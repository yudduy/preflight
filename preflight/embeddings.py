from __future__ import annotations

import math

import numpy as np

from preflight.loader import Sample


def _cosine_similarity(c: np.ndarray, r: np.ndarray) -> float:
    nc = np.linalg.norm(c)
    nr = np.linalg.norm(r)
    if nc == 0 or nr == 0:
        return 0.0
    return float(np.dot(c, r) / (nc * nr))


_model_cache: dict = {}


def compute_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    model = _model_cache[model_name]
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


def analyze_embedding_similarity(
    samples: list[Sample], model_name: str = "all-MiniLM-L6-v2"
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    chosen_texts = [s.chosen for s in samples]
    rejected_texts = [s.rejected for s in samples]
    chosen_emb = compute_embeddings(chosen_texts, model_name)
    rejected_emb = compute_embeddings(rejected_texts, model_name)

    similarities = np.array(
        [_cosine_similarity(c, r) for c, r in zip(chosen_emb, rejected_emb)]
    )

    low_contrast_mask = similarities > 0.9

    n = len(similarities)
    mean_sim = float(np.mean(similarities))
    std_sim = float(np.std(similarities))
    if n > 1 and std_sim > 0:
        se = std_sim / math.sqrt(n)
        ci_low = mean_sim - 1.96 * se
        ci_high = mean_sim + 1.96 * se
    else:
        ci_low = mean_sim
        ci_high = mean_sim

    return (
        {
            "mean_similarity": mean_sim,
            "std_similarity": std_sim,
            "similarity_ci_95": (ci_low, ci_high),
            "pct_above_0_9": float(np.mean(similarities > 0.9) * 100),
            "pct_above_0_8": float(np.mean(similarities > 0.8) * 100),
            "pct_above_0_7": float(np.mean(similarities > 0.7) * 100),
            "low_contrast_count": int(np.sum(low_contrast_mask)),
            "low_contrast_pct": float(np.mean(low_contrast_mask) * 100),
            "similarity_p25": float(np.percentile(similarities, 25)),
            "similarity_p50": float(np.percentile(similarities, 50)),
            "similarity_p75": float(np.percentile(similarities, 75)),
            "n_samples": len(samples),
        },
        chosen_emb,
        rejected_emb,
        similarities,
    )


def detect_easy_pairs(
    samples: list[Sample],
    similarities: np.ndarray | None = None,
    chosen_emb: np.ndarray | None = None,
    rejected_emb: np.ndarray | None = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict:
    if similarities is None:
        if chosen_emb is None or rejected_emb is None:
            raise ValueError("Provide similarities or embeddings")
        similarities = np.array(
            [_cosine_similarity(c, r) for c, r in zip(chosen_emb, rejected_emb)]
        )

    easy_by_embedding = similarities < 0.3

    easy_by_length = np.array(
        [
            len(s.chosen) / max(len(s.rejected), 1) > 3.0
            or len(s.chosen) / max(len(s.rejected), 1) < 0.33
            for s in samples
        ]
    )

    easy_mask = easy_by_embedding | easy_by_length
    easy_indices = [int(i) for i in np.where(easy_mask)[0]]

    return {
        "count": int(np.sum(easy_mask)),
        "pct": float(np.mean(easy_mask) * 100),
        "by_embedding": int(np.sum(easy_by_embedding)),
        "by_length_ratio": int(np.sum(easy_by_length)),
        "indices": easy_indices,
        "n_samples": len(samples),
    }
