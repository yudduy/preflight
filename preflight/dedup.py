from __future__ import annotations

from collections import Counter

import numpy as np

from preflight.loader import Sample


def analyze_duplicates(
    samples: list[Sample],
    prompt_embeddings: np.ndarray | None = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict:
    prompts = [s.prompt for s in samples]
    n = len(prompts)

    counts = Counter(prompts)
    exact_groups = {k: v for k, v in counts.items() if v > 1}
    exact_duplicate_count = sum(v for v in exact_groups.values()) - len(exact_groups)

    if prompt_embeddings is None:
        from preflight.embeddings import compute_embeddings

        prompt_embeddings = compute_embeddings(prompts, model_name)

    norms = np.linalg.norm(prompt_embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normed = prompt_embeddings / norms
    sim_matrix = normed @ normed.T

    np.fill_diagonal(sim_matrix, 0.0)

    exact_set = set()
    for prompt, count in exact_groups.items():
        indices = [i for i, p in enumerate(prompts) if p == prompt]
        for i in indices:
            for j in indices:
                if i < j:
                    exact_set.add((i, j))

    near_pairs = []
    rows, cols = np.where(np.triu(sim_matrix, k=1) > 0.95)
    for i, j in zip(rows, cols):
        if (int(i), int(j)) not in exact_set:
            near_pairs.append((int(i), int(j)))

    near_duplicate_count = len(near_pairs)
    near_duplicate_pct = near_duplicate_count / max(n * (n - 1) / 2, 1) * 100

    example_pairs = []
    for i, j in near_pairs[:5]:
        example_pairs.append(
            {
                "i": i,
                "j": j,
                "prompt_i": prompts[i][:100],
                "prompt_j": prompts[j][:100],
                "similarity": float(sim_matrix[i, j]),
            }
        )

    return {
        "n_samples": n,
        "exact_duplicate_count": exact_duplicate_count,
        "exact_duplicate_groups": len(exact_groups),
        "near_duplicate_count": near_duplicate_count,
        "near_duplicate_pct": round(near_duplicate_pct, 2),
        "near_duplicate_examples": example_pairs,
    }
