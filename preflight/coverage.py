from __future__ import annotations

import numpy as np

from preflight.loader import Sample


def analyze_coverage(
    samples: list[Sample],
    prompt_embeddings: np.ndarray | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    n_clusters: int = 8,
) -> dict:
    from sklearn.cluster import KMeans

    if prompt_embeddings is None:
        from preflight.embeddings import compute_embeddings

        prompt_embeddings = compute_embeddings([s.prompt for s in samples], model_name)

    k = min(n_clusters, len(samples))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(prompt_embeddings)

    clusters = []
    gaps = []
    for i in range(k):
        mask = labels == i
        indices = np.where(mask)[0]
        size = int(np.sum(mask))
        pct = float(size / len(samples) * 100)

        if size > 0:
            dists = np.linalg.norm(
                prompt_embeddings[mask] - kmeans.cluster_centers_[i], axis=1
            )
            nearest_idx = np.argsort(dists)[:3]
            representative = [samples[indices[j]].prompt[:100] for j in nearest_idx]
        else:
            representative = []

        if pct < 5.0:
            gaps.append(i)

        clusters.append(
            {
                "cluster_id": i,
                "size": size,
                "pct": round(pct, 1),
                "representative_prompts": representative,
            }
        )

    return {
        "n_clusters": k,
        "clusters": clusters,
        "gaps": gaps,
        "n_underrepresented": len(gaps),
    }
