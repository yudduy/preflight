from __future__ import annotations

import numpy as np

from preflight.loader import Sample
from preflight.coverage import analyze_coverage


def _make_samples(n: int) -> list[Sample]:
    return [
        Sample(prompt=f"prompt {i}", chosen=f"chosen {i}", rejected=f"rejected {i}")
        for i in range(n)
    ]


class TestAnalyzeCoverage:
    def test_basic_clustering(self):
        np.random.seed(42)
        samples = _make_samples(20)
        # 15 in cluster A, 5 in cluster B
        emb = np.vstack([
            np.random.randn(15, 2) + np.array([10, 0]),
            np.random.randn(5, 2) + np.array([-10, 0]),
        ])
        result = analyze_coverage(samples, prompt_embeddings=emb, n_clusters=2)

        assert result["n_clusters"] == 2
        assert len(result["clusters"]) == 2
        sizes = sorted([c["size"] for c in result["clusters"]])
        assert sizes == [5, 15]

    def test_gap_detection(self):
        np.random.seed(42)
        n = 30
        samples = _make_samples(n)
        emb = np.vstack([
            np.random.randn(29, 2) + np.array([10, 0]),
            np.array([[- 50, 0]]),  # isolated point
        ])
        result = analyze_coverage(samples, prompt_embeddings=emb, n_clusters=2)

        # The small cluster (1 out of 30 = 3.3%) should be a gap
        small = [c for c in result["clusters"] if c["size"] == 1]
        assert len(small) == 1
        assert small[0]["pct"] < 5.0
        assert result["n_underrepresented"] >= 1
        assert len(result["gaps"]) >= 1

    def test_representative_prompts(self):
        np.random.seed(42)
        samples = _make_samples(10)
        emb = np.random.randn(10, 4)
        result = analyze_coverage(samples, prompt_embeddings=emb, n_clusters=2)

        for c in result["clusters"]:
            assert len(c["representative_prompts"]) <= 3
            assert len(c["representative_prompts"]) > 0

    def test_clamp_n_clusters(self):
        np.random.seed(42)
        samples = _make_samples(3)
        emb = np.random.randn(3, 4)
        result = analyze_coverage(samples, prompt_embeddings=emb, n_clusters=100)

        assert result["n_clusters"] == 3

    def test_single_cluster(self):
        np.random.seed(42)
        samples = _make_samples(5)
        emb = np.random.randn(5, 2)
        result = analyze_coverage(samples, prompt_embeddings=emb, n_clusters=1)

        assert result["n_clusters"] == 1
        assert result["clusters"][0]["size"] == 5
        assert result["n_underrepresented"] == 0
