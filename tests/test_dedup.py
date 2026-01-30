from __future__ import annotations

import numpy as np

from preflight.dedup import analyze_duplicates
from preflight.loader import Sample
from preflight.recommend import generate_recommendations


def _make_sample(prompt: str) -> Sample:
    return Sample(prompt=prompt, chosen="chosen", rejected="rejected")


class TestAnalyzeDuplicates:
    def test_no_duplicates(self):
        samples = [_make_sample(f"unique prompt {i}") for i in range(5)]
        emb = np.random.RandomState(42).randn(5, 16).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        result = analyze_duplicates(samples, prompt_embeddings=emb)
        assert result["exact_duplicate_count"] == 0
        assert result["exact_duplicate_groups"] == 0
        assert result["near_duplicate_count"] == 0
        assert result["n_samples"] == 5

    def test_exact_duplicates(self):
        samples = [
            _make_sample("hello world"),
            _make_sample("hello world"),
            _make_sample("hello world"),
            _make_sample("different prompt"),
        ]
        emb = np.random.RandomState(0).randn(4, 16).astype(np.float32)
        result = analyze_duplicates(samples, prompt_embeddings=emb)
        assert result["exact_duplicate_count"] == 2
        assert result["exact_duplicate_groups"] == 1

    def test_near_duplicates(self):
        base = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        near = base + np.array([[0.01, 0.0, 0.0, 0.0]], dtype=np.float32)
        far = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        emb = np.vstack([base, near, far])
        samples = [_make_sample(f"prompt {i}") for i in range(3)]
        result = analyze_duplicates(samples, prompt_embeddings=emb)
        assert result["near_duplicate_count"] >= 1
        assert result["near_duplicate_pct"] > 0

    def test_near_duplicates_excludes_exact(self):
        base = np.array([[1.0, 0.0]], dtype=np.float32)
        samples = [
            _make_sample("same"),
            _make_sample("same"),
        ]
        emb = np.vstack([base, base])
        result = analyze_duplicates(samples, prompt_embeddings=emb)
        assert result["exact_duplicate_count"] == 1
        assert result["near_duplicate_count"] == 0


class TestDeduplicationRecommendations:
    def test_exact_duplicates_recommendation(self):
        dedup = {
            "exact_duplicate_count": 10,
            "exact_duplicate_groups": 3,
            "near_duplicate_count": 0,
            "near_duplicate_pct": 0.0,
        }
        recs = generate_recommendations(dedup=dedup)
        assert any("exact duplicates" in r for r in recs)

    def test_near_duplicates_recommendation(self):
        dedup = {
            "exact_duplicate_count": 0,
            "exact_duplicate_groups": 0,
            "near_duplicate_count": 50,
            "near_duplicate_pct": 8.0,
        }
        recs = generate_recommendations(dedup=dedup)
        assert any("near-duplicate" in r for r in recs)

    def test_no_dedup_issues(self):
        dedup = {
            "exact_duplicate_count": 0,
            "exact_duplicate_groups": 0,
            "near_duplicate_count": 1,
            "near_duplicate_pct": 0.5,
        }
        recs = generate_recommendations(dedup=dedup)
        assert "ready for DPO" in recs[0]
