from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from preflight.loader import Sample
from preflight.embeddings import (
    _model_cache,
    analyze_embedding_similarity,
    compute_embeddings,
    detect_easy_pairs,
)


def _make_samples(n: int = 4) -> list[Sample]:
    return [
        Sample(prompt=f"prompt {i}", chosen=f"chosen {i}" * (i + 1), rejected=f"rejected {i}")
        for i in range(n)
    ]


@pytest.fixture
def samples():
    return _make_samples(4)


class TestComputeEmbeddings:
    def test_returns_numpy_array(self):
        _model_cache.clear()
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, 8)
        with patch.dict(_model_cache, {"all-MiniLM-L6-v2": mock_model}):
            result = compute_embeddings(["a", "b", "c"])
            assert result.shape == (3, 8)
            mock_model.encode.assert_called_once()


class TestAnalyzeEmbeddingSimilarity:
    @patch("preflight.embeddings.compute_embeddings")
    def test_basic_output(self, mock_embed, samples):
        emb = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
        mock_embed.side_effect = [emb, emb]

        result, c_emb, r_emb, _ = analyze_embedding_similarity(samples)

        assert result["n_samples"] == 4
        assert result["mean_similarity"] == pytest.approx(1.0, abs=1e-6)
        assert result["low_contrast_count"] == 4
        assert result["pct_above_0_9"] == pytest.approx(100.0)
        assert "similarity_ci_95" in result
        assert result["similarity_p25"] == pytest.approx(1.0, abs=1e-6)
        assert result["similarity_p50"] == pytest.approx(1.0, abs=1e-6)
        assert result["similarity_p75"] == pytest.approx(1.0, abs=1e-6)

    @patch("preflight.embeddings.compute_embeddings")
    def test_orthogonal_vectors(self, mock_embed, samples):
        chosen = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
        rejected = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=float)
        mock_embed.side_effect = [chosen, rejected]

        result, _, _, _ = analyze_embedding_similarity(samples)

        assert result["mean_similarity"] == pytest.approx(0.0, abs=1e-6)
        assert result["low_contrast_count"] == 0

    @patch("preflight.embeddings.compute_embeddings")
    def test_returns_embeddings(self, mock_embed, samples):
        emb = np.ones((4, 5))
        mock_embed.side_effect = [emb, emb.copy()]

        _, c, r, _ = analyze_embedding_similarity(samples)
        assert c.shape == (4, 5)
        assert r.shape == (4, 5)

    @patch("preflight.embeddings.compute_embeddings")
    def test_zero_vector_no_nan(self, mock_embed, samples):
        chosen = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        rejected = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        mock_embed.side_effect = [chosen, rejected]

        result, _, _, _ = analyze_embedding_similarity(samples)

        assert not np.isnan(result["mean_similarity"])

    @patch("preflight.embeddings.compute_embeddings")
    def test_confidence_interval(self, mock_embed, samples):
        emb = np.array([[1, 0], [0, 1], [1, 1], [0, 0.5]], dtype=float)
        mock_embed.side_effect = [emb, emb]

        result, _, _, _ = analyze_embedding_similarity(samples)

        ci = result["similarity_ci_95"]
        assert ci[0] <= result["mean_similarity"] <= ci[1]


class TestDetectEasyPairs:
    def test_easy_by_embedding(self):
        samples = [
            Sample(prompt="p", chosen="a" * 10, rejected="b" * 10),
            Sample(prompt="p", chosen="a" * 10, rejected="b" * 10),
        ]
        sims = np.array([0.1, 0.5])
        result = detect_easy_pairs(samples, similarities=sims)

        assert result["by_embedding"] == 1
        assert result["count"] >= 1
        assert 0 in result["indices"]

    def test_easy_by_length_ratio(self):
        samples = [
            Sample(prompt="p", chosen="a" * 100, rejected="b" * 10),
            Sample(prompt="p", chosen="a" * 5, rejected="b" * 50),
            Sample(prompt="p", chosen="a" * 10, rejected="b" * 10),
        ]
        sims = np.array([0.5, 0.5, 0.5])
        result = detect_easy_pairs(samples, similarities=sims)

        assert result["by_length_ratio"] == 2
        assert result["by_embedding"] == 0
        assert result["count"] == 2
        assert set(result["indices"]) == {0, 1}

    def test_combined(self):
        samples = [
            Sample(prompt="p", chosen="a" * 100, rejected="b" * 10),
            Sample(prompt="p", chosen="a" * 10, rejected="b" * 10),
        ]
        sims = np.array([0.5, 0.2])
        result = detect_easy_pairs(samples, similarities=sims)

        assert result["count"] == 2
        assert result["pct"] == pytest.approx(100.0)

    def test_raises_without_inputs(self):
        samples = [Sample(prompt="p", chosen="a", rejected="b")]
        with pytest.raises(ValueError, match="Provide similarities or embeddings"):
            detect_easy_pairs(samples)

    def test_from_embeddings(self):
        samples = [
            Sample(prompt="p", chosen="a" * 10, rejected="b" * 10),
        ]
        chosen_emb = np.array([[1.0, 0.0]])
        rejected_emb = np.array([[0.0, 1.0]])
        result = detect_easy_pairs(
            samples, chosen_emb=chosen_emb, rejected_emb=rejected_emb
        )
        assert result["by_embedding"] == 1

    def test_zero_vector_in_detect(self):
        samples = [Sample(prompt="p", chosen="a" * 10, rejected="b" * 10)]
        chosen_emb = np.array([[0.0, 0.0]])
        rejected_emb = np.array([[1.0, 0.0]])
        result = detect_easy_pairs(samples, chosen_emb=chosen_emb, rejected_emb=rejected_emb)
        assert result["by_embedding"] == 1
