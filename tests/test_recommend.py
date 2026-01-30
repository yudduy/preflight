from __future__ import annotations

from preflight.recommend import generate_recommendations


class TestGenerateRecommendations:
    def test_no_issues(self):
        recs = generate_recommendations()
        assert len(recs) == 1
        assert "ready for DPO" in recs[0]

    def test_length_bias(self):
        lb = {"biased": True, "chosen_longer_pct": 85.0}
        recs = generate_recommendations(length_bias=lb)
        assert any("length bias" in r.lower() for r in recs)
        assert "85%" in recs[0]

    def test_length_bias_not_biased(self):
        lb = {"biased": False, "chosen_longer_pct": 55.0}
        recs = generate_recommendations(length_bias=lb)
        assert "ready for DPO" in recs[0]

    def test_embedding_similarity(self):
        es = {"low_contrast_count": 10, "low_contrast_pct": 20.0}
        recs = generate_recommendations(embedding_similarity=es)
        assert any("low-contrast" in r for r in recs)
        assert any("10 pairs" in r for r in recs)

    def test_embedding_no_low_contrast(self):
        es = {"low_contrast_count": 0, "low_contrast_pct": 0.0}
        recs = generate_recommendations(embedding_similarity=es)
        assert "ready for DPO" in recs[0]

    def test_coverage_gaps(self):
        cov = {"gaps": [0, 2], "n_underrepresented": 2}
        recs = generate_recommendations(coverage=cov)
        assert any("cluster" in r for r in recs)

    def test_coverage_no_gaps(self):
        cov = {"gaps": [], "n_underrepresented": 0}
        recs = generate_recommendations(coverage=cov)
        assert "ready for DPO" in recs[0]

    def test_easy_pairs(self):
        ep = {"count": 5, "pct": 12.5}
        recs = generate_recommendations(easy_pairs=ep)
        assert any("trivially easy" in r for r in recs)

    def test_easy_pairs_zero(self):
        ep = {"count": 0, "pct": 0.0}
        recs = generate_recommendations(easy_pairs=ep)
        assert "ready for DPO" in recs[0]

    def test_multiple_issues(self):
        recs = generate_recommendations(
            length_bias={"biased": True, "chosen_longer_pct": 90.0},
            embedding_similarity={"low_contrast_count": 5, "low_contrast_pct": 10.0},
            coverage={"gaps": [1], "n_underrepresented": 1},
            easy_pairs={"count": 3, "pct": 6.0},
        )
        assert len(recs) == 4
        assert not any("ready for DPO" in r for r in recs)
