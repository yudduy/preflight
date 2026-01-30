from __future__ import annotations

import argparse
import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from preflight.cli import main, run_audit
from preflight.loader import Sample


class TestArgParsing:
    def test_audit_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "preflight.cli", "audit", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--judge" in result.stdout
        assert "--output" in result.stdout

    def test_no_command_shows_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "preflight.cli"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


class TestRunAudit:
    def _make_args(self, tmp_path, judge=None):
        dataset = tmp_path / "data.jsonl"
        dataset.write_text(
            json.dumps({"prompt": "hi", "chosen": "good", "rejected": "bad"}) + "\n"
        )
        return argparse.Namespace(
            dataset=str(dataset),
            output=str(tmp_path / "report.json"),
            n_clusters=2,
            judge=judge,
            judge_base_url=None,
            judge_api_key=None,
            embedding_model="all-MiniLM-L6-v2",
        )

    @patch("preflight.cli.compute_embeddings")
    @patch("preflight.cli.analyze_coverage")
    @patch("preflight.cli.detect_easy_pairs")
    @patch("preflight.cli.analyze_embedding_similarity")
    @patch("preflight.cli.analyze_length_bias")
    def test_run_audit_no_judge(self, mock_lb, mock_emb, mock_easy, mock_cov, mock_ce, tmp_path):
        mock_lb.return_value = {"biased": False, "chosen_longer_pct": 50.0, "mean_chosen_length": 10, "mean_rejected_length": 10, "median_length_ratio": 1.0, "p_value": 1.0, "mean_chosen_tokens": 2.0, "mean_rejected_tokens": 2.0, "token_length_ratio": 1.0}
        mock_emb.return_value = ({"low_contrast_count": 0}, np.array([[1]]), np.array([[1]]), np.array([1.0]))
        mock_easy.return_value = {"count": 0, "pct": 0.0}
        mock_cov.return_value = {"gaps": [], "n_underrepresented": 0}
        mock_ce.return_value = np.array([[1]])

        args = self._make_args(tmp_path)
        run_audit(args)

        report_path = tmp_path / "report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert "judge_scores" not in report
        assert "recommendations" in report

    @patch("preflight.cli.compute_embeddings")
    @patch("preflight.cli.analyze_coverage")
    @patch("preflight.cli.detect_easy_pairs")
    @patch("preflight.cli.analyze_embedding_similarity")
    @patch("preflight.cli.analyze_length_bias")
    def test_run_audit_with_judge(self, mock_lb, mock_emb, mock_easy, mock_cov, mock_ce, tmp_path):
        mock_lb.return_value = {"biased": False, "chosen_longer_pct": 50.0, "mean_chosen_length": 10, "mean_rejected_length": 10, "median_length_ratio": 1.0, "p_value": 1.0, "mean_chosen_tokens": 2.0, "mean_rejected_tokens": 2.0, "token_length_ratio": 1.0}
        mock_emb.return_value = ({"low_contrast_count": 0}, np.array([[1]]), np.array([[1]]), np.array([1.0]))
        mock_easy.return_value = {"count": 0, "pct": 0.0}
        mock_cov.return_value = {"gaps": [], "n_underrepresented": 0}
        mock_ce.return_value = np.array([[1]])

        judge_result = {
            "scored_count": 1, "failed_count": 0, "mean_margin": 2.0,
            "std_margin": 0.0, "mislabeled_count": 0, "mislabeled_indices": [],
            "easy_by_judge_count": 0, "easy_by_judge_indices": [], "n_samples": 1,
        }

        with patch("preflight.judge.openai"), \
             patch("preflight.judge.judge_dataset", return_value=judge_result) as mock_jd, \
             patch("preflight.judge.JudgeClient"):
            args = self._make_args(tmp_path, judge="gpt-4")
            run_audit(args)

        report = json.loads((tmp_path / "report.json").read_text())
        assert report["judge_scores"]["scored_count"] == 1
