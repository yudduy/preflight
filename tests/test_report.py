from __future__ import annotations

import json
import tempfile
from pathlib import Path

from preflight.report import build_report, print_summary, write_json


META = {"path": "test.jsonl", "format": "together_ai", "n_samples": 10, "n_skipped": 0}
LENGTH_BIAS = {
    "mean_chosen_length": 100.0,
    "mean_rejected_length": 50.0,
    "median_length_ratio": 2.0,
    "chosen_longer_pct": 90.0,
    "p_value": 0.001,
    "biased": True,
    "mean_chosen_tokens": 20.0,
    "mean_rejected_tokens": 10.0,
    "token_length_ratio": 2.0,
    "n_samples": 10,
}


def test_build_all_sections():
    report = build_report(
        metadata=META,
        length_bias=LENGTH_BIAS,
        embedding_similarity={"mean": 0.5},
        coverage={"clusters": 8},
        easy_pairs={"pct": 10.0},
        judge_scores={"mean": 0.7},
        recommendations=["Fix length bias"],
    )
    assert "embedding_similarity" in report
    assert "coverage" in report
    assert "easy_pairs" in report
    assert "judge_scores" in report
    assert report["recommendations"] == ["Fix length bias"]


def test_build_length_only():
    report = build_report(metadata=META, length_bias=LENGTH_BIAS)
    assert "embedding_similarity" not in report
    assert "coverage" not in report
    assert report["recommendations"] == []


def test_write_json():
    report = build_report(metadata=META, length_bias=LENGTH_BIAS)
    path = tempfile.mktemp(suffix=".json")
    write_json(report, path)
    data = json.loads(Path(path).read_text())
    assert data["length_bias"]["biased"] is True


def test_print_summary(capsys):
    report = build_report(
        metadata=META,
        length_bias=LENGTH_BIAS,
        recommendations=["Fix length bias"],
    )
    print_summary(report)
    captured = capsys.readouterr()
    assert "BIASED" in captured.out
    assert "Fix length bias" in captured.out
