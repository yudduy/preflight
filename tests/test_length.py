from __future__ import annotations

import math

from preflight.length import analyze_length_bias
from preflight.loader import Sample


def _sample(chosen: str, rejected: str) -> Sample:
    return Sample(prompt="p", chosen=chosen, rejected=rejected)


def test_all_chosen_longer():
    samples = [_sample("a" * 100, "b" * 10) for _ in range(10)]
    result = analyze_length_bias(samples)
    assert result["biased"] is True
    assert result["chosen_longer_pct"] == 100.0
    assert result["p_value"] < 0.05


def test_all_rejected_longer():
    samples = [_sample("a" * 10, "b" * 100) for _ in range(10)]
    result = analyze_length_bias(samples)
    assert result["biased"] is True
    assert result["chosen_longer_pct"] == 0.0
    assert result["p_value"] < 0.05


def test_equal_lengths():
    samples = [_sample("a" * 100, "b" * 50) for _ in range(5)]
    samples += [_sample("a" * 50, "b" * 100) for _ in range(5)]
    result = analyze_length_bias(samples)
    assert result["biased"] is False
    assert result["chosen_longer_pct"] == 50.0
    assert result["p_value"] >= 0.05


def test_mixed():
    samples = [_sample("a" * 100, "b" * 10) for _ in range(6)]
    samples += [_sample("a" * 10, "b" * 100) for _ in range(4)]
    result = analyze_length_bias(samples)
    assert result["chosen_longer_pct"] == 60.0
    assert result["n_samples"] == 10
    assert "p_value" in result


def test_empty_rejected_no_inf():
    samples = [_sample("hello", ""), _sample("world", "x")]
    result = analyze_length_bias(samples)
    assert math.isfinite(result["median_length_ratio"])


def test_both_empty_strings():
    samples = [_sample("", "")]
    result = analyze_length_bias(samples)
    assert result["mean_chosen_length"] == 0.0
    assert result["mean_rejected_length"] == 0.0
    assert math.isfinite(result["median_length_ratio"])
    assert not math.isnan(result["p_value"])


def test_p_value_present():
    samples = [_sample("a" * 100, "b" * 10) for _ in range(20)]
    result = analyze_length_bias(samples)
    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0


def test_token_counts():
    samples = [
        _sample("hello world foo", "bar baz"),
        _sample("one two", "three four five six"),
    ]
    result = analyze_length_bias(samples)
    assert result["mean_chosen_tokens"] == (3 + 2) / 2
    assert result["mean_rejected_tokens"] == (2 + 4) / 2
    assert "token_length_ratio" in result
    assert result["token_length_ratio"] > 0
