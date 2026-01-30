from __future__ import annotations

import math

from preflight.loader import Sample


def _binomtest_pvalue(k: int, n: int, p: float = 0.5) -> float:
    if n == 0:
        return 1.0

    def _binom_pmf(x: int, n: int, p: float) -> float:
        coeff = math.comb(n, x)
        return coeff * (p ** x) * ((1 - p) ** (n - x))

    observed_pmf = _binom_pmf(k, n, p)
    pval = sum(_binom_pmf(x, n, p) for x in range(n + 1) if _binom_pmf(x, n, p) <= observed_pmf + 1e-15)
    return min(pval, 1.0)


def analyze_length_bias(samples: list[Sample]) -> dict:
    n = len(samples)
    chosen_lens = [len(s.chosen) for s in samples]
    rejected_lens = [len(s.rejected) for s in samples]
    ratios = [c / r if r > 0 else float("inf") for c, r in zip(chosen_lens, rejected_lens)]
    finite_ratios = [r for r in ratios if math.isfinite(r)]
    if finite_ratios:
        sorted_r = sorted(finite_ratios)
        mid = len(sorted_r) // 2
        median_ratio = (sorted_r[mid] + sorted_r[~mid]) / 2
    else:
        median_ratio = 0.0
    chosen_longer = sum(1 for c, r in zip(chosen_lens, rejected_lens) if c > r)
    pct = (chosen_longer / n) * 100 if n else 0.0
    p_value = _binomtest_pvalue(chosen_longer, n)

    chosen_tokens = [len(s.chosen.split()) for s in samples]
    rejected_tokens = [len(s.rejected.split()) for s in samples]
    token_ratios = [c / r if r > 0 else float("inf") for c, r in zip(chosen_tokens, rejected_tokens)]
    finite_token_ratios = [r for r in token_ratios if math.isfinite(r)]

    return {
        "mean_chosen_length": sum(chosen_lens) / n if n else 0.0,
        "mean_rejected_length": sum(rejected_lens) / n if n else 0.0,
        "median_length_ratio": median_ratio,
        "chosen_longer_pct": pct,
        "p_value": p_value,
        "biased": p_value < 0.05,
        "mean_chosen_tokens": sum(chosen_tokens) / n if n else 0.0,
        "mean_rejected_tokens": sum(rejected_tokens) / n if n else 0.0,
        "token_length_ratio": sum(finite_token_ratios) / len(finite_token_ratios) if finite_token_ratios else 0.0,
        "n_samples": n,
    }
