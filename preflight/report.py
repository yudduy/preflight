from __future__ import annotations

import json
from pathlib import Path


def build_report(
    metadata: dict,
    length_bias: dict,
    embedding_similarity: dict | None = None,
    coverage: dict | None = None,
    easy_pairs: dict | None = None,
    dedup: dict | None = None,
    judge_scores: dict | None = None,
    recommendations: list[str] | None = None,
) -> dict:
    report: dict = {"metadata": metadata, "length_bias": length_bias}
    if embedding_similarity is not None:
        report["embedding_similarity"] = embedding_similarity
    if coverage is not None:
        report["coverage"] = coverage
    if easy_pairs is not None:
        report["easy_pairs"] = easy_pairs
    if dedup is not None:
        report["dedup"] = dedup
    if judge_scores is not None:
        report["judge_scores"] = judge_scores
    report["recommendations"] = recommendations or []
    return report


def write_json(report: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def print_summary(report: dict) -> None:
    print("=" * 60)
    print("PREFLIGHT AUDIT REPORT")
    print("=" * 60)

    meta = report.get("metadata", {})
    print(f"\nDataset: {meta.get('path', 'N/A')}")
    print(f"Format:  {meta.get('format', 'N/A')}")
    print(f"Samples: {meta.get('n_samples', 'N/A')}")
    print(f"Skipped: {meta.get('n_skipped', 0)}")

    lb = report.get("length_bias")
    if lb:
        print(f"\n--- Length Bias ---")
        print(f"  Mean chosen length:  {lb['mean_chosen_length']:.1f}")
        print(f"  Mean rejected length:{lb['mean_rejected_length']:.1f}")
        print(f"  Median length ratio: {lb['median_length_ratio']:.2f}")
        print(f"  Chosen longer:       {lb['chosen_longer_pct']:.1f}%")
        flag = " *** BIASED ***" if lb["biased"] else ""
        print(f"  Biased:              {lb['biased']}{flag}")
        print(f"  Mean chosen tokens:  {lb['mean_chosen_tokens']:.1f}")
        print(f"  Mean rejected tokens:{lb['mean_rejected_tokens']:.1f}")
        print(f"  Token length ratio:  {lb['token_length_ratio']:.2f}")

    es = report.get("embedding_similarity")
    if es:
        print(f"\n--- Embedding Similarity ---")
        for k, v in es.items():
            print(f"  {k}: {v}")

    cov = report.get("coverage")
    if cov:
        print(f"\n--- Coverage ---")
        for k, v in cov.items():
            print(f"  {k}: {v}")

    dd = report.get("dedup")
    if dd:
        print(f"\n--- Prompt Deduplication ---")
        print(f"  Exact duplicate samples: {dd['exact_duplicate_count']}")
        print(f"  Exact duplicate groups:  {dd['exact_duplicate_groups']}")
        print(f"  Near-duplicate pairs:    {dd['near_duplicate_count']}")
        print(f"  Near-duplicate pct:      {dd['near_duplicate_pct']:.2f}%")

    ep = report.get("easy_pairs")
    if ep:
        print(f"\n--- Easy Pairs ---")
        for k, v in ep.items():
            print(f"  {k}: {v}")

    js = report.get("judge_scores")
    if js:
        print(f"\n--- Judge Scores ---")
        for k, v in js.items():
            print(f"  {k}: {v}")

    recs = report.get("recommendations", [])
    if recs:
        print(f"\n--- Recommendations ---")
        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r}")

    print("\n" + "=" * 60)
