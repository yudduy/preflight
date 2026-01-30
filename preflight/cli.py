from __future__ import annotations

import argparse

from preflight.loader import load_dataset
from preflight.length import analyze_length_bias
from preflight.embeddings import analyze_embedding_similarity, compute_embeddings, detect_easy_pairs
from preflight.coverage import analyze_coverage
from preflight.dedup import analyze_duplicates
from preflight.recommend import generate_recommendations
from preflight.report import build_report, write_json, print_summary


def run_audit(args: argparse.Namespace) -> None:
    samples, fmt, n_skipped = load_dataset(args.dataset)

    length_bias = analyze_length_bias(samples)

    embedding_sim, chosen_emb, rejected_emb, similarities = analyze_embedding_similarity(
        samples, model_name=args.embedding_model
    )

    prompt_embeddings = compute_embeddings([s.prompt for s in samples], args.embedding_model)
    coverage = analyze_coverage(
        samples,
        prompt_embeddings=prompt_embeddings,
        model_name=args.embedding_model,
        n_clusters=args.n_clusters,
    )

    easy_pairs = detect_easy_pairs(
        samples, similarities=similarities
    )

    dedup = analyze_duplicates(
        samples, prompt_embeddings=prompt_embeddings, model_name=args.embedding_model
    )

    judge_scores = None
    if args.judge:
        try:
            import asyncio
            from preflight.judge import JudgeClient, judge_dataset

            client = JudgeClient(
                base_url=args.judge_base_url,
                api_key=args.judge_api_key,
            )
            judge_scores = asyncio.run(judge_dataset(samples, args.judge, client))
        except ImportError:
            print(
                "ERROR: openai package is required for --judge. "
                "Install it with: pip install openai"
            )
            raise SystemExit(1)

    recommendations = generate_recommendations(
        length_bias=length_bias,
        embedding_similarity=embedding_sim,
        coverage=coverage,
        easy_pairs=easy_pairs,
        judge_scores=judge_scores,
        dedup=dedup,
    )

    metadata = {
        "path": args.dataset,
        "format": fmt,
        "n_samples": len(samples),
        "n_skipped": n_skipped,
    }

    report = build_report(
        metadata=metadata,
        length_bias=length_bias,
        embedding_similarity=embedding_sim,
        coverage=coverage,
        easy_pairs=easy_pairs,
        dedup=dedup,
        judge_scores=judge_scores,
        recommendations=recommendations,
    )

    write_json(report, args.output)
    print_summary(report)
    print(f"\nReport written to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="preflight", description="DPO preference data profiler")
    sub = parser.add_subparsers(dest="command")
    audit = sub.add_parser("audit", help="Audit a preference dataset")
    audit.add_argument("dataset", help="Path to JSONL dataset")
    audit.add_argument("--output", default="preflight_report.json")
    audit.add_argument("--n-clusters", type=int, default=8)
    audit.add_argument("--judge", help="Judge model for LLM-based scoring")
    audit.add_argument("--judge-base-url", default=None)
    audit.add_argument("--judge-api-key", default=None)
    audit.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    if args.command == "audit":
        run_audit(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
