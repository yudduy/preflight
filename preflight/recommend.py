from __future__ import annotations


def generate_recommendations(
    length_bias: dict | None = None,
    embedding_similarity: dict | None = None,
    coverage: dict | None = None,
    easy_pairs: dict | None = None,
    dedup: dict | None = None,
    judge_scores: dict | None = None,
) -> list[str]:
    recs: list[str] = []

    if length_bias and length_bias.get("biased"):
        pct = length_bias["chosen_longer_pct"]
        recs.append(
            f"Dataset has strong length bias ({pct:.0f}% chosen > rejected). "
            "Consider length-normalizing or using length-penalized DPO (LN-DPO)."
        )

    if embedding_similarity:
        lc = embedding_similarity.get("low_contrast_count", 0)
        lc_pct = embedding_similarity.get("low_contrast_pct", 0)
        if lc > 0:
            recs.append(
                f"{lc} pairs ({lc_pct:.1f}%) have near-identical chosen/rejected "
                "(similarity > 0.9). Consider removing or replacing these low-contrast pairs."
            )

    if coverage and coverage.get("gaps"):
        n = coverage["n_underrepresented"]
        recs.append(
            f"{n} prompt cluster(s) have < 5% coverage. "
            "Consider adding more examples in underrepresented categories."
        )

    if easy_pairs and easy_pairs.get("count", 0) > 0:
        c = easy_pairs["count"]
        pct = easy_pairs["pct"]
        recs.append(
            f"{c} pairs ({pct:.1f}%) are trivially easy (very different chosen/rejected). "
            "These add little learning signal — consider removing them."
        )

    if dedup:
        exact = dedup.get("exact_duplicate_count", 0)
        near_pct = dedup.get("near_duplicate_pct", 0)
        if exact > 0:
            recs.append(
                f"{exact} samples are exact duplicates of other prompts. "
                "Remove duplicates to avoid overfitting on repeated prompts."
            )
        if near_pct > 5.0:
            near = dedup.get("near_duplicate_count", 0)
            recs.append(
                f"{near} near-duplicate prompt pairs ({near_pct:.1f}% of pairs) detected. "
                "Consider deduplicating or diversifying prompts."
            )

    if judge_scores:
        ml = judge_scores.get("mislabeled_count", 0)
        if ml > 0:
            recs.append(
                f"{ml} pairs appear mislabeled (rejected scored higher than chosen by judge). Review these pairs."
            )
        ej = judge_scores.get("easy_by_judge_count", 0)
        if ej > 0:
            recs.append(
                f"{ej} pairs have large reward margin (> 5 points). These are trivially easy for DPO."
            )

    if not recs:
        recs.append("No major issues detected. Dataset looks ready for DPO training.")

    return recs
