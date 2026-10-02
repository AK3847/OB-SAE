"""Analyze cross-method CAFT latent rankings without changing source results."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METHOD_NAMES = {1: "method_1", 2: "method_2", 3: "method_3"}
SCORE_COLUMNS = ("mean_attribution", "attribution_effect", "mean_activation")
MEDICAL_TERMS = re.compile(
    r"\b(medical|medicine|medication|medications|drug|drugs|diagnos\w*|"
    r"treat\w*|symptom\w*|patient\w*|clinical|physician\w*|doctor\w*|"
    r"prescription\w*|therapy|therapies|dosage|dose|hospital\w*|healthcare)\b"
)
HEALTH_TERMS = re.compile(
    r"\b(healthcare|health care|mental health|physical health|public health|"
    r"wellness|well[- ]being|illness\w*|disease\w*|nutrition\w*|fitness|hygiene|"
    r"health condition\w*|health risk\w*)\b"
)
ADVICE_TERMS = re.compile(
    r"\b(advice|recommend\w*|instruct\w*|instruction\w*|should|must|guidance|"
    r"suggest\w*|imperative|directive|step[- ]by[- ]step)\b"
)
HARM_TERMS = re.compile(
    r"\b(harmful|unsafe|dangerous|misleading|incorrect|wrong|bad advice|"
    r"adverse|toxic|overdos\w*|contraindicat\w*|misdiagnos\w*|undertreat\w*)\b"
)
GENERIC_TERMS = re.compile(
    r"\b(function words?|grammar|grammatical|punctuation|prose|sentence structure|"
    r"common words?|conjunctions?|prepositions?|articles?|formatting|lists?)\b"
)


def parse_method(path: Path, frame: pd.DataFrame) -> int | None:
    """Infer a source method from an explicit column or the file/path name."""
    if "source_method" in frame:
        values = pd.to_numeric(frame["source_method"], errors="coerce").dropna().unique()
        if len(values) == 1 and int(values[0]) in METHOD_NAMES:
            return int(values[0])
    match = re.search(r"(?:^|[/_.-])m(?:ethod)?[_-]?([123])(?:$|[/_.-])", str(path).lower())
    if match:
        return int(match.group(1))
    return None


def first_value(row: pd.Series, names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row.index and pd.notna(row[name]) and str(row[name]).strip():
            return row[name]
    return None


def rank_column(method: int, frame: pd.DataFrame) -> str | None:
    candidates = (f"method{method}_rank", "ranking_rank", "rank")
    return next((name for name in candidates if name in frame.columns), None)


def infer_dimension(row: pd.Series, name: str, path: Path) -> Any:
    value = row.get(name)
    if pd.notna(value):
        return value
    match = re.search(rf"(?:^|[/_]){name}[_-]?(\d+)", str(path).lower())
    return int(match.group(1)) if match else None


def integer_or_none(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if numeric.is_integer() else None


def load_evidence(path: Path) -> tuple[str, list[str]]:
    """Read interpretation-sidecar examples if present; never infer examples."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", []
    examples = payload.get("examples", []) if isinstance(payload, dict) else []
    texts = [str(item["highlighted_text"]) for item in examples
             if isinstance(item, dict) and item.get("highlighted_text")]
    return "\n".join(texts), texts


def normalize_results(results_dir: Path, evidence_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str]]:
    """Load supported result CSVs, normalize fields, and account for discarded rows."""
    records: list[dict[str, Any]] = []
    discards: list[dict[str, Any]] = []
    skipped_files: list[str] = []
    paths = sorted(results_dir.rglob("*.csv"))
    if not paths:
        raise ValueError(f"No CSV files found recursively under {results_dir}")

    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            skipped_files.append(f"{path}: CSV read error: {exc}")
            continue
        method = parse_method(path, frame)
        if method is None:
            skipped_files.append(f"{path}: could not infer method")
            continue
        rank_name = rank_column(method, frame)
        if "latent_id" not in frame or rank_name is None:
            skipped_files.append(f"{path}: missing latent_id or rank column")
            continue

        for row_number, (_, row) in enumerate(frame.iterrows(), start=2):
            layer = integer_or_none(infer_dimension(row, "layer", path))
            k_value = integer_or_none(infer_dimension(row, "k", path))
            latent = integer_or_none(row.get("latent_id"))
            rank = pd.to_numeric(pd.Series([row.get(rank_name)]), errors="coerce").iloc[0]
            if latent is None:
                discards.append({"source_file": str(path), "source_row": row_number,
                                 "reason": "invalid_or_missing_latent_id"})
                continue
            if layer is None or k_value is None:
                discards.append({"source_file": str(path), "source_row": row_number,
                                 "reason": "missing_layer_or_k"})
                continue
            if pd.isna(rank) or rank < 1:
                discards.append({"source_file": str(path), "source_row": row_number,
                                 "reason": "invalid_or_missing_rank"})
                continue

            explanation_value = first_value(row, ("explanation", "interpretation", "judge_explanation"))
            explanation = str(explanation_value) if explanation_value is not None else ""
            status_value = first_value(row, ("status",))
            status = str(status_value).strip().lower() if status_value is not None else "missing"
            if not explanation and status in {"complete", "completed", "success", "successful"}:
                status = "missing_explanation"
            score_values = {}
            for score_name in SCORE_COLUMNS:
                score = pd.to_numeric(pd.Series([row.get(score_name)]), errors="coerce").iloc[0]
                score_values[score_name] = float(score) if pd.notna(score) else np.nan
            judge_score = pd.to_numeric(pd.Series([row.get("relevance_score")]), errors="coerce").iloc[0]
            if pd.isna(judge_score) or not 0 <= judge_score <= 100:
                judge_score = np.nan

            contexts: list[str] = []
            context_path = (evidence_dir / f"method_{method}" / "interpretation" /
                            f"layer_{layer}_k{k_value}" / f"latent_{latent}" / "examples.json")
            context_text, contexts = load_evidence(context_path)
            records.append({
                "method": method, "layer": layer, "k": k_value, "latent_id": latent,
                "rank": float(rank), **score_values,
                "relevance_score": float(judge_score) if pd.notna(judge_score) else np.nan,
                "explanation": explanation, "status": status,
                "interpretation_failed": status in {"failed", "error"},
                "interpretation_available": bool(explanation), "context_evidence": context_text,
                "context_count": len(contexts), "source_file": str(path), "source_row": row_number,
            })

    if not records:
        raise ValueError("No usable result rows found. See schema requirements in --help.")

    data = pd.DataFrame(records)
    # If both interpreted summaries and raw rankings were discovered, retain the
    # richest row for a method/configuration/latent rather than double-counting it.
    data["_richness"] = (data["interpretation_available"].astype(int) * 4
                         + data["relevance_score"].notna().astype(int) * 2
                         + data["context_count"].gt(0).astype(int))
    ordered = data.sort_values(["_richness", "rank", "source_file"],
                               ascending=[False, True, True], kind="stable")
    duplicate_mask = ordered.duplicated(["method", "layer", "k", "latent_id"], keep="first")
    for duplicate in ordered.loc[duplicate_mask].itertuples(index=False):
        discards.append({"source_file": duplicate.source_file, "source_row": duplicate.source_row,
                         "reason": "duplicate_method_layer_k_latent"})
    data = ordered.loc[~duplicate_mask].drop(columns="_richness").reset_index(drop=True)
    return data, discards, skipped_files


def classify_relevance(text: str, has_interpretation: bool) -> tuple[str, str, str]:
    """Classify textual evidence; judge rank/score alone cannot imply medical relevance."""
    if not text.strip():
        return "unclear", "unknown", "No usable interpretation or activating-context text."
    lowered = text.lower()
    medical = bool(MEDICAL_TERMS.search(lowered))
    health = bool(HEALTH_TERMS.search(lowered))
    advice = bool(ADVICE_TERMS.search(lowered))
    harm = bool(HARM_TERMS.search(lowered))
    generic = bool(GENERIC_TERMS.search(lowered))

    if medical:
        classification = "medical_specific"
    elif health and (advice or harm):
        classification = "health_related"
    elif health:
        classification = "health_related"
    elif advice:
        classification = "advice_or_instruction"
    elif generic:
        classification = "generic_language"
    else:
        classification = "unclear"

    if medical and harm and (advice or re.search(r"\b(medical|medicine|medication|treatment|diagnos\w*)\b", lowered)):
        relevance = "high"
        reason = "Explicit medical and potentially harmful/unsafe evidence appears in the text."
    elif medical and (advice or harm):
        relevance = "medium"
        reason = "Medical-specific concepts co-occur with advice or harm-related wording."
    elif health and (advice or harm) and harm:
        relevance = "medium"
        reason = "Health-related wording co-occurs with explicit harm language; medical specificity is limited."
    elif classification in {"medical_specific", "health_related"}:
        relevance = "low"
        reason = "Medical/health concepts appear without explicit harmful-advice evidence."
    elif classification in {"advice_or_instruction", "generic_language"}:
        relevance = "low"
        reason = "Text indicates a general language or instruction feature, not medical evidence."
    else:
        relevance = "unknown"
        reason = "The available text is too ambiguous for a relevance assessment."
    if not has_interpretation and relevance != "unknown":
        reason += " Classification relies on saved activating-context text."
    return classification, relevance, reason


def interpretation_for(group: pd.DataFrame) -> str:
    parts: list[str] = []
    for value in group["explanation"].tolist() + group["context_evidence"].tolist():
        if value and value not in parts:
            parts.append(value)
    return " | ".join(parts)


def selected(data: pd.DataFrame, top_n: int, methods: list[int], layers: list[int] | None,
             k_values: list[int] | None) -> pd.DataFrame:
    result = data[data.method.isin(methods) & (data["rank"] <= top_n)].copy()
    if layers:
        result = result[result.layer.isin(layers)]
    if k_values:
        result = result[result.k.isin(k_values)]
    return result


def analyze(data: pd.DataFrame, top_n: int, methods: list[int], layers: list[int] | None,
            k_values: list[int] | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top = selected(data, top_n, methods, layers, k_values)
    if top.empty:
        raise ValueError("No rows match the selected method/layer/k/top-N filters")
    aggregated_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    configs = sorted(set(zip(top.layer, top.k)))
    for layer, k_value in configs:
        config = top[(top.layer == layer) & (top.k == k_value)]
        available_methods = sorted(set(config.method))
        for first in range(1, 4):
            for second in range(first + 1, 4):
                if first not in methods or second not in methods:
                    continue
                first_ids = set(config.loc[config.method == first, "latent_id"])
                second_ids = set(config.loc[config.method == second, "latent_id"])
                overlap_rows.append({"layer": layer, "k": k_value,
                    "method_a": first, "method_b": second,
                    "method_a_available": first in available_methods,
                    "method_b_available": second in available_methods,
                    "method_a_count": len(first_ids), "method_b_count": len(second_ids),
                    "overlap_count": len(first_ids & second_ids),
                    "overlap_latent_ids": ";".join(map(str, sorted(first_ids & second_ids)))})

        for latent_id, group in config.groupby("latent_id", sort=True):
            method_rows = {int(row.method): row for row in group.itertuples(index=False)}
            ranks = [float(method_rows[m].rank) for m in sorted(method_rows)]
            reciprocal = [1.0 / rank for rank in ranks]
            scores = {method: (float(method_rows[method].mean_attribution)
                               if method in method_rows and pd.notna(method_rows[method].mean_attribution) else np.nan)
                      for method in (1, 2)}
            activation = (float(method_rows[3].mean_activation)
                          if 3 in method_rows and pd.notna(method_rows[3].mean_activation) else np.nan)
            explanation = interpretation_for(group)
            context_values = list(dict.fromkeys(value for value in group.context_evidence.tolist() if value))
            activating_contexts = " | ".join(context_values)
            has_interpretation = bool(group.interpretation_available.any())
            classification, relevance, reason = classify_relevance(explanation, has_interpretation)
            judge_scores = group.relevance_score.dropna()
            failed_count = int(group.interpretation_failed.sum())
            row: dict[str, Any] = {
                "latent_id": int(latent_id), "layer": int(layer), "k": int(k_value),
                "methods_supporting": ",".join(str(method) for method in sorted(method_rows)),
                "cross_method_support": len(method_rows), "aggregated_rank_score": float(sum(reciprocal)),
                "best_rank": min(ranks), "mean_rank": float(np.mean(ranks)),
                "rank_std": float(np.std(ranks, ddof=0)), "medical_relevance": classification,
                "bad_medical_advice_relevance": relevance, "relevance_reason": reason,
                "interpretation": explanation, "judge_relevance_score_mean": float(judge_scores.mean()) if len(judge_scores) else np.nan,
                "interpretation_failures": failed_count,
                "context_available": bool(group.context_count.gt(0).any()),
                "context_example_count": int(group.context_count.sum()),
                "activating_contexts": activating_contexts,
            }
            for method in (1, 2, 3):
                method_row = method_rows.get(method)
                row[f"method_{method}_rank"] = float(method_row.rank) if method_row else np.nan
                row[f"method_{method}_score"] = scores[method] if method in (1, 2) else activation
                row[f"method_{method}_status"] = method_row.status if method_row else "not_selected"
            # Transparent shortlist heuristic. Text-based relevance dominates; rank
            # support and judge relevance are supporting evidence, never classifiers.
            category_points = {"medical_specific": 5.0, "health_related": 3.0,
                               "advice_or_instruction": 1.0, "generic_language": 0.0, "unclear": 0.0}
            harmful_explicit = bool(HARM_TERMS.search(explanation.lower()) and
                                    (MEDICAL_TERMS.search(explanation.lower()) or HEALTH_TERMS.search(explanation.lower())))
            row["shortlist_score"] = (
                category_points[classification] + (2.0 if harmful_explicit else 0.0)
                + 1.5 * (len(method_rows) - 1) / 2
                + min(sum(reciprocal), 1.0)
                + (0.5 if has_interpretation else 0.0)
                + (0.5 if row["context_available"] else 0.0)
                + (0.5 * float(judge_scores.mean()) / 100 if len(judge_scores) else 0.0)
            )
            reasons = [f"supported by {len(method_rows)} method(s)", classification.replace("_", " ")]
            if harmful_explicit:
                reasons.append("explicit medical/health harm wording")
            if row["context_available"]:
                reasons.append("activating contexts available")
            row["reason_for_selection"] = "; ".join(reasons)
            aggregated_rows.append(row)

    aggregated = pd.DataFrame(aggregated_rows)
    aggregated = aggregated.sort_values(
        ["shortlist_score", "cross_method_support", "aggregated_rank_score", "layer", "k", "latent_id"],
        ascending=[False, False, False, True, True, True], kind="stable")
    overlap = pd.DataFrame(overlap_rows)
    consensus = aggregated[aggregated.cross_method_support == 3].copy()
    return top, overlap, aggregated, consensus


def method_summaries(data: pd.DataFrame, methods: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats: list[dict[str, Any]] = []
    tops: list[pd.DataFrame] = []
    for method in methods:
        frame = data[data.method == method]
        configs = frame.groupby(["layer", "k"], sort=True)
        for (layer, k_value), group in configs:
            scores = group[list(SCORE_COLUMNS)].stack().dropna().to_numpy(dtype=float)
            stats.append({
                "method": method, "layer": layer, "k": k_value, "total_rows": len(group),
                "valid_latents": group.latent_id.nunique(),
                "valid_interpretations": int(group.interpretation_available.sum()),
                "failed_interpretations": int(group.interpretation_failed.sum()),
                "missing_interpretations": int((~group.interpretation_available & ~group.interpretation_failed).sum()),
                "missing_scores": int(group[list(SCORE_COLUMNS)].isna().all(axis=1).sum()),
                "judge_scores_missing": int(group.relevance_score.isna().sum()),
                "score_count": len(scores), "score_mean": float(np.mean(scores)) if len(scores) else np.nan,
                "score_std": float(np.std(scores)) if len(scores) else np.nan,
                "score_min": float(np.min(scores)) if len(scores) else np.nan,
                "score_max": float(np.max(scores)) if len(scores) else np.nan,
            })
            ordered = group.sort_values(["rank", "latent_id"], kind="stable").head(25).copy()
            tops.append(ordered)
    return pd.DataFrame(stats), pd.concat(tops, ignore_index=True) if tops else pd.DataFrame()


def overall_method_summaries(data: pd.DataFrame, methods: list[int]) -> pd.DataFrame:
    rows = []
    for method in methods:
        group = data[data.method == method]
        rows.append({
            "method": method, "total_rows": len(group), "valid_latents": group.latent_id.nunique(),
            "valid_interpretations": int(group.interpretation_available.sum()),
            "failed_interpretations": int(group.interpretation_failed.sum()),
            "missing_interpretations": int((~group.interpretation_available & ~group.interpretation_failed).sum()),
            "missing_ranking_scores": int(group[list(SCORE_COLUMNS)].isna().all(axis=1).sum()),
            "missing_judge_scores": int(group.relevance_score.isna().sum()),
            "context_examples": int(group.context_count.sum()),
        })
    return pd.DataFrame(rows)


def layer_summaries(aggregated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for layer, group in aggregated.groupby("layer", sort=True):
        rows.append({
            "layer": layer, "unique_candidates": group.latent_id.nunique(),
            "cross_method_overlaps": int((group.cross_method_support >= 2).sum()),
            "health_related_candidates": int(group.medical_relevance.isin(["health_related", "medical_specific"]).sum()),
            "medical_specific_candidates": int((group.medical_relevance == "medical_specific").sum()),
            "three_way_consensus_candidates": int((group.cross_method_support == 3).sum()),
        })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render report tables without pandas' optional tabulate dependency."""
    if frame.empty:
        return "No rows."
    columns = [str(column) for column in frame.columns]

    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    rows = [[cell(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(output_dir: Path, data: pd.DataFrame, stats: pd.DataFrame, overlaps: pd.DataFrame,
                 aggregated: pd.DataFrame, consensus: pd.DataFrame, layers: pd.DataFrame,
                 overall_stats: pd.DataFrame,
                 discards: list[dict[str, Any]], skipped: list[str], top_n: int,
                 shortlist_size: int) -> None:
    methods = sorted(data.method.unique())
    context_rows = int(data.context_count.gt(0).sum())
    candidate_preview = aggregated.head(shortlist_size).copy()
    candidate_preview["interpretation"] = candidate_preview["interpretation"].fillna("").str.slice(0, 220)
    lines = ["# CAFT Latent Ranking Analysis", "",
             "## Dataset and result overview", "",
             f"- Source rows retained after deduplication: {len(data)}",
             f"- Methods found: {', '.join(METHOD_NAMES[m] for m in methods)}",
             f"- Configurations: {data.groupby(['layer', 'k']).ngroups}",
             f"- Top-N used for overlap and aggregation: {top_n}",
             f"- Rows excluded during normalization (invalid or duplicate): {len(discards)}",
             f"- Skipped CSV files: {len(skipped)}", "",
             "Method 1 and Method 2 attribution values and Method 3 activation-difference values are kept separate. "
             "Cross-method aggregation uses reciprocal rank (`1 / rank`), not raw score magnitudes.", "",
             "## Per-method statistics", "",
             markdown_table(stats) if not stats.empty else "No method statistics available.", "",
             "Overall totals by method:", "",
             markdown_table(overall_stats), "",
             "## Cross-method overlap", "",
             markdown_table(overlaps) if not overlaps.empty else "No pairwise comparisons available.", "",
             "Overlap counts indicate selection within the configured top-N, not failed interpretation.", "",
             "## Three-way consensus latents", "",
             markdown_table(consensus[["latent_id", "layer", "k", "method_1_rank", "method_2_rank", "method_3_rank",
                        "aggregated_rank_score", "medical_relevance", "bad_medical_advice_relevance"]])
             if not consensus.empty else "No three-way consensus latents in the selected top-N.", "",
             "## Aggregated rankings", "",
             "Reciprocal-rank scores are summed across methods that selected each latent. `rank_std` is the population standard deviation across those observed ranks.", "",
             markdown_table(aggregated.head(25)[["latent_id", "layer", "k", "cross_method_support", "aggregated_rank_score",
                                  "best_rank", "mean_rank", "rank_std", "medical_relevance"]]), "",
             "## Medical-relevance candidates", "",
             "Classification uses explanation text and saved activating contexts when present. A judge relevance score or a word such as 'health'/'advice' alone does not establish bad-medical-advice relevance.", "",
             "Categories distinguish generic language, advice/instruction, health-related, and medical-specific features. Relevance is qualitative and non-causal.", "",
             "Shortlist score formula: 5 points for medical-specific, 3 for health-related, 1 for advice/instruction, otherwise 0; +2 for explicit harm wording with medical/health evidence; +1.5 × (method support − 1) / 2; + min(sum of reciprocal ranks, 1); +0.5 when interpretation text exists; +0.5 when saved activating contexts exist; +0.5 × mean judge score / 100 when available. The judge score contributes only to prioritization, never classification.", "",
             markdown_table(aggregated.head(shortlist_size)[["latent_id", "layer", "k", "methods_supporting", "cross_method_support",
                                             "method_1_rank", "method_2_rank", "method_3_rank", "aggregated_rank_score",
                                             "interpretation", "medical_relevance", "bad_medical_advice_relevance", "reason_for_selection"]].assign(
                                                 interpretation=candidate_preview["interpretation"])), "",
             "## Layer comparison", "",
             markdown_table(layers) if not layers.empty else "No layer summary available.", "",
             "## Failed and missing interpretation statistics", "",
             f"- Total retained result rows: {len(data)}",
             f"- Valid interpretations: {int(data.interpretation_available.sum())}",
             f"- Failed interpretations: {int(data.interpretation_failed.sum())}",
             f"- Missing interpretations: {int((~data.interpretation_available & ~data.interpretation_failed).sum())}",
             f"- Missing all ranking scores: {int(data[list(SCORE_COLUMNS)].isna().all(axis=1).sum())}",
             "A `not_selected` method status means the latent was absent from that method's top-N; it is not classified as an interpretation failure.", "",
             "## Recommended shortlist", "",
             "The shortlist is a transparent prioritization for downstream validation, not a causal claim. See `medical_candidates.csv` for the score, evidence, and full method ranks.", "",
             "## Limitations", "",
             "- Interpretations are generated from generic FineWeb activating contexts, not necessarily bad-medical-advice examples.",
             "- The relevance judge score is an EM relevance score and is not itself proof of medical specificity or harmful advice.",
             f"- Activating-context sidecars were available for {context_rows} method/latent rows; missing sidecars are not interpreted as evidence.",
             "- Lexical classification is intentionally conservative and should be manually validated before intervention experiments.",
             "- Ranking overlap is limited to supplied CSV rows and configured top-N.", ""]
    (output_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_ints(value: str | None) -> list[int] | None:
    return [int(item.strip()) for item in value.split(",") if item.strip()] if value else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results",
                        help="Directory recursively containing method result CSVs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "analysis" / "outputs")
    parser.add_argument("--evidence-dir", type=Path,
                        help="Parent of method_1/interpretation sidecars (default: results directory parent)")
    parser.add_argument("--top-n", type=int, default=25, help="Per-method rank cutoff for overlap/aggregation")
    parser.add_argument("--shortlist-size", type=int, default=25)
    parser.add_argument("--methods", default="1,2,3", help="Comma-separated method IDs")
    parser.add_argument("--layers", help="Optional comma-separated zero-based layers")
    parser.add_argument("--k-values", help="Optional comma-separated SAE k values")
    args = parser.parse_args(argv)
    if args.top_n < 1 or args.shortlist_size < 1:
        parser.error("--top-n and --shortlist-size must be positive")
    methods = parse_ints(args.methods) or []
    if not methods or any(method not in METHOD_NAMES for method in methods):
        parser.error("--methods must contain one or more IDs from 1,2,3")
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    evidence_dir = (args.evidence_dir.expanduser().resolve() if args.evidence_dir
                    else results_dir.parent / "outputs")
    try:
        print(f"[load] recursively scanning {results_dir}", flush=True)
        data, discards, skipped = normalize_results(results_dir, evidence_dir)
        data = data[data.method.isin(methods)]
        layers_filter, k_filter = parse_ints(args.layers), parse_ints(args.k_values)
        if layers_filter:
            data = data[data.layer.isin(layers_filter)]
        if k_filter:
            data = data[data.k.isin(k_filter)]
        if data.empty:
            raise ValueError("No result rows remain after method/layer/k filters")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[analyze] {len(data)} normalized rows; {len(discards)} discarded; {len(skipped)} files skipped", flush=True)
        top, overlaps, aggregated, consensus = analyze(data, args.top_n, methods, layers_filter, k_filter)
        stats, per_method_top = method_summaries(data, methods)
        overall_stats = overall_method_summaries(data, methods)
        layers = layer_summaries(aggregated)
        medical = aggregated.sort_values(
            ["shortlist_score", "cross_method_support", "aggregated_rank_score"],
            ascending=[False, False, False], kind="stable")

        stats.to_csv(output_dir / "method_statistics.csv", index=False)
        overall_stats.to_csv(output_dir / "method_overall_statistics.csv", index=False)
        per_method_top.to_csv(output_dir / "per_method_top_latents.csv", index=False)
        overlaps.to_csv(output_dir / "overlap_summary.csv", index=False)
        consensus.to_csv(output_dir / "consensus_latents.csv", index=False)
        aggregated.to_csv(output_dir / "aggregated_rankings.csv", index=False)
        medical[medical.cross_method_support >= 2].to_csv(output_dir / "overlap_latents.csv", index=False)
        medical.to_csv(output_dir / "medical_candidates.csv", index=False)
        layers.to_csv(output_dir / "layer_summary.csv", index=False)
        pd.DataFrame(discards, columns=["source_file", "source_row", "reason"]).to_csv(
            output_dir / "discarded_rows.csv", index=False)
        (output_dir / "skipped_files.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
        summary = {
            "results_dir": str(results_dir), "output_dir": str(output_dir), "top_n": args.top_n,
            "methods": methods, "total_normalized_rows": len(data), "valid_interpretations": int(data.interpretation_available.sum()),
            "failed_interpretations": int(data.interpretation_failed.sum()),
            "missing_interpretations": int((~data.interpretation_available & ~data.interpretation_failed).sum()),
            "missing_scores": int(data[list(SCORE_COLUMNS)].isna().all(axis=1).sum()),
            "missing_judge_scores": int(data.relevance_score.isna().sum()),
            "discarded_rows": len(discards), "skipped_files": skipped,
            "method_statistics": overall_stats.to_dict(orient="records"),
            "three_way_consensus_count": len(consensus),
            "shortlist": medical.head(args.shortlist_size)[["latent_id", "layer", "k", "methods_supporting",
                "aggregated_rank_score", "medical_relevance", "bad_medical_advice_relevance", "shortlist_score"]]
                .replace({np.nan: None}).to_dict(orient="records"),
        }
        (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        write_report(output_dir, data, stats, overlaps, aggregated, consensus, layers, overall_stats,
                     discards, skipped, args.top_n, args.shortlist_size)
        print(f"[done] analyzed {len(top)} top-N rows across {aggregated.groupby(['layer', 'k']).ngroups} configurations", flush=True)
        print(f"[output] {output_dir}", flush=True)
        return 0
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())