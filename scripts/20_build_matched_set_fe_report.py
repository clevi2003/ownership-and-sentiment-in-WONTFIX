#!/usr/bin/env python3
"""
Build matched-set fixed-effect reporting tables for the WONTFIX revision.

Primary model:
    outcome ~ is_wontfix + C(matched_set_id)
    standard errors clustered by matched_set_id

Robustness model:
    outcome ~ is_wontfix + C(repo_full_name)
    HC3 robust standard errors by default

Design notes:
- Matched sets are the unit of comparison: one WONTFIX issue and up to N matched controls.
- Because matched sets are same-repository, matched-set FE absorbs repo-level differences.
- Repo FE models are retained only as a coarser robustness check corresponding to the original analysis tier.
- Valid matched sets are recomputed separately for every outcome after missing outcomes and denominator filters are applied.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover
    smf = None

try:
    from statsmodels.stats.multitest import multipletests
except Exception:  # pragma: no cover
    multipletests = None

SCRIPT_VERSION = "matched_set_fe_report_v2026_06_17_plot_fix"

DEFAULT_RQ1_DATASET = "data/final/analysis_dataset_rq1.parquet"
DEFAULT_RQ2_DATASET = "data/final/analysis_dataset_rq2.parquet"
DEFAULT_RQ3_DATASET = "data/final/analysis_dataset_rq3_issue_level_base.parquet"
DEFAULT_ANALYSIS_QA = "logs/qa/analysis_dataset_qa_summary.csv"
DEFAULT_COMPARISON_QA = "logs/qa/comparison_issue_set_summary.csv"
DEFAULT_OUTPUT_DIR = "outputs/matched_set_fe"

MIN_MODEL_N = 40
MIN_GROUP_N = 10
MIN_EVENT_COUNT = 10
REPO_CLUSTER_THRESHOLD = 30


@dataclass(frozen=True)
class OutcomeSpec:
    dataset: str
    family: str
    tier: str
    outcome: str
    label: str
    outcome_type: str
    denominator_filter: str | None = None
    requires_comments: bool = False
    run_primary: bool = True
    run_repo_fe: bool = True
    practical_significance_priority: bool = True
    notes: str = ""


def build_outcome_specs() -> list[OutcomeSpec]:
    specs: list[OutcomeSpec] = []

    # RQ1 sentiment: primary/secondary sentiment features.
    for outcome, label, outcome_type, notes in [
        ("mean_comment_sentiment", "Mean comment sentiment", "continuous", "Aggregate shift in issue discussion sentiment."),
        ("positive_comment_share", "Positive comment share", "proportion", "Professor specifically flagged this as a small-effect practical-significance example."),
        ("negative_comment_share", "Negative comment share", "proportion", "Share of comments classified as negative."),
        ("std_comment_sentiment", "Sentiment variability", "continuous", "Volatility/dispersion in comment sentiment."),
        ("comment_sentiment_slope", "Sentiment slope", "continuous", "Approximate within-thread sentiment trajectory."),
        ("comment_sentiment_change_late_minus_early", "Late-minus-early sentiment change", "continuous", "Late discussion sentiment minus early discussion sentiment."),
        ("median_comment_sentiment", "Median comment sentiment", "continuous", "Optional if available."),
        ("min_comment_sentiment", "Minimum comment sentiment", "continuous", "Optional if available."),
        ("max_comment_sentiment", "Maximum comment sentiment", "continuous", "Optional if available."),
        ("neutral_comment_share", "Neutral comment share", "proportion", "Optional if available."),
    ]:
        specs.append(OutcomeSpec(
            dataset="rq1",
            family="sentiment",
            tier="primary" if outcome in {
                "mean_comment_sentiment", "positive_comment_share", "negative_comment_share",
                "std_comment_sentiment", "comment_sentiment_slope",
                "comment_sentiment_change_late_minus_early",
            } else "secondary",
            outcome=outcome,
            label=label,
            outcome_type=outcome_type,
            requires_comments=True,
            notes=notes,
        ))

    # Optional issue-level emotion outcomes. These do not replace the role-stratified surprise script.
    for outcome, label, outcome_type in [
        ("surprise_comment_share", "Surprise comment share", "proportion"),
        ("dominant_emotion_surprise_flag", "Dominant emotion is surprise", "binary"),
        ("sadness_comment_share", "Sadness comment share", "proportion"),
        ("dominant_emotion_sadness_flag", "Dominant emotion is sadness", "binary"),
        ("anger_comment_share", "Anger comment share", "proportion"),
        ("dominant_emotion_anger_flag", "Dominant emotion is anger", "binary"),
    ]:
        specs.append(OutcomeSpec(
            dataset="rq1",
            family="emotion_issue_level",
            tier="secondary",
            outcome=outcome,
            label=label,
            outcome_type=outcome_type,
            requires_comments=True,
            practical_significance_priority=False,
            notes="Issue-level emotion model only; role-stratified surprise should be handled separately.",
        ))

    # RQ3 participation.
    for outcome, label, outcome_type, notes in [
        ("log1p_comment_count", "log1p(comment count)", "log_count", "Comment count informed matching; interpret volume effects cautiously."),
        ("log1p_unique_commenter_count", "log1p(unique commenters)", "log_count", "Discussion breadth."),
        ("log1p_num_distinct_non_author_commenters", "log1p(non-author commenters)", "log_count", "Non-author participation breadth."),
        ("non_author_comment_share", "Non-author comment share", "proportion", "Author-centered versus community-centered discussion."),
        ("top_commenter_share", "Top commenter share", "proportion", "Discussion concentration."),
        ("comment_concentration_ratio", "Comment concentration ratio", "proportion", "Discussion concentration."),
        ("issue_author_commented_flag", "Issue author commented", "binary", "Whether the issue author participated after opening."),
        ("has_any_non_author_comment", "Any non-author commenter", "binary", "Whether discussion included someone beyond the issue author."),
        ("multi_party_discussion_flag", "Multiple commenters", "binary", "Whether discussion involved multiple commenters."),
        ("only_author_commented_flag", "Only issue author commented", "binary", "Author-centered discussion indicator."),
        ("first_comment_by_author_flag", "First comment by issue author", "binary", "Author involvement in early thread."),
        ("last_comment_by_author_flag", "Last comment by issue author", "binary", "Author involvement at end of thread."),
    ]:
        specs.append(OutcomeSpec(
            dataset="rq3",
            family="participation",
            tier="primary",
            outcome=outcome,
            label=label,
            outcome_type=outcome_type,
            requires_comments=outcome not in {"log1p_comment_count", "log1p_unique_commenter_count", "log1p_num_distinct_non_author_commenters"},
            notes=notes,
        ))

    # RQ2 primary ownership-adjacent repo-level participant roles.
    for outcome, label, outcome_type in [
        ("issue_author_is_pre_issue_repo_contributor", "Issue author was prior repo contributor", "binary"),
        ("issue_author_is_pre_issue_major_repo_contributor", "Issue author was major prior repo contributor", "binary"),
        ("any_commenter_is_pre_issue_repo_contributor", "Any commenter was prior repo contributor", "binary"),
        ("any_commenter_is_pre_issue_major_repo_contributor", "Any commenter was major prior repo contributor", "binary"),
        ("share_commenters_pre_issue_repo_contributors", "Share of commenters who were prior repo contributors", "proportion"),
        ("share_commenters_pre_issue_major_repo_contributors", "Share of commenters who were major prior repo contributors", "proportion"),
        ("top_commenter_is_pre_issue_repo_contributor", "Top commenter was prior repo contributor", "binary"),
        ("top_commenter_is_pre_issue_major_repo_contributor", "Top commenter was major prior repo contributor", "binary"),
        ("commenter_count_pre_issue_repo_contributors", "Prior repo contributor commenters", "count"),
    ]:
        specs.append(OutcomeSpec(
            dataset="rq2",
            family="ownership_repo_primary",
            tier="primary",
            outcome=outcome,
            label=label,
            outcome_type=outcome_type,
            notes="Primary ownership-adjacent feature family: pre-issue repo-level participant roles.",
        ))

    # RQ2 secondary file-level participant roles.
    file_filter = "participant_role_file_features_applicable == 1"
    for outcome, label, outcome_type in [
        ("issue_author_is_pre_issue_file_contributor", "Issue author was prior linked-file contributor", "binary"),
        ("issue_author_is_pre_issue_major_file_contributor", "Issue author was major prior linked-file contributor", "binary"),
        ("any_commenter_is_pre_issue_file_contributor", "Any commenter was prior linked-file contributor", "binary"),
        ("any_commenter_is_pre_issue_major_file_contributor", "Any commenter was major prior linked-file contributor", "binary"),
        ("share_commenters_pre_issue_file_contributors", "Share of commenters who were prior linked-file contributors", "proportion"),
        ("share_commenters_pre_issue_major_file_contributors", "Share of commenters who were major prior linked-file contributors", "proportion"),
        ("top_commenter_is_pre_issue_file_contributor", "Top commenter was prior linked-file contributor", "binary"),
        ("top_commenter_is_pre_issue_major_file_contributor", "Top commenter was major prior linked-file contributor", "binary"),
    ]:
        specs.append(OutcomeSpec(
            dataset="rq2",
            family="ownership_file_secondary",
            tier="secondary_file_applicable",
            outcome=outcome,
            label=label,
            outcome_type=outcome_type,
            denominator_filter=file_filter,
            practical_significance_priority=False,
            notes="Secondary file-level participant-role model, restricted to file-applicable issues.",
        ))

    # RQ2 descriptive/conditional direct ownership and continuity outcomes.
    for outcome, label, outcome_type in [
        ("ownership_has_pre_issue_ownership", "Direct pre-issue ownership evidence", "binary"),
        ("ownership_has_post_issue_ownership", "Direct post-issue ownership evidence", "binary"),
        ("any_commenter_is_eventual_post_issue_owner", "Any commenter became post-issue owner", "binary"),
        ("share_post_issue_owners_with_pre_issue_repo_history", "Share of post-issue owners with prior repo history", "proportion"),
        ("share_post_issue_owners_with_pre_issue_file_history", "Share of post-issue owners with prior linked-file history", "proportion"),
        ("pre_post_owner_jaccard", "Pre/post owner Jaccard", "proportion"),
    ]:
        specs.append(OutcomeSpec(
            dataset="rq2",
            family="ownership_descriptive_conditional",
            tier="descriptive_conditional",
            outcome=outcome,
            label=label,
            outcome_type=outcome_type,
            practical_significance_priority=False,
            notes="Descriptive/conditional ownership feature; interpret as context rather than primary ownership inference.",
        ))

    return specs


# -----------------------------------------------------------------------------
# CLI and IO helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build matched-set FE report for WONTFIX analysis datasets.")
    parser.add_argument("--rq1-dataset", default=DEFAULT_RQ1_DATASET)
    parser.add_argument("--rq2-dataset", default=DEFAULT_RQ2_DATASET)
    parser.add_argument("--rq3-dataset", default=DEFAULT_RQ3_DATASET)
    parser.add_argument("--analysis-qa", default=DEFAULT_ANALYSIS_QA)
    parser.add_argument("--comparison-qa", default=DEFAULT_COMPARISON_QA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-model-n", type=int, default=MIN_MODEL_N)
    parser.add_argument("--min-group-n", type=int, default=MIN_GROUP_N)
    parser.add_argument("--min-event-count", type=int, default=MIN_EVENT_COUNT)
    parser.add_argument("--repo-cluster-threshold", type=int, default=REPO_CLUSTER_THRESHOLD)
    parser.add_argument("--repo-covariance", choices=["hc3", "auto", "cluster_repo"], default="hc3")
    parser.add_argument("--png-dpi", type=int, default=220)
    parser.add_argument("--allow-missing-rq1", action="store_true")
    parser.add_argument("--allow-missing-rq2", action="store_true")
    parser.add_argument("--allow-missing-rq3", action="store_true")
    parser.add_argument("--allow-missing-qa", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_table(path: Path | str, required: bool = True) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Input table does not exist: {p}")
        return pd.DataFrame()
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix == ".json":
        return pd.read_json(p)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported table format for {p}. Expected parquet/csv/json/jsonl.")


def write_csv(df: pd.DataFrame, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def write_markdown(text: str, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>", "nat"}:
        return None
    return text


def normalize_analysis_set(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return "missing"
    lowered = text.lower().replace("-", "_").replace(" ", "_")
    lowered = lowered.replace("'", "").replace("’", "")
    if lowered in {"wontfix", "wont_fix", "won_t_fix", "wonfix"}:
        return "wontfix"
    if lowered in {"comparison", "control", "controls", "non_wontfix", "nonwontfix", "matched_control"}:
        return "comparison"
    return lowered


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    for col in df.columns:
        col_l = str(col).lower()
        for candidate in candidates:
            if candidate.lower() in col_l:
                return col
    if required:
        raise KeyError(f"Required column not found. Tried: {candidates}")
    return None


def to_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")


# -----------------------------------------------------------------------------
# Dataset normalization and design summaries
# -----------------------------------------------------------------------------


def normalize_analysis_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    repo_col = find_col(out, ["repo_full_name", "repo_name", "full_name", "repo"], required=True)
    analysis_col = find_col(out, ["analysis_set", "comparison_group", "group"], required=False)
    issue_id_col = find_col(out, ["issue_id", "id"], required=False)
    issue_num_col = find_col(out, ["issue_number", "number"], required=False)
    matched_col = find_col(out, ["matched_set_id"], required=False)

    if repo_col != "repo_full_name":
        out["repo_full_name"] = out[repo_col].astype(str)
    else:
        out["repo_full_name"] = out["repo_full_name"].astype(str)

    if analysis_col:
        out["analysis_set"] = out[analysis_col].apply(normalize_analysis_set)
    elif "is_wontfix" in out.columns:
        out["analysis_set"] = to_numeric(out["is_wontfix"]).fillna(0).astype(int).map({1: "wontfix", 0: "comparison"})
    else:
        out["analysis_set"] = "missing"

    out["is_wontfix"] = (out["analysis_set"] == "wontfix").astype(int)
    out["is_comparison"] = (out["analysis_set"] == "comparison").astype(int)
    out["dataset_name"] = dataset_name

    if issue_id_col and issue_id_col != "issue_id":
        out["issue_id"] = out[issue_id_col].astype(str)
    elif "issue_id" in out.columns:
        out["issue_id"] = out["issue_id"].astype(str)
    else:
        out["issue_id"] = pd.NA

    if issue_num_col and issue_num_col != "issue_number":
        out["issue_number"] = to_numeric(out[issue_num_col])
    elif "issue_number" in out.columns:
        out["issue_number"] = to_numeric(out["issue_number"])
    else:
        out["issue_number"] = pd.NA

    if matched_col and matched_col != "matched_set_id":
        out["matched_set_id"] = out[matched_col]
    elif "matched_set_id" not in out.columns:
        out["matched_set_id"] = pd.NA
    out["matched_set_id"] = out["matched_set_id"].apply(clean_text)
    out["has_matched_set"] = out["matched_set_id"].notna().astype(int)

    return out


def get_metric_value(qa_df: pd.DataFrame, metric_name: str, default: Any = None) -> Any:
    if qa_df is None or qa_df.empty:
        return default
    metric_cols = [c for c in qa_df.columns if str(c).lower() in {"metric", "name", "key"}]
    value_cols = [c for c in qa_df.columns if str(c).lower() in {"value", "metric_value"}]
    if metric_cols and value_cols:
        metric_col = metric_cols[0]
        value_col = value_cols[0]
        sub = qa_df[qa_df[metric_col].astype(str) == metric_name]
        if not sub.empty:
            return sub[value_col].iloc[0]
    if metric_name in qa_df.columns and len(qa_df) > 0:
        return qa_df[metric_name].iloc[0]
    return default


def build_dataset_readiness(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in datasets.items():
        if df is None or df.empty:
            rows.append({
                "dataset": name,
                "rows": 0,
                "repos": 0,
                "matched_sets": 0,
                "rows_with_matched_set": 0,
                "global_usable_for_matched_set_fe_rows": 0,
                "wontfix_rows": 0,
                "comparison_rows": 0,
                "wontfix_rows_with_matched_set": 0,
                "comparison_rows_with_matched_set": 0,
            })
            continue
        usable_col = "usable_for_matched_set_fe" if "usable_for_matched_set_fe" in df.columns else None
        rows.append({
            "dataset": name,
            "rows": int(len(df)),
            "repos": int(df["repo_full_name"].nunique()) if "repo_full_name" in df.columns else 0,
            "matched_sets": int(df["matched_set_id"].dropna().nunique()) if "matched_set_id" in df.columns else 0,
            "rows_with_matched_set": int(df["matched_set_id"].notna().sum()) if "matched_set_id" in df.columns else 0,
            "global_usable_for_matched_set_fe_rows": int(to_numeric(df[usable_col]).fillna(0).sum()) if usable_col else 0,
            "wontfix_rows": int((df["analysis_set"] == "wontfix").sum()) if "analysis_set" in df.columns else 0,
            "comparison_rows": int((df["analysis_set"] == "comparison").sum()) if "analysis_set" in df.columns else 0,
            "wontfix_rows_with_matched_set": int(((df["analysis_set"] == "wontfix") & df["matched_set_id"].notna()).sum()) if "analysis_set" in df.columns and "matched_set_id" in df.columns else 0,
            "comparison_rows_with_matched_set": int(((df["analysis_set"] == "comparison") & df["matched_set_id"].notna()).sum()) if "analysis_set" in df.columns and "matched_set_id" in df.columns else 0,
        })
    return pd.DataFrame(rows)


def build_matched_design_summary(
    datasets: dict[str, pd.DataFrame],
    analysis_qa: pd.DataFrame,
    comparison_qa: pd.DataFrame,
) -> pd.DataFrame:
    # Prefer analysis QA values because they reflect downstream dataset attachment.
    rows = []
    def add(metric: str, value: Any, note: str = "") -> None:
        rows.append({"metric": metric, "value": value, "note": note})

    population_df = next((df for df in datasets.values() if df is not None and not df.empty), pd.DataFrame())
    if not population_df.empty:
        controls_per_set = (
            population_df[population_df["matched_set_id"].notna()]
            .groupby("matched_set_id")["analysis_set"]
            .apply(lambda s: int((s == "comparison").sum()))
        )
        distribution = controls_per_set.value_counts().sort_index().to_dict()
    else:
        distribution = {}

    add("script_version", SCRIPT_VERSION)
    add("matching_unit", "WONTFIX issue / matched_set_id", "Each set contains one WONTFIX issue and up to N non-WONTFIX controls.")
    add("matching_ratio", "1:N, up to 3 controls per WONTFIX", "Actual ratio is reported by controls_per_wontfix_distribution_json.")
    add("max_controls_per_wontfix", get_metric_value(comparison_qa, "max_controls_per_wontfix", get_metric_value(analysis_qa, "max_controls_per_wontfix", 3)))
    add("same_repository_only", "yes", "Comparison-set QA should show zero cross-repo pairs/sets.")
    add("time_window_days", get_metric_value(comparison_qa, "time_window_days", get_metric_value(analysis_qa, "time_window_days", 180)))
    add("issue_type_definition", "broad issue type inferred from GitHub label names", "Uses label_names_json and label_payload_json; does not use author_type/type scalar columns.")
    add("issue_type_source", get_metric_value(comparison_qa, "issue_type_source", get_metric_value(analysis_qa, "issue_type_source", "label_columns_only")))
    add("matching_algorithm", "greedy score-based matching within repo/time/type constraints", "Ranks candidates by creation-time distance, comment-count difference, issue-type overlap, linked-PR status, and comparison bucket.")
    add("controls_selected_without_replacement", "yes", "QA should show no reused comparison issues.")
    add("unmatched_wontfix_count", get_metric_value(comparison_qa, "wontfix_issues_with_zero_matches", get_metric_value(analysis_qa, "wontfix_rows_missing_matched_set_id", None)))
    add("controls_per_wontfix_distribution_json", get_metric_value(analysis_qa, "controls_per_wontfix_distribution_json", get_metric_value(comparison_qa, "controls_per_wontfix_distribution_json", json.dumps(distribution))))
    add("matched_sets_valid_for_fe", get_metric_value(analysis_qa, "matched_sets_valid_for_fe", None))
    add("analysis_rows_usable_for_matched_set_fe", get_metric_value(analysis_qa, "analysis_rows_usable_for_matched_set_fe", None))
    add("cross_repo_sets", get_metric_value(analysis_qa, "matched_sets_cross_repo_in_analysis", get_metric_value(comparison_qa, "pair_rows_cross_repo", None)))
    add("id_number_conflict_rows", get_metric_value(analysis_qa, "matched_set_issue_id_number_conflict_rows", None))
    add("run_timestamp_utc", datetime.now(timezone.utc).isoformat())
    return pd.DataFrame(rows)


def render_design_summary_md(design_df: pd.DataFrame) -> str:
    lookup = {row["metric"]: row["value"] for row in design_df.to_dict(orient="records")}
    lines = [
        "# Matched comparison design summary",
        "",
        f"- **Matching unit:** {lookup.get('matching_unit', '')}",
        f"- **Matching ratio:** {lookup.get('matching_ratio', '')}",
        f"- **Maximum controls per WONTFIX:** {lookup.get('max_controls_per_wontfix', '')}",
        f"- **Same-repository matching:** {lookup.get('same_repository_only', '')}",
        f"- **Creation-time window:** ±{lookup.get('time_window_days', '')} days",
        f"- **Category/type definition:** {lookup.get('issue_type_definition', '')}",
        f"- **Issue-type source:** {lookup.get('issue_type_source', '')}",
        f"- **Algorithm:** {lookup.get('matching_algorithm', '')}",
        f"- **Without replacement:** {lookup.get('controls_selected_without_replacement', '')}",
        f"- **Unmatched WONTFIX issues:** {lookup.get('unmatched_wontfix_count', '')}",
        f"- **Controls-per-WONTFIX distribution:** `{lookup.get('controls_per_wontfix_distribution_json', '')}`",
        f"- **Valid matched sets for FE:** {lookup.get('matched_sets_valid_for_fe', '')}",
        f"- **Rows usable for matched-set FE:** {lookup.get('analysis_rows_usable_for_matched_set_fe', '')}",
        "",
        "Primary statistical model: `outcome ~ WONTFIX + C(matched_set_id)`, with standard errors clustered by `matched_set_id`.",
        "Repo fixed-effect models are reported as coarser robustness checks corresponding to the original analysis tier.",
        "",
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Modeling helpers
# -----------------------------------------------------------------------------


def apply_denominator_filter(df: pd.DataFrame, filter_expr: str | None) -> tuple[pd.DataFrame, str | None]:
    if not filter_expr:
        return df.copy(), None
    expr = filter_expr.strip()
    if "==" not in expr:
        return df.copy(), f"unsupported_denominator_filter:{expr}"
    left, right = [x.strip() for x in expr.split("==", 1)]
    if left not in df.columns:
        return df.iloc[0:0].copy(), f"denominator_filter_column_missing:{left}"
    try:
        right_value = float(right)
    except Exception:
        right_value = right.strip('"\'')
    if isinstance(right_value, float):
        mask = to_numeric(df[left]).fillna(np.nan) == right_value
    else:
        mask = df[left].astype(str) == str(right_value)
    return df[mask].copy(), None


def valid_matched_set_ids(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    grouped = df.groupby("matched_set_id", dropna=True)["analysis_set"].agg(
        has_wontfix=lambda s: bool((s == "wontfix").any()),
        has_comparison=lambda s: bool((s == "comparison").any()),
    )
    valid = grouped[grouped["has_wontfix"] & grouped["has_comparison"]]
    return set(valid.index.astype(str))


def prepare_model_data(
    df: pd.DataFrame,
    spec: OutcomeSpec,
    min_model_n: int,
    min_group_n: int,
    min_event_count: int,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    base = df.copy() if df is not None else pd.DataFrame()
    readiness: dict[str, Any] = {
        "dataset": spec.dataset,
        "family": spec.family,
        "tier": spec.tier,
        "outcome": spec.outcome,
        "outcome_label": spec.label,
        "outcome_type": spec.outcome_type,
        "denominator_filter": spec.denominator_filter,
        "rows_before_filter": int(len(base)),
        "rows_after_denominator_filter": 0,
        "rows_nonmissing_outcome": 0,
        "rows_with_matched_set": 0,
        "rows_after_valid_set_filter": 0,
        "matched_sets_before_filter": 0,
        "matched_sets_after_filter": 0,
        "n_wontfix": 0,
        "n_comparison": 0,
        "wontfix_nonmissing_rate": np.nan,
        "comparison_nonmissing_rate": np.nan,
        "event_count_wontfix": np.nan,
        "event_count_comparison": np.nan,
        "low_n_flag": 0,
        "low_group_n_flag": 0,
        "low_event_count_flag": 0,
        "status": "ok",
        "skip_reason": "",
    }

    if base.empty:
        readiness.update(status="empty_dataset", skip_reason="dataset_has_no_rows")
        return None, readiness
    if spec.outcome not in base.columns:
        readiness.update(status="missing_outcome", skip_reason="outcome_column_missing")
        return None, readiness
    if "matched_set_id" not in base.columns:
        readiness.update(status="missing_matched_set_id", skip_reason="matched_set_id_column_missing")
        return None, readiness
    if "analysis_set" not in base.columns or "is_wontfix" not in base.columns:
        readiness.update(status="missing_analysis_set", skip_reason="analysis_set_or_is_wontfix_missing")
        return None, readiness

    denom_df, denom_error = apply_denominator_filter(base, spec.denominator_filter)
    readiness["rows_after_denominator_filter"] = int(len(denom_df))
    if denom_error:
        readiness.update(status="denominator_filter_failed", skip_reason=denom_error)
        return None, readiness
    if denom_df.empty:
        readiness.update(status="empty_after_denominator_filter", skip_reason="no_rows_after_denominator_filter")
        return None, readiness

    denom_df = denom_df.copy()
    denom_df["_outcome"] = to_numeric(denom_df[spec.outcome])
    denom_df["is_wontfix"] = to_numeric(denom_df["is_wontfix"]).fillna(0).astype(int)

    # Nonmissing rates by analysis set before dropping rows.
    for group, key in [("wontfix", "wontfix_nonmissing_rate"), ("comparison", "comparison_nonmissing_rate")]:
        sub = denom_df[denom_df["analysis_set"] == group]
        readiness[key] = float(sub["_outcome"].notna().mean()) if len(sub) else np.nan

    model_df = denom_df[denom_df["_outcome"].notna()].copy()
    readiness["rows_nonmissing_outcome"] = int(len(model_df))
    model_df = model_df[model_df["matched_set_id"].notna()].copy()
    readiness["rows_with_matched_set"] = int(len(model_df))
    readiness["matched_sets_before_filter"] = int(model_df["matched_set_id"].nunique()) if not model_df.empty else 0

    if model_df.empty:
        readiness.update(status="no_nonmissing_matched_rows", skip_reason="no_rows_with_outcome_and_matched_set")
        return None, readiness

    valid_ids = valid_matched_set_ids(model_df)
    model_df = model_df[model_df["matched_set_id"].astype(str).isin(valid_ids)].copy()
    readiness["rows_after_valid_set_filter"] = int(len(model_df))
    readiness["matched_sets_after_filter"] = int(model_df["matched_set_id"].nunique()) if not model_df.empty else 0
    readiness["n_wontfix"] = int((model_df["analysis_set"] == "wontfix").sum()) if not model_df.empty else 0
    readiness["n_comparison"] = int((model_df["analysis_set"] == "comparison").sum()) if not model_df.empty else 0

    if model_df.empty or readiness["matched_sets_after_filter"] == 0:
        readiness.update(status="no_valid_matched_sets", skip_reason="no_sets_with_both_wontfix_and_comparison_after_outcome_filter")
        return None, readiness

    if model_df["_outcome"].nunique(dropna=True) < 2:
        readiness.update(status="constant_outcome", skip_reason="outcome_has_no_variation_after_filtering")
        return None, readiness

    if readiness["rows_after_valid_set_filter"] < min_model_n:
        readiness["low_n_flag"] = 1
    if readiness["n_wontfix"] < min_group_n or readiness["n_comparison"] < min_group_n:
        readiness["low_group_n_flag"] = 1

    if spec.outcome_type == "binary":
        w_events = int(model_df.loc[model_df["analysis_set"] == "wontfix", "_outcome"].fillna(0).sum())
        c_events = int(model_df.loc[model_df["analysis_set"] == "comparison", "_outcome"].fillna(0).sum())
        readiness["event_count_wontfix"] = w_events
        readiness["event_count_comparison"] = c_events
        if w_events < min_event_count or c_events < min_event_count:
            readiness["low_event_count_flag"] = 1

    if readiness["low_n_flag"] or readiness["low_group_n_flag"]:
        readiness.update(status="insufficient_data", skip_reason="below_minimum_n_or_group_threshold")
        return None, readiness

    model_df["matched_set_id"] = model_df["matched_set_id"].astype(str)
    model_df["repo_full_name"] = model_df["repo_full_name"].astype(str)
    return model_df, readiness


def pooled_sd(x: pd.Series, y: pd.Series) -> float:
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    denom = max(len(x) + len(y) - 2, 1)
    val = math.sqrt(((len(x) - 1) * sx * sx + (len(y) - 1) * sy * sy) / denom)
    return float(val) if np.isfinite(val) else np.nan


def compute_descriptive_effects(model_df: pd.DataFrame, spec: OutcomeSpec) -> dict[str, Any]:
    w = model_df[model_df["analysis_set"] == "wontfix"]["_outcome"].astype(float)
    c = model_df[model_df["analysis_set"] == "comparison"]["_outcome"].astype(float)
    raw_w_mean = float(w.mean()) if len(w) else np.nan
    raw_c_mean = float(c.mean()) if len(c) else np.nan
    raw_diff = raw_w_mean - raw_c_mean if np.isfinite(raw_w_mean) and np.isfinite(raw_c_mean) else np.nan
    psd = pooled_sd(w, c)
    standardized = raw_diff / psd if psd and np.isfinite(psd) and psd != 0 else np.nan

    set_rows = []
    for _, sub in model_df.groupby("matched_set_id", dropna=True):
        sw = sub[sub["analysis_set"] == "wontfix"]["_outcome"].astype(float)
        sc = sub[sub["analysis_set"] == "comparison"]["_outcome"].astype(float)
        if len(sw) and len(sc):
            set_rows.append({
                "matched_set_id": sub["matched_set_id"].iloc[0],
                "w_mean": float(sw.mean()),
                "c_mean": float(sc.mean()),
                "diff": float(sw.mean() - sc.mean()),
            })
    set_df = pd.DataFrame(set_rows)
    return {
        "raw_wontfix_mean": raw_w_mean,
        "raw_comparison_mean": raw_c_mean,
        "raw_difference": raw_diff,
        "raw_wontfix_sd": float(w.std(ddof=1)) if len(w) > 1 else np.nan,
        "raw_comparison_sd": float(c.std(ddof=1)) if len(c) > 1 else np.nan,
        "standardized_effect_raw_d": standardized,
        "within_set_wontfix_mean": float(set_df["w_mean"].mean()) if not set_df.empty else np.nan,
        "within_set_comparison_mean": float(set_df["c_mean"].mean()) if not set_df.empty else np.nan,
        "within_set_difference": float(set_df["diff"].mean()) if not set_df.empty else np.nan,
    }


def fit_ols_formula(
    model_df: pd.DataFrame,
    formula: str,
    cov_type: str,
    cov_kwds: dict[str, Any] | None = None,
):
    if smf is None:
        raise RuntimeError("statsmodels is not installed; cannot fit FE models.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.ols(formula, data=model_df)
        if cov_kwds is None:
            return model.fit(cov_type=cov_type)
        return model.fit(cov_type=cov_type, cov_kwds=cov_kwds)


def extract_model_result(
    fit,
    model_df: pd.DataFrame,
    spec: OutcomeSpec,
    model_type: str,
    covariance: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = fit.params
    if "is_wontfix" not in params.index:
        raise RuntimeError("Model did not estimate is_wontfix coefficient.")
    conf = fit.conf_int()
    desc = compute_descriptive_effects(model_df, spec)
    coef = float(params["is_wontfix"])
    row = {
        "dataset": spec.dataset,
        "family": spec.family,
        "tier": spec.tier,
        "outcome": spec.outcome,
        "outcome_label": spec.label,
        "outcome_type": spec.outcome_type,
        "model_type": model_type,
        "covariance": covariance,
        "model_status": "ok",
        "error_message": "",
        "n_issues": int(len(model_df)),
        "n_matched_sets": int(model_df["matched_set_id"].nunique()) if "matched_set_id" in model_df.columns else np.nan,
        "n_repos": int(model_df["repo_full_name"].nunique()) if "repo_full_name" in model_df.columns else np.nan,
        "n_wontfix": int((model_df["analysis_set"] == "wontfix").sum()),
        "n_comparison": int((model_df["analysis_set"] == "comparison").sum()),
        "coef_wontfix": coef,
        "std_error": float(fit.bse["is_wontfix"]),
        "ci_low": float(conf.loc["is_wontfix", 0]),
        "ci_high": float(conf.loc["is_wontfix", 1]),
        "p_value": float(fit.pvalues["is_wontfix"]),
        "df_resid": float(fit.df_resid),
        "rsquared": float(getattr(fit, "rsquared", np.nan)),
        "denominator_filter": spec.denominator_filter,
        "requires_comments": int(spec.requires_comments),
        "notes": spec.notes,
    }
    row.update(desc)
    row["percentage_point_effect"] = 100.0 * coef if spec.outcome_type in {"binary", "proportion"} else np.nan
    row["approx_percent_effect"] = 100.0 * (math.exp(coef) - 1.0) if spec.outcome_type == "log_count" and np.isfinite(coef) else np.nan
    row["practical_interpretation"] = build_practical_interpretation(row, spec)
    if extra:
        row.update(extra)
    return row


def failed_result_row(spec: OutcomeSpec, model_type: str, covariance: str, status: str, error: str, model_df: pd.DataFrame | None = None) -> dict[str, Any]:
    row = {
        "dataset": spec.dataset,
        "family": spec.family,
        "tier": spec.tier,
        "outcome": spec.outcome,
        "outcome_label": spec.label,
        "outcome_type": spec.outcome_type,
        "model_type": model_type,
        "covariance": covariance,
        "model_status": status,
        "error_message": error,
        "n_issues": int(len(model_df)) if model_df is not None else 0,
        "n_matched_sets": int(model_df["matched_set_id"].nunique()) if model_df is not None and "matched_set_id" in model_df.columns else 0,
        "n_repos": int(model_df["repo_full_name"].nunique()) if model_df is not None and "repo_full_name" in model_df.columns else 0,
        "n_wontfix": int((model_df["analysis_set"] == "wontfix").sum()) if model_df is not None and "analysis_set" in model_df.columns else 0,
        "n_comparison": int((model_df["analysis_set"] == "comparison").sum()) if model_df is not None and "analysis_set" in model_df.columns else 0,
        "coef_wontfix": np.nan,
        "std_error": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": np.nan,
        "p_value_fdr_bh": np.nan,
        "reject_fdr_bh_05": False,
        "df_resid": np.nan,
        "rsquared": np.nan,
        "raw_wontfix_mean": np.nan,
        "raw_comparison_mean": np.nan,
        "raw_difference": np.nan,
        "raw_wontfix_sd": np.nan,
        "raw_comparison_sd": np.nan,
        "standardized_effect_raw_d": np.nan,
        "within_set_wontfix_mean": np.nan,
        "within_set_comparison_mean": np.nan,
        "within_set_difference": np.nan,
        "percentage_point_effect": np.nan,
        "approx_percent_effect": np.nan,
        "practical_interpretation": "",
        "denominator_filter": spec.denominator_filter,
        "requires_comments": int(spec.requires_comments),
        "notes": spec.notes,
    }
    return row


def fit_matched_set_fe(model_df: pd.DataFrame, spec: OutcomeSpec) -> dict[str, Any]:
    try:
        formula = "_outcome ~ is_wontfix + C(matched_set_id)"
        fit = fit_ols_formula(
            model_df,
            formula=formula,
            cov_type="cluster",
            cov_kwds={"groups": model_df["matched_set_id"]},
        )
        return extract_model_result(fit, model_df, spec, model_type="matched_set_fe", covariance="cluster_matched_set")
    except Exception as exc:
        return failed_result_row(spec, "matched_set_fe", "cluster_matched_set", "model_failed", str(exc), model_df)


def fit_repo_fe(model_df: pd.DataFrame, spec: OutcomeSpec, repo_covariance: str, repo_cluster_threshold: int) -> dict[str, Any]:
    try:
        n_repos = int(model_df["repo_full_name"].nunique())
        formula = "_outcome ~ is_wontfix + C(repo_full_name)"
        if repo_covariance == "cluster_repo" or (repo_covariance == "auto" and n_repos >= repo_cluster_threshold):
            cov_type = "cluster"
            cov_kwds = {"groups": model_df["repo_full_name"]}
            covariance_label = "cluster_repo"
            model_type = "repo_fe_clustered_repo"
        else:
            cov_type = "HC3"
            cov_kwds = None
            covariance_label = "HC3"
            model_type = "repo_fe_hc3"
        fit = fit_ols_formula(model_df, formula=formula, cov_type=cov_type, cov_kwds=cov_kwds)
        return extract_model_result(fit, model_df, spec, model_type=model_type, covariance=covariance_label)
    except Exception as exc:
        return failed_result_row(spec, "repo_fe", repo_covariance, "model_failed", str(exc), model_df)


def build_practical_interpretation(row: dict[str, Any], spec: OutcomeSpec) -> str:
    coef = row.get("coef_wontfix")
    d = row.get("standardized_effect_raw_d")
    if coef is None or pd.isna(coef):
        return ""
    direction = "higher" if coef > 0 else "lower" if coef < 0 else "no different"
    abs_coef = abs(float(coef))
    parts: list[str] = []
    if spec.outcome_type in {"binary", "proportion"}:
        parts.append(f"WONTFIX issues are estimated to be {abs_coef * 100:.2f} percentage points {direction} on {spec.label.lower()} within matched sets.")
    elif spec.outcome_type == "log_count":
        pct = 100.0 * (math.exp(float(coef)) - 1.0)
        direction_word = "higher" if pct > 0 else "lower" if pct < 0 else "unchanged"
        parts.append(f"WONTFIX issues are estimated to have approximately {abs(pct):.1f}% {direction_word} {spec.label.lower()} within matched sets.")
    else:
        parts.append(f"WONTFIX issues are estimated to be {abs_coef:.4f} units {direction} on {spec.label.lower()} within matched sets.")
    if d is not None and pd.notna(d):
        magnitude = "small" if abs(float(d)) < 0.2 else "moderate" if abs(float(d)) < 0.5 else "large"
        parts.append(f"The raw standardized difference is {float(d):.3f} SDs, conventionally {magnitude}.")
        if abs(float(d)) < 0.2:
            parts.append("Interpret this as an aggregate pattern rather than a standalone classifier or intervention signal.")
    return " ".join(parts)


# -----------------------------------------------------------------------------
# Results postprocessing
# -----------------------------------------------------------------------------


def add_fdr_by_family(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return results_df
    out = results_df.copy()
    out["p_value_fdr_bh"] = np.nan
    out["reject_fdr_bh_05"] = False
    if multipletests is None or "p_value" not in out.columns:
        return out
    group_cols = ["dataset", "family", "tier", "model_type"]
    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        idx = list(idx)
        valid_idx = [i for i in idx if pd.notna(out.loc[i, "p_value"])]
        if not valid_idx:
            continue
        reject, p_adj, _, _ = multipletests(out.loc[valid_idx, "p_value"].astype(float), method="fdr_bh")
        out.loc[valid_idx, "p_value_fdr_bh"] = p_adj
        out.loc[valid_idx, "reject_fdr_bh_05"] = reject
    return out


def compare_primary_and_robustness(msfe: pd.DataFrame, repo: pd.DataFrame) -> pd.DataFrame:
    if msfe.empty or repo.empty:
        return pd.DataFrame()
    left = msfe[msfe["model_status"] == "ok"].copy()
    right = repo[repo["model_status"] == "ok"].copy()
    if left.empty or right.empty:
        return pd.DataFrame()
    cols = ["dataset", "family", "tier", "outcome", "outcome_label"]
    lcols = cols + ["coef_wontfix", "p_value", "p_value_fdr_bh", "n_issues", "n_matched_sets", "model_type"]
    rcols = cols + ["coef_wontfix", "p_value", "p_value_fdr_bh", "n_issues", "n_repos", "model_type"]
    merged = left[lcols].merge(right[rcols], on=cols, how="outer", suffixes=("_matched_set_fe", "_repo_fe"))

    def direction(value: Any) -> str:
        if pd.isna(value):
            return "missing"
        if float(value) > 0:
            return "positive"
        if float(value) < 0:
            return "negative"
        return "zero"

    merged["matched_set_direction"] = merged["coef_wontfix_matched_set_fe"].apply(direction)
    merged["repo_fe_direction"] = merged["coef_wontfix_repo_fe"].apply(direction)
    merged["same_direction_flag"] = (
        (merged["matched_set_direction"] == merged["repo_fe_direction"])
        & ~merged["matched_set_direction"].isin(["missing", "zero"])
    )
    merged["both_significant_flag"] = (
        (pd.to_numeric(merged["p_value_matched_set_fe"], errors="coerce") < 0.05)
        & (pd.to_numeric(merged["p_value_repo_fe"], errors="coerce") < 0.05)
    )
    merged["primary_significant_only_flag"] = (
        (pd.to_numeric(merged["p_value_matched_set_fe"], errors="coerce") < 0.05)
        & ~(pd.to_numeric(merged["p_value_repo_fe"], errors="coerce") < 0.05)
    )
    merged["robustness_significant_only_flag"] = (
        ~(pd.to_numeric(merged["p_value_matched_set_fe"], errors="coerce") < 0.05)
        & (pd.to_numeric(merged["p_value_repo_fe"], errors="coerce") < 0.05)
    )

    def consistency(row: pd.Series) -> str:
        if pd.isna(row.get("coef_wontfix_matched_set_fe")) and pd.isna(row.get("coef_wontfix_repo_fe")):
            return "no_estimates"
        if pd.isna(row.get("coef_wontfix_matched_set_fe")):
            return "repo_only_no_primary_result"
        if pd.isna(row.get("coef_wontfix_repo_fe")):
            return "primary_only_no_repo_result"
        if row["matched_set_direction"] == "zero" and row["repo_fe_direction"] == "zero":
            return "both_near_zero"
        if row["same_direction_flag"]:
            return "consistent_direction"
        return "opposite_direction"

    merged["interpretation_consistency"] = merged.apply(consistency, axis=1)
    return merged


def build_practical_significance_summary(msfe_results: pd.DataFrame) -> pd.DataFrame:
    if msfe_results.empty:
        return pd.DataFrame()
    keep = msfe_results[(msfe_results["model_status"] == "ok") & (msfe_results["tier"].isin(["primary", "secondary"]))].copy()
    if keep.empty:
        return pd.DataFrame()
    cols = [
        "dataset", "family", "tier", "outcome", "outcome_label", "outcome_type",
        "raw_wontfix_mean", "raw_comparison_mean", "raw_difference",
        "within_set_difference", "coef_wontfix", "ci_low", "ci_high", "p_value", "p_value_fdr_bh",
        "standardized_effect_raw_d", "percentage_point_effect", "approx_percent_effect",
        "n_issues", "n_matched_sets", "practical_interpretation", "notes",
    ]
    return keep[[c for c in cols if c in keep.columns]].sort_values(["dataset", "family", "tier", "outcome"])


def render_practical_significance_md(practical_df: pd.DataFrame, max_rows: int = 30) -> str:
    if practical_df.empty:
        return "# Practical significance summary\n\nNo practical-significance rows were generated.\n"
    lines = ["# Practical significance summary", ""]
    lines.append("This table emphasizes magnitude and interpretation, not only p-values. Small standardized effects should be treated as aggregate patterns, not standalone prediction signals.")
    lines.append("")
    display_cols = ["family", "outcome_label", "coef_wontfix", "standardized_effect_raw_d", "percentage_point_effect", "approx_percent_effect", "practical_interpretation"]
    subset = practical_df[[c for c in display_cols if c in practical_df.columns]].head(max_rows).copy()
    for col in ["coef_wontfix", "standardized_effect_raw_d", "percentage_point_effect", "approx_percent_effect"]:
        if col in subset.columns:
            subset[col] = pd.to_numeric(subset[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    lines.append(subset.to_markdown(index=False))
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def safe_filename(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "plot"


def plot_matched_sets_retained(readiness_df: pd.DataFrame, path: Path, dpi: int) -> None:
    if readiness_df.empty or "matched_sets_after_filter" not in readiness_df.columns:
        return
    df = readiness_df[readiness_df["status"].isin(["ok", "insufficient_data"])].copy()
    df = df[pd.to_numeric(df["matched_sets_after_filter"], errors="coerce").notna()]
    if df.empty:
        return
    df["plot_label"] = df["family"].astype(str) + ": " + df["outcome_label"].astype(str)
    df = df.sort_values("matched_sets_after_filter", ascending=True).tail(35)
    fig_height = max(6, 0.26 * len(df))
    plt.figure(figsize=(11, fig_height))
    plt.barh(df["plot_label"], pd.to_numeric(df["matched_sets_after_filter"], errors="coerce"))
    plt.xlabel("Matched sets retained")
    plt.title("Matched sets retained after outcome-specific filtering")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_effects(results_df: pd.DataFrame, family: str, title: str, path: Path, dpi: int, max_rows: int = 25) -> None:
    if results_df.empty:
        return
    df = results_df[
        (results_df["model_status"] == "ok")
        & (results_df["family"] == family)
        & (results_df["model_type"] == "matched_set_fe")
    ].copy()
    if df.empty:
        return
    df = df[df["tier"].isin(["primary", "secondary", "secondary_file_applicable"])]
    df = df.sort_values("coef_wontfix", ascending=True).tail(max_rows)
    if df.empty:
        return

    # Matplotlib versions paired with newer pandas can fail when passed Series
    # objects because matplotlib internally tries Series[:, None]. Convert all
    # plotted vectors to numpy/list objects before calling errorbar.
    labels = df["outcome_label"].astype(str).tolist()
    coef = pd.to_numeric(df["coef_wontfix"], errors="coerce").to_numpy(dtype=float)
    ci_low = pd.to_numeric(df["ci_low"], errors="coerce").to_numpy(dtype=float)
    ci_high = pd.to_numeric(df["ci_high"], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(coef) & np.isfinite(ci_low) & np.isfinite(ci_high)
    if not valid.any():
        return
    labels = [label for label, keep in zip(labels, valid) if keep]
    coef = coef[valid]
    ci_low = ci_low[valid]
    ci_high = ci_high[valid]

    lower_err = np.maximum(coef - ci_low, 0)
    upper_err = np.maximum(ci_high - coef, 0)
    xerr = np.vstack([lower_err, upper_err])

    plt.figure(figsize=(10, max(5, 0.33 * len(labels))))
    plt.errorbar(coef, labels, xerr=xerr, fmt="o", capsize=3)
    plt.axvline(0, linewidth=1)
    plt.xlabel("WONTFIX coefficient with 95% CI")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_ms_vs_repo(compare_df: pd.DataFrame, path: Path, dpi: int) -> None:
    if compare_df.empty:
        return
    df = compare_df.copy()
    x = pd.to_numeric(df.get("coef_wontfix_matched_set_fe"), errors="coerce")
    y = pd.to_numeric(df.get("coef_wontfix_repo_fe"), errors="coerce")
    valid = x.notna() & y.notna()
    if valid.sum() < 2:
        return
    x = x[valid]
    y = y[valid]
    labels = df.loc[valid, "outcome_label"].astype(str)
    plt.figure(figsize=(8, 7))
    plt.scatter(x, y)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
        plt.plot([lo, hi], [lo, hi], linewidth=1)
    for xv, yv, label in zip(x, y, labels):
        if abs(xv) >= x.abs().quantile(0.85) or abs(yv) >= y.abs().quantile(0.85):
            plt.annotate(label[:30], (xv, yv), fontsize=7)
    plt.xlabel("Matched-set FE coefficient")
    plt.ylabel("Repo-FE robustness coefficient")
    plt.title("Matched-set FE vs repo-FE WONTFIX coefficients")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------------------


def render_report(
    design_md: str,
    dataset_readiness: pd.DataFrame,
    outcome_readiness: pd.DataFrame,
    msfe_results: pd.DataFrame,
    repo_results: pd.DataFrame,
    comparison_df: pd.DataFrame,
    practical_df: pd.DataFrame,
) -> str:
    def mini_table(df: pd.DataFrame, max_rows: int = 20) -> str:
        if df.empty:
            return "_No rows._"
        return df.head(max_rows).to_markdown(index=False)

    lines = [
        "# Matched-set fixed-effect report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Script version: `{SCRIPT_VERSION}`",
        "",
        "## Design summary",
        "",
        design_md.replace("# Matched comparison design summary\n\n", ""),
        "",
        "## Model specification",
        "",
        "Primary model: `outcome ~ WONTFIX + C(matched_set_id)`, with standard errors clustered by matched set.",
        "",
        "Robustness model: `outcome ~ WONTFIX + C(repo_full_name)`, with HC3 robust standard errors by default. This corresponds to the original repo-aware analysis tier.",
        "",
        "Outcome-specific valid matched sets are recomputed after applying denominator filters and dropping missing outcome rows.",
        "",
        "## Dataset readiness",
        "",
        mini_table(dataset_readiness),
        "",
        "## Primary matched-set FE results overview",
        "",
    ]
    if not msfe_results.empty:
        ok = msfe_results[msfe_results["model_status"] == "ok"].copy()
        overview_cols = ["family", "tier", "outcome_label", "coef_wontfix", "ci_low", "ci_high", "p_value", "p_value_fdr_bh", "n_issues", "n_matched_sets"]
        lines.append(mini_table(ok[[c for c in overview_cols if c in ok.columns]].sort_values(["family", "tier", "outcome_label"]), max_rows=40))
    else:
        lines.append("_No matched-set FE results._")
    lines += [
        "",
        "## Robustness consistency",
        "",
        mini_table(comparison_df[[c for c in ["family", "tier", "outcome_label", "coef_wontfix_matched_set_fe", "coef_wontfix_repo_fe", "same_direction_flag", "interpretation_consistency"] if c in comparison_df.columns]], max_rows=40) if not comparison_df.empty else "_No robustness comparison rows._",
        "",
        "## Practical significance",
        "",
        "Small standardized effects should be interpreted as aggregate patterns rather than classifier-ready signals.",
        "",
    ]
    if not practical_df.empty:
        lines.append(mini_table(practical_df[[c for c in ["family", "outcome_label", "coef_wontfix", "standardized_effect_raw_d", "percentage_point_effect", "approx_percent_effect", "practical_interpretation"] if c in practical_df.columns]], max_rows=30))
    else:
        lines.append("_No practical-significance rows._")
    lines += [
        "",
        "## Limitations and interpretation guardrails",
        "",
        "- Matched-set FE estimates are associational, not causal.",
        "- Repo-level differences are absorbed by matched-set FE because matched sets are same-repository.",
        "- Repo-FE robustness is a coarser replication of the original analysis, not the primary design.",
        "- File-level ownership results are secondary because file/commit evidence coverage can be affected by WONTFIX status.",
        "- Outcome-specific missingness can reduce the number of usable matched sets; inspect `model_readiness_by_outcome.csv` before interpreting coefficients.",
        "",
    ]
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def load_datasets(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    inputs = {
        "rq1": (args.rq1_dataset, args.allow_missing_rq1),
        "rq2": (args.rq2_dataset, args.allow_missing_rq2),
        "rq3": (args.rq3_dataset, args.allow_missing_rq3),
    }
    for name, (path, allow_missing) in inputs.items():
        df = read_table(path, required=not allow_missing)
        datasets[name] = normalize_analysis_dataset(df, name) if not df.empty else pd.DataFrame()
    return datasets


def run_models(args: argparse.Namespace, datasets: dict[str, pd.DataFrame], specs: list[OutcomeSpec]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    readiness_rows: list[dict[str, Any]] = []
    msfe_rows: list[dict[str, Any]] = []
    repo_rows: list[dict[str, Any]] = []

    for spec in specs:
        df = datasets.get(spec.dataset, pd.DataFrame())
        model_df, readiness = prepare_model_data(
            df,
            spec,
            min_model_n=args.min_model_n,
            min_group_n=args.min_group_n,
            min_event_count=args.min_event_count,
        )
        readiness_rows.append(readiness)
        if readiness["status"] != "ok" or model_df is None:
            msfe_rows.append(failed_result_row(spec, "matched_set_fe", "cluster_matched_set", readiness["status"], readiness.get("skip_reason", ""), model_df))
            repo_rows.append(failed_result_row(spec, "repo_fe_hc3", "HC3", readiness["status"], readiness.get("skip_reason", ""), model_df))
            continue
        if spec.run_primary:
            msfe_rows.append(fit_matched_set_fe(model_df, spec))
        if spec.run_repo_fe:
            repo_rows.append(fit_repo_fe(model_df, spec, args.repo_covariance, args.repo_cluster_threshold))

    readiness_df = pd.DataFrame(readiness_rows)
    msfe_df = add_fdr_by_family(pd.DataFrame(msfe_rows))
    repo_df = add_fdr_by_family(pd.DataFrame(repo_rows))
    return readiness_df, msfe_df, repo_df


def write_family_outputs(msfe_results: pd.DataFrame, out_dir: Path) -> None:
    if msfe_results.empty:
        return
    mapping = {
        "sentiment": "matched_set_fe_sentiment_results.csv",
        "emotion_issue_level": "matched_set_fe_emotion_issue_level_results.csv",
        "participation": "matched_set_fe_participation_results.csv",
        "ownership_repo_primary": "matched_set_fe_ownership_repo_primary_results.csv",
        "ownership_file_secondary": "matched_set_fe_ownership_file_secondary_results.csv",
        "ownership_descriptive_conditional": "matched_set_fe_ownership_descriptive_conditional_results.csv",
    }
    for family, filename in mapping.items():
        sub = msfe_results[msfe_results["family"] == family].copy()
        if not sub.empty:
            write_csv(sub, out_dir / filename)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    fig_dir = ensure_dir(out_dir / "figures")
    qa_dir = ensure_dir(out_dir / "qa")

    print(f"Running {SCRIPT_VERSION}")
    print(f"Output directory: {out_dir}")

    datasets = load_datasets(args)
    analysis_qa = read_table(args.analysis_qa, required=not args.allow_missing_qa)
    comparison_qa = read_table(args.comparison_qa, required=False)

    dataset_readiness = build_dataset_readiness(datasets)
    design_summary = build_matched_design_summary(datasets, analysis_qa, comparison_qa)
    design_md = render_design_summary_md(design_summary)

    specs = build_outcome_specs()
    outcome_readiness, msfe_results, repo_results = run_models(args, datasets, specs)
    comparison_results = compare_primary_and_robustness(msfe_results, repo_results)
    practical = build_practical_significance_summary(msfe_results)

    write_csv(design_summary, out_dir / "matched_design_summary.csv")
    write_markdown(design_md, out_dir / "matched_design_summary.md")
    write_csv(dataset_readiness, out_dir / "model_readiness_by_dataset.csv")
    write_csv(outcome_readiness, out_dir / "model_readiness_by_outcome.csv")
    write_csv(dataset_readiness, qa_dir / "model_readiness_by_dataset.csv")
    write_csv(outcome_readiness, qa_dir / "model_readiness_by_outcome.csv")

    write_csv(msfe_results, out_dir / "matched_set_fe_all_results.csv")
    write_family_outputs(msfe_results, out_dir)
    write_csv(repo_results, out_dir / "repo_fe_robustness_results.csv")
    write_csv(comparison_results, out_dir / "matched_set_vs_repo_fe_comparison.csv")
    write_csv(practical, out_dir / "practical_significance_summary.csv")
    write_markdown(render_practical_significance_md(practical), out_dir / "practical_significance_summary.md")

    plot_matched_sets_retained(outcome_readiness, fig_dir / "matched_sets_retained_by_outcome.png", args.png_dpi)
    plot_effects(msfe_results, "sentiment", "Primary sentiment matched-set FE effects", fig_dir / "primary_effects_sentiment.png", args.png_dpi)
    plot_effects(msfe_results, "participation", "Primary participation matched-set FE effects", fig_dir / "primary_effects_participation.png", args.png_dpi)
    plot_effects(msfe_results, "ownership_repo_primary", "Primary ownership repo-role matched-set FE effects", fig_dir / "primary_effects_ownership_repo_roles.png", args.png_dpi)
    plot_ms_vs_repo(comparison_results, fig_dir / "matched_set_vs_repo_fe_coefficients.png", args.png_dpi)

    report_md = render_report(
        design_md=design_md,
        dataset_readiness=dataset_readiness,
        outcome_readiness=outcome_readiness,
        msfe_results=msfe_results,
        repo_results=repo_results,
        comparison_df=comparison_results,
        practical_df=practical,
    )
    write_markdown(report_md, out_dir / "matched_set_fe_report.md")

    ok_primary = int(((msfe_results["model_status"] == "ok") & (msfe_results["tier"] == "primary")).sum()) if not msfe_results.empty else 0
    failed = int((msfe_results["model_status"] != "ok").sum()) if not msfe_results.empty else 0
    print(f"Primary matched-set FE models estimated successfully: {ok_primary}")
    print(f"Matched-set FE rows skipped/failed: {failed}")
    print(f"Wrote report: {out_dir / 'matched_set_fe_report.md'}")


if __name__ == "__main__":
    main()
