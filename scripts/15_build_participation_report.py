#!/usr/bin/env python3
"""
Build the participation analysis report for the WONTFIX pipeline.

This version implements:
  Part 1: Participation coverage and population QA
  Part 2: Core participation differences, WONTFIX vs comparison
  Part 3: Repo-aware effect sizes and simple repo-fixed-effect models
  Part 4: Participation + sentiment bridge

The script is designed to run after scripts/13_build_analysis_dataset.py and read
RQ3's merged issue-level analysis dataset.

Typical usage:
    python scripts/15_build_participation_report.py \
        --rq3-dataset data/final/analysis_dataset_rq3_issue_level_base.parquet \
        --analysis-qa logs/qa/analysis_dataset_qa_summary.csv \
        --output-dir outputs/participation_analysis

The implementation intentionally avoids type annotations so it fits the style of
most project scripts.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import statsmodels.formula.api as smf
except Exception:
    smf = None


# --- UTK brand colors ---
UTK_COLORS = {
    "orange": "#FF8200",
    "white": "#FFFFFF",
    "smokey_gray": "#58595B",
    "dark_gray": "#333333",
    "light_gray": "#A7A9AC",
}

ANALYSIS_COLORS = {
    "wontfix": UTK_COLORS["orange"],
    "comparison": UTK_COLORS["smokey_gray"],
    "pr_resolved": "#4C78A8",
    "closed_non_wontfix": "#72B7B2",
    "open": "#54A24B",
    "invalid": "#E45756",
    "missing": UTK_COLORS["light_gray"],
}

COVERAGE_NEUTRAL_COLORS = {
    "zero_comments": UTK_COLORS["light_gray"],
    "missing_participation_features": "#D9D9D9",
    "other": "#C7C7C7",
}

GROUP_LABELS = {
    "wontfix": "WONTFIX",
    "comparison": "Comparison",
}

DEFAULT_RQ3_DATASET = "data/final/analysis_dataset_rq3_issue_level_base.parquet"
DEFAULT_ANALYSIS_QA = "logs/qa/analysis_dataset_qa_summary.csv"
DEFAULT_OUTPUT_DIR = "outputs/participation_analysis"

PARTICIPATION_CORE_COLUMNS = [
    "comment_count",
    "zero_comment_flag",
    "unique_commenter_count",
    "issue_author_commented_flag",
    "num_distinct_non_author_commenters",
    "top_commenter_share",
    "comment_concentration_ratio",
    "non_author_comment_share",
    "top_2_commenters_share",
    "single_commenter_flag",
    "only_author_commented_flag",
    "first_comment_by_author_flag",
    "last_comment_by_author_flag",
    "participation_feature_coverage_flag",
    "has_participation_features",
    "usable_for_rq3",
]

RATIO_APPLICABILITY_COLUMNS = [
    "top_commenter_share",
    "comment_concentration_ratio",
    "non_author_comment_share",
    "top_2_commenters_share",
    "single_commenter_flag",
    "only_author_commented_flag",
]

PART2_NUMERIC_FEATURES = [
    {
        "feature": "comment_count",
        "label": "Comment count",
        "family": "volume",
        "requires_comments": False,
        "transform": None,
    },
    {
        "feature": "log1p_comment_count",
        "label": "log1p(comment count)",
        "family": "volume",
        "requires_comments": False,
        "transform": "log1p",
    },
    {
        "feature": "unique_commenter_count",
        "label": "Unique commenters",
        "family": "breadth",
        "requires_comments": False,
        "transform": None,
    },
    {
        "feature": "log1p_unique_commenter_count",
        "label": "log1p(unique commenters)",
        "family": "breadth",
        "requires_comments": False,
        "transform": "log1p",
    },
    {
        "feature": "num_distinct_non_author_commenters",
        "label": "Distinct non-author commenters",
        "family": "breadth",
        "requires_comments": False,
        "transform": None,
    },
    {
        "feature": "log1p_num_distinct_non_author_commenters",
        "label": "log1p(non-author commenters)",
        "family": "breadth",
        "requires_comments": False,
        "transform": "log1p",
    },
    {
        "feature": "top_commenter_share",
        "label": "Top commenter share",
        "family": "concentration",
        "requires_comments": True,
        "transform": None,
    },
    {
        "feature": "comment_concentration_ratio",
        "label": "Comment concentration ratio",
        "family": "concentration",
        "requires_comments": True,
        "transform": None,
    },
    {
        "feature": "non_author_comment_share",
        "label": "Non-author comment share",
        "family": "author_involvement",
        "requires_comments": True,
        "transform": None,
    },
]

PART2_BINARY_FEATURES = [
    {
        "feature": "zero_comment_flag",
        "label": "Zero-comment issue",
        "family": "coverage",
        "requires_comments": False,
    },
    {
        "feature": "issue_author_commented_flag",
        "label": "Issue author commented",
        "family": "author_involvement",
        "requires_comments": False,
    },
    {
        "feature": "has_any_non_author_comment",
        "label": "Any non-author commenter",
        "family": "breadth",
        "requires_comments": False,
    },
    {
        "feature": "multi_party_discussion_flag",
        "label": "Multiple commenters",
        "family": "breadth",
        "requires_comments": False,
    },
    {
        "feature": "single_commenter_flag",
        "label": "Single-commenter discussion",
        "family": "concentration",
        "requires_comments": True,
    },
    {
        "feature": "only_author_commented_flag",
        "label": "Only issue author commented",
        "family": "author_involvement",
        "requires_comments": True,
    },
    {
        "feature": "first_comment_by_author_flag",
        "label": "First comment by issue author",
        "family": "author_involvement",
        "requires_comments": True,
    },
    {
        "feature": "last_comment_by_author_flag",
        "label": "Last comment by issue author",
        "family": "author_involvement",
        "requires_comments": True,
    },
]


PART3_EFFECT_FEATURES = [
    {
        "feature": "log1p_comment_count",
        "label": "log1p(comment count)",
        "family": "volume",
        "feature_type": "numeric",
        "requires_comments": False,
    },
    {
        "feature": "log1p_unique_commenter_count",
        "label": "log1p(unique commenters)",
        "family": "breadth",
        "feature_type": "numeric",
        "requires_comments": False,
    },
    {
        "feature": "log1p_num_distinct_non_author_commenters",
        "label": "log1p(non-author commenters)",
        "family": "breadth",
        "feature_type": "numeric",
        "requires_comments": False,
    },
    {
        "feature": "top_commenter_share",
        "label": "Top commenter share",
        "family": "concentration",
        "feature_type": "numeric",
        "requires_comments": True,
    },
    {
        "feature": "comment_concentration_ratio",
        "label": "Comment concentration ratio",
        "family": "concentration",
        "feature_type": "numeric",
        "requires_comments": True,
    },
    {
        "feature": "non_author_comment_share",
        "label": "Non-author comment share",
        "family": "author_involvement",
        "feature_type": "numeric",
        "requires_comments": True,
    },
    {
        "feature": "zero_comment_flag",
        "label": "Zero-comment issue",
        "family": "coverage",
        "feature_type": "binary",
        "requires_comments": False,
    },
    {
        "feature": "issue_author_commented_flag",
        "label": "Issue author commented",
        "family": "author_involvement",
        "feature_type": "binary",
        "requires_comments": False,
    },
    {
        "feature": "has_any_non_author_comment",
        "label": "Any non-author commenter",
        "family": "breadth",
        "feature_type": "binary",
        "requires_comments": False,
    },
    {
        "feature": "multi_party_discussion_flag",
        "label": "Multiple commenters",
        "family": "breadth",
        "feature_type": "binary",
        "requires_comments": False,
    },
    {
        "feature": "single_commenter_flag",
        "label": "Single-commenter discussion",
        "family": "concentration",
        "feature_type": "binary",
        "requires_comments": True,
    },
    {
        "feature": "only_author_commented_flag",
        "label": "Only issue author commented",
        "family": "author_involvement",
        "feature_type": "binary",
        "requires_comments": True,
    },
    {
        "feature": "first_comment_by_author_flag",
        "label": "First comment by issue author",
        "family": "author_involvement",
        "feature_type": "binary",
        "requires_comments": True,
    },
    {
        "feature": "last_comment_by_author_flag",
        "label": "Last comment by issue author",
        "family": "author_involvement",
        "feature_type": "binary",
        "requires_comments": True,
    },
]


SENTIMENT_FEATURES = [
    {
        "feature": "mean_comment_sentiment",
        "label": "Mean comment sentiment",
        "family": "sentiment_level",
    },
    {
        "feature": "median_comment_sentiment",
        "label": "Median comment sentiment",
        "family": "sentiment_level",
    },
    {
        "feature": "min_comment_sentiment",
        "label": "Minimum comment sentiment",
        "family": "sentiment_extreme",
    },
    {
        "feature": "max_comment_sentiment",
        "label": "Maximum comment sentiment",
        "family": "sentiment_extreme",
    },
    {
        "feature": "std_comment_sentiment",
        "label": "Sentiment volatility",
        "family": "sentiment_variability",
    },
    {
        "feature": "negative_comment_share",
        "label": "Negative comment share",
        "family": "sentiment_share",
    },
    {
        "feature": "positive_comment_share",
        "label": "Positive comment share",
        "family": "sentiment_share",
    },
    {
        "feature": "comment_sentiment_change_late_minus_early",
        "label": "Late-minus-early sentiment change",
        "family": "sentiment_trajectory",
    },
    {
        "feature": "comment_sentiment_slope",
        "label": "Sentiment slope",
        "family": "sentiment_trajectory",
    },
]

PARTICIPATION_SENTIMENT_BRIDGE_FEATURES = [
    {
        "feature": "log1p_comment_count",
        "label": "log1p(comment count)",
        "family": "volume",
        "requires_comments": False,
    },
    {
        "feature": "log1p_unique_commenter_count",
        "label": "log1p(unique commenters)",
        "family": "breadth",
        "requires_comments": False,
    },
    {
        "feature": "log1p_num_distinct_non_author_commenters",
        "label": "log1p(non-author commenters)",
        "family": "breadth",
        "requires_comments": False,
    },
    {
        "feature": "top_commenter_share",
        "label": "Top commenter share",
        "family": "concentration",
        "requires_comments": True,
    },
    {
        "feature": "comment_concentration_ratio",
        "label": "Comment concentration ratio",
        "family": "concentration",
        "requires_comments": True,
    },
    {
        "feature": "non_author_comment_share",
        "label": "Non-author comment share",
        "family": "author_involvement",
        "requires_comments": True,
    },
    {
        "feature": "issue_author_commented_flag",
        "label": "Issue author commented",
        "family": "author_involvement",
        "requires_comments": False,
    },
    {
        "feature": "has_any_non_author_comment",
        "label": "Any non-author commenter",
        "family": "breadth",
        "requires_comments": False,
    },
    {
        "feature": "last_comment_by_author_flag",
        "label": "Last comment by issue author",
        "family": "author_involvement",
        "requires_comments": True,
    },
]

PART4_SENTIMENT_MODEL_OUTCOMES = [
    "mean_comment_sentiment",
    "positive_comment_share",
    "negative_comment_share",
    "std_comment_sentiment",
    "comment_sentiment_slope",
]

PART4_CONTROL_FEATURES = [
    "log1p_comment_count",
    "log1p_unique_commenter_count",
    "log1p_num_distinct_non_author_commenters",
    "top_commenter_share",
    "non_author_comment_share",
]

MIN_MODEL_GROUP_N = 10
MIN_MODEL_TOTAL_N = 40


# -----------------------------
# CLI / I/O helpers
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build participation coverage and WONTFIX-vs-comparison participation report."
    )
    parser.add_argument("--rq3-dataset", default=DEFAULT_RQ3_DATASET)
    parser.add_argument("--analysis-qa", default=DEFAULT_ANALYSIS_QA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--png-dpi", type=int, default=220)
    parser.add_argument(
        "--allow-missing-analysis-qa",
        action="store_true",
        help="Continue if the analysis-dataset QA summary CSV is unavailable.",
    )
    return parser.parse_args()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Input table does not exist: {0}".format(path))
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError("Unsupported input table format for {0}. Expected .parquet, .csv, or .json.".format(path))


def maybe_read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def write_text(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def clean_text(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def to_numeric(series):
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def to_flag(series):
    values = to_numeric(series)
    return values.fillna(0).astype(int)


def safe_divide(numer, denom):
    try:
        if denom is None or pd.isna(denom) or float(denom) == 0.0:
            return np.nan
        return float(numer) / float(denom)
    except Exception:
        return np.nan


def lower_map(columns):
    return {str(column).lower(): column for column in columns}


def find_col(df, candidates, required=False):
    cmap = lower_map(df.columns)
    for candidate in candidates:
        if candidate.lower() in cmap:
            return cmap[candidate.lower()]
    for column in df.columns:
        clean = str(column).lower()
        for candidate in candidates:
            if candidate.lower() in clean:
                return column
    if required:
        raise KeyError("Required column not found. Tried: {0}".format(candidates))
    return None


# -----------------------------
# Dataset preparation / QA
# -----------------------------


def normalize_analysis_set(value):
    text = clean_text(value)
    if text is None:
        return "missing"
    lowered = text.lower().replace("-", "_").replace(" ", "_")
    if lowered in {"wontfix", "won_t_fix", "won't_fix", "wont_fix"}:
        return "wontfix"
    if lowered in {"comparison", "control", "controls", "non_wontfix", "non_wontfix_comparison"}:
        return "comparison"
    return lowered


def normalize_comparison_group(value, analysis_set=None):
    if normalize_analysis_set(analysis_set) == "wontfix":
        return "wontfix"
    text = clean_text(value)
    if text is None:
        return "comparison"
    lowered = text.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "resolved_pr": "pr_resolved",
        "pr_resolved": "pr_resolved",
        "closed_non_wontfix": "closed_non_wontfix",
        "closed": "closed_non_wontfix",
        "comparison": "comparison",
        "open": "open",
        "invalid": "invalid",
    }
    return aliases.get(lowered, lowered)


def normalize_coverage_flag(value):
    text = clean_text(value)
    if text is None:
        return "missing_participation_features"
    lowered = text.lower().replace(" ", "_").replace("-", "_")
    if lowered in {"ok", "zero_comments", "missing_participation_features"}:
        return lowered
    return lowered


def validate_required_columns(df):
    required = ["repo_full_name", "issue_id", "analysis_set"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError("RQ3 dataset is missing required columns: {0}".format(missing))


def add_missing_participation_columns(df):
    out = df.copy()

    numeric_defaults = [
        "comment_count",
        "unique_commenter_count",
        "num_distinct_non_author_commenters",
        "top_commenter_share",
        "comment_concentration_ratio",
        "non_author_comment_share",
        "top_2_commenters_share",
        "mean_comments_per_commenter",
    ]
    for column in numeric_defaults:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = to_numeric(out[column])

    if "zero_comment_flag" not in out.columns:
        out["zero_comment_flag"] = (out["comment_count"].fillna(0) == 0).astype(int)
    else:
        out["zero_comment_flag"] = to_flag(out["zero_comment_flag"])

    flag_defaults = [
        "issue_author_commented_flag",
        "single_commenter_flag",
        "only_author_commented_flag",
        "first_comment_by_author_flag",
        "last_comment_by_author_flag",
    ]
    for column in flag_defaults:
        if column not in out.columns:
            out[column] = np.nan
        else:
            out[column] = to_numeric(out[column])

    if "has_participation_features" not in out.columns:
        if "__has_participation" in out.columns:
            out["has_participation_features"] = to_flag(out["__has_participation"])
        else:
            out["has_participation_features"] = 1
    else:
        out["has_participation_features"] = to_flag(out["has_participation_features"])

    if "usable_for_rq3" not in out.columns:
        out["usable_for_rq3"] = out["has_participation_features"].eq(1).astype(int)
    else:
        out["usable_for_rq3"] = to_flag(out["usable_for_rq3"])

    if "participation_feature_coverage_flag" not in out.columns:
        out["participation_feature_coverage_flag"] = np.where(
            out["has_participation_features"].eq(0),
            "missing_participation_features",
            np.where(out["zero_comment_flag"].eq(1), "zero_comments", "ok"),
        )
    else:
        out["participation_feature_coverage_flag"] = out["participation_feature_coverage_flag"].apply(normalize_coverage_flag)

    out["has_any_non_author_comment"] = (out["num_distinct_non_author_commenters"].fillna(0) > 0).astype(int)
    out["multi_party_discussion_flag"] = (out["unique_commenter_count"].fillna(0) > 1).astype(int)
    out["comment_bearing_flag"] = (out["comment_count"].fillna(0) > 0).astype(int)

    out["log1p_comment_count"] = np.log1p(out["comment_count"].fillna(0).clip(lower=0))
    out["log1p_unique_commenter_count"] = np.log1p(out["unique_commenter_count"].fillna(0).clip(lower=0))
    out["log1p_num_distinct_non_author_commenters"] = np.log1p(out["num_distinct_non_author_commenters"].fillna(0).clip(lower=0))

    for column in PARTICIPATION_CORE_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    return out


def normalize_dataset(df):
    out = df.copy()
    validate_required_columns(out)

    out["repo_full_name"] = out["repo_full_name"].astype(str)
    out["issue_id"] = out["issue_id"].astype(str)
    if "issue_number" in out.columns:
        out["issue_number"] = to_numeric(out["issue_number"])
    else:
        out["issue_number"] = np.nan

    out["analysis_set"] = out["analysis_set"].apply(normalize_analysis_set)
    if "comparison_group" not in out.columns:
        out["comparison_group"] = np.where(out["analysis_set"].eq("wontfix"), "wontfix", "comparison")
    else:
        out["comparison_group"] = [
            normalize_comparison_group(value, analysis_set=analysis_set)
            for value, analysis_set in zip(out["comparison_group"], out["analysis_set"])
        ]

    out = add_missing_participation_columns(out)
    out = out.sort_values(["repo_full_name", "analysis_set", "issue_number", "issue_id"], kind="stable").reset_index(drop=True)
    return out


def duplicate_issue_key_count(df):
    key_cols = ["repo_full_name", "issue_id", "issue_number"]
    existing = [column for column in key_cols if column in df.columns]
    if not existing:
        return len(df)
    return int(df.duplicated(subset=existing).sum())


def make_metric_rows(df, external_qa_df):
    rows = []

    def add(metric, value):
        rows.append({"metric": metric, "value": value})

    add("rq3_rows", int(len(df)))
    add("duplicate_issue_keys", duplicate_issue_key_count(df))
    add("repo_count", int(df["repo_full_name"].nunique()))
    add("analysis_set_count", int(df["analysis_set"].nunique()))
    add("wontfix_rows", int(df["analysis_set"].eq("wontfix").sum()))
    add("comparison_rows", int(df["analysis_set"].eq("comparison").sum()))
    add("rows_with_participation_features", int(df["has_participation_features"].fillna(0).sum()))
    add("rows_usable_for_rq3", int(df["usable_for_rq3"].fillna(0).sum()))
    add("zero_comment_rows", int(df["participation_feature_coverage_flag"].eq("zero_comments").sum()))
    add("ok_participation_rows", int(df["participation_feature_coverage_flag"].eq("ok").sum()))
    add("missing_participation_feature_rows", int(df["participation_feature_coverage_flag"].eq("missing_participation_features").sum()))
    add("zero_comment_share", safe_divide(df["participation_feature_coverage_flag"].eq("zero_comments").sum(), len(df)))

    if not external_qa_df.empty and {"metric", "value"}.issubset(external_qa_df.columns):
        metric_lookup = dict(zip(external_qa_df["metric"].astype(str), external_qa_df["value"]))
        for metric in [
            "population_rows_expected",
            "population_rows_final",
            "rows_with_participation_features",
            "rows_usable_for_rq3",
        ]:
            if metric in metric_lookup:
                add("analysis_dataset_qa__" + metric, metric_lookup[metric])

    return pd.DataFrame(rows)


def summarize_group(part, label_column, label_value):
    coverage_counts = part["participation_feature_coverage_flag"].value_counts(dropna=False).to_dict()
    return {
        label_column: label_value,
        "issue_count": int(len(part)),
        "repo_count": int(part["repo_full_name"].nunique()),
        "ok_count": int(coverage_counts.get("ok", 0)),
        "zero_comment_count": int(coverage_counts.get("zero_comments", 0)),
        "missing_participation_feature_count": int(coverage_counts.get("missing_participation_features", 0)),
        "zero_comment_share": safe_divide(coverage_counts.get("zero_comments", 0), len(part)),
        "has_participation_features_count": int(part["has_participation_features"].fillna(0).sum()),
        "usable_for_rq3_count": int(part["usable_for_rq3"].fillna(0).sum()),
    }


def make_population_summary(df):
    rows = []
    for analysis_set, part in df.groupby("analysis_set", dropna=False):
        rows.append(summarize_group(part, "analysis_set", analysis_set))
    rows.append(summarize_group(df, "analysis_set", "all"))
    return pd.DataFrame(rows)


def make_comparison_group_summary(df):
    rows = []
    for comparison_group, part in df.groupby("comparison_group", dropna=False):
        row = summarize_group(part, "comparison_group", comparison_group)
        row["analysis_sets_present"] = ", ".join(sorted(part["analysis_set"].dropna().astype(str).unique().tolist()))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["analysis_sets_present", "comparison_group"], kind="stable").reset_index(drop=True)


def make_coverage_by_repo(df):
    base = df.copy()
    base["coverage_flag"] = base["participation_feature_coverage_flag"].apply(normalize_coverage_flag)
    grouped = (
        base.groupby(["repo_full_name", "analysis_set", "coverage_flag"], dropna=False)
        .size()
        .reset_index(name="issue_count")
    )
    totals = (
        base.groupby(["repo_full_name", "analysis_set"], dropna=False)
        .size()
        .reset_index(name="total_issues")
    )
    out = grouped.merge(totals, on=["repo_full_name", "analysis_set"], how="left")
    out["share"] = out.apply(lambda row: safe_divide(row["issue_count"], row["total_issues"]), axis=1)
    return out.sort_values(["repo_full_name", "analysis_set", "coverage_flag"], kind="stable").reset_index(drop=True)


def make_zero_comment_by_repo(df):
    rows = []
    for (repo_name, analysis_set), part in df.groupby(["repo_full_name", "analysis_set"], dropna=False):
        zero_count = int(part["participation_feature_coverage_flag"].eq("zero_comments").sum())
        rows.append({
            "repo_full_name": repo_name,
            "analysis_set": analysis_set,
            "issue_count": int(len(part)),
            "zero_comment_count": zero_count,
            "zero_comment_share": safe_divide(zero_count, len(part)),
        })
    return pd.DataFrame(rows).sort_values(["repo_full_name", "analysis_set"], kind="stable").reset_index(drop=True)


def make_missingness_summary(df):
    rows = []
    for column in PARTICIPATION_CORE_COLUMNS:
        if column not in df.columns:
            rows.append({
                "column": column,
                "present": 0,
                "missing_count": len(df),
                "missing_share": 1.0 if len(df) else np.nan,
            })
            continue
        missing_count = int(df[column].isna().sum())
        rows.append({
            "column": column,
            "present": 1,
            "missing_count": missing_count,
            "missing_share": safe_divide(missing_count, len(df)),
        })
    return pd.DataFrame(rows)


def classify_zero_comment_values(series):
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = numeric.dropna()
    if len(non_missing) == 0:
        return "all_missing"
    zero_count = int(non_missing.eq(0).sum())
    nonzero_count = int(non_missing.ne(0).sum())
    missing_count = int(numeric.isna().sum())
    if zero_count == len(non_missing) and missing_count == 0:
        return "all_zero"
    if zero_count == len(non_missing) and missing_count > 0:
        return "zero_and_missing"
    if nonzero_count > 0:
        return "contains_nonzero_values"
    return "mixed_or_unknown"


def make_zero_comment_ratio_applicability(df):
    zero_df = df[df["participation_feature_coverage_flag"].eq("zero_comments")].copy()
    rows = []
    for column in RATIO_APPLICABILITY_COLUMNS:
        if column not in df.columns:
            rows.append({
                "column": column,
                "present": 0,
                "zero_comment_rows_checked": int(len(zero_df)),
                "missing_on_zero_comment_rows": int(len(zero_df)),
                "zero_values_on_zero_comment_rows": 0,
                "nonzero_values_on_zero_comment_rows": 0,
                "value_pattern": "column_missing",
                "analysis_guidance": "Column is not present.",
            })
            continue
        numeric = pd.to_numeric(zero_df[column], errors="coerce") if len(zero_df) else pd.Series(dtype="float64")
        missing_count = int(numeric.isna().sum())
        zero_count = int(numeric.dropna().eq(0).sum())
        nonzero_count = int(numeric.dropna().ne(0).sum())
        pattern = classify_zero_comment_values(numeric)
        if pattern in {"all_zero", "zero_and_missing", "all_missing"}:
            guidance = "Treat as not applicable for zero-comment issues; filter comment_count > 0 before ratio/concentration analysis."
        elif pattern == "contains_nonzero_values":
            guidance = "Inspect before analysis; nonzero ratio values on zero-comment issues may indicate unexpected coding."
        else:
            guidance = "Inspect before analysis."
        rows.append({
            "column": column,
            "present": 1,
            "zero_comment_rows_checked": int(len(zero_df)),
            "missing_on_zero_comment_rows": missing_count,
            "zero_values_on_zero_comment_rows": zero_count,
            "nonzero_values_on_zero_comment_rows": nonzero_count,
            "value_pattern": pattern,
            "analysis_guidance": guidance,
        })
    return pd.DataFrame(rows)


# -----------------------------
# Part 2 summaries
# -----------------------------


def describe_numeric(series):
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) == 0:
        return {
            "n": 0,
            "missing_count": int(len(series)),
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(len(numeric)),
        "missing_count": int(len(series) - len(numeric)),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
        "q1": float(numeric.quantile(0.25)),
        "q3": float(numeric.quantile(0.75)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def make_participation_group_summary(df):
    rows = []
    for spec in PART2_NUMERIC_FEATURES:
        feature = spec["feature"]
        if feature not in df.columns:
            continue
        base = df[df["comment_count"].fillna(0) > 0].copy() if spec.get("requires_comments") else df.copy()
        for analysis_set, part in base.groupby("analysis_set", dropna=False):
            desc = describe_numeric(part[feature])
            row = {
                "feature": feature,
                "feature_label": spec["label"],
                "feature_family": spec["family"],
                "feature_type": "numeric",
                "analysis_set": analysis_set,
                "requires_comment_bearing_issue": int(bool(spec.get("requires_comments"))),
                "issue_count_in_scope": int(len(part)),
            }
            row.update(desc)
            rows.append(row)
    for spec in PART2_BINARY_FEATURES:
        feature = spec["feature"]
        if feature not in df.columns:
            continue
        base = df[df["comment_count"].fillna(0) > 0].copy() if spec.get("requires_comments") else df.copy()
        for analysis_set, part in base.groupby("analysis_set", dropna=False):
            values = pd.to_numeric(part[feature], errors="coerce")
            valid = values.dropna()
            row = {
                "feature": feature,
                "feature_label": spec["label"],
                "feature_family": spec["family"],
                "feature_type": "binary",
                "analysis_set": analysis_set,
                "requires_comment_bearing_issue": int(bool(spec.get("requires_comments"))),
                "issue_count_in_scope": int(len(part)),
                "n": int(len(valid)),
                "missing_count": int(values.isna().sum()),
                "rate": float(valid.mean()) if len(valid) else np.nan,
                "positive_count": int(valid.eq(1).sum()) if len(valid) else 0,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def make_participation_repo_group_summary(df):
    rows = []
    useful_features = [
        "comment_count",
        "unique_commenter_count",
        "num_distinct_non_author_commenters",
        "top_commenter_share",
        "comment_concentration_ratio",
        "issue_author_commented_flag",
        "has_any_non_author_comment",
        "single_commenter_flag",
    ]
    for (repo_name, analysis_set), part in df.groupby(["repo_full_name", "analysis_set"], dropna=False):
        comment_bearing = part[part["comment_count"].fillna(0) > 0].copy()
        row = {
            "repo_full_name": repo_name,
            "analysis_set": analysis_set,
            "issue_count": int(len(part)),
            "comment_bearing_issue_count": int(len(comment_bearing)),
            "zero_comment_count": int(part["zero_comment_flag"].fillna(0).sum()),
            "zero_comment_share": safe_divide(part["zero_comment_flag"].fillna(0).sum(), len(part)),
        }
        for feature in useful_features:
            if feature not in part.columns:
                continue
            feature_base = comment_bearing if feature in {"top_commenter_share", "comment_concentration_ratio", "single_commenter_flag"} else part
            values = pd.to_numeric(feature_base[feature], errors="coerce").dropna()
            row[feature + "__mean"] = float(values.mean()) if len(values) else np.nan
            row[feature + "__median"] = float(values.median()) if len(values) else np.nan
            row[feature + "__n"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["repo_full_name", "analysis_set"], kind="stable").reset_index(drop=True)


def get_group_values(df, feature, analysis_set, requires_comments):
    base = df[df["analysis_set"].eq(analysis_set)].copy()
    if requires_comments:
        base = base[base["comment_count"].fillna(0) > 0].copy()
    if feature not in base.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(base[feature], errors="coerce").dropna()


def make_participation_difference_summary(df):
    rows = []
    for spec in PART2_NUMERIC_FEATURES:
        feature = spec["feature"]
        if feature not in df.columns:
            continue
        wontfix_values = get_group_values(df, feature, "wontfix", spec.get("requires_comments"))
        comparison_values = get_group_values(df, feature, "comparison", spec.get("requires_comments"))
        row = {
            "feature": feature,
            "feature_label": spec["label"],
            "feature_family": spec["family"],
            "feature_type": "numeric",
            "requires_comment_bearing_issue": int(bool(spec.get("requires_comments"))),
            "wontfix_n": int(len(wontfix_values)),
            "comparison_n": int(len(comparison_values)),
            "wontfix_mean": float(wontfix_values.mean()) if len(wontfix_values) else np.nan,
            "comparison_mean": float(comparison_values.mean()) if len(comparison_values) else np.nan,
            "mean_difference_wontfix_minus_comparison": np.nan,
            "wontfix_median": float(wontfix_values.median()) if len(wontfix_values) else np.nan,
            "comparison_median": float(comparison_values.median()) if len(comparison_values) else np.nan,
            "median_difference_wontfix_minus_comparison": np.nan,
        }
        row["mean_difference_wontfix_minus_comparison"] = row["wontfix_mean"] - row["comparison_mean"] if pd.notna(row["wontfix_mean"]) and pd.notna(row["comparison_mean"]) else np.nan
        row["median_difference_wontfix_minus_comparison"] = row["wontfix_median"] - row["comparison_median"] if pd.notna(row["wontfix_median"]) and pd.notna(row["comparison_median"]) else np.nan
        rows.append(row)
    for spec in PART2_BINARY_FEATURES:
        feature = spec["feature"]
        if feature not in df.columns:
            continue
        wontfix_values = get_group_values(df, feature, "wontfix", spec.get("requires_comments"))
        comparison_values = get_group_values(df, feature, "comparison", spec.get("requires_comments"))
        wontfix_rate = float(wontfix_values.mean()) if len(wontfix_values) else np.nan
        comparison_rate = float(comparison_values.mean()) if len(comparison_values) else np.nan
        rows.append({
            "feature": feature,
            "feature_label": spec["label"],
            "feature_family": spec["family"],
            "feature_type": "binary",
            "requires_comment_bearing_issue": int(bool(spec.get("requires_comments"))),
            "wontfix_n": int(len(wontfix_values)),
            "comparison_n": int(len(comparison_values)),
            "wontfix_rate": wontfix_rate,
            "comparison_rate": comparison_rate,
            "rate_difference_wontfix_minus_comparison": wontfix_rate - comparison_rate if pd.notna(wontfix_rate) and pd.notna(comparison_rate) else np.nan,
            "wontfix_positive_count": int(wontfix_values.eq(1).sum()) if len(wontfix_values) else 0,
            "comparison_positive_count": int(comparison_values.eq(1).sum()) if len(comparison_values) else 0,
        })
    return pd.DataFrame(rows)


# -----------------------------
# Part 3 repo-aware effect sizes and models
# -----------------------------


def get_feature_spec(feature):
    for spec in PART3_EFFECT_FEATURES:
        if spec.get("feature") == feature:
            return spec
    return None


def subset_for_feature(df, feature, requires_comments):
    if feature not in df.columns:
        return pd.DataFrame()
    base = df.copy()
    if requires_comments:
        base = base[base["comment_count"].fillna(0) > 0].copy()
    base = base[base["analysis_set"].isin(["wontfix", "comparison"])].copy()
    base[feature] = pd.to_numeric(base[feature], errors="coerce")
    base = base[base[feature].notna()].copy()
    return base


def cohen_d(wontfix_values, comparison_values):
    x = pd.Series(wontfix_values).dropna().astype(float)
    y = pd.Series(comparison_values).dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    denom_df = len(x) + len(y) - 2
    if denom_df <= 0:
        return np.nan
    pooled = np.sqrt(((len(x) - 1) * sx * sx + (len(y) - 1) * sy * sy) / denom_df)
    if pd.isna(pooled) or pooled == 0:
        return np.nan
    return (x.mean() - y.mean()) / pooled


def hedges_g(wontfix_values, comparison_values):
    d = cohen_d(wontfix_values, comparison_values)
    if pd.isna(d):
        return np.nan
    n = len(pd.Series(wontfix_values).dropna()) + len(pd.Series(comparison_values).dropna())
    if n <= 3:
        return d
    correction = 1.0 - (3.0 / (4.0 * n - 9.0))
    return d * correction


def log_odds_ratio_from_counts(wontfix_positive, wontfix_n, comparison_positive, comparison_n):
    if wontfix_n <= 0 or comparison_n <= 0:
        return np.nan
    a = float(wontfix_positive) + 0.5
    b = float(wontfix_n - wontfix_positive) + 0.5
    c = float(comparison_positive) + 0.5
    d = float(comparison_n - comparison_positive) + 0.5
    return np.log((a / b) / (c / d))


def standardize_within_repo(df, feature):
    values = pd.to_numeric(df[feature], errors="coerce")

    def zscore(series):
        std = series.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.mean()) / std

    temp = df.copy()
    temp["__value"] = values
    return temp.groupby("repo_full_name", dropna=False)["__value"].transform(zscore)


def make_repo_effect_sizes(df):
    rows = []
    for spec in PART3_EFFECT_FEATURES:
        feature = spec["feature"]
        base = subset_for_feature(df, feature, spec.get("requires_comments"))
        if base.empty:
            continue
        for repo_name, repo_df in base.groupby("repo_full_name"):
            wontfix = repo_df[repo_df["analysis_set"].eq("wontfix")][feature].dropna().astype(float)
            comparison = repo_df[repo_df["analysis_set"].eq("comparison")][feature].dropna().astype(float)
            if len(wontfix) == 0 or len(comparison) == 0:
                continue
            row = {
                "repo_full_name": repo_name,
                "feature": feature,
                "feature_label": spec["label"],
                "feature_family": spec["family"],
                "feature_type": spec["feature_type"],
                "requires_comment_bearing_issue": int(bool(spec.get("requires_comments"))),
                "wontfix_n": int(len(wontfix)),
                "comparison_n": int(len(comparison)),
                "wontfix_mean_or_rate": float(wontfix.mean()),
                "comparison_mean_or_rate": float(comparison.mean()),
                "raw_difference_wontfix_minus_comparison": float(wontfix.mean() - comparison.mean()),
                "weight_total_n": int(len(wontfix) + len(comparison)),
            }
            if spec["feature_type"] == "numeric":
                row["cohen_d_wontfix_minus_comparison"] = cohen_d(wontfix, comparison)
                row["hedges_g_wontfix_minus_comparison"] = hedges_g(wontfix, comparison)
                row["log_odds_ratio_wontfix_minus_comparison"] = np.nan
            else:
                wontfix_positive = int((wontfix == 1).sum())
                comparison_positive = int((comparison == 1).sum())
                row["wontfix_positive_count"] = wontfix_positive
                row["comparison_positive_count"] = comparison_positive
                row["cohen_d_wontfix_minus_comparison"] = np.nan
                row["hedges_g_wontfix_minus_comparison"] = np.nan
                row["log_odds_ratio_wontfix_minus_comparison"] = log_odds_ratio_from_counts(
                    wontfix_positive,
                    len(wontfix),
                    comparison_positive,
                    len(comparison),
                )
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["feature_family", "feature", "repo_full_name"], kind="stable").reset_index(drop=True)


def weighted_average(values, weights):
    values = pd.Series(values).astype(float)
    weights = pd.Series(weights).astype(float)
    mask = values.notna() & weights.notna() & (weights > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def make_overall_effect_sizes(df, repo_effect_df):
    rows = []
    for spec in PART3_EFFECT_FEATURES:
        feature = spec["feature"]
        base = subset_for_feature(df, feature, spec.get("requires_comments"))
        if base.empty:
            continue
        wontfix = base[base["analysis_set"].eq("wontfix")][feature].dropna().astype(float)
        comparison = base[base["analysis_set"].eq("comparison")][feature].dropna().astype(float)
        if len(wontfix) == 0 or len(comparison) == 0:
            continue
        repo_rows = repo_effect_df[repo_effect_df["feature"].eq(feature)].copy() if not repo_effect_df.empty else pd.DataFrame()
        row = {
            "feature": feature,
            "feature_label": spec["label"],
            "feature_family": spec["family"],
            "feature_type": spec["feature_type"],
            "requires_comment_bearing_issue": int(bool(spec.get("requires_comments"))),
            "wontfix_n": int(len(wontfix)),
            "comparison_n": int(len(comparison)),
            "wontfix_mean_or_rate": float(wontfix.mean()),
            "comparison_mean_or_rate": float(comparison.mean()),
            "pooled_raw_difference_wontfix_minus_comparison": float(wontfix.mean() - comparison.mean()),
            "repo_count_with_both_groups": int(repo_rows["repo_full_name"].nunique()) if not repo_rows.empty else 0,
            "repo_weighted_raw_difference_wontfix_minus_comparison": np.nan,
            "repo_unweighted_raw_difference_wontfix_minus_comparison": np.nan,
            "pooled_cohen_d_wontfix_minus_comparison": np.nan,
            "pooled_hedges_g_wontfix_minus_comparison": np.nan,
            "repo_weighted_hedges_g_wontfix_minus_comparison": np.nan,
            "pooled_log_odds_ratio_wontfix_minus_comparison": np.nan,
            "repo_weighted_log_odds_ratio_wontfix_minus_comparison": np.nan,
            "recommended_effect_for_plot": np.nan,
            "recommended_effect_units": "",
        }
        if not repo_rows.empty:
            row["repo_weighted_raw_difference_wontfix_minus_comparison"] = weighted_average(
                repo_rows["raw_difference_wontfix_minus_comparison"], repo_rows["weight_total_n"]
            )
            row["repo_unweighted_raw_difference_wontfix_minus_comparison"] = float(repo_rows["raw_difference_wontfix_minus_comparison"].mean())
        if spec["feature_type"] == "numeric":
            row["pooled_cohen_d_wontfix_minus_comparison"] = cohen_d(wontfix, comparison)
            row["pooled_hedges_g_wontfix_minus_comparison"] = hedges_g(wontfix, comparison)
            if not repo_rows.empty:
                row["repo_weighted_hedges_g_wontfix_minus_comparison"] = weighted_average(
                    repo_rows["hedges_g_wontfix_minus_comparison"], repo_rows["weight_total_n"]
                )
            row["recommended_effect_for_plot"] = row["repo_weighted_hedges_g_wontfix_minus_comparison"]
            row["recommended_effect_units"] = "Repo-weighted Hedges g"
        else:
            wontfix_positive = int((wontfix == 1).sum())
            comparison_positive = int((comparison == 1).sum())
            row["wontfix_positive_count"] = wontfix_positive
            row["comparison_positive_count"] = comparison_positive
            row["pooled_log_odds_ratio_wontfix_minus_comparison"] = log_odds_ratio_from_counts(
                wontfix_positive,
                len(wontfix),
                comparison_positive,
                len(comparison),
            )
            if not repo_rows.empty:
                row["repo_weighted_log_odds_ratio_wontfix_minus_comparison"] = weighted_average(
                    repo_rows["log_odds_ratio_wontfix_minus_comparison"], repo_rows["weight_total_n"]
                )
            row["recommended_effect_for_plot"] = row["repo_weighted_raw_difference_wontfix_minus_comparison"]
            row["recommended_effect_units"] = "Repo-weighted rate difference"
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["feature_family", "feature"], kind="stable").reset_index(drop=True)


def quote_formula_name(name):
    safe = str(name).replace('"', '\\"')
    return 'Q("{0}")'.format(safe)


def fit_repo_fixed_effect_model(df, spec):
    feature = spec["feature"]
    base = subset_for_feature(df, feature, spec.get("requires_comments"))
    if base.empty or smf is None:
        return {
            "feature": feature,
            "feature_label": spec["label"],
            "feature_family": spec["family"],
            "feature_type": spec["feature_type"],
            "model_type": "not_fit",
            "status": "statsmodels_unavailable" if smf is None else "empty_analysis_frame",
        }
    base = base[["repo_full_name", "analysis_set", feature]].dropna().copy()
    base["is_wontfix"] = base["analysis_set"].eq("wontfix").astype(int)
    group_counts = base.groupby("analysis_set")[feature].count().to_dict()
    if int(group_counts.get("wontfix", 0)) < MIN_MODEL_GROUP_N or int(group_counts.get("comparison", 0)) < MIN_MODEL_GROUP_N or len(base) < MIN_MODEL_TOTAL_N:
        return {
            "feature": feature,
            "feature_label": spec["label"],
            "feature_family": spec["family"],
            "feature_type": spec["feature_type"],
            "model_type": "not_fit",
            "status": "insufficient_group_n",
            "n": int(len(base)),
            "wontfix_n": int(group_counts.get("wontfix", 0)),
            "comparison_n": int(group_counts.get("comparison", 0)),
        }
    formula = "{0} ~ is_wontfix + C(repo_full_name)".format(quote_formula_name(feature))
    result_row = {
        "feature": feature,
        "feature_label": spec["label"],
        "feature_family": spec["family"],
        "feature_type": spec["feature_type"],
        "requires_comment_bearing_issue": int(bool(spec.get("requires_comments"))),
        "n": int(len(base)),
        "wontfix_n": int(group_counts.get("wontfix", 0)),
        "comparison_n": int(group_counts.get("comparison", 0)),
        "repo_count": int(base["repo_full_name"].nunique()),
        "model_formula": formula,
    }
    try:
        if spec["feature_type"] == "binary":
            try:
                model = smf.logit(formula, data=base).fit(disp=False, maxiter=200)
                result_row["model_type"] = "logit_repo_fixed_effects"
                result_row["status"] = "fit"
                result_row["coefficient_is_wontfix"] = float(model.params.get("is_wontfix", np.nan))
                result_row["std_error_is_wontfix"] = float(model.bse.get("is_wontfix", np.nan))
                result_row["p_value_is_wontfix"] = float(model.pvalues.get("is_wontfix", np.nan))
                ci = model.conf_int().loc["is_wontfix"] if "is_wontfix" in model.params.index else [np.nan, np.nan]
                result_row["ci_low_is_wontfix"] = float(ci[0])
                result_row["ci_high_is_wontfix"] = float(ci[1])
                result_row["odds_ratio_is_wontfix"] = float(np.exp(result_row["coefficient_is_wontfix"])) if pd.notna(result_row["coefficient_is_wontfix"]) else np.nan
                result_row["pseudo_r2"] = float(getattr(model, "prsquared", np.nan))
                return result_row
            except Exception as logit_exc:
                model = smf.ols(formula, data=base).fit(cov_type="HC3")
                result_row["model_type"] = "linear_probability_repo_fixed_effects_hc3"
                result_row["status"] = "fit_logit_failed_lpm_used"
                result_row["logit_error_message"] = str(logit_exc)[:300]
        else:
            model = smf.ols(formula, data=base).fit(cov_type="HC3")
            result_row["model_type"] = "ols_repo_fixed_effects_hc3"
            result_row["status"] = "fit"
        result_row["coefficient_is_wontfix"] = float(model.params.get("is_wontfix", np.nan))
        result_row["std_error_is_wontfix"] = float(model.bse.get("is_wontfix", np.nan))
        result_row["p_value_is_wontfix"] = float(model.pvalues.get("is_wontfix", np.nan))
        ci = model.conf_int().loc["is_wontfix"] if "is_wontfix" in model.params.index else [np.nan, np.nan]
        result_row["ci_low_is_wontfix"] = float(ci[0])
        result_row["ci_high_is_wontfix"] = float(ci[1])
        result_row["r_squared"] = float(getattr(model, "rsquared", np.nan))
        return result_row
    except Exception as exc:
        result_row["model_type"] = "not_fit"
        result_row["status"] = "model_error"
        result_row["error_message"] = str(exc)[:500]
        return result_row


def make_model_results(df):
    rows = []
    for spec in PART3_EFFECT_FEATURES:
        if spec["feature"] in df.columns:
            rows.append(fit_repo_fixed_effect_model(df, spec))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# -----------------------------
# Plotting
# -----------------------------


def style_axis(ax, grid_axis="y"):
    ax.set_facecolor(UTK_COLORS["white"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(UTK_COLORS["light_gray"])
    ax.spines["bottom"].set_color(UTK_COLORS["light_gray"])
    ax.tick_params(colors=UTK_COLORS["dark_gray"])
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E6E6E6", linewidth=0.8)
    ax.set_axisbelow(True)


def ok_color_for_analysis_set(analysis_set):
    return ANALYSIS_COLORS.get(str(analysis_set), ANALYSIS_COLORS["missing"])


def plot_coverage_by_repo(coverage_df, output_path, png_dpi):
    if coverage_df.empty:
        return None
    pivot = coverage_df.pivot_table(
        index=["repo_full_name", "analysis_set"],
        columns="coverage_flag",
        values="issue_count",
        aggfunc="sum",
        fill_value=0,
    )
    for column in ["ok", "zero_comments", "missing_participation_features"]:
        if column not in pivot.columns:
            pivot[column] = 0
    other_columns = [column for column in pivot.columns if column not in {"ok", "zero_comments", "missing_participation_features"}]
    pivot["other"] = pivot[other_columns].sum(axis=1) if other_columns else 0
    pivot = pivot[["ok", "zero_comments", "missing_participation_features", "other"]].sort_index(level=[0, 1])

    labels = []
    analysis_sets = []
    for repo_name, analysis_set in pivot.index:
        labels.append("{0} · {1}".format(repo_name, GROUP_LABELS.get(str(analysis_set), str(analysis_set).title())))
        analysis_sets.append(str(analysis_set))

    row_count = len(pivot)
    fig_height = max(4.8, row_count * 0.52 + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    fig.patch.set_facecolor(UTK_COLORS["white"])

    y_positions = np.arange(row_count)
    left = np.zeros(row_count)
    ok_values = pivot["ok"].to_numpy(dtype=float)
    ax.barh(y_positions, ok_values, left=left, height=0.74, color=[ok_color_for_analysis_set(value) for value in analysis_sets], edgecolor=UTK_COLORS["white"], linewidth=0.7)
    left = left + ok_values

    neutral_display = {
        "zero_comments": "Zero comments",
        "missing_participation_features": "Missing participation features",
        "other": "Other coverage flag",
    }
    for column in ["zero_comments", "missing_participation_features", "other"]:
        values = pivot[column].to_numpy(dtype=float)
        if np.nansum(values) == 0:
            continue
        ax.barh(y_positions, values, left=left, height=0.74, color=COVERAGE_NEUTRAL_COLORS.get(column, COVERAGE_NEUTRAL_COLORS["other"]), edgecolor=UTK_COLORS["white"], linewidth=0.7)
        left = left + values

    legend_handles = []
    legend_labels = []
    for group in ["wontfix", "comparison"]:
        if group in analysis_sets:
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=ANALYSIS_COLORS[group]))
            legend_labels.append("{0}: comment-bearing / OK".format(GROUP_LABELS[group]))
    for column in ["zero_comments", "missing_participation_features", "other"]:
        if column in pivot.columns and pivot[column].sum() > 0:
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=COVERAGE_NEUTRAL_COLORS.get(column, COVERAGE_NEUTRAL_COLORS["other"])))
            legend_labels.append(neutral_display.get(column, column))

    totals = pivot.sum(axis=1).to_numpy(dtype=float)
    max_total = max(totals) if len(totals) else 1
    for i, total in enumerate(totals):
        ax.text(total + max_total * 0.01, i, str(int(total)), va="center", ha="left", fontsize=8.5, color=UTK_COLORS["dark_gray"])

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Issue count", color=UTK_COLORS["dark_gray"])
    ax.set_title("Participation coverage by repository and analysis group", fontsize=14, color=UTK_COLORS["dark_gray"], pad=14)
    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    style_axis(ax, grid_axis="x")
    ax.set_xlim(0, max_total * 1.12 if len(totals) else 1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def group_arrays_for_feature(df, feature, filter_comment_bearing=False):
    base = df.copy()
    if filter_comment_bearing:
        base = base[base["comment_count"].fillna(0) > 0].copy()
    arrays = []
    labels = []
    colors = []
    for group in ["comparison", "wontfix"]:
        values = pd.to_numeric(base.loc[base["analysis_set"].eq(group), feature], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        arrays.append(values)
        labels.append(GROUP_LABELS.get(group, group.title()))
        colors.append(ANALYSIS_COLORS[group])
    return arrays, labels, colors


def color_boxplot(bp, colors):
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)
        patch.set_edgecolor(UTK_COLORS["dark_gray"])
    for element in ["whiskers", "caps", "medians"]:
        for item in bp[element]:
            item.set_color(UTK_COLORS["dark_gray"])
            item.set_linewidth(1.1)


def plot_comment_volume_distribution(df, output_path, png_dpi):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.patch.set_facecolor(UTK_COLORS["white"])
    max_value = float(df["log1p_comment_count"].max()) if len(df) else 1.0
    bins = np.linspace(0, max(max_value, 1.0), 28)
    for group in ["comparison", "wontfix"]:
        values = pd.to_numeric(df.loc[df["analysis_set"].eq(group), "log1p_comment_count"], errors="coerce").dropna()
        if len(values) == 0:
            continue
        ax.hist(values, bins=bins, alpha=0.68, color=ANALYSIS_COLORS[group], label=GROUP_LABELS[group], edgecolor=UTK_COLORS["white"], linewidth=0.45)
    ax.set_title("Comment volume distribution", fontsize=14, color=UTK_COLORS["dark_gray"], pad=14)
    ax.set_xlabel("log1p(comment count)", color=UTK_COLORS["dark_gray"])
    ax.set_ylabel("Issue count", color=UTK_COLORS["dark_gray"])
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="y")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_breadth_by_group(df, output_path, png_dpi):
    features = [
        ("log1p_unique_commenter_count", "log1p unique\ncommenters"),
        ("log1p_num_distinct_non_author_commenters", "log1p non-author\ncommenters"),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    fig.patch.set_facecolor(UTK_COLORS["white"])
    positions = []
    arrays = []
    colors = []
    tick_positions = []
    tick_labels = []
    pos = 1
    for feature, label in features:
        tick_positions.append(pos + 0.35)
        tick_labels.append(label)
        for group in ["comparison", "wontfix"]:
            values = pd.to_numeric(df.loc[df["analysis_set"].eq(group), feature], errors="coerce").dropna().to_numpy(dtype=float)
            arrays.append(values)
            positions.append(pos)
            colors.append(ANALYSIS_COLORS[group])
            pos += 0.7
        pos += 0.6
    bp = ax.boxplot(arrays, positions=positions, widths=0.48, patch_artist=True, showfliers=False)
    color_boxplot(bp, colors)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Log-scaled count", color=UTK_COLORS["dark_gray"])
    ax.set_title("Breadth of participation by analysis group", fontsize=14, color=UTK_COLORS["dark_gray"], pad=14)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ANALYSIS_COLORS["comparison"]), plt.Rectangle((0, 0), 1, 1, color=ANALYSIS_COLORS["wontfix"])]
    ax.legend(handles, ["Comparison", "WONTFIX"], frameon=False, loc="upper right")
    style_axis(ax, grid_axis="y")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_concentration_by_group(df, output_path, png_dpi):
    comment_df = df[df["comment_count"].fillna(0) > 0].copy()
    features = [
        ("top_commenter_share", "Top commenter\nshare"),
        ("comment_concentration_ratio", "Concentration\nratio"),
    ]
    fig, ax = plt.subplots(figsize=(9.4, 6.4))
    fig.patch.set_facecolor(UTK_COLORS["white"])
    positions = []
    arrays = []
    colors = []
    tick_positions = []
    tick_labels = []
    pos = 1
    for feature, label in features:
        tick_positions.append(pos + 0.35)
        tick_labels.append(label)
        for group in ["comparison", "wontfix"]:
            values = pd.to_numeric(comment_df.loc[comment_df["analysis_set"].eq(group), feature], errors="coerce").dropna().to_numpy(dtype=float)
            arrays.append(values)
            positions.append(pos)
            colors.append(ANALYSIS_COLORS[group])
            pos += 0.7
        pos += 0.6
    bp = ax.boxplot(arrays, positions=positions, widths=0.48, patch_artist=True, showfliers=False)
    color_boxplot(bp, colors)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("Share / ratio", color=UTK_COLORS["dark_gray"])
    ax.set_title("Discussion concentration among comment-bearing issues", fontsize=14, color=UTK_COLORS["dark_gray"], pad=14)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ANALYSIS_COLORS["comparison"]), plt.Rectangle((0, 0), 1, 1, color=ANALYSIS_COLORS["wontfix"])]
    ax.legend(handles, ["Comparison", "WONTFIX"], frameon=False, loc="upper right")
    style_axis(ax, grid_axis="y")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_author_participation(df, output_path, png_dpi):
    features = [
        ("issue_author_commented_flag", "Author\ncommented", False),
        ("has_any_non_author_comment", "Any non-author\ncommenter", False),
        ("multi_party_discussion_flag", "Multiple\ncommenters", False),
        ("single_commenter_flag", "Single-commenter\ndiscussion", True),
        ("only_author_commented_flag", "Only author\ncommented", True),
    ]
    x = np.arange(len(features))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11.0, 6.4))
    fig.patch.set_facecolor(UTK_COLORS["white"])
    for offset, group in [(-width / 2, "comparison"), (width / 2, "wontfix")]:
        rates = []
        for feature, _, requires_comments in features:
            base = df[df["analysis_set"].eq(group)].copy()
            if requires_comments:
                base = base[base["comment_count"].fillna(0) > 0].copy()
            values = pd.to_numeric(base[feature], errors="coerce").dropna() if feature in base.columns else pd.Series(dtype="float64")
            rates.append(float(values.mean()) if len(values) else np.nan)
        ax.bar(x + offset, rates, width=width, color=ANALYSIS_COLORS[group], label=GROUP_LABELS[group], edgecolor=UTK_COLORS["white"], linewidth=0.7)
        for xpos, rate in zip(x + offset, rates):
            if pd.notna(rate):
                ax.text(xpos, rate + 0.015, "{0:.1f}%".format(rate * 100), ha="center", va="bottom", fontsize=8, color=UTK_COLORS["dark_gray"], rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in features])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Share of issues", color=UTK_COLORS["dark_gray"])
    ax.set_title("Issue-author and non-author participation", fontsize=14, color=UTK_COLORS["dark_gray"], pad=14)
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax, grid_axis="y")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path



def plot_effect_sizes(effect_df, output_path, png_dpi):
    if effect_df.empty or "recommended_effect_for_plot" not in effect_df.columns:
        return None
    plot_df = effect_df.copy()
    plot_df = plot_df[plot_df["recommended_effect_for_plot"].notna()].copy()
    if plot_df.empty:
        return None
    plot_df["feature_display"] = plot_df["feature_label"]
    plot_df = plot_df.sort_values(["feature_family", "recommended_effect_for_plot"], kind="stable").reset_index(drop=True)

    y_positions = np.arange(len(plot_df))
    colors = []
    for _, row in plot_df.iterrows():
        if row.get("feature_type") == "numeric":
            colors.append(UTK_COLORS["orange"])
        else:
            colors.append(UTK_COLORS["smokey_gray"])

    fig_height = max(6.0, len(plot_df) * 0.43 + 2.0)
    fig, ax = plt.subplots(figsize=(11.0, fig_height))
    fig.patch.set_facecolor(UTK_COLORS["white"])
    values = plot_df["recommended_effect_for_plot"].astype(float).to_numpy()
    ax.barh(y_positions, values, color=colors, edgecolor=UTK_COLORS["white"], linewidth=0.7)
    ax.axvline(0, color=UTK_COLORS["dark_gray"], linewidth=1.0)
    for y, value, units in zip(y_positions, values, plot_df["recommended_effect_units"]):
        if pd.notna(value):
            label = "{0:.3f}".format(value)
            offset = 0.01 if value >= 0 else -0.01
            ha = "left" if value >= 0 else "right"
            ax.text(value + offset, y, label, va="center", ha=ha, fontsize=8.5, color=UTK_COLORS["dark_gray"])
    ax.set_yticks(y_positions)
    ax.set_yticklabels(plot_df["feature_display"].tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Effect size: numeric = repo-weighted Hedges g; binary = repo-weighted rate difference", color=UTK_COLORS["dark_gray"])
    ax.set_title("Repo-aware WONTFIX vs comparison effect sizes", fontsize=14, color=UTK_COLORS["dark_gray"], pad=14)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=UTK_COLORS["orange"]),
        plt.Rectangle((0, 0), 1, 1, color=UTK_COLORS["smokey_gray"]),
    ]
    ax.legend(handles, ["Numeric features", "Binary features"], frameon=False, loc="lower right")
    style_axis(ax, grid_axis="x")
    finite_values = plot_df["recommended_effect_for_plot"].dropna().astype(float)
    max_abs = max(abs(finite_values).max(), 0.10) if len(finite_values) else 0.10
    ax.set_xlim(-max_abs * 1.25, max_abs * 1.25)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_repo_effect_heatmap(repo_effect_df, output_path, png_dpi):
    if repo_effect_df.empty:
        return None
    heat_df = repo_effect_df.copy()
    heat_df["heatmap_effect"] = np.where(
        heat_df["feature_type"].eq("numeric"),
        heat_df["hedges_g_wontfix_minus_comparison"],
        heat_df["raw_difference_wontfix_minus_comparison"],
    )
    heat_df = heat_df[heat_df["heatmap_effect"].notna()].copy()
    if heat_df.empty:
        return None
    heat_df["display_feature"] = heat_df["feature_label"]
    pivot = heat_df.pivot_table(
        index="repo_full_name",
        columns="display_feature",
        values="heatmap_effect",
        aggfunc="mean",
    )
    if pivot.empty:
        return None
    feature_order = []
    for spec in PART3_EFFECT_FEATURES:
        if spec["label"] in pivot.columns:
            feature_order.append(spec["label"])
    pivot = pivot[feature_order]
    values = pivot.to_numpy(dtype=float)
    max_abs = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    if max_abs == 0 or pd.isna(max_abs):
        max_abs = 1.0

    fig_width = max(10.5, len(pivot.columns) * 0.85 + 3.0)
    fig_height = max(4.8, len(pivot.index) * 0.55 + 2.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(UTK_COLORS["white"])
    cmap = plt.get_cmap("coolwarm")
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Within-repo WONTFIX minus comparison effects", fontsize=14, color=UTK_COLORS["dark_gray"], pad=14)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if pd.notna(value):
                text_color = UTK_COLORS["white"] if abs(value) > max_abs * 0.55 else UTK_COLORS["dark_gray"]
                ax.text(j, i, "{0:.2f}".format(value), ha="center", va="center", fontsize=7.5, color=text_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Numeric = Hedges g; binary = rate difference", color=UTK_COLORS["dark_gray"])
    ax.tick_params(colors=UTK_COLORS["dark_gray"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path



# -----------------------------
# Part 4 participation + sentiment bridge
# -----------------------------


def available_sentiment_specs(df):
    specs = []
    for spec in SENTIMENT_FEATURES:
        if spec["feature"] in df.columns and pd.to_numeric(df[spec["feature"]], errors="coerce").notna().any():
            specs.append(spec)
    return specs


def sentiment_label_lookup():
    return {spec["feature"]: spec["label"] for spec in SENTIMENT_FEATURES}


def participation_label_lookup():
    lookup = {}
    for spec in PART2_NUMERIC_FEATURES + PART2_BINARY_FEATURES + PART3_EFFECT_FEATURES + PARTICIPATION_SENTIMENT_BRIDGE_FEATURES:
        lookup[spec["feature"]] = spec["label"]
    return lookup


def safe_corr(series_x, series_y, method):
    temp = pd.DataFrame({"x": pd.to_numeric(series_x, errors="coerce"), "y": pd.to_numeric(series_y, errors="coerce")}).dropna()
    if len(temp) < 10:
        return np.nan
    if temp["x"].nunique() < 2 or temp["y"].nunique() < 2:
        return np.nan
    try:
        return float(temp["x"].corr(temp["y"], method=method))
    except Exception:
        return np.nan


def make_participation_sentiment_correlation(issue_df):
    rows = []
    sentiment_specs = available_sentiment_specs(issue_df)
    if not sentiment_specs:
        return pd.DataFrame()
    scopes = [("all", issue_df)]
    for group in ["comparison", "wontfix"]:
        scopes.append((group, issue_df[issue_df["analysis_set"].eq(group)].copy()))

    for scope_name, scope_df in scopes:
        for part_spec in PARTICIPATION_SENTIMENT_BRIDGE_FEATURES:
            part_col = part_spec["feature"]
            if part_col not in scope_df.columns:
                continue
            part_base = scope_df.copy()
            if part_spec.get("requires_comments"):
                part_base = part_base[part_base["comment_count"].fillna(0) > 0].copy()
            for sent_spec in sentiment_specs:
                sent_col = sent_spec["feature"]
                pair = part_base[[part_col, sent_col]].copy().dropna()
                pair[part_col] = pd.to_numeric(pair[part_col], errors="coerce")
                pair[sent_col] = pd.to_numeric(pair[sent_col], errors="coerce")
                pair = pair.dropna()
                rows.append({
                    "scope": scope_name,
                    "participation_feature": part_col,
                    "participation_label": part_spec["label"],
                    "participation_family": part_spec["family"],
                    "sentiment_feature": sent_col,
                    "sentiment_label": sent_spec["label"],
                    "sentiment_family": sent_spec["family"],
                    "requires_comment_bearing_issue": int(bool(part_spec.get("requires_comments"))),
                    "n": int(len(pair)),
                    "pearson_correlation": safe_corr(pair[part_col], pair[sent_col], "pearson"),
                    "spearman_correlation": safe_corr(pair[part_col], pair[sent_col], "spearman"),
                })
    return pd.DataFrame(rows)


def assign_quantile_bucket(series, bucket_count=4):
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    valid = values.dropna()
    if len(valid) < bucket_count or valid.nunique() < 2:
        return out
    try:
        bucketed = pd.qcut(valid, q=bucket_count, labels=False, duplicates="drop")
    except Exception:
        try:
            bucketed = pd.qcut(valid.rank(method="first"), q=bucket_count, labels=False, duplicates="drop")
        except Exception:
            return out
    for index, bucket in bucketed.items():
        if pd.isna(bucket):
            continue
        out.at[index] = "Q{0}".format(int(bucket) + 1)
    return out


def make_sentiment_by_participation_bucket(issue_df):
    sentiment_specs = available_sentiment_specs(issue_df)
    if not sentiment_specs:
        return pd.DataFrame()

    dimensions = [
        {
            "dimension": "participation_breadth",
            "dimension_label": "Participation breadth",
            "bucket_feature": "log1p_unique_commenter_count",
            "bucket_feature_label": "log1p(unique commenters)",
            "requires_comments": False,
        },
        {
            "dimension": "non_author_breadth",
            "dimension_label": "Non-author participation breadth",
            "bucket_feature": "log1p_num_distinct_non_author_commenters",
            "bucket_feature_label": "log1p(non-author commenters)",
            "requires_comments": False,
        },
        {
            "dimension": "discussion_concentration",
            "dimension_label": "Discussion concentration",
            "bucket_feature": "top_commenter_share",
            "bucket_feature_label": "Top commenter share",
            "requires_comments": True,
        },
    ]

    rows = []
    for dim in dimensions:
        feature = dim["bucket_feature"]
        if feature not in issue_df.columns:
            continue
        base = issue_df.copy()
        if dim.get("requires_comments"):
            base = base[base["comment_count"].fillna(0) > 0].copy()
        if base.empty:
            continue
        base["participation_bucket"] = assign_quantile_bucket(base[feature])
        base = base[base["participation_bucket"].notna()].copy()
        if base.empty:
            continue
        for analysis_set in ["comparison", "wontfix"]:
            group_df = base[base["analysis_set"].eq(analysis_set)].copy()
            if group_df.empty:
                continue
            for bucket in sorted(group_df["participation_bucket"].dropna().unique().tolist()):
                bucket_df = group_df[group_df["participation_bucket"].eq(bucket)].copy()
                for sent_spec in sentiment_specs:
                    sent_col = sent_spec["feature"]
                    values = pd.to_numeric(bucket_df[sent_col], errors="coerce").dropna()
                    rows.append({
                        "dimension": dim["dimension"],
                        "dimension_label": dim["dimension_label"],
                        "bucket_feature": feature,
                        "bucket_feature_label": dim["bucket_feature_label"],
                        "participation_bucket": bucket,
                        "analysis_set": analysis_set,
                        "sentiment_feature": sent_col,
                        "sentiment_label": sent_spec["label"],
                        "sentiment_family": sent_spec["family"],
                        "n": int(len(values)),
                        "sentiment_mean": float(values.mean()) if len(values) else np.nan,
                        "sentiment_median": float(values.median()) if len(values) else np.nan,
                        "sentiment_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                        "sentiment_q1": float(values.quantile(0.25)) if len(values) else np.nan,
                        "sentiment_q3": float(values.quantile(0.75)) if len(values) else np.nan,
                    })
    return pd.DataFrame(rows)


def make_participation_sentiment_model_bridge(issue_df):
    rows = []
    if smf is None:
        return pd.DataFrame([{"status": "statsmodels_unavailable"}])
    if "repo_full_name" not in issue_df.columns or "analysis_set" not in issue_df.columns:
        return pd.DataFrame()

    available_outcomes = [col for col in PART4_SENTIMENT_MODEL_OUTCOMES if col in issue_df.columns]
    available_controls = [col for col in PART4_CONTROL_FEATURES if col in issue_df.columns]
    if not available_outcomes:
        return pd.DataFrame()

    for outcome in available_outcomes:
        needed_base = [outcome, "analysis_set", "repo_full_name"]
        base_df = issue_df[needed_base + available_controls].copy()
        base_df[outcome] = pd.to_numeric(base_df[outcome], errors="coerce")
        base_df["is_wontfix"] = base_df["analysis_set"].eq("wontfix").astype(int)
        for control in available_controls:
            base_df[control] = pd.to_numeric(base_df[control], errors="coerce")
        base_df = base_df[base_df[outcome].notna()].copy()
        if base_df.empty or base_df["is_wontfix"].nunique() < 2 or base_df["repo_full_name"].nunique() < 2:
            rows.append({
                "sentiment_feature": outcome,
                "sentiment_label": sentiment_label_lookup().get(outcome, outcome),
                "model_name": "base_repo_fe",
                "status": "insufficient_variation",
            })
            continue

        model_specs = [
            {
                "model_name": "base_repo_fe",
                "formula": "{0} ~ is_wontfix + C(repo_full_name)".format(outcome),
                "controls": [],
            },
        ]

        controls_for_model = []
        for control in available_controls:
            non_null = base_df[control].notna().sum()
            if non_null >= MIN_MODEL_TOTAL_N and base_df[control].nunique(dropna=True) > 1:
                controls_for_model.append(control)
        if controls_for_model:
            model_specs.append({
                "model_name": "participation_adjusted_repo_fe",
                "formula": "{0} ~ is_wontfix + {1} + C(repo_full_name)".format(outcome, " + ".join(controls_for_model)),
                "controls": controls_for_model,
            })

        base_coef = None
        for spec in model_specs:
            model_df = base_df[[outcome, "is_wontfix", "repo_full_name"] + spec["controls"]].dropna().copy()
            if len(model_df) < MIN_MODEL_TOTAL_N or model_df["is_wontfix"].nunique() < 2:
                rows.append({
                    "sentiment_feature": outcome,
                    "sentiment_label": sentiment_label_lookup().get(outcome, outcome),
                    "model_name": spec["model_name"],
                    "status": "insufficient_complete_cases",
                    "n": int(len(model_df)),
                    "controls": ",".join(spec["controls"]),
                })
                continue
            try:
                fit = smf.ols(spec["formula"], data=model_df).fit(cov_type="HC3")
                coef = fit.params.get("is_wontfix", np.nan)
                se = fit.bse.get("is_wontfix", np.nan)
                p_value = fit.pvalues.get("is_wontfix", np.nan)
                ci_low = coef - 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan
                ci_high = coef + 1.96 * se if pd.notna(coef) and pd.notna(se) else np.nan
                if spec["model_name"] == "base_repo_fe":
                    base_coef = coef
                rows.append({
                    "sentiment_feature": outcome,
                    "sentiment_label": sentiment_label_lookup().get(outcome, outcome),
                    "model_name": spec["model_name"],
                    "status": "fit",
                    "n": int(len(model_df)),
                    "wontfix_n": int(model_df["is_wontfix"].sum()),
                    "comparison_n": int((1 - model_df["is_wontfix"]).sum()),
                    "repo_count": int(model_df["repo_full_name"].nunique()),
                    "coefficient_is_wontfix": float(coef) if pd.notna(coef) else np.nan,
                    "std_error_is_wontfix": float(se) if pd.notna(se) else np.nan,
                    "ci_low_is_wontfix": float(ci_low) if pd.notna(ci_low) else np.nan,
                    "ci_high_is_wontfix": float(ci_high) if pd.notna(ci_high) else np.nan,
                    "p_value_is_wontfix": float(p_value) if pd.notna(p_value) else np.nan,
                    "r_squared": float(getattr(fit, "rsquared", np.nan)),
                    "controls": ",".join(spec["controls"]),
                    "base_coefficient_is_wontfix": float(base_coef) if pd.notna(base_coef) else np.nan,
                    "coefficient_change_from_base": float(coef - base_coef) if spec["model_name"] != "base_repo_fe" and pd.notna(coef) and pd.notna(base_coef) else np.nan,
                    "absolute_coefficient_change_from_base": float(abs(coef - base_coef)) if spec["model_name"] != "base_repo_fe" and pd.notna(coef) and pd.notna(base_coef) else np.nan,
                })
            except Exception as exc:
                rows.append({
                    "sentiment_feature": outcome,
                    "sentiment_label": sentiment_label_lookup().get(outcome, outcome),
                    "model_name": spec["model_name"],
                    "status": "failed",
                    "error_message": str(exc),
                    "controls": ",".join(spec["controls"]),
                })
    return pd.DataFrame(rows)


def compact_sentiment_correlation_table(correlation_df):
    if correlation_df.empty:
        return correlation_df
    display = correlation_df.copy()
    display = display[display["scope"].eq("all")].copy() if "scope" in display.columns else display
    if display.empty:
        display = correlation_df.copy()
    if "spearman_correlation" in display.columns:
        display = display.sort_values("spearman_correlation", key=lambda s: s.abs(), ascending=False, kind="stable")
    keep_cols = [
        "participation_label",
        "participation_family",
        "sentiment_label",
        "n",
        "pearson_correlation",
        "spearman_correlation",
    ]
    keep_cols = [col for col in keep_cols if col in display.columns]
    display = display[keep_cols].head(20).copy()
    for col in ["pearson_correlation", "spearman_correlation"]:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: format_float(x, 3))
    return display


def compact_bucket_table(bucket_df, dimension=None, sentiment_feature="mean_comment_sentiment"):
    if bucket_df.empty:
        return bucket_df
    display = bucket_df.copy()
    if dimension is not None and "dimension" in display.columns:
        display = display[display["dimension"].eq(dimension)].copy()
    if "sentiment_feature" in display.columns:
        display = display[display["sentiment_feature"].eq(sentiment_feature)].copy()
    keep_cols = [
        "dimension_label",
        "bucket_feature_label",
        "participation_bucket",
        "analysis_set",
        "sentiment_label",
        "n",
        "sentiment_mean",
        "sentiment_median",
        "sentiment_std",
    ]
    keep_cols = [col for col in keep_cols if col in display.columns]
    display = display[keep_cols].copy()
    for col in ["sentiment_mean", "sentiment_median", "sentiment_std"]:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: format_float(x, 4))
    return display


def compact_sentiment_bridge_model_table(model_df):
    if model_df.empty:
        return model_df
    keep_cols = [
        "sentiment_label",
        "model_name",
        "status",
        "n",
        "wontfix_n",
        "comparison_n",
        "repo_count",
        "coefficient_is_wontfix",
        "std_error_is_wontfix",
        "ci_low_is_wontfix",
        "ci_high_is_wontfix",
        "p_value_is_wontfix",
        "coefficient_change_from_base",
        "controls",
    ]
    keep_cols = [col for col in keep_cols if col in model_df.columns]
    display = model_df[keep_cols].copy()
    for col in ["coefficient_is_wontfix", "std_error_is_wontfix", "ci_low_is_wontfix", "ci_high_is_wontfix", "p_value_is_wontfix", "coefficient_change_from_base"]:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: format_float(x, 4))
    return display


def plot_sentiment_by_participation_bucket(bucket_df, dimension, sentiment_feature, title, output_path, png_dpi):
    if bucket_df.empty:
        return None
    plot_df = bucket_df[(bucket_df["dimension"].eq(dimension)) & (bucket_df["sentiment_feature"].eq(sentiment_feature))].copy()
    if plot_df.empty:
        return None
    buckets = sorted(plot_df["participation_bucket"].dropna().unique().tolist())
    if not buckets:
        return None
    x = np.arange(len(buckets))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    fig.patch.set_facecolor(UTK_COLORS["white"])
    for offset, group in [(-width / 2, "comparison"), (width / 2, "wontfix")]:
        values = []
        counts = []
        for bucket in buckets:
            row = plot_df[(plot_df["analysis_set"].eq(group)) & (plot_df["participation_bucket"].eq(bucket))]
            if row.empty:
                values.append(np.nan)
                counts.append(0)
            else:
                values.append(float(row.iloc[0]["sentiment_mean"]))
                counts.append(int(row.iloc[0]["n"]))
        ax.plot(x + offset, values, marker="o", linewidth=2.0, color=ANALYSIS_COLORS[group], label=GROUP_LABELS[group])
        for xpos, value, count in zip(x + offset, values, counts):
            if pd.notna(value):
                ax.text(xpos, value, "n={0}".format(count), ha="center", va="bottom", fontsize=7.5, color=UTK_COLORS["dark_gray"], rotation=45)
    ax.axhline(0, color=UTK_COLORS["light_gray"], linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel("Participation bucket", color=UTK_COLORS["dark_gray"])
    ax.set_ylabel("Mean sentiment feature value", color=UTK_COLORS["dark_gray"])
    ax.set_title(title, fontsize=14, color=UTK_COLORS["dark_gray"], pad=14)
    ax.legend(frameon=False)
    style_axis(ax, grid_axis="y")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def plot_sentiment_model_bridge(model_df, output_path, png_dpi):
    if model_df.empty or "model_name" not in model_df.columns:
        return None
    plot_df = model_df[model_df["status"].eq("fit")].copy()
    plot_df = plot_df[plot_df["model_name"].isin(["base_repo_fe", "participation_adjusted_repo_fe"])].copy()
    if plot_df.empty:
        return None
    pivot = plot_df.pivot_table(index="sentiment_label", columns="model_name", values="coefficient_is_wontfix", aggfunc="first")
    if pivot.empty:
        return None
    for col in ["base_repo_fe", "participation_adjusted_repo_fe"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot[["base_repo_fe", "participation_adjusted_repo_fe"]].dropna(how="all")
    if pivot.empty:
        return None
    y = np.arange(len(pivot.index))
    width = 0.34
    fig_height = max(5.0, len(pivot.index) * 0.58 + 2.2)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    fig.patch.set_facecolor(UTK_COLORS["white"])
    ax.barh(y - width / 2, pivot["base_repo_fe"], height=width, color=UTK_COLORS["smokey_gray"], label="Repo FE only", edgecolor=UTK_COLORS["white"], linewidth=0.7)
    ax.barh(y + width / 2, pivot["participation_adjusted_repo_fe"], height=width, color=UTK_COLORS["orange"], label="Repo FE + participation controls", edgecolor=UTK_COLORS["white"], linewidth=0.7)
    ax.axvline(0, color=UTK_COLORS["dark_gray"], linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index.tolist(), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("WONTFIX coefficient", color=UTK_COLORS["dark_gray"])
    ax.set_title("WONTFIX sentiment effect before/after participation controls", fontsize=14, color=UTK_COLORS["dark_gray"], pad=14)
    ax.legend(frameon=False, loc="lower right")
    style_axis(ax, grid_axis="x")
    finite = pivot.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    max_abs = max(abs(finite).max(), 0.02) if len(finite) else 0.02
    ax.set_xlim(-max_abs * 1.30, max_abs * 1.30)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=png_dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path

# -----------------------------
# Markdown report
# -----------------------------


def format_int(value):
    try:
        return "{0:,}".format(int(float(value)))
    except Exception:
        return str(value)


def format_pct(value):
    try:
        if pd.isna(value):
            return "NA"
        return "{0:.1f}%".format(float(value) * 100.0)
    except Exception:
        return str(value)


def format_float(value, digits=3):
    try:
        if pd.isna(value):
            return "NA"
        return ("{0:." + str(digits) + "f}").format(float(value))
    except Exception:
        return str(value)


def metric_value(metrics_df, metric, default=None):
    if metrics_df.empty or "metric" not in metrics_df.columns:
        return default
    matched = metrics_df[metrics_df["metric"].eq(metric)]
    if matched.empty:
        return default
    return matched.iloc[0]["value"]


def dataframe_to_markdown(df, max_rows=None):
    if max_rows is not None:
        df = df.head(max_rows).copy()
    if df.empty:
        return "_No rows._"
    return df.to_markdown(index=False)


def format_percent_columns(df, columns):
    out = df.copy()
    for column in columns:
        if column in out.columns:
            out[column] = out[column].apply(format_pct)
    return out


def compact_difference_table(difference_df):
    if difference_df.empty:
        return difference_df
    selected = difference_df.copy()
    keep_cols = [
        "feature_label",
        "feature_family",
        "feature_type",
        "requires_comment_bearing_issue",
        "wontfix_n",
        "comparison_n",
        "wontfix_mean",
        "comparison_mean",
        "mean_difference_wontfix_minus_comparison",
        "wontfix_rate",
        "comparison_rate",
        "rate_difference_wontfix_minus_comparison",
    ]
    keep_cols = [col for col in keep_cols if col in selected.columns]
    selected = selected[keep_cols].copy()
    for col in ["wontfix_mean", "comparison_mean", "mean_difference_wontfix_minus_comparison"]:
        if col in selected.columns:
            selected[col] = selected[col].apply(lambda x: format_float(x, 3))
    for col in ["wontfix_rate", "comparison_rate", "rate_difference_wontfix_minus_comparison"]:
        if col in selected.columns:
            selected[col] = selected[col].apply(format_pct)
    return selected




def compact_effect_size_table(effect_df):
    if effect_df.empty:
        return effect_df
    keep_cols = [
        "feature_label",
        "feature_family",
        "feature_type",
        "wontfix_n",
        "comparison_n",
        "wontfix_mean_or_rate",
        "comparison_mean_or_rate",
        "pooled_raw_difference_wontfix_minus_comparison",
        "repo_weighted_raw_difference_wontfix_minus_comparison",
        "pooled_hedges_g_wontfix_minus_comparison",
        "repo_weighted_hedges_g_wontfix_minus_comparison",
        "pooled_log_odds_ratio_wontfix_minus_comparison",
        "repo_weighted_log_odds_ratio_wontfix_minus_comparison",
        "recommended_effect_for_plot",
        "recommended_effect_units",
        "repo_count_with_both_groups",
    ]
    keep_cols = [col for col in keep_cols if col in effect_df.columns]
    out = effect_df[keep_cols].copy()
    count_columns = {"wontfix_n", "comparison_n", "repo_count_with_both_groups"}
    for col in out.columns:
        if col in {"feature_label", "feature_family", "feature_type", "recommended_effect_units"}:
            continue
        if col in count_columns:
            continue
        out[col] = out[col].apply(lambda x: format_float(x, 3))
    return out


def compact_model_results_table(model_df):
    if model_df.empty:
        return model_df
    keep_cols = [
        "feature_label",
        "feature_family",
        "feature_type",
        "model_type",
        "status",
        "n",
        "wontfix_n",
        "comparison_n",
        "repo_count",
        "coefficient_is_wontfix",
        "std_error_is_wontfix",
        "ci_low_is_wontfix",
        "ci_high_is_wontfix",
        "p_value_is_wontfix",
        "odds_ratio_is_wontfix",
        "r_squared",
        "pseudo_r2",
    ]
    keep_cols = [col for col in keep_cols if col in model_df.columns]
    out = model_df[keep_cols].copy()
    for col in ["coefficient_is_wontfix", "std_error_is_wontfix", "ci_low_is_wontfix", "ci_high_is_wontfix", "p_value_is_wontfix", "odds_ratio_is_wontfix", "r_squared", "pseudo_r2"]:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: format_float(x, 4))
    return out

def build_markdown_report(metrics_df, population_df, comparison_group_df, coverage_df, zero_comment_repo_df, missingness_df, ratio_applicability_df, group_summary_df, repo_group_summary_df, difference_df, repo_effect_df, effect_size_df, model_results_df, sentiment_correlation_df, sentiment_bucket_df, sentiment_model_bridge_df, figure_paths):
    rq3_rows = metric_value(metrics_df, "rq3_rows", 0)
    duplicate_keys = metric_value(metrics_df, "duplicate_issue_keys", 0)
    repo_count = metric_value(metrics_df, "repo_count", 0)
    wontfix_rows = metric_value(metrics_df, "wontfix_rows", 0)
    comparison_rows = metric_value(metrics_df, "comparison_rows", 0)
    zero_comment_rows = metric_value(metrics_df, "zero_comment_rows", 0)
    zero_comment_share = metric_value(metrics_df, "zero_comment_share", np.nan)
    rows_usable_for_rq3 = metric_value(metrics_df, "rows_usable_for_rq3", 0)

    lines = []
    lines.append("# Participation Analysis Report")
    lines.append("")
    lines.append("## Part 1: Participation coverage and population QA")
    lines.append("")
    lines.append("This section summarizes the issue population and participation-feature coverage for the RQ3 issue-level analysis dataset.")
    lines.append("")
    lines.append("### Headline QA")
    lines.append("")
    lines.append("- Issue rows: **{0}**".format(format_int(rq3_rows)))
    lines.append("- Repositories represented: **{0}**".format(format_int(repo_count)))
    lines.append("- WONTFIX rows: **{0}**".format(format_int(wontfix_rows)))
    lines.append("- Comparison rows: **{0}**".format(format_int(comparison_rows)))
    lines.append("- Rows usable for RQ3: **{0}**".format(format_int(rows_usable_for_rq3)))
    lines.append("- Duplicate issue keys: **{0}**".format(format_int(duplicate_keys)))
    lines.append("- Zero-comment issues: **{0}** ({1})".format(format_int(zero_comment_rows), format_pct(zero_comment_share)))
    lines.append("")

    lines.append("### Population summary")
    lines.append("")
    lines.append(dataframe_to_markdown(format_percent_columns(population_df, ["zero_comment_share"])))
    lines.append("")

    lines.append("### Comparison-group summary")
    lines.append("")
    lines.append("This table keeps comparison-set detail visible for later subgroup analyses.")
    lines.append("")
    lines.append(dataframe_to_markdown(format_percent_columns(comparison_group_df, ["zero_comment_share"])))
    lines.append("")

    lines.append("### Coverage by repository and analysis group")
    lines.append("")
    if figure_paths.get("coverage"):
        lines.append("![Participation coverage by repository and analysis group]({0})".format(figure_paths["coverage"]))
    else:
        lines.append("_Coverage figure was not generated._")
    lines.append("")
    display_coverage = coverage_df.copy()
    if "share" in display_coverage.columns:
        display_coverage["share"] = display_coverage["share"].apply(format_pct)
    lines.append(dataframe_to_markdown(display_coverage))
    lines.append("")

    lines.append("### Zero-comment issues by repository")
    lines.append("")
    lines.append("Zero-comment rows are retained because they are meaningful participation outcomes, not row-loss artifacts.")
    lines.append("")
    lines.append(dataframe_to_markdown(format_percent_columns(zero_comment_repo_df, ["zero_comment_share"])))
    lines.append("")

    lines.append("### Core participation feature missingness")
    lines.append("")
    lines.append(dataframe_to_markdown(format_percent_columns(missingness_df, ["missing_share"])))
    lines.append("")

    lines.append("### Zero-comment ratio/concentration applicability check")
    lines.append("")
    lines.append("This check prevents later analyses from treating ratio or concentration values on zero-comment issues as substantively meaningful zeros.")
    lines.append("")
    lines.append(dataframe_to_markdown(ratio_applicability_df))
    lines.append("")

    lines.append("## Part 2: Core participation differences, WONTFIX vs comparison")
    lines.append("")
    lines.append("This section compares WONTFIX issues with the pooled comparison set across discussion volume, breadth, concentration, and issue-author/non-author involvement. It is descriptive; later sections can add effect sizes, repo fixed effects, sentiment bridges, and ownership bridges.")
    lines.append("")

    if figure_paths.get("volume"):
        lines.append("### Discussion volume")
        lines.append("")
        lines.append("![Comment volume distribution]({0})".format(figure_paths["volume"]))
        lines.append("")
    if figure_paths.get("breadth"):
        lines.append("### Breadth of participation")
        lines.append("")
        lines.append("![Breadth of participation by analysis group]({0})".format(figure_paths["breadth"]))
        lines.append("")
    if figure_paths.get("concentration"):
        lines.append("### Discussion concentration")
        lines.append("")
        lines.append("![Discussion concentration by analysis group]({0})".format(figure_paths["concentration"]))
        lines.append("")
    if figure_paths.get("author"):
        lines.append("### Issue-author and non-author involvement")
        lines.append("")
        lines.append("![Issue-author and non-author participation]({0})".format(figure_paths["author"]))
        lines.append("")

    lines.append("### WONTFIX vs comparison difference summary")
    lines.append("")
    lines.append(dataframe_to_markdown(compact_difference_table(difference_df)))
    lines.append("")

    lines.append("### Group-level descriptive summary")
    lines.append("")
    display_group_summary = group_summary_df.copy()
    for col in ["rate"]:
        if col in display_group_summary.columns:
            display_group_summary[col] = display_group_summary[col].apply(format_pct)
    for col in ["mean", "median", "std", "q1", "q3", "min", "max"]:
        if col in display_group_summary.columns:
            display_group_summary[col] = display_group_summary[col].apply(lambda x: format_float(x, 3))
    lines.append(dataframe_to_markdown(display_group_summary, max_rows=50))
    if len(group_summary_df) > 50:
        lines.append("")
        lines.append("_Table truncated in Markdown; see `participation_group_summary.csv` for all rows._")
    lines.append("")

    lines.append("### Repo-aware descriptive summary")
    lines.append("")
    lines.append("The repo-level table is written for later richer analysis and should be used to check whether pooled WONTFIX/comparison differences are driven by one repository.")
    lines.append("")
    display_repo_summary = repo_group_summary_df.copy()
    for col in ["zero_comment_share"]:
        if col in display_repo_summary.columns:
            display_repo_summary[col] = display_repo_summary[col].apply(format_pct)
    for col in display_repo_summary.columns:
        if col.endswith("__mean") or col.endswith("__median"):
            display_repo_summary[col] = display_repo_summary[col].apply(lambda x: format_float(x, 3))
    lines.append(dataframe_to_markdown(display_repo_summary, max_rows=24))
    if len(repo_group_summary_df) > 24:
        lines.append("")
        lines.append("_Table truncated in Markdown; see `participation_repo_group_summary.csv` for all rows._")
    lines.append("")


    lines.append("## Part 3: Repo-aware effect sizes and models")
    lines.append("")
    lines.append("This section quantifies the WONTFIX-vs-comparison differences from Part 2 while accounting for repository-level participation norms. Numeric features are summarized with repo-weighted Hedges g; binary features are summarized with repo-weighted rate differences. The model table fits simple repo-fixed-effect models where possible, using logistic regression for binary outcomes and OLS for numeric outcomes.")
    lines.append("")

    if figure_paths.get("effect_sizes"):
        lines.append("### Repo-aware effect-size overview")
        lines.append("")
        lines.append("![Repo-aware WONTFIX vs comparison effect sizes]({0})".format(figure_paths["effect_sizes"]))
        lines.append("")
    if figure_paths.get("repo_heatmap"):
        lines.append("### Repo-specific WONTFIX minus comparison differences")
        lines.append("")
        lines.append("![Within-repo participation differences]({0})".format(figure_paths["repo_heatmap"]))
        lines.append("")

    lines.append("### Effect-size summary")
    lines.append("")
    lines.append(dataframe_to_markdown(compact_effect_size_table(effect_size_df)))
    lines.append("")

    lines.append("### Repo-fixed-effect model results")
    lines.append("")
    lines.append("These models are robustness checks, not causal estimates. They estimate the WONTFIX coefficient after absorbing baseline differences across repositories.")
    lines.append("")
    lines.append(dataframe_to_markdown(compact_model_results_table(model_results_df), max_rows=30))
    if len(model_results_df) > 30:
        lines.append("")
        lines.append("_Table truncated in Markdown; see `participation_model_results.csv` for all rows._")
    lines.append("")

    lines.append("### Repo-level effect-size detail")
    lines.append("")
    lines.append("The full repo-level effect table is written for diagnosing whether pooled effects are consistent across repositories or driven by one project.")
    lines.append("")
    display_repo_effect = repo_effect_df.copy()
    for col in ["wontfix_mean_or_rate", "comparison_mean_or_rate", "raw_difference_wontfix_minus_comparison", "cohen_d_wontfix_minus_comparison", "hedges_g_wontfix_minus_comparison", "log_odds_ratio_wontfix_minus_comparison"]:
        if col in display_repo_effect.columns:
            display_repo_effect[col] = display_repo_effect[col].apply(lambda x: format_float(x, 3))
    lines.append(dataframe_to_markdown(display_repo_effect, max_rows=32))
    if len(repo_effect_df) > 32:
        lines.append("")
        lines.append("_Table truncated in Markdown; see `participation_repo_effect_sizes.csv` for all rows._")
    lines.append("")

    lines.append("## Part 4: Participation + sentiment bridge")
    lines.append("")
    lines.append("This section connects participation structure to issue-level sentiment features. It is exploratory and descriptive: participation patterns are used to contextualize sentiment, not to claim that participation causes sentiment.")
    lines.append("")

    if figure_paths.get("sentiment_breadth"):
        lines.append("### Sentiment by participation breadth")
        lines.append("")
        lines.append("![Mean sentiment by participation breadth bucket]({0})".format(figure_paths["sentiment_breadth"]))
        lines.append("")
    if figure_paths.get("sentiment_concentration"):
        lines.append("### Sentiment by discussion concentration")
        lines.append("")
        lines.append("![Mean sentiment by discussion concentration bucket]({0})".format(figure_paths["sentiment_concentration"]))
        lines.append("")
    if figure_paths.get("sentiment_model_bridge"):
        lines.append("### WONTFIX sentiment effect before and after participation controls")
        lines.append("")
        lines.append("![WONTFIX sentiment effect before and after participation controls]({0})".format(figure_paths["sentiment_model_bridge"]))
        lines.append("")

    lines.append("### Participation/sentiment correlation summary")
    lines.append("")
    lines.append("The table shows the strongest absolute correlations in the full issue population. Correlations are descriptive and should be read as association, not directionality.")
    lines.append("")
    lines.append(dataframe_to_markdown(compact_sentiment_correlation_table(sentiment_correlation_df)))
    lines.append("")

    lines.append("### Sentiment by participation bucket")
    lines.append("")
    lines.append("These tables summarize mean comment sentiment by participation buckets. Breadth buckets can include zero-comment issues; concentration buckets are restricted to comment-bearing issues.")
    lines.append("")
    lines.append("#### Breadth buckets")
    lines.append("")
    lines.append(dataframe_to_markdown(compact_bucket_table(sentiment_bucket_df, dimension="participation_breadth", sentiment_feature="mean_comment_sentiment"), max_rows=20))
    lines.append("")
    lines.append("#### Concentration buckets")
    lines.append("")
    lines.append(dataframe_to_markdown(compact_bucket_table(sentiment_bucket_df, dimension="discussion_concentration", sentiment_feature="mean_comment_sentiment"), max_rows=20))
    lines.append("")

    lines.append("### Sentiment bridge model results")
    lines.append("")
    lines.append("These repo-fixed-effect OLS models compare the WONTFIX sentiment coefficient before and after adding basic participation controls. If the WONTFIX coefficient changes very little, participation structure probably does not explain much of the sentiment difference. If it shrinks meaningfully, participation may be part of the sentiment story.")
    lines.append("")
    lines.append(dataframe_to_markdown(compact_sentiment_bridge_model_table(sentiment_model_bridge_df), max_rows=30))
    if len(sentiment_model_bridge_df) > 30:
        lines.append("")
        lines.append("_Table truncated in Markdown; see `participation_sentiment_bridge_models.csv` for all rows._")
    lines.append("")

    lines.append("### Notes for interpretation")
    lines.append("")
    lines.append("- `zero_comments` rows are part of the issue population and should remain visible in participation analyses.")
    lines.append("- Ratio/concentration fields on zero-comment rows should be treated as not applicable, even when they are encoded as `0.0` rather than missing.")
    lines.append("- Comment-volume and breadth summaries can include zero-comment issues; concentration summaries should use comment-bearing issues only.")
    lines.append("- Repo-aware summaries are important because participation norms vary by repository.")
    lines.append("")

    lines.append("### Output tables")
    lines.append("")
    lines.append("- `participation_qa_summary.csv`")
    lines.append("- `participation_population_summary.csv`")
    lines.append("- `participation_comparison_group_summary.csv`")
    lines.append("- `participation_coverage_by_repo.csv`")
    lines.append("- `participation_zero_comment_by_repo.csv`")
    lines.append("- `participation_feature_missingness.csv`")
    lines.append("- `participation_zero_comment_ratio_applicability.csv`")
    lines.append("- `participation_group_summary.csv`")
    lines.append("- `participation_repo_group_summary.csv`")
    lines.append("- `participation_core_difference_summary.csv`")
    lines.append("- `participation_effect_sizes.csv`")
    lines.append("- `participation_repo_effect_sizes.csv`")
    lines.append("- `participation_model_results.csv`")
    lines.append("- `participation_sentiment_correlations.csv`")
    lines.append("- `participation_sentiment_by_bucket.csv`")
    lines.append("- `participation_sentiment_bridge_models.csv`")
    lines.append("")
    return "\n".join(lines)


# -----------------------------
# Main orchestration
# -----------------------------


def run_report(args):
    output_dir = ensure_dir(args.output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    rq3_path = Path(args.rq3_dataset)
    analysis_qa_path = Path(args.analysis_qa) if args.analysis_qa else None

    issue_df_raw = read_table(rq3_path)
    if analysis_qa_path is not None:
        analysis_qa_df = maybe_read_csv(analysis_qa_path)
        if analysis_qa_df.empty and not args.allow_missing_analysis_qa:
            raise FileNotFoundError(
                "Analysis QA summary not found or empty: {0}. Use --allow-missing-analysis-qa to continue.".format(analysis_qa_path)
            )
    else:
        analysis_qa_df = pd.DataFrame()

    issue_df = normalize_dataset(issue_df_raw)

    metrics_df = make_metric_rows(issue_df, analysis_qa_df)
    population_df = make_population_summary(issue_df)
    comparison_group_df = make_comparison_group_summary(issue_df)
    coverage_df = make_coverage_by_repo(issue_df)
    zero_comment_repo_df = make_zero_comment_by_repo(issue_df)
    missingness_df = make_missingness_summary(issue_df)
    ratio_applicability_df = make_zero_comment_ratio_applicability(issue_df)

    group_summary_df = make_participation_group_summary(issue_df)
    repo_group_summary_df = make_participation_repo_group_summary(issue_df)
    difference_df = make_participation_difference_summary(issue_df)
    repo_effect_df = make_repo_effect_sizes(issue_df)
    effect_size_df = make_overall_effect_sizes(issue_df, repo_effect_df)
    model_results_df = make_model_results(issue_df)
    sentiment_correlation_df = make_participation_sentiment_correlation(issue_df)
    sentiment_bucket_df = make_sentiment_by_participation_bucket(issue_df)
    sentiment_model_bridge_df = make_participation_sentiment_model_bridge(issue_df)

    qa_path = write_csv(metrics_df, output_dir / "participation_qa_summary.csv")
    population_path = write_csv(population_df, output_dir / "participation_population_summary.csv")
    comparison_group_path = write_csv(comparison_group_df, output_dir / "participation_comparison_group_summary.csv")
    coverage_path = write_csv(coverage_df, output_dir / "participation_coverage_by_repo.csv")
    zero_comment_repo_path = write_csv(zero_comment_repo_df, output_dir / "participation_zero_comment_by_repo.csv")
    missingness_path = write_csv(missingness_df, output_dir / "participation_feature_missingness.csv")
    ratio_applicability_path = write_csv(ratio_applicability_df, output_dir / "participation_zero_comment_ratio_applicability.csv")
    group_summary_path = write_csv(group_summary_df, output_dir / "participation_group_summary.csv")
    repo_group_summary_path = write_csv(repo_group_summary_df, output_dir / "participation_repo_group_summary.csv")
    difference_path = write_csv(difference_df, output_dir / "participation_core_difference_summary.csv")
    repo_effect_path = write_csv(repo_effect_df, output_dir / "participation_repo_effect_sizes.csv")
    effect_size_path = write_csv(effect_size_df, output_dir / "participation_effect_sizes.csv")
    model_results_path = write_csv(model_results_df, output_dir / "participation_model_results.csv")
    sentiment_correlation_path = write_csv(sentiment_correlation_df, output_dir / "participation_sentiment_correlations.csv")
    sentiment_bucket_path = write_csv(sentiment_bucket_df, output_dir / "participation_sentiment_by_bucket.csv")
    sentiment_model_bridge_path = write_csv(sentiment_model_bridge_df, output_dir / "participation_sentiment_bridge_models.csv")

    coverage_figure_path = figures_dir / "01_participation_coverage_by_repo.png"
    volume_figure_path = figures_dir / "02_comment_volume_distribution.png"
    breadth_figure_path = figures_dir / "03_participation_breadth_by_group.png"
    concentration_figure_path = figures_dir / "04_discussion_concentration_by_group.png"
    author_figure_path = figures_dir / "05_issue_author_participation.png"
    effect_size_figure_path = figures_dir / "06_participation_effect_sizes.png"
    repo_heatmap_figure_path = figures_dir / "07_participation_by_repo_heatmap.png"
    sentiment_breadth_figure_path = figures_dir / "08_sentiment_by_participation_breadth.png"
    sentiment_concentration_figure_path = figures_dir / "09_sentiment_by_discussion_concentration.png"
    sentiment_model_bridge_figure_path = figures_dir / "10_wontfix_sentiment_effect_with_participation_controls.png"

    plot_coverage_by_repo(coverage_df, coverage_figure_path, args.png_dpi)
    plot_comment_volume_distribution(issue_df, volume_figure_path, args.png_dpi)
    plot_breadth_by_group(issue_df, breadth_figure_path, args.png_dpi)
    plot_concentration_by_group(issue_df, concentration_figure_path, args.png_dpi)
    plot_author_participation(issue_df, author_figure_path, args.png_dpi)
    plot_effect_sizes(effect_size_df, effect_size_figure_path, args.png_dpi)
    plot_repo_effect_heatmap(repo_effect_df, repo_heatmap_figure_path, args.png_dpi)
    plot_sentiment_by_participation_bucket(
        sentiment_bucket_df,
        "participation_breadth",
        "mean_comment_sentiment",
        "Mean comment sentiment by participation breadth",
        sentiment_breadth_figure_path,
        args.png_dpi,
    )
    plot_sentiment_by_participation_bucket(
        sentiment_bucket_df,
        "discussion_concentration",
        "mean_comment_sentiment",
        "Mean comment sentiment by discussion concentration",
        sentiment_concentration_figure_path,
        args.png_dpi,
    )
    plot_sentiment_model_bridge(sentiment_model_bridge_df, sentiment_model_bridge_figure_path, args.png_dpi)

    figure_paths = {
        "coverage": "figures/{0}".format(coverage_figure_path.name) if coverage_figure_path.exists() else None,
        "volume": "figures/{0}".format(volume_figure_path.name) if volume_figure_path.exists() else None,
        "breadth": "figures/{0}".format(breadth_figure_path.name) if breadth_figure_path.exists() else None,
        "concentration": "figures/{0}".format(concentration_figure_path.name) if concentration_figure_path.exists() else None,
        "author": "figures/{0}".format(author_figure_path.name) if author_figure_path.exists() else None,
        "effect_sizes": "figures/{0}".format(effect_size_figure_path.name) if effect_size_figure_path.exists() else None,
        "repo_heatmap": "figures/{0}".format(repo_heatmap_figure_path.name) if repo_heatmap_figure_path.exists() else None,
        "sentiment_breadth": "figures/{0}".format(sentiment_breadth_figure_path.name) if sentiment_breadth_figure_path.exists() else None,
        "sentiment_concentration": "figures/{0}".format(sentiment_concentration_figure_path.name) if sentiment_concentration_figure_path.exists() else None,
        "sentiment_model_bridge": "figures/{0}".format(sentiment_model_bridge_figure_path.name) if sentiment_model_bridge_figure_path.exists() else None,
    }

    report_text = build_markdown_report(
        metrics_df,
        population_df,
        comparison_group_df,
        coverage_df,
        zero_comment_repo_df,
        missingness_df,
        ratio_applicability_df,
        group_summary_df,
        repo_group_summary_df,
        difference_df,
        repo_effect_df,
        effect_size_df,
        model_results_df,
        sentiment_correlation_df,
        sentiment_bucket_df,
        sentiment_model_bridge_df,
        figure_paths,
    )
    report_path = write_text(report_text, output_dir / "participation_analysis_report.md")

    manifest = {
        "status": "completed",
        "started_at_utc": None,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "rq3_dataset": str(rq3_path),
        "analysis_qa": str(analysis_qa_path) if analysis_qa_path else None,
        "rows_read": int(len(issue_df_raw)),
        "rows_normalized": int(len(issue_df)),
        "outputs": {
            "qa_summary": str(qa_path),
            "population_summary": str(population_path),
            "comparison_group_summary": str(comparison_group_path),
            "coverage_by_repo": str(coverage_path),
            "zero_comment_by_repo": str(zero_comment_repo_path),
            "feature_missingness": str(missingness_path),
            "zero_comment_ratio_applicability": str(ratio_applicability_path),
            "participation_group_summary": str(group_summary_path),
            "participation_repo_group_summary": str(repo_group_summary_path),
            "participation_core_difference_summary": str(difference_path),
            "participation_repo_effect_sizes": str(repo_effect_path),
            "participation_effect_sizes": str(effect_size_path),
            "participation_model_results": str(model_results_path),
            "participation_sentiment_correlations": str(sentiment_correlation_path),
            "participation_sentiment_by_bucket": str(sentiment_bucket_path),
            "participation_sentiment_bridge_models": str(sentiment_model_bridge_path),
            "coverage_figure": str(coverage_figure_path),
            "comment_volume_figure": str(volume_figure_path),
            "breadth_figure": str(breadth_figure_path),
            "concentration_figure": str(concentration_figure_path),
            "author_participation_figure": str(author_figure_path),
            "participation_effect_sizes_figure": str(effect_size_figure_path),
            "participation_by_repo_heatmap": str(repo_heatmap_figure_path),
            "sentiment_breadth_figure": str(sentiment_breadth_figure_path),
            "sentiment_concentration_figure": str(sentiment_concentration_figure_path),
            "sentiment_model_bridge_figure": str(sentiment_model_bridge_figure_path),
            "markdown_report": str(report_path),
        },
    }
    manifest_path = write_json(manifest, output_dir / "15_build_participation_report_run_manifest.json")

    return {
        "metrics": metrics_df,
        "population": population_df,
        "comparison_group": comparison_group_df,
        "coverage": coverage_df,
        "zero_comment_by_repo": zero_comment_repo_df,
        "missingness": missingness_df,
        "ratio_applicability": ratio_applicability_df,
        "group_summary": group_summary_df,
        "repo_group_summary": repo_group_summary_df,
        "difference": difference_df,
        "repo_effect_sizes": repo_effect_df,
        "effect_sizes": effect_size_df,
        "model_results": model_results_df,
        "sentiment_correlations": sentiment_correlation_df,
        "sentiment_by_bucket": sentiment_bucket_df,
        "sentiment_bridge_models": sentiment_model_bridge_df,
        "manifest_path": manifest_path,
        "report_path": report_path,
    }


def main():
    args = parse_args()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        result = run_report(args)
        manifest_path = result.get("manifest_path")
        if manifest_path and Path(manifest_path).exists():
            with Path(manifest_path).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["started_at_utc"] = started_at
            payload["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(payload, manifest_path)
        print("Participation report complete: {0}".format(result["report_path"]))
    except Exception as exc:
        output_dir = ensure_dir(getattr(args, "output_dir", DEFAULT_OUTPUT_DIR))
        failure_manifest = {
            "status": "failed",
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_message": str(exc),
            "rq3_dataset": getattr(args, "rq3_dataset", None),
            "analysis_qa": getattr(args, "analysis_qa", None),
        }
        write_json(failure_manifest, output_dir / "15_build_participation_report_run_manifest.json")
        raise


if __name__ == "__main__":
    main()
