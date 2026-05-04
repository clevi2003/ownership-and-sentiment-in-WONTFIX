#!/usr/bin/env python3
"""
Build the RQ2 ownership analysis report for the WONTFIX pipeline.

This script is designed to run after scripts/13_build_analysis_dataset.py. It reads
RQ2's merged issue-level analysis dataset and analyzes WONTFIX-vs-comparison
differences using the ownership feature families that passed the readiness stage.

Analysis stance
---------------
Primary:
  Repo-level participant roles. These use all RQ2 issues where issue/comment
  participants have resolved contributor keys.

Secondary / stratified:
  File-level participant roles. These are analyzed with the explicit applicable
  denominator: participant_role_file_features_applicable == 1.

Descriptive / conditional:
  Direct issue-linked ownership and pre/post continuity. These are useful for
  context but are conditional on PR/commit/file-link evidence and should not be
  treated as the primary WONTFIX-vs-comparison ownership result.

Typical usage:
    python scripts/17_build_ownership_analysis_report.py \
        --rq2-dataset data/final/analysis_dataset_rq2.parquet \
        --output-dir outputs/ownership_analysis
"""

import argparse
import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
except Exception:
    stats = None

try:
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
except Exception:
    smf = None
    ConvergenceWarning = None


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
    "missing": UTK_COLORS["light_gray"],
}

GROUP_LABELS = {
    "wontfix": "WONTFIX",
    "comparison": "Comparison",
    "missing": "Missing",
}

DEFAULT_RQ2_DATASET = "data/final/analysis_dataset_rq2.parquet"
DEFAULT_ANALYSIS_QA = "logs/qa/analysis_dataset_qa_summary.csv"
DEFAULT_READINESS_DIR = "outputs/ownership_readiness"
DEFAULT_OUTPUT_DIR = "outputs/ownership_analysis"
MIN_GROUP_N_FOR_TEST = 5
MIN_MODEL_TOTAL_N = 40
MIN_MODEL_GROUP_N = 10

PRIMARY_REPO_BINARY_FEATURES = [
    {
        "feature": "issue_author_is_pre_issue_repo_contributor",
        "label": "Issue author was prior repo contributor",
        "family": "repo_participant_roles",
        "denominator": "all_rq2_issues",
        "interpretation": "Whether the issue author had committed to the repository before the issue was created.",
    },
    {
        "feature": "issue_author_is_pre_issue_major_repo_contributor",
        "label": "Issue author was major prior repo contributor",
        "family": "repo_participant_roles",
        "denominator": "all_rq2_issues",
        "interpretation": "Whether the issue author met the configured major prior repo contributor threshold.",
    },
    {
        "feature": "any_commenter_is_pre_issue_repo_contributor",
        "label": "Any commenter was prior repo contributor",
        "family": "repo_participant_roles",
        "denominator": "issues_with_resolved_commenters",
        "interpretation": "Whether at least one resolved commenter had committed to the repository before the issue.",
    },
    {
        "feature": "any_commenter_is_pre_issue_major_repo_contributor",
        "label": "Any commenter was major prior repo contributor",
        "family": "repo_participant_roles",
        "denominator": "issues_with_resolved_commenters",
        "interpretation": "Whether at least one resolved commenter was a major prior repo contributor.",
    },
    {
        "feature": "top_commenter_is_pre_issue_repo_contributor",
        "label": "Top commenter was prior repo contributor",
        "family": "repo_participant_roles",
        "denominator": "issues_with_top_commenter",
        "interpretation": "Whether the most active resolved commenter had pre-issue repo history.",
    },
    {
        "feature": "top_commenter_is_pre_issue_major_repo_contributor",
        "label": "Top commenter was major prior repo contributor",
        "family": "repo_participant_roles",
        "denominator": "issues_with_top_commenter",
        "interpretation": "Whether the most active resolved commenter was a major prior repo contributor.",
    },
]

PRIMARY_REPO_NUMERIC_FEATURES = [
    {
        "feature": "share_commenters_pre_issue_repo_contributors",
        "label": "Share of commenters who were prior repo contributors",
        "family": "repo_participant_roles",
        "denominator": "issues_with_resolved_commenters",
        "interpretation": "Fraction of resolved commenters with pre-issue repo commit history.",
    },
    {
        "feature": "share_commenters_pre_issue_major_repo_contributors",
        "label": "Share of commenters who were major prior repo contributors",
        "family": "repo_participant_roles",
        "denominator": "issues_with_resolved_commenters",
        "interpretation": "Fraction of resolved commenters who met the major prior repo contributor threshold.",
    },
    {
        "feature": "commenter_count_pre_issue_repo_contributors",
        "label": "Count of prior repo contributor commenters",
        "family": "repo_participant_roles",
        "denominator": "issues_with_resolved_commenters",
        "interpretation": "Number of resolved commenters with pre-issue repo commit history.",
    },
]

SECONDARY_FILE_BINARY_FEATURES = [
    {
        "feature": "participant_role_file_features_applicable",
        "label": "File-role features applicable",
        "family": "file_participant_roles",
        "denominator": "all_rq2_issues",
        "interpretation": "Whether the issue has usable linked-file context and pre-issue file history.",
    },
    {
        "feature": "issue_author_is_pre_issue_file_contributor",
        "label": "Issue author was prior linked-file contributor",
        "family": "file_participant_roles",
        "denominator": "file_applicable_issues",
        "interpretation": "Whether the issue author previously touched at least one linked file.",
    },
    {
        "feature": "issue_author_is_pre_issue_major_file_contributor",
        "label": "Issue author was major prior linked-file contributor",
        "family": "file_participant_roles",
        "denominator": "file_applicable_issues",
        "interpretation": "Whether the issue author met the configured major linked-file contributor threshold.",
    },
    {
        "feature": "any_commenter_is_pre_issue_file_contributor",
        "label": "Any commenter was prior linked-file contributor",
        "family": "file_participant_roles",
        "denominator": "file_applicable_issues_with_resolved_commenters",
        "interpretation": "Whether at least one resolved commenter previously touched a linked file.",
    },
    {
        "feature": "any_commenter_is_pre_issue_major_file_contributor",
        "label": "Any commenter was major prior linked-file contributor",
        "family": "file_participant_roles",
        "denominator": "file_applicable_issues_with_resolved_commenters",
        "interpretation": "Whether at least one resolved commenter was a major linked-file contributor.",
    },
    {
        "feature": "top_commenter_is_pre_issue_file_contributor",
        "label": "Top commenter was prior linked-file contributor",
        "family": "file_participant_roles",
        "denominator": "file_applicable_issues_with_top_commenter",
        "interpretation": "Whether the most active resolved commenter previously touched a linked file.",
    },
]

SECONDARY_FILE_NUMERIC_FEATURES = [
    {
        "feature": "share_commenters_pre_issue_file_contributors",
        "label": "Share of commenters who were prior linked-file contributors",
        "family": "file_participant_roles",
        "denominator": "file_applicable_issues_with_resolved_commenters",
        "interpretation": "Fraction of resolved commenters with prior linked-file history.",
    },
    {
        "feature": "share_commenters_pre_issue_major_file_contributors",
        "label": "Share of commenters who were major prior linked-file contributors",
        "family": "file_participant_roles",
        "denominator": "file_applicable_issues_with_resolved_commenters",
        "interpretation": "Fraction of resolved commenters who were major contributors on linked files.",
    },
    {
        "feature": "commenter_count_pre_issue_file_contributors",
        "label": "Count of prior linked-file contributor commenters",
        "family": "file_participant_roles",
        "denominator": "file_applicable_issues_with_resolved_commenters",
        "interpretation": "Number of resolved commenters with prior linked-file history.",
    },
]

DESCRIPTIVE_BINARY_FEATURES = [
    {
        "feature": "has_post_issue_ownership",
        "label": "Direct post-issue ownership evidence",
        "family": "direct_issue_linked_ownership",
        "denominator": "all_rq2_issues",
        "interpretation": "Whether the issue had selected post-issue PR/commit/file ownership evidence.",
    },
    {
        "feature": "has_pre_issue_issue_linked_ownership",
        "label": "Direct pre-issue ownership evidence",
        "family": "direct_issue_linked_ownership",
        "denominator": "all_rq2_issues",
        "interpretation": "Whether the issue had selected pre-issue issue-linked ownership evidence.",
    },
    {
        "feature": "any_post_issue_owner_with_pre_issue_repo_history",
        "label": "Post-issue owner had prior repo history",
        "family": "continuity",
        "denominator": "issues_with_post_issue_owners",
        "interpretation": "Whether any eventual post-issue owner had pre-issue repo commit history.",
    },
    {
        "feature": "any_post_issue_owner_with_pre_issue_file_history",
        "label": "Post-issue owner had prior linked-file history",
        "family": "continuity",
        "denominator": "issues_with_post_issue_owners",
        "interpretation": "Whether any eventual post-issue owner had pre-issue linked-file history.",
    },
    {
        "feature": "any_commenter_is_eventual_post_issue_owner",
        "label": "Any commenter became post-issue owner",
        "family": "continuity",
        "denominator": "issues_with_post_issue_owners_and_resolved_commenters",
        "interpretation": "Whether a resolved commenter appears among eventual post-issue owners.",
    },
]

DESCRIPTIVE_NUMERIC_FEATURES = [
    {
        "feature": "share_post_issue_owners_with_pre_issue_repo_history",
        "label": "Share of post-issue owners with prior repo history",
        "family": "continuity",
        "denominator": "issues_with_post_issue_owners",
        "interpretation": "Fraction of eventual owners who had prior repo commit history.",
    },
    {
        "feature": "share_post_issue_owners_with_pre_issue_file_history",
        "label": "Share of post-issue owners with prior linked-file history",
        "family": "continuity",
        "denominator": "issues_with_post_issue_owners",
        "interpretation": "Fraction of eventual owners who had prior linked-file history.",
    },
    {
        "feature": "share_commenters_eventual_post_issue_owners",
        "label": "Share of commenters who became post-issue owners",
        "family": "continuity",
        "denominator": "issues_with_post_issue_owners_and_resolved_commenters",
        "interpretation": "Fraction of resolved commenters who appear among eventual post-issue owners.",
    },
]

ALL_FEATURE_SPECS = (
    PRIMARY_REPO_BINARY_FEATURES
    + PRIMARY_REPO_NUMERIC_FEATURES
    + SECONDARY_FILE_BINARY_FEATURES
    + SECONDARY_FILE_NUMERIC_FEATURES
    + DESCRIPTIVE_BINARY_FEATURES
    + DESCRIPTIVE_NUMERIC_FEATURES
)

BINARY_FEATURE_NAMES = {
    spec["feature"]
    for spec in PRIMARY_REPO_BINARY_FEATURES + SECONDARY_FILE_BINARY_FEATURES + DESCRIPTIVE_BINARY_FEATURES
}


# -----------------------------
# CLI / I/O helpers
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Build RQ2 ownership analysis report for WONTFIX issues.")
    parser.add_argument("--rq2-dataset", default=DEFAULT_RQ2_DATASET)
    parser.add_argument("--analysis-qa", default=DEFAULT_ANALYSIS_QA)
    parser.add_argument("--readiness-dir", default=DEFAULT_READINESS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--png-dpi", type=int, default=220)
    parser.add_argument("--min-group-n", type=int, default=MIN_GROUP_N_FOR_TEST)
    parser.add_argument("--allow-missing-readiness", action="store_true")
    return parser.parse_args()

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def read_table(path, required=True):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError("Input table does not exist: {0}".format(path))
        return pd.DataFrame()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError("Unsupported input table format for {0}. Expected .parquet, .csv, or .json.".format(path))

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
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return None
    return text

def normalize_analysis_set(value):
    text = clean_text(value)
    if text is None:
        return "missing"
    lowered = text.lower().replace("-", "_").replace(" ", "_")
    if lowered in {"wontfix", "wont_fix", "won_t_fix", "won't_fix"}:
        return "wontfix"
    if lowered in {"comparison", "control", "controls", "non_wontfix", "non_wontfix_comparison"}:
        return "comparison"
    return lowered

def to_numeric(series):
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")

def to_flag(series):
    return to_numeric(series).fillna(0).astype(int)

def safe_divide(numer, denom, default=np.nan):
    try:
        if denom is None or pd.isna(denom) or float(denom) == 0.0:
            return default
        return float(numer) / float(denom)
    except Exception:
        return default

def maybe_float(value):
    try:
        if value is None or pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan

def format_rate(value):
    if value is None or pd.isna(value):
        return "n/a"
    return "{0:.1%}".format(float(value))

def format_number(value):
    if value is None or pd.isna(value):
        return "n/a"
    return "{0:,.0f}".format(float(value))

def cohen_d(x, y):
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    pooled = math.sqrt(((len(x) - 1) * sx * sx + (len(y) - 1) * sy * sy) / max(len(x) + len(y) - 2, 1))
    if pooled == 0 or pd.isna(pooled):
        return np.nan
    return (x.mean() - y.mean()) / pooled

def cliffs_delta(x, y):
    x = pd.Series(x).dropna().astype(float).to_numpy()
    y = pd.Series(y).dropna().astype(float).to_numpy()
    if len(x) == 0 or len(y) == 0:
        return np.nan
    total = 0
    for xv in x:
        total += np.sum(xv > y) - np.sum(xv < y)
    return total / float(len(x) * len(y))

def odds_ratio_from_counts(a, b, c, d):
    # Haldane-Anscombe correction for zero cells.
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


# -----------------------------
# Dataset preparation
# -----------------------------


def denominator_mask(df, denominator_name):
    if denominator_name == "all_rq2_issues":
        return pd.Series(True, index=df.index)
    if denominator_name == "issues_with_resolved_commenters":
        return to_numeric(df.get("commenter_count_with_resolved_key", pd.Series(0, index=df.index))).fillna(0) > 0
    if denominator_name == "issues_with_top_commenter":
        return to_numeric(df.get("top_commenter_comment_count", pd.Series(0, index=df.index))).fillna(0) > 0
    if denominator_name == "file_applicable_issues":
        return to_numeric(df.get("participant_role_file_features_applicable", pd.Series(0, index=df.index))).fillna(0) > 0
    if denominator_name == "file_applicable_issues_with_resolved_commenters":
        applicable = to_numeric(df.get("participant_role_file_features_applicable", pd.Series(0, index=df.index))).fillna(0) > 0
        commenters = to_numeric(df.get("participant_role_file_commenter_count_with_resolved_key", df.get("commenter_count_with_resolved_key", pd.Series(0, index=df.index)))).fillna(0) > 0
        return applicable & commenters
    if denominator_name == "file_applicable_issues_with_top_commenter":
        applicable = to_numeric(df.get("participant_role_file_features_applicable", pd.Series(0, index=df.index))).fillna(0) > 0
        top_commenter = to_numeric(df.get("top_commenter_comment_count", pd.Series(0, index=df.index))).fillna(0) > 0
        return applicable & top_commenter
    if denominator_name == "issues_with_post_issue_owners":
        return to_numeric(df.get("post_issue_owner_count_for_continuity", pd.Series(0, index=df.index))).fillna(0) > 0
    if denominator_name == "issues_with_post_issue_owners_and_resolved_commenters":
        post = to_numeric(df.get("post_issue_owner_count_for_continuity", pd.Series(0, index=df.index))).fillna(0) > 0
        commenters = to_numeric(df.get("commenter_count_with_resolved_key", pd.Series(0, index=df.index))).fillna(0) > 0
        return post & commenters
    return pd.Series(True, index=df.index)

def add_missing_columns(df):
    out = df.copy()
    default_numeric = set()
    default_string = {
        "participant_role_file_coverage_flag": "missing",
    }
    for spec in ALL_FEATURE_SPECS:
        default_numeric.add(spec["feature"])
    for column in [
        "commenter_count_with_resolved_key",
        "top_commenter_comment_count",
        "participant_role_file_commenter_count_with_resolved_key",
        "participant_role_file_features_applicable",
        "post_issue_owner_count_for_continuity",
        "pre_issue_owner_count_for_continuity",
        "ownership_post_issue_contributor_count",
        "ownership_pre_issue_contributor_count",
        "ownership_has_post_issue_ownership",
        "ownership_has_pre_issue_ownership",
        "has_post_issue_ownership",
        "has_pre_issue_issue_linked_ownership",
        "usable_for_rq2",
        "usable_for_rq2_repo_participant_roles",
        "usable_for_rq2_file_participant_roles",
        "usable_for_rq2_direct_ownership",
        "usable_for_rq2_continuity",
        "has_repo_participant_role_signal",
        "has_file_participant_role_signal",
        "has_continuity_signal",
    ]:
        default_numeric.add(column)

    additions = {}
    for column, default in default_string.items():
        if column not in out.columns:
            additions[column] = pd.Series(default, index=out.index)
    for column in default_numeric:
        if column not in out.columns:
            additions[column] = pd.Series(0, index=out.index)
    if additions:
        out = pd.concat([out, pd.DataFrame(additions, index=out.index)], axis=1)
    return out.copy()

def normalize_dataset(df):
    if df.empty:
        raise ValueError("RQ2 dataset is empty.")
    required = ["repo_full_name", "issue_id", "analysis_set"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError("RQ2 dataset is missing required columns: {0}".format(missing))

    out = df.copy()
    out["repo_full_name"] = out["repo_full_name"].astype(str)
    out["issue_id"] = out["issue_id"].astype(str)
    if "issue_number" in out.columns:
        out["issue_number"] = to_numeric(out["issue_number"])
    else:
        out["issue_number"] = np.nan
    out["analysis_set"] = out["analysis_set"].apply(normalize_analysis_set)
    if "comparison_group" not in out.columns:
        out["comparison_group"] = out["analysis_set"]
    out["comparison_group"] = out["comparison_group"].fillna(out["analysis_set"]).astype(str)

    out = add_missing_columns(out)

    for spec in ALL_FEATURE_SPECS:
        feature = spec["feature"]
        if feature in BINARY_FEATURE_NAMES:
            out[feature] = to_flag(out[feature])
        else:
            out[feature] = to_numeric(out[feature])

    for column in [
        "commenter_count_with_resolved_key",
        "top_commenter_comment_count",
        "participant_role_file_commenter_count_with_resolved_key",
        "participant_role_file_features_applicable",
        "post_issue_owner_count_for_continuity",
        "usable_for_rq2",
        "usable_for_rq2_repo_participant_roles",
        "usable_for_rq2_file_participant_roles",
        "usable_for_rq2_direct_ownership",
        "usable_for_rq2_continuity",
        "has_repo_participant_role_signal",
        "has_file_participant_role_signal",
        "has_continuity_signal",
    ]:
        out[column] = to_numeric(out[column]).fillna(0)

    out["participant_role_file_coverage_flag"] = (
        out["participant_role_file_coverage_flag"]
        .fillna("missing")
        .astype(str)
        .str.strip()
        .replace({"": "missing", "nan": "missing", "None": "missing", "<NA>": "missing"})
    )

    out["has_post_issue_ownership"] = (
            (to_numeric(out.get("ownership_post_issue_contributor_count", pd.Series(0, index=out.index))).fillna(0) > 0)
            | (to_numeric(out.get("ownership_has_post_issue_ownership", pd.Series(0, index=out.index))).fillna(0) > 0)
            | (to_numeric(out.get("post_issue_owner_count_for_continuity", pd.Series(0, index=out.index))).fillna(
        0) > 0)
    ).astype(int)

    out["has_pre_issue_issue_linked_ownership"] = (
            (to_numeric(out.get("ownership_pre_issue_contributor_count", pd.Series(0, index=out.index))).fillna(0) > 0)
            | (to_numeric(out.get("ownership_has_pre_issue_ownership", pd.Series(0, index=out.index))).fillna(0) > 0)
            | (to_numeric(out.get("pre_issue_owner_count_for_continuity", pd.Series(0, index=out.index))).fillna(0) > 0)
    ).astype(int)

    out["analysis_set_wontfix"] = out["analysis_set"].eq("wontfix").astype(int)
    out["row_id"] = np.arange(len(out))
    return out.reset_index(drop=True).copy()

def load_and_prepare(paths):
    rq2_df = read_table(paths["rq2_dataset"], required=True)
    analysis_qa_df = read_table(paths["analysis_qa"], required=False)
    readiness_df = read_table(paths["feature_family_readiness"], required=False)
    candidate_df = read_table(paths["candidate_analysis_features"], required=False)

    qa = {
        "rq2_rows_raw": int(len(rq2_df)),
        "analysis_qa_rows": int(len(analysis_qa_df)),
        "readiness_rows": int(len(readiness_df)),
        "candidate_feature_rows": int(len(candidate_df)),
    }

    rq2_df = normalize_dataset(rq2_df)
    duplicate_keys = int(rq2_df.duplicated(subset=["repo_full_name", "issue_id", "issue_number"]).sum())
    qa["duplicate_issue_keys_before_dedupe"] = duplicate_keys
    rq2_df = rq2_df.drop_duplicates(subset=["repo_full_name", "issue_id", "issue_number"]).reset_index(drop=True)
    qa["rq2_rows_normalized"] = int(len(rq2_df))
    qa["repo_count"] = int(rq2_df["repo_full_name"].nunique())
    qa["wontfix_issue_count"] = int(rq2_df["analysis_set"].eq("wontfix").sum())
    qa["comparison_issue_count"] = int(rq2_df["analysis_set"].eq("comparison").sum())
    qa["usable_for_rq2_rows"] = int(to_numeric(rq2_df.get("usable_for_rq2", pd.Series(0, index=rq2_df.index))).fillna(0).sum())
    qa["file_applicable_rows"] = int(to_numeric(rq2_df.get("participant_role_file_features_applicable", pd.Series(0, index=rq2_df.index))).fillna(0).sum())
    qa["post_issue_owner_rows"] = int((to_numeric(rq2_df.get("post_issue_owner_count_for_continuity", pd.Series(0, index=rq2_df.index))).fillna(0) > 0).sum())

    return rq2_df, analysis_qa_df, readiness_df, candidate_df, qa

# -----------------------------
# Summary / effect calculations
# -----------------------------


def group_counts(df):
    rows = []
    for group, part in df.groupby("analysis_set", dropna=False):
        rows.append({
            "analysis_set": group,
            "issue_count": int(len(part)),
            "repo_count": int(part["repo_full_name"].nunique()),
            "resolved_commenter_issue_count": int((to_numeric(part.get("commenter_count_with_resolved_key", pd.Series(0, index=part.index))).fillna(0) > 0).sum()),
            "file_applicable_issue_count": int((to_numeric(part.get("participant_role_file_features_applicable", pd.Series(0, index=part.index))).fillna(0) > 0).sum()),
            "post_issue_owner_issue_count": int((to_numeric(part.get("post_issue_owner_count_for_continuity", pd.Series(0, index=part.index))).fillna(0) > 0).sum()),
        })
    return pd.DataFrame(rows).sort_values("analysis_set").reset_index(drop=True)

def population_summary(df):
    rows = []
    for repo, repo_part in df.groupby("repo_full_name", dropna=False):
        base = {"repo_full_name": repo, "analysis_set": "all", "issue_count": int(len(repo_part))}
        rows.append(base)
        for group, part in repo_part.groupby("analysis_set", dropna=False):
            rows.append({"repo_full_name": repo, "analysis_set": group, "issue_count": int(len(part))})
    rows.append({"repo_full_name": "ALL", "analysis_set": "all", "issue_count": int(len(df))})
    for group, part in df.groupby("analysis_set", dropna=False):
        rows.append({"repo_full_name": "ALL", "analysis_set": group, "issue_count": int(len(part))})
    return pd.DataFrame(rows)

def summarize_feature_by_group(df, spec, feature_type):
    feature = spec["feature"]
    denom_name = spec["denominator"]
    mask = denominator_mask(df, denom_name)
    rows = []
    for group, part in df[mask].groupby("analysis_set", dropna=False):
        values = to_numeric(part[feature]) if feature in part.columns else pd.Series(dtype="float64")
        valid = values.dropna()
        row = {
            "feature": feature,
            "label": spec["label"],
            "family": spec["family"],
            "feature_type": feature_type,
            "denominator": denom_name,
            "analysis_set": group,
            "n": int(len(part)),
            "valid_n": int(valid.notna().sum()),
            "interpretation": spec.get("interpretation", ""),
        }
        if feature_type == "binary":
            row["positive_count"] = int((values.fillna(0) > 0).sum())
            row["rate"] = safe_divide(row["positive_count"], row["n"])
            row["mean"] = row["rate"]
            row["median"] = np.nan
            row["std"] = np.nan
        else:
            row["positive_count"] = np.nan
            row["rate"] = np.nan
            row["mean"] = float(valid.mean()) if not valid.empty else np.nan
            row["median"] = float(valid.median()) if not valid.empty else np.nan
            row["std"] = float(valid.std(ddof=1)) if len(valid) > 1 else np.nan
        rows.append(row)
    return rows

def build_feature_summary_by_group(df):
    rows = []
    for spec in PRIMARY_REPO_BINARY_FEATURES + SECONDARY_FILE_BINARY_FEATURES + DESCRIPTIVE_BINARY_FEATURES:
        rows.extend(summarize_feature_by_group(df, spec, "binary"))
    for spec in PRIMARY_REPO_NUMERIC_FEATURES + SECONDARY_FILE_NUMERIC_FEATURES + DESCRIPTIVE_NUMERIC_FEATURES:
        rows.extend(summarize_feature_by_group(df, spec, "numeric"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["family", "feature", "analysis_set"]).reset_index(drop=True)

def binary_effect(df, spec, min_group_n):
    feature = spec["feature"]
    mask = denominator_mask(df, spec["denominator"])
    part = df[mask & df["analysis_set"].isin(["wontfix", "comparison"])].copy()
    if part.empty or feature not in part.columns:
        return {
            "feature": feature,
            "label": spec["label"],
            "family": spec["family"],
            "feature_type": "binary",
            "denominator": spec["denominator"],
            "status": "missing_or_empty",
        }

    w = to_flag(part.loc[part["analysis_set"] == "wontfix", feature])
    c = to_flag(part.loc[part["analysis_set"] == "comparison", feature])
    n_w = int(len(w))
    n_c = int(len(c))
    pos_w = int((w > 0).sum())
    pos_c = int((c > 0).sum())
    rate_w = safe_divide(pos_w, n_w)
    rate_c = safe_divide(pos_c, n_c)
    diff = rate_w - rate_c if pd.notna(rate_w) and pd.notna(rate_c) else np.nan
    or_value = odds_ratio_from_counts(pos_w, n_w - pos_w, pos_c, n_c - pos_c) if n_w > 0 and n_c > 0 else np.nan
    p_value = np.nan
    test_name = "not_run"
    if n_w >= min_group_n and n_c >= min_group_n:
        if stats is not None:
            try:
                _, p_value = stats.fisher_exact([[pos_w, n_w - pos_w], [pos_c, n_c - pos_c]])
                test_name = "fisher_exact"
            except Exception:
                p_value = np.nan
                test_name = "fisher_exact_failed"
        else:
            test_name = "scipy_unavailable"
    else:
        test_name = "insufficient_group_n"
    return {
        "feature": feature,
        "label": spec["label"],
        "family": spec["family"],
        "feature_type": "binary",
        "denominator": spec["denominator"],
        "status": "ok" if n_w >= min_group_n and n_c >= min_group_n else "limited_n",
        "wontfix_n": n_w,
        "comparison_n": n_c,
        "wontfix_positive_count": pos_w,
        "comparison_positive_count": pos_c,
        "wontfix_rate_or_mean": rate_w,
        "comparison_rate_or_mean": rate_c,
        "difference_wontfix_minus_comparison": diff,
        "odds_ratio": or_value,
        "effect_size": diff,
        "effect_size_name": "risk_difference",
        "test_name": test_name,
        "p_value": p_value,
        "interpretation": spec.get("interpretation", ""),
    }

def numeric_effect(df, spec, min_group_n):
    feature = spec["feature"]
    mask = denominator_mask(df, spec["denominator"])
    part = df[mask & df["analysis_set"].isin(["wontfix", "comparison"])].copy()
    if part.empty or feature not in part.columns:
        return {
            "feature": feature,
            "label": spec["label"],
            "family": spec["family"],
            "feature_type": "numeric",
            "denominator": spec["denominator"],
            "status": "missing_or_empty",
        }

    w = to_numeric(part.loc[part["analysis_set"] == "wontfix", feature]).dropna()
    c = to_numeric(part.loc[part["analysis_set"] == "comparison", feature]).dropna()
    n_w = int(len(w))
    n_c = int(len(c))
    mean_w = float(w.mean()) if n_w else np.nan
    mean_c = float(c.mean()) if n_c else np.nan
    diff = mean_w - mean_c if pd.notna(mean_w) and pd.notna(mean_c) else np.nan
    d_value = cohen_d(w, c)
    cliff = cliffs_delta(w, c)
    p_value = np.nan
    test_name = "not_run"
    if n_w >= min_group_n and n_c >= min_group_n:
        if stats is not None:
            try:
                _, p_value = stats.mannwhitneyu(w, c, alternative="two-sided")
                test_name = "mann_whitney_u"
            except Exception:
                p_value = np.nan
                test_name = "mann_whitney_failed"
        else:
            test_name = "scipy_unavailable"
    else:
        test_name = "insufficient_group_n"
    return {
        "feature": feature,
        "label": spec["label"],
        "family": spec["family"],
        "feature_type": "numeric",
        "denominator": spec["denominator"],
        "status": "ok" if n_w >= min_group_n and n_c >= min_group_n else "limited_n",
        "wontfix_n": n_w,
        "comparison_n": n_c,
        "wontfix_rate_or_mean": mean_w,
        "comparison_rate_or_mean": mean_c,
        "difference_wontfix_minus_comparison": diff,
        "odds_ratio": np.nan,
        "effect_size": d_value,
        "effect_size_name": "cohen_d",
        "cliffs_delta": cliff,
        "test_name": test_name,
        "p_value": p_value,
        "interpretation": spec.get("interpretation", ""),
    }

def build_effect_tables(df, min_group_n):
    primary_rows = []
    for spec in PRIMARY_REPO_BINARY_FEATURES:
        primary_rows.append(binary_effect(df, spec, min_group_n))
    for spec in PRIMARY_REPO_NUMERIC_FEATURES:
        primary_rows.append(numeric_effect(df, spec, min_group_n))

    file_rows = []
    for spec in SECONDARY_FILE_BINARY_FEATURES:
        file_rows.append(binary_effect(df, spec, min_group_n))
    for spec in SECONDARY_FILE_NUMERIC_FEATURES:
        file_rows.append(numeric_effect(df, spec, min_group_n))

    descriptive_rows = []
    for spec in DESCRIPTIVE_BINARY_FEATURES:
        descriptive_rows.append(binary_effect(df, spec, min_group_n))
    for spec in DESCRIPTIVE_NUMERIC_FEATURES:
        descriptive_rows.append(numeric_effect(df, spec, min_group_n))

    return pd.DataFrame(primary_rows), pd.DataFrame(file_rows), pd.DataFrame(descriptive_rows)

def build_repo_level_effects(df, feature_specs, min_group_n):
    rows = []
    for repo, repo_part in df.groupby("repo_full_name", dropna=False):
        for spec in feature_specs:
            if spec["feature"] in BINARY_FEATURE_NAMES:
                row = binary_effect(repo_part, spec, min_group_n)
            else:
                row = numeric_effect(repo_part, spec, min_group_n)
            row["repo_full_name"] = repo
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["family", "feature", "repo_full_name"]).reset_index(drop=True)

def build_file_coverage_summary(df):
    rows = []
    if "participant_role_file_coverage_flag" not in df.columns:
        return pd.DataFrame()
    for group_keys, part in df.groupby(["repo_full_name", "analysis_set"], dropna=False):
        repo, analysis_set = group_keys
        total = len(part)
        counts = part["participant_role_file_coverage_flag"].fillna("missing").astype(str).value_counts().to_dict()
        row = {
            "repo_full_name": repo,
            "analysis_set": analysis_set,
            "issue_count": int(total),
            "coverage_ok_count": int(counts.get("ok", 0)),
            "no_file_links_count": int(counts.get("no_file_links", 0)),
            "no_pre_issue_file_history_count": int(counts.get("no_pre_issue_file_history", 0)),
            "missing_issue_timestamp_count": int(counts.get("missing_issue_timestamp", 0)),
            "unknown_count": int(sum(value for key, value in counts.items() if key not in {"ok", "no_file_links", "no_pre_issue_file_history", "missing_issue_timestamp"})),
        }
        row["coverage_ok_rate"] = safe_divide(row["coverage_ok_count"], total)
        row["no_file_links_rate"] = safe_divide(row["no_file_links_count"], total)
        row["no_pre_issue_file_history_rate"] = safe_divide(row["no_pre_issue_file_history_count"], total)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["repo_full_name", "analysis_set"]).reset_index(drop=True)

def build_model_results(df, feature_specs):
    rows = []
    if smf is None:
        for spec in feature_specs:
            rows.append({
                "feature": spec["feature"],
                "label": spec["label"],
                "family": spec["family"],
                "model_type": "not_run",
                "status": "statsmodels_unavailable",
                "n": 0,
                "model_converged": np.nan,
                "model_warning_count": 0,
                "model_warnings": "",
            })
        return pd.DataFrame(rows)

    for spec in feature_specs:
        feature = spec["feature"]
        denom_name = spec["denominator"]
        mask = denominator_mask(df, denom_name)
        model_df = df[mask & df["analysis_set"].isin(["wontfix", "comparison"])].copy()

        if feature not in model_df.columns:
            rows.append({
                "feature": feature,
                "label": spec["label"],
                "family": spec["family"],
                "model_type": "not_run",
                "status": "missing_feature",
                "n": int(len(model_df)),
                "model_converged": np.nan,
                "model_warning_count": 0,
                "model_warnings": "",
            })
            continue

        model_df[feature] = to_numeric(model_df[feature])
        model_df = model_df[[feature, "analysis_set_wontfix", "repo_full_name"]].dropna().copy()

        n = len(model_df)
        n_w = int(model_df["analysis_set_wontfix"].sum())
        n_c = int(n - n_w)

        if n < MIN_MODEL_TOTAL_N or n_w < MIN_MODEL_GROUP_N or n_c < MIN_MODEL_GROUP_N:
            rows.append({
                "feature": feature,
                "label": spec["label"],
                "family": spec["family"],
                "model_type": "not_run",
                "status": "insufficient_n",
                "n": int(n),
                "wontfix_n": int(n_w),
                "comparison_n": int(n_c),
                "model_converged": np.nan,
                "model_warning_count": 0,
                "model_warnings": "",
            })
            continue

        if model_df[feature].nunique(dropna=True) < 2:
            rows.append({
                "feature": feature,
                "label": spec["label"],
                "family": spec["family"],
                "model_type": "not_run",
                "status": "no_outcome_variation",
                "n": int(n),
                "wontfix_n": int(n_w),
                "comparison_n": int(n_c),
                "model_converged": np.nan,
                "model_warning_count": 0,
                "model_warnings": "",
            })
            continue

        model_type = "logit" if feature in BINARY_FEATURE_NAMES else "ols"
        formula = "{0} ~ analysis_set_wontfix + C(repo_full_name)".format(feature)

        try:
            captured_warnings = []

            with warnings.catch_warnings(record=True) as warning_records:
                if ConvergenceWarning is not None:
                    warnings.simplefilter("always", ConvergenceWarning)
                else:
                    warnings.simplefilter("always")

                if model_type == "logit":
                    result = smf.logit(formula=formula, data=model_df).fit(disp=False, maxiter=200)
                else:
                    result = smf.ols(formula=formula, data=model_df).fit()

                captured_warnings = [
                    str(warning_record.message)
                    for warning_record in warning_records
                ]

            model_converged = True
            if model_type == "logit":
                mle_retvals = getattr(result, "mle_retvals", {}) or {}
                if "converged" in mle_retvals:
                    model_converged = bool(mle_retvals.get("converged"))

            coef = result.params.get("analysis_set_wontfix", np.nan)
            p_value = result.pvalues.get("analysis_set_wontfix", np.nan)

            conf_low = np.nan
            conf_high = np.nan
            try:
                ci = result.conf_int().loc["analysis_set_wontfix"]
                conf_low = ci.iloc[0]
                conf_high = ci.iloc[1]
            except Exception:
                pass

            warning_text = " | ".join(captured_warnings)
            warning_count = int(len(captured_warnings))

            if model_type == "logit" and not model_converged:
                status = "nonconverged"
            elif warning_count > 0:
                status = "ok_with_warning"
            else:
                status = "ok"

            rows.append({
                "feature": feature,
                "label": spec["label"],
                "family": spec["family"],
                "model_type": model_type,
                "status": status,
                "n": int(n),
                "wontfix_n": int(n_w),
                "comparison_n": int(n_c),
                "coef_analysis_set_wontfix": float(coef) if pd.notna(coef) else np.nan,
                "p_value": float(p_value) if pd.notna(p_value) else np.nan,
                "conf_low": float(conf_low) if pd.notna(conf_low) else np.nan,
                "conf_high": float(conf_high) if pd.notna(conf_high) else np.nan,
                "odds_ratio_if_logit": float(math.exp(coef)) if model_type == "logit" and pd.notna(coef) else np.nan,
                "model_converged": int(model_converged) if model_type == "logit" else np.nan,
                "model_warning_count": warning_count,
                "model_warnings": warning_text[:1000],
            })

        except Exception as exc:
            rows.append({
                "feature": feature,
                "label": spec["label"],
                "family": spec["family"],
                "model_type": model_type,
                "status": "model_failed",
                "n": int(n),
                "wontfix_n": int(n_w),
                "comparison_n": int(n_c),
                "model_converged": 0 if model_type == "logit" else np.nan,
                "model_warning_count": 0,
                "model_warnings": "",
                "error_message": str(exc)[:500],
            })

    return pd.DataFrame(rows)

def build_available_readiness_notes(readiness_df):
    if readiness_df is None or readiness_df.empty:
        return pd.DataFrame()
    expected = ["feature_family_label", "readiness_status", "wontfix_coverage_rate", "comparison_coverage_rate", "reason"]
    cols = [col for col in expected if col in readiness_df.columns]
    if not cols:
        return pd.DataFrame()
    return readiness_df[cols].copy()

# -----------------------------
# Plots
# -----------------------------

def save_grouped_bar(data, x_col, y_col, group_col, title, ylabel, path, png_dpi):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.empty:
        return None
    x_values = list(data[x_col].dropna().astype(str).unique())
    groups = [group for group in ["wontfix", "comparison"] if group in set(data[group_col].astype(str))]
    if not x_values or not groups:
        return None
    x = np.arange(len(x_values))
    width = 0.36 if len(groups) <= 2 else 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(x_values) * 1.4), 5))
    for i, group in enumerate(groups):
        offsets = x + (i - (len(groups) - 1) / 2.0) * width
        values = []
        for x_value in x_values:
            row = data[(data[x_col].astype(str) == x_value) & (data[group_col].astype(str) == group)]
            values.append(float(row[y_col].iloc[0]) if not row.empty and pd.notna(row[y_col].iloc[0]) else 0.0)
        ax.bar(offsets, values, width=width, label=GROUP_LABELS.get(group, group), color=ANALYSIS_COLORS.get(group, UTK_COLORS["light_gray"]))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_values, rotation=30, ha="right")
    ax.set_ylim(0, max(1.0, data[y_col].dropna().max() * 1.15 if not data[y_col].dropna().empty else 1.0))
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_primary_repo_role_rates(feature_summary, figures_dir, png_dpi):
    selected = feature_summary[
        (feature_summary["family"] == "repo_participant_roles")
        & (feature_summary["feature_type"] == "binary")
        & (feature_summary["feature"].isin([
            "issue_author_is_pre_issue_repo_contributor",
            "any_commenter_is_pre_issue_repo_contributor",
            "top_commenter_is_pre_issue_repo_contributor",
        ]))
    ].copy()
    if selected.empty:
        return None
    selected["short_label"] = selected["feature"].map({
        "issue_author_is_pre_issue_repo_contributor": "Author prior repo contributor",
        "any_commenter_is_pre_issue_repo_contributor": "Any commenter prior repo contributor",
        "top_commenter_is_pre_issue_repo_contributor": "Top commenter prior repo contributor",
    })
    return save_grouped_bar(
        selected,
        "short_label",
        "rate",
        "analysis_set",
        "Repo-level participant-role rates",
        "Rate",
        figures_dir / "01_repo_participant_role_rates.png",
        png_dpi,
    )

def plot_file_role_rates(feature_summary, figures_dir, png_dpi):
    selected = feature_summary[
        (feature_summary["family"] == "file_participant_roles")
        & (feature_summary["feature_type"] == "binary")
        & (feature_summary["feature"].isin([
            "participant_role_file_features_applicable",
            "issue_author_is_pre_issue_file_contributor",
            "any_commenter_is_pre_issue_file_contributor",
            "top_commenter_is_pre_issue_file_contributor",
        ]))
    ].copy()
    if selected.empty:
        return None
    selected["short_label"] = selected["feature"].map({
        "participant_role_file_features_applicable": "File-role applicable",
        "issue_author_is_pre_issue_file_contributor": "Author prior file contributor",
        "any_commenter_is_pre_issue_file_contributor": "Any commenter prior file contributor",
        "top_commenter_is_pre_issue_file_contributor": "Top commenter prior file contributor",
    })
    return save_grouped_bar(
        selected,
        "short_label",
        "rate",
        "analysis_set",
        "File-level participant-role rates",
        "Rate within feature denominator",
        figures_dir / "02_file_participant_role_rates.png",
        png_dpi,
    )

def plot_file_coverage(file_coverage, figures_dir, png_dpi):
    if file_coverage.empty:
        return None
    data = file_coverage[file_coverage["analysis_set"].isin(["wontfix", "comparison"])].copy()
    if data.empty:
        return None
    data["repo_group"] = data["repo_full_name"] + "\n" + data["analysis_set"].map(GROUP_LABELS).fillna(data["analysis_set"])
    categories = [
        ("coverage_ok_rate", "OK"),
        ("no_file_links_rate", "No file links"),
        ("no_pre_issue_file_history_rate", "No pre-issue file history"),
    ]
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(max(10, len(data) * 1.2), 5.5))
    bottom = np.zeros(len(data))
    colors = [UTK_COLORS["orange"], UTK_COLORS["smokey_gray"], UTK_COLORS["light_gray"]]
    for idx, (col, label) in enumerate(categories):
        values = data[col].fillna(0).astype(float).to_numpy()
        ax.bar(x, values, bottom=bottom, label=label, color=colors[idx])
        bottom += values
    ax.set_title("File-level participant-role coverage states")
    ax.set_ylabel("Share of issues")
    ax.set_xticks(x)
    ax.set_xticklabels(data["repo_group"], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = figures_dir / "03_file_role_coverage_states.png"
    fig.savefig(path, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_effect_sizes(effect_df, title, path, png_dpi, top_n=12):
    if effect_df.empty or "difference_wontfix_minus_comparison" not in effect_df.columns:
        return None
    data = effect_df.copy()
    data = data[data["status"].isin(["ok", "limited_n"])].copy()
    data = data[data["difference_wontfix_minus_comparison"].notna()].copy()
    if data.empty:
        return None
    data["abs_diff"] = data["difference_wontfix_minus_comparison"].abs()
    data = data.sort_values("abs_diff", ascending=False).head(top_n).sort_values("difference_wontfix_minus_comparison")
    fig, ax = plt.subplots(figsize=(9, max(4.5, len(data) * 0.45)))
    colors = [UTK_COLORS["orange"] if val >= 0 else UTK_COLORS["smokey_gray"] for val in data["difference_wontfix_minus_comparison"]]
    ax.barh(data["label"], data["difference_wontfix_minus_comparison"], color=colors)
    ax.axvline(0, color=UTK_COLORS["dark_gray"], linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("WONTFIX minus comparison")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=png_dpi, bbox_inches="tight")
    plt.close(fig)
    return path

def build_figures(feature_summary, primary_effects, file_effects, descriptive_effects, file_coverage, figures_dir, png_dpi):
    figures_dir = ensure_dir(figures_dir)
    paths = []
    for path in [
        plot_primary_repo_role_rates(feature_summary, figures_dir, png_dpi),
        plot_file_role_rates(feature_summary, figures_dir, png_dpi),
        plot_file_coverage(file_coverage, figures_dir, png_dpi),
        plot_effect_sizes(primary_effects, "Primary repo-level ownership-adjacent effects", figures_dir / "04_primary_repo_effects.png", png_dpi),
        plot_effect_sizes(file_effects, "Secondary file-level ownership-adjacent effects", figures_dir / "05_secondary_file_effects.png", png_dpi),
        plot_effect_sizes(descriptive_effects, "Descriptive direct/continuity effects", figures_dir / "06_descriptive_effects.png", png_dpi),
    ]:
        if path is not None:
            paths.append(str(path))
    return paths


# -----------------------------
# Markdown report
# -----------------------------

def df_to_markdown(df, max_rows=20):
    if df is None or df.empty:
        return "_No rows available._"
    display = df.head(max_rows).copy()
    return display.to_markdown(index=False)

def top_effects_text(effect_df, family_label, max_rows=4):
    if effect_df.empty:
        return "No effect rows were available."
    data = effect_df[effect_df["difference_wontfix_minus_comparison"].notna()].copy()
    if data.empty:
        return "No comparable WONTFIX-vs-comparison effects were available."
    data["abs_diff"] = data["difference_wontfix_minus_comparison"].abs()
    data = data.sort_values("abs_diff", ascending=False).head(max_rows)
    lines = []
    for _, row in data.iterrows():
        lines.append(
            "- {label}: WONTFIX={w}, comparison={c}, difference={d:.3f} ({denom}).".format(
                label=row.get("label"),
                w="n/a" if pd.isna(row.get("wontfix_rate_or_mean")) else "{0:.3f}".format(row.get("wontfix_rate_or_mean")),
                c="n/a" if pd.isna(row.get("comparison_rate_or_mean")) else "{0:.3f}".format(row.get("comparison_rate_or_mean")),
                d=row.get("difference_wontfix_minus_comparison"),
                denom=row.get("denominator"),
            )
        )
    return "\n".join(lines)

def build_markdown_report(df, qa, population_df, group_counts_df, feature_summary, primary_effects, file_effects, descriptive_effects, repo_effects, file_coverage, model_results, readiness_notes, candidate_df, figure_paths):
    generated = datetime.now(timezone.utc).isoformat()

    repo_signal_rate_w = safe_divide(
        int((df[df["analysis_set"] == "wontfix"].get("has_repo_participant_role_signal", pd.Series(0)).fillna(0) > 0).sum()),
        int(df["analysis_set"].eq("wontfix").sum()),
    )
    repo_signal_rate_c = safe_divide(
        int((df[df["analysis_set"] == "comparison"].get("has_repo_participant_role_signal", pd.Series(0)).fillna(0) > 0).sum()),
        int(df["analysis_set"].eq("comparison").sum()),
    )
    file_app_rate_w = safe_divide(
        int((df[df["analysis_set"] == "wontfix"].get("participant_role_file_features_applicable", pd.Series(0)).fillna(0) > 0).sum()),
        int(df["analysis_set"].eq("wontfix").sum()),
    )
    file_app_rate_c = safe_divide(
        int((df[df["analysis_set"] == "comparison"].get("participant_role_file_features_applicable", pd.Series(0)).fillna(0) > 0).sum()),
        int(df["analysis_set"].eq("comparison").sum()),
    )

    lines = []
    lines.append("# Ownership RQ2 Analysis Report")
    lines.append("")
    lines.append("Generated at: `{0}`".format(generated))
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append("- RQ2 dataset rows analyzed: **{0:,}** across **{1:,} repositories**.".format(len(df), df["repo_full_name"].nunique()))
    lines.append("- WONTFIX issues: **{0:,}**; comparison issues: **{1:,}**.".format(int(df["analysis_set"].eq("wontfix").sum()), int(df["analysis_set"].eq("comparison").sum())))
    lines.append("- Primary ownership analysis should use **repo-level participant roles**. Signal coverage is {0} for WONTFIX and {1} for comparison issues.".format(format_rate(repo_signal_rate_w), format_rate(repo_signal_rate_c)))
    lines.append("- File-level participant roles are secondary/stratified. File applicability is {0} for WONTFIX and {1} for comparison issues.".format(format_rate(file_app_rate_w), format_rate(file_app_rate_c)))
    lines.append("- Direct issue-linked ownership and continuity remain conditional on PR/commit/file-link evidence and should be interpreted descriptively unless explicitly modeled with their denominators.")
    lines.append("")

    lines.append("## Population and denominators")
    lines.append("")
    lines.append(df_to_markdown(group_counts_df))
    lines.append("")

    lines.append("## Readiness context")
    lines.append("")
    if readiness_notes is not None and not readiness_notes.empty:
        lines.append(df_to_markdown(readiness_notes, max_rows=12))
    else:
        lines.append("No readiness table was available. This report still applies the planned primary/secondary/descriptive feature framing.")
    lines.append("")

    lines.append("## Primary analysis: repo-level participant roles")
    lines.append("")
    lines.append("These features ask whether issue authors and commenters already had repository contribution history before the issue was created. This is the recommended primary RQ2 framing.")
    lines.append("")
    primary_display_cols = [
        "label", "denominator", "wontfix_n", "comparison_n", "wontfix_rate_or_mean",
        "comparison_rate_or_mean", "difference_wontfix_minus_comparison", "effect_size_name", "effect_size", "p_value"
    ]
    lines.append(df_to_markdown(primary_effects[[col for col in primary_display_cols if col in primary_effects.columns]], max_rows=20))
    lines.append("")
    lines.append("Largest primary contrasts:")
    lines.append(top_effects_text(primary_effects, "repo-level participant roles"))
    lines.append("")

    lines.append("## Secondary analysis: file-level participant roles")
    lines.append("")
    lines.append("These features are more specific but must use the applicable denominator because many issues do not have usable linked-file context.")
    lines.append("")
    lines.append("### File-role coverage states")
    lines.append(df_to_markdown(file_coverage, max_rows=16))
    lines.append("")
    lines.append("### File-level effects")
    lines.append(df_to_markdown(file_effects[[col for col in primary_display_cols if col in file_effects.columns]], max_rows=20))
    lines.append("")
    lines.append("Largest file-level contrasts:")
    lines.append(top_effects_text(file_effects, "file-level participant roles"))
    lines.append("")

    lines.append("## Descriptive/conditional ownership and continuity")
    lines.append("")
    lines.append("These results are useful for context, but they are conditional on direct ownership evidence and post-issue PR/commit/file-link availability.")
    lines.append("")
    lines.append(df_to_markdown(descriptive_effects[[col for col in primary_display_cols if col in descriptive_effects.columns]], max_rows=20))
    lines.append("")
    lines.append("Largest descriptive contrasts:")
    lines.append(top_effects_text(descriptive_effects, "descriptive ownership/continuity"))
    lines.append("")

    lines.append("## Repo-aware model results")
    lines.append("")
    if smf is None:
        lines.append("Statsmodels was unavailable, so repo-fixed-effect models were skipped.")
    else:
        model_cols = [
            "feature",
            "family",
            "model_type",
            "status",
            "n",
            "coef_analysis_set_wontfix",
            "odds_ratio_if_logit",
            "p_value",
            "model_converged",
            "model_warning_count",
        ]
        lines.append(
            df_to_markdown(model_results[[col for col in model_cols if col in model_results.columns]], max_rows=30))

        flagged_models = model_results[
            model_results.get("status", pd.Series("", index=model_results.index)).isin(
                ["nonconverged", "ok_with_warning", "model_failed"]
            )
        ].copy()

        if not flagged_models.empty:
            lines.append("")
            lines.append(
                "Model diagnostics note: one or more repo-aware models had convergence warnings, other fit warnings, or failed fits. Treat coefficients and p-values for those rows as provisional and prefer the descriptive effect table for interpretation.")
    lines.append("")

    lines.append("## Repo-level effects")
    lines.append("")
    repo_cols = ["repo_full_name", "feature", "label", "denominator", "wontfix_n", "comparison_n", "wontfix_rate_or_mean", "comparison_rate_or_mean", "difference_wontfix_minus_comparison", "status"]
    lines.append(df_to_markdown(repo_effects[[col for col in repo_cols if col in repo_effects.columns]], max_rows=30))
    lines.append("")

    lines.append("## Candidate analysis features")
    lines.append("")
    if candidate_df is not None and not candidate_df.empty:
        lines.append(df_to_markdown(candidate_df, max_rows=20))
    else:
        candidate_rows = []
        for spec in ALL_FEATURE_SPECS:
            candidate_rows.append({
                "feature": spec["feature"],
                "family": spec["family"],
                "denominator": spec["denominator"],
                "interpretation": spec.get("interpretation", ""),
            })
        lines.append(df_to_markdown(pd.DataFrame(candidate_rows), max_rows=20))
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    if figure_paths:
        for path in figure_paths:
            lines.append("- `{0}`".format(path))
    else:
        lines.append("No figures were generated.")
    lines.append("")

    lines.append("## Recommended interpretation")
    lines.append("")
    lines.append("Use repo-level participant-role results as the main ownership-adjacent RQ2 evidence. Use file-level participant-role results as a secondary, applicability-denominated analysis. Keep direct issue-linked ownership and continuity as descriptive/conditional context unless later modeling explicitly handles their evidence-selection denominators.")
    lines.append("")

    lines.append("## Run QA")
    lines.append("")
    qa_rows = [{"metric": key, "value": value} for key, value in qa.items()]
    lines.append(df_to_markdown(pd.DataFrame(qa_rows), max_rows=40))
    lines.append("")
    return "\n".join(lines)

# -----------------------------
# Main runner
# -----------------------------

def run_report(args):
    output_dir = ensure_dir(args.output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    readiness_dir = Path(args.readiness_dir)

    paths = {
        "rq2_dataset": Path(args.rq2_dataset),
        "analysis_qa": Path(args.analysis_qa),
        "feature_family_readiness": readiness_dir / "ownership_feature_family_readiness.csv",
        "candidate_analysis_features": readiness_dir / "ownership_candidate_analysis_features.csv",
    }

    df, analysis_qa_df, readiness_df, candidate_df, qa = load_and_prepare(paths)

    population_df = population_summary(df)
    group_counts_df = group_counts(df)
    feature_summary = build_feature_summary_by_group(df)
    primary_effects, file_effects, descriptive_effects = build_effect_tables(df, min_group_n=args.min_group_n)

    repo_effect_specs = PRIMARY_REPO_BINARY_FEATURES + PRIMARY_REPO_NUMERIC_FEATURES + SECONDARY_FILE_BINARY_FEATURES + SECONDARY_FILE_NUMERIC_FEATURES
    repo_effects = build_repo_level_effects(df, repo_effect_specs, min_group_n=args.min_group_n)
    file_coverage = build_file_coverage_summary(df)

    model_specs = PRIMARY_REPO_BINARY_FEATURES + PRIMARY_REPO_NUMERIC_FEATURES + SECONDARY_FILE_BINARY_FEATURES + SECONDARY_FILE_NUMERIC_FEATURES + DESCRIPTIVE_BINARY_FEATURES + DESCRIPTIVE_NUMERIC_FEATURES
    model_results = build_model_results(df, model_specs)
    qa["model_rows"] = int(len(model_results))
    if not model_results.empty and "status" in model_results.columns:
        qa["model_rows_ok"] = int(model_results["status"].eq("ok").sum())
        qa["model_rows_ok_with_warning"] = int(model_results["status"].eq("ok_with_warning").sum())
        qa["model_rows_nonconverged"] = int(model_results["status"].eq("nonconverged").sum())
        qa["model_rows_failed"] = int(model_results["status"].eq("model_failed").sum())
    else:
        qa["model_rows_ok"] = 0
        qa["model_rows_ok_with_warning"] = 0
        qa["model_rows_nonconverged"] = 0
        qa["model_rows_failed"] = 0
    readiness_notes = build_available_readiness_notes(readiness_df)

    written_paths = {}
    written_paths["population_summary"] = str(write_csv(population_df, output_dir / "ownership_rq2_population_summary.csv"))
    written_paths["group_counts"] = str(write_csv(group_counts_df, output_dir / "ownership_rq2_denominator_summary.csv"))
    written_paths["feature_summary_by_group"] = str(write_csv(feature_summary, output_dir / "ownership_rq2_feature_summary_by_group.csv"))
    written_paths["repo_level_effects"] = str(write_csv(primary_effects, output_dir / "ownership_rq2_repo_level_effects.csv"))
    written_paths["file_level_effects"] = str(write_csv(file_effects, output_dir / "ownership_rq2_file_level_effects.csv"))
    written_paths["descriptive_secondary_effects"] = str(write_csv(descriptive_effects, output_dir / "ownership_rq2_descriptive_secondary_effects.csv"))
    written_paths["repo_specific_effects"] = str(write_csv(repo_effects, output_dir / "ownership_rq2_repo_specific_effects.csv"))
    written_paths["file_coverage_summary"] = str(write_csv(file_coverage, output_dir / "ownership_rq2_file_coverage_summary.csv"))
    written_paths["model_results"] = str(write_csv(model_results, output_dir / "ownership_rq2_model_results.csv"))
    if not readiness_notes.empty:
        written_paths["readiness_notes"] = str(write_csv(readiness_notes, output_dir / "ownership_rq2_readiness_notes.csv"))

    figure_paths = build_figures(feature_summary, primary_effects, file_effects, descriptive_effects, file_coverage, figures_dir, args.png_dpi)

    report_text = build_markdown_report(
        df=df,
        qa=qa,
        population_df=population_df,
        group_counts_df=group_counts_df,
        feature_summary=feature_summary,
        primary_effects=primary_effects,
        file_effects=file_effects,
        descriptive_effects=descriptive_effects,
        repo_effects=repo_effects,
        file_coverage=file_coverage,
        model_results=model_results,
        readiness_notes=readiness_notes,
        candidate_df=candidate_df,
        figure_paths=figure_paths,
    )
    written_paths["markdown_report"] = str(write_text(report_text, output_dir / "ownership_rq2_analysis_report.md"))

    manifest = {
        "script": "17_build_ownership_analysis_report.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "inputs": {key: str(value) for key, value in paths.items()},
        "outputs": written_paths,
        "figures": figure_paths,
        "qa": qa,
        "dependencies": {
            "scipy_available": stats is not None,
            "statsmodels_available": smf is not None,
        },
    }
    written_paths["run_manifest"] = str(write_json(manifest, output_dir / "17_build_ownership_analysis_report_run_manifest.json"))
    return manifest

def main():
    args = parse_args()
    manifest = run_report(args)
    print(
        "Ownership RQ2 analysis report complete | rows={0} | output_dir={1}".format(
            manifest.get("qa", {}).get("rq2_rows_normalized"),
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
