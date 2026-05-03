#!/usr/bin/env python3
"""
Build the ownership QA and analysis-readiness report for the WONTFIX pipeline.

This script is designed to run after scripts/11_b_build_issue_ownership_features.py
has been run in strict and, ideally, fuzzy identity modes. It summarizes the
ownership feature layer, explicit file-role coverage flags, participant-role
coverage, pre/post continuity, leakage guards, and strict-vs-fuzzy sensitivity.

Typical usage:
    python scripts/16_build_ownership_report.py \
        --output-dir outputs/ownership_readiness

Optional explicit inputs:
    python scripts/16_build_ownership_report.py \
        --ownership-features data/features/ownership/issue_ownership_features.parquet \
        --ownership-features-fuzzy data/features/ownership_fuzzy/issue_ownership_features_fuzzy.parquet \
        --ownership-qa logs/qa/issue_ownership_feature_qa_summary.csv \
        --ownership-qa-fuzzy logs/qa/issue_ownership_feature_qa_summary_fuzzy.csv \
        --strict-manifest logs/qa/11_build_issue_ownership_features_run_manifest.json \
        --fuzzy-manifest logs/qa/11_build_issue_ownership_features_fuzzy_run_manifest.json \
        --output-dir outputs/ownership_readiness

The script intentionally avoids type annotations so it fits the style of most
project scripts.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

READINESS_SCORES = {
    "not_ready": 0,
    "descriptive_only": 1,
    "secondary_or_stratified": 2,
    "analysis_ready": 3,
}

READINESS_LABELS = {
    "not_ready": "Not ready",
    "descriptive_only": "Descriptive only",
    "secondary_or_stratified": "Secondary / stratified",
    "analysis_ready": "Analysis-ready",
}

DEFAULT_OWNERSHIP_FEATURES = "data/features/ownership/issue_ownership_features.parquet"
DEFAULT_OWNERSHIP_FEATURES_FUZZY = "data/features/ownership/issue_ownership_features_fuzzy.parquet"
DEFAULT_OWNERSHIP_QA = "logs/qa/issue_ownership_feature_qa_summary.csv"
DEFAULT_OWNERSHIP_QA_FUZZY = "logs/qa/issue_ownership_feature_qa_summary_fuzzy.csv"
DEFAULT_STRICT_MANIFEST = "logs/qa/11_build_issue_ownership_features_run_manifest.json"
DEFAULT_FUZZY_MANIFEST = "logs/qa/11_build_issue_ownership_features_fuzzy_run_manifest.json"
DEFAULT_CONTRIBUTOR_PROFILE_QA = "logs/qa/contributor_ownership_profile_qa_summary.csv"
DEFAULT_CONTRIBUTOR_PROFILE_QA_FUZZY = "logs/qa/contributor_ownership_profile_qa_summary_fuzzy.csv"
DEFAULT_ANALYSIS_QA = "logs/qa/analysis_dataset_qa_summary.csv"
DEFAULT_OUTPUT_DIR = "outputs/ownership_readiness"

LEAKAGE_COLUMNS = [
    "selected_conservative_pre_issue_rows_with_post_issue_bucket",
    "selected_conservative_pre_issue_rows_with_pre_issue_leakage",
    "selected_conservative_pre_issue_rows_with_post_issue_leakage",
    "participant_role_pre_issue_commit_leakage_rows",
    "participant_role_file_pre_issue_commit_leakage_rows",
    "continuity_pre_issue_repo_commit_leakage_rows",
    "continuity_pre_issue_file_commit_leakage_rows",
]

MISSING_TIMESTAMP_COLUMNS = [
    "issues_missing_issue_created_at",
    "participant_role_pre_issue_timestamp_missing_issues",
    "participant_role_file_pre_issue_timestamp_missing_issues",
    "continuity_pre_issue_repo_timestamp_missing_issues",
    "continuity_pre_issue_file_timestamp_missing_issues",
]

FEATURE_FAMILIES = {
    "issue_level_ownership_evidence": {
        "label": "Direct issue-linked ownership evidence",
        "coverage_column": "has_post_issue_ownership",
        "wontfix_threshold_metric": "post_issue_ownership_rate",
        "comparison_threshold_metric": "post_issue_ownership_rate",
        "notes": "Direct PR/commit-linked ownership evidence. Strongest for issues with PR/commit evidence; often sparse for WONTFIX issues.",
    },
    "repo_participant_roles": {
        "label": "Repo-level participant roles",
        "coverage_column": "has_repo_participant_role_signal",
        "wontfix_threshold_metric": "repo_role_signal_rate",
        "comparison_threshold_metric": "repo_role_signal_rate",
        "notes": "Whether authors/commenters had pre-issue repo contribution history. Likely the primary ownership-adjacent RQ2 feature family.",
    },
    "file_participant_roles": {
        "label": "File-level participant roles",
        "coverage_column": "participant_role_file_features_applicable",
        "wontfix_threshold_metric": "file_role_applicable_rate",
        "comparison_threshold_metric": "file_role_applicable_rate",
        "notes": "Whether authors/commenters had prior history on issue-linked files. Should use applicable-denominator rates because file links are sparse.",
    },
    "continuity": {
        "label": "Pre/post continuity",
        "coverage_column": "has_continuity_signal",
        "wontfix_threshold_metric": "continuity_signal_rate",
        "comparison_threshold_metric": "continuity_signal_rate",
        "notes": "Whether post-issue owners had prior repo/file history or appeared in discussion. Useful but conditional on post-issue ownership evidence.",
    },
}

CANDIDATE_FEATURE_ROWS = [
    {
        "feature": "issue_author_is_pre_issue_repo_contributor",
        "family": "repo_participant_roles",
        "recommended_status": "primary_candidate",
        "denominator": "all issues with resolved issue author",
        "interpretation": "Whether the issue author had pre-issue repo commit history.",
        "notes": "Use with repo controls or repo fixed effects in later RQ2 analysis.",
    },
    {
        "feature": "any_commenter_is_pre_issue_repo_contributor",
        "family": "repo_participant_roles",
        "recommended_status": "primary_candidate",
        "denominator": "issues with resolved commenters",
        "interpretation": "Whether the discussion included at least one commenter with pre-issue repo commit history.",
        "notes": "High-coverage ownership-adjacent participation signal.",
    },
    {
        "feature": "share_commenters_pre_issue_repo_contributors",
        "family": "repo_participant_roles",
        "recommended_status": "primary_candidate",
        "denominator": "issues with resolved commenters",
        "interpretation": "Share of resolved commenters with pre-issue repo contribution history.",
        "notes": "Continuous version of commenter repo-contributor participation.",
    },
    {
        "feature": "participant_role_file_features_applicable",
        "family": "file_participant_roles",
        "recommended_status": "coverage_control",
        "denominator": "all issues",
        "interpretation": "Whether file-level participant-role features are interpretable for the issue.",
        "notes": "Use to define applicable denominator or as a coverage control.",
    },
    {
        "feature": "any_commenter_is_pre_issue_file_contributor",
        "family": "file_participant_roles",
        "recommended_status": "secondary_candidate",
        "denominator": "file-applicable issues with resolved commenters",
        "interpretation": "Whether the discussion included someone with prior history on linked files.",
        "notes": "Report rates among applicable issues and all issues separately.",
    },
    {
        "feature": "share_commenters_pre_issue_file_contributors",
        "family": "file_participant_roles",
        "recommended_status": "secondary_candidate",
        "denominator": "file-applicable issues with resolved commenters",
        "interpretation": "Share of resolved commenters with prior linked-file history.",
        "notes": "More specific but more coverage-limited than repo-level participant roles.",
    },
    {
        "feature": "share_post_issue_owners_with_pre_issue_repo_history",
        "family": "continuity",
        "recommended_status": "secondary_candidate",
        "denominator": "issues with post-issue owners",
        "interpretation": "Share of eventual post-issue owners with prior repo history.",
        "notes": "Continuity signal; not a pre-issue ownership definition.",
    },
    {
        "feature": "share_post_issue_owners_with_pre_issue_file_history",
        "family": "continuity",
        "recommended_status": "secondary_candidate",
        "denominator": "issues with post-issue owners and file history",
        "interpretation": "Share of eventual post-issue owners with prior linked-file history.",
        "notes": "Specific continuity signal; coverage depends on file links and post-issue ownership evidence.",
    },
    {
        "feature": "any_commenter_is_eventual_post_issue_owner",
        "family": "continuity",
        "recommended_status": "secondary_candidate",
        "denominator": "issues with post-issue ownership evidence and resolved commenters",
        "interpretation": "Whether a discussion participant later appears as a post-issue owner.",
        "notes": "Useful for owner/participant continuity but conditional on post-issue evidence.",
    },
]


# -----------------------------
# CLI / I/O helpers
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Build ownership QA/readiness report for WONTFIX issue ownership features.")
    parser.add_argument("--ownership-features", default=DEFAULT_OWNERSHIP_FEATURES)
    parser.add_argument("--ownership-features-fuzzy", default=DEFAULT_OWNERSHIP_FEATURES_FUZZY)
    parser.add_argument("--ownership-qa", default=DEFAULT_OWNERSHIP_QA)
    parser.add_argument("--ownership-qa-fuzzy", default=DEFAULT_OWNERSHIP_QA_FUZZY)
    parser.add_argument("--strict-manifest", default=DEFAULT_STRICT_MANIFEST)
    parser.add_argument("--fuzzy-manifest", default=DEFAULT_FUZZY_MANIFEST)
    parser.add_argument("--contributor-profile-qa", default=DEFAULT_CONTRIBUTOR_PROFILE_QA)
    parser.add_argument("--contributor-profile-qa-fuzzy", default=DEFAULT_CONTRIBUTOR_PROFILE_QA_FUZZY)
    parser.add_argument("--analysis-qa", default=DEFAULT_ANALYSIS_QA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--png-dpi", type=int, default=220)
    parser.add_argument("--allow-missing-fuzzy", action="store_true", help="Continue when fuzzy ownership feature input is missing.")
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


def read_json(path, required=False):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError("Input JSON does not exist: {0}".format(path))
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def rate(count, denom):
    return safe_divide(count, denom, default=0.0)


def positive_count(df, column):
    if df is None or df.empty or column not in df.columns:
        return 0
    return int((to_numeric(df[column]).fillna(0) > 0).sum())


def mean_numeric(df, column):
    if df is None or df.empty or column not in df.columns:
        return np.nan
    values = to_numeric(df[column]).dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def median_numeric(df, column):
    if df is None or df.empty or column not in df.columns:
        return np.nan
    values = to_numeric(df[column]).dropna()
    if values.empty:
        return np.nan
    return float(values.median())


def sum_numeric(df, column):
    if df is None or df.empty or column not in df.columns:
        return 0
    return int(to_numeric(df[column]).fillna(0).sum())


def choose_existing(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def safe_columns(df, columns):
    return [column for column in columns if column in df.columns]


# -----------------------------
# Dataset preparation
# -----------------------------


def add_numeric_default(out, column, default=0):
    if column not in out.columns:
        out[column] = default
    out[column] = to_numeric(out[column]).fillna(default)
    return out


def add_share_default(out, column):
    if column not in out.columns:
        out[column] = np.nan
    out[column] = to_numeric(out[column])
    return out


def add_text_default(out, column, default="missing"):
    if column not in out.columns:
        out[column] = default
    out[column] = out[column].apply(lambda value: clean_text(value) or default)
    return out


def normalize_ownership_dataset(df, mode):
    out = df.copy()
    if out.empty:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number", "analysis_set", "identity_resolution_mode"])

    required = ["repo_full_name", "issue_id", "analysis_set"]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise KeyError("Ownership feature dataset is missing required columns: {0}".format(missing))

    out["repo_full_name"] = out["repo_full_name"].astype(str)
    out["issue_id"] = out["issue_id"].astype(str)
    if "issue_number" not in out.columns:
        out["issue_number"] = np.nan
    out["issue_number"] = to_numeric(out["issue_number"])
    out["analysis_set"] = out["analysis_set"].apply(normalize_analysis_set)
    out["identity_resolution_mode"] = mode

    flag_defaults = [
        "ownership_has_pre_issue_ownership",
        "ownership_has_post_issue_ownership",
        "ownership_has_selected_conservative_pre_issue_fallback",
        "issue_author_is_pre_issue_repo_contributor",
        "issue_author_is_pre_issue_major_repo_contributor",
        "any_commenter_is_pre_issue_repo_contributor",
        "any_commenter_is_pre_issue_major_repo_contributor",
        "top_commenter_is_pre_issue_repo_contributor",
        "top_commenter_is_pre_issue_major_repo_contributor",
        "participant_role_file_features_applicable",
        "participant_role_has_file_links",
        "participant_role_has_high_conf_file_links",
        "participant_role_has_pre_issue_file_history",
        "participant_role_file_author_applicable",
        "participant_role_file_commenter_applicable",
        "participant_role_file_top_commenter_applicable",
        "issue_author_is_pre_issue_file_contributor",
        "issue_author_is_pre_issue_major_file_contributor",
        "any_commenter_is_pre_issue_file_contributor",
        "any_commenter_is_pre_issue_major_file_contributor",
        "top_commenter_is_pre_issue_file_contributor",
        "top_commenter_is_pre_issue_major_file_contributor",
        "any_pre_issue_owner_became_post_issue_owner",
        "top_pre_issue_owner_became_post_issue_owner",
        "any_post_issue_owner_with_pre_issue_repo_history",
        "any_post_issue_owner_with_pre_issue_file_history",
        "issue_author_is_post_issue_owner",
        "issue_author_pre_issue_repo_contributor_became_post_issue_owner",
        "issue_author_pre_issue_file_contributor_became_post_issue_owner",
        "any_commenter_is_eventual_post_issue_owner",
        "top_commenter_is_eventual_post_issue_owner",
        "any_pre_issue_repo_contributor_commenter_is_eventual_post_issue_owner",
        "any_pre_issue_file_contributor_commenter_is_eventual_post_issue_owner",
        "participant_role_pre_issue_timestamp_missing",
        "participant_role_file_pre_issue_timestamp_missing",
        "continuity_pre_issue_repo_timestamp_missing",
        "continuity_pre_issue_file_timestamp_missing",
    ]
    count_defaults = [
        "ownership_pre_issue_contributor_count",
        "ownership_post_issue_contributor_count",
        "ownership_pre_issue_high_confidence_contributor_count",
        "ownership_post_issue_high_confidence_contributor_count",
        "commenter_count_with_resolved_key",
        "commenter_count_pre_issue_repo_contributors",
        "commenter_count_pre_issue_major_repo_contributors",
        "top_commenter_comment_count",
        "participant_role_linked_file_count",
        "participant_role_high_conf_linked_file_count",
        "participant_role_pre_issue_file_history_file_count",
        "participant_role_file_commenter_count_with_resolved_key",
        "commenter_count_pre_issue_file_contributors",
        "commenter_count_pre_issue_major_file_contributors",
        "pre_issue_owner_count_for_continuity",
        "post_issue_owner_count_for_continuity",
        "pre_post_owner_overlap_count",
        "post_issue_owners_with_pre_issue_repo_history_count",
        "post_issue_owners_with_pre_issue_file_history_count",
        "commenter_count_eventual_post_issue_owners",
        "participant_role_pre_issue_commit_leakage_rows",
        "participant_role_file_pre_issue_commit_leakage_rows",
        "continuity_pre_issue_repo_commit_leakage_rows",
        "continuity_pre_issue_file_commit_leakage_rows",
    ]
    share_defaults = [
        "share_commenters_pre_issue_repo_contributors",
        "share_commenters_pre_issue_major_repo_contributors",
        "share_commenters_pre_issue_file_contributors",
        "share_commenters_pre_issue_major_file_contributors",
        "pre_post_owner_jaccard",
        "share_post_issue_owners_with_pre_issue_repo_history",
        "share_post_issue_owners_with_pre_issue_file_history",
        "share_commenters_eventual_post_issue_owners",
    ]

    for column in flag_defaults + count_defaults:
        out = add_numeric_default(out, column, default=0)
    for column in share_defaults:
        out = add_share_default(out, column)
    out = add_text_default(out, "participant_role_file_coverage_flag", default="missing")

    # Derived issue-linked ownership flags.
    out["has_post_issue_ownership"] = (
        (to_numeric(out.get("ownership_post_issue_contributor_count")).fillna(0) > 0)
        | (to_numeric(out.get("ownership_has_post_issue_ownership")).fillna(0) > 0)
        | (to_numeric(out.get("post_issue_owner_count_for_continuity")).fillna(0) > 0)
    ).astype(int)

    out["has_pre_issue_issue_linked_ownership"] = (
        (to_numeric(out.get("ownership_pre_issue_contributor_count")).fillna(0) > 0)
        | (to_numeric(out.get("ownership_has_pre_issue_ownership")).fillna(0) > 0)
        | (to_numeric(out.get("pre_issue_owner_count_for_continuity")).fillna(0) > 0)
    ).astype(int)

    out["has_repo_participant_role_signal"] = (
        (to_numeric(out.get("issue_author_is_pre_issue_repo_contributor")).fillna(0) > 0)
        | (to_numeric(out.get("any_commenter_is_pre_issue_repo_contributor")).fillna(0) > 0)
        | (to_numeric(out.get("top_commenter_is_pre_issue_repo_contributor")).fillna(0) > 0)
    ).astype(int)

    out["has_file_participant_role_signal"] = (
        (to_numeric(out.get("participant_role_file_features_applicable")).fillna(0) > 0)
        & (
            (to_numeric(out.get("issue_author_is_pre_issue_file_contributor")).fillna(0) > 0)
            | (to_numeric(out.get("any_commenter_is_pre_issue_file_contributor")).fillna(0) > 0)
            | (to_numeric(out.get("top_commenter_is_pre_issue_file_contributor")).fillna(0) > 0)
        )
    ).astype(int)

    out["has_continuity_signal"] = (
        (to_numeric(out.get("any_post_issue_owner_with_pre_issue_repo_history")).fillna(0) > 0)
        | (to_numeric(out.get("any_post_issue_owner_with_pre_issue_file_history")).fillna(0) > 0)
        | (to_numeric(out.get("any_commenter_is_eventual_post_issue_owner")).fillna(0) > 0)
    ).astype(int)

    out["duplicate_issue_key"] = out.duplicated(
        subset=["repo_full_name", "issue_id", "issue_number", "identity_resolution_mode"],
        keep=False,
    ).astype(int)

    out = out.drop_duplicates(
        subset=["repo_full_name", "issue_id", "issue_number", "identity_resolution_mode"],
        keep="first",
    ).reset_index(drop=True)

    return out


# -----------------------------
# Summary builders
# -----------------------------


def build_population_summary(all_df, duplicate_counts):
    rows = []
    for mode, part in all_df.groupby("identity_resolution_mode", dropna=False):
        rows.append({
            "identity_resolution_mode": mode,
            "issue_rows": int(len(part)),
            "unique_repos": int(part["repo_full_name"].nunique()) if "repo_full_name" in part.columns else 0,
            "wontfix_issues": int((part["analysis_set"] == "wontfix").sum()),
            "comparison_issues": int((part["analysis_set"] == "comparison").sum()),
            "missing_analysis_set_rows": int((part["analysis_set"] == "missing").sum()),
            "duplicate_issue_keys_before_dedupe": int(duplicate_counts.get(mode, 0)),
        })
    return pd.DataFrame(rows).sort_values(["identity_resolution_mode"]).reset_index(drop=True)


def count_rate_row(part, base_payload):
    issue_count = int(len(part))
    row = dict(base_payload)
    row["issue_count"] = issue_count

    count_specs = {
        "pre_issue_ownership": "has_pre_issue_issue_linked_ownership",
        "post_issue_ownership": "has_post_issue_ownership",
        "both_pre_post_ownership": None,
        "repo_role_author_pre_issue": "issue_author_is_pre_issue_repo_contributor",
        "repo_role_any_commenter_pre_issue": "any_commenter_is_pre_issue_repo_contributor",
        "repo_role_top_commenter_pre_issue": "top_commenter_is_pre_issue_repo_contributor",
        "repo_role_signal": "has_repo_participant_role_signal",
        "file_role_applicable": "participant_role_file_features_applicable",
        "file_role_signal": "has_file_participant_role_signal",
        "file_role_author_pre_issue": "issue_author_is_pre_issue_file_contributor",
        "file_role_any_commenter_pre_issue": "any_commenter_is_pre_issue_file_contributor",
        "file_role_top_commenter_pre_issue": "top_commenter_is_pre_issue_file_contributor",
        "post_owner_pre_repo_history": "any_post_issue_owner_with_pre_issue_repo_history",
        "post_owner_pre_file_history": "any_post_issue_owner_with_pre_issue_file_history",
        "any_commenter_eventual_post_owner": "any_commenter_is_eventual_post_issue_owner",
        "top_commenter_eventual_post_owner": "top_commenter_is_eventual_post_issue_owner",
        "continuity_signal": "has_continuity_signal",
    }

    if issue_count == 0:
        for metric in count_specs:
            row[metric + "_count"] = 0
            row[metric + "_rate"] = 0.0
        return row

    both_mask = (to_numeric(part["has_pre_issue_issue_linked_ownership"]).fillna(0) > 0) & (
        to_numeric(part["has_post_issue_ownership"]).fillna(0) > 0
    )
    for metric, column in count_specs.items():
        if metric == "both_pre_post_ownership":
            count_value = int(both_mask.sum())
        else:
            count_value = positive_count(part, column)
        row[metric + "_count"] = count_value
        row[metric + "_rate"] = rate(count_value, issue_count)

    row["file_coverage_ok_count"] = int((part["participant_role_file_coverage_flag"] == "ok").sum())
    row["file_coverage_ok_rate"] = rate(row["file_coverage_ok_count"], issue_count)
    row["file_coverage_no_file_links_count"] = int((part["participant_role_file_coverage_flag"] == "no_file_links").sum())
    row["file_coverage_no_file_links_rate"] = rate(row["file_coverage_no_file_links_count"], issue_count)
    row["file_coverage_no_pre_issue_history_count"] = int((part["participant_role_file_coverage_flag"] == "no_pre_issue_file_history").sum())
    row["file_coverage_no_pre_issue_history_rate"] = rate(row["file_coverage_no_pre_issue_history_count"], issue_count)
    return row


def build_coverage_by_repo_and_group(all_df):
    rows = []
    group_cols = ["identity_resolution_mode", "repo_full_name", "analysis_set"]
    for keys, part in all_df.groupby(group_cols, dropna=False):
        mode, repo_name, analysis_set = keys
        rows.append(count_rate_row(part, {
            "identity_resolution_mode": mode,
            "repo_full_name": repo_name,
            "analysis_set": analysis_set,
        }))
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_issue_level_evidence_summary(all_df):
    rows = []
    for keys, part in all_df.groupby(["identity_resolution_mode", "analysis_set"], dropna=False):
        mode, analysis_set = keys
        issue_count = int(len(part))
        both_count = int(((part["has_pre_issue_issue_linked_ownership"] > 0) & (part["has_post_issue_ownership"] > 0)).sum())
        rows.append({
            "identity_resolution_mode": mode,
            "analysis_set": analysis_set,
            "issue_count": issue_count,
            "pre_issue_ownership_count": positive_count(part, "has_pre_issue_issue_linked_ownership"),
            "pre_issue_ownership_rate": rate(positive_count(part, "has_pre_issue_issue_linked_ownership"), issue_count),
            "post_issue_ownership_count": positive_count(part, "has_post_issue_ownership"),
            "post_issue_ownership_rate": rate(positive_count(part, "has_post_issue_ownership"), issue_count),
            "both_pre_post_ownership_count": both_count,
            "both_pre_post_ownership_rate": rate(both_count, issue_count),
            "mean_pre_issue_contributor_count": mean_numeric(part, "ownership_pre_issue_contributor_count"),
            "mean_post_issue_contributor_count": mean_numeric(part, "ownership_post_issue_contributor_count"),
        })
    return pd.DataFrame(rows).sort_values(["identity_resolution_mode", "analysis_set"]).reset_index(drop=True)


def build_repo_participant_role_summary(all_df):
    rows = []
    for keys, part in all_df.groupby(["identity_resolution_mode", "analysis_set"], dropna=False):
        mode, analysis_set = keys
        issue_count = int(len(part))
        resolved_commenter_count = positive_count(part, "commenter_count_with_resolved_key")
        rows.append({
            "identity_resolution_mode": mode,
            "analysis_set": analysis_set,
            "issue_count": issue_count,
            "issue_author_resolved_count": issue_count,
            "issue_author_resolved_rate": rate(issue_count, issue_count),
            "author_pre_issue_repo_contributor_count": positive_count(part, "issue_author_is_pre_issue_repo_contributor"),
            "author_pre_issue_repo_contributor_rate": rate(positive_count(part, "issue_author_is_pre_issue_repo_contributor"), issue_count),
            "author_pre_issue_major_repo_contributor_count": positive_count(part, "issue_author_is_pre_issue_major_repo_contributor"),
            "author_pre_issue_major_repo_contributor_rate": rate(positive_count(part, "issue_author_is_pre_issue_major_repo_contributor"), issue_count),
            "issues_with_resolved_commenters_count": resolved_commenter_count,
            "issues_with_resolved_commenters_rate": rate(resolved_commenter_count, issue_count),
            "any_commenter_pre_issue_repo_contributor_count": positive_count(part, "any_commenter_is_pre_issue_repo_contributor"),
            "any_commenter_pre_issue_repo_contributor_rate": rate(positive_count(part, "any_commenter_is_pre_issue_repo_contributor"), issue_count),
            "any_commenter_pre_issue_major_repo_contributor_count": positive_count(part, "any_commenter_is_pre_issue_major_repo_contributor"),
            "any_commenter_pre_issue_major_repo_contributor_rate": rate(positive_count(part, "any_commenter_is_pre_issue_major_repo_contributor"), issue_count),
            "top_commenter_pre_issue_repo_contributor_count": positive_count(part, "top_commenter_is_pre_issue_repo_contributor"),
            "top_commenter_pre_issue_repo_contributor_rate": rate(positive_count(part, "top_commenter_is_pre_issue_repo_contributor"), issue_count),
            "top_commenter_pre_issue_major_repo_contributor_count": positive_count(part, "top_commenter_is_pre_issue_major_repo_contributor"),
            "top_commenter_pre_issue_major_repo_contributor_rate": rate(positive_count(part, "top_commenter_is_pre_issue_major_repo_contributor"), issue_count),
            "mean_share_commenters_pre_issue_repo_contributors": mean_numeric(part, "share_commenters_pre_issue_repo_contributors"),
            "mean_share_commenters_pre_issue_major_repo_contributors": mean_numeric(part, "share_commenters_pre_issue_major_repo_contributors"),
        })
    return pd.DataFrame(rows).sort_values(["identity_resolution_mode", "analysis_set"]).reset_index(drop=True)


def build_file_participant_role_summary(all_df):
    rows = []
    for keys, part in all_df.groupby(["identity_resolution_mode", "analysis_set"], dropna=False):
        mode, analysis_set = keys
        issue_count = int(len(part))
        applicable_count = positive_count(part, "participant_role_file_features_applicable")
        commenter_applicable_count = positive_count(part, "participant_role_file_commenter_applicable")
        author_file_count = positive_count(part, "issue_author_is_pre_issue_file_contributor")
        commenter_file_count = positive_count(part, "any_commenter_is_pre_issue_file_contributor")
        top_commenter_file_count = positive_count(part, "top_commenter_is_pre_issue_file_contributor")
        rows.append({
            "identity_resolution_mode": mode,
            "analysis_set": analysis_set,
            "issue_count": issue_count,
            "file_features_applicable_count": applicable_count,
            "file_features_applicable_rate": rate(applicable_count, issue_count),
            "file_coverage_ok_count": int((part["participant_role_file_coverage_flag"] == "ok").sum()),
            "file_coverage_ok_rate": rate(int((part["participant_role_file_coverage_flag"] == "ok").sum()), issue_count),
            "file_coverage_no_file_links_count": int((part["participant_role_file_coverage_flag"] == "no_file_links").sum()),
            "file_coverage_no_file_links_rate": rate(int((part["participant_role_file_coverage_flag"] == "no_file_links").sum()), issue_count),
            "file_coverage_no_pre_issue_history_count": int((part["participant_role_file_coverage_flag"] == "no_pre_issue_file_history").sum()),
            "file_coverage_no_pre_issue_history_rate": rate(int((part["participant_role_file_coverage_flag"] == "no_pre_issue_file_history").sum()), issue_count),
            "author_file_applicable_count": positive_count(part, "participant_role_file_author_applicable"),
            "commenter_file_applicable_count": commenter_applicable_count,
            "top_commenter_file_applicable_count": positive_count(part, "participant_role_file_top_commenter_applicable"),
            "author_pre_issue_file_contributor_count": author_file_count,
            "author_pre_issue_file_contributor_rate_all_issues": rate(author_file_count, issue_count),
            "author_pre_issue_file_contributor_rate_applicable": rate(author_file_count, applicable_count),
            "any_commenter_pre_issue_file_contributor_count": commenter_file_count,
            "any_commenter_pre_issue_file_contributor_rate_all_issues": rate(commenter_file_count, issue_count),
            "any_commenter_pre_issue_file_contributor_rate_applicable": rate(commenter_file_count, commenter_applicable_count),
            "top_commenter_pre_issue_file_contributor_count": top_commenter_file_count,
            "top_commenter_pre_issue_file_contributor_rate_all_issues": rate(top_commenter_file_count, issue_count),
            "top_commenter_pre_issue_file_contributor_rate_applicable": rate(top_commenter_file_count, positive_count(part, "participant_role_file_top_commenter_applicable")),
            "mean_share_commenters_pre_issue_file_contributors": mean_numeric(part, "share_commenters_pre_issue_file_contributors"),
            "mean_participant_role_linked_file_count_when_applicable": mean_numeric(part[part["participant_role_file_features_applicable"] > 0], "participant_role_linked_file_count") if applicable_count else np.nan,
            "mean_participant_role_pre_issue_file_history_file_count": mean_numeric(part, "participant_role_pre_issue_file_history_file_count"),
        })
    return pd.DataFrame(rows).sort_values(["identity_resolution_mode", "analysis_set"]).reset_index(drop=True)


def build_pre_post_continuity_summary(all_df):
    rows = []
    for keys, part in all_df.groupby(["identity_resolution_mode", "analysis_set"], dropna=False):
        mode, analysis_set = keys
        issue_count = int(len(part))
        post_owner_count = positive_count(part, "post_issue_owner_count_for_continuity")
        rows.append({
            "identity_resolution_mode": mode,
            "analysis_set": analysis_set,
            "issue_count": issue_count,
            "post_issue_owner_issue_count": post_owner_count,
            "post_issue_owner_issue_rate": rate(post_owner_count, issue_count),
            "pre_post_owner_overlap_count": positive_count(part, "pre_post_owner_overlap_count"),
            "pre_post_owner_overlap_rate": rate(positive_count(part, "pre_post_owner_overlap_count"), issue_count),
            "post_owners_with_pre_issue_repo_history_count": positive_count(part, "any_post_issue_owner_with_pre_issue_repo_history"),
            "post_owners_with_pre_issue_repo_history_rate": rate(positive_count(part, "any_post_issue_owner_with_pre_issue_repo_history"), issue_count),
            "post_owners_with_pre_issue_file_history_count": positive_count(part, "any_post_issue_owner_with_pre_issue_file_history"),
            "post_owners_with_pre_issue_file_history_rate": rate(positive_count(part, "any_post_issue_owner_with_pre_issue_file_history"), issue_count),
            "issue_author_as_post_issue_owner_count": positive_count(part, "issue_author_is_post_issue_owner"),
            "issue_author_as_post_issue_owner_rate": rate(positive_count(part, "issue_author_is_post_issue_owner"), issue_count),
            "any_commenter_as_post_issue_owner_count": positive_count(part, "any_commenter_is_eventual_post_issue_owner"),
            "any_commenter_as_post_issue_owner_rate": rate(positive_count(part, "any_commenter_is_eventual_post_issue_owner"), issue_count),
            "top_commenter_as_post_issue_owner_count": positive_count(part, "top_commenter_is_eventual_post_issue_owner"),
            "top_commenter_as_post_issue_owner_rate": rate(positive_count(part, "top_commenter_is_eventual_post_issue_owner"), issue_count),
            "mean_pre_post_owner_jaccard": mean_numeric(part, "pre_post_owner_jaccard"),
            "mean_share_post_issue_owners_with_pre_issue_repo_history": mean_numeric(part, "share_post_issue_owners_with_pre_issue_repo_history"),
            "mean_share_post_issue_owners_with_pre_issue_file_history": mean_numeric(part, "share_post_issue_owners_with_pre_issue_file_history"),
            "mean_share_commenters_eventual_post_issue_owners": mean_numeric(part, "share_commenters_eventual_post_issue_owners"),
        })
    return pd.DataFrame(rows).sort_values(["identity_resolution_mode", "analysis_set"]).reset_index(drop=True)


def build_qa_summary(qa_frames, manifests):
    rows = []
    for mode, qa_df in qa_frames.items():
        if qa_df is not None and not qa_df.empty:
            for row in qa_df.to_dict(orient="records"):
                payload = dict(row)
                payload["identity_resolution_mode"] = mode
                payload["source"] = "qa_summary_csv"
                rows.append(payload)
        manifest = manifests.get(mode) or {}
        for row in manifest.get("summary_rows", []):
            payload = dict(row)
            payload["identity_resolution_mode"] = mode
            payload["source"] = "run_manifest"
            rows.append(payload)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_leakage_guard_summary(all_df, qa_summary_df, manifests):
    rows = []
    for mode in sorted(all_df["identity_resolution_mode"].dropna().unique().tolist()):
        part = all_df[all_df["identity_resolution_mode"] == mode]
        for column in LEAKAGE_COLUMNS:
            if column in part.columns:
                value = sum_numeric(part, column)
                source = "issue_features"
                status = "pass" if value == 0 else "warn"
            else:
                value = np.nan
                source = "issue_features"
                status = "missing"
            rows.append({
                "identity_resolution_mode": mode,
                "check_name": column,
                "source": source,
                "value": value,
                "status": status,
                "interpretation": "Expected zero. Nonzero values indicate possible pre/post leakage or invalid row selection.",
            })
        for column in MISSING_TIMESTAMP_COLUMNS:
            value = np.nan
            status = "missing"
            if column in part.columns:
                value = sum_numeric(part, column)
                status = "pass" if value == 0 else "warn"
            elif qa_summary_df is not None and not qa_summary_df.empty and column in qa_summary_df.columns:
                subset = qa_summary_df[qa_summary_df["identity_resolution_mode"] == mode]
                value = int(to_numeric(subset[column]).fillna(0).sum()) if not subset.empty else np.nan
                status = "pass" if pd.notna(value) and value == 0 else "warn"
            rows.append({
                "identity_resolution_mode": mode,
                "check_name": column,
                "source": "issue_features_or_manifest",
                "value": value,
                "status": status,
                "interpretation": "Expected zero or explainably small. Missing issue timestamps limit time-aware pre-issue features.",
            })
    return pd.DataFrame(rows)


def summarize_family_coverage(all_df, family_key):
    spec = FEATURE_FAMILIES[family_key]
    coverage_column = spec["coverage_column"]
    rows = []
    for mode, mode_df in all_df.groupby("identity_resolution_mode", dropna=False):
        leakage_pass = True
        leakage_cols = [col for col in LEAKAGE_COLUMNS if col in mode_df.columns]
        if leakage_cols:
            leakage_total = int(mode_df[leakage_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum().sum())
            leakage_pass = leakage_total == 0
        for analysis_set, part in mode_df.groupby("analysis_set", dropna=False):
            issue_count = int(len(part))
            covered_count = positive_count(part, coverage_column)
            repos_with_coverage = 0
            if issue_count > 0:
                repo_counts = part.groupby("repo_full_name")[coverage_column].apply(lambda s: int((to_numeric(s).fillna(0) > 0).sum()))
                repos_with_coverage = int((repo_counts > 0).sum())
            rows.append({
                "identity_resolution_mode": mode,
                "feature_family": family_key,
                "feature_family_label": spec["label"],
                "analysis_set": analysis_set,
                "issue_count": issue_count,
                "covered_count": covered_count,
                "coverage_rate": rate(covered_count, issue_count),
                "repos_with_coverage": repos_with_coverage,
                "leakage_pass": leakage_pass,
            })
    return pd.DataFrame(rows)


def classify_readiness_for_mode(family_coverage_df, family_key, mode):
    part = family_coverage_df[(family_coverage_df["feature_family"] == family_key) & (family_coverage_df["identity_resolution_mode"] == mode)]
    if part.empty:
        return {
            "readiness_status": "not_ready",
            "readiness_score": 0,
            "reason": "No coverage rows available for this feature family.",
        }

    leakage_pass = bool(part["leakage_pass"].all()) if "leakage_pass" in part.columns else False
    wontfix = part[part["analysis_set"] == "wontfix"]
    comparison = part[part["analysis_set"] == "comparison"]
    if wontfix.empty or comparison.empty:
        return {
            "readiness_status": "not_ready",
            "readiness_score": 0,
            "reason": "Missing WONTFIX or comparison group for readiness classification.",
        }

    wf_rate = float(wontfix["coverage_rate"].iloc[0])
    cmp_rate = float(comparison["coverage_rate"].iloc[0])
    wf_repos = int(wontfix["repos_with_coverage"].iloc[0])
    cmp_repos = int(comparison["repos_with_coverage"].iloc[0])

    if not leakage_pass:
        status = "not_ready"
        reason = "Leakage guard did not pass."
    elif wf_rate >= 0.20 and cmp_rate >= 0.20 and wf_repos >= 2 and cmp_repos >= 2:
        status = "analysis_ready"
        reason = "Coverage is adequate in both groups across multiple repositories and leakage checks pass."
    elif wf_rate >= 0.05 and cmp_rate >= 0.05:
        status = "secondary_or_stratified"
        reason = "Coverage exists in both groups but is limited or uneven; use as secondary/stratified analysis."
    elif wf_rate > 0 or cmp_rate > 0:
        status = "descriptive_only"
        reason = "Coverage exists but is too sparse or imbalanced for primary WONTFIX-vs-comparison claims."
    else:
        status = "not_ready"
        reason = "No usable coverage in one or both analysis groups."

    return {
        "readiness_status": status,
        "readiness_score": READINESS_SCORES[status],
        "reason": reason,
        "wontfix_coverage_rate": wf_rate,
        "comparison_coverage_rate": cmp_rate,
        "wontfix_repos_with_coverage": wf_repos,
        "comparison_repos_with_coverage": cmp_repos,
        "leakage_pass": leakage_pass,
    }


def build_feature_family_readiness(all_df):
    coverage_frames = []
    for family_key in FEATURE_FAMILIES:
        coverage_frames.append(summarize_family_coverage(all_df, family_key))
    family_coverage_df = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame()

    rows = []
    modes = sorted(all_df["identity_resolution_mode"].dropna().unique().tolist())
    for mode in modes:
        for family_key, spec in FEATURE_FAMILIES.items():
            result = classify_readiness_for_mode(family_coverage_df, family_key, mode)
            row = {
                "identity_resolution_mode": mode,
                "feature_family": family_key,
                "feature_family_label": spec["label"],
                "notes": spec["notes"],
            }
            row.update(result)
            rows.append(row)
    readiness_df = pd.DataFrame(rows).sort_values(["identity_resolution_mode", "readiness_score", "feature_family"], ascending=[True, False, True]).reset_index(drop=True)
    return readiness_df, family_coverage_df


def build_strict_vs_fuzzy_comparison(coverage_by_repo_group_df):
    if coverage_by_repo_group_df.empty:
        return pd.DataFrame()

    metrics = [
        "issue_count",
        "repo_role_any_commenter_pre_issue_count",
        "file_role_applicable_count",
        "file_role_any_commenter_pre_issue_count",
        "post_owner_pre_repo_history_count",
        "post_owner_pre_file_history_count",
        "any_commenter_eventual_post_owner_count",
        "file_coverage_ok_count",
        "file_coverage_no_file_links_count",
        "continuity_signal_count",
    ]
    available_metrics = [metric for metric in metrics if metric in coverage_by_repo_group_df.columns]
    long_rows = []
    for row in coverage_by_repo_group_df.to_dict(orient="records"):
        for metric in available_metrics:
            long_rows.append({
                "identity_resolution_mode": row.get("identity_resolution_mode"),
                "repo_full_name": row.get("repo_full_name"),
                "analysis_set": row.get("analysis_set"),
                "metric_name": metric,
                "value": row.get(metric),
            })
    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        return pd.DataFrame()
    strict = long_df[long_df["identity_resolution_mode"] == "strict"].rename(columns={"value": "strict_value"})
    fuzzy = long_df[long_df["identity_resolution_mode"] == "fuzzy"].rename(columns={"value": "fuzzy_value"})
    if strict.empty or fuzzy.empty:
        return pd.DataFrame()
    merged = strict[["repo_full_name", "analysis_set", "metric_name", "strict_value"]].merge(
        fuzzy[["repo_full_name", "analysis_set", "metric_name", "fuzzy_value"]],
        on=["repo_full_name", "analysis_set", "metric_name"],
        how="outer",
    )
    merged["strict_value"] = to_numeric(merged["strict_value"])
    merged["fuzzy_value"] = to_numeric(merged["fuzzy_value"])
    merged["delta_fuzzy_minus_strict"] = merged["fuzzy_value"] - merged["strict_value"]
    merged["relative_delta"] = merged.apply(
        lambda row: safe_divide(row["delta_fuzzy_minus_strict"], row["strict_value"], default=np.nan),
        axis=1,
    )
    return merged.sort_values(["metric_name", "repo_full_name", "analysis_set"]).reset_index(drop=True)


def build_candidate_analysis_features(readiness_df):
    rows = []
    status_by_family = {}
    if readiness_df is not None and not readiness_df.empty:
        strict_rows = readiness_df[readiness_df["identity_resolution_mode"] == "strict"]
        for row in strict_rows.to_dict(orient="records"):
            status_by_family[row.get("feature_family")] = row.get("readiness_status")
    for row in CANDIDATE_FEATURE_ROWS:
        payload = dict(row)
        family_status = status_by_family.get(row["family"], "unknown")
        payload["family_readiness_status_strict"] = family_status
        rows.append(payload)
    return pd.DataFrame(rows)


# -----------------------------
# Figures
# -----------------------------


def setup_axis(ax, title=None, ylabel=None, xlabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def save_figure(fig, output_path, dpi):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ownership_coverage_by_repo(coverage_df, output_path, dpi):
    strict = coverage_df[coverage_df["identity_resolution_mode"] == "strict"].copy()
    if strict.empty:
        return None
    summary = strict.groupby("repo_full_name", dropna=False).agg(
        pre_issue=("pre_issue_ownership_rate", "mean"),
        post_issue=("post_issue_ownership_rate", "mean"),
        repo_role=("repo_role_signal_rate", "mean"),
        file_applicable=("file_role_applicable_rate", "mean"),
    ).reset_index()
    x = np.arange(len(summary))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * width, summary["pre_issue"], width, label="Issue-linked pre")
    ax.bar(x - 0.5 * width, summary["post_issue"], width, label="Issue-linked post")
    ax.bar(x + 0.5 * width, summary["repo_role"], width, label="Repo participant role")
    ax.bar(x + 1.5 * width, summary["file_applicable"], width, label="File role applicable")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["repo_full_name"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    setup_axis(ax, "Ownership coverage by repository (strict)", "Mean coverage rate", None)
    ax.legend(fontsize=8)
    return save_figure(fig, output_path, dpi)


def plot_repo_participant_role_coverage(repo_summary, output_path, dpi):
    strict = repo_summary[repo_summary["identity_resolution_mode"] == "strict"].copy()
    if strict.empty:
        return None
    metrics = [
        ("author_pre_issue_repo_contributor_rate", "Author"),
        ("any_commenter_pre_issue_repo_contributor_rate", "Any commenter"),
        ("top_commenter_pre_issue_repo_contributor_rate", "Top commenter"),
    ]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, analysis_set in [(-width / 2, "comparison"), (width / 2, "wontfix")]:
        part = strict[strict["analysis_set"] == analysis_set]
        values = []
        for col, _ in metrics:
            values.append(float(part[col].iloc[0]) if not part.empty and col in part.columns else 0.0)
        ax.bar(x + offset, values, width, label=analysis_set.title(), color=ANALYSIS_COLORS.get(analysis_set))
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1)
    setup_axis(ax, "Repo-level participant-role coverage (strict)", "Rate among issues", None)
    ax.legend()
    return save_figure(fig, output_path, dpi)


def plot_file_participant_role_coverage(all_df, output_path, dpi):
    strict = all_df[all_df["identity_resolution_mode"] == "strict"].copy()
    if strict.empty:
        return None
    grouped = strict.groupby(["repo_full_name", "analysis_set", "participant_role_file_coverage_flag"], dropna=False).size().reset_index(name="count")
    totals = strict.groupby(["repo_full_name", "analysis_set"], dropna=False).size().reset_index(name="total")
    grouped = grouped.merge(totals, on=["repo_full_name", "analysis_set"], how="left")
    grouped["rate"] = grouped.apply(lambda row: safe_divide(row["count"], row["total"], default=0.0), axis=1)
    flags = ["ok", "no_file_links", "no_pre_issue_file_history", "no_high_conf_file_links", "missing"]
    labels = sorted(grouped[["repo_full_name", "analysis_set"]].drop_duplicates().apply(lambda r: "{0}\n{1}".format(r["repo_full_name"], r["analysis_set"]), axis=1).tolist())
    key_df = grouped[["repo_full_name", "analysis_set"]].drop_duplicates().copy()
    key_df["label"] = key_df.apply(lambda r: "{0}\n{1}".format(r["repo_full_name"], r["analysis_set"]), axis=1)
    key_df = key_df.sort_values("label").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(key_df))
    color_map = {
        "ok": UTK_COLORS["orange"],
        "no_file_links": UTK_COLORS["light_gray"],
        "no_pre_issue_file_history": "#BFC1C4",
        "no_high_conf_file_links": "#8D8F92",
        "missing": "#D9D9D9",
    }
    for flag in flags:
        values = []
        for row in key_df.to_dict(orient="records"):
            subset = grouped[
                (grouped["repo_full_name"] == row["repo_full_name"])
                & (grouped["analysis_set"] == row["analysis_set"])
                & (grouped["participant_role_file_coverage_flag"] == flag)
            ]
            values.append(float(subset["rate"].iloc[0]) if not subset.empty else 0.0)
        ax.bar(np.arange(len(key_df)), values, bottom=bottom, label=flag, color=color_map.get(flag))
        bottom += np.array(values)
    ax.set_xticks(np.arange(len(key_df)))
    ax.set_xticklabels(key_df["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    setup_axis(ax, "File-level participant-role applicability (strict)", "Share of issues", None)
    ax.legend(fontsize=8, ncol=2)
    return save_figure(fig, output_path, dpi)


def plot_pre_post_continuity_coverage(continuity_summary, output_path, dpi):
    strict = continuity_summary[continuity_summary["identity_resolution_mode"] == "strict"].copy()
    if strict.empty:
        return None
    metrics = [
        ("post_owners_with_pre_issue_repo_history_rate", "Post owners\npre-repo history"),
        ("post_owners_with_pre_issue_file_history_rate", "Post owners\npre-file history"),
        ("any_commenter_as_post_issue_owner_rate", "Commenter\neventual owner"),
    ]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, analysis_set in [(-width / 2, "comparison"), (width / 2, "wontfix")]:
        part = strict[strict["analysis_set"] == analysis_set]
        values = []
        for col, _ in metrics:
            values.append(float(part[col].iloc[0]) if not part.empty and col in part.columns else 0.0)
        ax.bar(x + offset, values, width, label=analysis_set.title(), color=ANALYSIS_COLORS.get(analysis_set))
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1)
    setup_axis(ax, "Pre/post continuity coverage (strict)", "Rate among issues", None)
    ax.legend()
    return save_figure(fig, output_path, dpi)


def plot_strict_vs_fuzzy_delta(delta_df, output_path, dpi):
    if delta_df.empty:
        return None
    summary = delta_df.groupby("metric_name", dropna=False)["delta_fuzzy_minus_strict"].apply(lambda s: float(pd.to_numeric(s, errors="coerce").abs().sum())).reset_index()
    summary = summary.sort_values("delta_fuzzy_minus_strict", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(summary["metric_name"], summary["delta_fuzzy_minus_strict"])
    setup_axis(ax, "Strict-vs-fuzzy absolute deltas", "Total absolute count delta", None)
    return save_figure(fig, output_path, dpi)


def plot_feature_family_readiness(readiness_df, output_path, dpi):
    strict = readiness_df[readiness_df["identity_resolution_mode"] == "strict"].copy()
    if strict.empty:
        strict = readiness_df.copy()
    if strict.empty:
        return None
    strict = strict.sort_values("readiness_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(strict["feature_family_label"], strict["readiness_score"])
    ax.set_xlim(0, 3)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Not ready", "Descriptive", "Secondary", "Ready"])
    setup_axis(ax, "Feature-family readiness (strict)", "Readiness score", None)
    return save_figure(fig, output_path, dpi)


def build_figures(all_df, summaries, figures_dir, dpi):
    ensure_dir(figures_dir)
    paths = {}
    paths["ownership_coverage_by_repo"] = plot_ownership_coverage_by_repo(
        summaries["coverage_by_repo_group"], figures_dir / "01_ownership_coverage_by_repo.png", dpi
    )
    paths["repo_participant_role_coverage"] = plot_repo_participant_role_coverage(
        summaries["repo_participant_role_summary"], figures_dir / "02_repo_participant_role_coverage.png", dpi
    )
    paths["file_participant_role_coverage"] = plot_file_participant_role_coverage(
        all_df, figures_dir / "03_file_participant_role_coverage.png", dpi
    )
    paths["pre_post_continuity_coverage"] = plot_pre_post_continuity_coverage(
        summaries["pre_post_continuity_summary"], figures_dir / "04_pre_post_continuity_coverage.png", dpi
    )
    paths["strict_vs_fuzzy_delta"] = plot_strict_vs_fuzzy_delta(
        summaries["strict_vs_fuzzy_comparison"], figures_dir / "05_strict_vs_fuzzy_delta.png", dpi
    )
    paths["feature_family_readiness"] = plot_feature_family_readiness(
        summaries["feature_family_readiness"], figures_dir / "06_feature_family_readiness.png", dpi
    )
    return {key: str(value) for key, value in paths.items() if value is not None}


# -----------------------------
# Markdown report
# -----------------------------


def markdown_table(df, max_rows=20):
    if df is None or df.empty:
        return "_No rows available._"
    display = df.head(max_rows).copy()
    return display.to_markdown(index=False)


def get_status_for_family(readiness_df, family, mode="strict"):
    part = readiness_df[(readiness_df["feature_family"] == family) & (readiness_df["identity_resolution_mode"] == mode)]
    if part.empty:
        return "unknown"
    return str(part["readiness_status"].iloc[0])


def format_percent(value):
    if value is None or pd.isna(value):
        return "NA"
    return "{0:.1f}%".format(float(value) * 100.0)


def build_markdown_report(summaries, paths_written, input_paths, figure_paths):
    population = summaries["population_summary"]
    leakage = summaries["leakage_guard_summary"]
    readiness = summaries["feature_family_readiness"]
    issue_summary = summaries["issue_level_evidence_summary"]
    repo_summary = summaries["repo_participant_role_summary"]
    file_summary = summaries["file_participant_role_summary"]
    continuity_summary = summaries["pre_post_continuity_summary"]

    strict_population = population[population["identity_resolution_mode"] == "strict"] if not population.empty else pd.DataFrame()
    strict_rows = int(strict_population["issue_rows"].iloc[0]) if not strict_population.empty else 0
    strict_repos = int(strict_population["unique_repos"].iloc[0]) if not strict_population.empty else 0
    leakage_pass = bool((leakage["status"] == "pass").all()) if not leakage.empty else False

    lines = []
    lines.append("# Ownership QA and Analysis Readiness Report")
    lines.append("")
    lines.append("Generated at: `{0}`".format(datetime.now(timezone.utc).isoformat()))
    lines.append("")
    lines.append("## Executive summary")
    lines.append("")
    lines.append("- Strict ownership features contain **{0:,} issue rows** across **{1:,} repositories**.".format(strict_rows, strict_repos))
    lines.append("- Leakage guard status: **{0}**.".format("PASS" if leakage_pass else "CHECK WARNINGS"))
    lines.append("- Repo-level participant roles are the most likely primary ownership-adjacent feature family for RQ2.")
    lines.append("- File-level participant roles are now explicitly coverage-flagged and should be analyzed using the applicable denominator.")
    lines.append("- Direct issue-linked ownership evidence remains useful as a descriptive/secondary layer because it is conditional on PR/commit/file-link evidence.")
    lines.append("")

    lines.append("## Input files")
    lines.append("")
    for key, value in input_paths.items():
        lines.append("- `{0}`: `{1}`".format(key, value))
    lines.append("")

    lines.append("## Population integrity")
    lines.append("")
    lines.append(markdown_table(population))
    lines.append("")

    lines.append("## Leakage guard results")
    lines.append("")
    lines.append(markdown_table(leakage[["identity_resolution_mode", "check_name", "value", "status"]] if not leakage.empty else leakage, max_rows=40))
    lines.append("")

    lines.append("## Feature-family readiness")
    lines.append("")
    readiness_display_cols = ["identity_resolution_mode", "feature_family_label", "readiness_status", "wontfix_coverage_rate", "comparison_coverage_rate", "reason"]
    lines.append(markdown_table(readiness[safe_columns(readiness, readiness_display_cols)], max_rows=20))
    lines.append("")

    lines.append("## Direct issue-linked ownership evidence")
    lines.append("")
    lines.append("Direct issue-linked ownership evidence captures selected pre/post ownership rows from issue-linked PR/commit/file evidence. This remains the most conservative ownership signal, but its denominator is limited by PR/commit/file linkage.")
    lines.append("")
    lines.append(markdown_table(issue_summary, max_rows=20))
    lines.append("")

    lines.append("## Repo-level participant roles")
    lines.append("")
    lines.append("Repo-level participant-role features ask whether the issue author, commenters, or top commenter had pre-issue repo contribution history. These are generally higher-coverage than direct issue-linked ownership.")
    lines.append("")
    lines.append(markdown_table(repo_summary, max_rows=20))
    lines.append("")

    lines.append("## File-level participant roles")
    lines.append("")
    lines.append("File-level participant-role features are now paired with explicit coverage flags. Use `participant_role_file_features_applicable` or `participant_role_file_coverage_flag == ok` to define the applicable denominator.")
    lines.append("")
    lines.append(markdown_table(file_summary, max_rows=20))
    lines.append("")

    lines.append("## Pre/post continuity")
    lines.append("")
    lines.append("Continuity features measure whether pre-issue owners or contributors overlap with post-issue owners. They should not be interpreted as pre-issue ownership definitions.")
    lines.append("")
    lines.append(markdown_table(continuity_summary, max_rows=20))
    lines.append("")

    lines.append("## Strict vs fuzzy identity sensitivity")
    lines.append("")
    delta = summaries["strict_vs_fuzzy_comparison"]
    if delta.empty:
        lines.append("_Strict-vs-fuzzy comparison was unavailable because one mode was missing._")
    else:
        delta_display = delta.sort_values("delta_fuzzy_minus_strict", key=lambda s: s.abs(), ascending=False).head(20)
        lines.append(markdown_table(delta_display, max_rows=20))
    lines.append("")

    lines.append("## Candidate features for RQ2 analysis")
    lines.append("")
    lines.append(markdown_table(summaries["candidate_analysis_features"], max_rows=20))
    lines.append("")

    if figure_paths:
        lines.append("## Figures")
        lines.append("")
        for key, path in figure_paths.items():
            rel = Path(path).name
            lines.append("- `{0}`: `figures/{1}`".format(key, rel))
        lines.append("")

    lines.append("## Recommended next step")
    lines.append("")
    repo_status = get_status_for_family(readiness, "repo_participant_roles")
    file_status = get_status_for_family(readiness, "file_participant_roles")
    continuity_status = get_status_for_family(readiness, "continuity")
    lines.append("Build or refactor the RQ2 ownership analysis dataset using the readiness results:")
    lines.append("")
    lines.append("- Repo participant roles: **{0}**".format(READINESS_LABELS.get(repo_status, repo_status)))
    lines.append("- File participant roles: **{0}**".format(READINESS_LABELS.get(file_status, file_status)))
    lines.append("- Continuity features: **{0}**".format(READINESS_LABELS.get(continuity_status, continuity_status)))
    lines.append("")
    lines.append("A good next implementation target is an RQ2 analysis dataset/report script that treats repo-level participant roles as primary candidates and file/continuity features as coverage-aware secondary analyses.")
    lines.append("")

    return "\n".join(lines)


# -----------------------------
# Main runner
# -----------------------------


def load_inputs(args):
    strict_df_raw = read_table(args.ownership_features, required=True)
    fuzzy_df_raw = read_table(args.ownership_features_fuzzy, required=not args.allow_missing_fuzzy)

    raw_frames = []
    duplicate_counts = {}
    strict_dup = int(strict_df_raw.duplicated(subset=[c for c in ["repo_full_name", "issue_id", "issue_number"] if c in strict_df_raw.columns]).sum()) if not strict_df_raw.empty else 0
    duplicate_counts["strict"] = strict_dup
    strict_df = normalize_ownership_dataset(strict_df_raw, "strict")
    raw_frames.append(strict_df)

    if fuzzy_df_raw is not None and not fuzzy_df_raw.empty:
        fuzzy_dup = int(fuzzy_df_raw.duplicated(subset=[c for c in ["repo_full_name", "issue_id", "issue_number"] if c in fuzzy_df_raw.columns]).sum())
        duplicate_counts["fuzzy"] = fuzzy_dup
        fuzzy_df = normalize_ownership_dataset(fuzzy_df_raw, "fuzzy")
        raw_frames.append(fuzzy_df)
    else:
        duplicate_counts["fuzzy"] = 0

    all_df = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()

    qa_frames = {
        "strict": read_table(args.ownership_qa, required=False),
        "fuzzy": read_table(args.ownership_qa_fuzzy, required=False),
    }
    manifests = {
        "strict": read_json(args.strict_manifest, required=False),
        "fuzzy": read_json(args.fuzzy_manifest, required=False),
    }
    optional_inputs = {
        "contributor_profile_qa": read_table(args.contributor_profile_qa, required=False),
        "contributor_profile_qa_fuzzy": read_table(args.contributor_profile_qa_fuzzy, required=False),
        "analysis_qa": read_table(args.analysis_qa, required=False),
    }
    return all_df, qa_frames, manifests, optional_inputs, duplicate_counts


def build_all_summaries(all_df, qa_frames, manifests, optional_inputs, duplicate_counts):
    qa_summary = build_qa_summary(qa_frames, manifests)
    population_summary = build_population_summary(all_df, duplicate_counts)
    coverage_by_repo_group = build_coverage_by_repo_and_group(all_df)
    issue_level_evidence_summary = build_issue_level_evidence_summary(all_df)
    repo_participant_role_summary = build_repo_participant_role_summary(all_df)
    file_participant_role_summary = build_file_participant_role_summary(all_df)
    pre_post_continuity_summary = build_pre_post_continuity_summary(all_df)
    leakage_guard_summary = build_leakage_guard_summary(all_df, qa_summary, manifests)
    feature_family_readiness, feature_family_coverage = build_feature_family_readiness(all_df)
    strict_vs_fuzzy_comparison = build_strict_vs_fuzzy_comparison(coverage_by_repo_group)
    candidate_analysis_features = build_candidate_analysis_features(feature_family_readiness)

    return {
        "ownership_qa_summary": qa_summary,
        "population_summary": population_summary,
        "coverage_by_repo_group": coverage_by_repo_group,
        "issue_level_evidence_summary": issue_level_evidence_summary,
        "repo_participant_role_summary": repo_participant_role_summary,
        "file_participant_role_summary": file_participant_role_summary,
        "pre_post_continuity_summary": pre_post_continuity_summary,
        "leakage_guard_summary": leakage_guard_summary,
        "feature_family_readiness": feature_family_readiness,
        "feature_family_coverage": feature_family_coverage,
        "strict_vs_fuzzy_comparison": strict_vs_fuzzy_comparison,
        "candidate_analysis_features": candidate_analysis_features,
        "contributor_profile_qa": optional_inputs.get("contributor_profile_qa", pd.DataFrame()),
        "contributor_profile_qa_fuzzy": optional_inputs.get("contributor_profile_qa_fuzzy", pd.DataFrame()),
        "analysis_qa": optional_inputs.get("analysis_qa", pd.DataFrame()),
    }


def write_outputs(summaries, all_df, args):
    output_dir = ensure_dir(args.output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    csv_specs = {
        "ownership_qa_summary.csv": summaries["ownership_qa_summary"],
        "ownership_population_summary.csv": summaries["population_summary"],
        "ownership_coverage_by_repo_and_group.csv": summaries["coverage_by_repo_group"],
        "ownership_feature_family_readiness.csv": summaries["feature_family_readiness"],
        "ownership_feature_family_coverage.csv": summaries["feature_family_coverage"],
        "ownership_issue_level_evidence_summary.csv": summaries["issue_level_evidence_summary"],
        "ownership_repo_participant_role_summary.csv": summaries["repo_participant_role_summary"],
        "ownership_file_participant_role_summary.csv": summaries["file_participant_role_summary"],
        "ownership_pre_post_continuity_summary.csv": summaries["pre_post_continuity_summary"],
        "ownership_strict_vs_fuzzy_comparison.csv": summaries["strict_vs_fuzzy_comparison"],
        "ownership_leakage_guard_summary.csv": summaries["leakage_guard_summary"],
        "ownership_candidate_analysis_features.csv": summaries["candidate_analysis_features"],
    }

    paths_written = {}
    for filename, df in csv_specs.items():
        paths_written[filename] = str(write_csv(df, output_dir / filename))

    figure_paths = build_figures(all_df, summaries, figures_dir, args.png_dpi)

    input_paths = {
        "ownership_features": args.ownership_features,
        "ownership_features_fuzzy": args.ownership_features_fuzzy,
        "ownership_qa": args.ownership_qa,
        "ownership_qa_fuzzy": args.ownership_qa_fuzzy,
        "strict_manifest": args.strict_manifest,
        "fuzzy_manifest": args.fuzzy_manifest,
        "contributor_profile_qa": args.contributor_profile_qa,
        "contributor_profile_qa_fuzzy": args.contributor_profile_qa_fuzzy,
        "analysis_qa": args.analysis_qa,
    }

    report_text = build_markdown_report(summaries, paths_written, input_paths, figure_paths)
    report_path = write_text(report_text, output_dir / "ownership_analysis_readiness_report.md")
    paths_written["ownership_analysis_readiness_report.md"] = str(report_path)

    manifest = {
        "script": "16_build_ownership_report.py",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": input_paths,
        "output_dir": str(output_dir),
        "output_paths": paths_written,
        "figure_paths": figure_paths,
        "row_counts": {
            "ownership_issue_rows_combined": int(len(all_df)),
            "strict_rows": int((all_df["identity_resolution_mode"] == "strict").sum()) if "identity_resolution_mode" in all_df.columns else 0,
            "fuzzy_rows": int((all_df["identity_resolution_mode"] == "fuzzy").sum()) if "identity_resolution_mode" in all_df.columns else 0,
            "feature_family_readiness_rows": int(len(summaries["feature_family_readiness"])),
            "leakage_guard_rows": int(len(summaries["leakage_guard_summary"])),
        },
    }
    manifest_path = write_json(manifest, output_dir / "16_build_ownership_report_run_manifest.json")
    paths_written["16_build_ownership_report_run_manifest.json"] = str(manifest_path)

    return paths_written, figure_paths, manifest


def main():
    args = parse_args()
    all_df, qa_frames, manifests, optional_inputs, duplicate_counts = load_inputs(args)
    if all_df.empty:
        raise RuntimeError("No ownership feature rows were loaded.")
    summaries = build_all_summaries(all_df, qa_frames, manifests, optional_inputs, duplicate_counts)
    paths_written, figure_paths, manifest = write_outputs(summaries, all_df, args)
    print("Ownership readiness report complete.")
    print("Output directory: {0}".format(args.output_dir))
    print("Report: {0}".format(paths_written.get("ownership_analysis_readiness_report.md")))


if __name__ == "__main__":
    main()
