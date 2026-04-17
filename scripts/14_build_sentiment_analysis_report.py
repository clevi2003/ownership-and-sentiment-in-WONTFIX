#!/usr/bin/env python3
"""
Comprehensive sentiment analysis and reporting script for the WONTFIX pipeline.

This script is designed to sit *after* the analysis-dataset build stage. It reads
the merged issue-level RQ1 dataset from data/final/analysis_dataset_rq1.parquet,
optionally reads comment-level sentiment features for trajectory-style plots,
produces analysis-ready plots/tables, and writes a Markdown summary report.

Key goals
---------
- keep the issue as the main inferential unit
- be repo-aware via within-repo standardization and repo fixed effects
- report effect sizes, not only p-values
- provide QA / coverage summaries so silent row loss is visible
- degrade gracefully when optional columns / files are unavailable

Typical usage
-------------
python build_sentiment_analysis_report.py \
  --rq1-dataset data/final/analysis_dataset_rq1.parquet \
  --comment-features data/features/sentiment/comment_sentiment_features.parquet \
  --output-dir outputs/sentiment_analysis
"""
from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scipy import stats

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

UTK_ORANGE_HEXBIN_CMAP = LinearSegmentedColormap.from_list(
    "utk_orange_hexbin",
    [UTK_COLORS["white"], "#FFE7CC", "#FFB866", UTK_COLORS["orange"], "#CC6800"],
)

UTK_GRAY_HEXBIN_CMAP = LinearSegmentedColormap.from_list(
    "utk_gray_hexbin",
    [UTK_COLORS["white"], "#E8E8E8", "#BFC1C4", UTK_COLORS["smokey_gray"], "#3F4042"],
)

GROUP_ALIASES = {
    "wontfix": "wontfix",
    "comparison": "comparison",
    "resolved_pr": "pr_resolved",
    "pr_resolved": "pr_resolved",
    "closed_non_wontfix": "closed_non_wontfix",
    "closed non wontfix": "closed_non_wontfix",
    "open": "open",
    "invalid": "invalid",
    "missing": "missing",
}
from statsmodels.formula.api import ols, logit
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm


# -----------------------------
# Configuration / helpers
# -----------------------------

DEFAULT_RQ1_DATASET = "data/final/analysis_dataset_rq1.parquet"
DEFAULT_COMMENT_FEATURES = "data/features/sentiment/comment_sentiment_features.parquet"

PRIMARY_CONTINUOUS_FEATURES = [
    "mean_comment_sentiment",
    "median_comment_sentiment",
    "min_comment_sentiment",
    "max_comment_sentiment",
    "std_comment_sentiment",
    "comment_sentiment_change_late_minus_early",
    "comment_sentiment_slope",
    "negative_comment_share",
    "positive_comment_share",
]

PARTICIPATION_FEATURES = [
    "comment_count",
    "unique_commenter_count",
    "top_commenter_share",
    "comment_concentration_ratio",
    "num_distinct_non_author_commenters",
]

NEGATIVE_THRESHOLD = -0.05
POSITIVE_THRESHOLD = 0.05
MIN_GROUP_N_FOR_TEST = 5


@dataclass
class Paths:
    rq1_dataset: Path
    comment_features: Path | None
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run comprehensive sentiment analysis for WONTFIX issue discussions.")
    parser.add_argument("--rq1-dataset", default=DEFAULT_RQ1_DATASET)
    parser.add_argument("--comment-features", default=DEFAULT_COMMENT_FEATURES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-comments-for-temporal", type=int, default=2)
    parser.add_argument("--min-comments-for-trajectory", type=int, default=3)
    parser.add_argument("--comment-trajectory-bins", type=int, default=5)
    parser.add_argument("--winsorize", action="store_true", help="Winsorize heavy-tailed numeric features at 1st/99th pct.")
    parser.add_argument("--exclude-zero-text-issues", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def maybe_read_parquet(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def lower_map(columns: Iterable[str]) -> dict[str, str]:
    return {str(c).lower(): str(c) for c in columns}


def find_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = False) -> str | None:
    cmap = lower_map(df.columns)
    for cand in candidates:
        if cand.lower() in cmap:
            return cmap[cand.lower()]
    for col in df.columns:
        cl = str(col).lower()
        for cand in candidates:
            if cand.lower() in cl:
                return str(col)
    if required:
        raise KeyError(f"Required column not found. Tried: {candidates}")
    return None


def clean_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def to_datetime(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(series, errors="coerce", utc=True)


def to_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def safe_divide(numer, denom, default=np.nan):
    try:
        if denom is None or pd.isna(denom) or float(denom) == 0.0:
            return default
        return float(numer) / float(denom)
    except Exception:
        return default


def winsorize_series(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    if series.dropna().empty:
        return series
    lo = series.quantile(lower_q)
    hi = series.quantile(upper_q)
    return series.clip(lower=lo, upper=hi)


def standardize_within_group(df: pd.DataFrame, value_col: str, group_col: str = "repo_full_name") -> pd.Series:
    def zscore(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        if pd.isna(std) or std == 0:
            return pd.Series(np.zeros(len(s)), index=s.index, dtype="float64")
        return (s - s.mean()) / std
    return df.groupby(group_col, dropna=False)[value_col].transform(zscore)


def cohen_d(x: Sequence[float], y: Sequence[float]) -> float:
    x = pd.Series(x).dropna().astype(float)
    y = pd.Series(y).dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    pooled = math.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / max(nx + ny - 2, 1))
    if pooled == 0 or pd.isna(pooled):
        return np.nan
    return (x.mean() - y.mean()) / pooled


def hedges_g(x: Sequence[float], y: Sequence[float]) -> float:
    d = cohen_d(x, y)
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    n = len(x) + len(y)
    if pd.isna(d) or n <= 3:
        return d
    correction = 1 - (3 / (4 * n - 9))
    return d * correction


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    x = pd.Series(x).dropna().astype(float).to_numpy()
    y = pd.Series(y).dropna().astype(float).to_numpy()
    if len(x) == 0 or len(y) == 0:
        return np.nan
    diff = 0
    for xv in x:
        diff += np.sum(xv > y) - np.sum(xv < y)
    return diff / (len(x) * len(y))


def benjamini_hochberg(df: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    out = df.copy()
    if out.empty or p_col not in out.columns:
        return out
    valid = out[p_col].notna()
    if valid.sum() == 0:
        out["p_value_fdr_bh"] = np.nan
        out["reject_fdr_bh_05"] = False
        return out
    reject, p_adj, _, _ = multipletests(out.loc[valid, p_col], method="fdr_bh")
    out["p_value_fdr_bh"] = np.nan
    out["reject_fdr_bh_05"] = False
    out.loc[valid, "p_value_fdr_bh"] = p_adj
    out.loc[valid, "reject_fdr_bh_05"] = reject
    return out


def normalize_issue_set(df: pd.DataFrame, analysis_set: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number", "analysis_set", "comparison_group", "issue_type"])
    repo_col = find_col(df, ["repo_full_name", "repo_name", "full_name", "repo"], required=True)
    issue_id_col = find_col(df, ["issue_id", "id"])
    issue_num_col = find_col(df, ["issue_number", "number"])
    issue_type_col = find_col(df, ["issue_type", "__issue_type", "type"])
    cmp_group_col = find_col(df, ["comparison_group", "comparison_bucket", "__comparison_bucket", "bucket"])
    out = pd.DataFrame({
        "repo_full_name": df[repo_col].astype(str),
        "analysis_set": analysis_set,
    })
    out["issue_id"] = df[issue_id_col].astype(str) if issue_id_col else None
    out["issue_number"] = to_numeric(df[issue_num_col]) if issue_num_col else np.nan
    out["issue_type"] = df[issue_type_col].astype(str) if issue_type_col else None
    out["comparison_group"] = df[cmp_group_col].astype(str) if cmp_group_col else (analysis_set if analysis_set == "wontfix" else "comparison")
    if analysis_set == "wontfix":
        out["comparison_group"] = "wontfix"
    return out.drop_duplicates().reset_index(drop=True)


# -----------------------------
# Loading / harmonization
# -----------------------------


def load_and_prepare(paths: Paths, winsorize: bool = False, exclude_zero_text_issues: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    issue_df = pd.read_parquet(paths.rq1_dataset)
    comment_df = maybe_read_parquet(paths.comment_features)

    qa: dict[str, object] = {
        "rq1_rows_raw": int(len(issue_df)),
        "comment_feature_rows_raw": int(len(comment_df)),
    }

    issue_df = normalize_rq1_issue_dataset(issue_df)
    validate_rq1_dataset(issue_df)
    comment_df = normalize_comment_features(comment_df)

    qa["rq1_rows_normalized"] = int(len(issue_df))
    qa["comment_feature_rows_normalized"] = int(len(comment_df))
    qa["rq1_duplicate_issue_keys"] = int(issue_df.duplicated(subset=["repo_full_name", "issue_id", "issue_number"]).sum())
    qa["comment_duplicates"] = int(comment_df.duplicated(subset=["repo_full_name", "issue_id", "issue_number", "comment_id"]).sum()) if not comment_df.empty else 0

    issue_df = issue_df.drop_duplicates(subset=["repo_full_name", "issue_id", "issue_number"]).reset_index(drop=True)
    if not comment_df.empty:
        comment_df = comment_df.drop_duplicates(subset=["repo_full_name", "issue_id", "issue_number", "comment_id"]).reset_index(drop=True)

    issue_df = add_derived_rq1_columns(
        issue_df,
        winsorize=winsorize,
        exclude_zero_text_issues=exclude_zero_text_issues,
        qa=qa,
    )

    qa["repos_represented"] = int(issue_df["repo_full_name"].nunique()) if "repo_full_name" in issue_df.columns else 0
    qa["analysis_groups"] = sorted(issue_df["analysis_set"].dropna().astype(str).unique().tolist()) if "analysis_set" in issue_df.columns else []
    qa["rows_missing_core_sentiment_feature"] = int(issue_df["mean_comment_sentiment"].isna().sum()) if "mean_comment_sentiment" in issue_df.columns else int(len(issue_df))
    participation_cols = [col for col in PARTICIPATION_FEATURES if col in issue_df.columns]
    qa["rows_missing_participation_covariates"] = int(issue_df[participation_cols].isna().all(axis=1).sum()) if participation_cols else int(len(issue_df))
    if "usable_for_rq1" in issue_df.columns:
        qa["rows_not_marked_usable_for_rq1"] = int((~issue_df["usable_for_rq1"].fillna(False)).sum())

    return issue_df.reset_index(drop=True), comment_df.reset_index(drop=True), qa


def normalize_rq1_issue_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    repo_col = find_col(out, ["repo_full_name", "repo_name", "full_name"], required=True)
    issue_id_col = find_col(out, ["issue_id", "id"], required=True)
    issue_num_col = find_col(out, ["issue_number", "number"])
    analysis_set_col = find_col(out, ["analysis_set"], required=True)
    comparison_group_col = find_col(out, ["comparison_group"])

    renames = {
        repo_col: "repo_full_name",
        issue_id_col: "issue_id",
        analysis_set_col: "analysis_set",
    }
    if issue_num_col:
        renames[issue_num_col] = "issue_number"
    if comparison_group_col:
        renames[comparison_group_col] = "comparison_group"
    out = out.rename(columns=renames)

    if "issue_number" not in out.columns:
        out["issue_number"] = np.nan
    if "comparison_group" not in out.columns:
        out["comparison_group"] = None

    out["repo_full_name"] = out["repo_full_name"].astype(str)
    out["issue_id"] = out["issue_id"].astype(str)
    out["issue_number"] = to_numeric(out["issue_number"]) if "issue_number" in out.columns else np.nan

    for col in ["created_at", "closed_at", "pushed_at", "created_at_repo"]:
        if col in out.columns:
            out[col] = to_datetime(out[col])

    for col in out.columns:
        if any(token in col for token in ["count", "share", "ratio", "sentiment", "length", "slope", "entropy", "range"]):
            out[col] = to_numeric(out[col])

    out["analysis_set"] = out["analysis_set"].astype(str).replace({"nan": None})
    if "comparison_group" in out.columns:
        out["comparison_group"] = out["comparison_group"].astype(str).replace({"nan": None})
        fallback = pd.Series(np.where(out["analysis_set"].eq("wontfix"), "wontfix", "comparison"), index=out.index)
        out["comparison_group"] = out["comparison_group"].fillna(fallback)

    return out


def validate_rq1_dataset(df: pd.DataFrame) -> None:
    required_columns = ["repo_full_name", "issue_id", "analysis_set"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"RQ1 dataset is missing required columns: {missing}")
    if not any(col in df.columns for col in PRIMARY_CONTINUOUS_FEATURES):
        raise KeyError("RQ1 dataset does not contain any primary sentiment feature columns.")


def add_derived_rq1_columns(issue_df: pd.DataFrame, winsorize: bool = False, exclude_zero_text_issues: bool = False, qa: dict[str, object] | None = None) -> pd.DataFrame:
    out = issue_df.copy()

    if exclude_zero_text_issues and "comments_with_text_count" in out.columns:
        before = len(out)
        out = out[out["comments_with_text_count"].fillna(0) > 0].copy()
        if qa is not None:
            qa["issues_removed_zero_text_filter"] = int(before - len(out))

    for col in [c for c in PRIMARY_CONTINUOUS_FEATURES if c in out.columns]:
        out[f"repo_z_{col}"] = standardize_within_group(out, col, group_col="repo_full_name")

    if "min_comment_sentiment" in out.columns:
        out["has_strongly_negative_comment"] = out["min_comment_sentiment"] < NEGATIVE_THRESHOLD
    if "max_comment_sentiment" in out.columns:
        out["has_strongly_positive_comment"] = out["max_comment_sentiment"] > POSITIVE_THRESHOLD
    if "comment_sentiment_change_late_minus_early" in out.columns:
        out["late_more_negative_than_early"] = out["comment_sentiment_change_late_minus_early"] < 0
    if "std_comment_sentiment" in out.columns and out["std_comment_sentiment"].notna().any():
        cutoff = out["std_comment_sentiment"].quantile(0.75)
        out["high_sentiment_volatility"] = out["std_comment_sentiment"] >= cutoff
    if {"max_comment_sentiment", "min_comment_sentiment"}.issubset(out.columns):
        out["comment_sentiment_range"] = out["max_comment_sentiment"] - out["min_comment_sentiment"]

    if winsorize:
        for col in [c for c in [*PRIMARY_CONTINUOUS_FEATURES, *PARTICIPATION_FEATURES] if c in out.columns]:
            out[f"winsor_{col}"] = winsorize_series(out[col])

    return out


def normalize_comment_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    repo_col = find_col(out, ["repo_full_name", "repo_name", "full_name"], required=True)
    issue_id_col = find_col(out, ["issue_id", "id"], required=True)
    issue_num_col = find_col(out, ["issue_number", "number"])
    comment_id_col = find_col(out, ["comment_id"], required=True)

    renames = {
        repo_col: "repo_full_name",
        issue_id_col: "issue_id",
        comment_id_col: "comment_id",
    }
    if issue_num_col:
        renames[issue_num_col] = "issue_number"
    out = out.rename(columns=renames)
    if "issue_number" not in out.columns:
        out["issue_number"] = np.nan

    out["repo_full_name"] = out["repo_full_name"].astype(str)
    out["issue_id"] = out["issue_id"].astype(str)
    out["comment_id"] = out["comment_id"].astype(str)
    out["issue_number"] = to_numeric(out["issue_number"])

    for col in ["created_at"]:
        if col in out.columns:
            out[col] = to_datetime(out[col])

    for col in out.columns:
        if any(token in col for token in ["count", "share", "ratio", "sentiment", "length", "sequence", "index"]):
            out[col] = to_numeric(out[col])

    return out


def enrich_issue_features(issue_df: pd.DataFrame, paths: Paths) -> pd.DataFrame:
    out = issue_df.copy()

    wontfix_df = normalize_issue_set(maybe_read_parquet(paths.wontfix_set), "wontfix")
    comparison_df = normalize_issue_set(maybe_read_parquet(paths.comparison_set), "comparison")
    issue_sets = pd.concat([wontfix_df, comparison_df], ignore_index=True)

    if not issue_sets.empty:
        out = merge_issue_metadata(out, issue_sets)

    issues_resolved = maybe_read_parquet(paths.issues_resolved)
    if not issues_resolved.empty:
        out = merge_issue_metadata(out, normalize_issues_resolved(issues_resolved))

    repos_df = maybe_read_parquet(paths.repositories)
    if not repos_df.empty:
        out = merge_repo_metadata(out, repos_df)

    # Backfill analysis_set if still missing
    if "analysis_set" in out.columns:
        out["analysis_set"] = out["analysis_set"].replace({"nan": None})
        if out["analysis_set"].isna().all() and "comparison_group" in out.columns:
            out["analysis_set"] = pd.Series(
                np.where(out["comparison_group"].eq("wontfix"), "wontfix", "comparison"),
                index=out.index,
            )

    # Backfill comparison_group
    fallback_comparison_group = pd.Series(
        np.where(out["analysis_set"].eq("wontfix"), "wontfix", "comparison"),
        index=out.index,
    )

    if "comparison_group" not in out.columns:
        out["comparison_group"] = fallback_comparison_group
    else:
        out["comparison_group"] = out["comparison_group"].replace({"nan": None})
        out["comparison_group"] = out["comparison_group"].fillna(fallback_comparison_group)
    return out


def merge_issue_metadata(base: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    merge_keys = ["repo_full_name", "issue_id"]
    if meta["issue_id"].isna().all() and "issue_number" in meta.columns:
        merge_keys = ["repo_full_name", "issue_number"]
    if "issue_number" in base.columns and "issue_number" in meta.columns:
        by_num = base.merge(meta.drop(columns=[c for c in ["issue_id"] if c in meta.columns]), on=["repo_full_name", "issue_number"], how="left", suffixes=("", "_meta_num"))
        by_id = base.merge(meta, on=["repo_full_name", "issue_id"], how="left", suffixes=("", "_meta_id"))
        out = base.copy()
        for col in meta.columns:
            if col in {"repo_full_name", "issue_id", "issue_number"}:
                continue
            if col in by_id.columns:
                out[col] = by_id[col]
            if col in by_num.columns:
                out[col] = out.get(col).combine_first(by_num[col]) if col in out.columns else by_num[col]
        return out
    return base.merge(meta, on=merge_keys, how="left")


def normalize_issues_resolved(df: pd.DataFrame) -> pd.DataFrame:
    repo_col = find_col(df, ["repo_full_name", "repo_name", "full_name"], required=True)
    issue_id_col = find_col(df, ["issue_id", "id"], required=True)
    issue_num_col = find_col(df, ["issue_number", "number"])
    state_col = find_col(df, ["state"])
    created_col = find_col(df, ["created_at"])
    closed_col = find_col(df, ["closed_at"])
    labels_col = find_col(df, ["label_names_json", "labels", "label_names"])
    issue_type_col = find_col(df, ["issue_type", "__issue_type", "type"])
    out = pd.DataFrame({
        "repo_full_name": df[repo_col].astype(str),
        "issue_id": df[issue_id_col].astype(str),
    })
    out["issue_number"] = to_numeric(df[issue_num_col]) if issue_num_col else np.nan
    if state_col:
        out["state"] = df[state_col].astype(str)
    if created_col:
        out["created_at"] = to_datetime(df[created_col])
    if closed_col:
        out["closed_at"] = to_datetime(df[closed_col])
    if issue_type_col:
        out["issue_type"] = df[issue_type_col].astype(str)
    if labels_col:
        out["label_names_json"] = df[labels_col]
    return out.drop_duplicates(subset=["repo_full_name", "issue_id", "issue_number"]) 


def merge_repo_metadata(base: pd.DataFrame, repos_df: pd.DataFrame) -> pd.DataFrame:
    repo_col = find_col(repos_df, ["repo_full_name", "full_name", "repo_name"], required=True)
    meta_cols = [repo_col]
    for cand in ["language", "primary_language", "stargazers_count", "stars", "created_at", "owner_login", "visibility"]:
        col = find_col(repos_df, [cand])
        if col and col not in meta_cols:
            meta_cols.append(col)
    meta = repos_df[meta_cols].copy().rename(columns={repo_col: "repo_full_name"})
    return base.merge(meta, on="repo_full_name", how="left")


# -----------------------------
# QA / summaries
# -----------------------------


def build_qa_summary(issue_df: pd.DataFrame, comment_df: pd.DataFrame, qa_seed: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    q = dict(qa_seed)

    def add(metric: str, value):
        rows.append({"metric": metric, "value": value})

    for k, v in q.items():
        add(k, v)

    if not issue_df.empty:
        add("issues_final", len(issue_df))
        add("repos_final", issue_df["repo_full_name"].nunique())
        if "analysis_set" in issue_df.columns:
            for grp, n in issue_df["analysis_set"].fillna("missing").value_counts(dropna=False).items():
                add(f"issues_analysis_set__{grp}", int(n))
        if "comparison_group" in issue_df.columns:
            for grp, n in issue_df["comparison_group"].fillna("missing").value_counts(dropna=False).items():
                add(f"issues_comparison_group__{grp}", int(n))
        for col in ["comment_count", "comments_with_text_count", "unique_commenter_count"]:
            if col in issue_df.columns:
                add(f"{col}_median", float(issue_df[col].median()))
                add(f"{col}_mean", float(issue_df[col].mean()))
        if "zero_comment_flag" in issue_df.columns:
            add("zero_comment_issue_share", float(issue_df["zero_comment_flag"].mean()))
        if "comments_with_text_count" in issue_df.columns:
            add("zero_text_comment_issue_share", float((issue_df["comments_with_text_count"].fillna(0) == 0).mean()))
        if "unique_commenter_count" in issue_df.columns:
            add("one_commenter_issue_share", float((issue_df["unique_commenter_count"].fillna(0) <= 1).mean()))
        for col in PRIMARY_CONTINUOUS_FEATURES:
            if col in issue_df.columns:
                add(f"missing_share__{col}", float(issue_df[col].isna().mean()))

    if not comment_df.empty:
        add("comments_final", len(comment_df))
        if "has_text" in comment_df.columns:
            add("comment_missing_text_share", float((comment_df["has_text"].fillna(0) != 1).mean()))

    return pd.DataFrame(rows)


def build_group_descriptives(issue_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if issue_df.empty:
        return pd.DataFrame()

    grouping_cols = ["analysis_set", "comparison_group"]
    group_labels = issue_df[grouping_cols].fillna("missing")
    grouped = issue_df.assign(
        analysis_set=group_labels["analysis_set"],
        comparison_group=group_labels["comparison_group"],
    ).groupby(grouping_cols, dropna=False)

    for (analysis_set, comparison_group), g in grouped:
        base = {
            "analysis_set": analysis_set,
            "comparison_group": comparison_group,
            "n_issues": len(g),
            "n_repos": g["repo_full_name"].nunique(),
        }
        for col in ["comment_count", "comments_with_text_count", "unique_commenter_count", *PRIMARY_CONTINUOUS_FEATURES]:
            if col not in g.columns:
                continue
            vals = pd.to_numeric(g[col], errors="coerce")
            base[f"{col}__mean"] = vals.mean()
            base[f"{col}__sd"] = vals.std(ddof=1)
            base[f"{col}__median"] = vals.median()
            base[f"{col}__q1"] = vals.quantile(0.25)
            base[f"{col}__q3"] = vals.quantile(0.75)
        rows.append(base)
    return pd.DataFrame(rows)


# -----------------------------
# Statistical tests
# -----------------------------


def run_two_group_tests(issue_df: pd.DataFrame) -> pd.DataFrame:
    if issue_df.empty or "analysis_set" not in issue_df.columns:
        return pd.DataFrame()
    a = issue_df[issue_df["analysis_set"] == "wontfix"].copy()
    b = issue_df[issue_df["analysis_set"] == "comparison"].copy()
    rows = []
    if len(a) < MIN_GROUP_N_FOR_TEST or len(b) < MIN_GROUP_N_FOR_TEST:
        return pd.DataFrame()

    for feature in [*PRIMARY_CONTINUOUS_FEATURES, *[f"repo_z_{f}" for f in PRIMARY_CONTINUOUS_FEATURES if f"repo_z_{f}" in issue_df.columns]]:
        if feature not in issue_df.columns:
            continue
        x = pd.to_numeric(a[feature], errors="coerce").dropna()
        y = pd.to_numeric(b[feature], errors="coerce").dropna()
        if len(x) < MIN_GROUP_N_FOR_TEST or len(y) < MIN_GROUP_N_FOR_TEST:
            continue
        welch = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
        try:
            mwu = stats.mannwhitneyu(x, y, alternative="two-sided")
            mwu_stat, mwu_p = float(mwu.statistic), float(mwu.pvalue)
        except Exception:
            mwu_stat, mwu_p = np.nan, np.nan
        rows.append({
            "feature": feature,
            "test_family": "two_group",
            "wontfix_n": len(x),
            "comparison_n": len(y),
            "wontfix_mean": x.mean(),
            "comparison_mean": y.mean(),
            "mean_difference": x.mean() - y.mean(),
            "welch_t_stat": float(welch.statistic),
            "p_value": float(welch.pvalue),
            "mann_whitney_u": mwu_stat,
            "mann_whitney_p": mwu_p,
            "hedges_g": hedges_g(x, y),
            "cliffs_delta": cliffs_delta(x, y),
        })
    return benjamini_hochberg(pd.DataFrame(rows), p_col="p_value")


def run_multigroup_tests(issue_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if issue_df.empty or "comparison_group" not in issue_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    valid = issue_df[issue_df["comparison_group"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()

    omnibus_rows = []
    pairwise_rows = []
    groups = sorted(valid["comparison_group"].astype(str).unique().tolist())
    if len(groups) < 3:
        return pd.DataFrame(), pd.DataFrame()

    for feature in PRIMARY_CONTINUOUS_FEATURES:
        if feature not in valid.columns:
            continue
        group_values = []
        for g in groups:
            vals = pd.to_numeric(valid.loc[valid["comparison_group"] == g, feature], errors="coerce").dropna()
            if len(vals) >= MIN_GROUP_N_FOR_TEST:
                group_values.append((g, vals))
        if len(group_values) < 3:
            continue

        try:
            kr = stats.kruskal(*[vals for _, vals in group_values])
            omnibus_rows.append({
                "feature": feature,
                "test_family": "multigroup",
                "test": "kruskal_wallis",
                "n_groups": len(group_values),
                "statistic": float(kr.statistic),
                "p_value": float(kr.pvalue),
            })
        except Exception:
            pass

        try:
            fit_df = valid[[feature, "comparison_group"]].dropna().copy()
            fit_df = fit_df.rename(columns={feature: "y"})
            model = ols("y ~ C(comparison_group)", data=fit_df).fit()
            anova_df = anova_lm(model, typ=2)
            if "C(comparison_group)" in anova_df.index:
                row = anova_df.loc["C(comparison_group)"]
                omnibus_rows.append({
                    "feature": feature,
                    "test_family": "multigroup",
                    "test": "anova",
                    "n_groups": len(group_values),
                    "statistic": float(row["F"]),
                    "p_value": float(row["PR(>F)"]),
                })
        except Exception:
            pass

        # pairwise Welch t-tests with BH later
        for i, (g1, x) in enumerate(group_values):
            for g2, y in group_values[i + 1:]:
                tt = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
                pairwise_rows.append({
                    "feature": feature,
                    "group_1": g1,
                    "group_2": g2,
                    "group_1_n": len(x),
                    "group_2_n": len(y),
                    "group_1_mean": x.mean(),
                    "group_2_mean": y.mean(),
                    "mean_difference": x.mean() - y.mean(),
                    "p_value": float(tt.pvalue),
                    "hedges_g": hedges_g(x, y),
                    "cliffs_delta": cliffs_delta(x, y),
                })

    omnibus = benjamini_hochberg(pd.DataFrame(omnibus_rows), p_col="p_value")
    pairwise = benjamini_hochberg(pd.DataFrame(pairwise_rows), p_col="p_value")
    return omnibus, pairwise


def run_proportion_tests(issue_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if issue_df.empty or "analysis_set" not in issue_df.columns:
        return pd.DataFrame()
    indicators = [
        "has_strongly_negative_comment",
        "has_strongly_positive_comment",
        "late_more_negative_than_early",
        "high_sentiment_volatility",
    ]
    a = issue_df[issue_df["analysis_set"] == "wontfix"]
    b = issue_df[issue_df["analysis_set"] == "comparison"]
    for indicator in indicators:
        if indicator not in issue_df.columns:
            continue
        ax = a[indicator].dropna().astype(int)
        bx = b[indicator].dropna().astype(int)
        if len(ax) < MIN_GROUP_N_FOR_TEST or len(bx) < MIN_GROUP_N_FOR_TEST:
            continue
        table = np.array([
            [int(ax.sum()), int((1 - ax).sum())],
            [int(bx.sum()), int((1 - bx).sum())],
        ])
        try:
            if (table < 5).any():
                _, p_value = stats.fisher_exact(table)
                test = "fishers_exact"
                stat = np.nan
            else:
                chi2, p_value, _, _ = stats.chi2_contingency(table)
                test = "chi_square"
                stat = float(chi2)
        except Exception:
            continue
        odds_ratio = safe_divide(table[0, 0] * table[1, 1], table[0, 1] * table[1, 0], default=np.nan)
        rows.append({
            "indicator": indicator,
            "test_family": "proportion",
            "test": test,
            "statistic": stat,
            "p_value": p_value,
            "wontfix_rate": float(ax.mean()),
            "comparison_rate": float(bx.mean()),
            "odds_ratio": odds_ratio,
        })
    return benjamini_hochberg(pd.DataFrame(rows), p_col="p_value")


def run_early_late_within_group(issue_df: pd.DataFrame) -> pd.DataFrame:
    req = ["early_mean_comment_sentiment", "late_mean_comment_sentiment", "comparison_group"]
    if any(col not in issue_df.columns for col in req):
        return pd.DataFrame()
    rows = []
    for grp, g in issue_df.groupby("comparison_group"):
        pair = g[["early_mean_comment_sentiment", "late_mean_comment_sentiment"]].dropna()
        if len(pair) < MIN_GROUP_N_FOR_TEST:
            continue
        early = pair["early_mean_comment_sentiment"]
        late = pair["late_mean_comment_sentiment"]
        try:
            paired_t = stats.ttest_rel(late, early, nan_policy="omit")
        except Exception:
            paired_t = None
        try:
            wilcoxon = stats.wilcoxon(late, early)
        except Exception:
            wilcoxon = None
        rows.append({
            "comparison_group": grp,
            "n_pairs": len(pair),
            "early_mean": early.mean(),
            "late_mean": late.mean(),
            "late_minus_early_mean": (late - early).mean(),
            "paired_t_p": float(paired_t.pvalue) if paired_t is not None else np.nan,
            "paired_t_stat": float(paired_t.statistic) if paired_t is not None else np.nan,
            "wilcoxon_p": float(wilcoxon.pvalue) if wilcoxon is not None else np.nan,
            "wilcoxon_stat": float(wilcoxon.statistic) if wilcoxon is not None else np.nan,
        })
    return pd.DataFrame(rows)


def run_models(issue_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if issue_df.empty or "analysis_set" not in issue_df.columns:
        return pd.DataFrame(), pd.DataFrame()
    ols_rows = []
    logit_rows = []

    repo_ok = issue_df["repo_full_name"].nunique() >= 2 if "repo_full_name" in issue_df.columns else False
    controls = []
    if "issue_type" in issue_df.columns:
        controls.append("C(issue_type)")
    if "comment_count" in issue_df.columns:
        issue_df = issue_df.copy()
        issue_df["log_comment_count"] = np.log1p(pd.to_numeric(issue_df["comment_count"], errors="coerce"))
        controls.append("log_comment_count")
    if "unique_commenter_count" in issue_df.columns:
        controls.append("unique_commenter_count")
    if repo_ok:
        controls.append("C(repo_full_name)")

    rhs = "C(analysis_set)"
    if controls:
        rhs += " + " + " + ".join(controls)

    for outcome in [
        "mean_comment_sentiment",
        "std_comment_sentiment",
        "comment_sentiment_change_late_minus_early",
    ]:
        if outcome not in issue_df.columns:
            continue
        fit_df = issue_df[[outcome, "analysis_set", "repo_full_name", "issue_type", "comment_count", "unique_commenter_count"]].copy()
        fit_df = fit_df.rename(columns={outcome: "y"})
        fit_df = fit_df.dropna(subset=["y", "analysis_set"])
        if len(fit_df) < 20:
            continue
        try:
            model = ols(f"y ~ {rhs}", data=fit_df).fit(cov_type="HC3")
            for term, coef in model.params.items():
                ols_rows.append({
                    "outcome": outcome,
                    "term": term,
                    "coef": coef,
                    "std_err": model.bse.get(term, np.nan),
                    "t_value": model.tvalues.get(term, np.nan),
                    "p_value": model.pvalues.get(term, np.nan),
                    "ci_low": model.conf_int().loc[term, 0] if term in model.conf_int().index else np.nan,
                    "ci_high": model.conf_int().loc[term, 1] if term in model.conf_int().index else np.nan,
                    "n_obs": int(model.nobs),
                    "r_squared": model.rsquared,
                })
        except Exception:
            continue

    if "has_strongly_negative_comment" in issue_df.columns:
        fit_df = issue_df[["has_strongly_negative_comment", "analysis_set", "repo_full_name", "issue_type", "comment_count", "unique_commenter_count"]].copy()
        fit_df = fit_df.dropna(subset=["has_strongly_negative_comment", "analysis_set"])
        fit_df["has_strongly_negative_comment"] = fit_df["has_strongly_negative_comment"].astype(int)
        if fit_df["has_strongly_negative_comment"].nunique() == 2 and len(fit_df) >= 30:
            try:
                model = logit(f"has_strongly_negative_comment ~ {rhs}", data=fit_df).fit(disp=False)
                conf = model.conf_int()
                for term, coef in model.params.items():
                    logit_rows.append({
                        "outcome": "has_strongly_negative_comment",
                        "term": term,
                        "coef": coef,
                        "odds_ratio": np.exp(coef),
                        "std_err": model.bse.get(term, np.nan),
                        "z_value": model.tvalues.get(term, np.nan),
                        "p_value": model.pvalues.get(term, np.nan),
                        "ci_low_or": np.exp(conf.loc[term, 0]) if term in conf.index else np.nan,
                        "ci_high_or": np.exp(conf.loc[term, 1]) if term in conf.index else np.nan,
                        "n_obs": int(model.nobs),
                        "pseudo_r2": getattr(model, "prsquared", np.nan),
                    })
            except Exception:
                pass

    return benjamini_hochberg(pd.DataFrame(ols_rows), p_col="p_value"), benjamini_hochberg(pd.DataFrame(logit_rows), p_col="p_value")


# -----------------------------
# Plots
# -----------------------------


def canonical_group_name(value: object) -> str:
    if value is None:
        return "missing"
    text = str(value).strip().lower()
    if not text:
        return "missing"
    text = text.replace('-', '_').replace(' ', '_')
    return GROUP_ALIASES.get(text, text)


def get_group_color(value: object) -> str:
    key = canonical_group_name(value)
    return ANALYSIS_COLORS.get(key, UTK_COLORS["smokey_gray"])


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(UTK_COLORS["white"])
    ax.grid(axis="y", color=UTK_COLORS["light_gray"], alpha=0.45, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(UTK_COLORS["dark_gray"])
    ax.spines["bottom"].set_color(UTK_COLORS["dark_gray"])
    ax.tick_params(colors=UTK_COLORS["dark_gray"])
    ax.title.set_color(UTK_COLORS["dark_gray"])
    ax.xaxis.label.set_color(UTK_COLORS["dark_gray"])
    ax.yaxis.label.set_color(UTK_COLORS["dark_gray"])


def set_plot_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": UTK_COLORS["white"],
        "axes.facecolor": UTK_COLORS["white"],
        "savefig.facecolor": UTK_COLORS["white"],
        "axes.edgecolor": UTK_COLORS["dark_gray"],
        "axes.labelcolor": UTK_COLORS["dark_gray"],
        "text.color": UTK_COLORS["dark_gray"],
        "xtick.color": UTK_COLORS["dark_gray"],
        "ytick.color": UTK_COLORS["dark_gray"],
        "grid.color": UTK_COLORS["light_gray"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 10,
    })


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sample_counts(issue_df: pd.DataFrame, path: Path) -> None:
    counts = issue_df["comparison_group"].fillna("missing").value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = counts.index.astype(str).tolist()
    colors = [get_group_color(label) for label in labels]
    ax.bar(labels, counts.values, color=colors, edgecolor=UTK_COLORS["dark_gray"], linewidth=0.8)
    ax.set_title("Issue counts by analysis group")
    ax.set_ylabel("Number of issues")
    ax.set_xlabel("Group")
    ax.tick_params(axis="x", rotation=30)
    style_axes(ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=9, color=UTK_COLORS["dark_gray"])
    save_figure(fig, path)


def _plot_violin_box_scatter_panel(
    ax: plt.Axes,
    panel_df: pd.DataFrame,
    feature: str,
    groups: list[str],
    panel_label: str,
    *,
    sample_cap: int = 120,
    annotate_counts: bool = False,
    y_limits: tuple[float, float] | None = None,
    show_box: bool = True,
    gap: float = 0.95,
    violin_width: float = 0.62,
    scatter_alpha: float = 0.10,
) -> None:
    data = [
        panel_df.loc[panel_df["comparison_group"] == g, feature].astype(float).to_numpy()
        for g in groups
    ]
    positions = 1.0 + np.arange(len(groups)) * gap

    vp = ax.violinplot(
        data,
        positions=positions,
        widths=violin_width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for group, body in zip(groups, vp["bodies"]):
        color = get_group_color(group)
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.18)
        body.set_linewidth(0.9)

    if show_box:
        box = ax.boxplot(
            data,
            positions=positions,
            widths=min(0.16, violin_width * 0.32),
            patch_artist=True,
            showfliers=False,
            medianprops={"color": UTK_COLORS["dark_gray"], "linewidth": 1.35},
            whiskerprops={"color": UTK_COLORS["dark_gray"], "linewidth": 0.9},
            capprops={"color": UTK_COLORS["dark_gray"], "linewidth": 0.9},
            boxprops={"edgecolor": UTK_COLORS["dark_gray"], "linewidth": 0.9},
        )
        for patch, group in zip(box["boxes"], groups):
            patch.set_facecolor(get_group_color(group))
            patch.set_alpha(0.38)

    for idx, values in enumerate(data, start=1):
        if len(values) == 0:
            continue
        rng = np.random.default_rng(4000 + idx)
        sample = values
        if len(sample) > sample_cap:
            sample = rng.choice(sample, sample_cap, replace=False)
        x = rng.normal(positions[idx - 1], 0.018, size=len(sample))
        ax.scatter(
            x,
            sample,
            s=7,
            alpha=scatter_alpha,
            color=get_group_color(groups[idx - 1]),
            edgecolors="none",
            zorder=3,
        )
        med = float(np.median(values))
        ax.scatter(positions[idx - 1], med, s=28, color=UTK_COLORS["dark_gray"], zorder=4)
        if annotate_counts:
            ax.text(
                positions[idx - 1],
                0.98,
                f"n={len(values)}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color=UTK_COLORS["dark_gray"],
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(groups, rotation=20)
    ax.set_ylabel(feature)
    ax.set_title(panel_label)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    left = positions[0] - violin_width * 0.8
    right = positions[-1] + violin_width * 0.8
    ax.set_xlim(left, right)
    style_axes(ax)


def _add_share_annotation(ax: plt.Axes, lines: list[str]) -> None:
    if not lines:
        return
    ax.text(
        0.99,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color=UTK_COLORS["dark_gray"],
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=UTK_COLORS["light_gray"], alpha=0.88),
    )


def plot_zero_inflated_distribution(
    issue_df: pd.DataFrame,
    feature: str,
    path: Path,
    title: str | None = None,
    *,
    nonzero_label: str = "Nonzero issues only",
    show_box_left: bool = False,
    show_box_right: bool = True,
    gap: float = 0.78,
) -> None:
    required_cols = ["comparison_group", feature]
    plot_df = issue_df[required_cols].dropna().copy()
    if plot_df.empty:
        return

    plot_df["comparison_group"] = plot_df["comparison_group"].astype(str)
    plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce")
    plot_df = plot_df.dropna(subset=[feature]).copy()
    if plot_df.empty:
        return

    groups = sorted(
        plot_df["comparison_group"].unique(),
        key=lambda x: (canonical_group_name(x) != "wontfix", str(x)),
    )

    nonzero_df = plot_df.loc[plot_df[feature] > 0].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), squeeze=False)
    axes = axes[0]

    _plot_violin_box_scatter_panel(
        axes[0],
        plot_df,
        feature,
        groups,
        panel_label="All issues",
        sample_cap=120,
        show_box=show_box_left,
        gap=gap,
        violin_width=0.54,
        scatter_alpha=0.08,
    )
    lines = []
    for g in groups:
        gvals = plot_df.loc[plot_df["comparison_group"] == g, feature]
        if len(gvals) == 0:
            continue
        lines.append(f"{g}: {(gvals <= 0).mean():.1%} zero")
    _add_share_annotation(axes[0], lines)

    if nonzero_df.empty:
        axes[1].axis("off")
    else:
        upper = float(nonzero_df[feature].quantile(0.98))
        upper = upper if upper > 0 else float(nonzero_df[feature].max())
        _plot_violin_box_scatter_panel(
            axes[1],
            nonzero_df,
            feature,
            groups,
            panel_label=nonzero_label,
            sample_cap=140,
            annotate_counts=True,
            y_limits=(0.0, upper * 1.08 if upper > 0 else None),
            show_box=show_box_right,
            gap=gap,
            violin_width=0.54,
            scatter_alpha=0.10,
        )

    fig.suptitle(title or f"Distribution of {feature}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, path)


def plot_signed_sentiment_distributions(
    issue_df: pd.DataFrame,
    feature: str,
    overall_path: Path,
    split_path: Path,
    title: str,
) -> None:
    required_cols = ["comparison_group", feature]
    plot_df = issue_df[required_cols].dropna().copy()
    if plot_df.empty:
        return

    plot_df["comparison_group"] = plot_df["comparison_group"].astype(str)
    plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce")
    plot_df = plot_df.dropna(subset=[feature]).copy()
    if plot_df.empty:
        return

    groups = sorted(
        plot_df["comparison_group"].unique(),
        key=lambda x: (canonical_group_name(x) != "wontfix", str(x)),
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    _plot_violin_box_scatter_panel(
        ax,
        plot_df,
        feature,
        groups,
        panel_label="All issues",
        sample_cap=140,
    )
    summary_lines = []
    for g in groups:
        gvals = plot_df.loc[plot_df["comparison_group"] == g, feature]
        if len(gvals) == 0:
            continue
        zero_share = float((gvals == 0).mean())
        pos_share = float((gvals > 0).mean())
        neg_share = float((gvals < 0).mean())
        summary_lines.append(f"{g}: {zero_share:.1%} zero | {pos_share:.1%} > 0 | {neg_share:.1%} < 0")
    _add_share_annotation(ax, summary_lines)
    ax.set_ylabel(feature)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, overall_path)

    positive_df = plot_df.loc[plot_df[feature] > 0].copy()
    negative_df = plot_df.loc[plot_df[feature] < 0].copy()
    if positive_df.empty and negative_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), squeeze=False)
    axes = axes[0]

    if negative_df.empty:
        axes[0].axis("off")
    else:
        neg_lo = float(negative_df[feature].quantile(0.02))
        _plot_violin_box_scatter_panel(
            axes[0],
            negative_df,
            feature,
            groups,
            panel_label=f"Negative-only {feature} (< 0)",
            sample_cap=120,
            annotate_counts=True,
            y_limits=(neg_lo * 1.05 if neg_lo < 0 else -0.05, 0.0),
        )

    if positive_df.empty:
        axes[1].axis("off")
    else:
        pos_hi = float(positive_df[feature].quantile(0.98))
        _plot_violin_box_scatter_panel(
            axes[1],
            positive_df,
            feature,
            groups,
            panel_label=f"Positive-only {feature} (> 0)",
            sample_cap=120,
            annotate_counts=True,
            y_limits=(0.0, pos_hi * 1.05 if pos_hi > 0 else 0.05),
        )

    fig.suptitle(f"{title} — signed nonzero subsets", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, split_path)


def plot_mean_sentiment_distributions(
    issue_df: pd.DataFrame,
    overall_path: Path,
    split_path: Path,
    title: str = "Distribution of mean issue sentiment",
) -> None:
    return plot_signed_sentiment_distributions(
        issue_df,
        "mean_comment_sentiment",
        overall_path,
        split_path,
        title,
    )


def plot_distribution(
    issue_df: pd.DataFrame,
    feature: str,
    path: Path,
    title: str | None = None,
) -> None:
    """Generic cleaner distribution plot used for most features."""
    if feature == "std_comment_sentiment":
        return plot_zero_inflated_distribution(
            issue_df,
            feature,
            path,
            title,
            nonzero_label="Nonzero issues only (std_comment_sentiment > 0)",
            show_box_left=False,
            show_box_right=True,
            gap=0.72,
        )

    required_cols = ["comparison_group", feature]
    plot_df = issue_df[required_cols].dropna().copy()
    if plot_df.empty:
        return

    plot_df["comparison_group"] = plot_df["comparison_group"].astype(str)
    plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce")
    plot_df = plot_df.dropna(subset=[feature]).copy()
    if plot_df.empty:
        return

    groups = sorted(
        plot_df["comparison_group"].unique(),
        key=lambda x: (canonical_group_name(x) != "wontfix", str(x)),
    )

    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    _plot_violin_box_scatter_panel(
        ax,
        plot_df,
        feature,
        groups,
        panel_label="All issues",
        sample_cap=140,
    )
    fig.suptitle(title or f"Distribution of {feature}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, path)


def plot_nonnegative_distribution_pair(
    issue_df: pd.DataFrame,
    feature: str,
    overall_path: Path,
    split_path: Path,
    title: str,
) -> None:
    required_cols = ["comparison_group", feature]
    plot_df = issue_df[required_cols].dropna().copy()
    if plot_df.empty:
        return
    plot_df["comparison_group"] = plot_df["comparison_group"].astype(str)
    plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce")
    plot_df = plot_df.dropna(subset=[feature]).copy()
    if plot_df.empty:
        return

    groups = sorted(
        plot_df["comparison_group"].unique(),
        key=lambda x: (canonical_group_name(x) != "wontfix", str(x)),
    )

    fig, ax = plt.subplots(figsize=(9.0, 5.7))
    _plot_violin_box_scatter_panel(ax, plot_df, feature, groups, panel_label="All issues", sample_cap=140)
    lines = []
    for g in groups:
        gvals = plot_df.loc[plot_df["comparison_group"] == g, feature]
        if len(gvals) == 0:
            continue
        lines.append(f"{g}: {(gvals == 0).mean():.1%} zero | {(gvals > 0).mean():.1%} > 0")
    _add_share_annotation(ax, lines)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, overall_path)

    nonzero_df = plot_df.loc[plot_df[feature] > 0].copy()
    if nonzero_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    upper = float(nonzero_df[feature].quantile(0.98))
    _plot_violin_box_scatter_panel(
        ax, nonzero_df, feature, groups, panel_label=f"Positive-only {feature} (> 0)",
        sample_cap=120, annotate_counts=True, y_limits=(0.0, upper * 1.05 if upper > 0 else 0.05)
    )
    fig.suptitle(f"{title} — nonzero subset", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, split_path)


def plot_early_late_dumbbell(issue_df: pd.DataFrame, path: Path) -> None:
    req = ["comparison_group", "early_mean_comment_sentiment", "late_mean_comment_sentiment"]
    if any(c not in issue_df.columns for c in req):
        return
    g = issue_df.groupby("comparison_group", dropna=False)[["early_mean_comment_sentiment", "late_mean_comment_sentiment"]].mean(numeric_only=True).reset_index()
    if g.empty:
        return
    g = g.sort_values("late_mean_comment_sentiment")
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(g))
    for idx, row in g.iterrows():
        color = get_group_color(row["comparison_group"])
        ax.hlines(y[idx], row["early_mean_comment_sentiment"], row["late_mean_comment_sentiment"], linewidth=2.5, color=color, alpha=0.9)
        ax.scatter(row["early_mean_comment_sentiment"], y[idx], s=60, color=UTK_COLORS["smokey_gray"], label="Early" if idx == 0 else None, zorder=3)
        ax.scatter(row["late_mean_comment_sentiment"], y[idx], s=60, color=color, label="Late" if idx == 0 else None, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(g["comparison_group"].astype(str))
    ax.set_xlabel("Mean sentiment")
    ax.set_title("Early vs late issue discussion sentiment")
    style_axes(ax)
    ax.legend(frameon=False)
    save_figure(fig, path)


def build_comment_trajectory(comment_df: pd.DataFrame, issue_df: pd.DataFrame, bins: int = 5, min_comments: int = 3) -> pd.DataFrame:
    needed = ["repo_full_name", "issue_id", "comparison_group", "sentiment_compound"]
    if any(c not in comment_df.columns for c in ["repo_full_name", "issue_id", "sentiment_compound"]):
        return pd.DataFrame()
    merge_cols = [c for c in ["repo_full_name", "issue_id", "comparison_group"] if c in issue_df.columns]
    if len(merge_cols) < 3:
        return pd.DataFrame()
    cdf = comment_df.merge(issue_df[merge_cols].drop_duplicates(), on=["repo_full_name", "issue_id"], how="inner")
    seq_col = "comment_sequence_index" if "comment_sequence_index" in cdf.columns else None
    if seq_col is None:
        if "created_at" in cdf.columns:
            cdf = cdf.sort_values(["repo_full_name", "issue_id", "created_at", "comment_id"], kind="stable")
        else:
            cdf = cdf.sort_values(["repo_full_name", "issue_id", "comment_id"], kind="stable")
        cdf["comment_sequence_index"] = cdf.groupby(["repo_full_name", "issue_id"]).cumcount() + 1
        seq_col = "comment_sequence_index"
    counts = cdf.groupby(["repo_full_name", "issue_id"]).size().rename("n_comments").reset_index()
    cdf = cdf.merge(counts, on=["repo_full_name", "issue_id"], how="left")
    cdf = cdf[cdf["n_comments"] >= min_comments].copy()
    if cdf.empty:
        return pd.DataFrame()
    cdf["position_fraction"] = (cdf[seq_col] - 1) / (cdf["n_comments"] - 1)
    cdf["trajectory_bin"] = np.minimum((cdf["position_fraction"] * bins).astype(int) + 1, bins)
    out = cdf.groupby(["comparison_group", "trajectory_bin"], dropna=False)["sentiment_compound"].agg(["mean", "count", "std"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    out["ci_low"] = out["mean"] - 1.96 * out["se"]
    out["ci_high"] = out["mean"] + 1.96 * out["se"]
    return out


def plot_comment_trajectory(trajectory_df: pd.DataFrame, path: Path) -> None:
    if trajectory_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for group, g in trajectory_df.groupby("comparison_group"):
        g = g.sort_values("trajectory_bin")
        x = g["trajectory_bin"].to_numpy(dtype=float)
        y = g["mean"].to_numpy(dtype=float)
        low = g["ci_low"].to_numpy(dtype=float)
        high = g["ci_high"].to_numpy(dtype=float)
        color = get_group_color(group)
        ax.plot(x, y, marker="o", label=str(group), color=color, linewidth=2.5)
        ax.fill_between(x, low, high, alpha=0.14, color=color)
    ax.set_xlabel("Normalized comment-position bin")
    ax.set_ylabel("Mean comment sentiment")
    ax.set_title("Comment-position sentiment trajectory")
    style_axes(ax)
    ax.legend(loc="best", fontsize=8, frameon=False)
    save_figure(fig, path)


def _filter_primary_groups(plot_df: pd.DataFrame) -> pd.DataFrame:
    plot_df = plot_df.copy()
    plot_df["comparison_group"] = plot_df["comparison_group"].map(canonical_group_name)
    wanted = [group for group in ["wontfix", "comparison"] if group in plot_df["comparison_group"].unique()]
    if not wanted:
        wanted = sorted(plot_df["comparison_group"].dropna().unique().tolist())[:2]
    return plot_df[plot_df["comparison_group"].isin(wanted)].copy()


def _compute_plot_limits(plot_df: pd.DataFrame, xcol: str, ycol: str) -> tuple[tuple[float, float], tuple[float, float]]:
    x = pd.to_numeric(plot_df[xcol], errors="coerce").dropna()
    y = pd.to_numeric(plot_df[ycol], errors="coerce").dropna()
    if x.empty or y.empty:
        return (0.0, 1.0), (0.0, 1.0)

    x_hi = float(x.quantile(0.99))
    x_max = float(x.max())
    x_upper = max(x_hi, min(x_max, x_hi * 1.15 if x_hi > 0 else x_max))
    if x_upper <= 0:
        x_upper = max(1.0, x_max)
    if float(x_max) > x_upper:
        x_upper = float(x_max)

    y_lo = float(y.quantile(0.01))
    y_hi = float(y.quantile(0.99))
    y_pad = max((y_hi - y_lo) * 0.08, 0.03)
    return (0.0, x_upper), (y_lo - y_pad, y_hi + y_pad)


def _build_count_bin_spec(xcol: str) -> tuple[list[float], list[str]]:
    if xcol == "comment_count":
        return (
            [-0.1, 0.5, 1.5, 2.5, 3.5, 5.5, 8.5, 12.5, 20.5, np.inf],
            ["0", "1", "2", "3", "4–5", "6–8", "9–12", "13–20", "21+"],
        )
    if xcol == "unique_commenter_count":
        return (
            [-0.1, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7.5, 10.5, np.inf],
            ["0", "1", "2", "3", "4", "5", "6–7", "8–10", "11+"],
        )
    return (
        [-0.1, 0.5, 1.5, 2.5, 4.5, 7.5, 12.5, np.inf],
        ["0", "1", "2", "3–4", "5–7", "8–12", "13+"],
    )


def get_hexbin_cmap(group: object):
    key = canonical_group_name(group)
    if key == "wontfix":
        return UTK_ORANGE_HEXBIN_CMAP
    if key == "comparison":
        return UTK_GRAY_HEXBIN_CMAP
    return UTK_ORANGE_HEXBIN_CMAP


def plot_hexbin_facet(issue_df: pd.DataFrame, xcol: str, ycol: str, path: Path, title: str | None = None) -> None:
    if xcol not in issue_df.columns or ycol not in issue_df.columns or "comparison_group" not in issue_df.columns:
        return
    plot_df = issue_df[[xcol, ycol, "comparison_group"]].dropna().copy()
    plot_df = _filter_primary_groups(plot_df)
    if plot_df.empty:
        return

    groups = [group for group in ["wontfix", "comparison"] if group in plot_df["comparison_group"].unique()]
    if not groups:
        groups = sorted(plot_df["comparison_group"].unique().tolist())

    (xlo, xhi), (ylo, yhi) = _compute_plot_limits(plot_df, xcol, ycol)

    fig, axes = plt.subplots(1, len(groups), figsize=(8.3 * len(groups), 5.2), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, group in zip(axes, groups):
        g = plot_df[plot_df["comparison_group"] == group].copy()
        if g.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        hb = ax.hexbin(
            pd.to_numeric(g[xcol], errors="coerce"),
            pd.to_numeric(g[ycol], errors="coerce"),
            gridsize=26,
            mincnt=1,
            bins="log",
            extent=(xlo, xhi, ylo, yhi),
            linewidths=0.0,
            cmap=get_hexbin_cmap(group),
        )
        ax.set_title(group)
        ax.set_xlabel(xcol.replace("_", " "))
        style_axes(ax)
        ax.grid(False)
        ax.text(
            0.98,
            0.97,
            f"n={len(g):,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color=UTK_COLORS["dark_gray"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=UTK_COLORS["light_gray"], alpha=0.85),
        )
        cbar = fig.colorbar(hb, ax=ax, shrink=0.9, pad=0.02)
        cbar.set_label("Issue density (log bins)")
        cbar.ax.tick_params(colors=UTK_COLORS["dark_gray"], labelsize=8)
        cbar.outline.set_edgecolor(UTK_COLORS["light_gray"])

    axes[0].set_ylabel(ycol.replace("_", " "))
    fig.suptitle(title or f"{ycol} vs {xcol}")
    save_figure(fig, path)


def plot_binned_trend(issue_df: pd.DataFrame, xcol: str, ycol: str, path: Path, title: str | None = None) -> None:
    if xcol not in issue_df.columns or ycol not in issue_df.columns or "comparison_group" not in issue_df.columns:
        return
    plot_df = issue_df[[xcol, ycol, "comparison_group"]].dropna().copy()
    plot_df = _filter_primary_groups(plot_df)
    if plot_df.empty:
        return

    bin_edges, bin_labels = _build_count_bin_spec(xcol)
    plot_df = plot_df.copy()
    plot_df["x_bin"] = pd.cut(pd.to_numeric(plot_df[xcol], errors="coerce"), bins=bin_edges, labels=bin_labels, include_lowest=True, ordered=True)
    plot_df = plot_df[plot_df["x_bin"].notna()].copy()
    if plot_df.empty:
        return

    summary = (
        plot_df.groupby(["comparison_group", "x_bin"], observed=True)[ycol]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    if summary.empty:
        return
    summary["se"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    summary["ci_low"] = summary["mean"] - 1.96 * summary["se"]
    summary["ci_high"] = summary["mean"] + 1.96 * summary["se"]

    groups = [group for group in ["wontfix", "comparison"] if group in summary["comparison_group"].unique()]
    if not groups:
        groups = sorted(summary["comparison_group"].unique().tolist())

    fig, axes = plt.subplots(1, len(groups), figsize=(7.8 * len(groups), 5.2), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax, group in zip(axes, groups):
        g = summary[summary["comparison_group"] == group].copy()
        if g.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        order = [label for label in bin_labels if label in g["x_bin"].astype(str).tolist()]
        g["x_bin"] = pd.Categorical(g["x_bin"].astype(str), categories=order, ordered=True)
        g = g.sort_values("x_bin")
        xpos = np.arange(len(g))
        color = get_group_color(group)
        ax.plot(xpos, g["mean"], marker="o", linewidth=2.2, color=color)
        ax.fill_between(xpos, g["ci_low"], g["ci_high"], color=color, alpha=0.14)
        ax.errorbar(xpos, g["mean"], yerr=1.96 * g["se"], fmt="none", ecolor=color, alpha=0.7, capsize=3, linewidth=1)
        ax.set_xticks(xpos)
        ax.set_xticklabels(g["x_bin"].astype(str), rotation=25)
        ax.set_title(group)
        ax.set_xlabel(xcol.replace("_", " ") + " (binned)")
        style_axes(ax)
        for xi, (_, row) in zip(xpos, g.iterrows()):
            ax.text(xi, row["mean"], f"n={int(row['count'])}", ha="center", va="bottom", fontsize=7.5, color=UTK_COLORS["dark_gray"])

    axes[0].set_ylabel(ycol.replace("_", " "))
    fig.suptitle(title or f"Binned trend: {ycol} vs {xcol}")
    save_figure(fig, path)


def _build_repo_forest_df(issue_df: pd.DataFrame, feature: str) -> pd.DataFrame:
    if feature not in issue_df.columns or "analysis_set" not in issue_df.columns:
        return pd.DataFrame()
    rows = []
    for repo, g in issue_df.groupby("repo_full_name"):
        a = pd.to_numeric(g.loc[g["analysis_set"] == "wontfix", feature], errors="coerce").dropna()
        b = pd.to_numeric(g.loc[g["analysis_set"] == "comparison", feature], errors="coerce").dropna()
        if len(a) < MIN_GROUP_N_FOR_TEST or len(b) < MIN_GROUP_N_FOR_TEST:
            continue
        diff = a.mean() - b.mean()
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if len(a) > 1 and len(b) > 1 else np.nan
        rows.append({
            "repo_full_name": repo,
            "effect": diff,
            "ci_low": diff - 1.96 * se if pd.notna(se) else np.nan,
            "ci_high": diff + 1.96 * se if pd.notna(se) else np.nan,
            "n": len(a) + len(b),
        })
    return pd.DataFrame(rows).sort_values("effect").reset_index(drop=True) if rows else pd.DataFrame()


def plot_repo_forest_panel(issue_df: pd.DataFrame, path: Path) -> None:
    feature_specs = [
        ("mean_comment_sentiment", "Mean"),
        ("max_comment_sentiment", "Max"),
        ("min_comment_sentiment", "Min"),
        ("comment_sentiment_range", "Range"),
        ("std_comment_sentiment", "Std"),
        ("negative_comment_share", "Negative Share"),
    ]

    forests = []
    for feature, label in feature_specs:
        forest = _build_repo_forest_df(issue_df, feature)
        if not forest.empty:
            forests.append((feature, label, forest))

    if not forests:
        return

    finite_bounds = []
    for _, _, forest in forests:
        for col in ["ci_low", "ci_high", "effect"]:
            vals = pd.to_numeric(forest[col], errors="coerce")
            vals = vals[np.isfinite(vals)]
            if len(vals):
                finite_bounds.extend(vals.tolist())
    if not finite_bounds:
        return

    x_min = min(finite_bounds)
    x_max = max(finite_bounds)
    span = max(abs(x_min), abs(x_max))
    x_lim = (-1.08 * span, 1.08 * span) if span > 0 else (-0.05, 0.05)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    for ax, (feature, label, forest) in zip(axes, forests):
        y = np.arange(len(forest))
        ax.hlines(y, forest["ci_low"], forest["ci_high"], linewidth=2.0, color=UTK_COLORS["smokey_gray"], alpha=0.95)
        ax.scatter(
            forest["effect"],
            y,
            s=34,
            color=UTK_COLORS["orange"],
            edgecolors=UTK_COLORS["dark_gray"],
            linewidth=0.5,
            zorder=3,
        )
        ax.axvline(0, linestyle="--", linewidth=1, color=UTK_COLORS["dark_gray"])
        ax.set_title(label)
        ax.set_xlim(*x_lim)
        ax.set_yticks(y)
        ax.set_yticklabels(forest["repo_full_name"].astype(str), fontsize=8)
        style_axes(ax)

    for ax in axes[len(forests):]:
        ax.axis("off")

    axes[0].set_ylabel("Repository")
    axes[3].set_ylabel("Repository")
    axes[3].set_xlabel("WONTFIX - comparison difference")
    axes[4].set_xlabel("WONTFIX - comparison difference")
    axes[5].set_xlabel("WONTFIX - comparison difference")
    fig.suptitle("Per-repo effect estimates across sentiment features", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, path)


def plot_correlation_heatmap(issue_df: pd.DataFrame, path: Path) -> None:
    cols = [c for c in [*PRIMARY_CONTINUOUS_FEATURES, *PARTICIPATION_FEATURES] if c in issue_df.columns]
    if len(cols) < 2:
        return
    corr = issue_df[cols].apply(pd.to_numeric, errors="coerce").corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 3, 0.7 * len(cols) + 2))
    im = ax.imshow(corr, aspect="auto", cmap="Oranges", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=90, fontsize=8)
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols, fontsize=8)
    ax.set_title("Correlation heatmap: sentiment and participation features")
    style_axes(ax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, path)


# -----------------------------
# Report generation
# -----------------------------


def df_to_markdown(df: pd.DataFrame, max_rows: int = 20, float_precision: int = 4) -> str:
    if df.empty:
        return "_No rows available._"
    sample = df.head(max_rows).copy()
    for col in sample.columns:
        if pd.api.types.is_float_dtype(sample[col]):
            sample[col] = sample[col].map(lambda x: None if pd.isna(x) else round(float(x), float_precision))
    return sample.to_markdown(index=False)


def summarize_findings(two_group: pd.DataFrame, ols_df: pd.DataFrame) -> str:
    lines = []
    if not two_group.empty:
        sig = two_group.sort_values(["reject_fdr_bh_05", "p_value_fdr_bh", "p_value"], ascending=[False, True, True])
        top = sig.head(5)
        for _, row in top.iterrows():
            direction = "higher" if row.get("mean_difference", 0) > 0 else "lower"
            lines.append(
                f"- `{row['feature']}` was {direction} in WONTFIX than comparison issues "
                f"(Δ={row.get('mean_difference', np.nan):.4f}, Hedges g={row.get('hedges_g', np.nan):.4f}, "
                f"BH-adjusted p={row.get('p_value_fdr_bh', np.nan):.4g})."
            )
    if not ols_df.empty:
        wf_terms = ols_df[ols_df["term"].astype(str).str.contains("analysis_set", case=False, na=False)].copy()
        if not wf_terms.empty:
            lines.append("- Adjusted models were also fit with repository fixed effects when possible.")
            best = wf_terms.sort_values("p_value").head(3)
            for _, row in best.iterrows():
                lines.append(
                    f"  - Outcome `{row['outcome']}`: coefficient for `{row['term']}` = {row['coef']:.4f} "
                    f"(95% CI {row['ci_low']:.4f} to {row['ci_high']:.4f}, p={row['p_value']:.4g})."
                )
    if not lines:
        return "- No strong inferential summary could be generated from the available data."
    return "\n".join(lines)


def write_report(
    path: Path,
    issue_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    descriptives: pd.DataFrame,
    two_group: pd.DataFrame,
    omnibus: pd.DataFrame,
    pairwise: pd.DataFrame,
    proportion: pd.DataFrame,
    early_late: pd.DataFrame,
    ols_df: pd.DataFrame,
    logit_df: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    issue_count = len(issue_df)
    repo_count = issue_df["repo_full_name"].nunique() if not issue_df.empty else 0
    groups = sorted(issue_df["comparison_group"].dropna().astype(str).unique().tolist()) if "comparison_group" in issue_df.columns else []

    text = f"""
# Sentiment Analysis Report

## Overview

This report summarizes an issue-level sentiment analysis of WONTFIX discussions versus comparison issues.
The primary inferential unit is the **issue**, while comment-level records are used mainly for temporal
trajectory analysis and descriptive summaries.

- Issues analyzed: **{issue_count}**
- Repositories represented: **{repo_count}**
- Groups present: **{', '.join(groups) if groups else 'n/a'}**

## QA / coverage summary

{df_to_markdown(qa_df, max_rows=100)}

## Group descriptives

{df_to_markdown(descriptives, max_rows=20)}

## Headline findings

{summarize_findings(two_group, ols_df)}

## Two-group tests: WONTFIX vs comparison

{df_to_markdown(two_group, max_rows=20)}

## Multi-group omnibus tests

{df_to_markdown(omnibus, max_rows=20)}

## Multi-group pairwise tests

{df_to_markdown(pairwise, max_rows=20)}

## Proportion / prevalence tests

{df_to_markdown(proportion, max_rows=20)}

## Early-vs-late within-group tests

{df_to_markdown(early_late, max_rows=20)}

## Adjusted OLS models

{df_to_markdown(ols_df, max_rows=25)}

## Adjusted logistic models

{df_to_markdown(logit_df, max_rows=25)}

## Figures generated

"""
    for fig in figure_paths:
        text += f"- `{fig.name}`\n"

    text += textwrap.dedent(
        """

## Interpretation guardrails

- These analyses use sentiment features derived from issue and comment text. They are useful for comparative
  discussion-tone analysis, but they are not the same thing as intent, civility, or maintainer motivation.
- Repository baselines differ, so raw and within-repository-standardized analyses should be interpreted together.
- Comment-level records are not treated as independent observations in the main inferential tests.
- If certain optional upstream files were unavailable, subgroup or issue-type enrichment may be partial.
- Statistical significance should be read alongside effect sizes and confidence intervals.
"""
    )

    path.write_text(text.strip() + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    args = parse_args()
    set_plot_style()
    paths = Paths(
        rq1_dataset=Path(args.rq1_dataset),
        comment_features=Path(args.comment_features) if args.comment_features else None,
        output_dir=Path(args.output_dir),
    )

    ensure_dir(paths.output_dir)
    tables_dir = ensure_dir(paths.output_dir / "tables")
    figures_dir = ensure_dir(paths.output_dir / "figures")

    if not paths.rq1_dataset.exists():
        raise FileNotFoundError(f"RQ1 dataset file not found: {paths.rq1_dataset}")
    if paths.comment_features is not None and not paths.comment_features.exists():
        paths.comment_features = None

    issue_df, comment_df, qa_seed = load_and_prepare(
        paths,
        winsorize=args.winsorize,
        exclude_zero_text_issues=args.exclude_zero_text_issues,
    )

    # Core tables
    qa_df = build_qa_summary(issue_df, comment_df, qa_seed)
    descriptives = build_group_descriptives(issue_df)
    two_group = run_two_group_tests(issue_df)
    omnibus, pairwise = run_multigroup_tests(issue_df)
    proportion = run_proportion_tests(issue_df)
    early_late = run_early_late_within_group(issue_df)
    ols_df, logit_df = run_models(issue_df)

    # Persist core analysis tables
    issue_df.to_parquet(tables_dir / "issue_analysis_dataset.parquet", index=False)
    comment_df.to_parquet(tables_dir / "comment_analysis_dataset.parquet", index=False)
    qa_df.to_csv(tables_dir / "qa_summary.csv", index=False)
    descriptives.to_csv(tables_dir / "group_descriptives.csv", index=False)
    two_group.to_csv(tables_dir / "two_group_tests.csv", index=False)
    omnibus.to_csv(tables_dir / "multigroup_omnibus_tests.csv", index=False)
    pairwise.to_csv(tables_dir / "multigroup_pairwise_tests.csv", index=False)
    proportion.to_csv(tables_dir / "proportion_tests.csv", index=False)
    early_late.to_csv(tables_dir / "early_late_tests.csv", index=False)
    ols_df.to_csv(tables_dir / "ols_models.csv", index=False)
    logit_df.to_csv(tables_dir / "logit_models.csv", index=False)

    # Plots
    figure_paths: list[Path] = []
    plot_jobs = [
        (plot_sample_counts, (issue_df, figures_dir / "01_issue_counts_by_group.png")),
        (plot_mean_sentiment_distributions, (issue_df, figures_dir / "02_mean_comment_sentiment_distribution.png", figures_dir / "02b_mean_comment_sentiment_signed_distribution.png", "Distribution of mean issue sentiment")),
        (plot_distribution, (issue_df, "std_comment_sentiment", figures_dir / "03_sentiment_volatility_distribution.png", "Distribution of sentiment volatility")),
        (plot_signed_sentiment_distributions, (issue_df, "min_comment_sentiment", figures_dir / "04_min_comment_sentiment_distribution.png", figures_dir / "04b_min_comment_sentiment_signed_distribution.png", "Distribution of minimum comment sentiment")),
        (plot_signed_sentiment_distributions, (issue_df, "max_comment_sentiment", figures_dir / "04c_max_comment_sentiment_distribution.png", figures_dir / "04d_max_comment_sentiment_signed_distribution.png", "Distribution of maximum comment sentiment")),
        (plot_nonnegative_distribution_pair, (issue_df, "comment_sentiment_range", figures_dir / "04e_comment_sentiment_range_distribution.png", figures_dir / "04f_comment_sentiment_range_nonzero_distribution.png", "Distribution of sentiment range (max - min)")),
        (plot_early_late_dumbbell, (issue_df, figures_dir / "05_early_vs_late_sentiment.png")),
        (plot_hexbin_facet, (issue_df, "comment_count", "mean_comment_sentiment", figures_dir / "06_mean_sentiment_vs_comment_count.png", "Mean issue sentiment vs comment count")),
        (plot_binned_trend, (issue_df, "comment_count", "mean_comment_sentiment", figures_dir / "06b_mean_sentiment_vs_comment_count_binned_trend.png", "Binned trend: mean issue sentiment vs comment count")),
        (plot_hexbin_facet, (issue_df, "unique_commenter_count", "std_comment_sentiment", figures_dir / "07_volatility_vs_unique_commenters.png", "Sentiment volatility vs unique commenters")),
        (plot_binned_trend, (issue_df, "unique_commenter_count", "std_comment_sentiment", figures_dir / "07b_volatility_vs_unique_commenters_binned_trend.png", "Binned trend: sentiment volatility vs unique commenters")),
        (plot_repo_forest_panel, (issue_df, figures_dir / "08_repo_forest_effects_panel.png")),
        (plot_correlation_heatmap, (issue_df, figures_dir / "09_feature_correlation_heatmap.png")),
    ]

    for func, params in plot_jobs:
        try:
            func(*params)
            for value in params:
                if isinstance(value, Path) and value.exists() and value not in figure_paths:
                    figure_paths.append(value)
        except Exception:
            # keep the script robust: do not fail the full run because a plot couldn't be rendered
            pass

    trajectory_df = build_comment_trajectory(
        comment_df,
        issue_df,
        bins=args.comment_trajectory_bins,
        min_comments=args.min_comments_for_trajectory,
    )
    trajectory_df.to_csv(tables_dir / "comment_trajectory_summary.csv", index=False)
    trajectory_fig = figures_dir / "10_comment_trajectory.png"
    try:
        plot_comment_trajectory(trajectory_df, trajectory_fig)
        if trajectory_fig.exists():
            figure_paths.append(trajectory_fig)
    except Exception:
        pass

    report_path = paths.output_dir / "sentiment_analysis_report.md"
    write_report(
        report_path,
        issue_df,
        qa_df,
        descriptives,
        two_group,
        omnibus,
        pairwise,
        proportion,
        early_late,
        ols_df,
        logit_df,
        figure_paths,
    )

    manifest = {
        "rq1_dataset": str(paths.rq1_dataset),
        "comment_features": str(paths.comment_features) if paths.comment_features is not None else None,
        "output_dir": str(paths.output_dir),
        "n_issue_rows": int(len(issue_df)),
        "n_comment_rows": int(len(comment_df)),
        "figure_files": [str(p) for p in figure_paths],
        "report_path": str(report_path),
        "tables_dir": str(tables_dir),
    }
    (paths.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote sentiment analysis outputs to: {paths.output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
