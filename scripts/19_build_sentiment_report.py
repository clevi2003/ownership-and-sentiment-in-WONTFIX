#!/usr/bin/env python3
"""
Generate a Markdown analysis report, CSV summary tables, and graphs from the
sentiment and emotion feature outputs.

Default output folder:
    /outputs/sentiment_analysis

Expected input files from your pipeline:
    data/features/emotion_features.parquet
    data/features/sentiment/issue_sentiment_features.parquet
    data/features/sentiment/comment_sentiment_features.parquet

Example:
    python generate_markdown_analysis_report.py \
      --emotion data/features/emotion_features.parquet \
      --issue-sentiment data/features/sentiment/issue_sentiment_features.parquet \
      --comment-sentiment data/features/sentiment/comment_sentiment_features.parquet

Optional custom output folder:
    python generate_markdown_analysis_report.py --output-dir /outputs/sentiment_analysis
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("/outputs/sentiment_analysis")
DEFAULT_REPORT_NAME = "emotion_sentiment_analysis_report.md"

UT_ORANGE = "#FF8200"
UT_GREY = "#4B4B4B"


# -----------------------------
# File loading helpers
# -----------------------------

def read_table(path: Optional[str]) -> pd.DataFrame:
    """Read a parquet/csv/json/jsonl file into a DataFrame. Missing paths return empty DataFrames."""
    if not path:
        return pd.DataFrame()

    p = Path(path)
    if not p.exists():
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

    raise ValueError(f"Unsupported file type for {p}. Use .parquet, .csv, .json, or .jsonl")


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Make sure expected columns exist so later operations do not crash."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def safe_count_unique(df: pd.DataFrame, column: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].dropna().nunique())


def fmt_float(value: object, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def md_table(df: pd.DataFrame, max_rows: int = 20, float_digits: int = 3) -> str:
    """Convert a DataFrame to a compact Markdown table."""
    if df is None or df.empty:
        return "_No data available._"

    out = df.head(max_rows).copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]) or pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: fmt_float(x, float_digits))
    return out.to_markdown(index=False)


def section(title: str) -> str:
    return f"\n## {title}\n"


def subsection(title: str) -> str:
    return f"\n### {title}\n"


def slugify_filename(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "file"


def save_dataframe_csv(df: pd.DataFrame, output_dir: Path, filename: str) -> Optional[Path]:
    """Save a DataFrame as CSV when it has data. Returns the path or None."""
    if df is None or df.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path, index=False)
    return path


def save_current_plot(output_dir: Path, title: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{slugify_filename(title)}.png"
    path = output_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def markdown_image(path: Path, report_dir: Path, alt_text: str) -> str:
    try:
        rel_path = path.relative_to(report_dir)
    except ValueError:
        rel_path = path
    return f"![{alt_text}]({rel_path.as_posix()})"


def prepare_issue_level_emotions(emotion_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer issue-level comment emotion summaries so each issue contributes once.
    If those rows are not available, fall back to all rows with dominant emotions.
    """
    if emotion_df.empty:
        return pd.DataFrame()

    df = ensure_columns(
        emotion_df,
        ["repo_full_name", "issue_number", "text_source", "dominant_emotion", "emotion_confidence", "comment_count_for_issue"],
    ).copy()
    issue_level = df[df["text_source"] == "issue_comment_sentiment_summary"].copy()
    if issue_level.empty:
        issue_level = df.copy()

    issue_level = issue_level[issue_level["dominant_emotion"].notna()].copy()
    issue_level["issue_number"] = pd.to_numeric(issue_level["issue_number"], errors="coerce")
    issue_level["emotion_confidence"] = pd.to_numeric(issue_level["emotion_confidence"], errors="coerce")
    return issue_level


# -----------------------------
# Summary tables
# -----------------------------

def basic_dataset_summary(emotion_df: pd.DataFrame, issue_sent_df: pd.DataFrame, comment_sent_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "dataset": "emotion_features",
            "rows": len(emotion_df),
            "repositories": safe_count_unique(emotion_df, "repo_full_name"),
            "issues": safe_count_unique(emotion_df, "issue_number"),
        },
        {
            "dataset": "issue_sentiment_features",
            "rows": len(issue_sent_df),
            "repositories": safe_count_unique(issue_sent_df, "repo_full_name"),
            "issues": safe_count_unique(issue_sent_df, "issue_number"),
        },
        {
            "dataset": "comment_sentiment_features",
            "rows": len(comment_sent_df),
            "repositories": safe_count_unique(comment_sent_df, "repo_full_name"),
            "issues": safe_count_unique(comment_sent_df, "issue_number"),
        },
    ]
    return pd.DataFrame(rows)


def emotion_distribution(emotion_df: pd.DataFrame) -> pd.DataFrame:
    if emotion_df.empty:
        return pd.DataFrame()
    df = ensure_columns(emotion_df, ["text_source", "dominant_emotion", "emotion_confidence"])
    df = df[df["dominant_emotion"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    df["emotion_confidence"] = pd.to_numeric(df["emotion_confidence"], errors="coerce")
    grouped = (
        df.groupby(["text_source", "dominant_emotion"], dropna=False)
        .agg(
            rows=("dominant_emotion", "size"),
            avg_confidence=("emotion_confidence", "mean"),
        )
        .reset_index()
        .sort_values(["text_source", "rows", "avg_confidence"], ascending=[True, False, False])
    )
    return grouped


def issue_comment_emotion_summary(emotion_df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_issue_level_emotions(emotion_df)
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby("dominant_emotion", dropna=False)
        .agg(
            issues=("dominant_emotion", "size"),
            repos=("repo_full_name", "nunique"),
            avg_confidence=("emotion_confidence", "mean"),
            avg_comment_count=("comment_count_for_issue", "mean"),
        )
        .reset_index()
        .sort_values(["issues", "avg_confidence"], ascending=[False, False])
    )
    grouped["dominant_emotion"] = grouped["dominant_emotion"].fillna("unknown")
    return grouped


def sentiment_by_analysis_set(issue_sent_df: pd.DataFrame) -> pd.DataFrame:
    if issue_sent_df.empty:
        return pd.DataFrame()
    df = ensure_columns(
        issue_sent_df,
        [
            "analysis_set",
            "issue_number",
            "comment_count",
            "positive_comment_share",
            "negative_comment_share",
            "neutral_comment_share",
            "mean_comment_sentiment",
            "comment_sentiment_change_late_minus_early",
            "comment_sentiment_slope",
            "unique_commenter_count",
            "comment_concentration_ratio",
        ],
    ).copy()
    if df["analysis_set"].isna().all():
        df["analysis_set"] = "all_issues"
    else:
        df["analysis_set"] = df["analysis_set"].fillna("unknown")

    numeric_cols = [
        "comment_count",
        "positive_comment_share",
        "negative_comment_share",
        "neutral_comment_share",
        "mean_comment_sentiment",
        "comment_sentiment_change_late_minus_early",
        "comment_sentiment_slope",
        "unique_commenter_count",
        "comment_concentration_ratio",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.groupby("analysis_set", dropna=False)
        .agg(
            issues=("issue_number", "size"),
            avg_comments=("comment_count", "mean"),
            avg_positive_share=("positive_comment_share", "mean"),
            avg_negative_share=("negative_comment_share", "mean"),
            avg_neutral_share=("neutral_comment_share", "mean"),
            avg_comment_sentiment=("mean_comment_sentiment", "mean"),
            avg_late_minus_early=("comment_sentiment_change_late_minus_early", "mean"),
            avg_sentiment_slope=("comment_sentiment_slope", "mean"),
            avg_unique_commenters=("unique_commenter_count", "mean"),
            avg_comment_concentration=("comment_concentration_ratio", "mean"),
        )
        .reset_index()
        .sort_values("issues", ascending=False)
    )
    return grouped


def comment_sentiment_distribution(comment_sent_df: pd.DataFrame) -> pd.DataFrame:
    if comment_sent_df.empty:
        return pd.DataFrame()
    df = ensure_columns(
        comment_sent_df,
        ["analysis_set", "is_positive_comment", "is_negative_comment", "is_neutral_comment", "sentiment_compound"],
    ).copy()
    if df["analysis_set"].isna().all():
        df["analysis_set"] = "all_comments"
    else:
        df["analysis_set"] = df["analysis_set"].fillna("unknown")

    for col in ["is_positive_comment", "is_negative_comment", "is_neutral_comment", "sentiment_compound"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.groupby("analysis_set", dropna=False)
        .agg(
            comments=("sentiment_compound", "size"),
            avg_compound=("sentiment_compound", "mean"),
            positive_comments=("is_positive_comment", "sum"),
            negative_comments=("is_negative_comment", "sum"),
            neutral_comments=("is_neutral_comment", "sum"),
        )
        .reset_index()
    )
    grouped["positive_share"] = grouped["positive_comments"] / grouped["comments"].replace(0, pd.NA)
    grouped["negative_share"] = grouped["negative_comments"] / grouped["comments"].replace(0, pd.NA)
    grouped["neutral_share"] = grouped["neutral_comments"] / grouped["comments"].replace(0, pd.NA)
    return grouped.sort_values("comments", ascending=False)


def repo_level_summary(issue_sent_df: pd.DataFrame) -> pd.DataFrame:
    if issue_sent_df.empty:
        return pd.DataFrame()
    df = ensure_columns(
        issue_sent_df,
        ["repo_full_name", "issue_number", "comment_count", "mean_comment_sentiment", "negative_comment_share"],
    ).copy()
    for col in ["comment_count", "mean_comment_sentiment", "negative_comment_share"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    grouped = (
        df.groupby("repo_full_name", dropna=False)
        .agg(
            issues=("issue_number", "size"),
            avg_comments=("comment_count", "mean"),
            avg_comment_sentiment=("mean_comment_sentiment", "mean"),
            avg_negative_share=("negative_comment_share", "mean"),
        )
        .reset_index()
        .sort_values(["issues", "avg_negative_share"], ascending=[False, False])
    )
    return grouped


def joined_emotion_sentiment_summary(emotion_df: pd.DataFrame, issue_sent_df: pd.DataFrame) -> pd.DataFrame:
    if emotion_df.empty or issue_sent_df.empty:
        return pd.DataFrame()

    emo = prepare_issue_level_emotions(emotion_df)
    if emo.empty:
        return pd.DataFrame()

    sent = ensure_columns(
        issue_sent_df,
        [
            "repo_full_name",
            "issue_number",
            "analysis_set",
            "comment_count",
            "mean_comment_sentiment",
            "negative_comment_share",
            "positive_comment_share",
            "comment_sentiment_change_late_minus_early",
        ],
    ).copy()

    sent["issue_number"] = pd.to_numeric(sent["issue_number"], errors="coerce")
    for col in [
        "comment_count",
        "mean_comment_sentiment",
        "negative_comment_share",
        "positive_comment_share",
        "comment_sentiment_change_late_minus_early",
    ]:
        sent[col] = pd.to_numeric(sent[col], errors="coerce")

    merged = sent.merge(
        emo[["repo_full_name", "issue_number", "dominant_emotion", "emotion_confidence"]],
        on=["repo_full_name", "issue_number"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    merged["analysis_set"] = merged["analysis_set"].fillna("unknown")
    merged["dominant_emotion"] = merged["dominant_emotion"].fillna("unknown")

    grouped = (
        merged.groupby(["analysis_set", "dominant_emotion"], dropna=False)
        .agg(
            issues=("issue_number", "size"),
            repos=("repo_full_name", "nunique"),
            avg_emotion_confidence=("emotion_confidence", "mean"),
            avg_comment_count=("comment_count", "mean"),
            avg_comment_sentiment=("mean_comment_sentiment", "mean"),
            avg_negative_share=("negative_comment_share", "mean"),
            avg_positive_share=("positive_comment_share", "mean"),
            avg_late_minus_early=("comment_sentiment_change_late_minus_early", "mean"),
        )
        .reset_index()
        .sort_values(["analysis_set", "issues", "avg_emotion_confidence"], ascending=[True, False, False])
    )
    return grouped


def emotion_set_comparison_across_repos(emotion_df: pd.DataFrame, issue_sent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare dominant emotions between comparison and wontfix issues.

    Uses issue-level emotion summaries and joins to issue sentiment features so the
    analysis_set column can be attached. Aggregation is done both globally and by
    repository, then the repository-level shares are averaged so large repos do not
    completely dominate the repo-level view.
    """
    if emotion_df.empty or issue_sent_df.empty:
        return pd.DataFrame()

    emo = prepare_issue_level_emotions(emotion_df)
    if emo.empty:
        return pd.DataFrame()

    sent = ensure_columns(issue_sent_df, ["repo_full_name", "issue_number", "analysis_set"]).copy()
    sent["issue_number"] = pd.to_numeric(sent["issue_number"], errors="coerce")
    sent = sent[sent["analysis_set"].isin(["comparison", "wontfix"])].copy()
    if sent.empty:
        return pd.DataFrame()

    merged = sent.merge(
        emo[["repo_full_name", "issue_number", "dominant_emotion", "emotion_confidence"]],
        on=["repo_full_name", "issue_number"],
        how="inner",
    )
    merged = merged[merged["dominant_emotion"].notna()].copy()
    if merged.empty:
        return pd.DataFrame()

    merged["emotion_confidence"] = pd.to_numeric(merged["emotion_confidence"], errors="coerce")

    global_counts = (
        merged.groupby(["dominant_emotion", "analysis_set"])
        .agg(
            issues=("issue_number", "size"),
            repos=("repo_full_name", "nunique"),
            avg_confidence=("emotion_confidence", "mean"),
        )
        .reset_index()
    )

    issue_counts = global_counts.pivot(index="dominant_emotion", columns="analysis_set", values="issues").fillna(0)
    repo_counts = global_counts.pivot(index="dominant_emotion", columns="analysis_set", values="repos").fillna(0)
    conf = global_counts.pivot(index="dominant_emotion", columns="analysis_set", values="avg_confidence")

    total_by_set = merged.groupby("analysis_set").size()

    # Repo-normalized shares: calculate each repo's emotion distribution inside each set, then average.
    repo_emotion_counts = (
        merged.groupby(["repo_full_name", "analysis_set", "dominant_emotion"])
        .size()
        .reset_index(name="emotion_issue_count")
    )
    repo_set_totals = (
        merged.groupby(["repo_full_name", "analysis_set"])
        .size()
        .reset_index(name="repo_set_issue_count")
    )
    repo_shares = repo_emotion_counts.merge(repo_set_totals, on=["repo_full_name", "analysis_set"], how="left")
    repo_shares["repo_share"] = repo_shares["emotion_issue_count"] / repo_shares["repo_set_issue_count"].replace(0, pd.NA)
    avg_repo_share = repo_shares.pivot_table(
        index="dominant_emotion",
        columns="analysis_set",
        values="repo_share",
        aggfunc="mean",
    )

    all_emotions = sorted(set(merged["dominant_emotion"].dropna().astype(str)))
    rows = []
    for emotion in all_emotions:
        comparison_issues = float(issue_counts.get("comparison", pd.Series(dtype=float)).get(emotion, 0))
        wontfix_issues = float(issue_counts.get("wontfix", pd.Series(dtype=float)).get(emotion, 0))
        comparison_share = comparison_issues / float(total_by_set.get("comparison", 0)) if total_by_set.get("comparison", 0) else 0.0
        wontfix_share = wontfix_issues / float(total_by_set.get("wontfix", 0)) if total_by_set.get("wontfix", 0) else 0.0

        comparison_avg_repo_share = avg_repo_share.get("comparison", pd.Series(dtype=float)).get(emotion, 0.0)
        wontfix_avg_repo_share = avg_repo_share.get("wontfix", pd.Series(dtype=float)).get(emotion, 0.0)

        rows.append(
            {
                "dominant_emotion": emotion,
                "comparison_issues": int(comparison_issues),
                "wontfix_issues": int(wontfix_issues),
                "comparison_share": comparison_share,
                "wontfix_share": wontfix_share,
                "share_diff_wontfix_minus_comparison": wontfix_share - comparison_share,
                "comparison_repos": int(repo_counts.get("comparison", pd.Series(dtype=float)).get(emotion, 0)),
                "wontfix_repos": int(repo_counts.get("wontfix", pd.Series(dtype=float)).get(emotion, 0)),
                "comparison_avg_repo_share": comparison_avg_repo_share,
                "wontfix_avg_repo_share": wontfix_avg_repo_share,
                "avg_repo_share_diff_wontfix_minus_comparison": wontfix_avg_repo_share - comparison_avg_repo_share,
                "comparison_avg_confidence": conf.get("comparison", pd.Series(dtype=float)).get(emotion, pd.NA),
                "wontfix_avg_confidence": conf.get("wontfix", pd.Series(dtype=float)).get(emotion, pd.NA),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["share_diff_wontfix_minus_comparison", "wontfix_issues"], ascending=[False, False])


def top_negative_issues(issue_sent_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    if issue_sent_df.empty:
        return pd.DataFrame()
    df = ensure_columns(
        issue_sent_df,
        [
            "repo_full_name",
            "issue_number",
            "analysis_set",
            "comment_count",
            "mean_comment_sentiment",
            "negative_comment_share",
            "positive_comment_share",
        ],
    ).copy()
    for col in ["comment_count", "mean_comment_sentiment", "negative_comment_share", "positive_comment_share"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["mean_comment_sentiment", "negative_comment_share"], ascending=[True, False])
    return df[
        [
            "repo_full_name",
            "issue_number",
            "analysis_set",
            "comment_count",
            "mean_comment_sentiment",
            "negative_comment_share",
            "positive_comment_share",
        ]
    ].head(limit)


def top_positive_issues(issue_sent_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    if issue_sent_df.empty:
        return pd.DataFrame()
    df = ensure_columns(
        issue_sent_df,
        [
            "repo_full_name",
            "issue_number",
            "analysis_set",
            "comment_count",
            "mean_comment_sentiment",
            "positive_comment_share",
            "negative_comment_share",
        ],
    ).copy()
    for col in ["comment_count", "mean_comment_sentiment", "positive_comment_share", "negative_comment_share"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["mean_comment_sentiment", "positive_comment_share"], ascending=[False, False])
    return df[
        [
            "repo_full_name",
            "issue_number",
            "analysis_set",
            "comment_count",
            "mean_comment_sentiment",
            "positive_comment_share",
            "negative_comment_share",
        ]
    ].head(limit)


# -----------------------------
# Graphs
# -----------------------------

def create_graphs(
    output_dir: Path,
    emotion_df: pd.DataFrame,
    issue_sent_df: pd.DataFrame,
    comment_sent_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    top_n: int = 10,
) -> list[dict[str, object]]:
    graphs: list[dict[str, object]] = []

    def add_graph(title: str, description: str, path: Path) -> None:
        graphs.append({"title": title, "description": description, "path": path})

    # 1. Required pie chart: distribution of emotions over all repos.
    issue_emo = prepare_issue_level_emotions(emotion_df)
    if not issue_emo.empty:
        counts = issue_emo["dominant_emotion"].value_counts().sort_values(ascending=False)
        if not counts.empty:
            plt.figure(figsize=(10, 8))
            wedges, _ = plt.pie(
                counts.values,
                labels=None,
                startangle=90,
            )
            legend_labels = [
                f"{emotion} ({(value / counts.sum()) * 100:.1f}%)"
                for emotion, value in zip(counts.index.astype(str), counts.values)
            ]
            plt.title("Distribution of Dominant Emotions Across All Repositories")
            plt.legend(
                wedges,
                legend_labels,
                title="Dominant emotion",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
            )
            path = save_current_plot(output_dir, "emotion_distribution_pie_all_repositories")
            add_graph(
                "Emotion Distribution Across All Repositories",
                "Pie chart of dominant emotions using issue-level comment emotion summaries when available.",
                path,
            )

            plt.figure(figsize=(9, 5))
            counts.plot(kind="bar", color=UT_ORANGE)
            plt.title("Dominant Emotion Counts Across All Repositories")
            plt.xlabel("Dominant emotion")
            plt.ylabel("Number of issue-level emotion summaries")
            plt.xticks(rotation=35, ha="right")
            path = save_current_plot(output_dir, "emotion_distribution_bar_all_repositories")
            add_graph(
                "Dominant Emotion Counts Across All Repositories",
                "Bar chart version of the emotion distribution, useful when the pie chart labels are crowded.",
                path,
            )

            # Extra neutral-removed charts make smaller emotion categories easier to compare.
            non_neutral_counts = counts[counts.index.astype(str).str.lower() != "neutral"]
            if not non_neutral_counts.empty:
                plt.figure(figsize=(10, 8))
                wedges, _ = plt.pie(
                    non_neutral_counts.values,
                    labels=None,
                    startangle=90,
                )
                legend_labels = [
                    f"{emotion} ({(value / non_neutral_counts.sum()) * 100:.1f}%)"
                    for emotion, value in zip(non_neutral_counts.index.astype(str), non_neutral_counts.values)
                ]
                plt.title("Distribution of Dominant Emotions Across All Repositories, Excluding Neutral")
                plt.legend(
                    wedges,
                    legend_labels,
                    title="Dominant emotion",
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    frameon=False,
                )
                path = save_current_plot(output_dir, "emotion_distribution_pie_all_repositories_excluding_neutral")
                add_graph(
                    "Emotion Distribution Across All Repositories, Excluding Neutral",
                    "Pie chart of dominant emotions after removing neutral so the non-neutral emotion categories are easier to see.",
                    path,
                )

                plt.figure(figsize=(9, 5))
                non_neutral_counts.plot(kind="bar", color=UT_ORANGE)
                plt.title("Dominant Emotion Counts Across All Repositories, Excluding Neutral")
                plt.xlabel("Dominant emotion")
                plt.ylabel("Number of issue-level emotion summaries")
                plt.xticks(rotation=35, ha="right")
                path = save_current_plot(output_dir, "emotion_distribution_bar_all_repositories_excluding_neutral")
                add_graph(
                    "Dominant Emotion Counts Across All Repositories, Excluding Neutral",
                    "Bar chart of dominant emotion counts after removing neutral to better show smaller non-neutral categories.",
                    path,
                )

    # 2. Comparison vs Wontfix emotion shares.
    if comparison_df is not None and not comparison_df.empty:
        needed = ["dominant_emotion", "comparison_share", "wontfix_share"]
        if all(col in comparison_df.columns for col in needed):
            plot_df = comparison_df[needed].copy().set_index("dominant_emotion")
            plot_df = plot_df.sort_values("wontfix_share", ascending=False)
            plt.figure(figsize=(10, 5))
            plot_df[["comparison_share", "wontfix_share"]].plot(kind="bar", ax=plt.gca(), color=[UT_ORANGE, UT_GREY])
            plt.title("Emotion Share by Analysis Set")
            plt.xlabel("Dominant emotion")
            plt.ylabel("Share of issues")
            plt.xticks(rotation=35, ha="right")
            plt.legend(["comparison", "wontfix"])
            path = save_current_plot(output_dir, "emotion_share_by_analysis_set")
            add_graph(
                "Emotion Share by Analysis Set",
                "Compares the share of each dominant emotion in the comparison and wontfix issue sets.",
                path,
            )

            diff_df = comparison_df[["dominant_emotion", "share_diff_wontfix_minus_comparison"]].copy()
            diff_df = diff_df.sort_values("share_diff_wontfix_minus_comparison", ascending=True)
            plt.figure(figsize=(10, max(5, 0.35 * len(diff_df))))
            plt.barh(diff_df["dominant_emotion"].astype(str), pd.to_numeric(diff_df["share_diff_wontfix_minus_comparison"], errors="coerce"), color=UT_ORANGE)
            plt.title("Emotion Share Difference: Wontfix Minus Comparison")
            plt.xlabel("Share difference")
            plt.ylabel("Dominant emotion")
            path = save_current_plot(output_dir, "emotion_share_difference_wontfix_minus_comparison")
            add_graph(
                "Emotion Share Difference: Wontfix Minus Comparison",
                "Positive values mean an emotion is more common in wontfix issues; negative values mean it is more common in comparison issues.",
                path,
            )

            # Neutral-removed comparison charts help show the distribution among non-neutral emotions.
            non_neutral_plot_df = plot_df[plot_df.index.astype(str).str.lower() != "neutral"].copy()
            if not non_neutral_plot_df.empty:
                plt.figure(figsize=(10, 5))
                non_neutral_plot_df[["comparison_share", "wontfix_share"]].plot(kind="bar", ax=plt.gca(), color=[UT_ORANGE, UT_GREY])
                plt.title("Emotion Share by Analysis Set, Excluding Neutral")
                plt.xlabel("Dominant emotion")
                plt.ylabel("Share of issues")
                plt.xticks(rotation=35, ha="right")
                plt.legend(["comparison", "wontfix"])
                path = save_current_plot(output_dir, "emotion_share_by_analysis_set_excluding_neutral")
                add_graph(
                    "Emotion Share by Analysis Set, Excluding Neutral",
                    "Compares comparison and wontfix emotion shares after removing neutral so non-neutral emotions are easier to compare.",
                    path,
                )

            non_neutral_diff_df = diff_df[diff_df["dominant_emotion"].astype(str).str.lower() != "neutral"].copy()
            if not non_neutral_diff_df.empty:
                plt.figure(figsize=(10, max(5, 0.35 * len(non_neutral_diff_df))))
                plt.barh(
                    non_neutral_diff_df["dominant_emotion"].astype(str),
                    pd.to_numeric(non_neutral_diff_df["share_diff_wontfix_minus_comparison"], errors="coerce"),
                    color=UT_ORANGE,
                )
                plt.title("Emotion Share Difference: Wontfix Minus Comparison, Excluding Neutral")
                plt.xlabel("Share difference")
                plt.ylabel("Dominant emotion")
                path = save_current_plot(output_dir, "emotion_share_difference_wontfix_minus_comparison_excluding_neutral")
                add_graph(
                    "Emotion Share Difference: Wontfix Minus Comparison, Excluding Neutral",
                    "Shows which non-neutral emotions are more common in wontfix or comparison issues after neutral is removed.",
                    path,
                )

    # 3. Average issue comment sentiment by analysis set.
    if not issue_sent_df.empty:
        sent = ensure_columns(issue_sent_df, ["analysis_set", "mean_comment_sentiment"]).copy()
        sent["mean_comment_sentiment"] = pd.to_numeric(sent["mean_comment_sentiment"], errors="coerce")
        sent = sent.dropna(subset=["analysis_set", "mean_comment_sentiment"])
        if not sent.empty:
            grouped = sent.groupby("analysis_set")["mean_comment_sentiment"].mean().sort_values(ascending=False)
            plt.figure(figsize=(8, 5))
            grouped.plot(kind="bar", color=UT_ORANGE)
            plt.title("Average Issue Comment Sentiment by Analysis Set")
            plt.xlabel("Analysis set")
            plt.ylabel("Average mean comment sentiment")
            plt.xticks(rotation=20, ha="right")
            path = save_current_plot(output_dir, "average_issue_comment_sentiment_by_analysis_set")
            add_graph(
                "Average Issue Comment Sentiment by Analysis Set",
                "Shows whether issue discussions are generally more positive or negative in each analysis set.",
                path,
            )

    # 4. Positive, negative, and neutral shares by analysis set.
    if not issue_sent_df.empty:
        share_cols = ["positive_comment_share", "negative_comment_share", "neutral_comment_share"]
        sent = ensure_columns(issue_sent_df, ["analysis_set", *share_cols]).copy()
        for col in share_cols:
            sent[col] = pd.to_numeric(sent[col], errors="coerce")
        sent = sent.dropna(subset=["analysis_set"])
        if not sent.empty and any(sent[col].notna().any() for col in share_cols):
            grouped = sent.groupby("analysis_set")[share_cols].mean()
            plt.figure(figsize=(9, 5))
            grouped.plot(kind="bar", ax=plt.gca())
            plt.title("Average Comment Sentiment Shares by Analysis Set")
            plt.xlabel("Analysis set")
            plt.ylabel("Average share")
            plt.xticks(rotation=20, ha="right")
            plt.legend(["positive", "negative", "neutral"])
            path = save_current_plot(output_dir, "average_comment_sentiment_shares_by_analysis_set")
            add_graph(
                "Average Comment Sentiment Shares by Analysis Set",
                "Compares average positive, negative, and neutral comment shares for each issue set.",
                path,
            )

    # 5. Distribution of issue-level mean comment sentiment.
    if not issue_sent_df.empty:
        values = numeric_series(issue_sent_df, "mean_comment_sentiment").dropna()
        if not values.empty:
            plt.figure(figsize=(9, 5))
            plt.hist(values, bins=30, color=UT_ORANGE)
            plt.title("Distribution of Issue-Level Mean Comment Sentiment")
            plt.xlabel("Mean comment sentiment")
            plt.ylabel("Number of issues")
            path = save_current_plot(output_dir, "distribution_of_issue_level_mean_comment_sentiment")
            add_graph(
                "Distribution of Issue-Level Mean Comment Sentiment",
                "Shows how issue discussions are distributed from more negative to more positive sentiment.",
                path,
            )

    # 6. Distribution of comment-level sentiment.
    if not comment_sent_df.empty:
        values = numeric_series(comment_sent_df, "sentiment_compound").dropna()
        if not values.empty:
            plt.figure(figsize=(9, 5))
            plt.hist(values, bins=30, color=UT_ORANGE)
            plt.title("Distribution of Comment-Level Sentiment")
            plt.xlabel("Comment sentiment compound score")
            plt.ylabel("Number of comments")
            path = save_current_plot(output_dir, "distribution_of_comment_level_sentiment")
            add_graph(
                "Distribution of Comment-Level Sentiment",
                "Shows the spread of individual comment sentiment scores across all comments.",
                path,
            )

    # 7. Repository-level average sentiment, limited to top_n by issue count.
    repo_summary = repo_level_summary(issue_sent_df)
    if not repo_summary.empty:
        plot_df = repo_summary.head(top_n).copy()
        if "avg_comment_sentiment" in plot_df.columns:
            plot_df["avg_comment_sentiment"] = pd.to_numeric(plot_df["avg_comment_sentiment"], errors="coerce")
            plot_df = plot_df.sort_values("avg_comment_sentiment", ascending=True)
            plt.figure(figsize=(10, max(5, 0.35 * len(plot_df))))
            plt.barh(plot_df["repo_full_name"].astype(str), plot_df["avg_comment_sentiment"], color=UT_ORANGE)
            plt.title(f"Average Comment Sentiment for Top {len(plot_df)} Repositories by Issue Count")
            plt.xlabel("Average comment sentiment")
            plt.ylabel("Repository")
            path = save_current_plot(output_dir, "average_comment_sentiment_for_top_repositories")
            add_graph(
                "Average Comment Sentiment for Top Repositories",
                "Shows average issue discussion sentiment for the repositories with the most analyzed issues.",
                path,
            )

    return graphs


# -----------------------------
# Report writing
# -----------------------------

def write_report(
    output_dir: str | Path,
    emotion_path: Optional[str],
    issue_sentiment_path: Optional[str],
    comment_sentiment_path: Optional[str],
    report_name: str = DEFAULT_REPORT_NAME,
    top_n: int = 10,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / report_name

    emotion_df = read_table(emotion_path)
    issue_sent_df = read_table(issue_sentiment_path)
    comment_sent_df = read_table(comment_sentiment_path)

    # Build all summary tables first.
    tables = {
        "dataset_overview": basic_dataset_summary(emotion_df, issue_sent_df, comment_sent_df),
        "dominant_emotion_distribution": emotion_distribution(emotion_df),
        "issue_comment_emotion_summary": issue_comment_emotion_summary(emotion_df),
        "issue_level_sentiment_summary": sentiment_by_analysis_set(issue_sent_df),
        "comment_level_sentiment_summary": comment_sentiment_distribution(comment_sent_df),
        "combined_emotion_sentiment_view": joined_emotion_sentiment_summary(emotion_df, issue_sent_df),
        "comparison_vs_wontfix_emotion_comparison": emotion_set_comparison_across_repos(emotion_df, issue_sent_df),
        "repository_level_summary": repo_level_summary(issue_sent_df),
        "most_negative_issues": top_negative_issues(issue_sent_df, limit=top_n),
        "most_positive_issues": top_positive_issues(issue_sent_df, limit=top_n),
    }

    # Save all non-empty summary tables as CSVs in the output folder.
    csv_paths: dict[str, Path] = {}
    for name, df in tables.items():
        csv_path = save_dataframe_csv(df, output_dir, f"{name}.csv")
        if csv_path:
            csv_paths[name] = csv_path

    # Create graphs after the comparison table is available.
    graphs = create_graphs(
        output_dir=output_dir,
        emotion_df=emotion_df,
        issue_sent_df=issue_sent_df,
        comment_sent_df=comment_sent_df,
        comparison_df=tables["comparison_vs_wontfix_emotion_comparison"],
        top_n=top_n,
    )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append("# Emotion and Sentiment Analysis Report")
    lines.append("")
    lines.append(f"Generated at: **{generated_at}**")
    lines.append("")
    lines.append("## Output Folder")
    lines.append(f"All report files, CSV summaries, and graphs were written to: `{output_dir}`")
    lines.append("")
    lines.append("## Input Files")
    lines.append(f"- Emotion features: `{emotion_path or 'not provided'}`")
    lines.append(f"- Issue sentiment features: `{issue_sentiment_path or 'not provided'}`")
    lines.append(f"- Comment sentiment features: `{comment_sentiment_path or 'not provided'}`")

    missing = []
    for label, path, df in [
        ("emotion features", emotion_path, emotion_df),
        ("issue sentiment features", issue_sentiment_path, issue_sent_df),
        ("comment sentiment features", comment_sentiment_path, comment_sent_df),
    ]:
        if not path or df.empty:
            missing.append(label)

    if missing:
        lines.append("")
        lines.append("> Note: Some inputs were missing or empty, so their sections were skipped: " + ", ".join(missing) + ".")

    lines.append(section("Dataset Overview"))
    lines.append(md_table(tables["dataset_overview"]))

    lines.append(section("Graphs and Charts"))
    if not graphs:
        lines.append("_No graphs could be created because the required data was missing or empty._")
    else:
        for graph in graphs:
            path = graph["path"]
            lines.append(subsection(str(graph["title"])))
            lines.append(str(graph["description"]))
            lines.append("")
            lines.append(markdown_image(Path(path), output_dir, str(graph["title"])))

    lines.append(section("Emotion Feature Summary"))
    if emotion_df.empty:
        lines.append("_Emotion feature file was missing or empty._")
    else:
        lines.append(
            "The emotion output includes rows for issue bodies, issue comments, and issue-level comment emotion summaries when those rows are present."
        )
        lines.append(subsection("Dominant Emotion Distribution"))
        lines.append(md_table(tables["dominant_emotion_distribution"], max_rows=30))
        lines.append(subsection("Issue-Level Comment Emotion Summary"))
        lines.append(md_table(tables["issue_comment_emotion_summary"], max_rows=20))

    lines.append(section("Issue-Level Sentiment Summary"))
    if issue_sent_df.empty:
        lines.append("_Issue sentiment feature file was missing or empty._")
    else:
        lines.append(
            "This section summarizes sentiment at the issue level, including average comment sentiment, positive/negative/neutral comment shares, and how sentiment changes from early to late comments."
        )
        lines.append(md_table(tables["issue_level_sentiment_summary"], max_rows=20))

    lines.append(section("Comment-Level Sentiment Summary"))
    if comment_sent_df.empty:
        lines.append("_Comment sentiment feature file was missing or empty._")
    else:
        lines.append("This section summarizes individual comment sentiment labels created from the compound sentiment score.")
        lines.append(md_table(tables["comment_level_sentiment_summary"], max_rows=20))

    lines.append(section("Combined Emotion + Sentiment View"))
    joined = tables["combined_emotion_sentiment_view"]
    if joined.empty:
        lines.append(
            "_Could not create a joined emotion/sentiment summary. This usually means one input is missing, or issue keys did not match between files._"
        )
    else:
        lines.append(
            "This table joins issue-level comment emotion summaries with issue-level sentiment features using `repo_full_name` and `issue_number`."
        )
        lines.append(md_table(joined, max_rows=40))

    lines.append(section("Comparison vs Wontfix Emotion Comparison Across Repositories"))
    comparison = tables["comparison_vs_wontfix_emotion_comparison"]
    if comparison.empty:
        lines.append(
            "_Could not create the comparison vs wontfix emotion comparison. This usually means `analysis_set`, `repo_full_name`, or `issue_number` did not match between the emotion and issue sentiment files._"
        )
    else:
        lines.append(
            "This section compares the dominant issue-level comment emotions between the `comparison` and `wontfix` sets. It is split into smaller tables so it is easier to read on narrow screens."
        )

        counts_table = comparison[
            [
                "dominant_emotion",
                "comparison_issues",
                "wontfix_issues",
                "comparison_share",
                "wontfix_share",
            ]
        ].copy()
        lines.append(subsection("Issue Counts and Overall Shares"))
        lines.append(md_table(counts_table, max_rows=30))

        repo_table = comparison[
            [
                "dominant_emotion",
                "comparison_repos",
                "wontfix_repos",
                "comparison_avg_repo_share",
                "wontfix_avg_repo_share",
            ]
        ].copy()
        lines.append(subsection("Repository Coverage"))
        lines.append(md_table(repo_table, max_rows=30))

        diff_table = comparison[
            [
                "dominant_emotion",
                "share_diff_wontfix_minus_comparison",
                "avg_repo_share_diff_wontfix_minus_comparison",
                "comparison_avg_confidence",
                "wontfix_avg_confidence",
            ]
        ].copy()
        lines.append(subsection("Differences and Confidence"))
        lines.append(md_table(diff_table, max_rows=30))

    lines.append(section("Repository-Level Summary"))
    repo_summary = tables["repository_level_summary"]
    if repo_summary.empty:
        lines.append("_No repository-level sentiment summary could be created._")
    else:
        lines.append(md_table(repo_summary, max_rows=top_n))

    lines.append(section("Most Negative Issues by Mean Comment Sentiment"))
    lines.append(md_table(tables["most_negative_issues"], max_rows=top_n))

    lines.append(section("Most Positive Issues by Mean Comment Sentiment"))
    lines.append(md_table(tables["most_positive_issues"], max_rows=top_n))

    lines.append(section("Generated Files"))
    lines.append("The script generated the following summary CSV files and chart images in the output folder.")
    if csv_paths:
        lines.append(subsection("CSV Summary Tables"))
        for name, path in sorted(csv_paths.items()):
            lines.append(f"- `{path.name}`")
    if graphs:
        lines.append(subsection("Graph Images"))
        for graph in graphs:
            lines.append(f"- `{Path(graph['path']).name}`")

    lines.append(section("Suggested Interpretation Notes"))
    lines.append(
        "- `mean_comment_sentiment` gives the overall tone of the issue discussion. Higher values are more positive; lower values are more negative."
    )
    lines.append(
        "- `comment_sentiment_change_late_minus_early` can help show whether the discussion became more positive or more negative over time."
    )
    lines.append(
        "- `dominant_emotion` from the issue-level comment summary shows the most common detected emotion across comments for an issue."
    )
    lines.append("- `comment_concentration_ratio` can help show whether a discussion was dominated by one or a few commenters.")
    lines.append("- Emotion confidence and sentiment compound scores should be treated as model-derived estimates, not ground-truth labels.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Markdown report to: {report_path}")
    print(f"Wrote all generated files to: {output_dir}")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Markdown report, CSV summaries, and graphs from emotion/sentiment outputs.")
    parser.add_argument("--emotion", default="data/features/emotion_features.parquet", help="Path to emotion_features parquet/csv/json file.")
    parser.add_argument(
        "--issue-sentiment",
        default="data/features/sentiment/issue_sentiment_features.parquet",
        help="Path to issue_sentiment_features parquet/csv/json file.",
    )
    parser.add_argument(
        "--comment-sentiment",
        default="data/features/sentiment/comment_sentiment_features.parquet",
        help="Path to comment_sentiment_features parquet/csv/json file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder where the Markdown report, CSV summaries, and graphs will be written.",
    )
    parser.add_argument(
        "--report-name",
        default=DEFAULT_REPORT_NAME,
        help="Markdown report filename to create inside --output-dir.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Number of top repos/issues to show in ranked tables and charts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_report(
        output_dir=args.output_dir,
        emotion_path=args.emotion,
        issue_sentiment_path=args.issue_sentiment,
        comment_sentiment_path=args.comment_sentiment,
        report_name=args.report_name,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
