#!/usr/bin/env python3
"""
Generate a Markdown analysis report from sentiment and emotion feature outputs.

This script is meant to run AFTER these two pipeline stages have created files such as:
- data/features/emotion_features.parquet
- data/features/sentiment/issue_sentiment_features.parquet
- data/features/sentiment/comment_sentiment_features.parquet

Example:
    python generate_markdown_analysis_report.py \
      --emotion data/features/emotion_features.parquet \
      --issue-sentiment data/features/sentiment/issue_sentiment_features.parquet \
      --comment-sentiment data/features/sentiment/comment_sentiment_features.parquet \
      --output reports/emotion_sentiment_analysis_report.md

If one of the input files is missing, the report will still be created and will note
which sections were skipped.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


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


def safe_mean(df: pd.DataFrame, column: str) -> float:
    values = numeric_series(df, column).dropna()
    return float(values.mean()) if not values.empty else 0.0


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


# -----------------------------
# Analysis helpers
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
    if emotion_df.empty:
        return pd.DataFrame()
    df = ensure_columns(emotion_df, ["text_source", "dominant_emotion", "emotion_confidence", "comment_count_for_issue"])
    df = df[df["text_source"] == "issue_comment_sentiment_summary"].copy()
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby("dominant_emotion", dropna=False)
        .agg(
            issues=("dominant_emotion", "size"),
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
    )
    if df["analysis_set"].isna().all():
        df["analysis_set"] = "all_issues"
    else:
        df["analysis_set"] = df["analysis_set"].fillna("unknown")

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
    )
    if df["analysis_set"].isna().all():
        df["analysis_set"] = "all_comments"
    else:
        df["analysis_set"] = df["analysis_set"].fillna("unknown")

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
    df = ensure_columns(issue_sent_df, ["repo_full_name", "issue_number", "comment_count", "mean_comment_sentiment", "negative_comment_share"])
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
    """Join issue-level comment emotion summaries to issue sentiment features."""
    if emotion_df.empty or issue_sent_df.empty:
        return pd.DataFrame()

    emo = ensure_columns(emotion_df, ["repo_full_name", "issue_number", "text_source", "dominant_emotion", "emotion_confidence"])
    emo = emo[emo["text_source"] == "issue_comment_sentiment_summary"].copy()
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

    # Avoid type mismatch when joining.
    emo["issue_number"] = pd.to_numeric(emo["issue_number"], errors="coerce")
    sent["issue_number"] = pd.to_numeric(sent["issue_number"], errors="coerce")

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
    df["mean_comment_sentiment"] = pd.to_numeric(df["mean_comment_sentiment"], errors="coerce")
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
    df["mean_comment_sentiment"] = pd.to_numeric(df["mean_comment_sentiment"], errors="coerce")
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


def write_report(
    output_path: str,
    emotion_path: Optional[str],
    issue_sentiment_path: Optional[str],
    comment_sentiment_path: Optional[str],
    top_n: int = 10,
) -> None:
    emotion_df = read_table(emotion_path)
    issue_sent_df = read_table(issue_sentiment_path)
    comment_sent_df = read_table(comment_sentiment_path)

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: list[str] = []
    lines.append("# Emotion and Sentiment Analysis Report")
    lines.append("")
    lines.append(f"Generated at: **{generated_at}**")
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
    lines.append(md_table(basic_dataset_summary(emotion_df, issue_sent_df, comment_sent_df)))

    lines.append(section("Emotion Feature Summary"))
    if emotion_df.empty:
        lines.append("_Emotion feature file was missing or empty._")
    else:
        lines.append(
            "The emotion output includes rows for issue bodies, issue comments, and issue-level comment emotion summaries when those rows are present."
        )
        lines.append(subsection("Dominant Emotion Distribution"))
        lines.append(md_table(emotion_distribution(emotion_df), max_rows=30))
        lines.append(subsection("Issue-Level Comment Emotion Summary"))
        lines.append(md_table(issue_comment_emotion_summary(emotion_df), max_rows=20))

    lines.append(section("Issue-Level Sentiment Summary"))
    if issue_sent_df.empty:
        lines.append("_Issue sentiment feature file was missing or empty._")
    else:
        lines.append(
            "This section summarizes sentiment at the issue level, including average comment sentiment, positive/negative/neutral comment shares, and how sentiment changes from early to late comments."
        )
        lines.append(md_table(sentiment_by_analysis_set(issue_sent_df), max_rows=20))

    lines.append(section("Comment-Level Sentiment Summary"))
    if comment_sent_df.empty:
        lines.append("_Comment sentiment feature file was missing or empty._")
    else:
        lines.append(
            "This section summarizes the individual comment sentiment labels created from the compound sentiment score."
        )
        lines.append(md_table(comment_sentiment_distribution(comment_sent_df), max_rows=20))

    lines.append(section("Combined Emotion + Sentiment View"))
    joined = joined_emotion_sentiment_summary(emotion_df, issue_sent_df)
    if joined.empty:
        lines.append(
            "_Could not create a joined emotion/sentiment summary. This usually means one input is missing, or issue keys did not match between files._"
        )
    else:
        lines.append(
            "This table joins issue-level comment emotion summaries with issue-level sentiment features using `repo_full_name` and `issue_number`."
        )
        lines.append(md_table(joined, max_rows=40))

    lines.append(section("Repository-Level Summary"))
    repo_summary = repo_level_summary(issue_sent_df)
    if repo_summary.empty:
        lines.append("_No repository-level sentiment summary could be created._")
    else:
        lines.append(md_table(repo_summary, max_rows=top_n))

    lines.append(section("Most Negative Issues by Mean Comment Sentiment"))
    lines.append(md_table(top_negative_issues(issue_sent_df, limit=top_n), max_rows=top_n))

    lines.append(section("Most Positive Issues by Mean Comment Sentiment"))
    lines.append(md_table(top_positive_issues(issue_sent_df, limit=top_n), max_rows=top_n))

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
    lines.append(
        "- `comment_concentration_ratio` can help show whether a discussion was dominated by one or a few commenters."
    )
    lines.append(
        "- Emotion confidence and sentiment compound scores should be treated as model-derived estimates, not ground-truth labels."
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote Markdown report to: {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Markdown report from emotion and sentiment feature outputs.")
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
    parser.add_argument("--output", default="data/features/sentiment/emotion_sentiment_analysis_report.md", help="Output Markdown report path.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top repos/issues to show in ranked tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_report(
        output_path=args.output,
        emotion_path=args.emotion,
        issue_sentiment_path=args.issue_sentiment,
        comment_sentiment_path=args.comment_sentiment,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
