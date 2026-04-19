import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random

analyzer = SentimentIntensityAnalyzer()

# Load files
wontfix_df = pd.read_parquet("data/processed/wontfix_issue_set.parquet")
issues_df = pd.read_parquet("data/linked/resolved_entities/issues_resolved.parquet")
comments_df = pd.read_parquet("data/linked/resolved_entities/issue_comments_resolved.parquet")

# Mark WONTFIX vs non-WONTFIX using full issue dataset
wontfix_targets = wontfix_df[["issue_number"]].drop_duplicates().copy()
wontfix_targets["analysis_set"] = "wontfix"

issues_df = issues_df.merge(wontfix_targets, on="issue_number", how="left")
issues_df["analysis_set"] = issues_df["analysis_set"].fillna("non_wontfix")

print("Issue groups in full dataset:")
print(issues_df["analysis_set"].value_counts(dropna=False))

# Detect comment columns
comment_text_col = None
for col in ["body", "comment_body", "text", "comment_text"]:
    if col in comments_df.columns:
        comment_text_col = col
        break

if comment_text_col is None:
    raise ValueError(f"Could not find comment text column. Available columns: {comments_df.columns.tolist()}")

comment_issue_col = None
for col in ["issue_number", "number"]:
    if col in comments_df.columns:
        comment_issue_col = col
        break

if comment_issue_col is None:
    raise ValueError(f"Could not find issue-number column in comments. Available columns: {comments_df.columns.tolist()}")

# Keep comments for all issues in full dataset
comments_df = comments_df.merge(
    issues_df[["issue_number", "analysis_set"]],
    left_on=comment_issue_col,
    right_on="issue_number",
    how="inner"
)

print("\nComments linked to issues in full dataset:", len(comments_df))

# Score sentiment
comments_df = comments_df.dropna(subset=[comment_text_col]).copy()
comments_df["comment_sentiment"] = comments_df[comment_text_col].astype(str).apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)

# Aggregate to issue level
issue_sentiment = (
    comments_df.groupby(["issue_number", "analysis_set"], as_index=False)
    .agg(
        mean_comment_sentiment=("comment_sentiment", "mean"),
        std_comment_sentiment=("comment_sentiment", "std"),
        comment_count=("comment_sentiment", "size"),
    )
)

print("\nIssue-level sentiment rows:")
print(issue_sentiment["analysis_set"].value_counts(dropna=False))

output_dir = Path("outputs/plots")
output_dir.mkdir(parents=True, exist_ok=True)

UTK_ORANGE = "#F77F00"
UTK_LIGHT_ORANGE = "#FFD79A"
UTK_LIGHTER = "#FFE8C2"
UTK_SMOKY = "#58595B"

def jitter(base_x, n, width=0.08):
    return [base_x + random.uniform(-width, width) for _ in range(n)]

def draw_boxplot(values_left, values_right, left_label, right_label, title, ylabel, save_path):
    left_series = pd.Series(values_left)
    right_series = pd.Series(values_right)

    left_median = left_series.median()
    right_median = right_series.median()

    plt.figure(figsize=(10, 6))

    bp = plt.boxplot(
        [values_left, values_right],
        tick_labels=[
            f"{left_label}\n(n={len(values_left)})",
            f"{right_label}\n(n={len(values_right)})"
        ],
        widths=0.5,
        patch_artist=True,
        medianprops=dict(color=UTK_ORANGE, linewidth=2.2),
        boxprops=dict(color=UTK_SMOKY, linewidth=1.5),
        whiskerprops=dict(color=UTK_SMOKY, linewidth=1.5),
        capprops=dict(color=UTK_SMOKY, linewidth=1.5),
        flierprops=dict(
            marker="o",
            markerfacecolor="white",
            markeredgecolor=UTK_SMOKY,
            markersize=7,
            linestyle="none"
        )
    )

    bp["boxes"][0].set_facecolor(UTK_LIGHT_ORANGE)
    bp["boxes"][1].set_facecolor(UTK_LIGHTER)

    plt.scatter(jitter(1, len(values_left)), values_left, alpha=0.70, s=35, color=UTK_ORANGE, zorder=3)
    plt.scatter(jitter(2, len(values_right)), values_right, alpha=0.45, s=35, color=UTK_SMOKY, zorder=3)

    plt.axhline(0, linestyle="--", linewidth=1, color=UTK_SMOKY, alpha=0.7)

    plt.text(1, left_median + 0.03, f"Median = {left_median:.2f}", ha="center", color=UTK_SMOKY)
    plt.text(2, right_median + 0.03, f"Median = {right_median:.2f}", ha="center", color=UTK_SMOKY)

    plt.title(title)
    plt.xlabel("Issue Group")
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Mean sentiment plot
wontfix_mean_vals = issue_sentiment.loc[
    issue_sentiment["analysis_set"] == "wontfix", "mean_comment_sentiment"
].dropna().tolist()

non_wontfix_mean_vals = issue_sentiment.loc[
    issue_sentiment["analysis_set"] == "non_wontfix", "mean_comment_sentiment"
].dropna().tolist()

draw_boxplot(
    wontfix_mean_vals,
    non_wontfix_mean_vals,
    "WONTFIX",
    "Non-WONTFIX",
    "Average Issue Discussion Sentiment (Full Dataset)",
    "Mean Comment Sentiment",
    output_dir / "boxplot_mean_comment_sentiment_full_dataset.png"
)

# Std sentiment plot
wontfix_std_vals = issue_sentiment.loc[
    issue_sentiment["analysis_set"] == "wontfix", "std_comment_sentiment"
].dropna().tolist()

non_wontfix_std_vals = issue_sentiment.loc[
    issue_sentiment["analysis_set"] == "non_wontfix", "std_comment_sentiment"
].dropna().tolist()

if len(wontfix_std_vals) > 0 and len(non_wontfix_std_vals) > 0:
    draw_boxplot(
        wontfix_std_vals,
        non_wontfix_std_vals,
        "WONTFIX",
        "Non-WONTFIX",
        "Issue-Level Sentiment Variability (Full Dataset)",
        "Std Comment Sentiment",
        output_dir / "boxplot_std_comment_sentiment_full_dataset.png"
    )

print("\nSaved plots to:")
print(output_dir / "boxplot_mean_comment_sentiment_full_dataset.png")
print(output_dir / "boxplot_std_comment_sentiment_full_dataset.png")