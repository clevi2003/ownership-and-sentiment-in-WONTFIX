import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random

analyzer = SentimentIntensityAnalyzer()

# Load data
pairs_df = pd.read_parquet("data/processed/wontfix_comparison_pairs.parquet")
comments_df = pd.read_parquet("data/linked/resolved_entities/issue_comments_resolved.parquet")

# Detect columns
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

# Score sentiment
comments_df = comments_df.dropna(subset=[comment_text_col]).copy()
comments_df["comment_sentiment"] = comments_df[comment_text_col].astype(str).apply(
    lambda x: analyzer.polarity_scores(x)["compound"]
)

# Aggregate to issue-level sentiment
issue_sentiment = (
    comments_df.groupby(comment_issue_col, as_index=False)
    .agg(
        mean_comment_sentiment=("comment_sentiment", "mean"),
        std_comment_sentiment=("comment_sentiment", "std"),
        comment_count=("comment_sentiment", "size"),
    )
    .rename(columns={comment_issue_col: "issue_number"})
)

# WONTFIX sentiment
wontfix_sentiment = (
    pairs_df[["wontfix_issue_number"]]
    .drop_duplicates()
    .merge(
        issue_sentiment.rename(columns={
            "issue_number": "wontfix_issue_number",
            "mean_comment_sentiment": "wontfix_mean_sentiment"
        })[["wontfix_issue_number", "wontfix_mean_sentiment"]],
        on="wontfix_issue_number",
        how="left"
    )
)

# Comparison sentiment
comparison_pairs = pairs_df.merge(
    issue_sentiment.rename(columns={
        "issue_number": "comparison_issue_number",
        "mean_comment_sentiment": "comparison_mean_sentiment"
    })[["comparison_issue_number", "comparison_mean_sentiment"]],
    on="comparison_issue_number",
    how="left"
)

# Average matched comparison sentiment per WONTFIX issue
matched_avg = (
    comparison_pairs.groupby("wontfix_issue_number", as_index=False)
    .agg(
        avg_matched_comparison_sentiment=("comparison_mean_sentiment", "mean"),
        matched_pair_count=("comparison_mean_sentiment", "count")
    )
)

# Compute difference
diff_df = wontfix_sentiment.merge(
    matched_avg,
    on="wontfix_issue_number",
    how="inner"
)

diff_df = diff_df.dropna(subset=["wontfix_mean_sentiment", "avg_matched_comparison_sentiment"]).copy()
diff_df["sentiment_difference"] = (
    diff_df["avg_matched_comparison_sentiment"] - diff_df["wontfix_mean_sentiment"]
)

print("Rows with matched sentiment difference:", len(diff_df))
print(diff_df[[
    "wontfix_issue_number",
    "wontfix_mean_sentiment",
    "avg_matched_comparison_sentiment",
    "sentiment_difference",
    "matched_pair_count"
]].head())

# Plot settings
output_dir = Path("outputs/plots")
output_dir.mkdir(parents=True, exist_ok=True)

UTK_ORANGE = "#F77F00"
UTK_LIGHT_ORANGE = "#FFD79A"
UTK_SMOKY = "#58595B"

def jitter(base_x, n, width=0.08):
    return [base_x + random.uniform(-width, width) for _ in range(n)]

vals = diff_df["sentiment_difference"].dropna().tolist()
median_val = pd.Series(vals).median()

plt.figure(figsize=(9, 6))

bp = plt.boxplot(
    [vals],
    tick_labels=[f"Matched Avg - WONTFIX\n(n={len(vals)})"],
    widths=0.45,
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

plt.scatter(jitter(1, len(vals)), vals, alpha=0.7, s=35, color=UTK_ORANGE)

plt.axhline(0, linestyle="--", linewidth=1.2, color=UTK_SMOKY, alpha=0.8)
plt.text(1, median_val + 0.03, f"Median = {median_val:.2f}", ha="center", color=UTK_SMOKY)

plt.title("Matched-Pair Sentiment Difference")
plt.xlabel("Difference Group")
plt.ylabel("Avg(Matched Comparison Sentiment) - WONTFIX Sentiment")
plt.grid(axis="y", linestyle="--", alpha=0.35)
plt.tight_layout()
plt.savefig(output_dir / "boxplot_matched_pair_sentiment_difference.png", dpi=300)
plt.close()

print("\nSaved plot to:")
print(output_dir / "boxplot_matched_pair_sentiment_difference.png")
