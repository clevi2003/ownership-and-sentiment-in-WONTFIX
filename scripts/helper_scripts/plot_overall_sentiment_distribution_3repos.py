import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random

# ---------------------------
# EDIT THESE 3 PATHS
# ---------------------------
repo_files = [
    "data/repo1/issue_sentiment_features.parquet",
    "data/repo2/issue_sentiment_features.parquet",
    "data/repo3/issue_sentiment_features.parquet",
]

# Example if your files are somewhere else:
# repo_files = [
#     "/Users/yourname/Desktop/repo_a_issue_sentiment_features.parquet",
#     "/Users/yourname/Desktop/repo_b_issue_sentiment_features.parquet",
#     "/Users/yourname/Desktop/repo_c_issue_sentiment_features.parquet",
# ]

# ---------------------------
# Load and combine
# ---------------------------
frames = []
for file_path in repo_files:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    df = pd.read_parquet(p)
    df["source_file"] = p.name
    frames.append(df)

pooled_df = pd.concat(frames, ignore_index=True)

print("Loaded rows:", len(pooled_df))
print("Columns:")
print(pooled_df.columns.tolist())

# ---------------------------
# Detect group column
# ---------------------------
group_col = None
for col in ["analysis_set", "is_wontfix", "label", "issue_group"]:
    if col in pooled_df.columns:
        group_col = col
        break

if group_col is None:
    raise ValueError(
        "Could not find group column. Expected one of: "
        "analysis_set, is_wontfix, label, issue_group"
    )

# Normalize to wontfix / non_wontfix
if group_col == "analysis_set":
    pooled_df["plot_group"] = pooled_df["analysis_set"].astype(str).str.lower().map(
        lambda x: "wontfix" if "wontfix" in x else "non_wontfix"
    )
elif group_col == "is_wontfix":
    pooled_df["plot_group"] = pooled_df["is_wontfix"].map(
        lambda x: "wontfix" if bool(x) else "non_wontfix"
    )
else:
    pooled_df["plot_group"] = pooled_df[group_col].astype(str).str.lower().map(
        lambda x: "wontfix" if "wontfix" in x else "non_wontfix"
    )

print("\nGroup counts:")
print(pooled_df["plot_group"].value_counts(dropna=False))

# ---------------------------
# Check required sentiment columns
# ---------------------------
if "mean_comment_sentiment" not in pooled_df.columns:
    raise ValueError(
        f"'mean_comment_sentiment' not found. Available columns: {pooled_df.columns.tolist()}"
    )

# Optional std column
has_std = "std_comment_sentiment" in pooled_df.columns

# ---------------------------
# Plot helpers
# ---------------------------
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

# ---------------------------
# Mean sentiment plot
# ---------------------------
wontfix_mean_vals = pooled_df.loc[
    pooled_df["plot_group"] == "wontfix", "mean_comment_sentiment"
].dropna().tolist()

non_wontfix_mean_vals = pooled_df.loc[
    pooled_df["plot_group"] == "non_wontfix", "mean_comment_sentiment"
].dropna().tolist()

if len(wontfix_mean_vals) == 0 or len(non_wontfix_mean_vals) == 0:
    raise ValueError("One of the groups has no mean sentiment values to plot.")

draw_boxplot(
    wontfix_mean_vals,
    non_wontfix_mean_vals,
    "WONTFIX",
    "Non-WONTFIX",
    "Average Issue Discussion Sentiment (3 Repos)",
    "Mean Comment Sentiment",
    output_dir / "boxplot_mean_comment_sentiment_3repos.png"
)

# ---------------------------
# Std sentiment plot
# ---------------------------
if has_std:
    wontfix_std_vals = pooled_df.loc[
        pooled_df["plot_group"] == "wontfix", "std_comment_sentiment"
    ].dropna().tolist()

    non_wontfix_std_vals = pooled_df.loc[
        pooled_df["plot_group"] == "non_wontfix", "std_comment_sentiment"
    ].dropna().tolist()

    if len(wontfix_std_vals) > 0 and len(non_wontfix_std_vals) > 0:
        draw_boxplot(
            wontfix_std_vals,
            non_wontfix_std_vals,
            "WONTFIX",
            "Non-WONTFIX",
            "Issue-Level Sentiment Variability (3 Repos)",
            "Std Comment Sentiment",
            output_dir / "boxplot_std_comment_sentiment_3repos.png"
        )

print("\nSaved plots to:")
print(output_dir / "boxplot_mean_comment_sentiment_3repos.png")
if has_std:
    print(output_dir / "boxplot_std_comment_sentiment_3repos.png")