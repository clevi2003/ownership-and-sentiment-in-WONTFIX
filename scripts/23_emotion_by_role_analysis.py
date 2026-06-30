import sys
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EMOTION_FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "emotion_features.parquet"
WONTFIX_ISSUES_PATH = PROJECT_ROOT / "data" / "final" / "issue_sets" / "wontfix_issue_set.parquet"
COMPARISON_ISSUES_PATH = PROJECT_ROOT / "data" / "final" / "issue_sets" / "comparison_issue_set.parquet"
OUTPUT_CSV_PATH = PROJECT_ROOT / "outputs" / "sentiment_analysis" / "emotion_by_role_breakdown.csv"

def load_and_combine_issues():
    """Loads wontfix and comparison issues to get the issue_author_login."""
    wontfix_df = pd.read_parquet(WONTFIX_ISSUES_PATH)
    comp_df = pd.read_parquet(COMPARISON_ISSUES_PATH)
    
    wontfix_df["analysis_set"] = "wontfix"
    comp_df["analysis_set"] = "comparison"
    
    combined = pd.concat([wontfix_df, comp_df], ignore_index=True)
    
    repo_col = "__repo" if "__repo" in combined.columns else "repo_full_name"
    num_col = "__issue_number" if "__issue_number" in combined.columns else "issue_number"
    author_col = "author_login"
    
    return combined[[repo_col, num_col, author_col, "analysis_set"]].rename(
        columns={
            repo_col: "repo_full_name", 
            num_col: "issue_number", 
            author_col: "issue_author_login"
        }
    ).drop_duplicates()

def main():
    print("Loading emotion features and issue sets...")
    emotions_df = pd.read_parquet(EMOTION_FEATURES_PATH)
    issues_df = load_and_combine_issues()
    
    comments_only = emotions_df[emotions_df["text_source"] == "issue_comment"].copy()
    
    merged_df = comments_only.merge(
        issues_df, 
        on=["repo_full_name", "issue_number"], 
        how="inner"
    )
    
    print(f"Successfully merged {len(merged_df)} comment emotion rows with issue author data.")
    
    merged_df["is_issue_author"] = merged_df["author_login"] == merged_df["issue_author_login"]
    merged_df["commenter_role"] = merged_df["is_issue_author"].map({True: "Issue Author", False: "Non-Author"})
    
    non_neutral = merged_df[merged_df["dominant_emotion"] != "neutral"].copy()
    
    print("Calculating role-based distributions...")
    breakdown = non_neutral.groupby(
        ["analysis_set", "commenter_role", "dominant_emotion"]
    ).size().reset_index(name="comment_count")
    
    group_totals = breakdown.groupby(["analysis_set", "commenter_role"])["comment_count"].transform('sum')
    breakdown["share_within_role"] = (breakdown["comment_count"] / group_totals).round(4)
    
    breakdown = breakdown.sort_values(
        by=["analysis_set", "commenter_role", "comment_count"], 
        ascending=[True, True, False]
    )
    
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    breakdown.to_csv(OUTPUT_CSV_PATH, index=False)
    
    print(f"\nAnalysis complete! Breakdown saved to: {OUTPUT_CSV_PATH}")
    print("\n--- Quick Preview of WONTFIX Non-Author vs Author Emotions ---")
    preview = breakdown[breakdown["analysis_set"] == "wontfix"]
    print(preview.to_string(index=False))

if __name__ == "__main__":
    main()
