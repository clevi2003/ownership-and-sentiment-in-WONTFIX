from pathlib import Path
import pandas as pd

from config.study_config_loader import load_study_config


def main():
    config = load_study_config()

    issues_path = Path(config["outputs"]["issues_table"])
    wontfix_output_path = Path(config["outputs"]["wontfix_issue_set_table"])
    comparison_output_path = Path(config["outputs"]["comparison_issue_set_table"])
    qa_summary_path = Path(config["outputs"]["comparison_issue_qa_summary_csv"])

    print("Loaded config successfully.")
    print(f"Issues input path: {issues_path}")
    print(f"WONTFIX output path: {wontfix_output_path}")
    print(f"Comparison output path: {comparison_output_path}")
    print(f"QA summary path: {qa_summary_path}")

    if not issues_path.exists():
        raise FileNotFoundError(f"Issues table not found: {issues_path}")

    issues_df = pd.read_parquet(issues_path)

    print(f"Loaded issues table with {len(issues_df)} rows.")
    print("Columns:")
    print(list(issues_df.columns))


if __name__ == "__main__":
    main()
