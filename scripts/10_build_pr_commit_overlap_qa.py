import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.study_config_loader import ensure_project_directories, load_study_config
from utils.checkpoints import get_stage_option
from utils.io_helpers import load_repo_list, load_table, repo_filter, clean_text, write_summary_csv

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"
LOG_FILENAME = "10_build_pr_commit_overlap_qa.log"


def setup_logger(config):
    logger = logging.getLogger("build_pr_commit_overlap_qa")
    logger.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if config.logging.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.logging.log_to_file:
        log_dir = Path(config.logging.qa_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / LOG_FILENAME, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_overlap_option(config, field_name, default_value):
    """
    Reuse ownership_features runtime controls for now so this script can slot into the
    existing project without requiring a new config section immediately.
    """
    return get_stage_option(config, "ownership_features", field_name, default_value)


def get_stage_paths(config):
    outputs = getattr(config, "outputs", None)

    qa_summary_path = getattr(outputs, "pr_commit_overlap_qa_summary_csv", None)
    if not qa_summary_path:
        qa_summary_path = "./logs/qa/pr_commit_overlap_qa_summary.csv"

    return {
        "qa_summary_path": Path(qa_summary_path),
        "run_manifest_path": Path(config.logging.qa_log_dir) / "10_build_pr_commit_overlap_qa_run_manifest.json",
    }


def load_stage_inputs_for_repo(config, repo_full_name):
    merge_mode = getattr(config.storage, "processed_merge_mode", "single_parquet")

    prs_df = load_table(
        config.outputs.pull_requests_table,
        repo_full_name=repo_full_name,
        merge_mode=merge_mode,
    )
    pr_commit_df = load_table(
        config.outputs.pr_commit_links_table,
        repo_full_name=repo_full_name,
        merge_mode=merge_mode,
    )
    commits_df = load_table(
        config.outputs.commits_table,
        repo_full_name=repo_full_name,
        merge_mode=merge_mode,
    )

    return {
        "pull_requests": repo_filter(prs_df, repo_full_name),
        "pr_commit_links": repo_filter(pr_commit_df, repo_full_name),
        "commits": repo_filter(commits_df, repo_full_name),
    }


def new_repo_result(repo_full_name, repo_id=None):
    return {
        "repo_full_name": repo_full_name,
        "repo_id": repo_id,
        "status": "started",
        "pr_rows_seen": 0,
        "pr_unique_ids_seen": 0,
        "pr_commit_link_rows_seen": 0,
        "pr_commit_unique_shas_seen": 0,
        "commit_rows_seen": 0,
        "commit_unique_shas_seen": 0,
        "pr_commit_shas_present_in_commits": 0,
        "pr_commit_sha_overlap_rate": None,
        "pr_head_sha_non_null_count": 0,
        "pr_head_sha_present_count": 0,
        "pr_head_sha_present_rate": None,
        "pr_merge_commit_sha_non_null_count": 0,
        "pr_merge_commit_sha_present_count": 0,
        "pr_merge_commit_sha_present_rate": None,
        "error_message": "",
    }


def normalize_sha_series(series):
    if series is None:
        return pd.Series(dtype="object")
    cleaned = series.apply(clean_text)
    cleaned = cleaned[cleaned.notna()].astype(str)
    return cleaned


def build_commit_sha_set(commits_df):
    if commits_df is None or commits_df.empty or "commit_sha" not in commits_df.columns:
        return set()
    return set(normalize_sha_series(commits_df["commit_sha"]).tolist())


def count_unique_non_null(df, column_name):
    if df is None or df.empty or column_name not in df.columns:
        return 0
    return int(normalize_sha_series(df[column_name]).nunique())


def process_repo(config, logger, repo_row):
    repo_full_name = repo_row["full_name"]
    repo_id = repo_row.get("repo_id")
    result = new_repo_result(repo_full_name, repo_id=repo_id)

    stage_inputs = load_stage_inputs_for_repo(config, repo_full_name)
    prs_df = stage_inputs["pull_requests"]
    pr_commit_df = stage_inputs["pr_commit_links"]
    commits_df = stage_inputs["commits"]

    result["pr_rows_seen"] = int(len(prs_df))
    if prs_df is not None and not prs_df.empty:
        if "pr_id" in prs_df.columns:
            result["pr_unique_ids_seen"] = int(prs_df["pr_id"].dropna().nunique())
        else:
            result["pr_unique_ids_seen"] = int(len(prs_df))

    result["pr_commit_link_rows_seen"] = int(len(pr_commit_df))
    result["commit_rows_seen"] = int(len(commits_df))

    commit_sha_set = build_commit_sha_set(commits_df)
    result["commit_unique_shas_seen"] = int(len(commit_sha_set))

    pr_commit_sha_set = set()
    if pr_commit_df is not None and not pr_commit_df.empty and "commit_sha" in pr_commit_df.columns:
        pr_commit_sha_set = set(normalize_sha_series(pr_commit_df["commit_sha"]).tolist())
    result["pr_commit_unique_shas_seen"] = int(len(pr_commit_sha_set))

    overlap_shas = pr_commit_sha_set & commit_sha_set
    result["pr_commit_shas_present_in_commits"] = int(len(overlap_shas))
    result["pr_commit_sha_overlap_rate"] = (
        float(len(overlap_shas)) / float(len(pr_commit_sha_set))
        if pr_commit_sha_set
        else None
    )

    if prs_df is not None and not prs_df.empty:
        if "head_sha" in prs_df.columns:
            head_sha_set = set(normalize_sha_series(prs_df["head_sha"]).tolist())
            result["pr_head_sha_non_null_count"] = int(len(head_sha_set))
            head_present = head_sha_set & commit_sha_set
            result["pr_head_sha_present_count"] = int(len(head_present))
            result["pr_head_sha_present_rate"] = (
                float(len(head_present)) / float(len(head_sha_set))
                if head_sha_set
                else None
            )

        if "merge_commit_sha" in prs_df.columns:
            merge_sha_set = set(normalize_sha_series(prs_df["merge_commit_sha"]).tolist())
            result["pr_merge_commit_sha_non_null_count"] = int(len(merge_sha_set))
            merge_present = merge_sha_set & commit_sha_set
            result["pr_merge_commit_sha_present_count"] = int(len(merge_present))
            result["pr_merge_commit_sha_present_rate"] = (
                float(len(merge_present)) / float(len(merge_sha_set))
                if merge_sha_set
                else None
            )

    result["status"] = "completed"

    logger.info(
        "PR-commit overlap | repo=%s | pr_commit_unique=%s | commit_unique=%s | pr_commit_overlap_rate=%s | head_rate=%s | merge_rate=%s",
        repo_full_name,
        result["pr_commit_unique_shas_seen"],
        result["commit_unique_shas_seen"],
        result["pr_commit_sha_overlap_rate"],
        result["pr_head_sha_present_rate"],
        result["pr_merge_commit_sha_present_rate"],
    )

    return result


def write_run_manifest(repo_rows, summary_rows, stage_paths):
    manifest_path = Path(stage_paths["run_manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "script": "10_build_pr_commit_overlap_qa.py",
        "repo_count_requested": len(repo_rows),
        "repo_count_processed": len(summary_rows),
        "completed_repo_count": sum(1 for row in summary_rows if row.get("status") == "completed"),
        "failed_repo_count": sum(1 for row in summary_rows if row.get("status") == "failed"),
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary_rows": summary_rows,
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main(config_path=None):
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = load_study_config(config_path)
    ensure_project_directories(config)

    logger = setup_logger(config)
    stage_paths = get_stage_paths(config)

    logger.info("Loaded config from %s", config_path)

    repo_rows = load_repo_list(config.outputs.repo_included_list)

    max_repos_per_run = get_overlap_option(config, "max_repos_per_run", None)
    if max_repos_per_run and max_repos_per_run > 0:
        repo_rows = repo_rows[:max_repos_per_run]

    summary_rows = []
    for repo_row in repo_rows:
        repo_full_name = repo_row["full_name"]
        logger.info("Starting PR-commit overlap QA for %s", repo_full_name)
        try:
            result = process_repo(config, logger, repo_row)
        except Exception as exc:
            logger.exception("Failed while building PR-commit overlap QA for %s", repo_full_name)
            result = new_repo_result(repo_full_name, repo_row.get("repo_id"))
            result["status"] = "failed"
            result["error_message"] = str(exc)

        summary_rows.append(result)

    write_summary_csv(summary_rows, stage_paths["qa_summary_path"])
    write_run_manifest(repo_rows, summary_rows, stage_paths)

    logger.info(
        "PR-commit overlap QA complete | repos_requested=%s | repos_processed=%s",
        len(repo_rows),
        len(summary_rows),
    )


if __name__ == "__main__":
    main()
