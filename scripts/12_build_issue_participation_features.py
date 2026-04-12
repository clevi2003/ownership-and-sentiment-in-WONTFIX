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
from utils.checkpoints import (
    get_batch_root,
    get_stage_option,
    reset_batch_root,
    sanitize_repo_name,
    should_skip_repo,
    write_repo_checkpoint,
)
from utils.io_helpers import (
    clean_text,
    collect_repo_part_files,
    load_repo_list,
    load_table,
    repo_filter,
    safe_divide,
    safe_to_datetime,
    take_mean,
    take_median,
    write_merged_or_partitioned_output,
    write_summary_csv,
)
from utils.chunk_writers import ParticipationFeatureRepoChunkWriter

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"
LOG_FILENAME = "12_build_issue_participation_features.log"
CHECKPOINT_PREFIX = "12_build_issue_participation_features"
BATCH_FOLDER_NAME = "participation_features"
RAW_FOLDER_NAME = "participation_features"


def setup_logger(config):
    logger = logging.getLogger("build_issue_participation_features")
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


def get_participation_option(config, field_name, default_value):
    return get_stage_option(config, "participation_features", field_name, default_value)


def get_stage_paths(config):
    outputs = getattr(config, "outputs", None)
    issue_output_path = getattr(outputs, "issue_participation_features_table", None)
    if not issue_output_path:
        issue_output_path = "./data/features/participation/issue_participation_features.parquet"
    qa_summary_path = getattr(outputs, "participation_feature_qa_summary_csv", None)
    if not qa_summary_path:
        qa_summary_path = "./logs/qa/issue_participation_feature_qa_summary.csv"
    return {
        "issue_output_path": Path(issue_output_path),
        "qa_summary_path": Path(qa_summary_path),
        "run_manifest_path": Path(config.logging.qa_log_dir) / "12_build_issue_participation_features_run_manifest.json",
    }


def find_first_present_column(df, candidates):
    lower_map = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def normalize_issue_set_columns(df):
    if df.empty:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number", "analysis_set"])
    normalized = df.copy()
    repo_col = find_first_present_column(normalized, ["repo_full_name", "repo_name", "full_name", "repo"])
    issue_id_col = find_first_present_column(normalized, ["issue_id", "id"])
    issue_number_col = find_first_present_column(normalized, ["issue_number", "number"])
    if repo_col is None:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number", "analysis_set"])

    out_df = pd.DataFrame()
    out_df["repo_full_name"] = normalized[repo_col].astype(str)
    out_df["issue_id"] = normalized[issue_id_col].astype(str) if issue_id_col else None
    out_df["issue_number"] = pd.to_numeric(normalized[issue_number_col], errors="coerce") if issue_number_col else None
    if "analysis_set" in normalized.columns:
        out_df["analysis_set"] = normalized["analysis_set"].astype(str)
    else:
        out_df["analysis_set"] = None
    return out_df.drop_duplicates().reset_index(drop=True)


def build_target_issue_lookup(config):
    merge_mode = getattr(config.storage, "processed_merge_mode", "single_parquet")
    wontfix_df = load_table(config.outputs.wontfix_issue_set_table, merge_mode=merge_mode)
    comparison_df = load_table(config.outputs.comparison_issue_set_table, merge_mode=merge_mode)
    wontfix_df = normalize_issue_set_columns(wontfix_df)
    comparison_df = normalize_issue_set_columns(comparison_df)

    if not wontfix_df.empty:
        wontfix_df["analysis_set"] = "wontfix"
    if not comparison_df.empty:
        comparison_df["analysis_set"] = "comparison"
    if wontfix_df.empty and comparison_df.empty:
        return {}

    combined = pd.concat([wontfix_df, comparison_df], ignore_index=True)
    combined = combined.sort_values(["repo_full_name", "analysis_set"], kind="stable").reset_index(drop=True)
    lookup = {}
    for row in combined.to_dict(orient="records"):
        repo_full_name = row.get("repo_full_name")
        if not repo_full_name:
            continue
        repo_payload = lookup.setdefault(repo_full_name, {"by_issue_id": {}, "by_issue_number": {}})
        issue_id = clean_text(row.get("issue_id"))
        issue_number = row.get("issue_number")
        analysis_set = row.get("analysis_set")
        if issue_id:
            repo_payload["by_issue_id"][issue_id] = analysis_set
        if pd.notna(issue_number):
            repo_payload["by_issue_number"][int(issue_number)] = analysis_set
    return lookup


def build_repo_id_lookup(config):
    merge_mode = getattr(config.storage, "processed_merge_mode", "single_parquet")
    repos_df = load_table(config.outputs.repositories_table, merge_mode=merge_mode)
    if repos_df.empty:
        return {}

    repo_full_name_col = find_first_present_column(repos_df, ["repo_full_name", "full_name"])
    repo_id_col = find_first_present_column(repos_df, ["repo_id", "id"])
    if repo_full_name_col is None or repo_id_col is None:
        return {}

    return {
        row[repo_full_name_col]: row[repo_id_col]
        for row in repos_df.to_dict(orient="records")
        if row.get(repo_full_name_col) and row.get(repo_id_col) is not None
    }


def load_stage_inputs_for_repo(config, repo_full_name):
    merge_mode = getattr(config.storage, "processed_merge_mode", "single_parquet")
    issues_df = load_table(config.outputs.issues_resolved_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    comments_df = load_table(config.outputs.issue_comments_resolved_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    return {
        "issues_resolved": repo_filter(issues_df, repo_full_name),
        "issue_comments_resolved": repo_filter(comments_df, repo_full_name),
    }


def new_repo_result(repo_full_name, repo_id=None):
    return {
        "repo_full_name": repo_full_name,
        "repo_id": repo_id,
        "status": "started",
        "target_issues_requested": 0,
        "issues_resolved_rows_seen": 0,
        "issue_comments_resolved_rows_seen": 0,
        "target_issues_kept": 0,
        "issues_with_zero_comments": 0,
        "issues_with_resolved_comments": 0,
        "issues_with_multiple_commenters": 0,
        "issues_with_only_author_comments": 0,
        "issues_ok": 0,
        "issues_zero_comments": 0,
        "issues_no_resolved_comment_authors": 0,
        "issues_missing_issue_author_key": 0,
        "issue_rows_written": 0,
        "median_comment_count": 0.0,
        "median_unique_commenter_count": 0.0,
        "mean_top_commenter_share": None,
        "share_issue_author_commented": None,
        "share_single_commenter": None,
        "share_only_author_commented": None,
        "error_message": "",
    }


def attach_analysis_set(issues_df, repo_lookup):
    if issues_df.empty:
        return issues_df
    df = issues_df.copy()
    df["analysis_set"] = None

    for index, row in df.iterrows():
        issue_id = clean_text(row.get("issue_id"))
        issue_number = row.get("issue_number")
        analysis_set = None
        if issue_id and issue_id in repo_lookup.get("by_issue_id", {}):
            analysis_set = repo_lookup["by_issue_id"].get(issue_id)
        elif pd.notna(issue_number):
            analysis_set = repo_lookup.get("by_issue_number", {}).get(int(issue_number))
        df.at[index, "analysis_set"] = analysis_set

    df = df[df["analysis_set"].notna()].copy()
    if df.empty:
        return df
    return df.reset_index(drop=True)


def prepare_issue_frame(issues_df):
    if issues_df.empty:
        return issues_df
    df = issues_df.copy()
    needed_columns = [
        "repo_id",
        "repo_full_name",
        "issue_id",
        "issue_number",
        "state",
        "created_at",
        "closed_at",
        "author_login",
        "issue_author_contributor_key",
        "analysis_set",
    ]
    existing_columns = [column for column in needed_columns if column in df.columns]
    df = df[existing_columns].copy()
    if "issue_number" in df.columns:
        df["issue_number"] = pd.to_numeric(df["issue_number"], errors="coerce")
    if "created_at" in df.columns:
        df["created_at"] = safe_to_datetime(df["created_at"])
    else:
        df["created_at"] = pd.NaT
    if "closed_at" in df.columns:
        df["closed_at"] = safe_to_datetime(df["closed_at"])
    else:
        df["closed_at"] = pd.NaT
    return df.reset_index(drop=True)

def backfill_repo_id(issues_df, repo_id):
    if issues_df.empty:
        return issues_df

    df = issues_df.copy()
    if "repo_id" not in df.columns:
        df["repo_id"] = repo_id
        return df

    if repo_id is None:
        return df

    missing_mask = df["repo_id"].isna()
    if missing_mask.any():
        df.loc[missing_mask, "repo_id"] = repo_id
    return df


def prepare_comment_frame(comments_df, target_issue_numbers):
    if comments_df.empty:
        return comments_df
    df = comments_df.copy()
    needed_columns = [
        "repo_full_name",
        "issue_number",
        "comment_id",
        "author_login",
        "comment_author_contributor_key",
        "created_at",
    ]
    existing_columns = [column for column in needed_columns if column in df.columns]
    df = df[existing_columns].copy()
    if "issue_number" in df.columns:
        df["issue_number"] = pd.to_numeric(df["issue_number"], errors="coerce")
        df = df[df["issue_number"].notna()].copy()
        df["issue_number"] = df["issue_number"].astype(int)
    df = df[df["issue_number"].isin(target_issue_numbers)].copy()
    if "created_at" in df.columns:
        df["created_at"] = safe_to_datetime(df["created_at"])
    else:
        df["created_at"] = pd.NaT
    return df.reset_index(drop=True)


def get_effective_commenter_key(comment_row):
    return clean_text(comment_row.get("comment_author_contributor_key")) or clean_text(comment_row.get("author_login"))


def build_issue_participation_row(issue_row, issue_comments_df):
    issue_author_key = clean_text(issue_row.get("issue_author_contributor_key")) or clean_text(issue_row.get("author_login"))
    comments_df = issue_comments_df.copy() if issue_comments_df is not None else pd.DataFrame()
    if not comments_df.empty:
        comments_df = comments_df.sort_values(["created_at", "comment_id"], kind="stable", na_position="last").reset_index(drop=True)

    comment_rows = comments_df.to_dict(orient="records") if not comments_df.empty else []
    comment_count = len(comment_rows)

    resolved_comment_count = 0
    commenter_counts = {}
    for comment_row in comment_rows:
        commenter_key = get_effective_commenter_key(comment_row)
        if not commenter_key:
            continue
        resolved_comment_count += 1
        commenter_counts[commenter_key] = commenter_counts.get(commenter_key, 0) + 1

    missing_resolved_comment_count = comment_count - resolved_comment_count
    unique_commenter_count = len(commenter_counts)
    sorted_commenter_counts = sorted(commenter_counts.values(), reverse=True)
    top_commenter_share = safe_divide(sorted_commenter_counts[0] if sorted_commenter_counts else 0, resolved_comment_count, default_value=None)
    top_2_commenters_share = safe_divide(sum(sorted_commenter_counts[:2]), resolved_comment_count, default_value=None)
    comment_concentration_ratio = top_2_commenters_share
    repeat_commenter_count = sum(1 for count in commenter_counts.values() if count > 1)
    mean_comments_per_commenter = safe_divide(resolved_comment_count, unique_commenter_count, default_value=None)

    issue_author_commented_flag = None
    num_distinct_non_author_commenters = 0
    non_author_comment_count = 0
    non_author_comment_share = None
    only_author_commented_flag = None
    first_comment_by_author_flag = None
    last_comment_by_author_flag = None

    coverage_flag = "ok"
    if comment_count == 0:
        coverage_flag = "zero_comments"
    elif resolved_comment_count == 0:
        coverage_flag = "no_resolved_comment_authors"
    elif not issue_author_key:
        coverage_flag = "missing_issue_author_key"

    if resolved_comment_count > 0:
        if issue_author_key:
            issue_author_commented_flag = 1 if issue_author_key in commenter_counts else 0
            num_distinct_non_author_commenters = len([key for key in commenter_counts if key != issue_author_key])
            non_author_comment_count = sum(count for key, count in commenter_counts.items() if key != issue_author_key)
            non_author_comment_share = safe_divide(non_author_comment_count, resolved_comment_count, default_value=None)
            only_author_commented_flag = 1 if issue_author_commented_flag == 1 and num_distinct_non_author_commenters == 0 else 0
        else:
            num_distinct_non_author_commenters = unique_commenter_count
            non_author_comment_count = resolved_comment_count
            non_author_comment_share = safe_divide(non_author_comment_count, resolved_comment_count, default_value=None)

        if unique_commenter_count > 0:
            single_commenter_flag = 1 if unique_commenter_count == 1 else 0
        else:
            single_commenter_flag = None

        ordered_commenter_keys = [get_effective_commenter_key(row) for row in comment_rows if get_effective_commenter_key(row)]
        if issue_author_key and ordered_commenter_keys:
            first_comment_by_author_flag = 1 if ordered_commenter_keys[0] == issue_author_key else 0
            last_comment_by_author_flag = 1 if ordered_commenter_keys[-1] == issue_author_key else 0
    else:
        single_commenter_flag = None

    row = {
        "repo_id": issue_row.get("repo_id"),
        "repo_full_name": issue_row.get("repo_full_name"),
        "issue_id": issue_row.get("issue_id"),
        "issue_number": issue_row.get("issue_number"),
        "analysis_set": issue_row.get("analysis_set"),
        "issue_created_at": issue_row.get("created_at"),
        "issue_closed_at": issue_row.get("closed_at"),
        "issue_author_contributor_key": issue_author_key,
        "comment_count": comment_count,
        "comments_with_resolved_author_count": resolved_comment_count,
        "comments_missing_resolved_author_count": missing_resolved_comment_count,
        "unique_commenter_count": unique_commenter_count,
        "participation_feature_coverage_flag": coverage_flag,
        "issue_author_commented_flag": issue_author_commented_flag,
        "num_distinct_non_author_commenters": num_distinct_non_author_commenters,
        "non_author_comment_count": non_author_comment_count,
        "non_author_comment_share": non_author_comment_share,
        "top_commenter_share": top_commenter_share,
        "top_2_commenters_share": top_2_commenters_share,
        "comment_concentration_ratio": comment_concentration_ratio,
        "repeat_commenter_count": repeat_commenter_count,
        "mean_comments_per_commenter": mean_comments_per_commenter,
        "single_commenter_flag": single_commenter_flag,
        "only_author_commented_flag": only_author_commented_flag,
        "first_comment_by_author_flag": first_comment_by_author_flag,
        "last_comment_by_author_flag": last_comment_by_author_flag,
    }
    return row


def mean_or_none(column_name, df):
    values = [
        value
        for value in df[column_name].tolist()
        if value is not None and not pd.isna(value)
    ]
    if not values:
        return None
    return take_mean(values)

def process_repo(config, logger, repo_row, target_issue_lookup, repo_id_lookup):
    repo_full_name = repo_row["full_name"]
    repo_id = repo_row.get("repo_id") or repo_id_lookup.get(repo_full_name)
    repo_lookup = target_issue_lookup.get(repo_full_name, {"by_issue_id": {}, "by_issue_number": {}})
    result = new_repo_result(repo_full_name, repo_id)

    stage_inputs = load_stage_inputs_for_repo(config, repo_full_name)
    issues_df = stage_inputs["issues_resolved"]
    comments_df = stage_inputs["issue_comments_resolved"]
    result["issues_resolved_rows_seen"] = len(issues_df)
    result["issue_comments_resolved_rows_seen"] = len(comments_df)
    result["target_issues_requested"] = len(repo_lookup.get("by_issue_id", {})) or len(repo_lookup.get("by_issue_number", {}))

    if issues_df.empty:
        result["status"] = "completed"
        return result

    issues_df = attach_analysis_set(issues_df, repo_lookup)
    issues_df = prepare_issue_frame(issues_df)
    issues_df = backfill_repo_id(issues_df, repo_id)
    if issues_df.empty:
        result["status"] = "completed"
        return result

    result["target_issues_kept"] = len(issues_df)
    target_issue_numbers = {
        int(value) for value in issues_df["issue_number"].dropna().tolist()
    } if "issue_number" in issues_df.columns else set()
    comments_df = prepare_comment_frame(comments_df, target_issue_numbers)

    comments_by_issue_number = {}
    if not comments_df.empty:
        for row in comments_df.to_dict(orient="records"):
            issue_number = row.get("issue_number")
            if pd.isna(issue_number):
                continue
            comments_by_issue_number.setdefault(int(issue_number), []).append(row)

    repo_dir = Path(config.paths.processed_root) / "_batches" / BATCH_FOLDER_NAME / sanitize_repo_name(repo_full_name)
    writer = ParticipationFeatureRepoChunkWriter(
        config=config,
        repo_dir=repo_dir,
        batch_size=get_participation_option(config, "write_batch_size", 5000),
    )

    issue_rows = []
    for issue_row in issues_df.to_dict(orient="records"):
        issue_number = int(issue_row.get("issue_number")) if pd.notna(issue_row.get("issue_number")) else None
        issue_comments_df = pd.DataFrame(comments_by_issue_number.get(issue_number, []))
        feature_row = build_issue_participation_row(issue_row, issue_comments_df)
        writer.add_issue_participation_row(feature_row)
        issue_rows.append(feature_row)
        result["issue_rows_written"] += 1

    writer.finalize()

    if issue_rows:
        issue_feature_df = pd.DataFrame(issue_rows)
        result["issues_with_zero_comments"] = int((issue_feature_df["comment_count"] == 0).sum())
        result["issues_with_resolved_comments"] = int((issue_feature_df["comments_with_resolved_author_count"] > 0).sum())
        result["issues_with_multiple_commenters"] = int((issue_feature_df["unique_commenter_count"] > 1).sum())
        result["issues_with_only_author_comments"] = int((issue_feature_df["only_author_commented_flag"] == 1).sum())
        result["issues_ok"] = int((issue_feature_df["participation_feature_coverage_flag"] == "ok").sum())
        result["issues_zero_comments"] = int((issue_feature_df["participation_feature_coverage_flag"] == "zero_comments").sum())
        result["issues_no_resolved_comment_authors"] = int((issue_feature_df["participation_feature_coverage_flag"] == "no_resolved_comment_authors").sum())
        result["issues_missing_issue_author_key"] = int((issue_feature_df["participation_feature_coverage_flag"] == "missing_issue_author_key").sum())
        result["median_comment_count"] = float(take_median(issue_feature_df["comment_count"].tolist()))
        result["median_unique_commenter_count"] = float(take_median(issue_feature_df["unique_commenter_count"].tolist()))
        result["mean_top_commenter_share"] = mean_or_none("top_commenter_share", issue_feature_df)
        result["share_issue_author_commented"] = mean_or_none("issue_author_commented_flag", issue_feature_df)
        result["share_single_commenter"] = mean_or_none("single_commenter_flag", issue_feature_df)
        result["share_only_author_commented"] = mean_or_none("only_author_commented_flag", issue_feature_df)
    result["status"] = "completed"
    return result


def merge_participation_feature_batches(config, logger, stage_paths):
    batch_root = get_batch_root(config, BATCH_FOLDER_NAME)
    if not batch_root.exists():
        logger.warning("Participation feature batch root does not exist: %s", batch_root)
        return

    issue_repo_parts = collect_repo_part_files(batch_root, "issue_participation_features_part_*.parquet")
    if issue_repo_parts:
        mode_used = write_merged_or_partitioned_output(
            repo_part_map=issue_repo_parts,
            output_path=stage_paths["issue_output_path"],
            config=config,
            table_name="issue_participation_features",
            sort_columns=["repo_full_name", "issue_number"],
            dedupe_subset=["repo_full_name", "issue_id", "issue_number"],
        )
        logger.info("Wrote issue participation features using %s mode to %s", mode_used, stage_paths["issue_output_path"])
    else:
        logger.warning("No issue participation feature parts found to merge.")


def write_run_manifest(repo_rows, summary_rows, stage_paths):
    manifest_path = Path(stage_paths["run_manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "build_issue_participation_features.py",
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
    repo_id_lookup = build_repo_id_lookup(config)

    repo_rows = load_repo_list(config.outputs.repo_included_list)
    target_issue_lookup = build_target_issue_lookup(config)
    batch_root = reset_batch_root(config, BATCH_FOLDER_NAME)
    logger.info("Reset batch root: %s", batch_root)

    max_repos_per_run = get_participation_option(config, "max_repos_per_run", None)
    if max_repos_per_run and max_repos_per_run > 0:
        repo_rows = repo_rows[:max_repos_per_run]

    summary_rows = []
    for repo_row in repo_rows:
        repo_full_name = repo_row["full_name"]
        should_skip, reason = should_skip_repo(
            config,
            repo_full_name,
            checkpoint_prefix=CHECKPOINT_PREFIX,
            raw_folder_name=RAW_FOLDER_NAME,
            section_name="participation_features",
            raw_source="features",
        )
        if should_skip:
            logger.info("Skipping repo %s due to %s.", repo_full_name, reason)
            summary_rows.append(
                {
                    "repo_full_name": repo_full_name,
                    "repo_id": repo_row.get("repo_id"),
                    "status": f"skipped_{reason}",
                }
            )
            continue

        logger.info("Processing repo %s", repo_full_name)
        try:
            result = process_repo(config, logger, repo_row, target_issue_lookup, repo_id_lookup)
        except Exception as exc:
            logger.exception("Failed while building participation features for %s", repo_full_name)
            result = new_repo_result(repo_full_name, repo_row.get("repo_id"))
            result["status"] = "failed"
            result["error_message"] = str(exc)

        write_repo_checkpoint(config, CHECKPOINT_PREFIX, repo_full_name, result)
        summary_rows.append(result)

    merge_participation_feature_batches(config, logger, stage_paths)
    write_summary_csv(summary_rows, stage_paths["qa_summary_path"])
    write_run_manifest(repo_rows, summary_rows, stage_paths)
    logger.info("Participation feature building complete. Repos processed: %s", len(summary_rows))


if __name__ == "__main__":
    main()
