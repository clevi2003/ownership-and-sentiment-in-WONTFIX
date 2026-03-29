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
from utils.checkpoints import get_batch_root, get_stage_option, reset_batch_root, sanitize_repo_name, should_skip_repo, \
    write_repo_checkpoint
from utils.chunk_writers import ResolvedEntityRepoChunkWriter
from utils.io_helpers import clean_text, collect_repo_part_files, has_real_value, load_repo_list, load_table, \
    repo_filter, write_merged_or_partitioned_output

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"
LOG_FILENAME = "07_attach_resolved_contributor_keys.log"
CHECKPOINT_PREFIX = "07_attach_resolved_contributor_keys"
BATCH_FOLDER_NAME = "resolved_entities"
RAW_FOLDER_NAME = "resolved_entities"
NULL_JOIN_SENTINEL = "__NULL__"


def setup_logger(config):
    logger = logging.getLogger("attach_resolved_contributor_keys")
    logger.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if config.logging.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    if config.logging.log_to_file:
        log_dir = Path(config.logging.linkage_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / LOG_FILENAME, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_attachment_option(config, field_name, default_value):
    return get_stage_option(config, "identity_resolution", field_name, default_value)


def resolve_repo_id_from_stage_inputs(stage_inputs, fallback_repo_id=None):
    if pd.notna(fallback_repo_id):
        return fallback_repo_id
    for table_name in ["issues", "issue_comments", "pull_requests", "commits", "identity_map"]:
        df = stage_inputs.get(table_name)
        if df is None or df.empty or "repo_id" not in df.columns:
            continue
        non_null_ids = df["repo_id"].dropna()
        if not non_null_ids.empty:
            return non_null_ids.iloc[0]
    return None


def load_stage_inputs_for_repo(config, repo_full_name):
    merge_mode = get_attachment_option(config, "input_merge_mode", None)
    issues_df = load_table(config.outputs.issues_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    comments_df = load_table(config.outputs.issue_comments_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    prs_df = load_table(config.outputs.pull_requests_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    commits_df = load_table(config.outputs.commits_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    identity_df = load_table(config.outputs.contributor_identity_table, repo_full_name=repo_full_name,
                             merge_mode=merge_mode)
    return {
        "issues": repo_filter(issues_df, repo_full_name),
        "issue_comments": repo_filter(comments_df, repo_full_name),
        "pull_requests": repo_filter(prs_df, repo_full_name),
        "commits": repo_filter(commits_df, repo_full_name),
        "identity_map": repo_filter(identity_df, repo_full_name),
    }


def new_repo_result(repo_full_name, repo_id=None):
    return {
        "repo_full_name": repo_full_name,
        "repo_id": repo_id,
        "status": "started",
        "identity_rows_seen": 0,
        "issues_rows_seen": 0,
        "issue_rows_with_author": 0,
        "issue_author_keys_attached": 0,
        "issue_rows_with_closer": 0,
        "issue_closer_keys_attached": 0,
        "issue_comments_rows_seen": 0,
        "comment_rows_with_author": 0,
        "comment_author_keys_attached": 0,
        "pull_requests_rows_seen": 0,
        "pr_rows_with_author": 0,
        "pr_author_keys_attached": 0,
        "commits_rows_seen": 0,
        "commit_rows_with_author": 0,
        "commit_author_keys_attached": 0,
        "issues_rows_written": 0,
        "issue_comments_rows_written": 0,
        "pull_requests_rows_written": 0,
        "commits_rows_written": 0,
        "ambiguous_identity_keys_dropped": 0,
        "error_message": "",
    }


def clean_join_value(value):
    cleaned = clean_text(value)
    if cleaned is None:
        return NULL_JOIN_SENTINEL
    return cleaned


def count_rows_with_any_actor(df, actor_columns):
    if df is None or df.empty:
        return 0
    mask = pd.Series(False, index=df.index)
    for column_name in actor_columns:
        if column_name not in df.columns:
            continue
        mask = mask | df[column_name].apply(has_real_value)
    return int(mask.sum())


def build_identity_lookup(identity_df, raw_source_type, join_field_pairs):
    if identity_df is None or identity_df.empty:
        return pd.DataFrame(), 0
    if "raw_source_type" not in identity_df.columns:
        return pd.DataFrame(), 0
    lookup_df = identity_df[identity_df["raw_source_type"] == raw_source_type].copy()
    if lookup_df.empty:
        return pd.DataFrame(), 0
    if "resolved_contributor_key" not in lookup_df.columns:
        return pd.DataFrame(), 0

    join_columns = ["repo_full_name"]
    for join_index, (_, identity_col) in enumerate(join_field_pairs):
        temp_col = f"__join_{join_index}"
        if identity_col in lookup_df.columns:
            lookup_df[temp_col] = lookup_df[identity_col].apply(clean_join_value)
        else:
            lookup_df[temp_col] = NULL_JOIN_SENTINEL
        join_columns.append(temp_col)

    lookup_df = lookup_df[join_columns + ["resolved_contributor_key"]].copy()
    lookup_df = lookup_df[lookup_df["resolved_contributor_key"].notna()].copy()
    if lookup_df.empty:
        return pd.DataFrame(), 0

    key_counts = (lookup_df.groupby(join_columns, dropna=False)["resolved_contributor_key"].nunique().reset_index(
        name="resolved_key_count"))
    valid_keys = key_counts[key_counts["resolved_key_count"] == 1][join_columns].copy()
    ambiguous_count = int((key_counts["resolved_key_count"] > 1).sum())
    if valid_keys.empty:
        return pd.DataFrame(columns=join_columns + ["resolved_contributor_key"]), ambiguous_count

    lookup_df = lookup_df.merge(valid_keys, on=join_columns, how="inner")
    lookup_df = lookup_df.drop_duplicates(subset=join_columns + ["resolved_contributor_key"]).copy()
    lookup_df = lookup_df.drop_duplicates(subset=join_columns, keep="first").reset_index(drop=True)
    return lookup_df, ambiguous_count


def attach_resolved_key_column(base_df, identity_df, raw_source_type, join_field_pairs, output_column):
    if base_df is None or base_df.empty:
        empty_df = pd.DataFrame(
            columns=list(base_df.columns) + [output_column]) if base_df is not None else pd.DataFrame()
        return empty_df, {"rows_with_actor": 0, "rows_attached": 0, "ambiguous_keys_dropped": 0}

    annotated_df = base_df.copy()
    annotated_df[output_column] = None

    actor_columns = [base_col for base_col, _ in join_field_pairs]
    rows_with_actor = count_rows_with_any_actor(annotated_df, actor_columns)
    if rows_with_actor == 0:
        return annotated_df, {"rows_with_actor": 0, "rows_attached": 0, "ambiguous_keys_dropped": 0}

    join_columns = ["repo_full_name"]
    temp_columns = []
    for join_index, (base_col, _) in enumerate(join_field_pairs):
        temp_col = f"__join_{join_index}"
        if base_col in annotated_df.columns:
            annotated_df[temp_col] = annotated_df[base_col].apply(clean_join_value)
        else:
            annotated_df[temp_col] = NULL_JOIN_SENTINEL
        join_columns.append(temp_col)
        temp_columns.append(temp_col)

    lookup_df, ambiguous_count = build_identity_lookup(identity_df, raw_source_type, join_field_pairs)
    if lookup_df.empty:
        annotated_df = annotated_df.drop(columns=temp_columns, errors="ignore")
        return annotated_df, {
            "rows_with_actor": rows_with_actor,
            "rows_attached": 0,
            "ambiguous_keys_dropped": ambiguous_count,
        }

    annotated_df = annotated_df.merge(lookup_df, on=join_columns, how="left")
    annotated_df[output_column] = annotated_df["resolved_contributor_key"]
    annotated_df = annotated_df.drop(columns=temp_columns + ["resolved_contributor_key"], errors="ignore")

    rows_attached = int(annotated_df[output_column].notna().sum())
    return annotated_df, {
        "rows_with_actor": rows_with_actor,
        "rows_attached": rows_attached,
        "ambiguous_keys_dropped": ambiguous_count,
    }


def annotate_issues(issues_df, identity_df):
    if issues_df is None:
        issues_df = pd.DataFrame()

    resolved_df = issues_df.copy()
    if "issue_author_contributor_key" not in resolved_df.columns:
        resolved_df["issue_author_contributor_key"] = None
    if "issue_closer_contributor_key" not in resolved_df.columns:
        resolved_df["issue_closer_contributor_key"] = None

    resolved_df, author_stats = attach_resolved_key_column(resolved_df, identity_df, "issue_author",
                                                           [("author_login", "raw_login")],
                                                           "issue_author_contributor_key")
    resolved_df, closer_stats = attach_resolved_key_column(resolved_df, identity_df, "issue_closer",
                                                           [("closed_by_login", "raw_login")],
                                                           "issue_closer_contributor_key")
    return resolved_df, author_stats, closer_stats


def annotate_issue_comments(comments_df, identity_df):
    if comments_df is None:
        comments_df = pd.DataFrame()
    resolved_df = comments_df.copy()
    if "comment_author_contributor_key" not in resolved_df.columns:
        resolved_df["comment_author_contributor_key"] = None

    resolved_df, author_stats = attach_resolved_key_column(resolved_df, identity_df, "issue_comment",
                                                           [("author_login", "raw_login")],
                                                           "comment_author_contributor_key")
    return resolved_df, author_stats


def annotate_pull_requests(prs_df, identity_df):
    if prs_df is None:
        prs_df = pd.DataFrame()
    resolved_df = prs_df.copy()
    if "pr_author_contributor_key" not in resolved_df.columns:
        resolved_df["pr_author_contributor_key"] = None

    resolved_df, author_stats = attach_resolved_key_column(resolved_df, identity_df, "pr_author",
                                                           [("author_login", "raw_login")], "pr_author_contributor_key")
    return resolved_df, author_stats


def annotate_commits(commits_df, identity_df):
    if commits_df is None:
        commits_df = pd.DataFrame()
    resolved_df = commits_df.copy()
    if "commit_author_contributor_key" not in resolved_df.columns:
        resolved_df["commit_author_contributor_key"] = None

    resolved_df, author_stats = attach_resolved_key_column(resolved_df, identity_df, "commit_author",
                                                           [("author_name", "raw_name"), ("author_email", "raw_email")],
                                                           "commit_author_contributor_key")
    return resolved_df, author_stats


def log_attachment_warning(logger, repo_full_name, entity_label, rows_with_actor, rows_attached):
    if rows_with_actor <= 0:
        return
    if rows_attached == rows_with_actor:
        return

    logger.warning(
        "Resolved contributor coverage gap | repo=%s | entity=%s | actor_rows=%s | attached=%s | missing=%s",
        repo_full_name,
        entity_label,
        rows_with_actor,
        rows_attached,
        rows_with_actor - rows_attached,
    )


def process_repo(config, logger, repo_row):
    repo_full_name = repo_row["full_name"]
    result = new_repo_result(repo_full_name, repo_row.get("repo_id"))

    stage_inputs = load_stage_inputs_for_repo(config, repo_full_name)
    repo_id = resolve_repo_id_from_stage_inputs(stage_inputs, fallback_repo_id=repo_row.get("repo_id"))
    result["repo_id"] = repo_id

    identity_df = stage_inputs["identity_map"]
    result["identity_rows_seen"] = len(identity_df)

    issues_resolved_df, issue_author_stats, issue_closer_stats = annotate_issues(stage_inputs["issues"], identity_df)
    comments_resolved_df, comment_author_stats = annotate_issue_comments(stage_inputs["issue_comments"], identity_df)
    prs_resolved_df, pr_author_stats = annotate_pull_requests(stage_inputs["pull_requests"], identity_df)
    commits_resolved_df, commit_author_stats = annotate_commits(stage_inputs["commits"], identity_df)

    result["issues_rows_seen"] = len(issues_resolved_df)
    result["issue_rows_with_author"] = issue_author_stats["rows_with_actor"]
    result["issue_author_keys_attached"] = issue_author_stats["rows_attached"]
    result["issue_rows_with_closer"] = issue_closer_stats["rows_with_actor"]
    result["issue_closer_keys_attached"] = issue_closer_stats["rows_attached"]

    result["issue_comments_rows_seen"] = len(comments_resolved_df)
    result["comment_rows_with_author"] = comment_author_stats["rows_with_actor"]
    result["comment_author_keys_attached"] = comment_author_stats["rows_attached"]

    result["pull_requests_rows_seen"] = len(prs_resolved_df)
    result["pr_rows_with_author"] = pr_author_stats["rows_with_actor"]
    result["pr_author_keys_attached"] = pr_author_stats["rows_attached"]

    result["commits_rows_seen"] = len(commits_resolved_df)
    result["commit_rows_with_author"] = commit_author_stats["rows_with_actor"]
    result["commit_author_keys_attached"] = commit_author_stats["rows_attached"]

    result["ambiguous_identity_keys_dropped"] = int(
        issue_author_stats["ambiguous_keys_dropped"]
        + issue_closer_stats["ambiguous_keys_dropped"]
        + comment_author_stats["ambiguous_keys_dropped"]
        + pr_author_stats["ambiguous_keys_dropped"]
        + commit_author_stats["ambiguous_keys_dropped"]
    )

    log_attachment_warning(logger, repo_full_name, "issues.issue_author", result["issue_rows_with_author"],
                           result["issue_author_keys_attached"])
    log_attachment_warning(logger, repo_full_name, "issues.issue_closer", result["issue_rows_with_closer"],
                           result["issue_closer_keys_attached"])
    log_attachment_warning(logger, repo_full_name, "issue_comments.author", result["comment_rows_with_author"],
                           result["comment_author_keys_attached"])
    log_attachment_warning(logger, repo_full_name, "pull_requests.author", result["pr_rows_with_author"],
                           result["pr_author_keys_attached"])
    log_attachment_warning(logger, repo_full_name, "commits.author", result["commit_rows_with_author"],
                           result["commit_author_keys_attached"])

    batch_size = get_attachment_option(config, "write_batch_size", 5000)
    repo_dir = get_batch_root(config, BATCH_FOLDER_NAME) / sanitize_repo_name(repo_full_name)
    writer = ResolvedEntityRepoChunkWriter(config=config, repo_dir=repo_dir, batch_size=batch_size)

    resolved_dfs = [issues_resolved_df, comments_resolved_df, prs_resolved_df, commits_resolved_df]
    count_labels = ["issues_rows_written", "issue_comments_rows_written", "pull_requests_rows_written",
                    "commits_rows_written"]
    add_row_funcs = [writer.add_issue_row, writer.add_comment_row, writer.add_pull_request_row, writer.add_commit_row]
    for resolved_df, count_label, add_row_func in zip(resolved_dfs, count_labels, add_row_funcs):
        for row in resolved_df.to_dict(orient="records"):
            add_row_func(row)
            result[count_label] += 1
    writer.finalize()
    result["status"] = "completed"
    return result


def merge_resolved_entity_batches(config, logger):
    batch_root = get_batch_root(config, BATCH_FOLDER_NAME)
    if not batch_root.exists():
        logger.warning("Resolved entity batch root does not exist: %s", batch_root)
        return

    issue_repo_parts = collect_repo_part_files(batch_root, "issues_resolved_part_*.parquet")
    comment_repo_parts = collect_repo_part_files(batch_root, "issue_comments_resolved_part_*.parquet")
    pr_repo_parts = collect_repo_part_files(batch_root, "pull_requests_resolved_part_*.parquet")
    commit_repo_parts = collect_repo_part_files(batch_root, "commits_resolved_part_*.parquet")

    if issue_repo_parts:
        mode_used = write_merged_or_partitioned_output(
            repo_part_map=issue_repo_parts,
            output_path=config.outputs.issues_resolved_table,
            config=config,
            table_name="issues_resolved",
            sort_columns=["repo_full_name", "issue_number"],
            dedupe_subset=["repo_full_name", "issue_id", "issue_number"],
        )
        logger.info("Wrote issues_resolved using %s mode to %s", mode_used, config.outputs.issues_resolved_table)
    else:
        logger.warning("No issues_resolved parts found to merge.")

    if comment_repo_parts:
        mode_used = write_merged_or_partitioned_output(
            repo_part_map=comment_repo_parts,
            output_path=config.outputs.issue_comments_resolved_table,
            config=config,
            table_name="issue_comments_resolved",
            sort_columns=["repo_full_name", "issue_number", "comment_id"],
            dedupe_subset=["repo_full_name", "comment_id"],
        )
        logger.info("Wrote issue_comments_resolved using %s mode to %s", mode_used,
                    config.outputs.issue_comments_resolved_table)
    else:
        logger.warning("No issue_comments_resolved parts found to merge.")

    if pr_repo_parts:
        mode_used = write_merged_or_partitioned_output(
            repo_part_map=pr_repo_parts,
            output_path=config.outputs.pull_requests_resolved_table,
            config=config,
            table_name="pull_requests_resolved",
            sort_columns=["repo_full_name", "pr_number"],
            dedupe_subset=["repo_full_name", "pr_id", "pr_number"],
        )
        logger.info("Wrote pull_requests_resolved using %s mode to %s", mode_used,
                    config.outputs.pull_requests_resolved_table, )
    else:
        logger.warning("No pull_requests_resolved parts found to merge.")

    if commit_repo_parts:
        mode_used = write_merged_or_partitioned_output(
            repo_part_map=commit_repo_parts,
            output_path=config.outputs.commits_resolved_table,
            config=config,
            table_name="commits_resolved",
            sort_columns=["repo_full_name", "commit_timestamp", "commit_sha"],
            dedupe_subset=["repo_full_name", "commit_sha"],
        )
        logger.info("Wrote commits_resolved using %s mode to %s", mode_used, config.outputs.commits_resolved_table)
    else:
        logger.warning("No commits_resolved parts found to merge.")


def write_summary_csv(summary_rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)


def write_run_manifest(config, repo_rows, summary_rows):
    manifest_path = Path(config.logging.linkage_log_dir) / "07_attach_resolved_contributor_keys_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "attach_resolved_contributor_keys.py",
        "repo_count_requested": len(repo_rows),
        "repo_count_processed": len(summary_rows),
        "completed_repo_count": sum(1 for row in summary_rows if row.get("status") == "completed"),
        "failed_repo_count": sum(1 for row in summary_rows if row.get("status") == "failed"),
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "processed_merge_mode": getattr(config.storage, "processed_merge_mode", "single_parquet"),
        "summary_rows": summary_rows,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    config = load_study_config(DEFAULT_CONFIG_PATH)
    ensure_project_directories(config)
    logger = setup_logger(config)

    if not getattr(config.identity_resolution, "enabled", True):
        logger.warning("identity_resolution.enabled is false; nothing to do.")
        return

    if get_attachment_option(config, "resume_mode", "fresh") == "fresh":
        reset_batch_root(config, BATCH_FOLDER_NAME)

    repo_rows = load_repo_list(config.outputs.repo_included_list)
    max_repos = get_attachment_option(config, "max_repos_per_run", None)
    if max_repos:
        repo_rows = repo_rows[:max_repos]

    summary_rows = []
    for repo_row in repo_rows:
        repo_full_name = repo_row["full_name"]
        skip_repo, reason = should_skip_repo(
            config,
            repo_full_name,
            checkpoint_prefix=CHECKPOINT_PREFIX,
            raw_folder_name=RAW_FOLDER_NAME,
            section_name="identity_resolution",
            raw_source="linked",
        )
        if skip_repo:
            logger.info("Skipping %s due to %s", repo_full_name, reason)
            summary_rows.append(
                {
                    "repo_full_name": repo_full_name,
                    "repo_id": repo_row.get("repo_id"),
                    "status": f"skipped_{reason}",
                }
            )
            continue

        try:
            logger.info("Starting resolved contributor attachment for %s", repo_full_name)
            result = process_repo(config, logger, repo_row)
            summary_rows.append(result)
            write_repo_checkpoint(config, CHECKPOINT_PREFIX, repo_full_name, result)
        except Exception as exc:
            logger.exception("Resolved contributor attachment failed for %s", repo_full_name)
            error_row = new_repo_result(repo_full_name, repo_row.get("repo_id"))
            error_row["status"] = "failed"
            error_row["error_message"] = str(exc)
            summary_rows.append(error_row)
            write_repo_checkpoint(config, CHECKPOINT_PREFIX, repo_full_name, error_row)

    merge_resolved_entity_batches(config, logger)
    write_summary_csv(summary_rows,
                      Path(config.logging.linkage_log_dir) / "07_attach_resolved_contributor_keys_summary.csv")
    write_run_manifest(config, repo_rows, summary_rows)
    logger.info("Resolved contributor attachment complete.")


if __name__ == "__main__":
    main()
