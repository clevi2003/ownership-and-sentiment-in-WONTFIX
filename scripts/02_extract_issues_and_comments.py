import csv
import gzip
import json
import logging
import math
import os
import shutil
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.study_config_loader import ensure_project_directories, load_study_config
from utils.github_api import build_session, fetch_repository_metadata, get_github_headers, make_request
from utils.io_helpers import load_repo_list, save_json, write_processed_table
from utils.chunk_writers import IssueCommentRepoChunkWriter
from utils.labels import get_wontfix_variants, issue_has_wontfix_label, normalize_label_name
from utils.checkpoints import get_batch_root, get_repo_output_root, reset_batch_root, should_skip_repo, write_repo_checkpoint, sanitize_repo_name


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"
LOG_FILENAME = "02_extract_issues_and_comments.log"
CHECKPOINT_PREFIX = "02_extract_issues_and_comments"
BATCH_FOLDER_NAME = "issues_and_comments"
RAW_FOLDER_NAME = "issues_and_comments"


def new_repo_result(repo_full_name, repo_id=None):
    return {
        "repo_full_name": repo_full_name,
        "repo_id": repo_id,
        "status": "started",
        "search_count_requests": 0,
        "search_shards_planned": 0,
        "search_shards_executed": 0,
        "search_pages_fetched": 0,
        "search_results_seen": 0,
        "search_results_truncated": 0,
        "pull_request_rows_excluded": 0,
        "issues_kept": 0,
        "issue_comment_requests": 0,
        "comment_pages_fetched": 0,
        "comments_kept": 0,
        "raw_files_written": 0,
        "error_message": "",
    }



def setup_logger(config):
    logger = logging.getLogger("extract_issues_and_comments")
    logger.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if config.logging.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.logging.log_to_file:
        log_dir = Path(config.logging.extraction_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / LOG_FILENAME, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_issue_extraction_option(config, field_name, default_value):
    if not hasattr(config, "issue_extraction"):
        return default_value
    if not hasattr(config.issue_extraction, field_name):
        return default_value
    value = getattr(config.issue_extraction, field_name)
    if value is None:
        return default_value
    return value


def flatten_repository(repo_payload, repo_row):
    owner = repo_payload.get("owner") or {}
    return {
        "repo_id": repo_payload.get("id") or repo_row.get("repo_id"),
        "repo_node_id": repo_payload.get("node_id"),
        "repo_name": repo_payload.get("name"),
        "full_name": repo_payload.get("full_name") or repo_row.get("full_name"),
        "owner_login": owner.get("login"),
        "html_url": repo_payload.get("html_url"),
        "description": repo_payload.get("description"),
        "language": repo_payload.get("language"),
        "stargazers_count": repo_payload.get("stargazers_count"),
        "forks_count": repo_payload.get("forks_count"),
        "open_issues_count": repo_payload.get("open_issues_count"),
        "watchers_count": repo_payload.get("watchers_count"),
        "default_branch": repo_payload.get("default_branch"),
        "visibility": repo_payload.get("visibility"),
        "is_fork": repo_payload.get("fork"),
        "is_archived": repo_payload.get("archived"),
        "is_disabled": repo_payload.get("disabled"),
        "is_template": repo_payload.get("is_template"),
        "created_at": repo_payload.get("created_at"),
        "updated_at": repo_payload.get("updated_at"),
        "pushed_at": repo_payload.get("pushed_at"),
        "topics_json": json.dumps(repo_payload.get("topics", [])),
        "source_repo_list_row_json": json.dumps(repo_row),
    }


def flatten_issue(issue_payload, repo_full_name, config):
    user = issue_payload.get("user") or {}
    assignees = issue_payload.get("assignees") or []
    labels = issue_payload.get("labels") or []
    milestone = issue_payload.get("milestone") or {}
    closed_by = issue_payload.get("closed_by") or {}

    label_names = []
    label_payloads = []
    for label in labels:
        if isinstance(label, dict):
            label_names.append(label.get("name"))
            label_payloads.append(label)
        else:
            label_names.append(str(label))
            label_payloads.append({"name": str(label)})

    return {
        "issue_id": issue_payload.get("id"),
        "issue_node_id": issue_payload.get("node_id"),
        "repo_full_name": repo_full_name,
        "issue_number": issue_payload.get("number"),
        "state": issue_payload.get("state"),
        "state_reason": issue_payload.get("state_reason"),
        "title": issue_payload.get("title") if config.issue_selection.store_issue_title else None,
        "body": issue_payload.get("body") if config.issue_selection.store_issue_body else None,
        "author_login": user.get("login") if config.issue_selection.store_issue_author else None,
        "author_type": user.get("type") if config.issue_selection.store_issue_author else None,
        "assignee_logins_json": json.dumps([item.get("login") for item in assignees]) if config.issue_selection.store_assignees else None,
        "label_names_json": json.dumps(label_names) if config.issue_selection.require_labels_loaded else None,
        "label_payload_json": json.dumps(label_payloads) if config.issue_selection.require_labels_loaded else None,
        "milestone_title": milestone.get("title") if config.issue_selection.store_milestones else None,
        "comments_count": issue_payload.get("comments", 0),
        "locked": issue_payload.get("locked"),
        "active_lock_reason": issue_payload.get("active_lock_reason"),
        "author_association": issue_payload.get("author_association"),
        "created_at": issue_payload.get("created_at"),
        "updated_at": issue_payload.get("updated_at"),
        "closed_at": issue_payload.get("closed_at"),
        "closed_by_login": closed_by.get("login"),
        "html_url": issue_payload.get("html_url"),
        "url": issue_payload.get("url"),
        "repository_url": issue_payload.get("repository_url"),
        "timeline_url": issue_payload.get("timeline_url") if config.issue_selection.store_timeline_events else None,
        "performed_via_github_app_json": json.dumps(issue_payload.get("performed_via_github_app")),
        "is_wontfix_labeled": issue_has_wontfix_label(issue_payload, config),
        "source_api": "search/issues",
    }


def flatten_issue_comment(comment_payload, repo_full_name, issue_number, config):
    user = comment_payload.get("user") or {}
    return {
        "comment_id": comment_payload.get("id"),
        "comment_node_id": comment_payload.get("node_id"),
        "repo_full_name": repo_full_name,
        "issue_number": issue_number,
        "issue_url": comment_payload.get("issue_url"),
        "author_login": user.get("login") if config.issue_selection.comments.include_comment_authors else None,
        "author_type": user.get("type") if config.issue_selection.comments.include_comment_authors else None,
        "author_association": comment_payload.get("author_association") if config.issue_selection.comments.include_comment_authors else None,
        "created_at": comment_payload.get("created_at") if config.issue_selection.comments.include_comment_timestamps else None,
        "updated_at": comment_payload.get("updated_at") if config.issue_selection.comments.include_comment_timestamps else None,
        "body": comment_payload.get("body") if config.issue_selection.comments.include_comment_bodies else None,
        "html_url": comment_payload.get("html_url"),
        "url": comment_payload.get("url"),
        "reactions_json": json.dumps(comment_payload.get("reactions")) if config.issue_selection.comments.include_comment_reactions else None,
        "performed_via_github_app_json": json.dumps(comment_payload.get("performed_via_github_app")),
    }


def format_date(value):
    return value.strftime("%Y-%m-%d")


def parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


def build_search_query(repo_full_name, shard_start, shard_end):
    return f"repo:{repo_full_name} is:issue created:{format_date(shard_start)}..{format_date(shard_end)}"


def count_search_results(session, headers, config, logger, repo_full_name, shard_start, shard_end):
    url = f"{config.github.api_base_url}/search/issues"
    params = {
        "q": build_search_query(repo_full_name, shard_start, shard_end),
        "sort": "created",
        "order": "asc",
        "per_page": 1,
        "page": 1,
    }
    response = make_request(session, url, headers, params, config, logger)
    payload = response.json()
    return payload.get("total_count", 0), payload


def split_date_window(shard_start, shard_end):
    delta_days = (shard_end - shard_start).days
    midpoint = shard_start + timedelta(days=delta_days // 2)
    left_start = shard_start
    left_end = midpoint
    right_start = midpoint + timedelta(days=1)
    right_end = shard_end
    return (left_start, left_end), (right_start, right_end)


def plan_search_shards(session, headers, config, logger, repo_full_name, result, raw_root):
    start_date = parse_date(config.study_windows.issue_collection.start_date)
    end_date = parse_date(config.study_windows.issue_collection.end_date)
    max_results_per_shard = get_issue_extraction_option(config, "search_max_results_per_shard", 900)
    max_shard_splits = get_issue_extraction_option(config, "search_max_shard_splits", 1000)

    queue = [(start_date, end_date)]
    planned = []
    split_operations = 0
    count_request_index = 1

    while queue:
        shard_start, shard_end = queue.pop(0)
        total_count, count_payload = count_search_results(
            session,
            headers,
            config,
            logger,
            repo_full_name,
            shard_start,
            shard_end,
        )
        result["search_count_requests"] += 1
        result["raw_files_written"] += save_json(
            count_payload,
            raw_root / "search_counts" / f"count_{count_request_index:05d}_{format_date(shard_start)}__{format_date(shard_end)}.json",
            use_gzip=config.storage.compression.raw_json_gzip,
        )
        count_request_index += 1

        if total_count == 0:
            continue

        if total_count <= max_results_per_shard or shard_start >= shard_end:
            shard_record = {
                "start_date": format_date(shard_start),
                "end_date": format_date(shard_end),
                "total_count": total_count,
                "is_truncated_risk": 1 if total_count > 1000 else 0,
            }
            planned.append(shard_record)
            continue

        if split_operations >= max_shard_splits:
            logger.warning(
                "Reached search_max_shard_splits=%s while planning shards for %s. Keeping oversized shard %s..%s with count=%s.",
                max_shard_splits,
                repo_full_name,
                format_date(shard_start),
                format_date(shard_end),
                total_count,
            )
            planned.append(
                {
                    "start_date": format_date(shard_start),
                    "end_date": format_date(shard_end),
                    "total_count": total_count,
                    "is_truncated_risk": 1,
                }
            )
            continue

        left_window, right_window = split_date_window(shard_start, shard_end)
        queue.append(left_window)
        queue.append(right_window)
        split_operations += 1

    planned = sorted(planned, key=lambda item: (item["start_date"], item["end_date"]))
    result["search_shards_planned"] = len(planned)
    return planned


def iter_search_pages_for_shard(session, headers, config, logger, repo_full_name, shard_record):
    url = f"{config.github.api_base_url}/search/issues"
    per_page = config.github.pagination.per_page
    page = 1
    total_count = shard_record["total_count"]
    max_pages = get_issue_extraction_option(
        config,
        "max_search_pages_per_shard",
        int(math.ceil(min(total_count, 1000) / float(per_page))) if total_count else 1,
    )

    while True:
        if page > max_pages:
            break

        params = {
            "q": build_search_query(repo_full_name, parse_date(shard_record["start_date"]), parse_date(shard_record["end_date"])),
            "sort": "created",
            "order": "asc",
            "per_page": per_page,
            "page": page,
        }
        response = make_request(session, url, headers, params, config, logger)
        payload = response.json()
        items = payload.get("items") or []
        if not items:
            break

        yield page, payload

        if len(items) < per_page:
            break
        page += 1


def iter_issue_comment_pages(session, headers, config, logger, comments_url):
    page = 1
    pages_seen = 0
    max_pages = get_issue_extraction_option(config, "max_comment_pages_per_issue", 20)

    while True:
        if pages_seen >= max_pages:
            logger.warning(
                "Reached max_comment_pages_per_issue=%s for %s",
                max_pages,
                comments_url,
            )
            break

        params = {
            "sort": "created",
            "direction": "asc",
            "per_page": config.github.pagination.per_page,
            "page": page,
        }
        response = make_request(session, comments_url, headers, params, config, logger)
        payload = response.json()
        if not payload:
            break

        pages_seen += 1
        yield page, payload

        if len(payload) < config.github.pagination.per_page:
            break
        page += 1


def merge_chunked_batches(config, logger, repository_rows=None):
    batch_root = get_batch_root(config, BATCH_FOLDER_NAME)
    issue_frames = []
    comment_frames = []
    if not batch_root.exists():
        logger.warning("Batch root does not exist: %s", batch_root)
        return

    for repo_dir in sorted(batch_root.iterdir()):
        if not repo_dir.is_dir():
            continue

        for issue_path in sorted(repo_dir.glob("issues_part_*.parquet")):
            issue_frames.append(pd.read_parquet(issue_path))
        for comment_path in sorted(repo_dir.glob("issue_comments_part_*.parquet")):
            comment_frames.append(pd.read_parquet(comment_path))

    repositories_df = pd.DataFrame(repository_rows or [])
    issues_df = pd.concat(issue_frames, ignore_index=True) if issue_frames else pd.DataFrame()
    comments_df = pd.concat(comment_frames, ignore_index=True) if comment_frames else pd.DataFrame()

    if not repositories_df.empty:
        if "repo_id" in repositories_df.columns:
            repositories_df = repositories_df.drop_duplicates(subset=["repo_id"])
        elif "full_name" in repositories_df.columns:
            repositories_df = repositories_df.drop_duplicates(subset=["full_name"])

    if not issues_df.empty and "issue_id" in issues_df.columns:
        issues_df = issues_df.drop_duplicates(subset=["issue_id"])
    if not comments_df.empty and "comment_id" in comments_df.columns:
        comments_df = comments_df.drop_duplicates(subset=["comment_id"])

    if get_issue_extraction_option(config, "sort_before_write", True):
        if not repositories_df.empty and "full_name" in repositories_df.columns:
            repositories_df = repositories_df.sort_values(["full_name"]).reset_index(drop=True)
        if not issues_df.empty and {"repo_full_name", "issue_number"}.issubset(issues_df.columns):
            issues_df = issues_df.sort_values(["repo_full_name", "issue_number"]).reset_index(drop=True)
        if not comments_df.empty and {"repo_full_name", "issue_number", "comment_id"}.issubset(comments_df.columns):
            comments_df = comments_df.sort_values(["repo_full_name", "issue_number", "comment_id"]).reset_index(drop=True)

    write_processed_table(repositories_df, Path(config.outputs.repositories_table), config)
    write_processed_table(issues_df, Path(config.outputs.issues_table), config)
    write_processed_table(comments_df, Path(config.outputs.issue_comments_table), config)

    logger.info("Merged issue/comment batch shards and wrote repositories table from collected repository rows.")


def write_summary_csv(summary_rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "repo_full_name",
        "repo_id",
        "status",
        "search_count_requests",
        "search_shards_planned",
        "search_shards_executed",
        "search_pages_fetched",
        "search_results_seen",
        "search_results_truncated",
        "pull_request_rows_excluded",
        "issues_kept",
        "issue_comment_requests",
        "comment_pages_fetched",
        "comments_kept",
        "raw_files_written",
        "error_message",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if summary_rows:
            writer.writerows(summary_rows)


def write_run_manifest(config, repo_rows, summary_rows):
    output_path = Path(config.outputs.run_manifest_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "study_name": config.study.name,
        "study_version": config.study.version,
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_count_requested": len(repo_rows),
        "repo_count_summarized": len(summary_rows),
        "repo_list_path": str(config.outputs.repo_included_list),
        "repositories_table": str(config.outputs.repositories_table),
        "issues_table": str(config.outputs.issues_table),
        "issue_comments_table": str(config.outputs.issue_comments_table),
        "issue_window": {
            "start_date": config.study_windows.issue_collection.start_date,
            "end_date": config.study_windows.issue_collection.end_date,
        },
        "extraction_options_used": {
            "resume_mode": get_issue_extraction_option(config, "resume_mode", "checkpoint_only"),
            "write_batch_size": get_issue_extraction_option(config, "write_batch_size", 5000),
            "search_max_results_per_shard": get_issue_extraction_option(config, "search_max_results_per_shard", 900),
            "search_max_shard_splits": get_issue_extraction_option(config, "search_max_shard_splits", 1000),
            "max_search_pages_per_shard": get_issue_extraction_option(config, "max_search_pages_per_shard", None),
            "max_comment_pages_per_issue": get_issue_extraction_option(config, "max_comment_pages_per_issue", 20),
            "max_repos_per_run": get_issue_extraction_option(config, "max_repos_per_run", None),
        },
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def write_repo_manifest(config, raw_root, repo_full_name, repo_result, shard_plan):
    if not get_issue_extraction_option(config, "write_repo_manifest", True):
        return 0

    payload = {
        "repo_full_name": repo_full_name,
        "repo_result": repo_result,
        "search_shards": shard_plan,
    }
    return save_json(
        payload,
        raw_root / "repo_manifest.json",
        use_gzip=config.storage.compression.raw_json_gzip,
    )


def fetch_comments_for_issue(session, headers, config, logger, repo_full_name, issue_payload, writer, result, raw_root):
    if not config.issue_selection.comments.include_comments:
        return
    if int(issue_payload.get("comments") or 0) <= 0:
        return
    comments_url = issue_payload.get("comments_url")
    issue_number = issue_payload.get("number")
    issue_comment_count = int(issue_payload.get("comments") or 0)
    if not comments_url or not issue_number:
        return

    result["issue_comment_requests"] += 1
    if result["issue_comment_requests"] % 25 == 0:
        logger.info(
            "Fetching comments | repo=%s | issue=%s | expected_comments=%s | issue_comment_requests=%s",
            repo_full_name,
            issue_number,
            issue_comment_count,
            result["issue_comment_requests"],
        )

    for page, payload in iter_issue_comment_pages(session, headers, config, logger, comments_url):
        result["comment_pages_fetched"] += 1
        result["raw_files_written"] += save_json(
            payload,
            raw_root / "comments" / f"issue_{issue_number:06d}_page_{page:03d}.json",
            use_gzip=config.storage.compression.raw_json_gzip,
        )

        for comment_payload in payload:
            writer.add_comment_row(flatten_issue_comment(comment_payload, repo_full_name, issue_number, config))
            result["comments_kept"] += 1


def extract_single_repo(session, headers, config, logger, repo_row):
    repo_full_name = repo_row["full_name"]
    result = new_repo_result(repo_full_name, repo_row.get("repo_id"))
    raw_root = get_repo_output_root(config, RAW_FOLDER_NAME, repo_full_name)
    batch_size = get_issue_extraction_option(config, "write_batch_size", 5000)
    safe_repo = sanitize_repo_name(repo_full_name)
    repo_dir = get_batch_root(config, BATCH_FOLDER_NAME) / safe_repo
    writer = IssueCommentRepoChunkWriter(config=config, repo_dir=repo_dir, batch_size=batch_size,)

    repository_payload = fetch_repository_metadata(session, headers, config, logger, repo_full_name)
    result["raw_files_written"] += save_json(
        repository_payload,
        raw_root / "repo.json",
        use_gzip=config.storage.compression.raw_json_gzip,
    )
    repository_row = flatten_repository(repository_payload, repo_row)
    result["repo_id"] = repository_row.get("repo_id")

    if get_issue_extraction_option(config, "fail_on_missing_repo_id", True) and not result["repo_id"]:
        raise ValueError(f"Missing repo_id for {repo_full_name}")

    writer.write_repository_row(repository_row)

    shard_plan = plan_search_shards(session, headers, config, logger, repo_full_name, result, raw_root)
    result["raw_files_written"] += write_repo_manifest(config, raw_root, repo_full_name, result, shard_plan)

    seen_issue_ids = set()

    for shard_index, shard_record in enumerate(shard_plan, start=1):
        result["search_shards_executed"] += 1
        if shard_record.get("is_truncated_risk"):
            result["search_results_truncated"] += 1

        logger.info(
            "Searching %s | shard=%s/%s | %s..%s | expected_count=%s",
            repo_full_name,
            shard_index,
            len(shard_plan),
            shard_record["start_date"],
            shard_record["end_date"],
            shard_record["total_count"],
        )

        for page, payload in iter_search_pages_for_shard(session, headers, config, logger, repo_full_name, shard_record):
            result["search_pages_fetched"] += 1
            result["raw_files_written"] += save_json(
                payload,
                raw_root / "search_pages" / f"shard_{shard_index:04d}_page_{page:03d}.json",
                use_gzip=config.storage.compression.raw_json_gzip,
            )

            for issue_payload in payload.get("items") or []:
                result["search_results_seen"] += 1

                if issue_payload.get("pull_request") is not None:
                    result["pull_request_rows_excluded"] += 1
                    if not config.issue_selection.include_pull_requests_from_issues_endpoint:
                        continue

                issue_id = issue_payload.get("id")
                if issue_id in seen_issue_ids:
                    continue
                seen_issue_ids.add(issue_id)

                writer.add_issue_row(flatten_issue(issue_payload, repo_full_name, config))
                result["issues_kept"] += 1

                fetch_comments_for_issue(
                    session,
                    headers,
                    config,
                    logger,
                    repo_full_name,
                    issue_payload,
                    writer,
                    result,
                    raw_root,
                )

            if config.checkpointing.enabled and config.checkpointing.write_status_after_each_page:
                write_repo_checkpoint(
                    config,
                    CHECKPOINT_PREFIX,
                    repo_full_name,
                    {
                        "status": "in_progress",
                        "repo_full_name": repo_full_name,
                        "repo_id": result["repo_id"],
                        "search_pages_fetched": result["search_pages_fetched"],
                        "issues_kept": result["issues_kept"],
                        "comments_kept": result["comments_kept"],
                    },
                )

    writer.finalize()
    result["status"] = "completed"

    if config.checkpointing.enabled and config.checkpointing.write_status_after_each_repo:
        write_repo_checkpoint(
            config,
            CHECKPOINT_PREFIX,
            repo_full_name,
            {
                "status": "in_progress",
                "repo_full_name": repo_full_name,
                "repo_id": result["repo_id"],
                "search_pages_fetched": result["search_pages_fetched"],
                "issues_kept": result["issues_kept"],
                "comments_kept": result["comments_kept"],
            },
        )

    return result, repository_row


def main():
    config = load_study_config(DEFAULT_CONFIG_PATH)
    ensure_project_directories(config)

    try:
        from config.study_config_loader import write_resolved_config_snapshot
        write_resolved_config_snapshot(config)
    except Exception:
        pass

    logger = setup_logger(config)
    logger.info("Loaded config from %s", DEFAULT_CONFIG_PATH)

    repo_list_path = Path(config.outputs.repo_included_list)
    repo_rows = load_repo_list(repo_list_path)

    max_repos_per_run = get_issue_extraction_option(config, "max_repos_per_run", None)
    if max_repos_per_run:
        repo_rows = repo_rows[:max_repos_per_run]

    if not repo_rows:
        logger.warning("Repo list is empty. Nothing to extract.")
        return

    resume_mode = get_issue_extraction_option(config, "resume_mode", "checkpoint_only")
    if resume_mode == "fresh" and not config.storage.append_processed_batches:
        reset_batch_root(config, BATCH_FOLDER_NAME)
    elif not get_batch_root(config, BATCH_FOLDER_NAME).exists():
        get_batch_root(config, BATCH_FOLDER_NAME).mkdir(parents=True, exist_ok=True)

    session = build_session(config)
    headers = get_github_headers(config)
    summary_rows = []
    repository_rows = []

    for repo_row in repo_rows:
        repo_full_name = repo_row["full_name"]
        should_skip, reason = should_skip_repo(config,
                                               repo_full_name,
                                               checkpoint_prefix=CHECKPOINT_PREFIX,
                                               raw_folder_name=RAW_FOLDER_NAME,
                                               section_name="issue_extraction")
        if should_skip:
            logger.info("Skipping %s because %s already exists.", repo_full_name, reason)
            summary_rows.append(
                {
                    "repo_full_name": repo_full_name,
                    "repo_id": repo_row.get("repo_id"),
                    "status": f"skipped_{reason}",
                    "search_count_requests": 0,
                    "search_shards_planned": 0,
                    "search_shards_executed": 0,
                    "search_pages_fetched": 0,
                    "search_results_seen": 0,
                    "search_results_truncated": 0,
                    "pull_request_rows_excluded": 0,
                    "issues_kept": 0,
                    "issue_comment_requests": 0,
                    "comment_pages_fetched": 0,
                    "comments_kept": 0,
                    "raw_files_written": 0,
                    "error_message": "",
                }
            )
            continue

        logger.info("Starting extraction for %s", repo_full_name)
        try:
            result, repository_row = extract_single_repo(session, headers, config, logger, repo_row)
            summary_rows.append(result)
            if repository_row:
                repository_rows.append(repository_row)
            logger.info(
                "Completed %s | issues=%s | comments=%s | search_pages=%s",
                repo_full_name,
                result["issues_kept"],
                result["comments_kept"],
                result["search_pages_fetched"],
            )
        except Exception as exc:
            logger.exception("Failed extraction for %s", repo_full_name)
            payload = {
                    "status": "failed",
                    "repo_full_name": repo_full_name,
                    "error_message": str(exc),
                }
            write_repo_checkpoint(config, CHECKPOINT_PREFIX, repo_full_name, payload)
            summary_rows.append(
                {
                    "repo_full_name": repo_full_name,
                    "repo_id": repo_row.get("repo_id"),
                    "status": "failed",
                    "search_count_requests": 0,
                    "search_shards_planned": 0,
                    "search_shards_executed": 0,
                    "search_pages_fetched": 0,
                    "search_results_seen": 0,
                    "search_results_truncated": 0,
                    "pull_request_rows_excluded": 0,
                    "issues_kept": 0,
                    "issue_comment_requests": 0,
                    "comment_pages_fetched": 0,
                    "comments_kept": 0,
                    "raw_files_written": 0,
                    "error_message": str(exc),
                }
            )

        pause_seconds = get_issue_extraction_option(config, "request_pause_seconds_between_repos", 0)
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    if summary_rows:
        merge_chunked_batches(config, logger, repository_rows=repository_rows)
    else:
        logger.warning("No repo rows were processed in this run; skipping batch merge.")

    summary_output_path = Path(config.outputs.extraction_summary_csv)
    write_summary_csv(summary_rows, summary_output_path)
    write_run_manifest(config, repo_rows, summary_rows)

    logger.info("Wrote repositories table to %s", config.outputs.repositories_table)
    logger.info("Wrote issues table to %s", config.outputs.issues_table)
    logger.info("Wrote issue comments table to %s", config.outputs.issue_comments_table)
    logger.info("Wrote extraction summary to %s", summary_output_path)
    logger.info("Wrote run manifest to %s", config.outputs.run_manifest_json)
    logger.info("Issue/comment extraction pipeline complete.")


if __name__ == "__main__":
    main()
