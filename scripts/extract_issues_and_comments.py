import csv
import gzip
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config.study_config_loader import ConfigError, ensure_project_directories, load_study_config

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"
LOG_FILENAME = "02_extract_issues_and_comments.log"
SUMMARY_FILENAME = "issues_comments_extraction_summary.csv"
CHECKPOINT_PREFIX = "02_extract_issues_and_comments"


def new_repo_result(repo_full_name, repo_id=None):
    return {
        "repo_full_name": repo_full_name,
        "repo_id": repo_id,
        "status": "started",
        "issue_pages_fetched": 0,
        "raw_issue_items_seen": 0,
        "pull_request_rows_excluded": 0,
        "issues_kept": 0,
        "issue_comment_requests": 0,
        "comment_pages_fetched": 0,
        "comments_kept": 0,
        "raw_files_written": 0,
        "error_message": "",
    }

def build_session(config):
    session = requests.Session()

    retry = Retry(
        total=config.github.rate_limit.max_retries,
        read=config.github.rate_limit.max_retries,
        connect=config.github.rate_limit.max_retries,
        status=config.github.rate_limit.max_retries,
        backoff_factor=max(config.github.rate_limit.retry_backoff_seconds / 10.0, 0.1),
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

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

def get_github_headers(config):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": config.github.requests.user_agent,
    }
    if config.github.auth.use_token:
        token = os.getenv(config.github.auth.token_env_var)
        if not token:
            raise ConfigError(
                f"GitHub token environment variable '{config.github.auth.token_env_var}' is not set."
            )
        headers["Authorization"] = f"Bearer {token}"
    return headers

def get_batch_root(config):
    return Path(config.paths.processed_root) / "_batches" / "issues_and_comments"

def reset_batch_root(config):
    batch_root = get_batch_root(config)
    if batch_root.exists():
        shutil.rmtree(batch_root)
    batch_root.mkdir(parents=True, exist_ok=True)
    return batch_root

def write_repo_batch_tables(config, repo_full_name, repository_rows, issue_rows, comment_rows):
    batch_root = get_batch_root(config)
    safe_repo_name = repo_full_name.replace("/", "__")
    repo_dir = batch_root / safe_repo_name
    repo_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(repository_rows).to_parquet(
        repo_dir / "repositories.parquet",
        index=False,
        compression=config.storage.compression.parquet_compression,
    )
    pd.DataFrame(issue_rows).to_parquet(
        repo_dir / "issues.parquet",
        index=False,
        compression=config.storage.compression.parquet_compression,
    )
    pd.DataFrame(comment_rows).to_parquet(
        repo_dir / "issue_comments.parquet",
        index=False,
        compression=config.storage.compression.parquet_compression,
    )

def merge_repo_batches(config, logger):
    batch_root = get_batch_root(config)
    repo_frames = []
    issue_frames = []
    comment_frames = []
    for repo_dir in sorted(batch_root.iterdir()):
        if not repo_dir.is_dir():
            continue
        repo_path = repo_dir / "repositories.parquet"
        issue_path = repo_dir / "issues.parquet"
        comment_path = repo_dir / "issue_comments.parquet"
        if repo_path.exists():
            repo_frames.append(pd.read_parquet(repo_path))
        if issue_path.exists():
            issue_frames.append(pd.read_parquet(issue_path))
        if comment_path.exists():
            comment_frames.append(pd.read_parquet(comment_path))

    repositories_df = pd.concat(repo_frames, ignore_index=True) if repo_frames else pd.DataFrame()
    issues_df = pd.concat(issue_frames, ignore_index=True) if issue_frames else pd.DataFrame()
    comments_df = pd.concat(comment_frames, ignore_index=True) if comment_frames else pd.DataFrame()

    if not repositories_df.empty:
        repositories_df = repositories_df.drop_duplicates(subset=["repo_id"])
    if not issues_df.empty:
        issues_df = issues_df.drop_duplicates(subset=["issue_id"])
    if not comments_df.empty:
        comments_df = comments_df.drop_duplicates(subset=["comment_id"])

    if config.issue_extraction.sort_before_write:
        if not repositories_df.empty and "full_name" in repositories_df.columns:
            repositories_df = repositories_df.sort_values(["full_name"]).reset_index(drop=True)
        if not issues_df.empty and {"repo_full_name", "issue_number"}.issubset(issues_df.columns):
            issues_df = issues_df.sort_values(["repo_full_name", "issue_number"]).reset_index(drop=True)
        if not comments_df.empty and {"repo_full_name", "issue_number", "comment_id"}.issubset(comments_df.columns):
            comments_df = comments_df.sort_values(["repo_full_name", "issue_number", "comment_id"]).reset_index(drop=True)

    write_processed_table(repositories_df, Path(config.outputs.repositories_table), config)
    write_processed_table(issues_df, Path(config.outputs.issues_table), config)
    write_processed_table(comments_df, Path(config.outputs.issue_comments_table), config)
    logger.info("Merged repo batches into final processed tables.")

def write_run_summary(config, repo_rows, summary_rows):
    output_path = Path(config.outputs.run_manifest_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
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
        "extraction_limits": {
            "max_issue_pages_per_repo_per_state": config.issue_extraction.max_issue_pages_per_repo_per_state,
            "max_comment_pages_per_issue": config.issue_extraction.max_comment_pages_per_issue,
            "max_repos_per_run": config.issue_extraction.max_repos_per_run,
        },
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

def make_request(session, url, headers, params, config, logger):
    retries = 0
    while True:
        response = session.get(
            url,
            headers=headers,
            params=params,
            timeout=config.github.requests.timeout_seconds,
        )

        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_time = response.headers.get("X-RateLimit-Reset")

        if response.status_code == 403 and remaining == "0":
            if config.github.rate_limit.respect_reset_header and reset_time:
                sleep_seconds = max(int(reset_time) - int(time.time()) + 5, 5)
            else:
                sleep_seconds = config.github.rate_limit.default_pause_seconds

            logger.warning("Rate limit reached. Sleeping for %s seconds.", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        if response.status_code in (500, 502, 503, 504, 403):
            retries += 1
            if retries > config.github.rate_limit.max_retries:
                response.raise_for_status()

            sleep_seconds = config.github.rate_limit.retry_backoff_seconds * retries
            logger.warning(
                "GitHub response %s. Retry %s/%s after %s seconds.",
                response.status_code,
                retries,
                config.github.rate_limit.max_retries,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
            continue

        response.raise_for_status()
        if remaining is not None:
            try:
                if int(remaining) <= config.github.rate_limit.min_remaining_before_pause:
                    logger.info(
                        "Approaching rate limit (remaining=%s). Pausing for %s seconds.",
                        remaining,
                        config.github.rate_limit.default_pause_seconds,
                    )
                    time.sleep(config.github.rate_limit.default_pause_seconds)
            except ValueError:
                pass

        return response

def save_json(data, output_path, use_gzip):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_gzip and output_path.suffix != ".gz":
        output_path = output_path.with_suffix(output_path.suffix + ".gz")
    if output_path.suffix == ".gz":
        with gzip.open(output_path, "wt", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
    else:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
    return 1


def parse_github_datetime(value):
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

def load_repo_list(repo_list_path):
    if not repo_list_path.exists():
        raise FileNotFoundError(f"Repo list does not exist: {repo_list_path}")
    repo_df = pd.read_csv(repo_list_path)
    if repo_df.empty:
        return []

    required_columns = {"full_name"}
    missing = required_columns - set(repo_df.columns)
    if missing:
        raise ValueError(
            "Repo list is missing required columns: " + ", ".join(sorted(missing))
        )
    return repo_df.to_dict(orient="records")

def get_repo_output_root(config, repo_full_name):
    safe_repo_name = repo_full_name.replace("/", "__")
    return Path(config.paths.raw_root) / "github_api" / "issues_and_comments" / safe_repo_name

def get_checkpoint_path(config, repo_full_name):
    safe_repo_name = repo_full_name.replace("/", "__")
    return Path(config.checkpointing.checkpoint_dir) / f"{CHECKPOINT_PREFIX}__{safe_repo_name}.json"

def should_skip_repo_from_checkpoint(config, repo_full_name):
    if not config.checkpointing.enabled:
        return False
    if not config.checkpointing.resume_from_checkpoints:
        return False
    checkpoint_path = get_checkpoint_path(config, repo_full_name)
    if not checkpoint_path.exists():
        return False

    with checkpoint_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("status") == "completed"

def write_repo_checkpoint(config, repo_full_name, payload):
    if not config.checkpointing.enabled:
        return
    checkpoint_path = get_checkpoint_path(config, repo_full_name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

def is_pull_request_issue(issue_payload):
    return issue_payload.get("pull_request") is not None

def get_issue_window_bounds(config):
    start_dt = datetime.strptime(
        config.study_windows.issue_collection.start_date, "%Y-%m-%d"
    ).replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(
        config.study_windows.issue_collection.end_date, "%Y-%m-%d"
    ).replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
    return start_dt, end_dt

def issue_created_in_window(issue_payload, start_dt, end_dt):
    created_at = parse_github_datetime(issue_payload.get("created_at"))
    if created_at is None:
        return False
    return start_dt <= created_at <= end_dt

def build_issues_params(config, page, state):
    return {
        "state": state,
        "sort": "created",
        "direction": "asc",
        "per_page": config.github.pagination.per_page,
        "page": page,
    }

def build_comments_params(config, page):
    return {
        "sort": "created",
        "direction": "asc",
        "per_page": config.github.pagination.per_page,
        "page": page,
    }

def flatten_repository(repo_payload, repo_row):
    owner = repo_payload.get("owner") or {}
    return {
        "repo_id": repo_payload.get("id"),
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
        "assignee_logins_json": json.dumps([a.get("login") for a in assignees]) if config.issue_selection.store_assignees else None,
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

def fetch_repository_metadata(session, headers, config, logger, repo_full_name):
    url = f"{config.github.api_base_url}/repos/{repo_full_name}"
    response = make_request(session, url, headers, None, config, logger)
    return response.json()

def iter_issue_pages(session, headers, config, logger, repo_full_name, state, end_dt):
    page = 1
    pages_seen = 0
    url = f"{config.github.api_base_url}/repos/{repo_full_name}/issues"
    max_pages = config.issue_extraction.max_issue_pages_per_repo_per_state
    while True:
        if pages_seen >= max_pages:
            logger.warning(
                "Reached max_issue_pages_per_repo_per_state=%s for %s state=%s",
                max_pages,
                repo_full_name,
                state
            )
            break
        params = build_issues_params(config, page, state)
        response = make_request(session, url, headers, params, config, logger)
        payload = response.json()
        if not payload:
            break
        yield page, payload
        pages_seen += 1
        last_created = parse_github_datetime(payload[-1].get("created_at"))
        if last_created and last_created > end_dt:
            break
        if len(payload) < config.github.pagination.per_page:
            break
        page += 1

def iter_issue_comment_pages(session, headers, config, logger, comments_url):
    page = 1
    pages_seen = 0
    max_pages = config.issue_extraction.max_comment_pages_per_issue
    while True:
        if pages_seen >= max_pages:
            logger.warning(
                "Reached max_comment_pages_per_issue=%s for %s",
                max_pages,
                comments_url,
            )
            break
        params = build_comments_params(config, page)
        response = make_request(session, comments_url, headers, params, config, logger)
        payload = response.json()
        if not payload:
            break
        pages_seen += 1
        yield page, payload
        if len(payload) < config.github.pagination.per_page:
            break
        page += 1

def deduplicate_rows(rows, key):
    deduped = []
    seen = set()
    for row in rows:
        value = row.get(key)
        if value in seen:
            continue
        seen.add(value)
        deduped.append(row)
    return deduped

def write_processed_table(df, output_path, config):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if config.storage.processed_format == "parquet":
        df.to_parquet(
            output_path,
            index=False,
            compression=config.storage.compression.parquet_compression,
        )
        return
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

def extract_single_repo(session, headers, config, logger, repo_row, start_dt, end_dt):
    repo_full_name = repo_row["full_name"]
    result = new_repo_result(repo_full_name, repo_row.get("repo_id"))
    output_root = get_repo_output_root(config, repo_full_name)
    repository_payload = fetch_repository_metadata(session, headers, config, logger, repo_full_name)
    result["raw_files_written"] += save_json(
        repository_payload,
        output_root / "repo.json",
        use_gzip=config.storage.compression.raw_json_gzip,
    )
    repository_row = flatten_repository(repository_payload, repo_row)
    result["repo_id"] = repository_row.get("repo_id")

    if config.issue_extraction.fail_on_missing_repo_id and not result["repo_id"]:
        raise ValueError(f"Missing repo_id for {repo_full_name}")

    issues_kept = []
    comments_kept = []
    issue_numbers_seen = set()
    for state in config.issue_selection.states:
        logger.info("Fetching issues for %s | state=%s", repo_full_name, state)
        for page, payload in iter_issue_pages(session, headers, config, logger, repo_full_name, state, end_dt):
            result["issue_pages_fetched"] += 1
            result["raw_issue_items_seen"] += len(payload)
            result["raw_files_written"] += save_json(
                payload,
                output_root / "issues" / f"issues_state_{state}_page_{page:03d}.json",
                use_gzip=config.storage.compression.raw_json_gzip,
            )
            for issue_payload in payload:
                if is_pull_request_issue(issue_payload):
                    result["pull_request_rows_excluded"] += 1
                    if not config.issue_selection.include_pull_requests_from_issues_endpoint:
                        continue
                if config.issue_selection.require_created_within_window:
                    if not issue_created_in_window(issue_payload, start_dt, end_dt):
                        continue
                issue_number = issue_payload.get("number")
                if issue_number in issue_numbers_seen:
                    continue
                issue_numbers_seen.add(issue_number)
                issues_kept.append(flatten_issue(issue_payload, repo_full_name, config))
                result["issues_kept"] += 1
                if not config.issue_selection.comments.include_comments:
                    continue
                if int(issue_payload.get("comments") or 0) <= 0:
                    continue
                comments_url = issue_payload.get("comments_url")
                if not comments_url:
                    continue
                result["issue_comment_requests"] += 1
                repo_comment_payloads = []
                for comment_page, comment_payload in iter_issue_comment_pages(
                    session,
                    headers,
                    config,
                    logger,
                    comments_url,
                ):
                    result["comment_pages_fetched"] += 1
                    repo_comment_payloads.extend(comment_payload)
                    for comment_item in comment_payload:
                        comments_kept.append(
                            flatten_issue_comment(comment_item, repo_full_name, issue_number, config)
                        )
                        result["comments_kept"] += 1
                result["raw_files_written"] += save_json(
                    repo_comment_payloads,
                    output_root / "comments" / f"issue_{issue_number}_comments.json",
                    use_gzip=config.storage.compression.raw_json_gzip,
                )
            if config.checkpointing.enabled and config.checkpointing.write_status_after_each_page:
                write_repo_checkpoint(
                    config,
                    repo_full_name,
                    {
                        "status": "in_progress",
                        "repo_full_name": repo_full_name,
                        "repo_id": result["repo_id"],
                        "issue_pages_fetched": result["issue_pages_fetched"],
                        "issues_kept": result["issues_kept"],
                        "comments_kept": result["comments_kept"],
                    },
                )
    issues_df = pd.DataFrame(issues_kept)
    comments_df = pd.DataFrame(comments_kept)

    if not issues_df.empty:
        issues_df = issues_df.drop_duplicates(subset=["issue_id"])
        issues_kept = issues_df.to_dict(orient="records")
    if not comments_df.empty:
        comments_df = comments_df.drop_duplicates(subset=["comment_id"])
        comments_kept = comments_df.to_dict(orient="records")

    result["status"] = "completed"
    if config.checkpointing.enabled and config.checkpointing.write_status_after_each_repo:
        write_repo_checkpoint(
            config,
            repo_full_name,
            {
                "status": "completed",
                "repo_full_name": repo_full_name,
                "repo_id": result["repo_id"],
                "issue_pages_fetched": result["issue_pages_fetched"],
                "issues_kept": len(issues_kept),
                "comments_kept": len(comments_kept),
                "raw_files_written": result["raw_files_written"],
            },
        )
    return [repository_row], issues_kept, comments_kept, result

def write_summary_csv(summary_rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not summary_rows:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "repo_full_name",
                    "repo_id",
                    "status",
                    "issue_pages_fetched",
                    "raw_issue_items_seen",
                    "pull_request_rows_excluded",
                    "issues_kept",
                    "issue_comment_requests",
                    "comment_pages_fetched",
                    "comments_kept",
                    "raw_files_written",
                    "error_message",
                ]
            )
        return

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

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

    if config.issue_extraction.max_repos_per_run:
        repo_rows = repo_rows[:config.issue_extraction.max_repos_per_run]
    if not repo_rows:
        logger.warning("Repo list is empty. Nothing to extract.")
        return
    if not config.storage.append_processed_batches:
        reset_batch_root(config)

    start_dt, end_dt = get_issue_window_bounds(config)
    session = build_session(config)
    headers = get_github_headers(config)
    summary_rows = []
    for repo_row in repo_rows:
        repo_full_name = repo_row["full_name"]
        if should_skip_repo_from_checkpoint(config, repo_full_name):
            logger.info("Skipping %s because completed checkpoint already exists.", repo_full_name)
            summary_rows.append(
                {
                    "repo_full_name": repo_full_name,
                    "repo_id": repo_row.get("repo_id"),
                    "status": "skipped_completed_checkpoint",
                    "issue_pages_fetched": 0,
                    "raw_issue_items_seen": 0,
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
            repository_rows, repo_issue_rows, repo_comment_rows, result = extract_single_repo(
                session,
                headers,
                config,
                logger,
                repo_row,
                start_dt,
                end_dt,
            )
            write_repo_batch_tables(
                config,
                repo_full_name,
                repository_rows,
                repo_issue_rows,
                repo_comment_rows,
            )
            summary_rows.append(result)
            logger.info(
                "Completed %s | issues=%s | comments=%s",
                repo_full_name,
                len(repo_issue_rows),
                len(repo_comment_rows),
            )
        except Exception as exc:
            logger.exception("Failed extraction for %s", repo_full_name)
            write_repo_checkpoint(
                config,
                repo_full_name,
                {
                    "status": "failed",
                    "repo_full_name": repo_full_name,
                    "error_message": str(exc),
                },
            )
            summary_rows.append(
                {
                    "repo_full_name": repo_full_name,
                    "repo_id": repo_row.get("repo_id"),
                    "status": "failed",
                    "issue_pages_fetched": 0,
                    "raw_issue_items_seen": 0,
                    "pull_request_rows_excluded": 0,
                    "issues_kept": 0,
                    "issue_comment_requests": 0,
                    "comment_pages_fetched": 0,
                    "comments_kept": 0,
                    "raw_files_written": 0,
                    "error_message": str(exc),
                }
            )
        if config.issue_extraction.request_pause_seconds_between_repos > 0:
            time.sleep(config.issue_extraction.request_pause_seconds_between_repos)

    merge_repo_batches(config, logger)
    summary_output_path = Path(config.outputs.extraction_summary_csv)
    write_summary_csv(summary_rows, summary_output_path)
    write_run_summary(config, repo_rows, summary_rows)

    logger.info("Wrote repositories table to %s", config.outputs.repositories_table)
    logger.info("Wrote issues table to %s", config.outputs.issues_table)
    logger.info("Wrote issue comments table to %s", config.outputs.issue_comments_table)
    logger.info("Wrote extraction summary to %s", summary_output_path)
    logger.info("Wrote run manifest to %s", config.outputs.run_manifest_json)
    logger.info("Issue/comment extraction pipeline complete.")


if __name__ == "__main__":
    main()
