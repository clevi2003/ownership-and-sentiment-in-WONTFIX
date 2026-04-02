import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.study_config_loader import ensure_project_directories, load_study_config
from utils.github_api import build_session, fetch_repository_metadata, get_github_headers, make_request
from utils.io_helpers import load_repo_list, save_json, write_processed_table
from utils.chunk_writers import PullRequestRepoChunkWriter
from utils.checkpoints import get_batch_root, get_repo_output_root, reset_batch_root, should_skip_repo, write_repo_checkpoint, sanitize_repo_name
from utils.regex_expressions import CLOSING_CLAUSE_PATTERN, ISSUE_REF_PATTERN, COMMIT_ISSUE_REF_PATTERN, ISSUE_NUMBER_FROM_REF_PATTERNS

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"
LOG_FILENAME = "03_extract_prs.log"
CHECKPOINT_PREFIX = "03_extract_prs"
BATCH_FOLDER_NAME = "pull_requests"
RAW_FOLDER_NAME = "pull_requests"



def setup_logger(config):
    logger = logging.getLogger("extract_prs")
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


def get_pr_selection_option(config, field_name, default_value):
    if not hasattr(config, "pull_request_selection"):
        return default_value
    if not hasattr(config.pull_request_selection, field_name):
        return default_value
    value = getattr(config.pull_request_selection, field_name)
    if value is None:
        return default_value
    return value


def new_repo_result(repo_full_name, repo_id=None):
    return {
        "repo_full_name": repo_full_name,
        "repo_id": repo_id,
        "status": "started",
        "timeline_requests": 0,
        "timeline_pages_fetched": 0,
        "timeline_events_seen": 0,
        "explicit_pr_candidates_found": 0,
        "unique_pr_numbers_discovered": 0,
        "pr_detail_requests": 0,
        "pr_rows_written": 0,
        "issue_pr_link_rows_written": 0,
        "pr_commit_requests": 0,
        "pr_commit_rows_written": 0,
        "raw_files_written": 0,
        "error_message": "",
    }


def parse_datetime(value):
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def flatten_pull_request(pr_payload, repo_full_name, repo_id, config):
    user = pr_payload.get("user") or {}
    base = pr_payload.get("base") or {}
    head = pr_payload.get("head") or {}

    return {
        "pr_id": pr_payload.get("id"),
        "pr_node_id": pr_payload.get("node_id"),
        "repo_id": repo_id,
        "repo_full_name": repo_full_name,
        "pr_number": pr_payload.get("number"),
        "author_login": user.get("login") if get_pr_selection_option(config, "include_pr_author", True) else None,
        "author_type": user.get("type") if get_pr_selection_option(config, "include_pr_author", True) else None,
        "state": pr_payload.get("state") if get_pr_selection_option(config, "include_pr_state", True) else None,
        "title": pr_payload.get("title"),
        "body": pr_payload.get("body") if get_pr_selection_option(config, "include_pr_body", True) else None,
        "created_at": pr_payload.get("created_at") if get_pr_selection_option(config, "include_pr_created_closed_merged_dates", True) else None,
        "updated_at": pr_payload.get("updated_at"),
        "closed_at": pr_payload.get("closed_at") if get_pr_selection_option(config, "include_pr_created_closed_merged_dates", True) else None,
        "merged_at": pr_payload.get("merged_at") if get_pr_selection_option(config, "include_pr_created_closed_merged_dates", True) else None,
        "merge_commit_sha": pr_payload.get("merge_commit_sha"),
        "draft": pr_payload.get("draft"),
        "html_url": pr_payload.get("html_url"),
        "url": pr_payload.get("url"),
        "issue_url": pr_payload.get("issue_url"),
        "commits_url": pr_payload.get("commits_url"),
        "comments_url": pr_payload.get("comments_url"),
        "review_comments_url": pr_payload.get("review_comments_url"),
        "base_ref": base.get("ref"),
        "base_sha": base.get("sha"),
        "head_ref": head.get("ref"),
        "head_sha": head.get("sha"),
        "additions": pr_payload.get("additions"),
        "deletions": pr_payload.get("deletions"),
        "changed_files": pr_payload.get("changed_files"),
        "commits_count": pr_payload.get("commits"),
        "source_api": "pulls/detail",
    }


def make_issue_pr_link_row(
    *,
    repo_id,
    repo_full_name,
    issue_id,
    issue_number,
    pr_id,
    pr_number,
    link_type,
    config,
    source_event_type=None,
    source_text=None,
    source_url=None,
):
    return {
        "repo_id": repo_id,
        "repo_full_name": repo_full_name,
        "issue_id": issue_id,
        "issue_number": issue_number,
        "pr_id": pr_id,
        "pr_number": pr_number,
        "link_type": link_type,
        "link_confidence": config.linkage.issue_pr.confidence_levels.get(link_type),
        "source_event_type": source_event_type,
        "source_text": source_text,
        "source_url": source_url,
    }


def make_pr_commit_link_row(repo_id, repo_full_name, pr_id, pr_number, commit_sha):
    return {
        "repo_id": repo_id,
        "repo_full_name": repo_full_name,
        "pr_id": pr_id,
        "pr_number": pr_number,
        "commit_sha": commit_sha,
    }


def sanitize_issue_key(issue_number):
    try:
        return int(issue_number)
    except Exception:
        return None


def fetch_issue_timeline_pages(session, headers, config, logger, repo_full_name, issue_number):
    url = f"{config.github.api_base_url}/repos/{repo_full_name}/issues/{issue_number}/timeline"
    page = 1

    while True:
        params = {
            "per_page": config.github.pagination.per_page,
            "page": page,
        }
        response = make_request(session, url, headers, params, config, logger)
        payload = response.json()

        if not payload:
            break

        yield page, payload

        if len(payload) < config.github.pagination.per_page:
            break
        page += 1


def extract_pr_numbers_from_timeline_event(event):
    numbers = set()

    def maybe_add_issue_object(obj):
        if not isinstance(obj, dict):
            return
        number = obj.get("number")
        pull_request = obj.get("pull_request")
        if number and pull_request is not None:
            numbers.add(int(number))

    maybe_add_issue_object(event.get("issue"))
    maybe_add_issue_object(event.get("subject"))

    source = event.get("source") or {}
    maybe_add_issue_object(source.get("issue"))

    return sorted(numbers)


def fetch_pull_request_detail(session, headers, config, logger, repo_full_name, pr_number):
    url = f"{config.github.api_base_url}/repos/{repo_full_name}/pulls/{pr_number}"
    response = make_request(session, url, headers, None, config, logger)
    return response.json()


def fetch_pull_request_commits(session, headers, config, logger, repo_full_name, pr_number):
    url = f"{config.github.api_base_url}/repos/{repo_full_name}/pulls/{pr_number}/commits"
    page = 1

    while True:
        params = {
            "per_page": config.github.pagination.per_page,
            "page": page,
        }
        response = make_request(session, url, headers, params, config, logger)
        payload = response.json()

        if not payload:
            break

        yield page, payload

        if len(payload) < config.github.pagination.per_page:
            break
        page += 1


def extract_pr_body_issue_numbers(text):
    if not text:
        return set()
    issue_numbers = set()
    for clause_match in CLOSING_CLAUSE_PATTERN.finditer(text):
        tail = clause_match.group("tail") or ""
        for ref_match in ISSUE_REF_PATTERN.finditer(tail):
            for group_name in ("plain", "gh", "repo", "url"):
                value = ref_match.group(group_name)
                if value:
                    issue_numbers.add(int(value))
                    break
    return issue_numbers


def extract_commit_message_issue_numbers(text):
    if not text:
        return set()
    issue_numbers = set()
    for match in COMMIT_ISSUE_REF_PATTERN.finditer(text):
        for group_name in ("plain", "gh", "repo", "url"):
            value = match.group(group_name)
            if value:
                issue_numbers.add(int(value))
                break
    return issue_numbers


def merge_part_files(part_paths, sort_columns=None):
    frames = []
    for path in sorted(part_paths):
        df = pd.read_parquet(path)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)

    if sort_columns:
        existing = [col for col in sort_columns if col in merged.columns]
        if existing:
            merged = merged.sort_values(existing, kind="stable").reset_index(drop=True)

    return merged


def write_summary_csv(summary_rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)


def write_run_manifest(config, repo_rows, summary_rows):
    manifest_path = Path(config.logging.extraction_log_dir) / "03_extract_prs_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "script": "extract_prs.py",
        "repo_count_requested": len(repo_rows),
        "repo_count_processed": len(summary_rows),
        "completed_repo_count": sum(1 for row in summary_rows if row.get("status") == "completed"),
        "failed_repo_count": sum(1 for row in summary_rows if row.get("status") == "failed"),
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary_rows": summary_rows,
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def merge_pr_batches(config, logger):
    batch_root = get_batch_root(config, BATCH_FOLDER_NAME)
    if not batch_root.exists():
        logger.warning("PR batch root does not exist: %s", batch_root)
        return

    pr_parts = list(batch_root.glob("*/pull_requests_part_*.parquet"))
    issue_pr_parts = list(batch_root.glob("*/issue_pr_links_part_*.parquet"))
    pr_commit_parts = list(batch_root.glob("*/pr_commit_links_part_*.parquet"))

    pr_df = merge_part_files(
        pr_parts,
        sort_columns=["repo_full_name", "pr_number"],
    )
    issue_pr_df = merge_part_files(
        issue_pr_parts,
        sort_columns=["repo_full_name", "issue_number", "pr_number", "link_type"],
    )
    pr_commit_df = merge_part_files(
        pr_commit_parts,
        sort_columns=["repo_full_name", "pr_number", "commit_sha"],
    )

    if not pr_df.empty:
        pr_df = pr_df.drop_duplicates(subset=["repo_full_name", "pr_id", "pr_number"])
        write_processed_table(pr_df, Path(config.outputs.pull_requests_table), config)
        logger.info("Wrote merged pull requests table to %s", config.outputs.pull_requests_table)

    if not issue_pr_df.empty:
        issue_pr_df = issue_pr_df.drop_duplicates(
            subset=["repo_full_name", "issue_id", "pr_id", "link_type"]
        )
        write_processed_table(issue_pr_df, Path(config.outputs.issue_pr_links_table), config)
        logger.info("Wrote merged issue-PR links table to %s", config.outputs.issue_pr_links_table)

    pr_commit_output = getattr(config.outputs, "pr_commit_links_table", None)
    if pr_commit_output and not pr_commit_df.empty:
        pr_commit_df = pr_commit_df.drop_duplicates(
            subset=["repo_full_name", "pr_id", "commit_sha"]
        )
        write_processed_table(pr_commit_df, Path(pr_commit_output), config)
        logger.info("Wrote merged PR-commit links table to %s", pr_commit_output)


def normalize_issue_reference(ref_text):
    ref_text = ref_text.strip()
    for pattern in ISSUE_NUMBER_FROM_REF_PATTERNS:
        match = pattern.match(ref_text)
        if match:
            return int(match.group("num"))
    return None


def extract_commit_ids_from_timeline_event(event):
    commit_ids = set()

    commit_id = event.get("commit_id")
    if commit_id:
        commit_ids.add(commit_id)

    source = event.get("source") or {}
    source_commit = source.get("commit") or {}
    source_commit_id = source_commit.get("sha") or source_commit.get("oid")
    if source_commit_id:
        commit_ids.add(source_commit_id)

    return sorted(commit_ids)


def fetch_pull_requests_for_commit(session, headers, config, logger, repo_full_name, commit_sha):
    url = f"{config.github.api_base_url}/repos/{repo_full_name}/commits/{commit_sha}/pulls"

    try:
        response = make_request(session, url, headers, None, config, logger)
        return response.json()
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None

        if status_code == 422:
            logger.debug(
                "Skipping commit->PR lookup due to 422 | repo=%s | commit=%s",
                repo_full_name,
                commit_sha,
            )
            return []

        raise


def process_repo(session, headers, config, logger, repo_row, issues_df):
    repo_full_name = repo_row["full_name"]

    repo_payload = fetch_repository_metadata(session, headers, config, logger, repo_full_name)
    repo_id = repo_payload.get("id") or repo_row.get("repo_id")

    result = new_repo_result(repo_full_name, repo_id=repo_id)
    raw_root = get_repo_output_root(config, RAW_FOLDER_NAME, repo_full_name)
    batch_size = get_issue_extraction_option(config, "write_batch_size", 5000)
    safe_repo = sanitize_repo_name(repo_full_name)
    repo_dir = get_batch_root(config, BATCH_FOLDER_NAME) / safe_repo
    writer = PullRequestRepoChunkWriter(config=config, repo_dir=repo_dir, batch_size=batch_size,)

    repo_issues = issues_df[issues_df["repo_full_name"] == repo_full_name].copy()
    if repo_issues.empty:
        logger.info("No issues found in issues.parquet for %s. Skipping.", repo_full_name)
        result["status"] = "completed"
        return result

    repo_issues["issue_number"] = repo_issues["issue_number"].astype(int)

    issue_number_to_issue_id = {
        int(row["issue_number"]): row["issue_id"]
        for _, row in repo_issues[["issue_number", "issue_id"]].drop_duplicates().iterrows()
    }
    target_issue_numbers = set(issue_number_to_issue_id.keys())

    explicit_links_by_pr_number = {}
    fetched_pr_numbers = set()

    timeline_headers = get_github_headers(
        config,
        accept="application/vnd.github+json",
    )

    issue_numbers_sorted = sorted(target_issue_numbers)
    total_issue_count = len(issue_numbers_sorted)

    for issue_index, issue_number in enumerate(issue_numbers_sorted, start=1):
        if issue_index == 1 or issue_index % 50 == 0 or issue_index == total_issue_count:
            logger.info(
                "Timeline progress | repo=%s | issues_processed=%s/%s | prs_found=%s",
                repo_full_name,
                issue_index,
                total_issue_count,
                len(explicit_links_by_pr_number),
            )

        result["timeline_requests"] += 1

        commit_headers = get_github_headers(config, accept="application/vnd.github+json")

        for page, payload in fetch_issue_timeline_pages(
                session,
                timeline_headers,
                config,
                logger,
                repo_full_name,
                issue_number,
        ):
            result["timeline_pages_fetched"] += 1
            result["raw_files_written"] += save_json(
                payload,
                raw_root / "issue_timelines" / f"issue_{issue_number:06d}_page_{page:03d}.json",
                use_gzip=config.storage.compression.raw_json_gzip,
            )

            for event in payload:
                result["timeline_events_seen"] += 1

                # direct PR number discovery from timeline event
                pr_numbers = extract_pr_numbers_from_timeline_event(event)
                for pr_number in pr_numbers:
                    explicit_links_by_pr_number.setdefault(pr_number, []).append(
                        {
                            "issue_number": issue_number,
                            "link_type": "explicit_github_reference",
                            "source_event_type": event.get("event"),
                            "source_text": event.get("commit_id") or event.get("url"),
                            "source_url": event.get("url"),
                        }
                    )
                    result["explicit_pr_candidates_found"] += 1

                # commit links to PR (some timeline events like closed are triggered by commits, try looking up associated PRs for those commits)
                if event.get("event") in {"closed", "referenced"}:
                    for commit_sha in extract_commit_ids_from_timeline_event(event):
                        commit_pr_payload = fetch_pull_requests_for_commit(
                            session,
                            commit_headers,
                            config,
                            logger,
                            repo_full_name,
                            commit_sha,
                        )

                        result["raw_files_written"] += save_json(
                            commit_pr_payload,
                            raw_root / "commit_to_prs" / f"{commit_sha}.json",
                            use_gzip=config.storage.compression.raw_json_gzip,
                        )

                        for pr_item in commit_pr_payload:
                            pr_number = pr_item.get("number")
                            if not pr_number:
                                continue

                            explicit_links_by_pr_number.setdefault(int(pr_number), []).append(
                                {
                                    "issue_number": issue_number,
                                    "link_type": "explicit_github_reference",
                                    "source_event_type": f"{event.get('event')}+commit_lookup",
                                    "source_text": commit_sha,
                                    "source_url": event.get("url"),
                                }
                            )
                            result["explicit_pr_candidates_found"] += 1

            if config.checkpointing.enabled and config.checkpointing.write_status_after_each_page:
                write_repo_checkpoint(
                    config,
                    CHECKPOINT_PREFIX,
                    repo_full_name,
                    {
                        "status": "in_progress",
                        "repo_full_name": repo_full_name,
                        "repo_id": repo_id,
                        "timeline_pages_fetched": result["timeline_pages_fetched"],
                        "unique_pr_numbers_discovered": len(explicit_links_by_pr_number),
                        "pr_rows_written": result["pr_rows_written"],
                        "issue_pr_link_rows_written": result["issue_pr_link_rows_written"],
                        "pr_commit_rows_written": result["pr_commit_rows_written"],
                    },
                )

    result["unique_pr_numbers_discovered"] = len(explicit_links_by_pr_number)
    logger.info(
        "Timeline discovery complete | repo=%s | issues_scanned=%s | unique_prs_found=%s",
        repo_full_name,
        total_issue_count,
        result["unique_pr_numbers_discovered"],
    )

    pr_rows = []
    issue_pr_rows = []
    pr_commit_rows = []

    for pr_number in sorted(explicit_links_by_pr_number.keys()):
        if pr_number in fetched_pr_numbers:
            continue
        fetched_pr_numbers.add(pr_number)

        logger.info("Fetching PR detail | repo=%s | pr=%s", repo_full_name, pr_number)
        result["pr_detail_requests"] += 1

        pr_payload = fetch_pull_request_detail(session, headers, config, logger, repo_full_name, pr_number)
        result["raw_files_written"] += save_json(
            pr_payload,
            raw_root / "pr_details" / f"pr_{pr_number:06d}.json",
            use_gzip=config.storage.compression.raw_json_gzip,
        )

        pr_row = flatten_pull_request(pr_payload, repo_full_name, repo_id, config)
        pr_rows.append(pr_row)

        pr_id = pr_row["pr_id"]

        for link_meta in explicit_links_by_pr_number.get(pr_number, []):
            issue_number = link_meta["issue_number"]
            issue_id = issue_number_to_issue_id.get(issue_number)
            if issue_id is None:
                continue

            issue_pr_rows.append(
                make_issue_pr_link_row(
                    repo_id=repo_id,
                    repo_full_name=repo_full_name,
                    issue_id=issue_id,
                    issue_number=issue_number,
                    pr_id=pr_id,
                    pr_number=pr_number,
                    link_type=link_meta["link_type"],
                    config=config,
                    source_event_type=link_meta.get("source_event_type"),
                    source_text=link_meta.get("source_text"),
                    source_url=link_meta.get("source_url"),
                )
            )

        if get_pr_selection_option(config, "include_pr_body", True):
            for linked_issue_number in sorted(extract_pr_body_issue_numbers(pr_payload.get("body"))):
                if linked_issue_number not in target_issue_numbers:
                    continue
                issue_pr_rows.append(
                    make_issue_pr_link_row(
                        repo_id=repo_id,
                        repo_full_name=repo_full_name,
                        issue_id=issue_number_to_issue_id[linked_issue_number],
                        issue_number=linked_issue_number,
                        pr_id=pr_id,
                        pr_number=pr_number,
                        link_type="pr_body_closes_reference",
                        config=config,
                        source_text=pr_payload.get("body"),
                        source_url=pr_payload.get("html_url"),
                    )
                )

        if config.linkage.pr_commit.enabled and get_pr_selection_option(config, "include_pr_commits", True):
            for page, payload in fetch_pull_request_commits(
                session,
                headers,
                config,
                logger,
                repo_full_name,
                pr_number,
            ):
                result["pr_commit_requests"] += 1
                result["raw_files_written"] += save_json(
                    payload,
                    raw_root / "pr_commits" / f"pr_{pr_number:06d}_page_{page:03d}.json",
                    use_gzip=config.storage.compression.raw_json_gzip,
                )

                for commit_payload in payload:
                    commit_sha = commit_payload.get("sha")
                    if not commit_sha:
                        continue

                    pr_commit_rows.append(
                        make_pr_commit_link_row(repo_id, repo_full_name, pr_id, pr_number, commit_sha)
                    )

                    commit_message = (
                        (commit_payload.get("commit") or {}).get("message")
                        or ""
                    )
                    for linked_issue_number in sorted(extract_commit_message_issue_numbers(commit_message)):
                        if linked_issue_number not in target_issue_numbers:
                            continue
                        issue_pr_rows.append(
                            make_issue_pr_link_row(
                                repo_id=repo_id,
                                repo_full_name=repo_full_name,
                                issue_id=issue_number_to_issue_id[linked_issue_number],
                                issue_number=linked_issue_number,
                                pr_id=pr_id,
                                pr_number=pr_number,
                                link_type="commit_message_issue_reference",
                                config=config,
                                source_text=commit_message,
                                source_url=commit_payload.get("html_url"),
                            )
                        )

    pr_rows = writer.dedupe_rows(pr_rows, ["repo_full_name", "pr_id", "pr_number"])
    issue_pr_rows = writer.dedupe_rows(issue_pr_rows, ["repo_full_name", "issue_id", "pr_id", "link_type"])
    pr_commit_rows = writer.dedupe_rows(pr_commit_rows, ["repo_full_name", "pr_id", "commit_sha"])

    for row in pr_rows:
        writer.add_pr_row(row)
    for row in issue_pr_rows:
        writer.add_issue_pr_row(row)
    for row in pr_commit_rows:
        writer.add_pr_commit_row(row)

    logger.info(
        "Write prep | repo=%s | pr_rows=%s | issue_pr_rows=%s | pr_commit_rows=%s",
        repo_full_name,
        len(pr_rows),
        len(issue_pr_rows),
        len(pr_commit_rows),
    )
    writer.finalize()

    result["pr_rows_written"] = len(pr_rows)
    result["issue_pr_link_rows_written"] = len(issue_pr_rows)
    result["pr_commit_rows_written"] = len(pr_commit_rows)
    result["status"] = "completed"

    if config.checkpointing.enabled and config.checkpointing.write_status_after_each_repo:
        write_repo_checkpoint(
            config,
            CHECKPOINT_PREFIX,
            repo_full_name,
            {
                "status": "completed",
                "repo_full_name": repo_full_name,
                "repo_id": repo_id,
                "timeline_requests": result["timeline_requests"],
                "timeline_pages_fetched": result["timeline_pages_fetched"],
                "unique_pr_numbers_discovered": result["unique_pr_numbers_discovered"],
                "pr_detail_requests": result["pr_detail_requests"],
                "pr_rows_written": result["pr_rows_written"],
                "issue_pr_link_rows_written": result["issue_pr_link_rows_written"],
                "pr_commit_rows_written": result["pr_commit_rows_written"],
                "raw_files_written": result["raw_files_written"],
            },
        )

    return result


def main():
    config = load_study_config(DEFAULT_CONFIG_PATH)
    ensure_project_directories(config)
    logger = setup_logger(config)

    logger.info("Loaded config from %s", DEFAULT_CONFIG_PATH)

    repo_rows = load_repo_list(Path(config.outputs.repo_included_list))
    max_repos_per_run = get_issue_extraction_option(config, "max_repos_per_run", None)
    if max_repos_per_run:
        repo_rows = repo_rows[:max_repos_per_run]

    if not repo_rows:
        logger.warning("Repo list is empty. Nothing to extract.")
        return

    issues_path = Path(config.outputs.issues_table)
    if not issues_path.exists():
        raise FileNotFoundError(f"issues.parquet does not exist: {issues_path}")

    issues_df = pd.read_parquet(issues_path)
    if issues_df.empty:
        logger.warning("issues.parquet is empty. Nothing to extract.")
        return

    resume_mode = get_issue_extraction_option(config, "resume_mode", "checkpoint_only")
    batch_root = get_batch_root(config, BATCH_FOLDER_NAME)
    if resume_mode == "fresh" and not config.storage.append_processed_batches:
        reset_batch_root(config, BATCH_FOLDER_NAME)
    elif not batch_root.exists():
        batch_root.mkdir(parents=True, exist_ok=True)

    session = build_session(config)
    headers = get_github_headers(config)
    summary_rows = []

    for repo_row in repo_rows:
        repo_full_name = repo_row["full_name"]

        skip_repo, skip_reason = should_skip_repo(
            config,
            repo_full_name,
            checkpoint_prefix=CHECKPOINT_PREFIX,
            raw_folder_name=RAW_FOLDER_NAME,
            section_name="issue_extraction",
        )
        if skip_repo:
            logger.info("Skipping repo %s (%s)", repo_full_name, skip_reason)
            summary_rows.append(
                {
                    "repo_full_name": repo_full_name,
                    "status": "skipped",
                    "skip_reason": skip_reason,
                }
            )
            continue

        logger.info("Starting PR extraction for %s", repo_full_name)

        try:
            result = process_repo(session, headers, config, logger, repo_row, issues_df)
        except Exception as exc:
            logger.exception("Failed PR extraction for %s", repo_full_name)
            result = new_repo_result(repo_full_name, repo_id=repo_row.get("repo_id"))
            result["status"] = "failed"
            result["error_message"] = str(exc)

            if config.checkpointing.enabled and config.checkpointing.write_status_after_each_repo:
                payload = {
                    "status": "failed",
                    "repo_full_name": repo_full_name,
                    "error_message": str(exc)}
                write_repo_checkpoint(
                    config,
                    CHECKPOINT_PREFIX,
                    repo_full_name,
                    payload,
                )

        summary_rows.append(result)

        pause_seconds = get_issue_extraction_option(config, "request_pause_seconds_between_repos", 0)
        if pause_seconds:
            time.sleep(pause_seconds)

    merge_pr_batches(config, logger)

    summary_output = Path(config.logging.extraction_log_dir) / "03_extract_prs_summary.csv"
    write_summary_csv(summary_rows, summary_output)
    write_run_manifest(config, repo_rows, summary_rows)

    logger.info("PR extraction complete.")


if __name__ == "__main__":
    main()