"""
IF MEMORY ISSUES FOR REALLY BIG REPOS:
write per-pass raw JSONL keyed by commit
or write per-pass parquet shards
then merge them locally for each repo in a second streaming phase
(maybe test with smth big like vs code to see if we need this)
"""

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.study_config_loader import ensure_project_directories, load_study_config
from utils.github_api import build_session, fetch_repository_metadata, get_github_headers
from utils.io_helpers import load_repo_list, write_processed_table, append_jsonl_row, reset_output_file
from utils.checkpoints import get_batch_root, get_repo_output_root, reset_batch_root, should_skip_repo, write_repo_checkpoint, sanitize_repo_name
from utils.chunk_writers import CommitHistoryRepoChunkWriter

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"
LOG_FILENAME = "04_extract_commit_histories.log"
CHECKPOINT_PREFIX = "04_extract_commit_histories"
BATCH_FOLDER_NAME = "commit_histories"
RAW_FOLDER_NAME = "git_logs"
FIELD_SEPARATOR = "\x1f"
COMMIT_HEADER = "__COMMIT__"


def setup_logger(config):
    logger = logging.getLogger("extract_commit_histories")
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


# def get_issue_extraction_option(config, field_name, default_value):
#     if not hasattr(config, "issue_extraction"):
#         return default_value
#     if not hasattr(config.issue_extraction, field_name):
#         return default_value
#     value = getattr(config.issue_extraction, field_name)
#     if value is None:
#         return default_value
#     return value

def get_git_runtime_option(config, field_name, default_value):
    if not hasattr(config, "git_history_extraction"):
        return default_value
    if not hasattr(config.git_history_extraction, field_name):
        return default_value
    value = getattr(config.git_history_extraction, field_name)
    if value is None:
        return default_value
    return value


def get_git_history_option(config, field_name, default_value):
    if not hasattr(config, "git_history_extraction"):
        return default_value
    if not hasattr(config.git_history_extraction, field_name):
        return default_value
    value = getattr(config.git_history_extraction, field_name)
    if value is None:
        return default_value
    return value


def get_git_extract_option(config, field_name, default_value):
    if not hasattr(config, "git_history_extraction"):
        return default_value
    extract = getattr(config.git_history_extraction, "extract", None)
    if extract is None or not hasattr(extract, field_name):
        return default_value
    value = getattr(extract, field_name)
    if value is None:
        return default_value
    return value


def get_git_message_mode(config):
    if not hasattr(config, "git_history_extraction"):
        return "full"
    value = getattr(config.git_history_extraction, "commit_message_mode", "full")
    if value is None:
        return "full"
    return value


def get_effective_git_history_window(config):
    git_cfg = getattr(config, "git_history_extraction", None)
    if git_cfg is None:
        return None, None
    if not getattr(git_cfg, "fast_mode", False):
        return getattr(git_cfg, "history_start_date", None), getattr(git_cfg, "history_end_date", None)

    window_mode = getattr(git_cfg, "fast_mode_date_window", "participation_analysis")
    if window_mode == "issue_collection":
        return config.study_windows.issue_collection.start_date, config.study_windows.issue_collection.end_date
    if window_mode == "participation_analysis":
        return config.study_windows.participation_analysis.start_date, config.study_windows.participation_analysis.end_date
    if window_mode == "explicit_history_dates":
        return getattr(git_cfg, "history_start_date", None), getattr(git_cfg, "history_end_date", None)

    return getattr(git_cfg, "history_start_date", None), getattr(git_cfg, "history_end_date", None)


def trim_commit_message(value, config):
    if value is None:
        return None
    max_chars = getattr(config.git_history_extraction, "max_commit_message_chars", None)
    if max_chars is None:
        return value
    return value[:max_chars]


def new_repo_result(repo_full_name, repo_id=None):
    return {
        "repo_full_name": repo_full_name,
        "repo_id": repo_id,
        "status": "started",
        "clone_performed": False,
        "clone_refreshed": False,
        "git_commits_seen": 0,
        "git_file_records_seen": 0,
        "commit_rows_written": 0,
        "commit_file_rows_written": 0,
        "raw_files_written": 0,
        "raw_commits_written": 0,
        "raw_commit_files_written": 0,
        "error_message": "",
    }


def run_subprocess(command, *, cwd=None):
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
        capture_output=True,
        text=True,
    )


def stream_subprocess_lines(command, *, cwd=None):
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        assert process.stdout is not None
        for line in process.stdout:
            yield line.rstrip("\n")

        return_code = process.wait()
        if return_code != 0:
            stderr_text = ""
            if process.stderr is not None:
                stderr_text = process.stderr.read()
            raise subprocess.CalledProcessError(
                return_code,
                command,
                output=None,
                stderr=stderr_text,
            )
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


def clone_or_update_repo(config, logger, repo_full_name):
    clone_root = Path(config.git_history_extraction.clone_root)
    clone_root.mkdir(parents=True, exist_ok=True)
    safe_repo = sanitize_repo_name(repo_full_name)
    repo_clone_dir = clone_root / safe_repo
    clone_performed = False
    clone_refreshed = False

    if not repo_clone_dir.exists():
        clone_url = f"https://github.com/{repo_full_name}.git"
        logger.info("Cloning repo %s into %s", repo_full_name, repo_clone_dir)
        run_subprocess(["git", "clone", clone_url, str(repo_clone_dir)])
        clone_performed = True
        return repo_clone_dir, clone_performed, clone_refreshed

    logger.info("Refreshing existing clone for %s", repo_full_name)
    run_subprocess(["git", "fetch", "--all", "--tags", "--prune"], cwd=repo_clone_dir)
    clone_refreshed = True
    return repo_clone_dir, clone_performed, clone_refreshed


def resolve_repo_id(session, headers, config, logger, repo_row):
    repo_full_name = repo_row["full_name"]
    repo_id = repo_row.get("repo_id")
    if pd.notna(repo_id):
        return repo_id, None
    logger.info("repo_id missing in repo list; fetching repository metadata for %s", repo_full_name)
    repo_payload = fetch_repository_metadata(session, headers, config, logger, repo_full_name)
    return repo_payload.get("id"), repo_payload


def build_git_pretty_format(config):
    message_mode = get_git_message_mode(config)
    include_commit_message = (
        get_git_extract_option(config, "commit_message", True)
        and message_mode != "none"
    )
    if include_commit_message:
        message_field = "%s" if message_mode == "subject_only" else "%B"
    else:
        message_field = ""
    pretty_fields = ["%H", "%P", "%an", "%ae", "%aI", message_field]
    return f"{COMMIT_HEADER}%n" + FIELD_SEPARATOR.join(pretty_fields)


def append_history_window_args(command, config):
    history_start_date, history_end_date = get_effective_git_history_window(config)
    include_full_history = get_git_history_option(config, "include_full_history", True)
    fast_mode = get_git_history_option(config, "fast_mode", False)
    if fast_mode:
        include_full_history = False
    if not include_full_history and history_start_date:
        command.append(f"--since={history_start_date}")
    elif history_start_date:
        command.append(f"--since={history_start_date}")
    if history_end_date:
        command.append(f"--until={history_end_date}")
    return command


def build_git_log_command_numstat(config):
    command = ["git", "log", "--all", "--date=iso-strict", f"--pretty=format:{build_git_pretty_format(config)}", "--numstat"]
    return append_history_window_args(command, config)


def build_git_log_command_name_status(config):
    command = ["git", "log", "--all", "--date=iso-strict", f"--pretty=format:{build_git_pretty_format(config)}", "--name-status"]
    if get_git_extract_option(config, "renames_when_detectable", True):
        command.append("--find-renames")
        command.append("--find-copies")
    return append_history_window_args(command, config)


def is_numstat_token(value):
    return value == "-" or value.isdigit()


def looks_like_numstat_line(line):
    parts = line.split("\t")
    if len(parts) < 3:
        return False
    left = parts[0]
    right = parts[1]
    return is_numstat_token(left) and is_numstat_token(right)


def looks_like_name_status_line(line):
    parts = line.split("\t")
    if len(parts) < 2:
        return False
    status = parts[0]
    if not status:
        return False
    lead = status[0]
    return lead in {"A", "M", "D", "R", "C", "T", "U"}


def iter_git_log_records(command, *, cwd=None, file_detail_mode=None, expect_multiline_message=True):
    current_record = None
    commit_message_lines = []
    current_section = None
    for line in stream_subprocess_lines(command, cwd=cwd):
        if line == COMMIT_HEADER:
            if current_record is not None:
                current_record["commit_message"] = "\n".join(commit_message_lines).strip()
                yield current_record

            current_record = {
                "commit_sha": None,
                "parent_shas": [],
                "author_name": None,
                "author_email": None,
                "commit_timestamp": None,
                "commit_message": "",
                "numstat_rows": [],
                "name_status_rows": [],
            }
            commit_message_lines = []
            current_section = "metadata"
            continue
        if current_record is None:
            continue

        if current_section == "metadata":
            parts = line.split(FIELD_SEPARATOR)
            commit_sha = parts[0] if len(parts) > 0 else None
            parent_shas_text = parts[1] if len(parts) > 1 else ""
            author_name = parts[2] if len(parts) > 2 else None
            author_email = parts[3] if len(parts) > 3 else None
            commit_timestamp = parts[4] if len(parts) > 4 else None
            message_first_line = FIELD_SEPARATOR.join(parts[5:]) if len(parts) > 5 else ""

            current_record["commit_sha"] = commit_sha
            current_record["parent_shas"] = [value for value in parent_shas_text.split() if value]
            current_record["author_name"] = author_name
            current_record["author_email"] = author_email
            current_record["commit_timestamp"] = commit_timestamp
            if message_first_line:
                commit_message_lines.append(message_first_line)
            if expect_multiline_message:
                current_section = "message"
            else:
                current_section = "file_detail"
            continue

        if current_section == "message":
            if line == "":
                current_section = "file_detail"
                continue
            commit_message_lines.append(line)
            continue

        if current_section in {"file_detail"}:
            if line == "":
                continue
            if file_detail_mode == "numstat":
                parts = line.split("\t")
                if len(parts) >= 3 and looks_like_numstat_line(line):
                    current_record["numstat_rows"].append(parts)
                continue
            if file_detail_mode == "name_status":
                parts = line.split("\t")
                if len(parts) >= 2 and looks_like_name_status_line(line):
                    current_record["name_status_rows"].append(parts)
                continue
            continue

        if current_section == "done":
            continue

    if current_record is not None:
        current_record["commit_message"] = "\n".join(commit_message_lines).strip()
        yield current_record


def parse_int_or_none(value):
    if value in {None, "", "-"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_name_status_row(parts):
    status = parts[0] if parts else None
    change_type = status[0] if status else None

    old_path = None
    new_path = None

    if change_type in {"R", "C"} and len(parts) >= 3:
        old_path = parts[1]
        new_path = parts[2]
    elif len(parts) >= 2:
        new_path = parts[1]

    return {
        "status": status,
        "change_type": change_type,
        "old_path": old_path,
        "file_path": new_path,
    }


def make_numstat_entry(parts):
    return {
        "file_path": parts[2] if len(parts) >= 3 else None,
        "additions": parse_int_or_none(parts[0]) if len(parts) > 0 else None,
        "deletions": parse_int_or_none(parts[1]) if len(parts) > 1 else None,
    }


def choose_merge_key_from_name_status(entry):
    change_type = entry.get("change_type")
    if change_type in {"R", "C"}:
        # use the new path if possible
        return entry.get("file_path") or entry.get("old_path")
    return entry.get("file_path")


def combine_file_changes(record):
    name_status_entries = [parse_name_status_row(parts) for parts in record.get("name_status_rows", [])]
    numstat_entries = [make_numstat_entry(parts) for parts in record.get("numstat_rows", [])]
    if not name_status_entries and not numstat_entries:
        return []

    merged_by_key = {}
    ordered_keys = []

    def ensure_key(key):
        if key not in merged_by_key:
            merged_by_key[key] = {
                "file_path": None,
                "old_file_path": None,
                "additions": None,
                "deletions": None,
                "change_type": None,
                "raw_change_status": None,
            }
            ordered_keys.append(key)
        return merged_by_key[key]

    for entry in numstat_entries:
        key = entry.get("file_path")
        if not key:
            continue
        row = ensure_key(key)
        row["file_path"] = entry.get("file_path")
        row["additions"] = entry.get("additions")
        row["deletions"] = entry.get("deletions")

    for entry in name_status_entries:
        key = choose_merge_key_from_name_status(entry)
        if not key:
            continue
        row = ensure_key(key)
        row["file_path"] = entry.get("file_path") or row["file_path"]
        row["old_file_path"] = entry.get("old_path")
        row["change_type"] = entry.get("change_type")
        row["raw_change_status"] = entry.get("status")

    return [merged_by_key[key] for key in ordered_keys]


def flatten_commit_row(record, repo_id, repo_full_name, config):
    return {
        "repo_id": repo_id,
        "repo_full_name": repo_full_name,
        "commit_sha": record.get("commit_sha"),
        "author_name": record.get("author_name") if get_git_extract_option(config, "commit_author_name", True) else None,
        "author_email": record.get("author_email") if get_git_extract_option(config, "commit_author_email", True) else None,
        "commit_timestamp": record.get("commit_timestamp") if get_git_extract_option(config, "commit_timestamp", True) else None,
        "commit_message": (trim_commit_message(record.get("commit_message"), config)
                           if get_git_extract_option(config, "commit_message", True)
                           and get_git_message_mode(config) != "none" else None),
        "parent_shas_json": json.dumps(record.get("parent_shas") or []) if get_git_extract_option(config, "parent_shas", True) else None,
        "source_system": "local_git",
    }


def flatten_commit_file_rows(record, repo_id, repo_full_name, config):
    if not get_git_extract_option(config, "modified_files", True):
        return []

    rows = []
    for file_row in combine_file_changes(record):
        rows.append(
            {
                "repo_id": repo_id,
                "repo_full_name": repo_full_name,
                "commit_sha": record.get("commit_sha"),
                "file_path": file_row.get("file_path"),
                "old_file_path": file_row.get("old_file_path") if get_git_extract_option(config, "renames_when_detectable", True) else None,
                "additions": file_row.get("additions") if get_git_extract_option(config, "additions_deletions", True) else None,
                "deletions": file_row.get("deletions") if get_git_extract_option(config, "additions_deletions", True) else None,
                "change_type": file_row.get("change_type") if get_git_extract_option(config, "file_change_type", True) else None,
                "raw_change_status": file_row.get("raw_change_status"),
            }
        )
    return rows


def make_raw_commit_row(record, repo_id, repo_full_name):
    return {
        "repo_id": repo_id,
        "repo_full_name": repo_full_name,
        "commit_sha": record.get("commit_sha"),
        "parent_shas": record.get("parent_shas") or [],
        "author_name": record.get("author_name"),
        "author_email": record.get("author_email"),
        "commit_timestamp": record.get("commit_timestamp"),
        "commit_message": record.get("commit_message"),
    }


def collect_repo_part_files(batch_root, part_glob):
    repo_part_map = {}
    for repo_dir in sorted(batch_root.iterdir()):
        if not repo_dir.is_dir():
            continue
        repo_name = repo_dir.name
        part_paths = sorted(repo_dir.glob(part_glob))
        if part_paths:
            repo_part_map[repo_name] = part_paths
    return repo_part_map


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
    manifest_path = Path(config.logging.extraction_log_dir) / "04_extract_commit_histories_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "extract_commit_histories.py",
        "repo_count_requested": len(repo_rows),
        "repo_count_processed": len(summary_rows),
        "completed_repo_count": sum(1 for row in summary_rows if row.get("status") == "completed"),
        "failed_repo_count": sum(1 for row in summary_rows if row.get("status") == "failed"),
        "processed_merge_mode": "single_parquet",
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary_rows": summary_rows,
        "git_history_fast_mode": get_git_history_option(config, "fast_mode", False),
        "git_history_fast_mode_date_window": get_git_history_option(config, "fast_mode_date_window", "participation_analysis"),
        "commit_message_mode": get_git_message_mode(config),
        "max_commit_message_chars": get_git_history_option(config, "max_commit_message_chars", None),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def merge_commit_batches(config, logger):
    batch_root = get_batch_root(config, BATCH_FOLDER_NAME)
    if not batch_root.exists():
        logger.warning("Commit-history batch root does not exist: %s", batch_root)
        return

    commit_repo_parts = collect_repo_part_files(batch_root, "commits_part_*.parquet")
    commit_file_repo_parts = collect_repo_part_files(batch_root, "commit_files_part_*.parquet")
    commit_parts = [path for paths in commit_repo_parts.values() for path in paths]
    commit_file_parts = [path for paths in commit_file_repo_parts.values() for path in paths]

    commit_df = merge_part_files(commit_parts, sort_columns=["repo_full_name", "commit_timestamp", "commit_sha"])
    commit_file_df = merge_part_files(commit_file_parts, sort_columns=["repo_full_name", "commit_sha", "file_path", "change_type"])

    if not commit_df.empty:
        commit_df = commit_df.drop_duplicates(subset=["repo_full_name", "commit_sha"])
        write_processed_table(commit_df, Path(config.outputs.commits_table), config)
        logger.info("Wrote merged commits table to %s", config.outputs.commits_table)
    else:
        logger.warning("No commit parts found to merge.")

    if not commit_file_df.empty:
        commit_file_df = commit_file_df.drop_duplicates(subset=["repo_full_name", "commit_sha", "file_path", "old_file_path", "change_type"])
        write_processed_table(commit_file_df, Path(config.outputs.commit_files_table), config)
        logger.info("Wrote merged commit-files table to %s", config.outputs.commit_files_table)
    else:
        logger.warning("No commit-file parts found to merge.")


def process_repo(session, headers, config, logger, repo_row):
    repo_full_name = repo_row["full_name"]
    repo_id, repo_payload = resolve_repo_id(session, headers, config, logger, repo_row)
    if pd.isna(repo_id):
        repo_id = None
    if repo_id is None and get_git_runtime_option(config, "fail_on_missing_repo_id", True):
        raise ValueError(f"Missing repo_id for {repo_full_name}")
    if repo_payload is None:
        logger.info("Using repo_id from repo list | repo=%s | repo_id=%s", repo_full_name, repo_id)
    else:
        logger.info("Fetched repo_id from GitHub metadata | repo=%s | repo_id=%s", repo_full_name, repo_id)

    result = new_repo_result(repo_full_name, repo_id=repo_id)
    raw_root = get_repo_output_root(config, RAW_FOLDER_NAME, repo_full_name, raw_source="git_logs")
    batch_size = get_git_runtime_option(config, "write_batch_size", 5000)
    safe_repo = sanitize_repo_name(repo_full_name)
    repo_dir = get_batch_root(config, BATCH_FOLDER_NAME) / safe_repo
    writer = CommitHistoryRepoChunkWriter(config=config, repo_dir=repo_dir, batch_size=batch_size)

    local_repo_path, clone_performed, clone_refreshed = clone_or_update_repo(config, logger, repo_full_name)
    result["clone_performed"] = clone_performed
    result["clone_refreshed"] = clone_refreshed

    logger.info("Streaming commit history | repo=%s | local_path=%s", repo_full_name, local_repo_path)

    raw_commits_path = raw_root / "commits.jsonl"
    raw_commit_files_path = raw_root / "commit_files.jsonl"
    reset_output_file(raw_commits_path, use_gzip=config.storage.compression.raw_json_gzip)
    reset_output_file(raw_commit_files_path, use_gzip=config.storage.compression.raw_json_gzip)
    seen_commit_keys = set()
    seen_commit_file_keys = set()
    expect_multiline_message = (get_git_extract_option(config, "commit_message", True)
                                and get_git_message_mode(config) == "full")
    numstat_records_by_sha = {}
    name_status_records_by_sha = {}

    # part 1: metadata and numstat
    numstat_command = build_git_log_command_numstat(config)
    for record in iter_git_log_records(
            numstat_command,
            cwd=local_repo_path,
            file_detail_mode="numstat",
            expect_multiline_message=expect_multiline_message,
    ):
        commit_sha = record.get("commit_sha")
        if commit_sha:
            numstat_records_by_sha[commit_sha] = record
        if len(numstat_records_by_sha) and len(numstat_records_by_sha) % 50000 == 0:
            logger.warning("Large numstat record map | repo=%s | commits_buffered=%s", repo_full_name, len(numstat_records_by_sha))

    # part 2: metadata and name-status
    name_status_command = build_git_log_command_name_status(config)
    for record in iter_git_log_records(
            name_status_command,
            cwd=local_repo_path,
            file_detail_mode="name_status",
            expect_multiline_message=expect_multiline_message,
    ):
        commit_sha = record.get("commit_sha")
        if commit_sha:
            name_status_records_by_sha[commit_sha] = record
        if len(name_status_records_by_sha) and len(name_status_records_by_sha) % 50000 == 0:
            logger.warning("Large name-status record map | repo=%s | commits_buffered=%s", repo_full_name, len(name_status_records_by_sha))

    missing_in_name_status = set(numstat_records_by_sha) - set(name_status_records_by_sha)
    missing_in_numstat = set(name_status_records_by_sha) - set(numstat_records_by_sha)
    if missing_in_name_status:
        logger.warning(
            "Some commits seen in numstat pass but not name-status pass | repo=%s | count=%s",
            repo_full_name,
            len(missing_in_name_status),
        )
    if missing_in_numstat:
        logger.warning(
            "Some commits seen in name-status pass but not numstat pass | repo=%s | count=%s",
            repo_full_name,
            len(missing_in_numstat),
        )

    all_commit_shas = sorted(set(numstat_records_by_sha) | set(name_status_records_by_sha))
    for commit_sha in all_commit_shas:
        numstat_record = numstat_records_by_sha.get(commit_sha)
        name_status_record = name_status_records_by_sha.get(commit_sha)

        base_record = numstat_record or name_status_record
        merged_record = dict(base_record)
        merged_record["numstat_rows"] = (numstat_record.get("numstat_rows") or []) if numstat_record else []
        merged_record["name_status_rows"] = (
                    name_status_record.get("name_status_rows") or []) if name_status_record else []

        result["git_commits_seen"] += 1

        commit_row = flatten_commit_row(merged_record, repo_id, repo_full_name, config)
        commit_key = (commit_row.get("repo_full_name"), commit_row.get("commit_sha"))
        if commit_key not in seen_commit_keys:
            seen_commit_keys.add(commit_key)
            writer.add_commit_row(commit_row)
            result["commit_rows_written"] += 1

        raw_commit_row = make_raw_commit_row(merged_record, repo_id, repo_full_name)
        result["raw_files_written"] += append_jsonl_row(
            raw_commit_row,
            raw_commits_path,
            use_gzip=config.storage.compression.raw_json_gzip,
        )
        result["raw_commits_written"] += 1

        file_rows = flatten_commit_file_rows(merged_record, repo_id, repo_full_name, config)
        result["git_file_records_seen"] += len(file_rows)

        for file_row in file_rows:
            file_key = (file_row.get("repo_full_name"),
                        file_row.get("commit_sha"),
                        file_row.get("file_path"),
                        file_row.get("old_file_path"),
                        file_row.get("change_type"))
            if file_key in seen_commit_file_keys:
                continue

            seen_commit_file_keys.add(file_key)
            writer.add_commit_file_row(file_row)
            result["commit_file_rows_written"] += 1

            result["raw_files_written"] += append_jsonl_row(
                file_row,
                raw_commit_files_path,
                use_gzip=config.storage.compression.raw_json_gzip,
            )
            result["raw_commit_files_written"] += 1

    logger.info(
        "Finished streaming | repo=%s | commits_written=%s | commit_file_rows_written=%s",
        repo_full_name,
        result["commit_rows_written"],
        result["commit_file_rows_written"],
    )
    writer.finalize()
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
                "clone_performed": clone_performed,
                "clone_refreshed": clone_refreshed,
                "git_commits_seen": result["git_commits_seen"],
                "git_file_records_seen": result["git_file_records_seen"],
                "commit_rows_written": result["commit_rows_written"],
                "commit_file_rows_written": result["commit_file_rows_written"],
                "raw_files_written": result["raw_files_written"],
            },
        )

    return result


def main():
    config = load_study_config(DEFAULT_CONFIG_PATH)
    ensure_project_directories(config)
    logger = setup_logger(config)

    logger.info("Loaded config from %s", DEFAULT_CONFIG_PATH)

    if not get_git_history_option(config, "enabled", True):
        logger.warning("git_history_extraction.enabled is false. Nothing to do.")
        return

    repo_rows = load_repo_list(Path(config.outputs.repo_included_list))
    max_repos_per_run = get_git_runtime_option(config, "max_repos_per_run", None)
    if max_repos_per_run:
        repo_rows = repo_rows[:max_repos_per_run]

    if not repo_rows:
        logger.warning("Repo list is empty. Nothing to extract.")
        return

    resume_mode = get_git_runtime_option(config, "resume_mode", "checkpoint_only")
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
            section_name="git_history_extraction",
            raw_source="git_logs"
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

        logger.info("Starting commit-history extraction for %s", repo_full_name)

        try:
            result = process_repo(session, headers, config, logger, repo_row)
        except Exception as exc:
            logger.exception("Failed commit-history extraction for %s", repo_full_name)
            result = new_repo_result(repo_full_name, repo_id=repo_row.get("repo_id"))
            result["status"] = "failed"
            result["error_message"] = str(exc)

            if config.checkpointing.enabled and config.checkpointing.write_status_after_each_repo:
                write_repo_checkpoint(
                    config,
                    CHECKPOINT_PREFIX,
                    repo_full_name,
                    {
                        "status": "failed",
                        "repo_full_name": repo_full_name,
                        "repo_id": repo_row.get("repo_id"),
                        "error_message": str(exc),
                    },
                )

        summary_rows.append(result)

        pause_seconds = get_git_runtime_option(config, "request_pause_seconds_between_repos", 0)
        if pause_seconds:
            time.sleep(pause_seconds)

    merge_commit_batches(config, logger)

    summary_path = Path(config.logging.extraction_log_dir) / "04_extract_commit_histories_summary.csv"
    write_summary_csv(summary_rows, summary_path)
    write_run_manifest(config, repo_rows, summary_rows)

    logger.info(
        "Commit-history extraction complete | repos_requested=%s | repos_processed=%s",
        len(repo_rows),
        len(summary_rows),
    )


if __name__ == "__main__":
    main()
