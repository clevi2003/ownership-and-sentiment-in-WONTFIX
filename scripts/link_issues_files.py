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
from utils.checkpoints import get_batch_root, get_stage_option, reset_batch_root, should_skip_repo, write_repo_checkpoint, sanitize_repo_name
from utils.chunk_writers import IssueFileLinkRepoChunkWriter
from utils.io_helpers import load_repo_list, write_csv_rows, write_processed_table, read_repo_partitioned_dataset, read_parquet_if_exists
from utils.regex_expressions import BACKTICK_PATTERN, FILENAME_PATTERN, LEADING_PUNCTUATION, PATHISH_PATTERN, TRAILING_PUNCTUATION

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"
LOG_FILENAME = "05_link_issue_files.log"
CHECKPOINT_PREFIX = "05_link_issue_files"
BATCH_FOLDER_NAME = "issue_file_links"
RAW_FOLDER_NAME = "issue_file_links"


def setup_logger(config):
    logger = logging.getLogger("link_issue_files")
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


def get_partitioned_output_root(output_path):
    output_path = Path(output_path)
    return output_path.with_suffix("").with_name(output_path.stem + "_dataset")


def get_issue_file_linking_option(config, field_name, default_value):
    return get_stage_option(config, "issue_file_linking", field_name, default_value)


def get_issue_file_confidence(config, source_name, default_value):
    linkage_cfg = getattr(config, "linkage", None)
    if linkage_cfg is None:
        return default_value
    issue_file_cfg = getattr(linkage_cfg, "issue_file", None)
    if issue_file_cfg is None:
        return default_value
    confidence_levels = getattr(issue_file_cfg, "confidence_levels", None)
    if not isinstance(confidence_levels, dict):
        return default_value
    return confidence_levels.get(source_name, default_value)


def new_repo_result(repo_full_name, repo_id=None):
    return {
        "repo_full_name": repo_full_name,
        "repo_id": repo_id,
        "status": "started",
        "issues_seen": 0,
        "issue_comments_seen": 0,
        "issue_pr_rows_seen": 0,
        "pr_commit_rows_seen": 0,
        "commit_file_rows_seen": 0,
        "issues_linked_by_pr_chain": 0,
        "issues_linked_by_text_fallback": 0,
        "issues_with_no_file_link": 0,
        "issue_file_rows_written": 0,
        "fallback_candidates_seen": 0,
        "fallback_candidates_matched": 0,
        "error_message": "",
    }


def load_stage_inputs_for_repo(config, repo_full_name):
    issues_df = read_parquet_if_exists(config.outputs.issues_table)
    comments_df = read_parquet_if_exists(config.outputs.issue_comments_table)
    issue_pr_df = read_parquet_if_exists(config.outputs.issue_pr_links_table)
    pr_commit_df = read_parquet_if_exists(config.outputs.pr_commit_links_table)

    merge_mode = getattr(config.storage, "processed_merge_mode", "single_parquet")
    if merge_mode == "partitioned_dataset":
        commit_files_df = read_repo_partitioned_dataset(config.outputs.commit_files_table, repo_full_name)
    else:
        commit_files_df = read_parquet_if_exists(config.outputs.commit_files_table)

    def repo_filter(df):
        if df.empty or "repo_full_name" not in df.columns:
            return df.iloc[0:0].copy()
        return df[df["repo_full_name"] == repo_full_name].copy()

    return {"issues": repo_filter(issues_df),
            "comments": repo_filter(comments_df),
            "issue_pr": repo_filter(issue_pr_df),
            "pr_commit": repo_filter(pr_commit_df),
            "commit_files": repo_filter(commit_files_df)}


def normalize_issue_number_series(df):
    if df.empty or "issue_number" not in df.columns:
        return df
    df = df.copy()
    df["issue_number"] = pd.to_numeric(df["issue_number"], errors="coerce")
    df = df[df["issue_number"].notna()].copy()
    df["issue_number"] = df["issue_number"].astype(int)
    return df


def normalize_pr_number_series(df):
    if df.empty or "pr_number" not in df.columns:
        return df
    df = df.copy()
    df["pr_number"] = pd.to_numeric(df["pr_number"], errors="coerce")
    df = df[df["pr_number"].notna()].copy()
    df["pr_number"] = df["pr_number"].astype(int)
    return df


def build_repo_indexes(issues_df, comments_df, issue_pr_df, pr_commit_df, commit_files_df):
    issues_df = normalize_issue_number_series(issues_df)
    comments_df = normalize_issue_number_series(comments_df)
    issue_pr_df = normalize_issue_number_series(normalize_pr_number_series(issue_pr_df))
    pr_commit_df = normalize_pr_number_series(pr_commit_df)

    issue_rows_by_number = {}
    for row in issues_df.to_dict(orient="records"):
        issue_rows_by_number[int(row["issue_number"])] = row

    comments_by_issue_number = {}
    if not comments_df.empty:
        for row in comments_df.to_dict(orient="records"):
            issue_number = int(row["issue_number"])
            comments_by_issue_number.setdefault(issue_number, []).append(row)

    issue_pr_rows_by_issue_number = {}
    if not issue_pr_df.empty:
        for row in issue_pr_df.to_dict(orient="records"):
            issue_number = int(row["issue_number"])
            issue_pr_rows_by_issue_number.setdefault(issue_number, []).append(row)

    commit_shas_by_pr_id = {}
    commit_shas_by_pr_number = {}
    if not pr_commit_df.empty:
        for row in pr_commit_df.to_dict(orient="records"):
            commit_sha = row.get("commit_sha")
            if not commit_sha:
                continue

            pr_id = row.get("pr_id")
            if pd.notna(pr_id):
                commit_shas_by_pr_id.setdefault(pr_id, []).append(commit_sha)

            pr_number = row.get("pr_number")
            if pd.notna(pr_number):
                commit_shas_by_pr_number.setdefault(int(pr_number), []).append(commit_sha)

    commit_file_rows_by_sha = {}
    if not commit_files_df.empty:
        for row in commit_files_df.to_dict(orient="records"):
            commit_sha = row.get("commit_sha")
            if not commit_sha:
                continue
            commit_file_rows_by_sha.setdefault(commit_sha, []).append(row)

    return {
        "issues_by_number": issue_rows_by_number,
        "comments_by_issue_number": comments_by_issue_number,
        "issue_pr_rows_by_issue_number": issue_pr_rows_by_issue_number,
        "commit_shas_by_pr_id": commit_shas_by_pr_id,
        "commit_shas_by_pr_number": commit_shas_by_pr_number,
        "commit_file_rows_by_sha": commit_file_rows_by_sha,
    }


def normalize_text_candidate(candidate):
    if candidate is None:
        return None

    value = str(candidate).strip()
    if not value:
        return None

    value = value.strip(LEADING_PUNCTUATION).rstrip(TRAILING_PUNCTUATION)
    value = value.replace("\\", "/")
    while "//" in value:
        value = value.replace("//", "/")
    if value.startswith("./"):
        value = value[2:]

    return value.strip() or None


def extract_candidate_file_mentions(text):
    if text is None:
        return []
    if isinstance(text, float) and pd.isna(text):
        return []
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return []

    candidates = []

    for match in PATHISH_PATTERN.finditer(text):
        value = normalize_text_candidate(match.group("path"))
        if value:
            candidates.append(value)

    for match in BACKTICK_PATTERN.finditer(text):
        raw_value = normalize_text_candidate(match.group(1))
        if not raw_value:
            continue
        if "/" in raw_value or FILENAME_PATTERN.search(raw_value):
            candidates.append(raw_value)

    for match in FILENAME_PATTERN.finditer(text):
        value = normalize_text_candidate(match.group("name"))
        if value:
            candidates.append(value)

    deduped = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)

    return deduped


def build_repo_file_universe(commit_files_df):
    all_paths = set()
    basename_to_paths = {}

    if commit_files_df.empty:
        return all_paths, basename_to_paths

    for row in commit_files_df.to_dict(orient="records"):
        for field_name in ("file_path", "old_file_path"):
            value = row.get(field_name)
            normalized = normalize_text_candidate(value)
            if not normalized:
                continue
            all_paths.add(normalized)
            basename = Path(normalized).name
            basename_to_paths.setdefault(basename, set()).add(normalized)

    return all_paths, basename_to_paths


def resolve_text_candidates_to_repo_paths(candidates, all_paths, basename_to_paths, allow_unique_basename_match=True):
    resolved = []
    for candidate in candidates:
        normalized = normalize_text_candidate(candidate)
        if not normalized:
            continue

        if normalized in all_paths:
            resolved.append(normalized)
            continue

        basename = Path(normalized).name
        if allow_unique_basename_match and basename in basename_to_paths:
            matching_paths = sorted(basename_to_paths[basename])
            if len(matching_paths) == 1:
                resolved.append(matching_paths[0])

    deduped = []
    seen = set()
    for path in resolved:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)

    return deduped


def json_dumps_sorted(values):
    cleaned = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        cleaned.append(value)
    return json.dumps(sorted(cleaned))


def make_issue_file_link_row(
    *,
    repo_id,
    repo_full_name,
    issue_id,
    issue_number,
    file_path,
    source,
    confidence_level,
    source_pr_ids=None,
    source_pr_numbers=None,
    source_commit_shas=None,
    evidence_count=None,
    matched_text_source=None,
    matched_text_snippet=None,
):
    return {
        "repo_id": repo_id,
        "repo_full_name": repo_full_name,
        "issue_id": issue_id,
        "issue_number": issue_number,
        "file_path": file_path,
        "source": source,
        "confidence_level": confidence_level,
        "source_pr_ids_json": json_dumps_sorted(source_pr_ids or []),
        "source_pr_numbers_json": json_dumps_sorted(source_pr_numbers or []),
        "source_commit_shas_json": json_dumps_sorted(source_commit_shas or []),
        "evidence_count": evidence_count,
        "matched_text_source": matched_text_source,
        "matched_text_snippet": matched_text_snippet,
    }


def link_issue_files_via_pr_chain(config, repo_full_name, repo_result, indexes):
    confidence = get_issue_file_confidence(config, "pr_commit_chain", "high")
    rows = []
    linked_issue_numbers = set()

    for issue_number, issue_row in sorted(indexes["issues_by_number"].items()):
        issue_pr_rows = indexes["issue_pr_rows_by_issue_number"].get(issue_number, [])
        if not issue_pr_rows:
            continue

        source_pr_ids, source_pr_numbers, source_commit_shas, linked_file_paths = set(), set(), set(), set()

        for issue_pr_row in issue_pr_rows:
            pr_id = issue_pr_row.get("pr_id")
            pr_number = issue_pr_row.get("pr_number")
            if pd.notna(pr_id):
                source_pr_ids.add(pr_id)
            if pd.notna(pr_number):
                source_pr_numbers.add(int(pr_number))

            commit_shas = []
            if pd.notna(pr_id) and pr_id in indexes["commit_shas_by_pr_id"]:
                commit_shas.extend(indexes["commit_shas_by_pr_id"][pr_id])
            elif pd.notna(pr_number) and int(pr_number) in indexes["commit_shas_by_pr_number"]:
                commit_shas.extend(indexes["commit_shas_by_pr_number"][int(pr_number)])

            for commit_sha in commit_shas:
                source_commit_shas.add(commit_sha)
                for file_row in indexes["commit_file_rows_by_sha"].get(commit_sha, []):
                    file_path = normalize_text_candidate(file_row.get("file_path"))
                    if file_path:
                        linked_file_paths.add(file_path)

        if not linked_file_paths:
            continue

        linked_issue_numbers.add(issue_number)
        repo_result["issues_linked_by_pr_chain"] += 1

        for file_path in sorted(linked_file_paths):
            rows.append(make_issue_file_link_row(repo_id=issue_row.get("repo_id"),
                                                repo_full_name=repo_full_name,
                                                issue_id=issue_row.get("issue_id"),
                                                issue_number=issue_number,
                                                file_path=file_path,
                                                source="pr_commit_chain",
                                                confidence_level=confidence,
                                                source_pr_ids=source_pr_ids,
                                                source_pr_numbers=source_pr_numbers,
                                                source_commit_shas=source_commit_shas,
                                                evidence_count=len(source_commit_shas))
                        )
    return rows, linked_issue_numbers


def iter_issue_text_sources(issue_row, comment_rows, include_comment_text_fallback=True):
    def clean_text(value):
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        if not isinstance(value, str):
            value = str(value)
        value = value.strip()
        return value or None

    title = clean_text(issue_row.get("title"))
    if title:
        yield "title", title
    body = clean_text(issue_row.get("body"))
    if body:
        yield "body", body

    if include_comment_text_fallback:
        for comment_row in comment_rows:
            comment_body = clean_text(comment_row.get("body"))
            if comment_body:
                yield "comment", comment_body


def clip_snippet(text, max_chars=240):
    if text is None:
        return None
    text = str(text).strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def link_issue_files_via_text_fallback(config, repo_full_name, repo_result, indexes, excluded_issue_numbers=None):
    excluded_issue_numbers = excluded_issue_numbers or set()
    confidence = get_issue_file_confidence(config, "issue_text_file_reference", "medium")
    include_comments = get_issue_file_linking_option(config, "include_comment_text_fallback", True)
    require_repo_match = get_issue_file_linking_option(config, "require_repo_file_match_for_text_links", True)
    allow_unique_basename_match = get_issue_file_linking_option(config, "allow_unique_basename_match", True)

    commit_file_rows = [row for rows in indexes["commit_file_rows_by_sha"].values() for row in rows]
    all_paths, basename_to_paths = build_repo_file_universe(pd.DataFrame(commit_file_rows))

    rows = []
    linked_issue_numbers = set()

    for issue_number, issue_row in sorted(indexes["issues_by_number"].items()):
        if issue_number in excluded_issue_numbers:
            continue

        comment_rows = indexes["comments_by_issue_number"].get(issue_number, [])
        matched_paths = {}

        for text_source, text_value in iter_issue_text_sources(issue_row, comment_rows, include_comment_text_fallback=include_comments):
            candidates = extract_candidate_file_mentions(text_value)
            repo_result["fallback_candidates_seen"] += len(candidates)
            if not candidates:
                continue

            if require_repo_match:
                resolved_paths = resolve_text_candidates_to_repo_paths(
                    candidates,
                    all_paths,
                    basename_to_paths,
                    allow_unique_basename_match=allow_unique_basename_match,
                )
            else:
                resolved_paths = [value for value in candidates if normalize_text_candidate(value)]

            for resolved_path in resolved_paths:
                repo_result["fallback_candidates_matched"] += 1
                matched_paths.setdefault(resolved_path, {"matched_text_source": text_source, "matched_text_snippet": clip_snippet(text_value)})

        if not matched_paths:
            continue

        linked_issue_numbers.add(issue_number)
        repo_result["issues_linked_by_text_fallback"] += 1

        for file_path in sorted(matched_paths.keys()):
            evidence = matched_paths[file_path]
            rows.append(
                make_issue_file_link_row(repo_id=issue_row.get("repo_id"),
                                        repo_full_name=repo_full_name,
                                        issue_id=issue_row.get("issue_id"),
                                        issue_number=issue_number,
                                        file_path=file_path,
                                        source="issue_text_file_reference",
                                        confidence_level=confidence,
                                        evidence_count=1,
                                        matched_text_source=evidence["matched_text_source"],
                                        matched_text_snippet=evidence["matched_text_snippet"]))
    return rows, linked_issue_numbers

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


def merge_issue_file_batches(config, logger):
    batch_root = get_batch_root(config, BATCH_FOLDER_NAME)
    if not batch_root.exists():
        logger.warning("Issue-file batch root does not exist: %s", batch_root)
        return

    part_paths = list(batch_root.glob("*/issue_file_links_part_*.parquet"))
    merged = merge_part_files(
        part_paths,
        sort_columns=["repo_full_name", "issue_number", "file_path", "source"],
    )

    if merged.empty:
        logger.warning("No issue-file linkage parts found to merge.")
        return

    dedupe_subset = ["repo_full_name", "issue_id", "file_path", "source"]
    existing_subset = [col for col in dedupe_subset if col in merged.columns]
    if existing_subset:
        merged = merged.drop_duplicates(subset=existing_subset).reset_index(drop=True)

    write_processed_table(merged, Path(config.outputs.issue_file_links_table), config)
    logger.info("Wrote merged issue-file links table to %s", config.outputs.issue_file_links_table)


def write_summary_csv(summary_rows, output_path):
    fieldnames = [
        "repo_full_name",
        "repo_id",
        "status",
        "issues_seen",
        "issue_comments_seen",
        "issue_pr_rows_seen",
        "pr_commit_rows_seen",
        "commit_file_rows_seen",
        "issues_linked_by_pr_chain",
        "issues_linked_by_text_fallback",
        "issues_with_no_file_link",
        "issue_file_rows_written",
        "fallback_candidates_seen",
        "fallback_candidates_matched",
        "error_message",
    ]
    write_csv_rows(summary_rows, output_path, fieldnames=fieldnames)


def write_run_manifest(config, repo_rows, summary_rows):
    manifest_path = Path(config.logging.extraction_log_dir) / "05_link_issue_files_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "script": "link_issue_files.py",
        "repo_count_requested": len(repo_rows),
        "repo_count_processed": len(summary_rows),
        "completed_repo_count": sum(1 for row in summary_rows if row.get("status") == "completed"),
        "failed_repo_count": sum(1 for row in summary_rows if row.get("status") == "failed"),
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary_rows": summary_rows,
        "output_table": str(config.outputs.issue_file_links_table),
            "options_used": {
                            "resume_mode": get_issue_file_linking_option(config, "resume_mode", "checkpoint_only"),
                            "write_batch_size": get_issue_file_linking_option(config, "write_batch_size", 5000),
                            "include_comment_text_fallback": get_issue_file_linking_option(config, "include_comment_text_fallback", True),
                            "require_repo_file_match_for_text_links": get_issue_file_linking_option(config, "require_repo_file_match_for_text_links", True),
                            "allow_unique_basename_match": get_issue_file_linking_option(config, "allow_unique_basename_match", True),
                            "max_repos_per_run": get_issue_file_linking_option(config, "max_repos_per_run", None),
                        }
    }

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def process_repo(config, logger, repo_row):
    repo_full_name = repo_row["full_name"]
    repo_id = repo_row.get("repo_id")
    result = new_repo_result(repo_full_name, repo_id=repo_id)

    skip_repo, skip_reason = should_skip_repo(
        config,
        repo_full_name,
        checkpoint_prefix=CHECKPOINT_PREFIX,
        raw_folder_name=RAW_FOLDER_NAME,
        section_name="issue_file_linking",
        raw_source="github_api",
    )
    if skip_repo:
        result["status"] = "skipped"
        result["error_message"] = skip_reason
        return result

    batch_size = get_issue_file_linking_option(
        config,
        "write_batch_size",
        get_stage_option(config, "issue_extraction", "write_batch_size", 5000),
    )
    repo_dir = get_batch_root(config, BATCH_FOLDER_NAME) / sanitize_repo_name(repo_full_name)
    writer = IssueFileLinkRepoChunkWriter(config=config, repo_dir=repo_dir, batch_size=batch_size)

    stage_inputs = load_stage_inputs_for_repo(config, repo_full_name)
    issues_df = normalize_issue_number_series(stage_inputs["issues"])
    comments_df = normalize_issue_number_series(stage_inputs["comments"])
    issue_pr_df = normalize_issue_number_series(normalize_pr_number_series(stage_inputs["issue_pr"]))
    pr_commit_df = normalize_pr_number_series(stage_inputs["pr_commit"])
    commit_files_df = stage_inputs["commit_files"]

    result["issues_seen"] = len(issues_df)
    result["issue_comments_seen"] = len(comments_df)
    result["issue_pr_rows_seen"] = len(issue_pr_df)
    result["pr_commit_rows_seen"] = len(pr_commit_df)
    result["commit_file_rows_seen"] = len(commit_files_df)

    if issues_df.empty:
        logger.info("No issues found for %s. Marking as completed.", repo_full_name)
        result["status"] = "completed"
        write_repo_checkpoint(
            config,
            CHECKPOINT_PREFIX,
            repo_full_name,
            {
                "status": "completed",
                "repo_full_name": repo_full_name,
                "issue_file_rows_written": 0,
            },
        )
        return result

    indexes = build_repo_indexes(issues_df, comments_df, issue_pr_df, pr_commit_df, commit_files_df)

    rows_pr_chain, linked_by_pr_chain = link_issue_files_via_pr_chain(
        config,
        repo_full_name,
        result,
        indexes,
    )
    for row in rows_pr_chain:
        writer.add_issue_file_link_row(row)
    result["issue_file_rows_written"] += len(rows_pr_chain)

    rows_text, linked_by_text = link_issue_files_via_text_fallback(
        config,
        repo_full_name,
        result,
        indexes,
        excluded_issue_numbers=linked_by_pr_chain,
    )
    for row in rows_text:
        writer.add_issue_file_link_row(row)
    result["issue_file_rows_written"] += len(rows_text)

    linked_issue_numbers = set(linked_by_pr_chain) | set(linked_by_text)
    result["issues_with_no_file_link"] = max(result["issues_seen"] - len(linked_issue_numbers), 0)

    writer.finalize()
    result["status"] = "completed"

    write_repo_checkpoint(
        config,
        CHECKPOINT_PREFIX,
        repo_full_name,
        {
            "status": "completed",
            "repo_full_name": repo_full_name,
            "repo_id": repo_id,
            "issue_file_rows_written": result["issue_file_rows_written"],
            "issues_linked_by_pr_chain": result["issues_linked_by_pr_chain"],
            "issues_linked_by_text_fallback": result["issues_linked_by_text_fallback"],
            "issues_with_no_file_link": result["issues_with_no_file_link"],
        },
    )

    return result


def main(config_path=None):
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = load_study_config(config_path)
    ensure_project_directories(config)
    logger = setup_logger(config)

    if not get_issue_file_linking_option(config, "enabled", True):
        logger.info("Issue-file linkage is disabled in config. Exiting.")
        return

    repo_rows = load_repo_list(config.outputs.repo_included_list)
    max_repos_per_run = get_issue_file_linking_option(config, "max_repos_per_run", None)
    if max_repos_per_run is not None:
        repo_rows = repo_rows[: int(max_repos_per_run)]

    resume_mode = get_issue_file_linking_option(config, "resume_mode", "checkpoint_only")
    if resume_mode == "fresh":
        reset_batch_root(config, BATCH_FOLDER_NAME)

    summary_rows = []
    for repo_row in repo_rows:
        repo_full_name = repo_row["full_name"]
        logger.info("Starting issue-file linkage for %s", repo_full_name)
        try:
            result = process_repo(config, logger, repo_row)
        except Exception as exc:
            logger.exception("Issue-file linkage failed for %s", repo_full_name)
            result = new_repo_result(repo_full_name, repo_id=repo_row.get("repo_id"))
            result["status"] = "failed"
            result["error_message"] = str(exc)
        summary_rows.append(result)

    merge_issue_file_batches(config, logger)

    summary_path = Path(config.logging.extraction_log_dir) / "05_link_issue_files_summary.csv"
    write_summary_csv(summary_rows, summary_path)
    write_run_manifest(config, repo_rows, summary_rows)

    logger.info("Wrote issue-file linkage summary to %s", summary_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
