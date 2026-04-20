import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.study_config_loader import ensure_project_directories, load_study_config
from utils.checkpoints import get_batch_root, get_stage_option, reset_batch_root, sanitize_repo_name, should_skip_repo, write_repo_checkpoint
from utils.io_helpers import clean_text, collect_repo_part_files, has_real_value, load_repo_list, load_table, normalize_value, \
    repo_filter, safe_divide, safe_to_datetime, take_mean, take_median, write_merged_or_partitioned_output, write_summary_csv, mean_or_none
from utils.chunk_writers import OwnershipFeatureRepoChunkWriter

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"
CONFIDENCE_RANK = {"very_high": 5, "highest": 5, "high": 4, "medium": 3, "moderate": 3, "low": 2, "very_low": 1, "unknown": 0, None: 0}
PR_EVIDENCE_CONFIDENCE = {"pr_merge": "high", "pr_exact_commit": "high", "pr_head": "moderate", "file_fallback": "low"}


def get_ownership_identity_mode(config):
    mode = get_stage_option(config, "identity_resolution", "attachment_identity_mode", "strict")
    mode = str(mode).strip().lower()
    if mode not in {"strict", "fuzzy"}:
        raise ValueError(
            f"identity_resolution.attachment_identity_mode must be 'strict' or 'fuzzy', got: {mode}"
        )
    return mode

def get_ownership_runtime_names(config):
    mode = get_ownership_identity_mode(config)
    if mode == "fuzzy":
        return {
            "log_filename": "11_build_issue_ownership_features_fuzzy.log",
            "checkpoint_prefix": "11_build_issue_ownership_features_fuzzy",
            "batch_folder_name": "ownership_features_fuzzy",
            "raw_folder_name": "ownership_features_fuzzy",
            "summary_filename": "11_build_issue_ownership_features_fuzzy_summary.csv",
            "run_manifest_filename": "11_build_issue_ownership_features_fuzzy_run_manifest.json",
        }

    return {
        "log_filename": "11_build_issue_ownership_features.log",
        "checkpoint_prefix": "11_build_issue_ownership_features",
        "batch_folder_name": "ownership_features",
        "raw_folder_name": "ownership_features",
        "summary_filename": "11_build_issue_ownership_features_summary.csv",
        "run_manifest_filename": "11_build_issue_ownership_features_run_manifest.json",
    }

def setup_logger(config):
    runtime_names = get_ownership_runtime_names(config)

    logger = logging.getLogger("build_issue_ownership_features")
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
        file_handler = logging.FileHandler(log_dir / runtime_names["log_filename"], encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def get_ownership_option(config, field_name, default_value):
    return get_stage_option(config, "ownership_features", field_name, default_value)

def get_conservative_pre_issue_option(config, field_name, default_value):
    ownership_cfg = getattr(config, "ownership_features", None)
    if ownership_cfg is None:
        return default_value
    if not hasattr(ownership_cfg, field_name):
        return default_value
    value = getattr(ownership_cfg, field_name)
    if value is None:
        return default_value
    return value

def get_stage_paths(config):
    outputs = getattr(config, "outputs", None)
    mode = get_ownership_identity_mode(config)
    runtime_names = get_ownership_runtime_names(config)

    if mode == "fuzzy":
        issue_output_path = getattr(outputs, "issue_ownership_features_table_fuzzy", None)
        if not issue_output_path:
            issue_output_path = "./data/features/ownership_fuzzy/issue_ownership_features_fuzzy.parquet"

        evidence_output_path = getattr(outputs, "issue_file_ownership_evidence_table_fuzzy", None)
        if not evidence_output_path:
            evidence_output_path = "./data/features/ownership_fuzzy/issue_file_ownership_evidence_fuzzy.parquet"

        qa_summary_path = getattr(outputs, "ownership_feature_qa_summary_csv_fuzzy", None)
        if not qa_summary_path:
            qa_summary_path = "./logs/qa/issue_ownership_feature_qa_summary_fuzzy.csv"
    else:
        issue_output_path = getattr(outputs, "issue_ownership_features_table", None)
        if not issue_output_path:
            issue_output_path = "./data/features/ownership/issue_ownership_features.parquet"

        evidence_output_path = getattr(outputs, "issue_file_ownership_evidence_table", None)
        if not evidence_output_path:
            evidence_output_path = "./data/features/ownership/issue_file_ownership_evidence.parquet"

        qa_summary_path = getattr(outputs, "ownership_feature_qa_summary_csv", None)
        if not qa_summary_path:
            qa_summary_path = "./logs/qa/issue_ownership_feature_qa_summary.csv"

    overlap_qa_summary_path = getattr(outputs, "pr_commit_overlap_qa_summary_csv", None)
    if not overlap_qa_summary_path:
        overlap_qa_summary_path = "./logs/qa/pr_commit_overlap_qa_summary.csv"

    return {
        "identity_resolution_mode": mode,
        "issue_output_path": Path(issue_output_path),
        "evidence_output_path": Path(evidence_output_path),
        "qa_summary_path": Path(qa_summary_path),
        "overlap_qa_summary_path": Path(overlap_qa_summary_path),
        "run_manifest_path": Path(config.logging.qa_log_dir) / runtime_names["run_manifest_filename"],
    }

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

def find_first_present_column(df, candidates):
    lower_map = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None

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

def load_stage_inputs_for_repo(config, repo_full_name):
    merge_mode = getattr(config.storage, "processed_merge_mode", "single_parquet")
    mode = get_ownership_identity_mode(config)

    if mode == "fuzzy":
        issues_resolved_path = getattr(config.outputs, "issues_resolved_table_fuzzy")
        issue_comments_resolved_path = getattr(config.outputs, "issue_comments_resolved_table_fuzzy")
        commits_resolved_path = getattr(config.outputs, "commits_resolved_table_fuzzy")
    else:
        issues_resolved_path = config.outputs.issues_resolved_table
        issue_comments_resolved_path = config.outputs.issue_comments_resolved_table
        commits_resolved_path = config.outputs.commits_resolved_table

    issues_df = load_table(issues_resolved_path, repo_full_name=repo_full_name, merge_mode=merge_mode)
    comments_df = load_table(issue_comments_resolved_path, repo_full_name=repo_full_name, merge_mode=merge_mode)
    issue_pr_links_df = load_table(config.outputs.issue_pr_links_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    pull_requests_df = load_table(config.outputs.pull_requests_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    pr_commit_links_df = load_table(config.outputs.pr_commit_links_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    issue_file_links_df = load_table(config.outputs.issue_file_links_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    commit_files_df = load_table(config.outputs.commit_files_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    commits_df = load_table(config.outputs.commits_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    commits_resolved_df = load_table(commits_resolved_path, repo_full_name=repo_full_name, merge_mode=merge_mode)

    return {
        "issues_resolved": repo_filter(issues_df, repo_full_name),
        "issue_comments_resolved": repo_filter(comments_df, repo_full_name),
        "issue_pr_links": repo_filter(issue_pr_links_df, repo_full_name),
        "pull_requests": repo_filter(pull_requests_df, repo_full_name),
        "pr_commit_links": repo_filter(pr_commit_links_df, repo_full_name),
        "issue_file_links": repo_filter(issue_file_links_df, repo_full_name),
        "commit_files": repo_filter(commit_files_df, repo_full_name),
        "commits": repo_filter(commits_df, repo_full_name),
        "commits_resolved": repo_filter(commits_resolved_df, repo_full_name),
    }

def new_repo_result(repo_full_name, repo_id=None):
    return {
        "repo_full_name": repo_full_name,
        "repo_id": repo_id,
        "identity_resolution_mode": None,
        "status": "started",
        "target_issues_requested": 0,
        "issues_resolved_rows_seen": 0,
        "issue_comments_resolved_rows_seen": 0,
        "issue_pr_links_rows_seen": 0,
        "pull_requests_rows_seen": 0,
        "pr_commit_links_rows_seen": 0,
        "issue_file_links_rows_seen": 0,
        "commit_files_rows_seen": 0,
        "commits_rows_seen": 0,
        "commits_resolved_rows_seen": 0,
        "target_issues_kept": 0,
        "ownership_policy_used": None,
        "issues_with_pr_links": 0,
        "issues_with_pr_merge_evidence": 0,
        "issues_with_pr_exact_commit_evidence": 0,
        "issues_with_pr_head_evidence": 0,
        "known_commit_rows_in_lookup": 0,
        "issues_with_any_pr_based_evidence": 0,
        "issues_with_only_file_fallback_evidence": 0,
        "issues_with_file_links": 0,
        "issues_with_high_conf_file_links": 0,
        "issues_with_commit_matches": 0,
        "issues_with_resolved_commit_authors": 0,
        "issues_with_high_confidence_ownership": 0,
        "issues_ok": 0,
        "issues_sparse": 0,
        "issues_no_file_links": 0,
        "issues_no_commit_matches": 0,
        "issues_no_resolved_commit_authors": 0,
        "issues_missing_issue_created_at": 0,
        "issue_rows_written": 0,
        "evidence_rows_written": 0,
        "median_linked_file_count_all": 0.0,
        "median_linked_file_count_high_confidence": 0.0,
        "median_commit_evidence_row_count": 0.0,
        "median_resolved_commit_evidence_row_count": 0.0,
        "median_ownership_contributor_count": 0.0,
        "mean_ownership_top_contributor_share_churn": None,
        "mean_ownership_entropy_churn": None,
        "mean_ownership_discussion_overlap_fraction": None,
        "share_issue_author_is_owner": None,
        "share_top_owner_commented": None,
        "error_message": "",
        "issues_selected_pr_merge_evidence": 0,
        "issues_selected_pr_exact_commit_evidence": 0,
        "issues_selected_pr_head_evidence": 0,
        "issues_selected_file_fallback_evidence": 0,
        "issues_selected_fallback_only": 0,
        "issues_with_pre_issue_ownership": 0,
        "issues_with_post_issue_ownership": 0,
        "issues_with_both_pre_and_post_issue_ownership": 0,
        "issues_with_only_pre_issue_ownership": 0,
        "issues_with_only_post_issue_ownership": 0,

        "issues_with_pre_issue_high_confidence_ownership": 0,
        "issues_with_post_issue_high_confidence_ownership": 0,
        "issues_with_pre_issue_any_ownership": 0,
        "issues_with_post_issue_any_ownership": 0,
        "issues_with_any_conservative_pre_issue_fallback": 0,
        "issues_with_pre_issue_conservative_fallback_only": 0,
        "issues_with_pre_issue_any_but_not_high_confidence": 0,

        "issues_selected_for_high_confidence_features": 0,
        "issues_selected_for_any_features": 0,
        "issues_selected_for_conservative_pre_issue_fallback": 0,

        "total_selected_high_confidence_pre_issue_rows": 0,
        "total_selected_any_pre_issue_rows": 0,
        "total_selected_conservative_pre_issue_rows": 0,

        "mean_pre_issue_high_confidence_contributor_count": None,
        "mean_pre_issue_any_contributor_count": None,
        "mean_pre_issue_conservative_fallback_contributor_count": None,
    }

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

def load_pr_overlap_policy_lookup(stage_paths):
    path = stage_paths["overlap_qa_summary_path"]
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty or "repo_full_name" not in df.columns:
        return {}
    lookup = {}
    for row in df.to_dict(orient="records"):
        repo_name = clean_text(row.get("repo_full_name"))
        if repo_name:
            lookup[repo_name] = row
    return lookup

def select_repo_ownership_policy(config, overlap_row):
    if not overlap_row:
        return "fallback_heavy"
    merge_rate = pd.to_numeric(overlap_row.get("pr_merge_commit_sha_present_rate"), errors="coerce")
    exact_rate = pd.to_numeric(overlap_row.get("pr_commit_sha_overlap_rate"), errors="coerce")
    head_rate = pd.to_numeric(overlap_row.get("pr_head_sha_present_rate"), errors="coerce")

    merge_min = float(get_ownership_option(config, "merge_sha_min_present_rate", 0.80))
    exact_min = float(get_ownership_option(config, "exact_pr_commit_min_overlap_rate", 0.50))
    head_min = float(get_ownership_option(config, "head_sha_min_present_rate", 0.50))

    if pd.notna(merge_rate) and merge_rate >= merge_min:
        if pd.notna(exact_rate) and exact_rate >= exact_min:
            return "exact_plus_merge"
        if pd.notna(head_rate) and head_rate >= head_min:
            return "merge_plus_head"
        return "merge_first"
    if pd.notna(exact_rate) and exact_rate >= exact_min:
        return "exact_first"
    return "fallback_heavy"

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
    return df.reset_index(drop=True)

def confidence_rank(value):
    clean = normalize_value(value)
    return CONFIDENCE_RANK.get(clean, 0)

def build_conservative_pre_issue_support_frame(evidence_df, config):
    columns = [
        "repo_full_name",
        "issue_id",
        "issue_number",
        "commit_author_contributor_key",
        "conservative_pre_issue_distinct_commits",
        "conservative_pre_issue_distinct_files",
        "conservative_pre_issue_distinct_days",
        "conservative_pre_issue_support_ok",
    ]
    if evidence_df is None or evidence_df.empty:
        return pd.DataFrame(columns=columns)

    df = evidence_df.copy()

    enabled = bool(get_conservative_pre_issue_option(config, "enable_conservative_pre_issue_fallback", False))
    if not enabled:
        return pd.DataFrame(columns=columns)

    allowed_sources = {
        normalize_value(value)
        for value in get_conservative_pre_issue_option(
            config,
            "conservative_pre_issue_allowed_sources",
            ["file_fallback"],
        )
    }
    allowed_conf_levels = {
        normalize_value(value)
        for value in get_conservative_pre_issue_option(
            config,
            "conservative_pre_issue_allowed_confidence_levels",
            ["high"],
        )
    }

    min_distinct_commits = int(
        get_conservative_pre_issue_option(config, "conservative_pre_issue_min_distinct_commits", 2)
    )
    min_distinct_files = int(
        get_conservative_pre_issue_option(config, "conservative_pre_issue_min_distinct_files", 1)
    )
    min_distinct_days = int(
        get_conservative_pre_issue_option(config, "conservative_pre_issue_min_distinct_days", 1)
    )
    require_resolved_author = bool(
        get_conservative_pre_issue_option(config, "conservative_pre_issue_require_resolved_author", True)
    )

    if "ownership_time_bucket" not in df.columns:
        return pd.DataFrame(columns=columns)

    df["ownership_time_bucket"] = df["ownership_time_bucket"].apply(normalize_value)
    df = df[df["ownership_time_bucket"] == "pre_issue"].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    if "evidence_type" not in df.columns:
        return pd.DataFrame(columns=columns)
    df["evidence_type"] = df["evidence_type"].apply(normalize_value)
    df = df[df["evidence_type"].isin(allowed_sources)].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    if "issue_file_confidence_level" in df.columns:
        df["issue_file_confidence_level"] = df["issue_file_confidence_level"].apply(normalize_value)
        df = df[df["issue_file_confidence_level"].isin(allowed_conf_levels)].copy()
        if df.empty:
            return pd.DataFrame(columns=columns)

    if require_resolved_author:
        if "commit_author_contributor_key" not in df.columns:
            return pd.DataFrame(columns=columns)
        df["commit_author_contributor_key"] = df["commit_author_contributor_key"].apply(clean_text)
        df = df[df["commit_author_contributor_key"].notna()].copy()
        if df.empty:
            return pd.DataFrame(columns=columns)

    if "commit_sha" in df.columns:
        df["commit_sha"] = df["commit_sha"].apply(clean_text)
    else:
        df["commit_sha"] = None

    if "file_path" in df.columns:
        df["file_path"] = df["file_path"].apply(clean_text)
    else:
        df["file_path"] = None

    if "commit_timestamp" in df.columns:
        df["commit_timestamp"] = safe_to_datetime(df["commit_timestamp"])
        df["commit_date"] = df["commit_timestamp"].dt.strftime("%Y-%m-%d")
    else:
        df["commit_date"] = None

    group_columns = ["repo_full_name", "issue_id", "issue_number", "commit_author_contributor_key"]

    support_df = (
        df.groupby(group_columns, dropna=False)
        .agg(
            conservative_pre_issue_distinct_commits=("commit_sha", lambda s: int(pd.Series(s).dropna().nunique())),
            conservative_pre_issue_distinct_files=("file_path", lambda s: int(pd.Series(s).dropna().nunique())),
            conservative_pre_issue_distinct_days=("commit_date", lambda s: int(pd.Series(s).dropna().nunique())),
        )
        .reset_index()
    )

    support_df["conservative_pre_issue_support_ok"] = (
        (support_df["conservative_pre_issue_distinct_commits"] >= min_distinct_commits)
        & (support_df["conservative_pre_issue_distinct_files"] >= min_distinct_files)
        & (support_df["conservative_pre_issue_distinct_days"] >= min_distinct_days)
    ).astype(int)

    return support_df

def annotate_conservative_pre_issue_selection(evidence_rows, config):
    if evidence_rows is None:
        return []

    if isinstance(evidence_rows, pd.DataFrame):
        input_df = evidence_rows.copy()
    else:
        input_rows = list(evidence_rows)
        if not input_rows:
            return []
        input_df = pd.DataFrame(input_rows)

    if input_df.empty:
        if isinstance(evidence_rows, pd.DataFrame):
            out_df = input_df.copy()
            if "selected_for_conservative_pre_issue_fallback" not in out_df.columns:
                out_df["selected_for_conservative_pre_issue_fallback"] = pd.Series(dtype="int64")
            return out_df
        return []

    out = input_df.copy()
    out["selected_for_conservative_pre_issue_fallback"] = 0

    enabled = bool(get_conservative_pre_issue_option(config, "enable_conservative_pre_issue_fallback", False))
    if not enabled:
        return out.to_dict(orient="records") if not isinstance(evidence_rows, pd.DataFrame) else out

    required_columns = ["repo_full_name", "issue_id", "issue_number", "commit_author_contributor_key"]
    for column_name in required_columns:
        if column_name not in out.columns:
            return out.to_dict(orient="records") if not isinstance(evidence_rows, pd.DataFrame) else out

    support_df = build_conservative_pre_issue_support_frame(out, config)
    if support_df.empty:
        return out.to_dict(orient="records") if not isinstance(evidence_rows, pd.DataFrame) else out

    support_ok_df = support_df[support_df["conservative_pre_issue_support_ok"] == 1].copy()
    if support_ok_df.empty:
        return out.to_dict(orient="records") if not isinstance(evidence_rows, pd.DataFrame) else out

    support_key_columns = ["repo_full_name", "issue_id", "issue_number", "commit_author_contributor_key"]
    support_ok_df = support_ok_df[support_key_columns].drop_duplicates().copy()
    support_ok_df["__conservative_pre_issue_selected"] = 1

    out = out.merge(
        support_ok_df,
        on=support_key_columns,
        how="left",
    )

    out["selected_for_conservative_pre_issue_fallback"] = (
        pd.to_numeric(out["__conservative_pre_issue_selected"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    out = out.drop(columns=["__conservative_pre_issue_selected"], errors="ignore")

    exclude_if_any_high_conf_pr = bool(
        get_conservative_pre_issue_option(
            config,
            "conservative_pre_issue_exclude_if_any_high_conf_pr_evidence",
            False,
        )
    )

    if exclude_if_any_high_conf_pr:
        if "evidence_type" in out.columns and "selected_for_high_confidence_features" in out.columns:
            temp = out.copy()
            temp["evidence_type"] = temp["evidence_type"].apply(normalize_value)
            temp["selected_for_high_confidence_features"] = (
                pd.to_numeric(temp["selected_for_high_confidence_features"], errors="coerce").fillna(0).astype(int)
            )

            strict_pr_issue_keys = (
                temp[
                    (temp["selected_for_high_confidence_features"] == 1)
                    & (temp["evidence_type"].isin(["pr_merge", "pr_exact_commit", "pr_head"]))
                ][["repo_full_name", "issue_id", "issue_number"]]
                .drop_duplicates()
                .copy()
            )
            strict_pr_issue_keys["__drop_conservative_pre_issue"] = 1

            out = out.merge(
                strict_pr_issue_keys,
                on=["repo_full_name", "issue_id", "issue_number"],
                how="left",
            )
            drop_mask = pd.to_numeric(out["__drop_conservative_pre_issue"], errors="coerce").fillna(0).astype(int) == 1
            out.loc[drop_mask, "selected_for_conservative_pre_issue_fallback"] = 0
            out = out.drop(columns=["__drop_conservative_pre_issue"], errors="ignore")

    if isinstance(evidence_rows, pd.DataFrame):
        return out
    return out.to_dict(orient="records")

def prepare_issue_pr_links_frame(issue_pr_links_df, target_issue_ids, target_issue_numbers):
    if issue_pr_links_df.empty:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number", "pr_id", "pr_number", "link_type", "link_confidence"])
    df = issue_pr_links_df.copy()
    needed_columns = ["repo_full_name", "issue_id", "issue_number", "pr_id", "pr_number", "link_type", "link_confidence"]
    existing_columns = [column for column in needed_columns if column in df.columns]
    df = df[existing_columns].copy()
    if "issue_id" in df.columns:
        df["issue_id"] = df["issue_id"].astype(str)
    if "issue_number" in df.columns:
        df["issue_number"] = pd.to_numeric(df["issue_number"], errors="coerce")
    if "pr_id" in df.columns:
        df["pr_id"] = pd.to_numeric(df["pr_id"], errors="coerce")
    if "pr_number" in df.columns:
        df["pr_number"] = pd.to_numeric(df["pr_number"], errors="coerce")
    issue_id_mask = df["issue_id"].isin(target_issue_ids) if "issue_id" in df.columns else pd.Series(False, index=df.index)
    issue_number_mask = df["issue_number"].isin(target_issue_numbers) if "issue_number" in df.columns else pd.Series(False, index=df.index)
    df = df[issue_id_mask | issue_number_mask].copy()
    if "link_type" in df.columns:
        df["link_type"] = df["link_type"].apply(clean_text)
    if "link_confidence" in df.columns:
        df["link_confidence"] = df["link_confidence"].apply(clean_text)
    return df.reset_index(drop=True)

def prepare_pull_requests_frame(prs_df):
    if prs_df.empty:
        return pd.DataFrame(columns=["repo_full_name", "pr_id", "pr_number", "merge_commit_sha", "head_sha", "author_login", "created_at", "merged_at"])
    df = prs_df.copy()
    needed_columns = ["repo_full_name", "pr_id", "pr_number", "merge_commit_sha", "head_sha", "author_login", "created_at", "merged_at"]
    existing_columns = [column for column in needed_columns if column in df.columns]
    df = df[existing_columns].copy()
    if "pr_id" in df.columns:
        df["pr_id"] = pd.to_numeric(df["pr_id"], errors="coerce")
    if "pr_number" in df.columns:
        df["pr_number"] = pd.to_numeric(df["pr_number"], errors="coerce")
    if "merge_commit_sha" in df.columns:
        df["merge_commit_sha"] = df["merge_commit_sha"].apply(clean_text)
    if "head_sha" in df.columns:
        df["head_sha"] = df["head_sha"].apply(clean_text)
    if "created_at" in df.columns:
        df["created_at"] = safe_to_datetime(df["created_at"])
    if "merged_at" in df.columns:
        df["merged_at"] = safe_to_datetime(df["merged_at"])
    return df.reset_index(drop=True)

def prepare_pr_commit_links_frame(pr_commit_links_df):
    if pr_commit_links_df.empty:
        return pd.DataFrame(columns=["repo_full_name", "pr_id", "pr_number", "commit_sha"])
    df = pr_commit_links_df.copy()
    needed_columns = ["repo_full_name", "pr_id", "pr_number", "commit_sha"]
    existing_columns = [column for column in needed_columns if column in df.columns]
    df = df[existing_columns].copy()
    if "pr_id" in df.columns:
        df["pr_id"] = pd.to_numeric(df["pr_id"], errors="coerce")
    if "pr_number" in df.columns:
        df["pr_number"] = pd.to_numeric(df["pr_number"], errors="coerce")
    if "commit_sha" in df.columns:
        df["commit_sha"] = df["commit_sha"].apply(clean_text)
        df = df[df["commit_sha"].notna()].copy()
    return df.reset_index(drop=True)

def normalize_issue_file_links_frame(issue_file_links_df, target_issue_ids, target_issue_numbers):
    if issue_file_links_df.empty:
        return pd.DataFrame(
            columns=[
                "repo_full_name",
                "issue_id",
                "issue_number",
                "file_path",
                "source",
                "confidence_level",
            ]
        )

    df = issue_file_links_df.copy()
    needed_columns = [
        "repo_id",
        "repo_full_name",
        "issue_id",
        "issue_number",
        "file_path",
        "source",
        "confidence_level",
    ]
    existing_columns = [column for column in needed_columns if column in df.columns]
    df = df[existing_columns].copy()
    if "issue_number" in df.columns:
        df["issue_number"] = pd.to_numeric(df["issue_number"], errors="coerce")
    df["issue_id"] = df["issue_id"].astype(str) if "issue_id" in df.columns else None
    df["file_path"] = df["file_path"].apply(clean_text) if "file_path" in df.columns else None
    df = df[df["file_path"].notna()].copy()

    issue_id_mask = df["issue_id"].isin(target_issue_ids) if "issue_id" in df.columns else pd.Series(False, index=df.index)
    issue_number_mask = df["issue_number"].isin(target_issue_numbers) if "issue_number" in df.columns else pd.Series(False, index=df.index)
    df = df[issue_id_mask | issue_number_mask].copy()
    if df.empty:
        return df

    df["source"] = df["source"].apply(clean_text) if "source" in df.columns else None
    df["confidence_level"] = df["confidence_level"].apply(clean_text) if "confidence_level" in df.columns else None
    return df.reset_index(drop=True)

def prepare_commit_files_frame(commit_files_df):
    if commit_files_df.empty:
        return commit_files_df
    df = commit_files_df.copy()
    needed_columns = [
        "repo_full_name",
        "commit_sha",
        "file_path",
        "old_file_path",
        "additions",
        "deletions",
        "change_type",
    ]
    existing_columns = [column for column in needed_columns if column in df.columns]
    df = df[existing_columns].copy()
    if "file_path" in df.columns:
        df["file_path"] = df["file_path"].apply(clean_text)
    if "old_file_path" in df.columns:
        df["old_file_path"] = df["old_file_path"].apply(clean_text)
    if "additions" in df.columns:
        df["additions"] = pd.to_numeric(df["additions"], errors="coerce")
    else:
        df["additions"] = pd.NA
    if "deletions" in df.columns:
        df["deletions"] = pd.to_numeric(df["deletions"], errors="coerce")
    else:
        df["deletions"] = pd.NA
    return df.reset_index(drop=True)

def prepare_commits_frame(commits_df):
    if commits_df.empty:
        return pd.DataFrame(
            columns=[
                "repo_id",
                "repo_full_name",
                "commit_sha",
                "commit_timestamp",
                "author_email",
                "author_name",
                "author_login",
            ]
        )

    df = commits_df.copy()
    needed_columns = [
        "repo_id",
        "repo_full_name",
        "commit_sha",
        "commit_timestamp",
        "author_email",
        "author_name",
        "author_login",
    ]
    existing_columns = [column for column in needed_columns if column in df.columns]
    df = df[existing_columns].copy()

    if "commit_sha" in df.columns:
        df["commit_sha"] = df["commit_sha"].apply(clean_text)
        df = df[df["commit_sha"].notna()].copy()
    else:
        df["commit_sha"] = None

    if "commit_timestamp" in df.columns:
        df["commit_timestamp"] = safe_to_datetime(df["commit_timestamp"])
    else:
        df["commit_timestamp"] = pd.NaT

    if "author_email" in df.columns:
        df["author_email"] = df["author_email"].apply(clean_text)
    else:
        df["author_email"] = None

    if "author_name" in df.columns:
        df["author_name"] = df["author_name"].apply(clean_text)
    else:
        df["author_name"] = None

    if "author_login" in df.columns:
        df["author_login"] = df["author_login"].apply(clean_text)
    else:
        df["author_login"] = None

    return df.drop_duplicates(subset=["repo_full_name", "commit_sha"]).reset_index(drop=True)

def prepare_commits_resolved_frame(commits_resolved_df, exclude_bots=False):
    if commits_resolved_df.empty:
        return commits_resolved_df
    df = commits_resolved_df.copy()
    needed_columns = [
        "repo_id",
        "repo_full_name",
        "commit_sha",
        "commit_timestamp",
        "commit_author_contributor_key",
        "author_email",
        "author_name",
        "author_login",
        "bot_flag",
        "commit_author_bot_flag",
    ]
    existing_columns = [column for column in needed_columns if column in df.columns]
    df = df[existing_columns].copy()
    if "commit_timestamp" in df.columns:
        df["commit_timestamp"] = safe_to_datetime(df["commit_timestamp"])
    else:
        df["commit_timestamp"] = pd.NaT

    bot_column = None
    if "commit_author_bot_flag" in df.columns:
        bot_column = "commit_author_bot_flag"
    elif "bot_flag" in df.columns:
        bot_column = "bot_flag"

    if exclude_bots and bot_column:
        df = df[df[bot_column] != True].copy()

    return df.reset_index(drop=True)

def build_issue_file_summary(issue_links_df, high_conf_levels):
    if issue_links_df.empty:
        return {}, {}

    issue_summary = {}
    grouped = issue_links_df.groupby(["repo_full_name", "issue_id", "issue_number", "file_path"], dropna=False)
    for group_key, group in grouped:
        repo_full_name, issue_id, issue_number, file_path = group_key
        issue_key = (repo_full_name, clean_text(issue_id), int(issue_number) if pd.notna(issue_number) else None)
        payload = issue_summary.setdefault(
            issue_key,
            {
                "file_rows": [],
                "all_link_row_count": 0,
                "all_file_paths": set(),
                "high_conf_file_paths": set(),
            },
        )

        payload["all_link_row_count"] += len(group)
        payload["all_file_paths"].add(file_path)
        sources = sorted({clean_text(value) for value in group["source"].tolist() if clean_text(value)})
        confidence_values = [clean_text(value) for value in group["confidence_level"].tolist() if clean_text(value)]
        best_confidence = None
        if confidence_values:
            best_confidence = sorted(confidence_values, key=lambda value: confidence_rank(value), reverse=True)[0]
        is_high_conf = 1 if normalize_value(best_confidence) in high_conf_levels else 0
        if is_high_conf:
            payload["high_conf_file_paths"].add(file_path)
        payload["file_rows"].append(
            {
                "file_path": file_path,
                "issue_file_link_source": json.dumps(sources),
                "issue_file_link_confidence": best_confidence,
                "issue_file_link_is_high_confidence": is_high_conf,
            }
        )

    issue_level_lookup = {}
    for issue_key, payload in issue_summary.items():
        issue_level_lookup[issue_key] = {
            "all_link_row_count": int(payload["all_link_row_count"]),
            "linked_file_count_all": int(len(payload["all_file_paths"])),
            "linked_file_count_high_confidence": int(len(payload["high_conf_file_paths"])),
            "file_rows": sorted(payload["file_rows"], key=lambda row: row["file_path"]),
        }
    return issue_level_lookup, issue_summary

def build_commit_file_index(commit_files_df):
    path_index = {}
    if commit_files_df.empty:
        return path_index
    for row in commit_files_df.to_dict(orient="records"):
        for column_name in ["file_path", "old_file_path"]:
            file_path = clean_text(row.get(column_name))
            if not file_path:
                continue
            path_index.setdefault(file_path, []).append(row)
    return path_index

def build_commits_lookup(commits_df, commits_resolved_df):
    commits_lookup = {}

    if commits_df is not None and not commits_df.empty:
        for row in commits_df.to_dict(orient="records"):
            commit_sha = clean_text(row.get("commit_sha"))
            if not commit_sha:
                continue

            commits_lookup[commit_sha] = {
                "repo_id": row.get("repo_id"),
                "repo_full_name": row.get("repo_full_name"),
                "commit_sha": commit_sha,
                "commit_timestamp": row.get("commit_timestamp"),
                "commit_author_contributor_key": None,
                "author_email": clean_text(row.get("author_email")),
                "author_name": clean_text(row.get("author_name")),
                "author_login": clean_text(row.get("author_login")),
            }

    if commits_resolved_df is not None and not commits_resolved_df.empty:
        for row in commits_resolved_df.to_dict(orient="records"):
            commit_sha = clean_text(row.get("commit_sha"))
            if not commit_sha:
                continue

            existing = commits_lookup.get(commit_sha)
            if existing is None:
                existing = {
                    "repo_id": row.get("repo_id"),
                    "repo_full_name": row.get("repo_full_name"),
                    "commit_sha": commit_sha,
                    "commit_timestamp": row.get("commit_timestamp"),
                    "commit_author_contributor_key": None,
                    "author_email": clean_text(row.get("author_email")),
                    "author_name": clean_text(row.get("author_name")),
                    "author_login": clean_text(row.get("author_login")),
                }

            resolved_key = clean_text(row.get("commit_author_contributor_key"))
            if resolved_key:
                existing["commit_author_contributor_key"] = resolved_key

            if existing.get("repo_id") is None and row.get("repo_id") is not None:
                existing["repo_id"] = row.get("repo_id")

            if pd.isna(existing.get("commit_timestamp")) and not pd.isna(row.get("commit_timestamp")):
                existing["commit_timestamp"] = row.get("commit_timestamp")

            if not clean_text(existing.get("author_email")) and clean_text(row.get("author_email")):
                existing["author_email"] = clean_text(row.get("author_email"))

            if not clean_text(existing.get("author_name")) and clean_text(row.get("author_name")):
                existing["author_name"] = clean_text(row.get("author_name"))

            if not clean_text(existing.get("author_login")) and clean_text(row.get("author_login")):
                existing["author_login"] = clean_text(row.get("author_login"))

            commits_lookup[commit_sha] = existing

    return commits_lookup

def build_pr_lookup_maps(issue_pr_links_df, pull_requests_df, pr_commit_links_df, commits_lookup):
    issue_to_pr_ids = {}
    issue_to_pr_numbers = {}
    for row in issue_pr_links_df.to_dict(orient="records"):
        issue_id = clean_text(row.get("issue_id"))
        issue_number = row.get("issue_number")
        pr_id = row.get("pr_id")
        pr_number = row.get("pr_number")
        if issue_id and pd.notna(pr_id):
            issue_to_pr_ids.setdefault(issue_id, set()).add(int(pr_id))
        if issue_id and pd.notna(pr_number):
            issue_to_pr_numbers.setdefault(issue_id, set()).add(int(pr_number))
        if pd.notna(issue_number) and pd.notna(pr_id):
            issue_to_pr_ids.setdefault(int(issue_number), set()).add(int(pr_id))
        if pd.notna(issue_number) and pd.notna(pr_number):
            issue_to_pr_numbers.setdefault(int(issue_number), set()).add(int(pr_number))

    pr_rows_by_pr_id = {}
    pr_rows_by_pr_number = {}
    for row in pull_requests_df.to_dict(orient="records"):
        pr_id = row.get("pr_id")
        pr_number = row.get("pr_number")
        if pd.notna(pr_id):
            pr_rows_by_pr_id[int(pr_id)] = row
        if pd.notna(pr_number):
            pr_rows_by_pr_number[int(pr_number)] = row

    exact_commit_shas_by_pr_id = {}
    exact_commit_shas_by_pr_number = {}
    for row in pr_commit_links_df.to_dict(orient="records"):
        commit_sha = clean_text(row.get("commit_sha"))
        if not commit_sha:
            continue
        pr_id = row.get("pr_id")
        pr_number = row.get("pr_number")
        if pd.notna(pr_id):
            exact_commit_shas_by_pr_id.setdefault(int(pr_id), set()).add(commit_sha)
        if pd.notna(pr_number):
            exact_commit_shas_by_pr_number.setdefault(int(pr_number), set()).add(commit_sha)

    return {
        "issue_to_pr_ids": issue_to_pr_ids,
        "issue_to_pr_numbers": issue_to_pr_numbers,
        "pr_rows_by_pr_id": pr_rows_by_pr_id,
        "pr_rows_by_pr_number": pr_rows_by_pr_number,
        "exact_commit_shas_by_pr_id": exact_commit_shas_by_pr_id,
        "exact_commit_shas_by_pr_number": exact_commit_shas_by_pr_number,
        "known_commit_rows_by_sha": commits_lookup,
    }

def calculate_churn_weight(additions, deletions):
    has_add = additions is not None and not pd.isna(additions)
    has_del = deletions is not None and not pd.isna(deletions)
    if not has_add and not has_del:
        return None
    add_value = 0.0 if not has_add else abs(float(additions))
    del_value = 0.0 if not has_del else abs(float(deletions))
    return add_value + del_value

def build_generic_evidence_row(issue_row, commit_row, evidence_type, pr_row=None, file_row=None, commit_file_row=None):
    commit_timestamp = commit_row.get("commit_timestamp") if commit_row else pd.NaT
    additions = None
    deletions = None
    change_type = None
    file_path = None
    issue_file_link_source = None
    issue_file_link_confidence = None
    issue_file_link_is_high_confidence = None

    if commit_file_row:
        additions = pd.to_numeric(commit_file_row.get("additions"), errors="coerce")
        deletions = pd.to_numeric(commit_file_row.get("deletions"), errors="coerce")
        change_type = commit_file_row.get("change_type")
        file_path = clean_text(commit_file_row.get("file_path")) or clean_text(commit_file_row.get("old_file_path"))

    if file_row:
        file_path = clean_text(file_row.get("file_path")) or file_path
        issue_file_link_source = file_row.get("issue_file_link_source")
        issue_file_link_confidence = file_row.get("issue_file_link_confidence")
        issue_file_link_is_high_confidence = file_row.get("issue_file_link_is_high_confidence")

    churn_weight = calculate_churn_weight(additions, deletions)
    contributor_key = clean_text(commit_row.get("commit_author_contributor_key")) if commit_row else None

    return {
        "repo_id": issue_row.get("repo_id"),
        "repo_full_name": issue_row.get("repo_full_name"),
        "issue_id": issue_row.get("issue_id"),
        "issue_number": issue_row.get("issue_number"),
        "analysis_set": issue_row.get("analysis_set"),
        "issue_created_at": issue_row.get("created_at"),
        "pr_id": None if pr_row is None or pd.isna(pr_row.get("pr_id")) else int(pr_row.get("pr_id")),
        "pr_number": None if pr_row is None or pd.isna(pr_row.get("pr_number")) else int(pr_row.get("pr_number")),
        "file_path": file_path,
        "issue_file_link_source": issue_file_link_source,
        "issue_file_link_confidence": issue_file_link_confidence,
        "issue_file_link_is_high_confidence": issue_file_link_is_high_confidence,
        "commit_sha": clean_text(commit_row.get("commit_sha")) if commit_row else None,
        "commit_timestamp": commit_timestamp,
        "commit_author_contributor_key": contributor_key,
        "additions": None if additions is None or pd.isna(additions) else float(additions),
        "deletions": None if deletions is None or pd.isna(deletions) else float(deletions),
        "change_type": change_type,
        "ownership_weight_churn": churn_weight if churn_weight is not None else 1.0,
        "ownership_weight_commit": 1.0,
        "evidence_type": evidence_type,
        "evidence_confidence": PR_EVIDENCE_CONFIDENCE.get(evidence_type, "low"),
        "ownership_time_bucket": None,
        "evidence_selected_for_features": 0,
    }

def classify_evidence_time_bucket(issue_created_at, commit_timestamp):
    if pd.isna(issue_created_at) or pd.isna(commit_timestamp):
        return "unknown"
    if commit_timestamp <= issue_created_at:
        return "pre_issue"
    return "post_issue"

def resolve_linked_pr_rows(issue_row, pr_maps):
    issue_id = clean_text(issue_row.get("issue_id"))
    issue_number = issue_row.get("issue_number")
    pr_rows = []
    seen = set()
    for key in [issue_id, int(issue_number) if pd.notna(issue_number) else None]:
        if key is None:
            continue
        for pr_id in pr_maps["issue_to_pr_ids"].get(key, set()):
            row = pr_maps["pr_rows_by_pr_id"].get(pr_id)
            if row and ("id", pr_id) not in seen:
                seen.add(("id", pr_id)); pr_rows.append(row)
        for pr_number in pr_maps["issue_to_pr_numbers"].get(key, set()):
            row = pr_maps["pr_rows_by_pr_number"].get(pr_number)
            if row and ("num", pr_number) not in seen:
                seen.add(("num", pr_number)); pr_rows.append(row)
    return pr_rows

def build_issue_pr_merge_evidence(issue_row, pr_maps):
    evidence_rows = []
    seen = set()

    for pr_row in resolve_linked_pr_rows(issue_row, pr_maps):
        merge_commit_sha = clean_text(pr_row.get("merge_commit_sha"))
        if not merge_commit_sha or merge_commit_sha in seen:
            continue

        commit_row = pr_maps["known_commit_rows_by_sha"].get(merge_commit_sha)
        if not commit_row:
            continue

        seen.add(merge_commit_sha)
        evidence_row = build_generic_evidence_row(issue_row, commit_row, "pr_merge", pr_row=pr_row)
        evidence_row["ownership_time_bucket"] = classify_evidence_time_bucket(
            issue_row.get("created_at"),
            evidence_row.get("commit_timestamp"),
        )
        evidence_rows.append(evidence_row)

    return evidence_rows

def build_issue_pr_exact_commit_evidence(issue_row, pr_maps):
    evidence_rows = []
    seen = set()

    for pr_row in resolve_linked_pr_rows(issue_row, pr_maps):
        pr_id = pr_row.get("pr_id")
        pr_number = pr_row.get("pr_number")

        sha_set = set()
        if pd.notna(pr_id):
            sha_set |= pr_maps["exact_commit_shas_by_pr_id"].get(int(pr_id), set())
        if pd.notna(pr_number):
            sha_set |= pr_maps["exact_commit_shas_by_pr_number"].get(int(pr_number), set())

        for commit_sha in sorted(sha_set):
            if commit_sha in seen:
                continue

            commit_row = pr_maps["known_commit_rows_by_sha"].get(commit_sha)
            if not commit_row:
                continue

            seen.add(commit_sha)
            evidence_row = build_generic_evidence_row(issue_row, commit_row, "pr_exact_commit", pr_row=pr_row)
            evidence_row["ownership_time_bucket"] = classify_evidence_time_bucket(
                issue_row.get("created_at"),
                evidence_row.get("commit_timestamp"),
            )
            evidence_rows.append(evidence_row)

    return evidence_rows

def build_issue_pr_head_evidence(issue_row, pr_maps):
    evidence_rows = []
    seen = set()

    for pr_row in resolve_linked_pr_rows(issue_row, pr_maps):
        head_sha = clean_text(pr_row.get("head_sha"))
        if not head_sha or head_sha in seen:
            continue

        commit_row = pr_maps["known_commit_rows_by_sha"].get(head_sha)
        if not commit_row:
            continue

        seen.add(head_sha)
        evidence_row = build_generic_evidence_row(issue_row, commit_row, "pr_head", pr_row=pr_row)
        evidence_row["ownership_time_bucket"] = classify_evidence_time_bucket(
            issue_row.get("created_at"),
            evidence_row.get("commit_timestamp"),
        )
        evidence_rows.append(evidence_row)

    return evidence_rows

def calculate_churn_weight(additions, deletions):
    has_add = additions is not None and not pd.isna(additions)
    has_del = deletions is not None and not pd.isna(deletions)
    if not has_add and not has_del:
        return None
    add_value = 0.0 if not has_add else abs(float(additions))
    del_value = 0.0 if not has_del else abs(float(deletions))
    return add_value + del_value

def build_issue_file_commit_evidence(issue_row, issue_file_payload, commit_file_index, commits_lookup):
    evidence_rows = []
    seen_issue_file_commit = set()

    for file_row in issue_file_payload.get("file_rows", []):
        file_path = file_row.get("file_path")
        if not file_path:
            continue

        commit_file_rows = commit_file_index.get(file_path, [])
        for commit_file_row in commit_file_rows:
            commit_sha = clean_text(commit_file_row.get("commit_sha"))
            if not commit_sha:
                continue

            dedupe_key = (file_path, commit_sha)
            if dedupe_key in seen_issue_file_commit:
                continue

            commit_row = commits_lookup.get(commit_sha)
            if not commit_row:
                continue

            seen_issue_file_commit.add(dedupe_key)
            evidence_row = build_generic_evidence_row(
                issue_row,
                commit_row,
                "file_fallback",
                file_row=file_row,
                commit_file_row=commit_file_row,
            )
            evidence_row["ownership_time_bucket"] = classify_evidence_time_bucket(
                issue_row.get("created_at"),
                evidence_row.get("commit_timestamp"),
            )
            evidence_rows.append(evidence_row)

    return evidence_rows

def filter_fallback_rows_for_selection(config, fallback_rows):
    if not fallback_rows:
        return []

    allowed_sources = {
        normalize_value(value)
        for value in list(get_ownership_option(config, "allow_fallback_sources", ["pr_commit_chain"]) or [])
        if normalize_value(value)
    }
    allowed_conf_levels = {
        normalize_value(value)
        for value in list(get_ownership_option(config, "allow_fallback_confidence_levels", ["high"]) or [])
        if normalize_value(value)
    }

    filtered_rows = []
    for row in fallback_rows:
        source_blob = clean_text(row.get("issue_file_link_source"))
        confidence_value = normalize_value(row.get("issue_file_link_confidence"))
        source_values = set()
        if source_blob:
            try:
                parsed = json.loads(source_blob)
                if isinstance(parsed, list):
                    source_values = {normalize_value(value) for value in parsed if normalize_value(value)}
                else:
                    normalized = normalize_value(source_blob)
                    if normalized:
                        source_values = {normalized}
            except Exception:
                normalized = normalize_value(source_blob)
                if normalized:
                    source_values = {normalized}

        source_ok = bool(source_values.intersection(allowed_sources)) if allowed_sources else True
        confidence_ok = confidence_value in allowed_conf_levels if allowed_conf_levels else True
        if source_ok and confidence_ok:
            filtered_rows.append(row)

    return filtered_rows

def dedupe_evidence_rows(rows):
    deduped = []
    seen = set()
    for row in rows:
        key = (
            row.get("repo_full_name"),
            clean_text(row.get("issue_id")),
            clean_text(row.get("commit_sha")),
            row.get("evidence_type"),
            clean_text(row.get("file_path")),
            row.get("pr_id"),
            row.get("pr_number"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped

def combine_issue_ownership_evidence(config, repo_policy, pr_merge_rows, pr_exact_rows, pr_head_rows, fallback_rows):
    fallback_rows = filter_fallback_rows_for_selection(config, fallback_rows)
    all_rows = dedupe_evidence_rows(pr_merge_rows + pr_exact_rows + pr_head_rows + fallback_rows)

    high_conf_selected_rows = []
    if repo_policy == "exact_plus_merge":
        high_conf_selected_rows = dedupe_evidence_rows(pr_merge_rows + pr_exact_rows)
    elif repo_policy == "merge_plus_head":
        high_conf_selected_rows = dedupe_evidence_rows(pr_merge_rows + pr_head_rows)
    elif repo_policy == "merge_first":
        high_conf_selected_rows = dedupe_evidence_rows(pr_merge_rows)
        if not high_conf_selected_rows:
            high_conf_selected_rows = dedupe_evidence_rows(pr_exact_rows)
    elif repo_policy == "exact_first":
        high_conf_selected_rows = dedupe_evidence_rows(pr_exact_rows)
        if not high_conf_selected_rows:
            high_conf_selected_rows = dedupe_evidence_rows(pr_merge_rows)
    else:
        high_conf_selected_rows = dedupe_evidence_rows(pr_merge_rows + pr_exact_rows + pr_head_rows)

    if not high_conf_selected_rows and bool(get_ownership_option(config, "allow_file_fallback_when_no_pr_evidence", True)):
        high_conf_selected_rows = dedupe_evidence_rows(fallback_rows)

    high_conf_keys = {
        (
            row.get("repo_full_name"),
            clean_text(row.get("issue_id")),
            clean_text(row.get("commit_sha")),
            row.get("evidence_type"),
            clean_text(row.get("file_path")),
            row.get("pr_id"),
            row.get("pr_number"),
        )
        for row in high_conf_selected_rows
    }

    all_rows = annotate_conservative_pre_issue_selection(all_rows, config)

    conservative_selected_rows = []
    conservative_keys = set()
    for row in all_rows:
        row_key = (
            row.get("repo_full_name"),
            clean_text(row.get("issue_id")),
            clean_text(row.get("commit_sha")),
            row.get("evidence_type"),
            clean_text(row.get("file_path")),
            row.get("pr_id"),
            row.get("pr_number"),
        )
        if int(pd.to_numeric(row.get("selected_for_conservative_pre_issue_fallback"), errors="coerce") or 0) == 1:
            conservative_selected_rows.append(row)
            conservative_keys.add(row_key)

    counts_toward_usable_any = bool(
        get_conservative_pre_issue_option(
            config,
            "conservative_pre_issue_counts_toward_usable_any",
            True,
        )
    )

    if counts_toward_usable_any:
        any_selected_keys = high_conf_keys.union(conservative_keys)
    else:
        any_selected_keys = set(high_conf_keys)

    for row in all_rows:
        row_key = (
            row.get("repo_full_name"),
            clean_text(row.get("issue_id")),
            clean_text(row.get("commit_sha")),
            row.get("evidence_type"),
            clean_text(row.get("file_path")),
            row.get("pr_id"),
            row.get("pr_number"),
        )
        row["selected_for_high_confidence_features"] = 1 if row_key in high_conf_keys else 0
        conservative_value = pd.to_numeric(
            row.get("selected_for_conservative_pre_issue_fallback"),
            errors="coerce",
        )
        row["selected_for_conservative_pre_issue_fallback"] = (
            int(conservative_value) if pd.notna(conservative_value) else 0
        )
        row["selected_for_any_features"] = 1 if row_key in any_selected_keys else 0
        row["evidence_selected_for_features"] = row["selected_for_any_features"]

    selected_high_conf_pre_issue_rows = [
        row
        for row in all_rows
        if row["selected_for_high_confidence_features"] == 1 and clean_text(row.get("ownership_time_bucket")) == "pre_issue"
    ]
    selected_high_conf_post_issue_rows = [
        row
        for row in all_rows
        if row["selected_for_high_confidence_features"] == 1 and clean_text(row.get("ownership_time_bucket")) == "post_issue"
    ]
    selected_any_rows = [row for row in all_rows if row["selected_for_any_features"] == 1]
    selected_any_pre_issue_rows = [
        row
        for row in all_rows
        if row["selected_for_any_features"] == 1 and clean_text(row.get("ownership_time_bucket")) == "pre_issue"
    ]
    selected_any_post_issue_rows = [
        row
        for row in all_rows
        if row["selected_for_any_features"] == 1 and clean_text(row.get("ownership_time_bucket")) == "post_issue"
    ]

    return {
        "all_evidence_rows": all_rows,
        "selected_high_confidence_rows": high_conf_selected_rows,
        "selected_high_confidence_pre_issue_rows": selected_high_conf_pre_issue_rows,
        "selected_high_confidence_post_issue_rows": selected_high_conf_post_issue_rows,
        "selected_conservative_pre_issue_rows": conservative_selected_rows,
        "selected_any_rows": selected_any_rows,
        "selected_any_pre_issue_rows": selected_any_pre_issue_rows,
        "selected_any_post_issue_rows": selected_any_post_issue_rows,
    }

def build_discussion_summary(issue_comments_df):
    if issue_comments_df.empty:
        return {
            "discussion_participant_count": 0,
            "participant_keys": set(),
            "total_comments_with_resolved_author": 0,
            "comment_counts_by_participant": {},
        }

    participant_keys = set()
    comment_counts_by_participant = {}
    total_comments_with_resolved_author = 0
    for row in issue_comments_df.to_dict(orient="records"):
        commenter_key = clean_text(row.get("comment_author_contributor_key")) or clean_text(row.get("author_login"))
        if not commenter_key:
            continue
        participant_keys.add(commenter_key)
        total_comments_with_resolved_author += 1
        comment_counts_by_participant[commenter_key] = comment_counts_by_participant.get(commenter_key, 0) + 1
    return {
        "discussion_participant_count": len(participant_keys),
        "participant_keys": participant_keys,
        "total_comments_with_resolved_author": total_comments_with_resolved_author,
        "comment_counts_by_participant": comment_counts_by_participant,
    }

def shannon_entropy(shares):
    valid = [float(value) for value in shares if value is not None and not pd.isna(value) and float(value) > 0.0]
    if not valid:
        return None
    return -sum(value * math.log(value) for value in valid)

def compute_contributor_summary(issue_row, evidence_rows):
    contributor_map = {}
    for row in evidence_rows:
        contributor_key = clean_text(row.get("commit_author_contributor_key"))
        if not contributor_key:
            continue
        payload = contributor_map.setdefault(
            contributor_key,
            {
                "ownership_churn_sum": 0.0,
                "ownership_commit_count": 0.0,
                "linked_file_paths": set(),
                "first_touch_at": pd.NaT,
                "last_touch_at": pd.NaT,
                "evidence_types": set(),
            },
        )
        churn_weight = row.get("ownership_weight_churn")
        if churn_weight is not None and not pd.isna(churn_weight):
            payload["ownership_churn_sum"] += float(churn_weight)
        payload["ownership_commit_count"] += 1.0
        file_path = clean_text(row.get("file_path"))
        if file_path:
            payload["linked_file_paths"].add(file_path)
        payload["evidence_types"].add(clean_text(row.get("evidence_type")))
        commit_timestamp = row.get("commit_timestamp")
        if pd.notna(commit_timestamp):
            if pd.isna(payload["first_touch_at"]) or commit_timestamp < payload["first_touch_at"]:
                payload["first_touch_at"] = commit_timestamp
            if pd.isna(payload["last_touch_at"]) or commit_timestamp > payload["last_touch_at"]:
                payload["last_touch_at"] = commit_timestamp

    if not contributor_map:
        return []

    total_churn = sum(payload["ownership_churn_sum"] for payload in contributor_map.values())
    total_commit_count = sum(payload["ownership_commit_count"] for payload in contributor_map.values())
    summary_rows = []
    issue_created_at = issue_row.get("created_at")

    for contributor_key, payload in contributor_map.items():
        last_touch = payload.get("last_touch_at")
        days_since_last_touch = None
        if pd.notna(issue_created_at) and pd.notna(last_touch):
            delta = issue_created_at - last_touch
            days_since_last_touch = float(delta.total_seconds()) / 86400.0
        summary_rows.append(
            {
                "commit_author_contributor_key": contributor_key,
                "ownership_churn_sum": float(payload["ownership_churn_sum"]),
                "ownership_commit_count": float(payload["ownership_commit_count"]),
                "ownership_linked_file_count": int(len(payload["linked_file_paths"])),
                "first_touch_at": payload.get("first_touch_at"),
                "last_touch_at": last_touch,
                "days_since_last_touch_before_issue": days_since_last_touch,
                "ownership_share_churn": None if total_churn <= 0 else float(payload["ownership_churn_sum"]) / float(total_churn),
                "ownership_share_commit": None if total_commit_count <= 0 else float(payload["ownership_commit_count"]) / float(total_commit_count),
                "contributor_evidence_types": json.dumps(sorted([v for v in payload["evidence_types"] if v])),
            }
        )

    summary_rows = sorted(
        summary_rows,
        key=lambda row: (
            row.get("ownership_share_churn") if row.get("ownership_share_churn") is not None else -1.0,
            row.get("ownership_share_commit") if row.get("ownership_share_commit") is not None else -1.0,
            row.get("commit_author_contributor_key") or "",
        ),
        reverse=True,
    )
    return summary_rows

def summarize_contributor_metrics(contributor_summary, discussion_summary):
    shares_churn = [row.get("ownership_share_churn") for row in contributor_summary if row.get("ownership_share_churn") is not None]
    shares_commit = [row.get("ownership_share_commit") for row in contributor_summary if row.get("ownership_share_commit") is not None]
    sorted_churn = sorted(shares_churn, reverse=True)
    sorted_commit = sorted(shares_commit, reverse=True)

    owner_keys = {
        clean_text(row.get("commit_author_contributor_key"))
        for row in contributor_summary
        if clean_text(row.get("commit_author_contributor_key"))
    }
    participant_keys = set(discussion_summary.get("participant_keys", set()))
    overlap_keys = owner_keys.intersection(participant_keys)

    return {
        "owner_count": int(len(contributor_summary)),
        "top_share_churn": sorted_churn[0] if sorted_churn else None,
        "top_share_commit": sorted_commit[0] if sorted_commit else None,
        "entropy_churn": shannon_entropy(shares_churn),
        "entropy_commit": shannon_entropy(shares_commit),
        "discussion_overlap_count": int(len(overlap_keys)),
        "discussion_overlap_fraction": safe_divide(len(overlap_keys), len(contributor_summary), default_value=None) if contributor_summary else None,
        "owner_comment_presence_flag": 1 if len(overlap_keys) > 0 else 0,
    }

def resolve_coverage_flag(issue_row, issue_file_payload, evidence_rows, contributor_summary, sparse_thresholds):
    if pd.isna(issue_row.get("created_at")):
        return "missing_issue_created_at"
    linked_file_count_all = issue_file_payload.get("linked_file_count_all", 0)
    if linked_file_count_all <= 0:
        return "no_file_links"
    if len(evidence_rows) <= 0:
        return "no_commit_matches"
    if len(contributor_summary) <= 0:
        return "no_resolved_commit_authors"

    contributor_count = len(contributor_summary)
    resolved_commit_rows = len([row for row in evidence_rows if clean_text(row.get("commit_author_contributor_key"))])
    if (
        linked_file_count_all < sparse_thresholds["min_linked_files"]
        or resolved_commit_rows < sparse_thresholds["min_resolved_commit_rows"]
        or contributor_count < sparse_thresholds["min_contributors"]
    ):
        return "sparse_evidence"
    return "ok"

def summarize_issue_evidence_types(selected_rows):
    types = {clean_text(row.get("evidence_type")) for row in selected_rows if clean_text(row.get("evidence_type"))}
    if "pr_merge" in types:
        return "pr_merge"
    if "pr_exact_commit" in types:
        return "pr_exact_commit"
    if "pr_head" in types:
        return "pr_head"
    if "file_fallback" in types:
        return "file_fallback"
    return None

def build_issue_feature_row(issue_row, issue_file_payload, all_evidence_rows, selected_high_confidence_rows, selected_high_confidence_pre_issue_rows, selected_high_confidence_post_issue_rows, selected_conservative_pre_issue_rows, selected_any_rows, selected_any_pre_issue_rows, selected_any_post_issue_rows, contributor_summary_high_confidence, contributor_summary_high_confidence_pre, contributor_summary_high_confidence_post, contributor_summary_any, contributor_summary_any_pre, contributor_summary_any_post, discussion_summary, sparse_thresholds, repo_policy):
    coverage_flag = resolve_coverage_flag(
        issue_row,
        issue_file_payload,
        selected_any_rows,
        contributor_summary_any,
        sparse_thresholds,
    )

    all_linked_file_count = int(issue_file_payload.get("linked_file_count_all", 0))
    high_conf_linked_file_count = int(issue_file_payload.get("linked_file_count_high_confidence", 0))
    raw_link_row_count = int(issue_file_payload.get("all_link_row_count", 0))

    commit_evidence_row_count = int(len(selected_any_rows))
    resolved_evidence_rows = [row for row in selected_any_rows if clean_text(row.get("commit_author_contributor_key"))]
    resolved_commit_evidence_row_count = int(len(resolved_evidence_rows))
    contributor_count_any = int(len(contributor_summary_any))
    contributor_count_high_conf = int(len(contributor_summary_high_confidence))

    shares_churn = [row.get("ownership_share_churn") for row in contributor_summary_any if row.get("ownership_share_churn") is not None]
    shares_commit = [row.get("ownership_share_commit") for row in contributor_summary_any if row.get("ownership_share_commit") is not None]
    sorted_churn = sorted(shares_churn, reverse=True)
    sorted_commit = sorted(shares_commit, reverse=True)

    top_owner_row = contributor_summary_any[0] if contributor_summary_any else None
    top_owner_key = clean_text(top_owner_row.get("commit_author_contributor_key")) if top_owner_row else None
    owner_keys = {
        clean_text(row.get("commit_author_contributor_key"))
        for row in contributor_summary_any
        if clean_text(row.get("commit_author_contributor_key"))
    }
    participant_keys = set(discussion_summary.get("participant_keys", set()))
    overlap_keys = owner_keys.intersection(participant_keys)

    issue_author_key = clean_text(issue_row.get("issue_author_contributor_key")) or clean_text(issue_row.get("author_login"))
    issue_author_row = None
    if issue_author_key:
        for row in contributor_summary_any:
            if clean_text(row.get("commit_author_contributor_key")) == issue_author_key:
                issue_author_row = row
                break

    owner_comment_count = sum(discussion_summary.get("comment_counts_by_participant", {}).get(key, 0) for key in overlap_keys)
    total_comments_with_resolved_author = int(discussion_summary.get("total_comments_with_resolved_author", 0))
    top_owner_comment_count = discussion_summary.get("comment_counts_by_participant", {}).get(top_owner_key, 0) if top_owner_key else 0

    row = {
        "repo_id": issue_row.get("repo_id"),
        "repo_full_name": issue_row.get("repo_full_name"),
        "issue_id": issue_row.get("issue_id"),
        "issue_number": issue_row.get("issue_number"),
        "analysis_set": issue_row.get("analysis_set"),
        "ownership_coverage_flag": coverage_flag,
        "ownership_policy_used": repo_policy,
        "ownership_has_file_links": 1 if all_linked_file_count > 0 else 0,
        "ownership_link_row_count": raw_link_row_count,
        "ownership_linked_file_count_all": all_linked_file_count,
        "ownership_linked_file_count_high_confidence": high_conf_linked_file_count,
        "ownership_commit_evidence_row_count": commit_evidence_row_count,
        "ownership_resolved_commit_evidence_row_count": resolved_commit_evidence_row_count,
        "ownership_has_resolved_commit_authors": 1 if resolved_commit_evidence_row_count > 0 else 0,
        "ownership_contributor_count": contributor_count_any,
        "ownership_high_confidence_contributor_count": contributor_count_high_conf,
        "ownership_pre_issue_contributor_count": int(len(contributor_summary_any_pre)),
        "ownership_post_issue_contributor_count": int(len(contributor_summary_any_post)),
        "ownership_pre_issue_high_confidence_contributor_count": int(len(contributor_summary_high_confidence_pre)),
        "ownership_post_issue_high_confidence_contributor_count": int(len(contributor_summary_high_confidence_post)),
        "ownership_pre_issue_conservative_fallback_contributor_count": int(
            len(
                {
                    clean_text(r.get("commit_author_contributor_key"))
                    for r in selected_conservative_pre_issue_rows
                    if clean_text(r.get("commit_author_contributor_key"))
                }
            )
        ),
        "ownership_pre_issue_conservative_fallback_commit_count": int(len(selected_conservative_pre_issue_rows)),
        "ownership_pre_issue_conservative_fallback_file_count": int(
            len(
                {
                    clean_text(r.get("file_path"))
                    for r in selected_conservative_pre_issue_rows
                    if clean_text(r.get("file_path"))
                }
            )
        ),
        "ownership_usable_high_confidence": 1 if len(selected_high_confidence_rows) > 0 and len(contributor_summary_high_confidence) > 0 else 0,
        "ownership_usable_any": 1 if len(selected_any_rows) > 0 and len(contributor_summary_any) > 0 else 0,
        "ownership_usable_any_including_conservative_pre_issue": 1 if len(selected_any_rows) > 0 and len(contributor_summary_any) > 0 else 0,
        "ownership_usable_pre_issue_conservative_fallback": 1 if len(selected_conservative_pre_issue_rows) > 0 else 0,
        "ownership_has_selected_conservative_pre_issue_fallback": 1 if len(selected_conservative_pre_issue_rows) > 0 else 0,
        "ownership_selected_evidence_type": summarize_issue_evidence_types(selected_any_rows),
        "ownership_selected_high_confidence_evidence_type": summarize_issue_evidence_types(selected_high_confidence_rows),
        "ownership_selected_any_evidence_type": summarize_issue_evidence_types(selected_any_rows),
        "ownership_selected_evidence_types": json.dumps(sorted({clean_text(row.get("evidence_type")) for row in selected_any_rows if clean_text(row.get("evidence_type"))})),
        "ownership_selected_high_confidence_evidence_types": json.dumps(sorted({clean_text(row.get("evidence_type")) for row in selected_high_confidence_rows if clean_text(row.get("evidence_type"))})),
        "ownership_selected_any_evidence_types": json.dumps(sorted({clean_text(row.get("evidence_type")) for row in selected_any_rows if clean_text(row.get("evidence_type"))})),
        "ownership_pr_merge_commit_count": int(sum(1 for row in selected_any_rows if clean_text(row.get("evidence_type")) == "pr_merge")),
        "ownership_pr_exact_commit_count": int(sum(1 for row in selected_any_rows if clean_text(row.get("evidence_type")) == "pr_exact_commit")),
        "ownership_pr_head_commit_count": int(sum(1 for row in selected_any_rows if clean_text(row.get("evidence_type")) == "pr_head")),
        "ownership_file_fallback_commit_count": int(sum(1 for row in selected_any_rows if clean_text(row.get("evidence_type")) == "file_fallback")),
        "ownership_has_pr_merge_evidence": 1 if any(clean_text(row.get("evidence_type")) == "pr_merge" for row in all_evidence_rows) else 0,
        "ownership_has_pr_exact_commit_evidence": 1 if any(clean_text(row.get("evidence_type")) == "pr_exact_commit" for row in all_evidence_rows) else 0,
        "ownership_has_pr_head_evidence": 1 if any(clean_text(row.get("evidence_type")) == "pr_head" for row in all_evidence_rows) else 0,
        "ownership_has_file_fallback_evidence": 1 if any(clean_text(row.get("evidence_type")) == "file_fallback" for row in all_evidence_rows) else 0,
    }

    if sorted_churn:
        row["ownership_top_contributor_share_churn"] = sorted_churn[0]
        if len(sorted_churn) > 1:
            row["ownership_second_contributor_share_churn"] = sorted_churn[1]
    if sorted_commit:
        row["ownership_top_contributor_share_commit"] = sorted_commit[0]
        if len(sorted_commit) > 1:
            row["ownership_second_contributor_share_commit"] = sorted_commit[1]

    row["ownership_entropy_churn"] = shannon_entropy(shares_churn)
    row["ownership_entropy_commit"] = shannon_entropy(shares_commit)

    days_since_last_touch_values = [
        v for v in [summary_row.get("days_since_last_touch_before_issue") for summary_row in contributor_summary_any]
        if v is not None and not pd.isna(v)
    ]
    if top_owner_row:
        row["ownership_top_owner_days_since_last_touch"] = top_owner_row.get("days_since_last_touch_before_issue")

    if days_since_last_touch_values:
        row["ownership_median_owner_days_since_last_touch"] = take_median(days_since_last_touch_values)
        row["ownership_min_days_since_last_touch"] = min(days_since_last_touch_values)
        row["ownership_max_days_since_last_touch"] = max(days_since_last_touch_values)

    if issue_author_key:
        row["issue_author_is_owner_flag"] = 1 if issue_author_key in owner_keys else 0
        row["issue_author_is_top_owner_churn_flag"] = 1 if top_owner_key and issue_author_key == top_owner_key else 0
        if issue_author_row:
            row["issue_author_ownership_share_churn"] = issue_author_row.get("ownership_share_churn")
            row["issue_author_ownership_share_commit"] = issue_author_row.get("ownership_share_commit")
            if row.get("ownership_top_contributor_share_churn") is not None and row["issue_author_ownership_share_churn"] is not None:
                row["issue_author_vs_top_owner_gap_churn"] = float(row["ownership_top_contributor_share_churn"]) - float(row["issue_author_ownership_share_churn"])

    if contributor_count_any > 0:
        row["ownership_discussion_overlap_fraction"] = safe_divide(len(overlap_keys), contributor_count_any, default_value=None)

    participant_count = int(discussion_summary.get("discussion_participant_count", 0))
    if participant_count > 0:
        row["discussion_ownership_overlap_fraction"] = safe_divide(len(overlap_keys), participant_count, default_value=None)

    if top_owner_key:
        row["top_owner_commented_flag"] = 1 if top_owner_key in participant_keys else 0

    if total_comments_with_resolved_author > 0:
        row["owner_comment_share"] = safe_divide(owner_comment_count, total_comments_with_resolved_author, default_value=None)
        if top_owner_key:
            row["top_owner_comment_share"] = safe_divide(top_owner_comment_count, total_comments_with_resolved_author, default_value=None)

    return row

def summarize_repo_metrics(result, issue_feature_rows):
    if not issue_feature_rows:
        return result

    df = pd.DataFrame(issue_feature_rows)

    def positive_count(column_name):
        if column_name not in df.columns:
            return 0
        return int((pd.to_numeric(df[column_name], errors="coerce").fillna(0) > 0).sum())

    def mean_numeric(column_name):
        if column_name not in df.columns:
            return None
        series = pd.to_numeric(df[column_name], errors="coerce").dropna()
        if series.empty:
            return None
        return float(series.mean())

    def median_numeric(column_name, default_value=0.0):
        if column_name not in df.columns:
            return default_value
        series = pd.to_numeric(df[column_name], errors="coerce").dropna()
        if series.empty:
            return default_value
        return float(series.median())

    def mean_or_none_numeric(column_name):
        if column_name not in df.columns:
            return None
        series = pd.to_numeric(df[column_name], errors="coerce").dropna()
        if series.empty:
            return None
        return mean_or_none(series.tolist())

    result["target_issues_kept"] = int(len(df))

    result["issues_with_high_confidence_ownership"] = positive_count("ownership_usable_high_confidence")
    result["issues_ok"] = positive_count("ownership_coverage_flag")
    if "ownership_coverage_flag" in df.columns:
        sparse_mask = df["ownership_coverage_flag"].astype(str) == "sparse"
        ok_mask = df["ownership_coverage_flag"].astype(str) == "ok"
        result["issues_sparse"] = int(sparse_mask.sum())
        result["issues_ok"] = int(ok_mask.sum())

    result["issues_with_file_links"] = positive_count("ownership_has_file_links")
    result["issues_with_high_conf_file_links"] = positive_count("ownership_linked_file_count_high_confidence")
    result["issues_with_commit_matches"] = positive_count("ownership_commit_evidence_row_count")
    result["issues_with_resolved_commit_authors"] = positive_count("ownership_has_resolved_commit_authors")

    result["issues_no_file_links"] = int((pd.to_numeric(df.get("ownership_has_file_links", pd.Series(dtype="float64")), errors="coerce").fillna(0) <= 0).sum()) if "ownership_has_file_links" in df.columns else 0
    result["issues_no_commit_matches"] = int((pd.to_numeric(df.get("ownership_commit_evidence_row_count", pd.Series(dtype="float64")), errors="coerce").fillna(0) <= 0).sum()) if "ownership_commit_evidence_row_count" in df.columns else 0
    result["issues_no_resolved_commit_authors"] = int((pd.to_numeric(df.get("ownership_has_resolved_commit_authors", pd.Series(dtype="float64")), errors="coerce").fillna(0) <= 0).sum()) if "ownership_has_resolved_commit_authors" in df.columns else 0
    result["issues_missing_issue_created_at"] = int(df["issue_created_at"].isna().sum()) if "issue_created_at" in df.columns else 0

    result["issues_selected_pr_merge_evidence"] = positive_count("ownership_pr_merge_commit_count")
    result["issues_selected_pr_exact_commit_evidence"] = positive_count("ownership_pr_exact_commit_count")
    result["issues_selected_pr_head_evidence"] = positive_count("ownership_pr_head_commit_count")
    result["issues_selected_file_fallback_evidence"] = positive_count("ownership_file_fallback_commit_count")

    if {"ownership_file_fallback_commit_count", "ownership_pr_merge_commit_count", "ownership_pr_exact_commit_count", "ownership_pr_head_commit_count"}.issubset(df.columns):
        fallback_only_mask = (
            (pd.to_numeric(df["ownership_file_fallback_commit_count"], errors="coerce").fillna(0) > 0)
            & (pd.to_numeric(df["ownership_pr_merge_commit_count"], errors="coerce").fillna(0) <= 0)
            & (pd.to_numeric(df["ownership_pr_exact_commit_count"], errors="coerce").fillna(0) <= 0)
            & (pd.to_numeric(df["ownership_pr_head_commit_count"], errors="coerce").fillna(0) <= 0)
        )
        result["issues_selected_fallback_only"] = int(fallback_only_mask.sum())
    else:
        result["issues_selected_fallback_only"] = 0

    result["issues_with_pre_issue_ownership"] = positive_count("ownership_pre_issue_contributor_count")
    result["issues_with_post_issue_ownership"] = positive_count("ownership_post_issue_contributor_count")
    result["issues_with_pre_issue_high_confidence_ownership"] = positive_count("ownership_pre_issue_high_confidence_contributor_count")
    result["issues_with_post_issue_high_confidence_ownership"] = positive_count("ownership_post_issue_high_confidence_contributor_count")
    result["issues_with_pre_issue_any_ownership"] = positive_count("ownership_pre_issue_contributor_count")
    result["issues_with_post_issue_any_ownership"] = positive_count("ownership_post_issue_contributor_count")
    result["issues_with_any_conservative_pre_issue_fallback"] = positive_count("ownership_has_selected_conservative_pre_issue_fallback")

    if {"ownership_pre_issue_contributor_count", "ownership_post_issue_contributor_count"}.issubset(df.columns):
        pre_any = pd.to_numeric(df["ownership_pre_issue_contributor_count"], errors="coerce").fillna(0) > 0
        post_any = pd.to_numeric(df["ownership_post_issue_contributor_count"], errors="coerce").fillna(0) > 0
        result["issues_with_both_pre_and_post_issue_ownership"] = int((pre_any & post_any).sum())
        result["issues_with_only_pre_issue_ownership"] = int((pre_any & (~post_any)).sum())
        result["issues_with_only_post_issue_ownership"] = int(((~pre_any) & post_any).sum())
    else:
        result["issues_with_both_pre_and_post_issue_ownership"] = 0
        result["issues_with_only_pre_issue_ownership"] = 0
        result["issues_with_only_post_issue_ownership"] = 0

    if {"ownership_has_selected_conservative_pre_issue_fallback", "ownership_pre_issue_high_confidence_contributor_count"}.issubset(df.columns):
        conservative_any = pd.to_numeric(df["ownership_has_selected_conservative_pre_issue_fallback"], errors="coerce").fillna(0) > 0
        pre_high = pd.to_numeric(df["ownership_pre_issue_high_confidence_contributor_count"], errors="coerce").fillna(0) > 0
        result["issues_with_pre_issue_conservative_fallback_only"] = int((conservative_any & (~pre_high)).sum())
        result["issues_with_pre_issue_any_but_not_high_confidence"] = int((pd.to_numeric(df["ownership_pre_issue_contributor_count"], errors="coerce").fillna(0) > 0 & (~pre_high)).sum())
    else:
        result["issues_with_pre_issue_conservative_fallback_only"] = 0
        result["issues_with_pre_issue_any_but_not_high_confidence"] = 0

    result["issues_selected_for_high_confidence_features"] = positive_count("ownership_usable_high_confidence")
    result["issues_selected_for_any_features"] = positive_count("ownership_usable_any_including_conservative_pre_issue")
    result["issues_selected_for_conservative_pre_issue_fallback"] = positive_count("ownership_has_selected_conservative_pre_issue_fallback")

    if "ownership_pre_issue_high_confidence_contributor_count" in df.columns:
        result["total_selected_high_confidence_pre_issue_rows"] = int(pd.to_numeric(df["ownership_pre_issue_high_confidence_contributor_count"], errors="coerce").fillna(0).sum())
    else:
        result["total_selected_high_confidence_pre_issue_rows"] = 0

    if "ownership_pre_issue_contributor_count" in df.columns:
        result["total_selected_any_pre_issue_rows"] = int(pd.to_numeric(df["ownership_pre_issue_contributor_count"], errors="coerce").fillna(0).sum())
    else:
        result["total_selected_any_pre_issue_rows"] = 0

    if "ownership_pre_issue_conservative_fallback_commit_count" in df.columns:
        result["total_selected_conservative_pre_issue_rows"] = int(pd.to_numeric(df["ownership_pre_issue_conservative_fallback_commit_count"], errors="coerce").fillna(0).sum())
    else:
        result["total_selected_conservative_pre_issue_rows"] = 0

    result["mean_pre_issue_high_confidence_contributor_count"] = mean_numeric("ownership_pre_issue_high_confidence_contributor_count")
    result["mean_pre_issue_any_contributor_count"] = mean_numeric("ownership_pre_issue_contributor_count")
    result["mean_pre_issue_conservative_fallback_contributor_count"] = mean_numeric("ownership_pre_issue_conservative_fallback_contributor_count")

    result["median_linked_file_count_all"] = median_numeric("ownership_linked_file_count_all", default_value=0.0)
    result["median_linked_file_count_high_confidence"] = median_numeric("ownership_linked_file_count_high_confidence", default_value=0.0)
    result["median_commit_evidence_row_count"] = median_numeric("ownership_commit_evidence_row_count", default_value=0.0)
    result["median_resolved_commit_evidence_row_count"] = median_numeric("ownership_resolved_commit_evidence_row_count", default_value=0.0)
    result["median_ownership_contributor_count"] = median_numeric("ownership_contributor_count", default_value=0.0)

    result["mean_ownership_top_contributor_share_churn"] = mean_or_none_numeric("ownership_top_contributor_share_churn")
    result["mean_ownership_entropy_churn"] = mean_or_none_numeric("ownership_entropy_churn")
    result["mean_ownership_discussion_overlap_fraction"] = mean_or_none_numeric("ownership_discussion_overlap_fraction")
    result["share_issue_author_is_owner"] = mean_or_none_numeric("issue_author_is_owner_flag")
    result["share_top_owner_commented"] = mean_or_none_numeric("top_owner_commented_flag")

    return result

def process_repo(config, logger, repo_row, target_issue_lookup, repo_id_lookup, stage_paths, overlap_lookup):
    repo_full_name = repo_row["full_name"]
    repo_lookup = target_issue_lookup.get(repo_full_name)
    result = new_repo_result(repo_full_name, repo_row.get("repo_id"))
    result["identity_resolution_mode"] = get_ownership_identity_mode(config)

    if not repo_lookup:
        result["status"] = "skipped_no_target_issues"
        return result

    requested_issue_count = len(repo_lookup.get("by_issue_id", {})) + len(repo_lookup.get("by_issue_number", {}))
    result["target_issues_requested"] = requested_issue_count

    stage_inputs = load_stage_inputs_for_repo(config, repo_full_name)
    issues_df = stage_inputs["issues_resolved"]
    comments_df = stage_inputs["issue_comments_resolved"]
    issue_pr_links_df = stage_inputs["issue_pr_links"]
    pull_requests_df = stage_inputs["pull_requests"]
    pr_commit_links_df = stage_inputs["pr_commit_links"]
    issue_file_links_df = stage_inputs["issue_file_links"]
    commit_files_df = stage_inputs["commit_files"]
    commits_df = stage_inputs["commits"]
    commits_resolved_df = stage_inputs["commits_resolved"]

    result["issues_resolved_rows_seen"] = len(issues_df)
    result["issue_comments_resolved_rows_seen"] = len(comments_df)
    result["issue_pr_links_rows_seen"] = len(issue_pr_links_df)
    result["pull_requests_rows_seen"] = len(pull_requests_df)
    result["pr_commit_links_rows_seen"] = len(pr_commit_links_df)
    result["issue_file_links_rows_seen"] = len(issue_file_links_df)
    result["commit_files_rows_seen"] = len(commit_files_df)
    result["commits_rows_seen"] = len(commits_df)
    result["commits_resolved_rows_seen"] = len(commits_resolved_df)

    repo_policy = select_repo_ownership_policy(config, overlap_lookup.get(repo_full_name))
    result["ownership_policy_used"] = repo_policy

    if issues_df.empty:
        result["status"] = "completed"
        return result

    issues_df = attach_analysis_set(issues_df, repo_lookup)
    issues_df = prepare_issue_frame(issues_df)
    issues_df["repo_id"] = issues_df["repo_full_name"].map(repo_id_lookup)
    if issues_df.empty:
        result["status"] = "completed"
        return result

    result["target_issues_kept"] = len(issues_df)
    target_issue_ids = set(issues_df["issue_id"].astype(str).tolist()) if "issue_id" in issues_df.columns else set()
    target_issue_numbers = set(issues_df["issue_number"].dropna().astype(int).tolist()) if "issue_number" in issues_df.columns else set()

    comments_df = prepare_comment_frame(comments_df, target_issue_numbers)
    issue_pr_links_df = prepare_issue_pr_links_frame(issue_pr_links_df, target_issue_ids, target_issue_numbers)
    pull_requests_df = prepare_pull_requests_frame(pull_requests_df)
    pr_commit_links_df = prepare_pr_commit_links_frame(pr_commit_links_df)
    issue_file_links_df = normalize_issue_file_links_frame(issue_file_links_df, target_issue_ids, target_issue_numbers)
    exclude_bots = bool(get_ownership_option(config, "exclude_bots_from_ownership",
                                             getattr(config.bot_handling, "exclude_bots_from_ownership_metrics",
                                                     False)))
    commits_df = prepare_commits_frame(commits_df)
    commits_resolved_df = prepare_commits_resolved_frame(commits_resolved_df, exclude_bots=exclude_bots)
    commit_files_df = prepare_commit_files_frame(commit_files_df)

    high_conf_levels = get_ownership_option(config, "high_confidence_issue_file_levels", ["high"])
    high_conf_levels = {normalize_value(value) for value in list(high_conf_levels or ["high"]) if
                        normalize_value(value)}
    issue_file_lookup, _ = build_issue_file_summary(issue_file_links_df, high_conf_levels)
    commit_file_index = build_commit_file_index(commit_files_df)
    commits_lookup = build_commits_lookup(commits_df, commits_resolved_df)
    result["known_commit_rows_in_lookup"] = int(len(commits_lookup))
    pr_maps = build_pr_lookup_maps(issue_pr_links_df, pull_requests_df, pr_commit_links_df, commits_lookup)
    sparse_thresholds = {
        "min_linked_files": int(get_ownership_option(config, "min_linked_files_for_ok", 1)),
        "min_resolved_commit_rows": int(get_ownership_option(config, "min_resolved_commit_rows_for_ok", 1)),
        "min_contributors": int(get_ownership_option(config, "min_contributors_for_ok", 1)),
    }

    batch_size = int(get_ownership_option(config, "write_batch_size", 5000))
    runtime_names = get_ownership_runtime_names(config)
    repo_dir = get_batch_root(config, runtime_names["batch_folder_name"]) / sanitize_repo_name(repo_full_name)
    writer = OwnershipFeatureRepoChunkWriter(config=config, repo_dir=repo_dir, batch_size=batch_size)
    write_evidence_table = bool(get_ownership_option(config, "write_evidence_table", True))
    comments_by_issue_number = {}
    if not comments_df.empty:
        for row in comments_df.to_dict(orient="records"):
            issue_number = row.get("issue_number")
            if pd.isna(issue_number):
                continue
            comments_by_issue_number.setdefault(int(issue_number), []).append(row)

    issue_feature_rows = []
    for issue_row in issues_df.to_dict(orient="records"):
        issue_key = (
            issue_row.get("repo_full_name"),
            clean_text(issue_row.get("issue_id")),
            int(issue_row.get("issue_number")) if pd.notna(issue_row.get("issue_number")) else None,
        )
        issue_file_payload = issue_file_lookup.get(
            issue_key,
            {"all_link_row_count": 0, "linked_file_count_all": 0, "linked_file_count_high_confidence": 0, "file_rows": []},
        )
        issue_comments = comments_by_issue_number.get(issue_key[2], []) if issue_key[2] is not None else []
        discussion_summary = build_discussion_summary(pd.DataFrame(issue_comments))

        pr_merge_rows = build_issue_pr_merge_evidence(issue_row, pr_maps)
        pr_exact_rows = build_issue_pr_exact_commit_evidence(issue_row, pr_maps)
        pr_head_rows = build_issue_pr_head_evidence(issue_row, pr_maps)
        fallback_rows = build_issue_file_commit_evidence(issue_row, issue_file_payload, commit_file_index, commits_lookup)

        evidence_payload = combine_issue_ownership_evidence(
            config,
            repo_policy,
            pr_merge_rows,
            pr_exact_rows,
            pr_head_rows,
            fallback_rows,
        )

        all_evidence_rows = evidence_payload["all_evidence_rows"]
        selected_high_confidence_rows = evidence_payload["selected_high_confidence_rows"]
        selected_high_confidence_pre_issue_rows = evidence_payload["selected_high_confidence_pre_issue_rows"]
        selected_high_confidence_post_issue_rows = evidence_payload["selected_high_confidence_post_issue_rows"]
        selected_conservative_pre_issue_rows = evidence_payload["selected_conservative_pre_issue_rows"]
        selected_any_rows = evidence_payload["selected_any_rows"]
        selected_any_pre_issue_rows = evidence_payload["selected_any_pre_issue_rows"]
        selected_any_post_issue_rows = evidence_payload["selected_any_post_issue_rows"]

        contributor_summary_high_confidence = compute_contributor_summary(issue_row, selected_high_confidence_rows)
        contributor_summary_high_confidence_pre = compute_contributor_summary(issue_row,
                                                                              selected_high_confidence_pre_issue_rows)
        contributor_summary_high_confidence_post = compute_contributor_summary(issue_row,
                                                                               selected_high_confidence_post_issue_rows)

        contributor_summary_any = compute_contributor_summary(issue_row, selected_any_rows)
        contributor_summary_any_pre = compute_contributor_summary(issue_row, selected_any_pre_issue_rows)
        contributor_summary_any_post = compute_contributor_summary(issue_row, selected_any_post_issue_rows)

        issue_feature_row = build_issue_feature_row(
            issue_row,
            issue_file_payload,
            all_evidence_rows,
            selected_high_confidence_rows,
            selected_high_confidence_pre_issue_rows,
            selected_high_confidence_post_issue_rows,
            selected_conservative_pre_issue_rows,
            selected_any_rows,
            selected_any_pre_issue_rows,
            selected_any_post_issue_rows,
            contributor_summary_high_confidence,
            contributor_summary_high_confidence_pre,
            contributor_summary_high_confidence_post,
            contributor_summary_any,
            contributor_summary_any_pre,
            contributor_summary_any_post,
            discussion_summary,
            sparse_thresholds,
            repo_policy,
        )
        issue_feature_rows.append(issue_feature_row)
        writer.add_issue_row(issue_feature_row)
        result["issue_rows_written"] += 1

        linked_pr_rows = resolve_linked_pr_rows(issue_row, pr_maps)
        if linked_pr_rows:
            result["issues_with_pr_links"] += 1

        has_pr_merge_evidence = int(
            pd.to_numeric(issue_feature_row.get("ownership_has_pr_merge_evidence"), errors="coerce") or 0)
        has_pr_exact_commit_evidence = int(
            pd.to_numeric(issue_feature_row.get("ownership_has_pr_exact_commit_evidence"), errors="coerce") or 0)
        has_pr_head_evidence = int(
            pd.to_numeric(issue_feature_row.get("ownership_has_pr_head_evidence"), errors="coerce") or 0)
        has_any_pr_based_evidence = 1 if (
                has_pr_merge_evidence == 1
                or has_pr_exact_commit_evidence == 1
                or has_pr_head_evidence == 1
        ) else 0

        result["issues_with_pr_merge_evidence"] += has_pr_merge_evidence
        result["issues_with_pr_exact_commit_evidence"] += has_pr_exact_commit_evidence
        result["issues_with_pr_head_evidence"] += has_pr_head_evidence
        result["issues_with_any_pr_based_evidence"] += has_any_pr_based_evidence

        selected_pr_merge_count = int(
            sum(1 for row in selected_any_rows if clean_text(row.get("evidence_type")) == "pr_merge"))
        selected_pr_exact_commit_count = int(
            sum(1 for row in selected_any_rows if clean_text(row.get("evidence_type")) == "pr_exact_commit"))
        selected_pr_head_count = int(
            sum(1 for row in selected_any_rows if clean_text(row.get("evidence_type")) == "pr_head"))
        selected_file_fallback_count = int(
            sum(1 for row in selected_any_rows if clean_text(row.get("evidence_type")) == "file_fallback"))

        result["issues_selected_pr_merge_evidence"] += 1 if selected_pr_merge_count > 0 else 0
        result["issues_selected_pr_exact_commit_evidence"] += 1 if selected_pr_exact_commit_count > 0 else 0
        result["issues_selected_pr_head_evidence"] += 1 if selected_pr_head_count > 0 else 0
        result["issues_selected_file_fallback_evidence"] += 1 if selected_file_fallback_count > 0 else 0
        result["issues_selected_fallback_only"] += 1 if (
                selected_file_fallback_count > 0
                and selected_pr_merge_count == 0
                and selected_pr_exact_commit_count == 0
                and selected_pr_head_count == 0
        ) else 0

        if write_evidence_table:
            for evidence_row in all_evidence_rows:
                writer.add_evidence_row(evidence_row)
                result["evidence_rows_written"] += 1

    writer.finalize()
    summarize_repo_metrics(result, issue_feature_rows)
    result["status"] = "completed"
    return result

def merge_ownership_feature_batches(config, logger, stage_paths):
    runtime_names = get_ownership_runtime_names(config)
    batch_root = get_batch_root(config, runtime_names["batch_folder_name"])
    if not batch_root.exists():
        logger.warning("Ownership feature batch root does not exist: %s", batch_root)
        return

    issue_repo_parts = collect_repo_part_files(batch_root, "issue_ownership_features_part_*.parquet")
    evidence_repo_parts = collect_repo_part_files(batch_root, "issue_file_ownership_evidence_part_*.parquet")

    if issue_repo_parts:
        mode_used = write_merged_or_partitioned_output(
            repo_part_map=issue_repo_parts,
            output_path=stage_paths["issue_output_path"],
            config=config,
            table_name="issue_ownership_features",
            sort_columns=["repo_full_name", "issue_number"],
            dedupe_subset=["repo_full_name", "issue_id", "issue_number"],
        )
        logger.info("Wrote issue ownership features using %s mode to %s", mode_used, stage_paths["issue_output_path"])
    else:
        logger.warning("No issue ownership feature parts found to merge.")

    if evidence_repo_parts:
        mode_used = write_merged_or_partitioned_output(
            repo_part_map=evidence_repo_parts,
            output_path=stage_paths["evidence_output_path"],
            config=config,
            table_name="issue_file_ownership_evidence",
            sort_columns=["repo_full_name", "issue_number", "file_path", "commit_timestamp", "commit_sha"],
            dedupe_subset=["repo_full_name", "issue_id", "file_path", "commit_sha"],
        )
        logger.info("Wrote issue-file ownership evidence using %s mode to %s", mode_used, stage_paths["evidence_output_path"])
    else:
        logger.warning("No issue-file ownership evidence parts found to merge.")

def write_run_manifest(repo_rows, summary_rows, stage_paths):
    manifest_path = Path(stage_paths["run_manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "build_issue_ownership_features.py",
        "identity_resolution_mode": stage_paths.get("identity_resolution_mode"),
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
    runtime_names = get_ownership_runtime_names(config)
    repo_id_lookup = build_repo_id_lookup(config)
    overlap_lookup = load_pr_overlap_policy_lookup(stage_paths)

    repo_rows = load_repo_list(config.outputs.repo_included_list)
    target_issue_lookup = build_target_issue_lookup(config)
    batch_root = reset_batch_root(config, runtime_names["batch_folder_name"])
    logger.info("Reset batch root: %s", batch_root)

    max_repos_per_run = get_ownership_option(config, "max_repos_per_run", None)
    if max_repos_per_run and max_repos_per_run > 0:
        repo_rows = repo_rows[:max_repos_per_run]

    summary_rows = []
    for repo_row in repo_rows:
        repo_full_name = repo_row["full_name"]
        should_skip, reason = should_skip_repo(
            config,
            repo_full_name,
            checkpoint_prefix=runtime_names["checkpoint_prefix"],
            raw_folder_name=runtime_names["raw_folder_name"],
            section_name="ownership_features",
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

        logger.info(
            "Processing repo %s | identity_mode=%s",
            repo_full_name,
            get_ownership_identity_mode(config),
        )
        try:
            result = process_repo(config, logger, repo_row, target_issue_lookup, repo_id_lookup, stage_paths, overlap_lookup)
        except Exception as exc:
            logger.exception("Failed while building ownership features for %s", repo_full_name)
            result = new_repo_result(repo_full_name, repo_row.get("repo_id"))
            result["status"] = "failed"
            result["error_message"] = str(exc)

        write_repo_checkpoint(config, runtime_names["checkpoint_prefix"], repo_full_name, result)
        summary_rows.append(result)

    merge_ownership_feature_batches(config, logger, stage_paths)
    write_summary_csv(summary_rows, stage_paths["qa_summary_path"])
    write_run_manifest(repo_rows, summary_rows, stage_paths)
    logger.info("Ownership feature building complete. Repos processed: %s", len(summary_rows))


if __name__ == "__main__":
    main()
