import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

SCRIPT_VERSION = "matched_set_id_attachment_v2026_06_17"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.study_config_loader import ensure_project_directories, load_study_config
from utils.checkpoints import get_stage_option
from utils.io_helpers import (
    clean_text,
    has_real_value,
    load_table,
    safe_to_datetime,
    write_processed_table,
    write_summary_csv,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"
LOG_FILENAME = "13_build_analysis_dataset.log"
RQ2_REPO_PARTICIPANT_ROLE_COLUMNS = [
    "issue_author_has_resolved_key",
    "issue_author_is_pre_issue_repo_contributor",
    "issue_author_pre_issue_repo_commit_count",
    "issue_author_pre_issue_repo_commit_share",
    "issue_author_pre_issue_repo_distinct_days",
    "issue_author_pre_issue_repo_rank",
    "issue_author_is_pre_issue_major_repo_contributor",
    "commenter_count_with_resolved_key",
    "commenter_count_pre_issue_repo_contributors",
    "share_commenters_pre_issue_repo_contributors",
    "any_commenter_is_pre_issue_repo_contributor",
    "commenter_count_pre_issue_major_repo_contributors",
    "share_commenters_pre_issue_major_repo_contributors",
    "any_commenter_is_pre_issue_major_repo_contributor",
    "top_commenter_contributor_key",
    "top_commenter_comment_count",
    "top_commenter_is_pre_issue_repo_contributor",
    "top_commenter_is_pre_issue_major_repo_contributor",
]

RQ2_FILE_PARTICIPANT_ROLE_COLUMNS = [
    "participant_role_file_coverage_flag",
    "participant_role_file_features_applicable",
    "participant_role_linked_file_count",
    "participant_role_high_conf_linked_file_count",
    "participant_role_has_file_links",
    "participant_role_has_high_conf_file_links",
    "participant_role_has_pre_issue_file_history",
    "participant_role_pre_issue_file_history_file_count",
    "participant_role_file_author_has_resolved_key",
    "participant_role_file_commenter_count_with_resolved_key",
    "participant_role_file_has_resolved_commenters",
    "participant_role_file_top_commenter_has_resolved_key",
    "participant_role_file_author_applicable",
    "participant_role_file_commenter_applicable",
    "participant_role_file_top_commenter_applicable",
    "issue_author_is_pre_issue_file_contributor",
    "issue_author_pre_issue_linked_file_commit_count",
    "issue_author_pre_issue_linked_file_count",
    "issue_author_pre_issue_linked_file_share",
    "issue_author_pre_issue_best_file_rank",
    "issue_author_pre_issue_max_file_commit_share",
    "issue_author_is_pre_issue_major_file_contributor",
    "commenter_count_pre_issue_file_contributors",
    "share_commenters_pre_issue_file_contributors",
    "any_commenter_is_pre_issue_file_contributor",
    "commenter_count_pre_issue_major_file_contributors",
    "share_commenters_pre_issue_major_file_contributors",
    "any_commenter_is_pre_issue_major_file_contributor",
    "top_commenter_is_pre_issue_file_contributor",
    "top_commenter_is_pre_issue_major_file_contributor",
]

RQ2_CONTINUITY_COLUMNS = [
    "pre_issue_owner_count_for_continuity",
    "post_issue_owner_count_for_continuity",
    "pre_post_owner_overlap_count",
    "pre_post_owner_jaccard",
    "any_pre_issue_owner_became_post_issue_owner",
    "top_pre_issue_owner_contributor_key",
    "top_pre_issue_owner_became_post_issue_owner",
    "post_issue_owners_with_pre_issue_repo_history_count",
    "post_issue_owners_with_pre_issue_major_repo_history_count",
    "share_post_issue_owners_with_pre_issue_repo_history",
    "share_post_issue_owners_with_pre_issue_major_repo_history",
    "any_post_issue_owner_with_pre_issue_repo_history",
    "any_post_issue_owner_with_pre_issue_major_repo_history",
    "post_issue_owners_with_pre_issue_file_history_count",
    "post_issue_owners_with_pre_issue_major_file_history_count",
    "share_post_issue_owners_with_pre_issue_file_history",
    "share_post_issue_owners_with_pre_issue_major_file_history",
    "any_post_issue_owner_with_pre_issue_file_history",
    "any_post_issue_owner_with_pre_issue_major_file_history",
    "issue_author_is_post_issue_owner",
    "issue_author_pre_issue_repo_contributor_became_post_issue_owner",
    "issue_author_pre_issue_file_contributor_became_post_issue_owner",
    "commenter_count_eventual_post_issue_owners",
    "share_commenters_eventual_post_issue_owners",
    "any_commenter_is_eventual_post_issue_owner",
    "top_commenter_is_eventual_post_issue_owner",
    "commenter_count_pre_issue_repo_contributors_eventual_post_issue_owners",
    "any_pre_issue_repo_contributor_commenter_is_eventual_post_issue_owner",
    "commenter_count_pre_issue_file_contributors_eventual_post_issue_owners",
    "any_pre_issue_file_contributor_commenter_is_eventual_post_issue_owner",
]

RQ2_DIRECT_OWNERSHIP_COLUMNS = [
    "ownership_has_pre_issue_ownership",
    "ownership_has_post_issue_ownership",
    "ownership_pre_issue_contributor_count",
    "ownership_post_issue_contributor_count",
    "ownership_has_selected_conservative_pre_issue_fallback",
    "ownership_top_contributor_share_churn",
    "ownership_entropy_churn",
    "ownership_discussion_overlap_fraction",
]

RQ2_BINARY_COLUMNS = [
    "issue_author_has_resolved_key",
    "issue_author_is_pre_issue_repo_contributor",
    "issue_author_is_pre_issue_major_repo_contributor",
    "any_commenter_is_pre_issue_repo_contributor",
    "any_commenter_is_pre_issue_major_repo_contributor",
    "top_commenter_is_pre_issue_repo_contributor",
    "top_commenter_is_pre_issue_major_repo_contributor",
    "participant_role_file_features_applicable",
    "participant_role_has_file_links",
    "participant_role_has_high_conf_file_links",
    "participant_role_has_pre_issue_file_history",
    "participant_role_file_author_has_resolved_key",
    "participant_role_file_has_resolved_commenters",
    "participant_role_file_top_commenter_has_resolved_key",
    "participant_role_file_author_applicable",
    "participant_role_file_commenter_applicable",
    "participant_role_file_top_commenter_applicable",
    "issue_author_is_pre_issue_file_contributor",
    "issue_author_is_pre_issue_major_file_contributor",
    "any_commenter_is_pre_issue_file_contributor",
    "any_commenter_is_pre_issue_major_file_contributor",
    "top_commenter_is_pre_issue_file_contributor",
    "top_commenter_is_pre_issue_major_file_contributor",
    "any_pre_issue_owner_became_post_issue_owner",
    "top_pre_issue_owner_became_post_issue_owner",
    "any_post_issue_owner_with_pre_issue_repo_history",
    "any_post_issue_owner_with_pre_issue_major_repo_history",
    "any_post_issue_owner_with_pre_issue_file_history",
    "any_post_issue_owner_with_pre_issue_major_file_history",
    "issue_author_is_post_issue_owner",
    "issue_author_pre_issue_repo_contributor_became_post_issue_owner",
    "issue_author_pre_issue_file_contributor_became_post_issue_owner",
    "any_commenter_is_eventual_post_issue_owner",
    "top_commenter_is_eventual_post_issue_owner",
    "any_pre_issue_repo_contributor_commenter_is_eventual_post_issue_owner",
    "any_pre_issue_file_contributor_commenter_is_eventual_post_issue_owner",
    "ownership_has_pre_issue_ownership",
    "ownership_has_post_issue_ownership",
    "ownership_has_selected_conservative_pre_issue_fallback",
]

RQ2_COUNT_COLUMNS = [
    "issue_author_pre_issue_repo_commit_count",
    "issue_author_pre_issue_repo_distinct_days",
    "issue_author_pre_issue_repo_rank",
    "commenter_count_with_resolved_key",
    "commenter_count_pre_issue_repo_contributors",
    "commenter_count_pre_issue_major_repo_contributors",
    "top_commenter_comment_count",
    "participant_role_linked_file_count",
    "participant_role_high_conf_linked_file_count",
    "participant_role_pre_issue_file_history_file_count",
    "participant_role_file_commenter_count_with_resolved_key",
    "issue_author_pre_issue_linked_file_commit_count",
    "issue_author_pre_issue_linked_file_count",
    "issue_author_pre_issue_best_file_rank",
    "commenter_count_pre_issue_file_contributors",
    "commenter_count_pre_issue_major_file_contributors",
    "pre_issue_owner_count_for_continuity",
    "post_issue_owner_count_for_continuity",
    "pre_post_owner_overlap_count",
    "post_issue_owners_with_pre_issue_repo_history_count",
    "post_issue_owners_with_pre_issue_major_repo_history_count",
    "post_issue_owners_with_pre_issue_file_history_count",
    "post_issue_owners_with_pre_issue_major_file_history_count",
    "commenter_count_eventual_post_issue_owners",
    "commenter_count_pre_issue_repo_contributors_eventual_post_issue_owners",
    "commenter_count_pre_issue_file_contributors_eventual_post_issue_owners",
    "ownership_pre_issue_contributor_count",
    "ownership_post_issue_contributor_count",
]

RQ2_SHARE_COLUMNS = [
    "issue_author_pre_issue_repo_commit_share",
    "share_commenters_pre_issue_repo_contributors",
    "share_commenters_pre_issue_major_repo_contributors",
    "issue_author_pre_issue_linked_file_share",
    "issue_author_pre_issue_max_file_commit_share",
    "share_commenters_pre_issue_file_contributors",
    "share_commenters_pre_issue_major_file_contributors",
    "pre_post_owner_jaccard",
    "share_post_issue_owners_with_pre_issue_repo_history",
    "share_post_issue_owners_with_pre_issue_major_repo_history",
    "share_post_issue_owners_with_pre_issue_file_history",
    "share_post_issue_owners_with_pre_issue_major_file_history",
    "share_commenters_eventual_post_issue_owners",
    "ownership_top_contributor_share_churn",
    "ownership_entropy_churn",
    "ownership_discussion_overlap_fraction",
]


def setup_logger(config):
    logger = logging.getLogger("build_analysis_dataset")
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

def get_analysis_dataset_option(config, field_name, default_value):
    return get_stage_option(config, "analysis_dataset", field_name, default_value)

def get_stage_paths(config):
    outputs = getattr(config, "outputs", None)

    full_output_path = getattr(outputs, "analysis_dataset_full_issue_level_table", None)
    if not full_output_path:
        full_output_path = "./data/final/analysis_dataset_full_issue_level.parquet"

    rq1_output_path = getattr(outputs, "analysis_dataset_rq1_table", None)
    if not rq1_output_path:
        rq1_output_path = "./data/final/analysis_dataset_rq1.parquet"

    rq2_output_path = getattr(outputs, "analysis_dataset_rq2_table", None)
    if not rq2_output_path:
        rq2_output_path = "./data/final/analysis_dataset_rq2.parquet"

    rq3_issue_base_output_path = getattr(outputs, "analysis_dataset_rq3_issue_level_base_table", None)
    if not rq3_issue_base_output_path:
        rq3_issue_base_output_path = "./data/final/analysis_dataset_rq3_issue_level_base.parquet"

    qa_summary_path = getattr(outputs, "analysis_dataset_qa_summary_csv", None)
    if not qa_summary_path:
        qa_summary_path = "./logs/qa/analysis_dataset_qa_summary.csv"

    qa_output_dir = getattr(outputs, "analysis_dataset_qa_output_dir", None)
    if not qa_output_dir:
        qa_output_dir = "./outputs/qa/dataset_building"

    return {
        "full_output_path": Path(full_output_path),
        "rq1_output_path": Path(rq1_output_path),
        "rq2_output_path": Path(rq2_output_path),
        "rq3_issue_base_output_path": Path(rq3_issue_base_output_path),
        "qa_summary_path": Path(qa_summary_path),
        "qa_output_dir": Path(qa_output_dir),
        "run_manifest_path": Path(config.logging.qa_log_dir) / "13_build_analysis_dataset_run_manifest.json",
    }

def find_first_present_column(df, candidates):
    lower_map = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None

def drop_wontfix_comparison_overlap(df, dataset_name="analysis_dataset"):
    """Remove rows where the same issue appears as both WONTFIX and comparison.

    This should never happen after a clean matched lookup. If it does happen,
    WONTFIX wins and the comparison copy is dropped.
    """
    if df is None or df.empty:
        return df

    required = {"repo_full_name", "issue_id", "analysis_set"}
    if not required.issubset(set(df.columns)):
        return df

    out = df.copy()
    out["__analysis_set_norm"] = out["analysis_set"].astype(str).str.lower().str.strip()
    out["__issue_key_for_overlap"] = (
        out["repo_full_name"].astype(str) + "::" + out["issue_id"].astype(str)
    )

    wontfix_keys = set(
        out.loc[out["__analysis_set_norm"].eq("wontfix"), "__issue_key_for_overlap"]
    )

    comparison_overlap_mask = (
        out["__analysis_set_norm"].eq("comparison")
        & out["__issue_key_for_overlap"].isin(wontfix_keys)
    )

    dropped = int(comparison_overlap_mask.sum())
    if dropped:
        print(
            f"[WARN] {dataset_name}: dropping {dropped} comparison rows whose issue_id also appears as WONTFIX"
        )

    out = out.loc[~comparison_overlap_mask].copy()
    out = out.drop(columns=["__analysis_set_norm", "__issue_key_for_overlap"], errors="ignore")
    return out.reset_index(drop=True)

def normalize_issue_set_columns(df, analysis_set_name):
    if df.empty:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number", "analysis_set"])

    normalized = df.copy()
    repo_col = find_first_present_column(normalized, ["repo_full_name", "repo_name", "full_name", "repo"])
    issue_id_col = find_first_present_column(normalized, ["issue_id", "id"])
    issue_number_col = find_first_present_column(normalized, ["issue_number", "number"])
    comparison_group_col = find_first_present_column(normalized, ["comparison_group", "comparison_bucket", "bucket"])
    issue_type_col = find_first_present_column(normalized, ["issue_type", "type", "__issue_type"])

    if repo_col is None:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number", "analysis_set"])

    out_df = pd.DataFrame()
    out_df["repo_full_name"] = normalized[repo_col].astype(str)
    out_df["issue_id"] = normalized[issue_id_col].astype(str) if issue_id_col else None
    out_df["issue_number"] = pd.to_numeric(normalized[issue_number_col], errors="coerce") if issue_number_col else pd.NA
    out_df["analysis_set"] = analysis_set_name
    out_df["comparison_group"] = (
        normalized[comparison_group_col].astype(str) if comparison_group_col else ("wontfix" if analysis_set_name == "wontfix" else "comparison")
    )
    out_df["issue_type"] = normalized[issue_type_col].astype(str) if issue_type_col else None

    out_df = out_df.drop_duplicates().reset_index(drop=True)
    return out_df

def build_population_frame(config):
    merge_mode = getattr(config.storage, "processed_merge_mode", "single_parquet")
    wontfix_df = load_table(config.outputs.wontfix_issue_set_table, merge_mode=merge_mode)
    comparison_df = load_table(config.outputs.comparison_issue_set_table, merge_mode=merge_mode)

    wontfix_df = normalize_issue_set_columns(wontfix_df, "wontfix")
    comparison_df = normalize_issue_set_columns(comparison_df, "comparison")

    if wontfix_df.empty and comparison_df.empty:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number", "analysis_set", "comparison_group", "issue_type"])

    combined = pd.concat([wontfix_df, comparison_df], ignore_index=True)
    combined["repo_full_name"] = combined["repo_full_name"].astype(str)
    combined["issue_id"] = combined["issue_id"].astype(str)
    combined["issue_number"] = pd.to_numeric(combined["issue_number"], errors="coerce")
    combined = combined.drop_duplicates(subset=["repo_full_name", "issue_id", "issue_number", "analysis_set"]).reset_index(drop=True)
    return combined

def normalize_issues_resolved_frame(df):
    if df.empty:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number"])

    repo_col = find_first_present_column(df, ["repo_full_name", "repo_name", "full_name", "repo"])
    issue_id_col = find_first_present_column(df, ["issue_id", "id"])
    issue_number_col = find_first_present_column(df, ["issue_number", "number"])

    if repo_col is None or issue_id_col is None:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number"])

    selected_columns = [repo_col, issue_id_col]
    optional_map = {
        "issue_number": ["issue_number", "number"],
        "repo_id": ["repo_id", "id"],
        "state": ["state"],
        "created_at": ["created_at"],
        "closed_at": ["closed_at"],
        "author_login": ["author_login"],
        "issue_author_contributor_key": ["issue_author_contributor_key"],
        "closed_by_login": ["closed_by_login"],
        "issue_closer_contributor_key": ["issue_closer_contributor_key"],
        "state_reason": ["state_reason"],
        "title": ["title"],
        "label_names_json": ["label_names_json", "labels", "label_names"],
    }

    rename_map = {repo_col: "repo_full_name", issue_id_col: "issue_id"}
    if issue_number_col:
        rename_map[issue_number_col] = "issue_number"
        selected_columns.append(issue_number_col)

    for output_col, candidates in optional_map.items():
        col = find_first_present_column(df, candidates)
        if col and col not in selected_columns:
            selected_columns.append(col)
            rename_map[col] = output_col

    out_df = df[selected_columns].copy().rename(columns=rename_map)
    out_df["repo_full_name"] = out_df["repo_full_name"].astype(str)
    out_df["issue_id"] = out_df["issue_id"].astype(str)
    if "issue_number" not in out_df.columns:
        out_df["issue_number"] = pd.NA
    out_df["issue_number"] = pd.to_numeric(out_df["issue_number"], errors="coerce")
    if "created_at" in out_df.columns:
        out_df["created_at"] = safe_to_datetime(out_df["created_at"])
    if "closed_at" in out_df.columns:
        out_df["closed_at"] = safe_to_datetime(out_df["closed_at"])
    return out_df.drop_duplicates(subset=["repo_full_name", "issue_id", "issue_number"]).reset_index(drop=True)

def normalize_repo_metadata_frame(df):
    if df.empty:
        return pd.DataFrame(columns=["repo_full_name"])

    repo_col = find_first_present_column(df, ["repo_full_name", "full_name", "repo_name"])
    if repo_col is None:
        return pd.DataFrame(columns=["repo_full_name"])

    selected_columns = [repo_col]
    rename_map = {repo_col: "repo_full_name"}
    optional_map = {
        "repo_id": ["repo_id", "id"],
        "language": ["language", "primary_language"],
        "stargazers_count": ["stargazers_count", "stars"],
        "owner_login": ["owner_login"],
        "visibility": ["visibility"],
        "pushed_at": ["pushed_at"],
        "created_at_repo": ["created_at"],
    }
    for output_col, candidates in optional_map.items():
        col = find_first_present_column(df, candidates)
        if col and col not in selected_columns:
            selected_columns.append(col)
            rename_map[col] = output_col

    out_df = df[selected_columns].copy().rename(columns=rename_map)
    out_df["repo_full_name"] = out_df["repo_full_name"].astype(str)
    if "pushed_at" in out_df.columns:
        out_df["pushed_at"] = safe_to_datetime(out_df["pushed_at"])
    if "created_at_repo" in out_df.columns:
        out_df["created_at_repo"] = safe_to_datetime(out_df["created_at_repo"])
    return out_df.drop_duplicates(subset=["repo_full_name"]).reset_index(drop=True)

def normalize_feature_frame(df, family_name):
    if df.empty:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number", f"__has_{family_name}"])

    normalized = df.copy()
    repo_col = find_first_present_column(normalized, ["repo_full_name", "repo_name", "full_name", "repo"])
    issue_id_col = find_first_present_column(normalized, ["issue_id", "id"])
    issue_number_col = find_first_present_column(normalized, ["issue_number", "number"])

    if repo_col is None:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number", f"__has_{family_name}"])

    rename_map = {repo_col: "repo_full_name"}
    if issue_id_col:
        rename_map[issue_id_col] = "issue_id"
    if issue_number_col:
        rename_map[issue_number_col] = "issue_number"

    out_df = normalized.rename(columns=rename_map).copy()
    if "issue_id" not in out_df.columns:
        out_df["issue_id"] = None
    if "issue_number" not in out_df.columns:
        out_df["issue_number"] = pd.NA

    out_df["repo_full_name"] = out_df["repo_full_name"].astype(str)
    out_df["issue_id"] = out_df["issue_id"].astype(str)
    out_df["issue_number"] = pd.to_numeric(out_df["issue_number"], errors="coerce")

    for date_col in ["created_at", "closed_at", "issue_created_at", "issue_closed_at"]:
        if date_col in out_df.columns:
            out_df[date_col] = safe_to_datetime(out_df[date_col])

    out_df[f"__has_{family_name}"] = 1
    out_df = out_df.drop_duplicates().reset_index(drop=True)
    return out_df

def dedupe_feature_frame(df, family_name, logger):
    if df.empty:
        return df, 0

    exact_deduped = df.drop_duplicates().reset_index(drop=True)

    duplicate_count = int(exact_deduped.duplicated(subset=["repo_full_name", "issue_id", "issue_number"]).sum())
    if duplicate_count > 0:
        logger.warning(
            "Duplicate feature keys detected | family=%s | duplicates=%s | keeping first stable row per key",
            family_name,
            duplicate_count,
        )
        exact_deduped = exact_deduped.drop_duplicates(
            subset=["repo_full_name", "issue_id", "issue_number"], keep="first"
        ).reset_index(drop=True)

    return exact_deduped, duplicate_count

def merge_with_issue_fallback(base_df, add_df, family_name):
    if base_df.empty or add_df.empty:
        return base_df.copy()

    add_df = add_df.copy()
    key_columns = {"repo_full_name", "issue_id", "issue_number"}
    non_key_cols = [col for col in add_df.columns if col not in key_columns]
    if not non_key_cols:
        return base_df.copy()

    id_ready = add_df[add_df["issue_id"].apply(has_real_value)].copy()
    num_ready = add_df[add_df["issue_number"].notna()].copy()

    out_df = base_df.copy()

    if not id_ready.empty:
        id_merge_cols = ["repo_full_name", "issue_id"] + non_key_cols
        by_id = out_df.merge(
            id_ready[id_merge_cols],
            on=["repo_full_name", "issue_id"],
            how="left",
        )
    else:
        by_id = pd.DataFrame(index=out_df.index)

    if not num_ready.empty:
        num_merge_cols = ["repo_full_name", "issue_number"] + non_key_cols
        by_num = out_df.merge(
            num_ready[num_merge_cols],
            on=["repo_full_name", "issue_number"],
            how="left",
            suffixes=("", f"__{family_name}__num"),
        )
    else:
        by_num = pd.DataFrame(index=out_df.index)

    combined_columns = {}
    for col in non_key_cols:
        id_series = by_id[col] if col in by_id.columns else pd.Series(pd.NA, index=out_df.index)
        num_series = by_num[col] if col in by_num.columns else pd.Series(pd.NA, index=out_df.index)
        combined_columns[col] = id_series.reset_index(drop=True).combine_first(
            num_series.reset_index(drop=True)
        )

    additions_df = pd.DataFrame(combined_columns, index=out_df.index)

    output_base = out_df.drop(columns=[col for col in non_key_cols if col in out_df.columns], errors="ignore")
    merged = pd.concat(
        [
            output_base.reset_index(drop=True),
            additions_df.reset_index(drop=True),
        ],
        axis=1,
    )

    return merged.copy()

def reconcile_shared_columns(df, left_col, right_col, canonical_col, qa_metrics):
    if left_col not in df.columns and right_col not in df.columns:
        return df

    left_series = df[left_col] if left_col in df.columns else pd.Series(pd.NA, index=df.index)
    right_series = df[right_col] if right_col in df.columns else pd.Series(pd.NA, index=df.index)

    mismatch_mask = left_series.notna() & right_series.notna() & (left_series != right_series)
    qa_metrics[f"shared_metric_mismatch__{canonical_col}"] = int(mismatch_mask.sum())

    df[canonical_col] = left_series.combine_first(right_series)
    return df

def add_rq2_ownership_defaults_and_derived_fields(df):
    out_df = df.copy()

    string_defaults = {
        "top_commenter_contributor_key": None,
        "participant_role_file_coverage_flag": "missing",
        "top_pre_issue_owner_contributor_key": None,
    }

    numeric_defaults = {}

    for column in RQ2_BINARY_COLUMNS:
        numeric_defaults[column] = 0

    for column in RQ2_COUNT_COLUMNS:
        numeric_defaults[column] = 0

    for column in RQ2_SHARE_COLUMNS:
        numeric_defaults[column] = pd.NA

    default_columns = {}
    for column, default_value in string_defaults.items():
        if column not in out_df.columns:
            default_columns[column] = pd.Series(default_value, index=out_df.index)

    for column, default_value in numeric_defaults.items():
        if column not in out_df.columns:
            default_columns[column] = pd.Series(default_value, index=out_df.index)

    if default_columns:
        out_df = pd.concat([out_df, pd.DataFrame(default_columns, index=out_df.index)], axis=1).copy()

    for column in RQ2_BINARY_COLUMNS:
        if column in out_df.columns:
            out_df[column] = pd.to_numeric(out_df[column], errors="coerce").fillna(0).astype(int)

    for column in RQ2_COUNT_COLUMNS:
        if column in out_df.columns:
            out_df[column] = pd.to_numeric(out_df[column], errors="coerce").fillna(0)

    for column in RQ2_SHARE_COLUMNS:
        if column in out_df.columns:
            out_df[column] = pd.to_numeric(out_df[column], errors="coerce")

    if "participant_role_file_coverage_flag" in out_df.columns:
        out_df["participant_role_file_coverage_flag"] = (
            out_df["participant_role_file_coverage_flag"]
            .fillna("missing")
            .astype(str)
            .str.strip()
            .replace({"": "missing", "nan": "missing", "None": "missing", "<NA>": "missing"})
        )

    out_df["has_post_issue_ownership"] = (
        (pd.to_numeric(out_df.get("ownership_post_issue_contributor_count", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out_df.get("ownership_has_post_issue_ownership", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out_df.get("post_issue_owner_count_for_continuity", 0), errors="coerce").fillna(0) > 0)
    ).astype(int)

    out_df["has_pre_issue_issue_linked_ownership"] = (
        (pd.to_numeric(out_df.get("ownership_pre_issue_contributor_count", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out_df.get("ownership_has_pre_issue_ownership", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out_df.get("pre_issue_owner_count_for_continuity", 0), errors="coerce").fillna(0) > 0)
    ).astype(int)

    out_df["has_repo_participant_role_signal"] = (
        (pd.to_numeric(out_df.get("issue_author_is_pre_issue_repo_contributor", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out_df.get("any_commenter_is_pre_issue_repo_contributor", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out_df.get("top_commenter_is_pre_issue_repo_contributor", 0), errors="coerce").fillna(0) > 0)
    ).astype(int)

    out_df["has_file_participant_role_signal"] = (
        (pd.to_numeric(out_df.get("participant_role_file_features_applicable", 0), errors="coerce").fillna(0) > 0)
        & (
            (pd.to_numeric(out_df.get("issue_author_is_pre_issue_file_contributor", 0), errors="coerce").fillna(0) > 0)
            | (pd.to_numeric(out_df.get("any_commenter_is_pre_issue_file_contributor", 0), errors="coerce").fillna(0) > 0)
            | (pd.to_numeric(out_df.get("top_commenter_is_pre_issue_file_contributor", 0), errors="coerce").fillna(0) > 0)
        )
    ).astype(int)

    out_df["has_continuity_signal"] = (
        (pd.to_numeric(out_df.get("any_post_issue_owner_with_pre_issue_repo_history", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out_df.get("any_post_issue_owner_with_pre_issue_file_history", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out_df.get("any_commenter_is_eventual_post_issue_owner", 0), errors="coerce").fillna(0) > 0)
    ).astype(int)

    return out_df.copy()

def add_family_presence_flags(df):
    out_df = df.copy()

    out_df["has_sentiment_features"] = out_df.get("__has_sentiment", 0).fillna(0).astype(int)
    out_df["has_participation_features"] = out_df.get("__has_participation", 0).fillna(0).astype(int)

    ownership_flag_col = find_first_present_column(
        out_df,
        [
            "ownership_feature_coverage_flag",
            "ownership_coverage_flag",
        ],
    )

    if ownership_flag_col is not None:
        hard_missing = {
            "no_file_links",
            "no_commit_matches",
            "no_resolved_commit_authors",
            "missing_issue_created_at",
        }
        out_df["has_direct_issue_linked_ownership_features"] = (
            ~out_df[ownership_flag_col].isin(hard_missing)
            & out_df[ownership_flag_col].notna()
        ).astype(int)
    else:
        out_df["has_direct_issue_linked_ownership_features"] = out_df.get("__has_ownership", 0).fillna(0).astype(int)

    out_df["has_repo_participant_role_features"] = (
        (pd.to_numeric(out_df.get("issue_author_has_resolved_key", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out_df.get("commenter_count_with_resolved_key", 0), errors="coerce").fillna(0) > 0)
    ).astype(int)

    out_df["has_file_participant_role_features"] = (
        pd.to_numeric(out_df.get("participant_role_file_features_applicable", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    out_df["has_continuity_features"] = (
        pd.to_numeric(out_df.get("post_issue_owner_count_for_continuity", 0), errors="coerce")
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    out_df["has_ownership_features"] = (
        (out_df.get("__has_ownership", 0).fillna(0).astype(int) > 0)
        | (out_df["has_repo_participant_role_features"] > 0)
        | (out_df["has_file_participant_role_features"] > 0)
        | (out_df["has_direct_issue_linked_ownership_features"] > 0)
        | (out_df["has_continuity_features"] > 0)
    ).astype(int)

    return out_df.copy()

def add_rq_usability_flags(df, config):
    out_df = df.copy()
    require_sentiment_for_rq1 = bool(get_analysis_dataset_option(config, "require_sentiment_for_rq1", True))
    require_participation_for_rq1 = bool(get_analysis_dataset_option(config, "require_participation_for_rq1", False))
    ownership_include_sparse = bool(get_analysis_dataset_option(config, "ownership_usable_flags_include_sparse", True))

    rq1_mask = pd.Series(True, index=out_df.index)

    if require_sentiment_for_rq1:
        if "has_sentiment_features" in out_df.columns:
            rq1_mask = rq1_mask & out_df["has_sentiment_features"].eq(1)
        else:
            rq1_mask = rq1_mask & False

    if require_participation_for_rq1:
        if "has_participation_features" in out_df.columns:
            rq1_mask = rq1_mask & out_df["has_participation_features"].eq(1)
        else:
            rq1_mask = rq1_mask & False

    out_df["usable_for_rq1"] = rq1_mask.astype(int)

    out_df["usable_for_rq2_repo_participant_roles"] = (
        (pd.to_numeric(out_df.get("issue_author_has_resolved_key", 0), errors="coerce").fillna(0) > 0)
        | (pd.to_numeric(out_df.get("commenter_count_with_resolved_key", 0), errors="coerce").fillna(0) > 0)
    ).astype(int)

    out_df["usable_for_rq2_file_participant_roles"] = (
        pd.to_numeric(out_df.get("participant_role_file_features_applicable", 0), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    ownership_coverage_col = find_first_present_column(
        out_df,
        [
            "ownership_feature_coverage_flag",
            "ownership_coverage_flag",
        ],
    )

    if ownership_coverage_col is not None:
        ownership_coverage = out_df[ownership_coverage_col]
    else:
        ownership_coverage = pd.Series(pd.NA, index=out_df.index)

    acceptable_flags = {"ok"}
    if ownership_include_sparse:
        acceptable_flags.add("sparse_evidence")

    out_df["usable_for_rq2_direct_ownership"] = ownership_coverage.isin(acceptable_flags).astype(int)
    out_df["usable_for_rq2_direct_ownership_strict"] = ownership_coverage.eq("ok").astype(int)

    out_df["usable_for_rq2_continuity"] = (
        pd.to_numeric(out_df.get("post_issue_owner_count_for_continuity", 0), errors="coerce")
        .fillna(0)
        .gt(0)
        .astype(int)
    )

    out_df["usable_for_rq2"] = out_df["usable_for_rq2_repo_participant_roles"].astype(int)

    out_df["usable_for_rq2_strict"] = out_df["usable_for_rq2_direct_ownership_strict"].astype(int)

    if "has_participation_features" in out_df.columns:
        out_df["usable_for_rq3"] = out_df["has_participation_features"].eq(1).astype(int)
    else:
        out_df["usable_for_rq3"] = 0

    return out_df.copy()

def build_rq_views(full_df, config):
    rq1_df = full_df[full_df["usable_for_rq1"] == 1].copy()

    filter_rq2 = bool(get_analysis_dataset_option(config, "filter_rq2_to_ownership_usable", False))
    if filter_rq2:
        rq2_df = full_df[full_df["usable_for_rq2"] == 1].copy()
    else:
        rq2_df = full_df.copy()

    if "usable_for_rq3" in full_df.columns:
        rq3_issue_base_df = full_df[full_df["usable_for_rq3"] == 1].copy()
    else:
        rq3_issue_base_df = full_df.copy()

    return (
        rq1_df.reset_index(drop=True).copy(),
        rq2_df.reset_index(drop=True).copy(),
        rq3_issue_base_df.reset_index(drop=True).copy(),
    )

def build_qa_summary_rows(full_df, rq1_df, rq2_df, rq3_issue_base_df, qa_metrics):
    rows = []

    def add(metric, value):
        rows.append({"metric": metric, "value": value})

    for metric, value in qa_metrics.items():
        add(metric, value)

    add("population_rows_final", int(len(full_df)))
    add("rows_written__rq1", int(len(rq1_df)))
    add("rows_written__rq2", int(len(rq2_df)))
    add("rows_written__rq3_issue_level_base", int(len(rq3_issue_base_df)))

    if not full_df.empty:
        add("repos_represented", int(full_df["repo_full_name"].nunique()))
        add("rows_with_sentiment_features", int(full_df["has_sentiment_features"].sum()))
        add("rows_with_participation_features", int(full_df["has_participation_features"].sum()))
        add("rows_with_ownership_features", int(full_df["has_ownership_features"].sum()))
        add("rows_usable_for_rq1", int(full_df["usable_for_rq1"].sum()))
        add("rows_usable_for_rq2", int(full_df["usable_for_rq2"].sum()))
        add(
            "rows_usable_for_rq2_repo_participant_roles",
            int(full_df.get("usable_for_rq2_repo_participant_roles", pd.Series(0, index=full_df.index)).sum()),
        )
        add(
            "rows_usable_for_rq2_file_participant_roles",
            int(full_df.get("usable_for_rq2_file_participant_roles", pd.Series(0, index=full_df.index)).sum()),
        )
        add(
            "rows_usable_for_rq2_direct_ownership",
            int(full_df.get("usable_for_rq2_direct_ownership", pd.Series(0, index=full_df.index)).sum()),
        )
        add(
            "rows_usable_for_rq2_direct_ownership_strict",
            int(full_df.get("usable_for_rq2_direct_ownership_strict", pd.Series(0, index=full_df.index)).sum()),
        )
        add(
            "rows_usable_for_rq2_continuity",
            int(full_df.get("usable_for_rq2_continuity", pd.Series(0, index=full_df.index)).sum()),
        )
        add("rows_usable_for_rq2_strict", int(full_df["usable_for_rq2_strict"].sum()))

        if "analysis_set" in full_df.columns:
            for key, value in full_df["analysis_set"].fillna("missing").astype(str).value_counts(dropna=False).items():
                add(f"analysis_set_count__{key}", int(value))

        if "comparison_group" in full_df.columns:
            for key, value in full_df["comparison_group"].fillna("missing").astype(str).value_counts(dropna=False).items():
                add(f"comparison_group_count__{key}", int(value))

        if "matched_set_join_method" in full_df.columns:
            for key, value in full_df["matched_set_join_method"].fillna("missing").astype(str).value_counts(dropna=False).items():
                add(f"matched_set_join_method_count__{key}", int(value))

        if "usable_for_matched_set_fe" in full_df.columns:
            add("rows_usable_for_matched_set_fe", int(pd.to_numeric(full_df["usable_for_matched_set_fe"], errors="coerce").fillna(0).sum()))

        if "participant_role_file_coverage_flag" in full_df.columns:
            for key, value in full_df["participant_role_file_coverage_flag"].fillna("missing").astype(str).value_counts(
                    dropna=False).items():
                add(f"participant_role_file_coverage_count__{key}", int(value))

        for column_name in [
            "has_repo_participant_role_signal",
            "has_file_participant_role_signal",
            "has_continuity_signal",
            "has_direct_issue_linked_ownership_features",
            "has_repo_participant_role_features",
            "has_file_participant_role_features",
            "has_continuity_features",
        ]:
            if column_name in full_df.columns:
                add(f"rows_with__{column_name}",
                    int(pd.to_numeric(full_df[column_name], errors="coerce").fillna(0).sum()))

        if "participation_feature_coverage_flag" in full_df.columns:
            for key, value in full_df["participation_feature_coverage_flag"].fillna("missing").astype(str).value_counts(dropna=False).items():
                add(f"participation_coverage_count__{key}", int(value))

        for column_name in [
            "mean_comment_sentiment",
            "top_commenter_share",
            "ownership_top_contributor_share_churn",
        ]:
            if column_name in full_df.columns:
                add(f"missing_share__{column_name}", float(full_df[column_name].isna().mean()))

    return rows

def sort_output_frame(df):
    if df.empty:
        return df
    sort_columns = [col for col in ["repo_full_name", "analysis_set", "issue_number", "issue_id"] if col in df.columns]
    if not sort_columns:
        return df.reset_index(drop=True)
    return df.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def read_optional_table(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=suffix in {".jsonl", ".ndjson"})
    raise ValueError(f"Unsupported matched-set lookup file format: {path}")


def get_matched_set_lookup_path(config):
    outputs = getattr(config, "outputs", None)
    explicit_path = None
    for attr_name in [
        "wontfix_comparison_pairs_issue_lookup_csv",
        "wontfix_comparison_pairs_issue_lookup_table",
        "matched_set_issue_lookup_csv",
        "matched_set_issue_lookup_table",
    ]:
        if outputs is not None and hasattr(outputs, attr_name):
            value = getattr(outputs, attr_name)
            if value:
                explicit_path = value
                break
    if explicit_path:
        return Path(explicit_path)

    pair_path = None
    for attr_name in ["wontfix_comparison_pairs_table", "wontfix_comparison_pairs"]:
        if outputs is not None and hasattr(outputs, attr_name):
            value = getattr(outputs, attr_name)
            if value:
                pair_path = Path(value)
                break
    if pair_path:
        return pair_path.with_name(pair_path.stem + "_issue_lookup.csv")

    return Path("./data/processed/wontfix_comparison_pairs_issue_lookup.csv")


def normalize_matched_set_lookup(df):
    columns = ["repo_full_name", "issue_id", "issue_number", "analysis_set", "matched_set_id"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    normalized = df.copy()
    repo_col = find_first_present_column(normalized, ["repo_full_name", "repo_name", "full_name", "repo"])
    issue_id_col = find_first_present_column(normalized, ["issue_id", "id"])
    issue_number_col = find_first_present_column(normalized, ["issue_number", "number"])
    analysis_set_col = find_first_present_column(normalized, ["analysis_set", "matched_set_analysis_set", "group"])
    matched_set_col = find_first_present_column(normalized, ["matched_set_id", "match_set_id", "set_id"])

    if repo_col is None or matched_set_col is None:
        return pd.DataFrame(columns=columns)

    out_df = pd.DataFrame()
    out_df["repo_full_name"] = normalized[repo_col].astype(str)
    out_df["issue_id"] = normalized[issue_id_col].apply(clean_text) if issue_id_col else None
    out_df["issue_number"] = pd.to_numeric(normalized[issue_number_col], errors="coerce") if issue_number_col else pd.NA
    out_df["analysis_set"] = normalized[analysis_set_col].apply(clean_text).str.lower() if analysis_set_col else None
    out_df["matched_set_id"] = normalized[matched_set_col].apply(clean_text)

    out_df = out_df[out_df["matched_set_id"].apply(has_real_value)].copy()
    out_df["analysis_set"] = out_df["analysis_set"].fillna("missing")
    out_df["issue_id"] = out_df["issue_id"].apply(clean_text)
    out_df["issue_number"] = pd.to_numeric(out_df["issue_number"], errors="coerce")
    return out_df[columns].drop_duplicates().reset_index(drop=True)


def load_matched_set_lookup(config, logger):
    lookup_path = get_matched_set_lookup_path(config)
    lookup_df = read_optional_table(lookup_path)
    lookup_df = normalize_matched_set_lookup(lookup_df)
    if lookup_df.empty:
        logger.warning("Matched-set lookup not found or empty | path=%s", lookup_path)
    else:
        logger.info(
            "Loaded matched-set issue lookup | path=%s | rows=%s | matched_sets=%s",
            lookup_path,
            len(lookup_df),
            lookup_df["matched_set_id"].nunique(),
        )
    return lookup_df, lookup_path


def attach_matched_set_ids(base_df, matched_lookup_df, lookup_path, qa_metrics):
    out_df = base_df.copy()
    default_columns = {
        "matched_set_id": pd.NA,
        "matched_set_analysis_set": pd.NA,
        "matched_set_join_method": "unmatched",
        "matched_set_total_rows_in_analysis": 0,
        "matched_set_wontfix_rows_in_analysis": 0,
        "matched_set_comparison_rows_in_analysis": 0,
        "has_matched_set": 0,
        "usable_for_matched_set_fe": 0,
    }
    for column, default_value in default_columns.items():
        if column not in out_df.columns:
            out_df[column] = default_value

    qa_metrics["matched_set_lookup_path"] = str(lookup_path)
    qa_metrics["matched_set_lookup_rows"] = int(len(matched_lookup_df))
    qa_metrics["matched_set_lookup_distinct_sets"] = int(matched_lookup_df["matched_set_id"].nunique()) if not matched_lookup_df.empty else 0
    qa_metrics["matched_set_lookup_distinct_issues"] = int(matched_lookup_df.drop_duplicates(subset=["repo_full_name", "issue_id", "issue_number"]).shape[0]) if not matched_lookup_df.empty else 0

    if out_df.empty or matched_lookup_df.empty:
        qa_metrics["analysis_rows_with_matched_set_id"] = 0
        qa_metrics["analysis_rows_missing_matched_set_id"] = int(len(out_df))
        qa_metrics["matched_sets_in_analysis"] = 0
        qa_metrics["matched_sets_valid_for_fe"] = 0
        qa_metrics["analysis_rows_usable_for_matched_set_fe"] = 0
        return out_df

    lookup = matched_lookup_df.copy()
    lookup["issue_id"] = lookup["issue_id"].apply(clean_text)
    lookup["issue_number"] = pd.to_numeric(lookup["issue_number"], errors="coerce")

    id_lookup = lookup[lookup["issue_id"].apply(has_real_value)].copy()
    num_lookup = lookup[lookup["issue_number"].notna()].copy()

    qa_metrics["matched_set_lookup_duplicate_repo_issue_id_keys"] = int(
        id_lookup.duplicated(subset=["repo_full_name", "issue_id"]).sum()
    ) if not id_lookup.empty else 0
    qa_metrics["matched_set_lookup_duplicate_repo_issue_number_keys"] = int(
        num_lookup.duplicated(subset=["repo_full_name", "issue_number"]).sum()
    ) if not num_lookup.empty else 0

    id_lookup = id_lookup.drop_duplicates(subset=["repo_full_name", "issue_id"], keep="first")
    num_lookup = num_lookup.drop_duplicates(subset=["repo_full_name", "issue_number"], keep="first")

    base_for_join = out_df[["repo_full_name", "issue_id", "issue_number", "analysis_set"]].copy()
    base_for_join["issue_id"] = base_for_join["issue_id"].apply(clean_text)
    base_for_join["issue_number"] = pd.to_numeric(base_for_join["issue_number"], errors="coerce")

    if not id_lookup.empty:
        id_join = base_for_join.merge(
            id_lookup[["repo_full_name", "issue_id", "analysis_set", "matched_set_id"]].rename(
                columns={"analysis_set": "matched_set_analysis_set_id_join"}
            ),
            on=["repo_full_name", "issue_id"],
            how="left",
        )
        id_set = id_join["matched_set_id"]
        id_analysis_set = id_join["matched_set_analysis_set_id_join"]
    else:
        id_set = pd.Series(pd.NA, index=out_df.index)
        id_analysis_set = pd.Series(pd.NA, index=out_df.index)

    if not num_lookup.empty:
        num_join = base_for_join.merge(
            num_lookup[["repo_full_name", "issue_number", "analysis_set", "matched_set_id"]].rename(
                columns={"analysis_set": "matched_set_analysis_set_number_join"}
            ),
            on=["repo_full_name", "issue_number"],
            how="left",
        )
        num_set = num_join["matched_set_id"]
        num_analysis_set = num_join["matched_set_analysis_set_number_join"]
    else:
        num_set = pd.Series(pd.NA, index=out_df.index)
        num_analysis_set = pd.Series(pd.NA, index=out_df.index)

    id_set = id_set.reset_index(drop=True)
    num_set = num_set.reset_index(drop=True)
    id_analysis_set = id_analysis_set.reset_index(drop=True)
    num_analysis_set = num_analysis_set.reset_index(drop=True)

    both_match_disagree = id_set.notna() & num_set.notna() & (id_set.astype(str) != num_set.astype(str))
    qa_metrics["matched_set_issue_id_number_conflict_rows"] = int(both_match_disagree.sum())

    out_df["matched_set_id"] = id_set.combine_first(num_set)
    out_df["matched_set_analysis_set"] = id_analysis_set.combine_first(num_analysis_set)
    out_df["matched_set_join_method"] = "unmatched"
    out_df.loc[id_set.notna(), "matched_set_join_method"] = "issue_id"
    out_df.loc[id_set.isna() & num_set.notna(), "matched_set_join_method"] = "issue_number"
    out_df["has_matched_set"] = out_df["matched_set_id"].notna().astype(int)

    analysis_set_norm = out_df["analysis_set"].fillna("missing").astype(str).str.lower()
    lookup_set_norm = out_df["matched_set_analysis_set"].fillna("missing").astype(str).str.lower()
    mismatch_mask = out_df["has_matched_set"].eq(1) & lookup_set_norm.ne("missing") & analysis_set_norm.ne(lookup_set_norm)
    qa_metrics["matched_set_analysis_set_mismatch_rows"] = int(mismatch_mask.sum())

    matched_rows = out_df[out_df["has_matched_set"] == 1].copy()
    if matched_rows.empty:
        qa_metrics["analysis_rows_with_matched_set_id"] = 0
        qa_metrics["analysis_rows_missing_matched_set_id"] = int(len(out_df))
        qa_metrics["matched_sets_in_analysis"] = 0
        qa_metrics["matched_sets_valid_for_fe"] = 0
        qa_metrics["analysis_rows_usable_for_matched_set_fe"] = 0
        return out_df

    set_summary = (
        matched_rows.assign(
            __is_wontfix=analysis_set_norm.loc[matched_rows.index].eq("wontfix").astype(int),
            __is_comparison=analysis_set_norm.loc[matched_rows.index].eq("comparison").astype(int),
        )
        .groupby("matched_set_id", dropna=False)
        .agg(
            matched_set_total_rows_in_analysis=("matched_set_id", "size"),
            matched_set_wontfix_rows_in_analysis=("__is_wontfix", "sum"),
            matched_set_comparison_rows_in_analysis=("__is_comparison", "sum"),
            matched_set_repo_count=("repo_full_name", "nunique"),
        )
        .reset_index()
    )
    valid_set_ids = set(
        set_summary.loc[
            (set_summary["matched_set_wontfix_rows_in_analysis"] >= 1)
            & (set_summary["matched_set_comparison_rows_in_analysis"] >= 1),
            "matched_set_id",
        ].astype(str)
    )

    for metric_col in [
        "matched_set_total_rows_in_analysis",
        "matched_set_wontfix_rows_in_analysis",
        "matched_set_comparison_rows_in_analysis",
    ]:
        metric_map = set_summary.set_index("matched_set_id")[metric_col]
        out_df[metric_col] = out_df["matched_set_id"].map(metric_map).fillna(0).astype(int)

    out_df["usable_for_matched_set_fe"] = out_df["matched_set_id"].astype(str).isin(valid_set_ids).astype(int)

    qa_metrics["analysis_rows_with_matched_set_id"] = int(out_df["has_matched_set"].sum())
    qa_metrics["analysis_rows_missing_matched_set_id"] = int((out_df["has_matched_set"] == 0).sum())
    qa_metrics["wontfix_rows_with_matched_set_id"] = int(((analysis_set_norm == "wontfix") & (out_df["has_matched_set"] == 1)).sum())
    qa_metrics["comparison_rows_with_matched_set_id"] = int(((analysis_set_norm == "comparison") & (out_df["has_matched_set"] == 1)).sum())
    qa_metrics["wontfix_rows_missing_matched_set_id"] = int(((analysis_set_norm == "wontfix") & (out_df["has_matched_set"] == 0)).sum())
    qa_metrics["comparison_rows_missing_matched_set_id"] = int(((analysis_set_norm == "comparison") & (out_df["has_matched_set"] == 0)).sum())
    qa_metrics["matched_sets_in_analysis"] = int(set_summary["matched_set_id"].nunique())
    qa_metrics["matched_sets_with_wontfix"] = int((set_summary["matched_set_wontfix_rows_in_analysis"] >= 1).sum())
    qa_metrics["matched_sets_with_comparison"] = int((set_summary["matched_set_comparison_rows_in_analysis"] >= 1).sum())
    qa_metrics["matched_sets_valid_for_fe"] = int(len(valid_set_ids))
    qa_metrics["matched_sets_cross_repo_in_analysis"] = int((set_summary["matched_set_repo_count"] > 1).sum())
    qa_metrics["analysis_rows_usable_for_matched_set_fe"] = int(out_df["usable_for_matched_set_fe"].sum())
    qa_metrics["wontfix_rows_usable_for_matched_set_fe"] = int(((analysis_set_norm == "wontfix") & (out_df["usable_for_matched_set_fe"] == 1)).sum())
    qa_metrics["comparison_rows_usable_for_matched_set_fe"] = int(((analysis_set_norm == "comparison") & (out_df["usable_for_matched_set_fe"] == 1)).sum())

    controls_per_wontfix = []
    wontfix_rows = out_df[analysis_set_norm == "wontfix"].copy()
    comp_count_map = set_summary.set_index("matched_set_id")["matched_set_comparison_rows_in_analysis"]
    for _, row in wontfix_rows.iterrows():
        matched_set_id = row.get("matched_set_id")
        if has_real_value(matched_set_id):
            controls_per_wontfix.append(int(comp_count_map.get(matched_set_id, 0)))
        else:
            controls_per_wontfix.append(0)
    controls_dist = pd.Series(controls_per_wontfix, dtype="int64").value_counts().sort_index().to_dict() if controls_per_wontfix else {}
    qa_metrics["controls_per_wontfix_distribution_json"] = json.dumps({str(k): int(v) for k, v in controls_dist.items()}, sort_keys=True)

    return out_df.copy()


def write_dataset_building_qa_artifacts(full_df, qa_output_dir, logger):
    qa_output_dir = Path(qa_output_dir)
    qa_output_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    if full_df.empty:
        return written

    # Attachment status by analysis set.
    if "analysis_set" in full_df.columns and "has_matched_set" in full_df.columns:
        attachment_df = (
            full_df.assign(
                matched_set_status=full_df["has_matched_set"].map({1: "has_matched_set", 0: "missing_matched_set"}).fillna("missing_matched_set")
            )
            .groupby(["analysis_set", "matched_set_status"], dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values(["analysis_set", "matched_set_status"])
        )
        attachment_csv = qa_output_dir / "matched_set_attachment_by_analysis_set.csv"
        attachment_df.to_csv(attachment_csv, index=False)
        written["matched_set_attachment_by_analysis_set_csv"] = str(attachment_csv)

        if plt is not None and not attachment_df.empty:
            pivot = attachment_df.pivot(index="analysis_set", columns="matched_set_status", values="rows").fillna(0)
            ax = pivot.plot(kind="bar")
            ax.set_title("Matched-set ID attachment by analysis set")
            ax.set_xlabel("Analysis set")
            ax.set_ylabel("Rows")
            ax.legend(title="Matched-set status")
            fig = ax.get_figure()
            fig.tight_layout()
            png_path = qa_output_dir / "matched_set_attachment_by_analysis_set.png"
            fig.savefig(png_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            written["matched_set_attachment_by_analysis_set_png"] = str(png_path)

    # Controls-per-WONTFIX distribution in the final analysis population.
    required_cols = {"analysis_set", "matched_set_id"}
    if required_cols.issubset(set(full_df.columns)):
        analysis_set_norm = full_df["analysis_set"].fillna("missing").astype(str).str.lower()
        matched_rows = full_df[full_df["matched_set_id"].notna()].copy()
        if not matched_rows.empty:
            comp_counts = (
                matched_rows[analysis_set_norm.loc[matched_rows.index].eq("comparison")]
                .groupby("matched_set_id")
                .size()
            )
        else:
            comp_counts = pd.Series(dtype="int64")
        wontfix_rows = full_df[analysis_set_norm.eq("wontfix")].copy()
        if not wontfix_rows.empty:
            controls = []
            for _, row in wontfix_rows.iterrows():
                matched_set_id = row.get("matched_set_id")
                controls.append(int(comp_counts.get(matched_set_id, 0)) if has_real_value(matched_set_id) else 0)
            dist_df = (
                pd.Series(controls, name="controls_per_wontfix")
                .value_counts()
                .sort_index()
                .rename_axis("controls_per_wontfix")
                .reset_index(name="wontfix_issues")
            )
        else:
            dist_df = pd.DataFrame(columns=["controls_per_wontfix", "wontfix_issues"])

        dist_csv = qa_output_dir / "controls_per_wontfix_distribution.csv"
        dist_df.to_csv(dist_csv, index=False)
        written["controls_per_wontfix_distribution_csv"] = str(dist_csv)

        if plt is not None and not dist_df.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(dist_df["controls_per_wontfix"].astype(str), dist_df["wontfix_issues"])
            ax.set_title("Controls per WONTFIX issue")
            ax.set_xlabel("Number of matched comparison controls")
            ax.set_ylabel("WONTFIX issues")
            fig.tight_layout()
            png_path = qa_output_dir / "controls_per_wontfix_distribution.png"
            fig.savefig(png_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            written["controls_per_wontfix_distribution_png"] = str(png_path)

    if written:
        logger.info("Wrote dataset-building QA artifacts | output_dir=%s | files=%s", qa_output_dir, len(written))
    return written

def load_input_frames(config, logger):
    merge_mode = getattr(config.storage, "processed_merge_mode", "single_parquet")

    population_df = build_population_frame(config)
    issues_resolved_df = normalize_issues_resolved_frame(load_table(config.outputs.issues_resolved_table, merge_mode=merge_mode))
    repositories_df = normalize_repo_metadata_frame(load_table(config.outputs.repositories_table, merge_mode=merge_mode))

    outputs = getattr(config, "outputs", None)
    sentiment_path = getattr(outputs, "issue_sentiment_features_table", None) or "./data/features/sentiment/issue_sentiment_features.parquet"
    participation_path = getattr(outputs, "issue_participation_features_table", None) or "./data/features/participation/issue_participation_features.parquet"
    ownership_path = getattr(outputs, "issue_ownership_features_table", None) or "./data/features/ownership/issue_ownership_features.parquet"

    sentiment_df = normalize_feature_frame(load_table(sentiment_path, merge_mode=merge_mode), "sentiment")
    participation_df = normalize_feature_frame(load_table(participation_path, merge_mode=merge_mode), "participation")
    ownership_df = normalize_feature_frame(load_table(ownership_path, merge_mode=merge_mode), "ownership")
    matched_lookup_df, matched_lookup_path = load_matched_set_lookup(config, logger)

    sentiment_df, sentiment_dupes = dedupe_feature_frame(sentiment_df, "sentiment", logger)
    participation_df, participation_dupes = dedupe_feature_frame(participation_df, "participation", logger)
    ownership_df, ownership_dupes = dedupe_feature_frame(ownership_df, "ownership", logger)

    qa_metrics = {
        "population_rows_expected": int(len(population_df)),
        "issues_resolved_rows_seen": int(len(issues_resolved_df)),
        "repositories_rows_seen": int(len(repositories_df)),
        "sentiment_rows_seen": int(len(sentiment_df)),
        "participation_rows_seen": int(len(participation_df)),
        "ownership_rows_seen": int(len(ownership_df)),
        "matched_set_lookup_path": str(matched_lookup_path),
        "matched_set_lookup_rows_seen": int(len(matched_lookup_df)),
        "duplicate_sentiment_keys_detected": int(sentiment_dupes),
        "duplicate_participation_keys_detected": int(participation_dupes),
        "duplicate_ownership_keys_detected": int(ownership_dupes),
    }
    return population_df, issues_resolved_df, repositories_df, sentiment_df, participation_df, ownership_df, matched_lookup_df, matched_lookup_path, qa_metrics

def build_full_analysis_dataset(config, logger):
    (
        population_df,
        issues_resolved_df,
        repositories_df,
        sentiment_df,
        participation_df,
        ownership_df,
        matched_lookup_df,
        matched_lookup_path,
        qa_metrics,
    ) = load_input_frames(config, logger)

    if population_df.empty:
        return pd.DataFrame(), qa_metrics

    population_df = population_df.drop_duplicates(subset=["repo_full_name", "issue_id", "issue_number", "analysis_set"]).reset_index(drop=True)
    full_df = population_df.copy()

    full_df = attach_matched_set_ids(full_df, matched_lookup_df, matched_lookup_path, qa_metrics)

    if not issues_resolved_df.empty:
        full_df = merge_with_issue_fallback(full_df, issues_resolved_df, "issues_resolved")

    if not repositories_df.empty:
        repo_join_cols = [col for col in repositories_df.columns if col != "repo_full_name"]
        full_df = full_df.merge(repositories_df[["repo_full_name"] + repo_join_cols], on="repo_full_name", how="left")

    if not sentiment_df.empty:
        full_df = merge_with_issue_fallback(full_df, sentiment_df, "sentiment")

    if not participation_df.empty:
        full_df = merge_with_issue_fallback(full_df, participation_df, "participation")

    if not ownership_df.empty:
        full_df = merge_with_issue_fallback(full_df, ownership_df, "ownership")

    # reconcile shared columns if both sentiment and participation carry them via suffix-free merge fallback
    # keep canonical names and also retain family-specific columns when present.
    if "comment_count_x" in full_df.columns or "comment_count_y" in full_df.columns:
        full_df = reconcile_shared_columns(full_df, "comment_count_x", "comment_count_y", "comment_count", qa_metrics)
    if "unique_commenter_count_x" in full_df.columns or "unique_commenter_count_y" in full_df.columns:
        full_df = reconcile_shared_columns(full_df, "unique_commenter_count_x", "unique_commenter_count_y", "unique_commenter_count", qa_metrics)

    # backfill repo_id from repo metadata where issue metadata did not provide it
    if "repo_id" not in full_df.columns:
        full_df["repo_id"] = pd.NA
    if "repo_id_repo" in full_df.columns:
        full_df["repo_id"] = full_df["repo_id"].combine_first(full_df["repo_id_repo"])

    # clean up common duplicate suffix artifacts from merges
    for base_name in ["comment_count", "unique_commenter_count", "repo_id"]:
        left_col = f"{base_name}_x"
        right_col = f"{base_name}_y"
        if left_col in full_df.columns or right_col in full_df.columns:
            if base_name not in full_df.columns:
                left_series = full_df[left_col] if left_col in full_df.columns else pd.Series(pd.NA, index=full_df.index)
                right_series = full_df[right_col] if right_col in full_df.columns else pd.Series(pd.NA, index=full_df.index)
                full_df[base_name] = left_series.combine_first(right_series)
            full_df = full_df.drop(columns=[col for col in [left_col, right_col] if col in full_df.columns], errors="ignore")

    full_df = add_rq2_ownership_defaults_and_derived_fields(full_df)
    full_df = add_family_presence_flags(full_df)
    full_df = add_rq_usability_flags(full_df, config)

    # Normalize key/time columns after merging.
    full_df["repo_full_name"] = full_df["repo_full_name"].astype(str)
    full_df["issue_id"] = full_df["issue_id"].astype(str)
    full_df["issue_number"] = pd.to_numeric(full_df["issue_number"], errors="coerce")
    for date_col in ["created_at", "closed_at", "issue_created_at", "issue_closed_at", "pushed_at", "created_at_repo"]:
        if date_col in full_df.columns:
            full_df[date_col] = safe_to_datetime(full_df[date_col])

    full_df = sort_output_frame(full_df)

    qa_metrics["population_rows_final_pre_filter"] = int(len(full_df))
    qa_metrics["population_rows_final"] = int(len(full_df))
    qa_metrics["rows_with_sentiment_features"] = int(full_df["has_sentiment_features"].sum())
    qa_metrics["rows_with_participation_features"] = int(full_df["has_participation_features"].sum())
    qa_metrics["rows_with_ownership_features"] = int(full_df["has_ownership_features"].sum())
    qa_metrics["rows_usable_for_rq1"] = int(full_df["usable_for_rq1"].sum())
    qa_metrics["rows_usable_for_rq2"] = int(full_df["usable_for_rq2"].sum())
    qa_metrics["rows_usable_for_rq2_repo_participant_roles"] = int(
        full_df.get("usable_for_rq2_repo_participant_roles", pd.Series(0, index=full_df.index)).sum()
    )
    qa_metrics["rows_usable_for_rq2_file_participant_roles"] = int(
        full_df.get("usable_for_rq2_file_participant_roles", pd.Series(0, index=full_df.index)).sum()
    )
    qa_metrics["rows_usable_for_rq2_direct_ownership"] = int(
        full_df.get("usable_for_rq2_direct_ownership", pd.Series(0, index=full_df.index)).sum()
    )
    qa_metrics["rows_usable_for_rq2_direct_ownership_strict"] = int(
        full_df.get("usable_for_rq2_direct_ownership_strict", pd.Series(0, index=full_df.index)).sum()
    )
    qa_metrics["rows_usable_for_rq2_continuity"] = int(
        full_df.get("usable_for_rq2_continuity", pd.Series(0, index=full_df.index)).sum()
    )
    qa_metrics["rows_usable_for_rq2_strict"] = int(full_df["usable_for_rq2_strict"].sum())
    qa_metrics["rows_usable_for_rq3"] = int(
        full_df.get("usable_for_rq3", pd.Series(0, index=full_df.index)).sum()
    )
    qa_metrics["rows_with_matched_set_id_final"] = int(
        full_df.get("has_matched_set", pd.Series(0, index=full_df.index)).sum()
    )
    qa_metrics["rows_usable_for_matched_set_fe_final"] = int(
        full_df.get("usable_for_matched_set_fe", pd.Series(0, index=full_df.index)).sum()
    )

    return full_df, qa_metrics

def write_run_manifest(run_manifest_path, payload):
    run_manifest_path = Path(run_manifest_path)
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with run_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

def main(config_path=None):
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = load_study_config(config_path)
    ensure_project_directories(config)
    logger = setup_logger(config)
    stage_paths = get_stage_paths(config)

    started_at = datetime.utcnow()
    logger.info("Starting analysis dataset build")

    full_df, qa_metrics = build_full_analysis_dataset(config, logger)

    if full_df.empty:
        logger.warning("No analysis dataset rows were produced.")
        write_summary_csv(build_qa_summary_rows(full_df, full_df, full_df, full_df, qa_metrics), stage_paths["qa_summary_path"])
        write_run_manifest(
            stage_paths["run_manifest_path"],
            {
                "status": "completed_empty",
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": datetime.utcnow().isoformat(),
                "rows_written": 0,
                "config_path": str(config_path),
            },
        )
        return

    rq1_df, rq2_df, rq3_df = build_rq_views(full_df, config)

    rq1_df = drop_wontfix_comparison_overlap(rq1_df, dataset_name="rq1")
    rq2_df = drop_wontfix_comparison_overlap(rq2_df, dataset_name="rq2")
    rq3_df = drop_wontfix_comparison_overlap(rq3_df, dataset_name="rq3")

    write_processed_table(full_df, stage_paths["full_output_path"], config)
    write_processed_table(rq1_df, stage_paths["rq1_output_path"], config)
    write_processed_table(rq2_df, stage_paths["rq2_output_path"], config)
    write_processed_table(rq3_df, stage_paths["rq3_issue_base_output_path"], config)

    qa_artifact_paths = write_dataset_building_qa_artifacts(full_df, stage_paths["qa_output_dir"], logger)
    qa_metrics["dataset_building_qa_output_dir"] = str(stage_paths["qa_output_dir"])
    for artifact_key, artifact_path in qa_artifact_paths.items():
        qa_metrics[artifact_key] = artifact_path

    qa_rows = build_qa_summary_rows(full_df, rq1_df, rq2_df, rq3_df, qa_metrics)
    write_summary_csv(qa_rows, stage_paths["qa_summary_path"])

    write_run_manifest(
        stage_paths["run_manifest_path"],
        {
            "status": "completed",
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": datetime.utcnow().isoformat(),
            "config_path": str(config_path),
            "rows_written_full": int(len(full_df)),
            "rows_written_rq1": int(len(rq1_df)),
            "rows_written_rq2": int(len(rq2_df)),
            "rows_written_rq3_issue_level_base": int(len(rq3_df)),
            "output_paths": {
                "full_output_path": str(stage_paths["full_output_path"]),
                "rq1_output_path": str(stage_paths["rq1_output_path"]),
                "rq2_output_path": str(stage_paths["rq2_output_path"]),
                "rq3_issue_base_output_path": str(stage_paths["rq3_issue_base_output_path"]),
                "qa_summary_path": str(stage_paths["qa_summary_path"]),
                "qa_output_dir": str(stage_paths["qa_output_dir"]),
            },
        },
    )

    logger.info(
        "Analysis dataset build complete | full=%s | rq1=%s | rq2=%s | rq3_issue_base=%s",
        len(full_df),
        len(rq1_df),
        len(rq2_df),
        len(rq3_df),
    )


if __name__ == "__main__":
    main()
