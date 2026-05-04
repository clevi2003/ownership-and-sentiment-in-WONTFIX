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
    has_real_value,
    load_repo_list,
    load_table,
    repo_filter,
    safe_divide,
    safe_to_datetime,
    write_merged_or_partitioned_output,
    write_summary_csv,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"


class ContributorOwnershipProfileRepoChunkWriter:
    def __init__(self, config, repo_dir, batch_size=5000):
        self.config = config
        self.repo_dir = Path(repo_dir)
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = max(int(batch_size or 1), 1)
        self.repo_profile_rows = []
        self.file_profile_rows = []
        self.repo_part_index = 1
        self.file_part_index = 1

    def add_repo_profile_row(self, row):
        if row is None:
            return
        self.repo_profile_rows.append(dict(row))
        if len(self.repo_profile_rows) >= self.batch_size:
            self._flush_repo_profile_rows()

    def add_file_profile_row(self, row):
        if row is None:
            return
        self.file_profile_rows.append(dict(row))
        if len(self.file_profile_rows) >= self.batch_size:
            self._flush_file_profile_rows()

    def _flush_repo_profile_rows(self):
        if not self.repo_profile_rows:
            return
        output_path = self.repo_dir / f"contributor_repo_ownership_profiles_part_{self.repo_part_index:05d}.parquet"
        pd.DataFrame(self.repo_profile_rows).to_parquet(
            output_path,
            index=False,
            compression=self.config.storage.compression.parquet_compression,
        )
        self.repo_profile_rows = []
        self.repo_part_index += 1

    def _flush_file_profile_rows(self):
        if not self.file_profile_rows:
            return
        output_path = self.repo_dir / f"contributor_file_ownership_profiles_part_{self.file_part_index:05d}.parquet"
        pd.DataFrame(self.file_profile_rows).to_parquet(
            output_path,
            index=False,
            compression=self.config.storage.compression.parquet_compression,
        )
        self.file_profile_rows = []
        self.file_part_index += 1

    def finalize(self):
        self._flush_repo_profile_rows()
        self._flush_file_profile_rows()


def get_contributor_profile_identity_mode(config):
    mode = get_stage_option(config, "identity_resolution", "attachment_identity_mode", "strict")
    mode = str(mode).strip().lower()
    if mode not in {"strict", "fuzzy"}:
        raise ValueError(
            f"identity_resolution.attachment_identity_mode must be 'strict' or 'fuzzy', got: {mode}"
        )
    return mode


def get_contributor_profile_runtime_names(config):
    mode = get_contributor_profile_identity_mode(config)
    if mode == "fuzzy":
        return {
            "log_filename": "11_a_build_contributor_ownership_profiles_fuzzy.log",
            "checkpoint_prefix": "11_a_build_contributor_ownership_profiles_fuzzy",
            "batch_folder_name": "contributor_ownership_profiles_fuzzy",
            "raw_folder_name": "contributor_ownership_profiles_fuzzy",
            "summary_filename": "11_a_build_contributor_ownership_profiles_fuzzy_summary.csv",
            "run_manifest_filename": "11_a_build_contributor_ownership_profiles_fuzzy_run_manifest.json",
        }

    return {
        "log_filename": "11_a_build_contributor_ownership_profiles.log",
        "checkpoint_prefix": "11_a_build_contributor_ownership_profiles",
        "batch_folder_name": "contributor_ownership_profiles",
        "raw_folder_name": "contributor_ownership_profiles",
        "summary_filename": "11_a_build_contributor_ownership_profiles_summary.csv",
        "run_manifest_filename": "11_a_build_contributor_ownership_profiles_run_manifest.json",
    }


def setup_logger(config):
    runtime_names = get_contributor_profile_runtime_names(config)
    logger = logging.getLogger("build_contributor_ownership_profiles")
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


def get_contributor_profile_option(config, field_name, default_value):
    ownership_cfg = getattr(config, "ownership_features", None)
    if ownership_cfg is None:
        return default_value
    if not hasattr(ownership_cfg, field_name):
        return default_value
    value = getattr(ownership_cfg, field_name)
    if value is None:
        return default_value
    return value


def get_contributor_profile_stage_paths(config):
    outputs = getattr(config, "outputs", None)
    mode = get_contributor_profile_identity_mode(config)
    runtime_names = get_contributor_profile_runtime_names(config)

    if mode == "fuzzy":
        repo_output_path = getattr(outputs, "contributor_repo_ownership_profiles_table_fuzzy", None)
        if not repo_output_path:
            repo_output_path = "./data/features/ownership_fuzzy/contributor_repo_ownership_profiles_fuzzy.parquet"

        file_output_path = getattr(outputs, "contributor_file_ownership_profiles_table_fuzzy", None)
        if not file_output_path:
            file_output_path = "./data/features/ownership_fuzzy/contributor_file_ownership_profiles_fuzzy.parquet"

        qa_summary_path = getattr(outputs, "contributor_ownership_profile_qa_summary_csv_fuzzy", None)
        if not qa_summary_path:
            qa_summary_path = "./logs/qa/contributor_ownership_profile_qa_summary_fuzzy.csv"
    else:
        repo_output_path = getattr(outputs, "contributor_repo_ownership_profiles_table", None)
        if not repo_output_path:
            repo_output_path = "./data/features/ownership/contributor_repo_ownership_profiles.parquet"

        file_output_path = getattr(outputs, "contributor_file_ownership_profiles_table", None)
        if not file_output_path:
            file_output_path = "./data/features/ownership/contributor_file_ownership_profiles.parquet"

        qa_summary_path = getattr(outputs, "contributor_ownership_profile_qa_summary_csv", None)
        if not qa_summary_path:
            qa_summary_path = "./logs/qa/contributor_ownership_profile_qa_summary.csv"

    return {
        "identity_resolution_mode": mode,
        "repo_output_path": Path(repo_output_path),
        "file_output_path": Path(file_output_path),
        "qa_summary_path": Path(qa_summary_path),
        "run_manifest_path": Path(config.logging.qa_log_dir) / runtime_names["run_manifest_filename"],
    }


def build_repo_id_lookup(config):
    merge_mode = getattr(config.storage, "processed_merge_mode", "single_parquet")
    repos_df = load_table(config.outputs.repositories_table, merge_mode=merge_mode)
    if repos_df.empty:
        return {}

    repo_name_col = "repo_full_name" if "repo_full_name" in repos_df.columns else "full_name" if "full_name" in repos_df.columns else None
    repo_id_col = "repo_id" if "repo_id" in repos_df.columns else "id" if "id" in repos_df.columns else None
    if repo_name_col is None or repo_id_col is None:
        return {}

    lookup = {}
    for row in repos_df.to_dict(orient="records"):
        repo_name = clean_text(row.get(repo_name_col))
        repo_id = row.get(repo_id_col)
        if repo_name and pd.notna(repo_id):
            lookup[repo_name] = repo_id
    return lookup


def load_stage_inputs_for_repo(config, repo_full_name):
    merge_mode = getattr(config.storage, "processed_merge_mode", "single_parquet")
    mode = get_contributor_profile_identity_mode(config)

    commits_resolved_path = (
        getattr(config.outputs, "commits_resolved_table_fuzzy")
        if mode == "fuzzy"
        else config.outputs.commits_resolved_table
    )

    commits_resolved_df = load_table(commits_resolved_path, repo_full_name=repo_full_name, merge_mode=merge_mode)
    commit_files_df = load_table(config.outputs.commit_files_table, repo_full_name=repo_full_name, merge_mode=merge_mode)

    return {
        "commits_resolved": repo_filter(commits_resolved_df, repo_full_name),
        "commit_files": repo_filter(commit_files_df, repo_full_name),
    }


def new_repo_result(repo_full_name, repo_id=None):
    return {
        "repo_full_name": repo_full_name,
        "repo_id": repo_id,
        "identity_resolution_mode": None,
        "status": "started",
        "commits_resolved_rows_seen": 0,
        "commits_resolved_rows_with_contributor_key": 0,
        "commit_files_rows_seen": 0,
        "joined_commit_file_rows": 0,
        "distinct_commit_shas_in_resolved": 0,
        "distinct_commit_shas_joined_to_files": 0,
        "share_commit_shas_joined_to_commit_files": 0.0,
        "contributors_with_any_profile": 0,
        "contributors_with_file_profile": 0,
        "repo_contributor_profile_rows_written": 0,
        "file_contributor_profile_rows_written": 0,
        "top_contributors_by_commit_count": "",
        "top_files_by_distinct_contributors": "",
        "error_message": "",
    }


def prepare_commits_resolved_frame(commits_resolved_df):
    if commits_resolved_df is None or commits_resolved_df.empty:
        return pd.DataFrame(
            columns=[
                "repo_id",
                "repo_full_name",
                "commit_sha",
                "commit_author_contributor_key",
                "commit_timestamp",
            ]
        )

    df = commits_resolved_df.copy()
    if "commit_sha" not in df.columns or "repo_full_name" not in df.columns:
        return pd.DataFrame(
            columns=[
                "repo_id",
                "repo_full_name",
                "commit_sha",
                "commit_author_contributor_key",
                "commit_timestamp",
            ]
        )

    if "repo_id" not in df.columns:
        df["repo_id"] = None
    if "commit_author_contributor_key" not in df.columns:
        df["commit_author_contributor_key"] = None
    if "commit_timestamp" not in df.columns:
        df["commit_timestamp"] = pd.NaT

    df["repo_full_name"] = df["repo_full_name"].astype(str)
    df["commit_sha"] = df["commit_sha"].apply(clean_text)
    df["commit_author_contributor_key"] = df["commit_author_contributor_key"].apply(clean_text)
    df["commit_timestamp"] = safe_to_datetime(df["commit_timestamp"])

    keep_cols = [
        col for col in [
            "repo_id",
            "repo_full_name",
            "commit_sha",
            "commit_author_contributor_key",
            "commit_timestamp",
        ]
        if col in df.columns
    ]
    df = df[keep_cols].copy()
    df = df[df["commit_sha"].notna()].copy()
    df = df[df["commit_author_contributor_key"].notna()].copy()
    df = df.drop_duplicates(subset=["repo_full_name", "commit_sha"], keep="first").reset_index(drop=True)
    return df


def prepare_commit_files_frame(commit_files_df):
    if commit_files_df is None or commit_files_df.empty:
        return pd.DataFrame(columns=["repo_full_name", "commit_sha", "file_path", "additions", "deletions"])

    df = commit_files_df.copy()
    if "repo_full_name" not in df.columns or "commit_sha" not in df.columns:
        return pd.DataFrame(columns=["repo_full_name", "commit_sha", "file_path", "additions", "deletions"])

    df["repo_full_name"] = df["repo_full_name"].astype(str)
    df["commit_sha"] = df["commit_sha"].apply(clean_text)

    file_path_col = None
    for candidate in ["file_path", "path", "filename"]:
        if candidate in df.columns:
            file_path_col = candidate
            break
    if file_path_col is None:
        df["file_path"] = None
    elif file_path_col != "file_path":
        df = df.rename(columns={file_path_col: "file_path"})

    if "additions" not in df.columns:
        df["additions"] = pd.NA
    if "deletions" not in df.columns:
        df["deletions"] = pd.NA

    df["file_path"] = df["file_path"].apply(clean_text)
    df["additions"] = pd.to_numeric(df["additions"], errors="coerce")
    df["deletions"] = pd.to_numeric(df["deletions"], errors="coerce")

    keep_cols = ["repo_full_name", "commit_sha", "file_path", "additions", "deletions"]
    df = df[keep_cols].copy()
    df = df[df["commit_sha"].notna()].copy()
    return df.reset_index(drop=True)


def build_commit_file_join(commits_resolved_df, commit_files_df):
    if commits_resolved_df.empty or commit_files_df.empty:
        return pd.DataFrame(
            columns=[
                "repo_id",
                "repo_full_name",
                "commit_sha",
                "commit_author_contributor_key",
                "commit_timestamp",
                "file_path",
                "additions",
                "deletions",
            ]
        )

    joined_df = commits_resolved_df.merge(
        commit_files_df,
        on=["repo_full_name", "commit_sha"],
        how="inner",
        validate="one_to_many",
    )
    return joined_df.reset_index(drop=True)


def _series_iso(value):
    if pd.isna(value):
        return None
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def _build_top_pairs_as_string(df, label_col, value_col, limit=5):
    if df is None or df.empty or label_col not in df.columns or value_col not in df.columns:
        return ""
    top_rows = df.sort_values([value_col, label_col], ascending=[False, True], kind="stable").head(limit)
    pairs = []
    for row in top_rows.to_dict(orient="records"):
        label = clean_text(row.get(label_col)) or "<missing>"
        value = row.get(value_col)
        try:
            rendered = int(value)
        except Exception:
            rendered = value
        pairs.append(f"{label}:{rendered}")
    return " | ".join(pairs)


def build_repo_contributor_profiles(joined_commit_file_df, commits_resolved_df, identity_resolution_mode, config):
    columns = [
        "repo_id",
        "repo_full_name",
        "resolved_contributor_key",
        "identity_resolution_mode",
        "commit_count_total",
        "distinct_files_touched_total",
        "first_commit_timestamp",
        "last_commit_timestamp",
        "commit_days_active",
        "median_files_per_commit",
        "lines_added_total",
        "lines_deleted_total",
        "share_of_repo_commits",
        "share_of_repo_file_touches",
        "dominant_repo_contributor_flag",
        "top_k_repo_contributor_rank",
    ]
    if commits_resolved_df.empty:
        return pd.DataFrame(columns=columns)

    commit_df = commits_resolved_df.copy()
    commit_df = commit_df.rename(columns={"commit_author_contributor_key": "resolved_contributor_key"})
    commit_df["commit_day"] = commit_df["commit_timestamp"].dt.floor("D")

    commit_counts = (
        commit_df.groupby(["repo_full_name", "resolved_contributor_key"], dropna=False)["commit_sha"]
        .nunique()
        .reset_index(name="commit_count_total")
    )
    timestamp_stats = (
        commit_df.groupby(["repo_full_name", "resolved_contributor_key"], dropna=False)
        .agg(
            first_commit_timestamp=("commit_timestamp", "min"),
            last_commit_timestamp=("commit_timestamp", "max"),
            commit_days_active=("commit_day", "nunique"),
            repo_id=("repo_id", "first"),
        )
        .reset_index()
    )

    if joined_commit_file_df.empty:
        distinct_files = commit_counts[["repo_full_name", "resolved_contributor_key"]].copy()
        distinct_files["distinct_files_touched_total"] = 0
        median_files = distinct_files[["repo_full_name", "resolved_contributor_key"]].copy()
        median_files["median_files_per_commit"] = 0.0
        line_stats = distinct_files[["repo_full_name", "resolved_contributor_key"]].copy()
        line_stats["lines_added_total"] = pd.NA
        line_stats["lines_deleted_total"] = pd.NA
        file_touch_counts = distinct_files[["repo_full_name", "resolved_contributor_key"]].copy()
        file_touch_counts["file_touch_events_total"] = 0
    else:
        joined = joined_commit_file_df.copy()
        joined = joined.rename(columns={"commit_author_contributor_key": "resolved_contributor_key"})
        distinct_files = (
            joined[joined["file_path"].notna()]
            .groupby(["repo_full_name", "resolved_contributor_key"], dropna=False)["file_path"]
            .nunique()
            .reset_index(name="distinct_files_touched_total")
        )
        per_commit_file_counts = (
            joined[joined["file_path"].notna()]
            .groupby(["repo_full_name", "resolved_contributor_key", "commit_sha"], dropna=False)["file_path"]
            .nunique()
            .reset_index(name="files_touched_in_commit")
        )
        median_files = (
            per_commit_file_counts.groupby(["repo_full_name", "resolved_contributor_key"], dropna=False)["files_touched_in_commit"]
            .median()
            .reset_index(name="median_files_per_commit")
        )
        line_stats = (
            joined.groupby(["repo_full_name", "resolved_contributor_key"], dropna=False)
            .agg(lines_added_total=("additions", "sum"), lines_deleted_total=("deletions", "sum"))
            .reset_index()
        )
        file_touch_counts = (
            joined[joined["file_path"].notna()]
            .groupby(["repo_full_name", "resolved_contributor_key"], dropna=False)
            .size()
            .reset_index(name="file_touch_events_total")
        )

    repo_profiles = commit_counts.merge(
        timestamp_stats,
        on=["repo_full_name", "resolved_contributor_key"],
        how="left",
    )
    repo_profiles = repo_profiles.merge(
        distinct_files,
        on=["repo_full_name", "resolved_contributor_key"],
        how="left",
    )
    repo_profiles = repo_profiles.merge(
        median_files,
        on=["repo_full_name", "resolved_contributor_key"],
        how="left",
    )
    repo_profiles = repo_profiles.merge(
        line_stats,
        on=["repo_full_name", "resolved_contributor_key"],
        how="left",
    )
    repo_profiles = repo_profiles.merge(
        file_touch_counts,
        on=["repo_full_name", "resolved_contributor_key"],
        how="left",
    )

    repo_totals = (
        repo_profiles.groupby("repo_full_name", dropna=False)
        .agg(
            repo_commit_total=("commit_count_total", "sum"),
            repo_file_touch_total=("file_touch_events_total", "sum"),
        )
        .reset_index()
    )
    repo_profiles = repo_profiles.merge(repo_totals, on="repo_full_name", how="left")
    repo_profiles["share_of_repo_commits"] = repo_profiles.apply(
        lambda row: safe_divide(row.get("commit_count_total"), row.get("repo_commit_total"), default_value=0.0),
        axis=1,
    )
    repo_profiles["share_of_repo_file_touches"] = repo_profiles.apply(
        lambda row: safe_divide(row.get("file_touch_events_total"), row.get("repo_file_touch_total"), default_value=0.0),
        axis=1,
    )
    repo_profiles["top_k_repo_contributor_rank"] = (
        repo_profiles.groupby("repo_full_name", dropna=False)["commit_count_total"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    rank_threshold = int(get_contributor_profile_option(config, "pre_issue_repo_major_contributor_rank_threshold", 5))
    share_threshold = float(get_contributor_profile_option(config, "pre_issue_repo_major_contributor_share_threshold", 0.05))
    repo_profiles["dominant_repo_contributor_flag"] = (
        (repo_profiles["top_k_repo_contributor_rank"] <= rank_threshold)
        | (repo_profiles["share_of_repo_commits"] >= share_threshold)
    ).astype(int)

    repo_profiles["distinct_files_touched_total"] = repo_profiles["distinct_files_touched_total"].fillna(0).astype(int)
    repo_profiles["commit_days_active"] = repo_profiles["commit_days_active"].fillna(0).astype(int)
    repo_profiles["median_files_per_commit"] = repo_profiles["median_files_per_commit"].fillna(0.0)
    repo_profiles["identity_resolution_mode"] = identity_resolution_mode
    repo_profiles["first_commit_timestamp"] = repo_profiles["first_commit_timestamp"].apply(_series_iso)
    repo_profiles["last_commit_timestamp"] = repo_profiles["last_commit_timestamp"].apply(_series_iso)

    repo_profiles = repo_profiles.rename(columns={"repo_full_name": "repo_full_name"})
    repo_profiles = repo_profiles.sort_values(
        ["repo_full_name", "top_k_repo_contributor_rank", "resolved_contributor_key"],
        kind="stable",
    ).reset_index(drop=True)
    return repo_profiles[columns]


def build_file_contributor_profiles(joined_commit_file_df, identity_resolution_mode, config):
    columns = [
        "repo_id",
        "repo_full_name",
        "resolved_contributor_key",
        "file_path",
        "identity_resolution_mode",
        "commit_count_on_file",
        "first_commit_timestamp_on_file",
        "last_commit_timestamp_on_file",
        "distinct_commit_days_on_file",
        "share_of_file_commits",
        "file_owner_rank",
        "major_file_contributor_flag",
    ]
    if joined_commit_file_df.empty:
        return pd.DataFrame(columns=columns)

    joined = joined_commit_file_df.copy()
    joined = joined.rename(columns={"commit_author_contributor_key": "resolved_contributor_key"})
    joined = joined[joined["file_path"].notna()].copy()
    if joined.empty:
        return pd.DataFrame(columns=columns)

    joined["commit_day"] = joined["commit_timestamp"].dt.floor("D")
    file_profiles = (
        joined.groupby(["repo_full_name", "file_path", "resolved_contributor_key"], dropna=False)
        .agg(
            repo_id=("repo_id", "first"),
            commit_count_on_file=("commit_sha", "nunique"),
            first_commit_timestamp_on_file=("commit_timestamp", "min"),
            last_commit_timestamp_on_file=("commit_timestamp", "max"),
            distinct_commit_days_on_file=("commit_day", "nunique"),
        )
        .reset_index()
    )

    file_totals = (
        file_profiles.groupby(["repo_full_name", "file_path"], dropna=False)["commit_count_on_file"]
        .sum()
        .reset_index(name="file_commit_total")
    )
    file_profiles = file_profiles.merge(file_totals, on=["repo_full_name", "file_path"], how="left")
    file_profiles["share_of_file_commits"] = file_profiles.apply(
        lambda row: safe_divide(row.get("commit_count_on_file"), row.get("file_commit_total"), default_value=0.0),
        axis=1,
    )
    file_profiles["file_owner_rank"] = (
        file_profiles.groupby(["repo_full_name", "file_path"], dropna=False)["commit_count_on_file"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    rank_threshold = int(get_contributor_profile_option(config, "pre_issue_file_major_contributor_rank_threshold", 3))
    share_threshold = float(get_contributor_profile_option(config, "pre_issue_file_major_contributor_share_threshold", 0.10))
    file_profiles["major_file_contributor_flag"] = (
        (file_profiles["file_owner_rank"] <= rank_threshold)
        | (file_profiles["share_of_file_commits"] >= share_threshold)
    ).astype(int)

    file_profiles["identity_resolution_mode"] = identity_resolution_mode
    file_profiles["first_commit_timestamp_on_file"] = file_profiles["first_commit_timestamp_on_file"].apply(_series_iso)
    file_profiles["last_commit_timestamp_on_file"] = file_profiles["last_commit_timestamp_on_file"].apply(_series_iso)
    file_profiles = file_profiles.sort_values(
        ["repo_full_name", "file_path", "file_owner_rank", "resolved_contributor_key"],
        kind="stable",
    ).reset_index(drop=True)
    return file_profiles[columns]


def process_repo(config, logger, repo_row, repo_id_lookup):
    repo_full_name = repo_row["full_name"]
    result = new_repo_result(repo_full_name, repo_id=repo_row.get("repo_id") or repo_id_lookup.get(repo_full_name))
    identity_mode = get_contributor_profile_identity_mode(config)
    result["identity_resolution_mode"] = identity_mode

    stage_inputs = load_stage_inputs_for_repo(config, repo_full_name)
    commits_resolved_df = prepare_commits_resolved_frame(stage_inputs["commits_resolved"])
    commit_files_df = prepare_commit_files_frame(stage_inputs["commit_files"])
    joined_df = build_commit_file_join(commits_resolved_df, commit_files_df)

    result["commits_resolved_rows_seen"] = int(len(stage_inputs["commits_resolved"]))
    result["commits_resolved_rows_with_contributor_key"] = int(len(commits_resolved_df))
    result["commit_files_rows_seen"] = int(len(stage_inputs["commit_files"]))
    result["joined_commit_file_rows"] = int(len(joined_df))
    result["distinct_commit_shas_in_resolved"] = int(commits_resolved_df["commit_sha"].nunique()) if not commits_resolved_df.empty else 0
    result["distinct_commit_shas_joined_to_files"] = int(joined_df["commit_sha"].nunique()) if not joined_df.empty else 0
    result["share_commit_shas_joined_to_commit_files"] = safe_divide(
        result["distinct_commit_shas_joined_to_files"],
        result["distinct_commit_shas_in_resolved"],
        default_value=0.0,
    )

    repo_profiles_df = build_repo_contributor_profiles(joined_df, commits_resolved_df, identity_mode, config)
    file_profiles_df = build_file_contributor_profiles(joined_df, identity_mode, config)

    result["contributors_with_any_profile"] = int(repo_profiles_df["resolved_contributor_key"].nunique()) if not repo_profiles_df.empty else 0
    result["contributors_with_file_profile"] = int(file_profiles_df["resolved_contributor_key"].nunique()) if not file_profiles_df.empty else 0
    result["top_contributors_by_commit_count"] = _build_top_pairs_as_string(repo_profiles_df, "resolved_contributor_key", "commit_count_total", limit=5)

    if file_profiles_df.empty:
        result["top_files_by_distinct_contributors"] = ""
    else:
        top_files_df = (
            file_profiles_df.groupby("file_path", dropna=False)["resolved_contributor_key"]
            .nunique()
            .reset_index(name="distinct_contributors")
        )
        result["top_files_by_distinct_contributors"] = _build_top_pairs_as_string(
            top_files_df,
            "file_path",
            "distinct_contributors",
            limit=5,
        )

    runtime_names = get_contributor_profile_runtime_names(config)
    batch_root = get_batch_root(config, runtime_names["batch_folder_name"])
    repo_dir = batch_root / sanitize_repo_name(repo_full_name)
    writer = ContributorOwnershipProfileRepoChunkWriter(
        config=config,
        repo_dir=repo_dir,
        batch_size=get_contributor_profile_option(config, "write_batch_size", 5000),
    )

    for row in repo_profiles_df.to_dict(orient="records"):
        writer.add_repo_profile_row(row)
        result["repo_contributor_profile_rows_written"] += 1
    for row in file_profiles_df.to_dict(orient="records"):
        writer.add_file_profile_row(row)
        result["file_contributor_profile_rows_written"] += 1
    writer.finalize()

    result["status"] = "completed"
    return result


def merge_contributor_profile_batches(config, logger, stage_paths):
    runtime_names = get_contributor_profile_runtime_names(config)
    batch_root = get_batch_root(config, runtime_names["batch_folder_name"])
    if not batch_root.exists():
        logger.warning("Contributor ownership profile batch root does not exist: %s", batch_root)
        return

    repo_part_map = collect_repo_part_files(batch_root, "contributor_repo_ownership_profiles_part_*.parquet")
    file_part_map = collect_repo_part_files(batch_root, "contributor_file_ownership_profiles_part_*.parquet")

    if repo_part_map:
        mode_used = write_merged_or_partitioned_output(
            repo_part_map=repo_part_map,
            output_path=stage_paths["repo_output_path"],
            config=config,
            table_name="contributor_repo_ownership_profiles",
            sort_columns=["repo_full_name", "top_k_repo_contributor_rank", "resolved_contributor_key"],
            dedupe_subset=["repo_full_name", "resolved_contributor_key"],
        )
        logger.info(
            "Wrote contributor repo ownership profiles using %s mode to %s",
            mode_used,
            stage_paths["repo_output_path"],
        )
    else:
        logger.warning("No contributor repo ownership profile parts found to merge.")

    if file_part_map:
        mode_used = write_merged_or_partitioned_output(
            repo_part_map=file_part_map,
            output_path=stage_paths["file_output_path"],
            config=config,
            table_name="contributor_file_ownership_profiles",
            sort_columns=["repo_full_name", "file_path", "file_owner_rank", "resolved_contributor_key"],
            dedupe_subset=["repo_full_name", "resolved_contributor_key", "file_path"],
        )
        logger.info(
            "Wrote contributor file ownership profiles using %s mode to %s",
            mode_used,
            stage_paths["file_output_path"],
        )
    else:
        logger.warning("No contributor file ownership profile parts found to merge.")


def write_run_manifest(repo_rows, summary_rows, stage_paths):
    manifest_path = Path(stage_paths["run_manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "11_a_build_contributor_ownership_profiles.py",
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
    stage_paths = get_contributor_profile_stage_paths(config)
    runtime_names = get_contributor_profile_runtime_names(config)
    repo_id_lookup = build_repo_id_lookup(config)

    repo_rows = load_repo_list(config.outputs.repo_included_list)
    batch_root = reset_batch_root(config, runtime_names["batch_folder_name"])
    logger.info("Reset batch root: %s", batch_root)

    max_repos_per_run = get_contributor_profile_option(config, "max_repos_per_run", None)
    if max_repos_per_run and max_repos_per_run > 0:
        repo_rows = repo_rows[:max_repos_per_run]

    summary_rows = []
    for repo_row in repo_rows:
        repo_full_name = repo_row["full_name"]
        skip_repo, reason = should_skip_repo(
            config,
            repo_full_name,
            checkpoint_prefix=runtime_names["checkpoint_prefix"],
            raw_folder_name=runtime_names["raw_folder_name"],
            section_name="ownership_features",
            raw_source="features",
        )
        if skip_repo:
            logger.info("Skipping %s due to %s", repo_full_name, reason)
            summary_rows.append(
                {
                    "repo_full_name": repo_full_name,
                    "repo_id": repo_row.get("repo_id"),
                    "identity_resolution_mode": stage_paths.get("identity_resolution_mode"),
                    "status": f"skipped_{reason}",
                }
            )
            continue

        try:
            logger.info(
                "Starting contributor ownership profile build for %s | identity_mode=%s",
                repo_full_name,
                stage_paths.get("identity_resolution_mode"),
            )
            result = process_repo(config, logger, repo_row, repo_id_lookup)
            summary_rows.append(result)
            write_repo_checkpoint(config, runtime_names["checkpoint_prefix"], repo_full_name, result)
        except Exception as exc:
            logger.exception("Contributor ownership profile build failed for %s", repo_full_name)
            error_row = new_repo_result(repo_full_name, repo_row.get("repo_id") or repo_id_lookup.get(repo_full_name))
            error_row["identity_resolution_mode"] = stage_paths.get("identity_resolution_mode")
            error_row["status"] = "failed"
            error_row["error_message"] = str(exc)
            summary_rows.append(error_row)
            write_repo_checkpoint(config, runtime_names["checkpoint_prefix"], repo_full_name, error_row)

    merge_contributor_profile_batches(config, logger, stage_paths)
    write_summary_csv(summary_rows, stage_paths["qa_summary_path"])
    write_run_manifest(repo_rows, summary_rows, stage_paths)
    logger.info("Contributor ownership profile build complete.")


if __name__ == "__main__":
    main()
