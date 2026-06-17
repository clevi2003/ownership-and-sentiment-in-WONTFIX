from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.study_config_loader import load_study_config


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"

DEFAULT_WONTFIX_LABELS = [
    "wontfix",
    "won't fix",
    "won’t fix",
    "not planned",
    "declined",
]

DEFAULT_INVALID_LABELS = [
    "invalid",
    "question",
    "works as intended",
    "duplicate",
]

DEFAULT_ISSUE_TYPE_MAP = {
    "bug": ["bug", "type: bug", "kind/bug", "kind: bug"],
    "feature": ["enhancement", "feature", "feature request", "type: feature", "kind/feature", "kind: feature"],
    "documentation": ["documentation", "docs", "type: docs", "kind/documentation", "kind: documentation"],
    "question": ["question", "support", "type: question"],
}


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def get_cfg(cfg: Any, *path: str, default: Any = None) -> Any:
    cur = cfg
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def find_column(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    for col in df.columns:
        col_l = str(col).lower()
        for cand in candidates:
            if cand.lower() in col_l:
                return col

    if required:
        raise KeyError(f"Could not find required column. Tried: {candidates}")
    return None


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>", "nat"}:
        return None
    return text


def normalize_text_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="object")
    return series.apply(clean_text)


def normalize_lower_text_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="object")
    return normalize_text_series(series).apply(lambda x: x.lower() if x else None)


def normalize_datetime(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    return pd.to_datetime(series, errors="coerce", utc=True)


def normalize_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def parse_label_value(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        return [str(x).strip().lower() for x in value if str(x).strip()]

    if isinstance(value, dict):
        return [str(k).strip().lower() for k in value.keys() if str(k).strip()]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return [str(x).strip().lower() for x in parsed if str(x).strip()]
            except Exception:
                pass

    parts = re.split(r"[|,;]", text)
    if len(parts) > 1:
        return [p.strip().lower() for p in parts if p.strip()]

    return [text.lower()]


def contains_any(labels: list[str], keywords: list[str]) -> bool:
    label_blob = " | ".join(labels)
    for kw in keywords:
        kw_l = str(kw).lower().strip()
        if kw_l and kw_l in label_blob:
            return True
    return False


def infer_issue_type(labels: list[str], issue_type_map: dict[str, list[str]]) -> str | None:
    for issue_type, aliases in issue_type_map.items():
        if contains_any(labels, aliases):
            return issue_type
    return None


def parse_boolish_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    raw_num = pd.to_numeric(series, errors="coerce")
    if raw_num.notna().any():
        return raw_num.fillna(0).astype(float) > 0

    normalized = series.apply(clean_text).astype("object")
    return normalized.apply(lambda x: str(x).strip().lower() in {"true", "1", "yes", "y", "t"} if x else False)


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=suffix in {".jsonl", ".ndjson"})
    raise ValueError(f"Unsupported table format for {path}. Use .parquet, .csv, .json, .jsonl, or .ndjson.")


def write_table(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".json":
        df.to_json(path, orient="records", indent=2)
        return
    if suffix in {".jsonl", ".ndjson"}:
        df.to_json(path, orient="records", lines=True)
        return
    # Default project format is parquet.
    df.to_parquet(path, index=False)


# -----------------------------------------------------------------------------
# Config/path helpers
# -----------------------------------------------------------------------------


def pick_paths(config: Any) -> tuple[Path, Path, Path, Path, Path, Path | None]:
    issues_path = Path(
        get_cfg(config, "outputs", "issues_table")
        or get_cfg(config, "issues_table")
        or "data/processed/issues.parquet"
    )

    wontfix_output_path = Path(
        get_cfg(config, "outputs", "wontfix_issue_set_table")
        or get_cfg(config, "outputs", "wontfix_issue_set")
        or "data/final/wontfix_issue_set.parquet"
    )

    comparison_output_path = Path(
        get_cfg(config, "outputs", "comparison_issue_set_table")
        or get_cfg(config, "outputs", "comparison_issue_set")
        or "data/final/comparison_issue_set.parquet"
    )

    qa_summary_path = Path(
        get_cfg(config, "outputs", "comparison_issue_qa_summary_csv")
        or get_cfg(config, "outputs", "comparison_issue_set_qa_summary_csv")
        or "logs/comparison_issue_qa_summary.csv"
    )

    pair_output_path = Path(
        get_cfg(config, "outputs", "wontfix_comparison_pairs_table")
        or get_cfg(config, "outputs", "wontfix_comparison_pairs")
        or "data/processed/wontfix_comparison_pairs.parquet"
    )

    issue_pr_links_path_raw = (
        get_cfg(config, "outputs", "issue_pr_links_table")
        or get_cfg(config, "issue_pr_links_table")
    )
    issue_pr_links_path = Path(issue_pr_links_path_raw) if issue_pr_links_path_raw else None

    return (
        issues_path,
        wontfix_output_path,
        comparison_output_path,
        qa_summary_path,
        pair_output_path,
        issue_pr_links_path,
    )


def resolve_matching_settings(config: Any) -> tuple[int, int]:
    max_controls = (
        get_cfg(config, "comparison_set", "matching_rules", "max_controls_per_wontfix")
        or get_cfg(config, "comparison_set", "max_controls_per_wontfix")
        or get_cfg(config, "comparison_set", "max_controls")
        or 3
    )
    time_window_days = (
        get_cfg(config, "comparison_set", "time_window_days")
        or get_cfg(config, "comparison_set", "matching_time_window_days")
        or 180
    )
    return int(max_controls), int(time_window_days)


def _unique_lowered(values: list[str]) -> list[str]:
    cleaned = []
    seen = set()
    for value in values:
        text = str(value).strip().lower()
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    return cleaned


def resolve_label_settings(config: Any) -> tuple[list[str], list[str], dict[str, list[str]]]:
    wontfix_labels = (
        get_cfg(config, "label_normalization", "outcome_labels", "wontfix", "variants")
        or get_cfg(config, "comparison_set", "wontfix_label_variants")
        or get_cfg(config, "labels", "wontfix")
        or DEFAULT_WONTFIX_LABELS
    )

    invalid_labels = (
        get_cfg(config, "label_normalization", "outcome_labels", "invalid", "variants")
        or get_cfg(config, "comparison_set", "invalid_label_variants")
        or get_cfg(config, "labels", "invalid")
        or DEFAULT_INVALID_LABELS
    )

    issue_type_map = get_cfg(config, "comparison_set", "issue_type_label_groups")
    if not issue_type_map:
        issue_type_map = {
            "bug": get_cfg(config, "label_normalization", "issue_type_labels", "bug", "variants", default=[]),
            "feature": get_cfg(config, "label_normalization", "issue_type_labels", "feature_request", "variants", default=[]),
            "documentation": get_cfg(config, "label_normalization", "issue_type_labels", "documentation", "variants", default=[]),
            "question": get_cfg(config, "label_normalization", "issue_type_labels", "question", "variants", default=[]),
        }

    if not any(dict(issue_type_map).values()):
        issue_type_map = get_cfg(config, "issue_types") or DEFAULT_ISSUE_TYPE_MAP

    normalized_issue_type_map = {
        str(key): _unique_lowered(list(values or []))
        for key, values in dict(issue_type_map).items()
    }

    return (
        _unique_lowered(list(wontfix_labels)),
        _unique_lowered(list(invalid_labels)),
        normalized_issue_type_map,
    )


# -----------------------------------------------------------------------------
# Feature derivation and matching
# -----------------------------------------------------------------------------


def add_derived_columns(
    issues_df: pd.DataFrame,
    wontfix_labels: list[str],
    invalid_labels: list[str],
    issue_type_map: dict[str, list[str]],
    issue_pr_links_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    df = issues_df.copy()

    colmap = {
        "issue_id": find_column(df, ["issue_id", "id", "node_id", "number"], required=True),
        "repo": find_column(df, ["repo_full_name", "repo_name", "repository_name", "repo", "full_name"], required=True),
        "created_at": find_column(df, ["created_at", "issue_created_at", "created"], required=True),
        "closed_at": find_column(df, ["closed_at", "issue_closed_at", "closed"]),
        "state": find_column(df, ["state", "issue_state", "status"]),
        "labels": find_column(df, ["label_names_json", "labels", "label_names", "issue_labels"], required=True),
        "comment_count": find_column(df, ["comments", "comment_count", "num_comments", "comments_count"]),
        "linked_pr": find_column(df, ["has_linked_pr", "linked_pr", "linked_pull_request", "has_pr", "pull_request"]),
        "issue_type": find_column(df, ["issue_type", "type"]),
        "issue_number": find_column(df, ["issue_number", "number"]),
        "is_wontfix_labeled": find_column(df, ["is_wontfix_labeled"]),
    }

    df["__issue_id"] = df[colmap["issue_id"]].apply(clean_text).astype(str)
    df["__repo"] = df[colmap["repo"]].apply(clean_text).astype(str)
    df["__issue_key"] = df["__repo"] + "::" + df["__issue_id"]

    if colmap["issue_number"]:
        df["__issue_number"] = pd.to_numeric(df[colmap["issue_number"]], errors="coerce")
    else:
        df["__issue_number"] = pd.NA

    df["__created_at"] = normalize_datetime(df[colmap["created_at"]])
    df["__closed_at"] = normalize_datetime(df[colmap["closed_at"]]) if colmap["closed_at"] else pd.NaT
    df["__state"] = normalize_lower_text_series(df[colmap["state"]]) if colmap["state"] else None
    df["__comment_count"] = normalize_numeric(df[colmap["comment_count"]]).fillna(0) if colmap["comment_count"] else 0
    df["__labels_list"] = df[colmap["labels"]].apply(parse_label_value)

    if colmap["is_wontfix_labeled"]:
        df["__is_wontfix"] = parse_boolish_series(df[colmap["is_wontfix_labeled"]])
    else:
        df["__is_wontfix"] = df["__labels_list"].apply(lambda x: contains_any(x, wontfix_labels))

    df["__is_invalid"] = df["__labels_list"].apply(lambda x: contains_any(x, invalid_labels))

    if colmap["issue_type"]:
        df["__issue_type"] = normalize_lower_text_series(df[colmap["issue_type"]])
    else:
        df["__issue_type"] = df["__labels_list"].apply(lambda x: infer_issue_type(x, issue_type_map))

    if colmap["linked_pr"]:
        df["__has_linked_pr"] = parse_boolish_series(df[colmap["linked_pr"]])
    else:
        df["__has_linked_pr"] = False

    # Prefer explicit issue-PR link evidence when the link table exists. This keeps
    # PR-resolved controls visible even if the issues table itself lacks a flag.
    if issue_pr_links_df is not None and not issue_pr_links_df.empty:
        links_df = issue_pr_links_df.copy()
        repo_link_col = find_column(links_df, ["repo_full_name", "repo_name", "repo", "full_name"])
        issue_number_link_col = find_column(links_df, ["issue_number", "number"])
        issue_id_link_col = find_column(links_df, ["issue_id", "id", "node_id"])

        if repo_link_col and issue_number_link_col and colmap["issue_number"]:
            repo_series = links_df[repo_link_col].apply(clean_text).astype(str)
            issue_number_series = pd.to_numeric(links_df[issue_number_link_col], errors="coerce")
            valid = issue_number_series.notna()
            linked_issue_keys = set(zip(repo_series[valid], issue_number_series[valid].astype(int)))

            issue_number_values = pd.to_numeric(df[colmap["issue_number"]], errors="coerce")
            has_link = [
                (repo, int(issue_num)) in linked_issue_keys if pd.notna(issue_num) else False
                for repo, issue_num in zip(df["__repo"], issue_number_values)
            ]
            df["__has_linked_pr"] = df["__has_linked_pr"] | pd.Series(has_link, index=df.index)

        elif repo_link_col and issue_id_link_col:
            linked_issue_id_keys = set(
                zip(
                    links_df[repo_link_col].apply(clean_text).astype(str),
                    links_df[issue_id_link_col].apply(clean_text).astype(str),
                )
            )
            has_link = [
                (repo, issue_id) in linked_issue_id_keys
                for repo, issue_id in zip(df["__repo"], df["__issue_id"])
            ]
            df["__has_linked_pr"] = df["__has_linked_pr"] | pd.Series(has_link, index=df.index)

    df["__is_open"] = df["__state"].eq("open") if "__state" in df else False
    df["__is_closed"] = df["__state"].eq("closed") | df["__closed_at"].notna() if "__state" in df else df["__closed_at"].notna()

    # IMPORTANT: assign lower-priority buckets first and PR-resolved last.
    # The previous version assigned resolved_pr and then overwrote all closed
    # non-WONTFIX rows as closed_non_wontfix, which made selected_resolved_pr
    # incorrectly appear as zero in QA summaries.
    df["__comparison_bucket"] = "other"
    non_wontfix = ~df["__is_wontfix"]
    df.loc[df["__is_open"] & non_wontfix, "__comparison_bucket"] = "open"
    df.loc[df["__is_closed"] & non_wontfix, "__comparison_bucket"] = "closed_non_wontfix"
    df.loc[df["__is_invalid"] & non_wontfix, "__comparison_bucket"] = "invalid"
    df.loc[df["__is_closed"] & non_wontfix & df["__has_linked_pr"], "__comparison_bucket"] = "resolved_pr"

    return df, colmap


def candidate_score(wf_row: pd.Series, cand_row: pd.Series) -> float:
    score = 0.0

    if pd.notna(wf_row["__created_at"]) and pd.notna(cand_row["__created_at"]):
        day_diff = abs((cand_row["__created_at"] - wf_row["__created_at"]).days)
        score += min(day_diff, 3650) / 30.0
    else:
        score += 50

    comment_diff = abs(float(cand_row["__comment_count"]) - float(wf_row["__comment_count"]))
    score += min(comment_diff, 100)

    if wf_row["__issue_type"] and cand_row["__issue_type"] and wf_row["__issue_type"] == cand_row["__issue_type"]:
        score -= 15

    if bool(cand_row["__has_linked_pr"]):
        score -= 8

    bucket_bonus = {
        "resolved_pr": -12,
        "closed_non_wontfix": -8,
        "invalid": -3,
        "open": 0,
        "other": 3,
    }
    score += bucket_bonus.get(cand_row["__comparison_bucket"], 3)

    return float(score)


def add_pair_diagnostics(candidates: pd.DataFrame, wf_row: pd.Series) -> pd.DataFrame:
    candidates = candidates.copy()
    if pd.notna(wf_row["__created_at"]):
        candidates["__created_at_day_diff"] = (candidates["__created_at"] - wf_row["__created_at"]).abs().dt.days
    else:
        candidates["__created_at_day_diff"] = pd.NA
    candidates["__comment_count_abs_diff"] = (candidates["__comment_count"].astype(float) - float(wf_row["__comment_count"])).abs()
    candidates["__same_issue_type_flag"] = (
        candidates["__issue_type"].notna()
        & pd.notna(wf_row["__issue_type"])
        & candidates["__issue_type"].eq(wf_row["__issue_type"])
    ).astype(int)
    return candidates


def select_comparison_controls(
    derived_df: pd.DataFrame,
    max_controls_per_wontfix: int,
    time_window_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wontfix_df = derived_df[derived_df["__is_wontfix"]].copy()
    non_wontfix_df = derived_df[~derived_df["__is_wontfix"]].copy()

    selected_rows: list[pd.Series] = []
    used_candidate_keys: set[str] = set()

    for _, wf_row in wontfix_df.iterrows():
        candidates = non_wontfix_df[non_wontfix_df["__repo"] == wf_row["__repo"]].copy()

        if pd.notna(wf_row["__created_at"]):
            min_ts = wf_row["__created_at"] - pd.Timedelta(days=time_window_days)
            max_ts = wf_row["__created_at"] + pd.Timedelta(days=time_window_days)
            candidates = candidates[
                candidates["__created_at"].between(min_ts, max_ts, inclusive="both")
            ]

        if wf_row["__issue_type"]:
            same_type = candidates[candidates["__issue_type"] == wf_row["__issue_type"]]
            if not same_type.empty:
                candidates = same_type

        if candidates.empty:
            continue

        candidates = candidates[~candidates["__issue_key"].isin(used_candidate_keys)].copy()
        if candidates.empty:
            continue

        candidates = add_pair_diagnostics(candidates, wf_row)
        candidates["__match_score"] = candidates.apply(lambda r: candidate_score(wf_row, r), axis=1)
        candidates = candidates.sort_values(
            by=["__match_score", "__has_linked_pr", "__comment_count", "__created_at"],
            ascending=[True, False, True, True],
        )

        selected = candidates.head(max_controls_per_wontfix).copy()
        if selected.empty:
            continue

        selected["__match_rank_for_wontfix"] = range(1, len(selected) + 1)
        selected["__matched_to_wontfix_issue_id"] = wf_row["__issue_id"]
        selected["__matched_to_wontfix_issue_key"] = wf_row["__issue_key"]
        selected["__matched_to_repo"] = wf_row["__repo"]
        selected["__matched_to_issue_type"] = wf_row["__issue_type"]
        selected["__matched_to_wontfix_issue_number"] = wf_row["__issue_number"]
        selected["__matched_to_wontfix_created_at"] = wf_row["__created_at"]
        selected["__matched_to_wontfix_comment_count"] = wf_row["__comment_count"]

        selected_rows.extend([row for _, row in selected.iterrows()])
        used_candidate_keys.update(selected["__issue_key"].tolist())

    comparison_df = pd.DataFrame(selected_rows) if selected_rows else derived_df.iloc[0:0].copy()
    return wontfix_df, comparison_df


# -----------------------------------------------------------------------------
# Pair mapping and QA
# -----------------------------------------------------------------------------


def build_pair_mapping(comparison_df: pd.DataFrame, derived_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "repo_full_name",
        "matched_set_id",
        "wontfix_issue_id",
        "wontfix_issue_number",
        "comparison_issue_id",
        "comparison_issue_number",
        "match_rank_for_wontfix",
        "match_score",
        "comparison_bucket",
        "wontfix_issue_type",
        "comparison_issue_type",
        "same_issue_type_flag",
        "created_at_day_diff",
        "wontfix_created_at",
        "comparison_created_at",
        "wontfix_comment_count",
        "comparison_comment_count",
        "comment_count_abs_diff",
        "comparison_has_linked_pr",
        "comparison_is_invalid",
        "comparison_state",
    ]

    if comparison_df.empty:
        return pd.DataFrame(columns=output_columns)

    pair_df = pd.DataFrame({
        "repo_full_name": comparison_df["__repo"].astype(str),
        "wontfix_issue_id": comparison_df["__matched_to_wontfix_issue_id"].astype(str),
        "wontfix_issue_number": comparison_df["__matched_to_wontfix_issue_number"],
        "comparison_issue_id": comparison_df["__issue_id"].astype(str),
        "comparison_issue_number": comparison_df["__issue_number"],
        "match_rank_for_wontfix": comparison_df["__match_rank_for_wontfix"],
        "match_score": comparison_df["__match_score"],
        "comparison_bucket": comparison_df["__comparison_bucket"],
        "wontfix_issue_type": comparison_df["__matched_to_issue_type"],
        "comparison_issue_type": comparison_df["__issue_type"],
        "same_issue_type_flag": comparison_df["__same_issue_type_flag"],
        "created_at_day_diff": comparison_df["__created_at_day_diff"],
        "wontfix_created_at": comparison_df["__matched_to_wontfix_created_at"],
        "comparison_created_at": comparison_df["__created_at"],
        "wontfix_comment_count": comparison_df["__matched_to_wontfix_comment_count"],
        "comparison_comment_count": comparison_df["__comment_count"],
        "comment_count_abs_diff": comparison_df["__comment_count_abs_diff"],
        "comparison_has_linked_pr": comparison_df["__has_linked_pr"].astype(int),
        "comparison_is_invalid": comparison_df["__is_invalid"].astype(int),
        "comparison_state": comparison_df["__state"],
    })
    pair_df["matched_set_id"] = pair_df["repo_full_name"].astype(str) + "::" + pair_df["wontfix_issue_id"].astype(str)

    return pair_df[output_columns].drop_duplicates().sort_values(
        ["repo_full_name", "wontfix_issue_number", "match_rank_for_wontfix"],
        kind="stable",
    ).reset_index(drop=True)


def build_issue_level_matched_set_lookup(pair_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["repo_full_name", "issue_id", "issue_number", "analysis_set", "matched_set_id"]
    if pair_df.empty:
        return pd.DataFrame(columns=columns)

    wontfix_lookup = pair_df[["repo_full_name", "wontfix_issue_id", "wontfix_issue_number", "matched_set_id"]].copy()
    wontfix_lookup = wontfix_lookup.rename(
        columns={"wontfix_issue_id": "issue_id", "wontfix_issue_number": "issue_number"}
    )
    wontfix_lookup["analysis_set"] = "wontfix"

    comparison_lookup = pair_df[["repo_full_name", "comparison_issue_id", "comparison_issue_number", "matched_set_id"]].copy()
    comparison_lookup = comparison_lookup.rename(
        columns={"comparison_issue_id": "issue_id", "comparison_issue_number": "issue_number"}
    )
    comparison_lookup["analysis_set"] = "comparison"

    lookup = pd.concat([wontfix_lookup, comparison_lookup], ignore_index=True)
    lookup["issue_id"] = lookup["issue_id"].astype(str)
    return lookup[columns].drop_duplicates().reset_index(drop=True)


def build_qa_summary(
    full_df: pd.DataFrame,
    wontfix_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    max_controls_per_wontfix: int,
    time_window_days: int,
) -> pd.DataFrame:
    matched_counts = (
        comparison_df.groupby("__matched_to_wontfix_issue_key").size()
        if not comparison_df.empty and "__matched_to_wontfix_issue_key" in comparison_df.columns
        else pd.Series(dtype=int)
    )

    zero_match_count = 0
    if not wontfix_df.empty:
        zero_match_count = int((~wontfix_df["__issue_key"].isin(set(matched_counts.index.astype(str)))).sum())

    non_wontfix_df = full_df[~full_df["__is_wontfix"]].copy()
    selected_keys = set(comparison_df["__issue_key"].tolist()) if not comparison_df.empty else set()
    wontfix_keys = set(wontfix_df["__issue_key"].tolist()) if not wontfix_df.empty else set()
    comparison_keys = set(comparison_df["__issue_key"].tolist()) if not comparison_df.empty else set()

    pair_rows_cross_repo = 0
    if not comparison_df.empty and "__matched_to_repo" in comparison_df.columns:
        pair_rows_cross_repo = int((comparison_df["__repo"] != comparison_df["__matched_to_repo"]).sum())

    pair_rows_outside_time_window = 0
    if not comparison_df.empty and "__created_at_day_diff" in comparison_df.columns:
        pair_rows_outside_time_window = int((pd.to_numeric(comparison_df["__created_at_day_diff"], errors="coerce") > time_window_days).sum())

    comparison_issue_reused_count = 0
    if not pair_df.empty:
        comparison_issue_reused_count = int(
            pair_df.duplicated(subset=["repo_full_name", "comparison_issue_id"]).sum()
        )

    controls_per_set = pair_df.groupby("matched_set_id").size() if not pair_df.empty else pd.Series(dtype=int)

    summary = {
        "total_issues": int(len(full_df)),
        "total_non_wontfix_pool_issues": int(len(non_wontfix_df)),
        "total_wontfix_issues": int(len(wontfix_df)),
        "total_selected_comparison_issues": int(len(comparison_df)),
        "pair_rows": int(len(pair_df)),
        "matched_sets": int(pair_df["matched_set_id"].nunique()) if not pair_df.empty else 0,
        "selected_resolved_pr": int((comparison_df["__comparison_bucket"] == "resolved_pr").sum()) if not comparison_df.empty else 0,
        "selected_closed_non_wontfix": int((comparison_df["__comparison_bucket"] == "closed_non_wontfix").sum()) if not comparison_df.empty else 0,
        "selected_invalid": int((comparison_df["__comparison_bucket"] == "invalid").sum()) if not comparison_df.empty else 0,
        "selected_open": int((comparison_df["__comparison_bucket"] == "open").sum()) if not comparison_df.empty else 0,
        "selected_other": int((comparison_df["__comparison_bucket"] == "other").sum()) if not comparison_df.empty else 0,
        "pool_resolved_pr": int((non_wontfix_df["__comparison_bucket"] == "resolved_pr").sum()) if not non_wontfix_df.empty else 0,
        "pool_closed_non_wontfix": int((non_wontfix_df["__comparison_bucket"] == "closed_non_wontfix").sum()) if not non_wontfix_df.empty else 0,
        "pool_invalid": int((non_wontfix_df["__comparison_bucket"] == "invalid").sum()) if not non_wontfix_df.empty else 0,
        "pool_open": int((non_wontfix_df["__comparison_bucket"] == "open").sum()) if not non_wontfix_df.empty else 0,
        "pool_other": int((non_wontfix_df["__comparison_bucket"] == "other").sum()) if not non_wontfix_df.empty else 0,
        "avg_controls_per_wontfix": float(round(len(comparison_df) / len(wontfix_df), 4)) if len(wontfix_df) else 0.0,
        "min_controls_per_matched_wontfix": int(controls_per_set.min()) if not controls_per_set.empty else 0,
        "max_controls_per_matched_wontfix": int(controls_per_set.max()) if not controls_per_set.empty else 0,
        "wontfix_issues_with_zero_matches": zero_match_count,
        "matched_sets_over_max_controls": int((controls_per_set > max_controls_per_wontfix).sum()) if not controls_per_set.empty else 0,
        "max_controls_per_wontfix": int(max_controls_per_wontfix),
        "time_window_days": int(time_window_days),
        "wontfix_duplicate_issue_keys": int(wontfix_df.duplicated(subset=["__issue_key"]).sum()) if not wontfix_df.empty else 0,
        "comparison_duplicate_issue_keys": int(comparison_df.duplicated(subset=["__issue_key"]).sum()) if not comparison_df.empty else 0,
        "wontfix_comparison_overlap_issue_keys": int(len(wontfix_keys & comparison_keys)),
        "comparison_rows_with_wontfix_label": int(comparison_df["__is_wontfix"].sum()) if not comparison_df.empty else 0,
        "comparison_issue_reused_count": comparison_issue_reused_count,
        "pair_duplicate_rows": int(pair_df.duplicated().sum()) if not pair_df.empty else 0,
        "pair_rows_cross_repo": pair_rows_cross_repo,
        "pair_rows_outside_time_window": pair_rows_outside_time_window,
        "pair_rows_same_issue_type_selected": int(pair_df["same_issue_type_flag"].sum()) if not pair_df.empty and "same_issue_type_flag" in pair_df.columns else 0,
        "pair_rows_issue_type_mismatch_selected": int((pair_df["same_issue_type_flag"] == 0).sum()) if not pair_df.empty and "same_issue_type_flag" in pair_df.columns else 0,
        "selected_comparison_keys_not_in_non_wontfix_pool": int(len(selected_keys - set(non_wontfix_df["__issue_key"].tolist()))),
    }

    quality_flags = []
    if summary["comparison_rows_with_wontfix_label"]:
        quality_flags.append("comparison_contains_wontfix_labeled_rows")
    if summary["wontfix_comparison_overlap_issue_keys"]:
        quality_flags.append("wontfix_comparison_overlap_detected")
    if summary["pair_rows_cross_repo"]:
        quality_flags.append("cross_repo_pairs_detected")
    if summary["pair_rows_outside_time_window"]:
        quality_flags.append("pairs_outside_time_window_detected")
    if summary["comparison_issue_reused_count"]:
        quality_flags.append("comparison_issue_reused_across_sets")
    if summary["matched_sets_over_max_controls"]:
        quality_flags.append("matched_set_exceeds_max_controls")
    if summary["wontfix_issues_with_zero_matches"]:
        quality_flags.append("some_wontfix_issues_unmatched")

    summary["quality_flags_json"] = json.dumps(quality_flags)
    summary["quality_status"] = "ok" if not quality_flags else "check"

    return pd.DataFrame([summary])


def drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    internal_cols = [c for c in df.columns if str(c).startswith("__")]
    return df.drop(columns=internal_cols, errors="ignore")


def maybe_load_issue_pr_links(issue_pr_links_path: Path | None) -> pd.DataFrame | None:
    if issue_pr_links_path is None:
        return None
    if not issue_pr_links_path.exists():
        return None
    return read_table(issue_pr_links_path)


def write_optional_lookup(pair_df: pd.DataFrame, pair_output_path: Path) -> Path:
    lookup_df = build_issue_level_matched_set_lookup(pair_df)
    lookup_path = pair_output_path.with_name(pair_output_path.stem + "_issue_lookup.csv")
    lookup_df.to_csv(lookup_path, index=False)
    return lookup_path


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WONTFIX and matched non-WONTFIX comparison issue sets.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to study_config.yaml. Defaults to config/study_config.yaml relative to project root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)

    (
        issues_path,
        wontfix_output_path,
        comparison_output_path,
        qa_summary_path,
        pair_output_path,
        issue_pr_links_path,
    ) = pick_paths(config)

    max_controls_per_wontfix, time_window_days = resolve_matching_settings(config)
    wontfix_labels, invalid_labels, issue_type_map = resolve_label_settings(config)

    print("Loaded config successfully.")
    print(f"Issues input path: {issues_path}")
    print(f"Issue-PR links input path: {issue_pr_links_path}")
    print(f"WONTFIX output path: {wontfix_output_path}")
    print(f"Comparison output path: {comparison_output_path}")
    print(f"QA summary path: {qa_summary_path}")
    print(f"Pair mapping output path: {pair_output_path}")
    print(f"Max controls per WONTFIX: {max_controls_per_wontfix}")
    print(f"Time window (days): {time_window_days}")

    issues_df = read_table(issues_path)
    issue_pr_links_df = maybe_load_issue_pr_links(issue_pr_links_path)

    print(f"Loaded issues table with {len(issues_df)} rows.")
    if issue_pr_links_df is not None:
        print(f"Loaded issue_pr_links table with {len(issue_pr_links_df)} rows.")
    else:
        print("Issue-PR links table not found or not configured; using issue-only fallback for linked PR detection.")

    print("Columns:")
    print(list(issues_df.columns))

    derived_df, colmap = add_derived_columns(
        issues_df=issues_df,
        wontfix_labels=wontfix_labels,
        invalid_labels=invalid_labels,
        issue_type_map=issue_type_map,
        issue_pr_links_df=issue_pr_links_df,
    )

    print("Resolved column mapping:")
    print(colmap)

    wontfix_df, comparison_df = select_comparison_controls(
        derived_df=derived_df,
        max_controls_per_wontfix=max_controls_per_wontfix,
        time_window_days=time_window_days,
    )

    pair_df = build_pair_mapping(comparison_df=comparison_df, derived_df=derived_df)

    qa_df = build_qa_summary(
        full_df=derived_df,
        wontfix_df=wontfix_df,
        comparison_df=comparison_df,
        pair_df=pair_df,
        max_controls_per_wontfix=max_controls_per_wontfix,
        time_window_days=time_window_days,
    )

    write_table(drop_internal_columns(wontfix_df), wontfix_output_path)
    write_table(drop_internal_columns(comparison_df), comparison_output_path)
    ensure_parent_dir(qa_summary_path)
    qa_df.to_csv(qa_summary_path, index=False)
    write_table(pair_df, pair_output_path)
    lookup_path = write_optional_lookup(pair_df, pair_output_path)

    print(f"Saved WONTFIX issue set: {wontfix_output_path} ({len(wontfix_df)} rows)")
    print(f"Saved comparison issue set: {comparison_output_path} ({len(comparison_df)} rows)")
    print(f"Saved QA summary CSV: {qa_summary_path}")
    print(f"Saved pair mapping: {pair_output_path} ({len(pair_df)} rows)")
    print(f"Saved issue-level matched-set lookup CSV: {lookup_path}")
    print("QA status:")
    print(qa_df.to_string(index=False))


if __name__ == "__main__":
    main()
