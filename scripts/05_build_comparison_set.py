#!/usr/bin/env python3
"""
Build WONTFIX and matched non-WONTFIX comparison issue sets.

Fresh replacement focused on one important rule:
    Issue type is inferred from GitHub label JSON columns only.

This script deliberately does NOT search for a scalar issue_type/type column,
because GitHub issue exports often contain author_type, and substring matching
on "type" can incorrectly turn every issue type into "user".

Outputs:
  - WONTFIX issue set
  - matched comparison issue set
  - pair mapping with matched_set_id
  - issue-level matched-set lookup CSV
  - QA summary CSV
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_VERSION = "label_json_force_first3_by_wontfix_issue_id_v2026_06_21"

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
    "wont fix",
    "wont-fix",
    "won't-fix",
    "status: wontfix",
    "type: wontfix",
    "not planned",
    "declined",
]

DEFAULT_INVALID_LABELS = [
    "invalid",
    "question",
    "works as intended",
    "duplicate",
    "cannot-reproduce",
    "cant-reproduce",
    "need more information",
    "incomplete",
]

# Broad, paper-friendly type buckets. These are intentionally label-derived.
# They include the labels observed in the current subset, especially feature-request,
# site-bug, site-request, site-enhancement, and docs/meta/cleanup.
DEFAULT_ISSUE_TYPE_MAP = {
    "bug": [
        "bug",
        "type: bug",
        "kind: bug",
        "kind/bug",
        "site-bug",
        "regression",
        "confirmed",
    ],
    "feature": [
        "enhancement",
        "feature",
        "feature request",
        "feature-request",
        "type: feature",
        "kind: feature",
        "kind/feature",
        "site-request",
        "site-enhancement",
        "plugin request",
        "ui/ux",
        "accessibility",
    ],
    "documentation": [
        "documentation",
        "docs",
        "type: docs",
        "kind: documentation",
        "kind/documentation",
        "docs/meta/cleanup",
        "wiki",
    ],
    "question": [
        "question",
        "support",
        "type: question",
        "info:feedback-needed",
        "awaiting-reply",
    ],
}

ISSUE_TYPE_PRIORITY = ["bug", "feature", "documentation", "question"]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def get_cfg(cfg: Any, *path: str, default: Any = None) -> Any:
    cur = cfg
    for key in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def ensure_parent_dir(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>", "nat"}:
        return None
    return text


def normalize_key(value: Any) -> str:
    text = clean_text(value) or ""
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label_for_matching(value: Any) -> str:
    """Normalize label names enough for robust exact matching.

    We keep meaningful separators like '/', ':', and '-' because project labels
    such as docs/meta/cleanup, site-bug, and type: bug are meaningful.
    """
    text = normalize_key(value)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def unique_preserve_order(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = normalize_label_for_matching(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def find_column_exact(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    """Exact-name column lookup only.

    This intentionally does not do substring matching. Substring matching is what
    caused candidate "type" to resolve to author_type in the old script.
    """
    lower_map = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        hit = lower_map.get(str(candidate).lower())
        if hit is not None:
            return hit
    if required:
        raise KeyError(f"Required column not found. Tried exact names: {candidates}")
    return None


def normalize_datetime(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(series, errors="coerce", utc=True)


def normalize_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def normalize_lower_text_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="object")
    return series.apply(lambda value: normalize_key(value) or None)


def parse_boolish_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    raw_num = pd.to_numeric(series, errors="coerce")
    if raw_num.notna().any():
        return raw_num.fillna(0).astype(float) > 0

    return series.apply(lambda value: normalize_key(value) in {"true", "1", "yes", "y", "t"})


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input table not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported table format for {path}. Use .parquet, .csv, .json, .jsonl, or .ndjson.")


def write_table(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".json":
        df.to_json(path, orient="records", indent=2)
        return
    if suffix in {".jsonl", ".ndjson"}:
        df.to_json(path, orient="records", lines=True)
        return
    df.to_parquet(path, index=False)


# -----------------------------------------------------------------------------
# Label parsing and issue-type inference
# -----------------------------------------------------------------------------


def parse_possible_json(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (list, dict)):
        return value
    text = clean_text(value)
    if text is None:
        return None
    if not ((text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}"))):
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass
    return value


def extract_label_names_from_value(value: Any) -> list[str]:
    """Extract label names from strings, JSON lists, and GitHub label payloads.

    Handles:
      - ["bug", "wontfix"]
      - [{"name": "bug"}, {"name": "wontfix"}]
      - {"name": "bug"}
      - comma/semicolon/pipe-delimited strings
    """
    parsed = parse_possible_json(value)

    if parsed is None or (isinstance(parsed, float) and pd.isna(parsed)):
        return []

    if isinstance(parsed, dict):
        # GitHub label payload object: use the label object's name field.
        if "name" in parsed:
            return unique_preserve_order([parsed.get("name")])
        # Some exports may store a mapping name -> payload/bool.
        return unique_preserve_order(list(parsed.keys()))

    if isinstance(parsed, list):
        labels: list[Any] = []
        for item in parsed:
            if isinstance(item, dict):
                if "name" in item:
                    labels.append(item.get("name"))
                else:
                    labels.extend(list(item.keys()))
            else:
                labels.append(item)
        return unique_preserve_order(labels)

    text = clean_text(parsed)
    if text is None:
        return []
    if any(separator in text for separator in ["|", ";", ","]):
        return unique_preserve_order(re.split(r"[|,;]", text))
    return unique_preserve_order([text])


def find_label_columns(df: pd.DataFrame) -> list[str]:
    """Return label columns in preferred order, exact matches only."""
    candidates = [
        "label_names_json",
        "label_payload_json",
        "labels_json",
        "labels",
        "label_names",
        "issue_labels",
    ]
    label_columns: list[str] = []
    lower_map = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        col = lower_map.get(candidate.lower())
        if col is not None and col not in label_columns:
            label_columns.append(col)
    if not label_columns:
        raise KeyError("No label columns found. Expected label_names_json and/or label_payload_json.")
    return label_columns


def combine_label_columns(row: pd.Series, label_columns: list[str]) -> list[str]:
    labels: list[str] = []
    for col in label_columns:
        labels.extend(extract_label_names_from_value(row.get(col)))
    return unique_preserve_order(labels)


def label_matches_alias(label: str, alias: str) -> bool:
    label_norm = normalize_label_for_matching(label)
    alias_norm = normalize_label_for_matching(alias)
    if not label_norm or not alias_norm:
        return False
    if label_norm == alias_norm:
        return True

    # Allow common separator variants for labels like feature-request vs feature request.
    label_space = re.sub(r"[-_/]+", " ", label_norm)
    alias_space = re.sub(r"[-_/]+", " ", alias_norm)
    label_space = re.sub(r"\s+", " ", label_space).strip()
    alias_space = re.sub(r"\s+", " ", alias_space).strip()
    return label_space == alias_space


def contains_any_label(labels: list[str], aliases: list[str]) -> bool:
    return any(label_matches_alias(label, alias) for label in labels for alias in aliases)


def infer_issue_types(labels: list[str], issue_type_map: dict[str, list[str]]) -> list[str]:
    issue_types: list[str] = []
    for issue_type in ISSUE_TYPE_PRIORITY:
        aliases = issue_type_map.get(issue_type, [])
        if contains_any_label(labels, aliases):
            issue_types.append(issue_type)

    # Include any extra configured issue-type groups after the priority groups.
    for issue_type, aliases in issue_type_map.items():
        if issue_type in issue_types or issue_type in ISSUE_TYPE_PRIORITY:
            continue
        if contains_any_label(labels, aliases):
            issue_types.append(str(issue_type))
    return issue_types


def primary_issue_type(issue_types: list[str]) -> str | None:
    if not issue_types:
        return None
    for preferred in ISSUE_TYPE_PRIORITY:
        if preferred in issue_types:
            return preferred
    return issue_types[0]


def type_lists_overlap(left: Any, right: Any) -> bool:
    if not isinstance(left, list):
        left = [] if left is None or pd.isna(left) else [left]
    if not isinstance(right, list):
        right = [] if right is None or pd.isna(right) else [right]
    return bool(set(left) & set(right))


def merge_issue_type_maps(config_map: dict[str, Any] | None) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {key: list(values) for key, values in DEFAULT_ISSUE_TYPE_MAP.items()}
    if config_map:
        for key, values in dict(config_map).items():
            canonical_key = str(key)
            if canonical_key == "feature_request":
                canonical_key = "feature"
            existing = merged.get(canonical_key, [])
            merged[canonical_key] = existing + list(values or [])
    return {key: unique_preserve_order(values) for key, values in merged.items()}


# -----------------------------------------------------------------------------
# Config and path helpers
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
        or "logs/qa/comparison_issue_set_summary.csv"
    )

    pair_output_path = Path(
        get_cfg(config, "outputs", "wontfix_comparison_pairs_table")
        or get_cfg(config, "outputs", "wontfix_comparison_pairs")
        or "data/processed/wontfix_comparison_pairs.parquet"
    )

    issue_pr_links_raw = (
        get_cfg(config, "outputs", "issue_pr_links_table")
        or get_cfg(config, "issue_pr_links_table")
    )
    issue_pr_links_path = Path(issue_pr_links_raw) if issue_pr_links_raw else None

    return issues_path, wontfix_output_path, comparison_output_path, qa_summary_path, pair_output_path, issue_pr_links_path


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

    config_issue_type_map = get_cfg(config, "comparison_set", "issue_type_label_groups")
    if not config_issue_type_map:
        config_issue_type_map = {
            "bug": get_cfg(config, "label_normalization", "issue_type_labels", "bug", "variants", default=[]),
            "feature": get_cfg(config, "label_normalization", "issue_type_labels", "feature_request", "variants", default=[]),
            "documentation": get_cfg(config, "label_normalization", "issue_type_labels", "documentation", "variants", default=[]),
            "question": get_cfg(config, "label_normalization", "issue_type_labels", "question", "variants", default=[]),
        }

    issue_type_map = merge_issue_type_maps(config_issue_type_map)
    return unique_preserve_order(wontfix_labels), unique_preserve_order(invalid_labels), issue_type_map


# -----------------------------------------------------------------------------
# Feature derivation and matching
# -----------------------------------------------------------------------------


def add_derived_columns(
    issues_df: pd.DataFrame,
    wontfix_labels: list[str],
    invalid_labels: list[str],
    issue_type_map: dict[str, list[str]],
    issue_pr_links_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = issues_df.copy()

    label_columns = find_label_columns(df)
    colmap: dict[str, Any] = {
        "issue_id": find_column_exact(df, ["issue_id", "id", "node_id"], required=True),
        "repo": find_column_exact(df, ["repo_full_name", "repo_name", "repository_name", "repo", "full_name"], required=True),
        "created_at": find_column_exact(df, ["created_at", "issue_created_at", "created"], required=True),
        "closed_at": find_column_exact(df, ["closed_at", "issue_closed_at", "closed"]),
        "state": find_column_exact(df, ["state", "issue_state", "status"]),
        "labels": ",".join(label_columns),
        "label_columns": label_columns,
        "comment_count": find_column_exact(df, ["comments_count", "comment_count", "comments", "num_comments"]),
        "linked_pr": find_column_exact(df, ["has_linked_pr", "linked_pr", "linked_pull_request", "has_pr", "pull_request"]),
        # Hard rule: issue type is derived from label JSON columns only.
        "issue_type": label_columns,
        "issue_type_source": "label_columns_only",
        "issue_number": find_column_exact(df, ["issue_number", "number"]),
        "is_wontfix_labeled": find_column_exact(df, ["is_wontfix_labeled"]),
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

    df["__labels_list"] = df.apply(lambda row: combine_label_columns(row, label_columns), axis=1)
    df["__labels_json"] = df["__labels_list"].apply(json.dumps)

    if colmap["is_wontfix_labeled"]:
        df["__is_wontfix"] = parse_boolish_series(df[colmap["is_wontfix_labeled"]])
        df["__is_wontfix"] = df["__is_wontfix"] | df["__labels_list"].apply(lambda labels: contains_any_label(labels, wontfix_labels))
    else:
        df["__is_wontfix"] = df["__labels_list"].apply(lambda labels: contains_any_label(labels, wontfix_labels))

    df["__is_invalid"] = df["__labels_list"].apply(lambda labels: contains_any_label(labels, invalid_labels))

    # The central fix: infer issue type from labels only. No direct scalar issue_type/type lookup exists here.
    df["__issue_types_list"] = df["__labels_list"].apply(lambda labels: infer_issue_types(labels, issue_type_map))
    df["__issue_types_json"] = df["__issue_types_list"].apply(json.dumps)
    df["__issue_type"] = df["__issue_types_list"].apply(primary_issue_type)

    if colmap["linked_pr"]:
        df["__has_linked_pr"] = parse_boolish_series(df[colmap["linked_pr"]])
    else:
        df["__has_linked_pr"] = False

    if issue_pr_links_df is not None and not issue_pr_links_df.empty:
        links_df = issue_pr_links_df.copy()
        repo_link_col = find_column_exact(links_df, ["repo_full_name", "repo_name", "repo", "full_name"])
        issue_number_link_col = find_column_exact(links_df, ["issue_number", "number"])
        issue_id_link_col = find_column_exact(links_df, ["issue_id", "id", "node_id"])

        if repo_link_col and issue_number_link_col and colmap["issue_number"]:
            repo_series = links_df[repo_link_col].apply(clean_text).astype(str)
            issue_number_series = pd.to_numeric(links_df[issue_number_link_col], errors="coerce")
            valid = issue_number_series.notna()
            linked_issue_keys = set(zip(repo_series[valid], issue_number_series[valid].astype(int)))

            issue_numbers = pd.to_numeric(df[colmap["issue_number"]], errors="coerce")
            has_link = [
                (repo, int(issue_num)) in linked_issue_keys if pd.notna(issue_num) else False
                for repo, issue_num in zip(df["__repo"], issue_numbers)
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
    df["__is_closed"] = (df["__state"].eq("closed") | df["__closed_at"].notna()) if "__state" in df else df["__closed_at"].notna()

    # Assign lower-priority buckets first and PR-resolved last so resolved_pr is not overwritten.
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
        score += 50.0

    comment_diff = abs(float(cand_row["__comment_count"]) - float(wf_row["__comment_count"]))
    score += min(comment_diff, 100)

    if type_lists_overlap(wf_row.get("__issue_types_list"), cand_row.get("__issue_types_list")):
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
    candidates["__same_issue_type_flag"] = candidates["__issue_types_list"].apply(
        lambda types: int(type_lists_overlap(wf_row.get("__issue_types_list"), types))
    )
    return candidates



def deduplicate_issue_rows_for_matching(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one deterministic row per repo/issue key before matching.

    This is a defensive guard for duplicated issue exports. Duplicate WONTFIX
    rows can make one matched set receive more than max_controls controls, and
    duplicate candidate rows can make the same comparison issue appear at
    multiple ranks.
    """
    if df is None:
        return pd.DataFrame()
    if df.empty or "__issue_key" not in df.columns:
        return df.copy()
    out = df.copy()
    out["__dedupe_label_count"] = out["__labels_list"].apply(lambda x: len(x) if isinstance(x, list) else 0) if "__labels_list" in out.columns else 0
    out["__dedupe_type_count"] = out["__issue_types_list"].apply(lambda x: len(x) if isinstance(x, list) else 0) if "__issue_types_list" in out.columns else out["__issue_type"].notna().astype(int) if "__issue_type" in out.columns else 0
    out["__dedupe_linked_pr"] = out["__has_linked_pr"].astype(int) if "__has_linked_pr" in out.columns else 0
    out["__dedupe_closed"] = out["__is_closed"].astype(int) if "__is_closed" in out.columns else 0
    sort_cols = [c for c in ["__repo", "__issue_key", "__dedupe_linked_pr", "__dedupe_type_count", "__dedupe_label_count", "__dedupe_closed", "__created_at"] if c in out.columns]
    ascending = [True, True, False, False, False, False, True][:len(sort_cols)]
    out = out.sort_values(sort_cols, ascending=ascending, kind="stable")
    out = out.drop_duplicates(subset=["__issue_key"], keep="first").copy()
    return out.drop(columns=[c for c in out.columns if str(c).startswith("__dedupe_")], errors="ignore")


def select_comparison_controls(
    derived_df: pd.DataFrame,
    max_controls_per_wontfix: int,
    time_window_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Deduplicate both treatment and candidate pools before matching. This keeps
    # the intended one-row-per-issue matching unit even when upstream issue
    # extraction produced duplicate rows.
    wontfix_df = deduplicate_issue_rows_for_matching(derived_df[derived_df["__is_wontfix"]]).copy()
    non_wontfix_df = deduplicate_issue_rows_for_matching(derived_df[~derived_df["__is_wontfix"]]).copy()

    selected_rows: list[pd.Series] = []
    used_candidate_keys: set[str] = set()

    for _, wf_row in wontfix_df.iterrows():
        candidates = non_wontfix_df[non_wontfix_df["__repo"] == wf_row["__repo"]].copy()

        if pd.notna(wf_row["__created_at"]):
            min_ts = wf_row["__created_at"] - pd.Timedelta(days=time_window_days)
            max_ts = wf_row["__created_at"] + pd.Timedelta(days=time_window_days)
            candidates = candidates[candidates["__created_at"].between(min_ts, max_ts, inclusive="both")]

        wf_types = wf_row.get("__issue_types_list") or []
        if wf_types:
            same_type_mask = candidates["__issue_types_list"].apply(lambda types: type_lists_overlap(wf_types, types))
            same_type_candidates = candidates[same_type_mask].copy()
            if not same_type_candidates.empty:
                candidates = same_type_candidates

        if candidates.empty:
            continue

        # Re-deduplicate after all filters, then remove globally used controls.
        candidates = deduplicate_issue_rows_for_matching(candidates)
        candidates = candidates[~candidates["__issue_key"].isin(used_candidate_keys)].copy()
        if candidates.empty:
            continue

        candidates = add_pair_diagnostics(candidates, wf_row)
        candidates["__match_score"] = candidates.apply(lambda row: candidate_score(wf_row, row), axis=1)
        candidates = candidates.sort_values(
            by=["__match_score", "__has_linked_pr", "__comment_count", "__created_at", "__issue_key"],
            ascending=[True, False, True, True, True],
            kind="stable",
        )
        candidates = candidates.drop_duplicates(subset=["__issue_key"], keep="first")

        selected = candidates.head(int(max_controls_per_wontfix)).copy()
        if selected.empty:
            continue

        selected["__match_rank_for_wontfix"] = range(1, len(selected) + 1)
        selected["__matched_to_wontfix_issue_id"] = wf_row["__issue_id"]
        selected["__matched_to_wontfix_issue_key"] = wf_row["__issue_key"]
        selected["__matched_to_repo"] = wf_row["__repo"]
        selected["__matched_to_issue_type"] = wf_row["__issue_type"]
        selected["__matched_to_issue_types_json"] = wf_row["__issue_types_json"]
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


def build_pair_mapping(comparison_df: pd.DataFrame, max_controls_per_wontfix: int = 3) -> pd.DataFrame:
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
        "wontfix_issue_types_json",
        "comparison_issue_types_json",
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
        "match_rank_for_wontfix": pd.to_numeric(comparison_df["__match_rank_for_wontfix"], errors="coerce"),
        "match_score": pd.to_numeric(comparison_df["__match_score"], errors="coerce"),
        "comparison_bucket": comparison_df["__comparison_bucket"],
        "wontfix_issue_type": comparison_df["__matched_to_issue_type"],
        "comparison_issue_type": comparison_df["__issue_type"],
        "wontfix_issue_types_json": comparison_df["__matched_to_issue_types_json"],
        "comparison_issue_types_json": comparison_df["__issue_types_json"],
        "same_issue_type_flag": comparison_df["__same_issue_type_flag"],
        "created_at_day_diff": pd.to_numeric(comparison_df["__created_at_day_diff"], errors="coerce"),
        "wontfix_created_at": comparison_df["__matched_to_wontfix_created_at"],
        "comparison_created_at": comparison_df["__created_at"],
        "wontfix_comment_count": comparison_df["__matched_to_wontfix_comment_count"],
        "comparison_comment_count": comparison_df["__comment_count"],
        "comment_count_abs_diff": pd.to_numeric(comparison_df["__comment_count_abs_diff"], errors="coerce"),
        "comparison_has_linked_pr": comparison_df["__has_linked_pr"].astype(int),
        "comparison_is_invalid": comparison_df["__is_invalid"].astype(int),
        "comparison_state": comparison_df["__state"],
    })
    pair_df["matched_set_id"] = pair_df["repo_full_name"].astype(str) + "::" + pair_df["wontfix_issue_id"].astype(str)

    # Final source-of-truth cap. The requested rule is simple and explicit:
    # for each WONTFIX issue, keep only the first N matched controls. Use the
    # WONTFIX issue id rather than matched_set_id so this guard still works if
    # a downstream diagnostic groups by the original WONTFIX id.
    #
    # Ordering starts with the rank assigned during matching. Score and distance
    # fields are tie-breakers only. This preserves "first 3" semantics while
    # remaining deterministic for duplicated upstream rows.
    pair_df = pair_df.sort_values(
        [
            "repo_full_name",
            "wontfix_issue_id",
            "match_rank_for_wontfix",
            "match_score",
            "created_at_day_diff",
            "comment_count_abs_diff",
            "comparison_issue_id",
        ],
        ascending=[True, True, True, True, True, True, True],
        kind="stable",
    )

    # Remove duplicate instances of the same WONTFIX-control pair before capping.
    pair_df = pair_df.drop_duplicates(
        subset=["repo_full_name", "wontfix_issue_id", "comparison_issue_id"],
        keep="first",
    ).copy()

    # Preserve the original no-reuse design: one comparison issue can serve only
    # one WONTFIX issue within a repository. Keep the earliest/best assignment.
    pair_df = pair_df.drop_duplicates(
        subset=["repo_full_name", "comparison_issue_id"],
        keep="first",
    ).copy()

    # HARD CAP: first N controls per WONTFIX issue id.
    pair_df = (
        pair_df
        .groupby(["repo_full_name", "wontfix_issue_id"], group_keys=False, sort=False)
        .head(int(max_controls_per_wontfix))
        .copy()
    )

    # Recompute ranks after dedupe/no-reuse/capping.
    pair_df["match_rank_for_wontfix"] = (
        pair_df.groupby(["repo_full_name", "wontfix_issue_id"], sort=False).cumcount() + 1
    )
    pair_df["matched_set_id"] = pair_df["repo_full_name"].astype(str) + "::" + pair_df["wontfix_issue_id"].astype(str)

    # Hard validation. If this fails, do not run downstream scripts.
    controls_by_wf = pair_df.groupby(["repo_full_name", "wontfix_issue_id"])["comparison_issue_id"].nunique()
    if not controls_by_wf.empty and int(controls_by_wf.max()) > int(max_controls_per_wontfix):
        offenders = controls_by_wf[controls_by_wf > int(max_controls_per_wontfix)].head(10)
        raise RuntimeError(
            "Pair mapping still has WONTFIX issues with more than "
            f"{int(max_controls_per_wontfix)} controls after final cap. Examples: {offenders.to_dict()}"
        )

    duplicate_pairs = int(pair_df.duplicated(subset=["repo_full_name", "wontfix_issue_id", "comparison_issue_id"]).sum())
    if duplicate_pairs:
        raise RuntimeError(f"Pair mapping still has {duplicate_pairs} duplicate WONTFIX-control rows after final cap.")

    reused_controls = int(pair_df.duplicated(subset=["repo_full_name", "comparison_issue_id"]).sum())
    if reused_controls:
        raise RuntimeError(f"Pair mapping still reuses {reused_controls} comparison issues after final cap.")

    return pair_df[output_columns].sort_values(
        ["repo_full_name", "wontfix_issue_id", "match_rank_for_wontfix"],
        kind="stable",
    ).reset_index(drop=True)


def restrict_comparison_df_to_pair_mapping(comparison_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    """Align the written comparison issue set to the capped pair mapping.

    The previous emergency version only filtered by allowed comparison issue id.
    That was not enough: if an allowed comparison issue appeared in several
    pre-cap candidate rows, the extra rows could survive into comparison_issue_set
    and later make diagnostics report >3 controls per WONTFIX.

    This version filters by the exact pair:
        repo_full_name + wontfix_issue_id + comparison_issue_id
    and then applies the same first-N-by-WONTFIX guard to the rows that will be
    written as comparison_issue_set.
    """
    if comparison_df is None or comparison_df.empty or pair_df is None or pair_df.empty:
        return comparison_df.iloc[0:0].copy() if comparison_df is not None else pd.DataFrame()

    allowed_pairs = set(
        zip(
            pair_df["repo_full_name"].astype(str),
            pair_df["wontfix_issue_id"].astype(str),
            pair_df["comparison_issue_id"].astype(str),
        )
    )

    out = comparison_df.copy()
    mask = [
        (str(repo), str(wf_id), str(cmp_id)) in allowed_pairs
        for repo, wf_id, cmp_id in zip(
            out["__repo"],
            out["__matched_to_wontfix_issue_id"],
            out["__issue_id"],
        )
    ]
    out = out.loc[mask].copy()
    if out.empty:
        return out.reset_index(drop=True)

    # Preserve only the row that corresponds to the final pair table for each
    # WONTFIX-control pair. This prevents duplicate candidate rows from leaking.
    out = out.sort_values(
        ["__repo", "__matched_to_wontfix_issue_id", "__match_rank_for_wontfix", "__match_score", "__created_at", "__issue_id"],
        kind="stable",
    )
    out = out.drop_duplicates(
        subset=["__repo", "__matched_to_wontfix_issue_id", "__issue_id"],
        keep="first",
    ).copy()

    # Mirror the written pair ranks onto the comparison rows so downstream
    # diagnostics see the capped ranks, not stale pre-cap ranks.
    rank_lookup = {
        (str(row.repo_full_name), str(row.wontfix_issue_id), str(row.comparison_issue_id)): int(row.match_rank_for_wontfix)
        for row in pair_df[["repo_full_name", "wontfix_issue_id", "comparison_issue_id", "match_rank_for_wontfix"]].itertuples(index=False)
    }
    out["__match_rank_for_wontfix"] = [
        rank_lookup.get((str(repo), str(wf_id), str(cmp_id)))
        for repo, wf_id, cmp_id in zip(out["__repo"], out["__matched_to_wontfix_issue_id"], out["__issue_id"])
    ]

    # Defensive validation on the actual comparison_issue_set rows that will be
    # written. This is intentionally grouped by WONTFIX issue id, matching the
    # user's diagnostic.
    controls_by_wf = out.groupby(["__repo", "__matched_to_wontfix_issue_id"])["__issue_id"].nunique()
    if not controls_by_wf.empty and int(controls_by_wf.max()) > int(pair_df.groupby(["repo_full_name", "wontfix_issue_id"]).size().max()):
        raise RuntimeError("comparison_issue_set has more controls per WONTFIX than the capped pair mapping.")

    return out.reset_index(drop=True)


def build_issue_level_matched_set_lookup(pair_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["repo_full_name", "issue_id", "issue_number", "analysis_set", "matched_set_id"]
    if pair_df.empty:
        return pd.DataFrame(columns=columns)

    wontfix_lookup = pair_df[["repo_full_name", "wontfix_issue_id", "wontfix_issue_number", "matched_set_id"]].copy()
    wontfix_lookup = wontfix_lookup.rename(columns={"wontfix_issue_id": "issue_id", "wontfix_issue_number": "issue_number"})
    wontfix_lookup["analysis_set"] = "wontfix"

    comparison_lookup = pair_df[["repo_full_name", "comparison_issue_id", "comparison_issue_number", "matched_set_id"]].copy()
    comparison_lookup = comparison_lookup.rename(columns={"comparison_issue_id": "issue_id", "comparison_issue_number": "issue_number"})
    comparison_lookup["analysis_set"] = "comparison"

    lookup = pd.concat([wontfix_lookup, comparison_lookup], ignore_index=True)
    lookup["issue_id"] = lookup["issue_id"].astype(str)
    return lookup[columns].drop_duplicates().reset_index(drop=True)


def summarize_issue_type_counts(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in df["__issue_type"].fillna("missing"):
        key = str(value) if value else "missing"
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_qa_summary(
    full_df: pd.DataFrame,
    wontfix_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    colmap: dict[str, Any],
    max_controls_per_wontfix: int,
    time_window_days: int,
) -> pd.DataFrame:
    matched_counts = (
        pair_df.groupby(["repo_full_name", "wontfix_issue_id"])["comparison_issue_id"].nunique()
        if not pair_df.empty
        else pd.Series(dtype=int)
    )
    matched_wf_ids = set(pair_df["matched_set_id"].astype(str)) if not pair_df.empty else set()
    zero_match_count = int((~wontfix_df["__issue_key"].astype(str).isin(matched_wf_ids)).sum()) if not wontfix_df.empty else 0
    non_wontfix_df = full_df[~full_df["__is_wontfix"]].copy()
    selected_keys = set(comparison_df["__issue_key"].tolist()) if not comparison_df.empty else set()
    wontfix_keys = set(wontfix_df["__issue_key"].tolist()) if not wontfix_df.empty else set()
    comparison_keys = set(comparison_df["__issue_key"].tolist()) if not comparison_df.empty else set()

    pair_rows_cross_repo = int((comparison_df["__repo"] != comparison_df["__matched_to_repo"]).sum()) if not comparison_df.empty else 0
    pair_rows_outside_time_window = int((pd.to_numeric(comparison_df["__created_at_day_diff"], errors="coerce") > time_window_days).sum()) if not comparison_df.empty else 0
    comparison_issue_reused_count = int(pair_df.duplicated(subset=["repo_full_name", "comparison_issue_id"]).sum()) if not pair_df.empty else 0
    controls_per_set = pair_df.groupby(["repo_full_name", "wontfix_issue_id"])["comparison_issue_id"].nunique() if not pair_df.empty else pd.Series(dtype=int)
    raw_rows_per_wf_in_comparison_set = comparison_df.groupby(["__repo", "__matched_to_wontfix_issue_id"])["__issue_id"].nunique() if not comparison_df.empty else pd.Series(dtype=int)
    controls_distribution = controls_per_set.reindex(
        pd.MultiIndex.from_tuples(
            [(str(row["__repo"]), str(row["__issue_id"])) for _, row in wontfix_df.iterrows()],
            names=["repo_full_name", "wontfix_issue_id"],
        ),
        fill_value=0,
    ).value_counts().sort_index().to_dict() if not wontfix_df.empty else {}
    controls_distribution = {str(int(k)): int(v) for k, v in controls_distribution.items()}

    issue_type_lengths = full_df["__issue_types_list"].apply(lambda values: len(values) if isinstance(values, list) else 0)
    pair_missing_wf_type = int(pair_df["wontfix_issue_type"].isna().sum()) if not pair_df.empty else 0
    pair_missing_cmp_type = int(pair_df["comparison_issue_type"].isna().sum()) if not pair_df.empty else 0

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
        "comparison_set_wontfix_groups_over_max_controls": int((raw_rows_per_wf_in_comparison_set > max_controls_per_wontfix).sum()) if not raw_rows_per_wf_in_comparison_set.empty else 0,
        "max_controls_per_wontfix": int(max_controls_per_wontfix),
        "time_window_days": int(time_window_days),
        "controls_per_wontfix_distribution_json": json.dumps(controls_distribution, sort_keys=True),
        "wontfix_duplicate_issue_keys": int(wontfix_df.duplicated(subset=["__issue_key"]).sum()) if not wontfix_df.empty else 0,
        "comparison_duplicate_issue_keys": int(comparison_df.duplicated(subset=["__issue_key"]).sum()) if not comparison_df.empty else 0,
        "wontfix_comparison_overlap_issue_keys": int(len(wontfix_keys & comparison_keys)),
        "comparison_rows_with_wontfix_label": int(comparison_df["__is_wontfix"].sum()) if not comparison_df.empty else 0,
        "comparison_issue_reused_count": comparison_issue_reused_count,
        "pair_duplicate_rows": int(pair_df.duplicated().sum()) if not pair_df.empty else 0,
        "pair_rows_cross_repo": pair_rows_cross_repo,
        "pair_rows_outside_time_window": pair_rows_outside_time_window,
        "pair_rows_same_issue_type_selected": int(pair_df["same_issue_type_flag"].sum()) if not pair_df.empty else 0,
        "pair_rows_issue_type_mismatch_selected": int((pair_df["same_issue_type_flag"] == 0).sum()) if not pair_df.empty else 0,
        "pair_rows_issue_type_missing_wontfix": pair_missing_wf_type,
        "pair_rows_issue_type_missing_comparison": pair_missing_cmp_type,
        "selected_comparison_keys_not_in_non_wontfix_pool": int(len(selected_keys - set(non_wontfix_df["__issue_key"].tolist()))),
        "label_columns_json": json.dumps(colmap.get("label_columns", [])),
        "issue_type_source": colmap.get("issue_type_source"),
        "issue_type_column_mapping_json": json.dumps(colmap.get("issue_type", [])),
        "issues_with_any_label": int(full_df["__labels_list"].apply(lambda labels: len(labels) > 0).sum()),
        "issue_type_from_labels_rows": int(issue_type_lengths.gt(0).sum()),
        "issue_type_missing_rows": int(issue_type_lengths.eq(0).sum()),
        "issue_type_multitype_rows": int(issue_type_lengths.gt(1).sum()),
        "issue_type_value_counts_json": json.dumps(summarize_issue_type_counts(full_df), sort_keys=True),
    }

    quality_flags: list[str] = []
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
    if summary.get("comparison_set_wontfix_groups_over_max_controls", 0):
        quality_flags.append("comparison_issue_set_exceeds_max_controls")
    if summary["wontfix_issues_with_zero_matches"]:
        quality_flags.append("some_wontfix_issues_unmatched")
    if '"user"' in summary["issue_type_value_counts_json"].lower():
        quality_flags.append("issue_type_contains_user_check_label_inference")

    summary["quality_flags_json"] = json.dumps(quality_flags)
    summary["quality_status"] = "ok" if not quality_flags else "check"
    return pd.DataFrame([summary])


def drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    internal_cols = [column for column in df.columns if str(column).startswith("__")]
    return df.drop(columns=internal_cols, errors="ignore")


def maybe_load_issue_pr_links(issue_pr_links_path: Path | None) -> pd.DataFrame | None:
    if issue_pr_links_path is None or not issue_pr_links_path.exists():
        return None
    return read_table(issue_pr_links_path)


def write_optional_lookup(pair_df: pd.DataFrame, pair_output_path: Path) -> Path:
    lookup_df = build_issue_level_matched_set_lookup(pair_df)
    lookup_path = pair_output_path.with_name(pair_output_path.stem + "_issue_lookup.csv")
    ensure_parent_dir(lookup_path)
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

    print(f"Script version: {SCRIPT_VERSION}")
    print("Loaded config successfully.")
    print(f"Issues input path: {issues_path}")
    print(f"Issue-PR links input path: {issue_pr_links_path}")
    print(f"WONTFIX output path: {wontfix_output_path}")
    print(f"Comparison output path: {comparison_output_path}")
    print(f"QA summary path: {qa_summary_path}")
    print(f"Pair mapping output path: {pair_output_path}")
    print(f"Max controls per WONTFIX: {max_controls_per_wontfix}")
    print(f"Time window (days): {time_window_days}")
    print("Issue type inference source: label JSON columns only; scalar type/author_type columns are ignored.")

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
    print("Issue type value counts:")
    print(pd.Series(summarize_issue_type_counts(derived_df)).sort_index().to_string())

    wontfix_df, comparison_df = select_comparison_controls(
        derived_df=derived_df,
        max_controls_per_wontfix=max_controls_per_wontfix,
        time_window_days=time_window_days,
    )

    pair_df = build_pair_mapping(comparison_df=comparison_df, max_controls_per_wontfix=max_controls_per_wontfix)

    comparison_df = restrict_comparison_df_to_pair_mapping(comparison_df, pair_df)

    final_pair_counts = pair_df.groupby(["repo_full_name", "wontfix_issue_id"])["comparison_issue_id"].nunique() if not pair_df.empty else pd.Series(dtype=int)
    final_comparison_counts = comparison_df.groupby(["__repo", "__matched_to_wontfix_issue_id"])["__issue_id"].nunique() if not comparison_df.empty else pd.Series(dtype=int)
    if not final_pair_counts.empty and int(final_pair_counts.max()) > int(max_controls_per_wontfix):
        raise RuntimeError("Final pair mapping still exceeds max_controls_per_wontfix grouped by WONTFIX issue id.")
    if not final_comparison_counts.empty and int(final_comparison_counts.max()) > int(max_controls_per_wontfix):
        raise RuntimeError("Final comparison issue set still exceeds max_controls_per_wontfix grouped by WONTFIX issue id.")

    qa_df = build_qa_summary(
        full_df=derived_df,
        wontfix_df=wontfix_df,
        comparison_df=comparison_df,
        pair_df=pair_df,
        colmap=colmap,
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
