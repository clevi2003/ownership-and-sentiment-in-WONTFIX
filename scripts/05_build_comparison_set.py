from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_VERSION = "label_json_controls_distribution_v2026_06_17"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.study_config_loader import load_study_config

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"

DEFAULT_WONTFIX_LABELS = [
    "wontfix", "won't fix", "won’t fix", "wont fix", "wont-fix",
    "not planned", "declined",
]

DEFAULT_INVALID_LABELS = [
    "invalid", "duplicate", "works as intended", "cannot-reproduce",
    "cant-reproduce", "need more information", "incomplete",
]

# Broad matching categories inferred from GitHub labels. The script never infers
# issue type from author_type, type, or any scalar issue column.
DEFAULT_ISSUE_TYPE_MAP = {
    "bug": ["bug", "type: bug", "kind: bug", "kind/bug", "site-bug", "regression"],
    "feature": [
        "enhancement", "feature", "feature request", "feature-request",
        "type: feature", "kind: feature", "kind/feature", "site-request",
        "site-enhancement", "plugin request",
    ],
    "documentation": ["documentation", "docs", "type: docs", "docs/meta/cleanup", "wiki"],
    "question": ["question", "support", "type: question", "info:feedback-needed"],
}
ISSUE_TYPE_PRIORITY = ["bug", "feature", "documentation", "question"]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def get_cfg(cfg: Any, *path: str, default: Any = None) -> Any:
    cur = cfg
    for key in path:
        if cur is None:
            return default
        cur = cur.get(key) if isinstance(cur, dict) else getattr(cur, key, None)
    return default if cur is None else cur


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>", "nat"}:
        return None
    return text


def ensure_parent_dir(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = False,
    allow_substring: bool = True,
) -> str | None:
    """Find ordinary non-issue-type columns. Do not use this for issue type."""
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    if allow_substring:
        for col in df.columns:
            col_l = str(col).lower()
            for cand in candidates:
                if cand.lower() in col_l:
                    return col
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None


def find_label_columns(df: pd.DataFrame) -> list[str]:
    """Find label columns by exact name only."""
    preferred = [
        "label_names_json", "label_payload_json", "label_names",
        "labels_json", "labels", "issue_labels",
    ]
    lower = {str(c).lower(): c for c in df.columns}
    out: list[str] = []
    for name in preferred:
        col = lower.get(name)
        if col and col not in out:
            out.append(col)
    return out


def norm_label(value: Any) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def add_unique(out: list[str], seen: set[str], value: Any) -> None:
    label = norm_label(value)
    if label and label not in seen:
        seen.add(label)
        out.append(label)


def extract_label_names(parsed: Any) -> list[str]:
    """Extract label names from strings, lists, or GitHub label payload dicts."""
    out: list[str] = []
    seen: set[str] = set()

    def visit(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            add_unique(out, seen, obj)
            return
        if isinstance(obj, dict):
            # Critical path for label_payload_json.
            if "name" in obj:
                add_unique(out, seen, obj.get("name"))
                return
            for key in ("label", "labels", "nodes", "edges"):
                if key in obj:
                    visit(obj.get(key))
            return
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                visit(item)
            return
        add_unique(out, seen, obj)

    visit(parsed)
    return out


def parse_label_value(value: Any) -> list[str]:
    """Parse label_names_json or label_payload_json into normalized label names.

    Examples:
      '["bug", "wontfix"]' -> ['bug', 'wontfix']
      '[{"name":"bug"}]' -> ['bug']
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set, dict)):
        return extract_label_names(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return []

    if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                return extract_label_names(parsed)
            except Exception:
                pass

    parts = re.split(r"[|,;]", text)
    out: list[str] = []
    seen: set[str] = set()
    for part in parts:
        add_unique(out, seen, part)
    return out


def parse_row_labels(row: pd.Series, label_cols: list[str]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for col in label_cols:
        for label in parse_label_value(row.get(col)):
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def label_forms(label: str) -> set[str]:
    label = norm_label(label) or ""
    return {f for f in {
        label,
        label.replace("-", " "),
        label.replace(" ", "-"),
        label.replace("_", " "),
        label.replace("_", "-"),
    } if f}


def alias_forms(aliases: list[str]) -> set[str]:
    forms: set[str] = set()
    for alias in aliases or []:
        forms.update(label_forms(str(alias)))
    return forms


def labels_contain_any(labels: list[str], aliases: list[str]) -> bool:
    aliases_norm = alias_forms(aliases)
    return any(label_forms(label) & aliases_norm for label in labels)


def infer_issue_types(labels: list[str], issue_type_map: dict[str, list[str]]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for issue_type in ISSUE_TYPE_PRIORITY:
        if labels_contain_any(labels, issue_type_map.get(issue_type, [])):
            found.append(issue_type)
            seen.add(issue_type)
    for issue_type, aliases in issue_type_map.items():
        issue_type = str(issue_type)
        if issue_type not in seen and labels_contain_any(labels, aliases):
            found.append(issue_type)
            seen.add(issue_type)
    return found


def primary_issue_type(types: list[str]) -> str | None:
    if not types:
        return None
    for issue_type in ISSUE_TYPE_PRIORITY:
        if issue_type in types:
            return issue_type
    return types[0]


def type_overlap(left: Any, right: Any) -> bool:
    left_set = set(left if isinstance(left, list) else [])
    right_set = set(right if isinstance(right, list) else [])
    return bool(left_set and right_set and left_set.intersection(right_set))


def parse_boolish(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    nums = pd.to_numeric(series, errors="coerce")
    if nums.notna().any():
        return nums.fillna(0).astype(float) > 0
    cleaned = series.apply(clean_text)
    return cleaned.apply(lambda x: str(x).lower() in {"true", "1", "yes", "y", "t"} if x else False)


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
    raise ValueError(f"Unsupported table format for {path}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir(path)
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".json":
        df.to_json(path, orient="records", indent=2)
    elif suffix in {".jsonl", ".ndjson"}:
        df.to_json(path, orient="records", lines=True)
    else:
        df.to_parquet(path, index=False)


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------


def pick_paths(config: Any) -> tuple[Path, Path, Path, Path, Path, Path | None]:
    issues_path = Path(get_cfg(config, "outputs", "issues_table") or "data/processed/issues.parquet")
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
    issue_pr_links_raw = get_cfg(config, "outputs", "issue_pr_links_table") or get_cfg(config, "issue_pr_links_table")
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


def unique_lowered(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = norm_label(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


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
    cfg_map = get_cfg(config, "comparison_set", "issue_type_label_groups")
    if not cfg_map:
        cfg_map = {
            "bug": get_cfg(config, "label_normalization", "issue_type_labels", "bug", "variants", default=[]),
            "feature": get_cfg(config, "label_normalization", "issue_type_labels", "feature_request", "variants", default=[]),
            "documentation": get_cfg(config, "label_normalization", "issue_type_labels", "documentation", "variants", default=[]),
            "question": get_cfg(config, "label_normalization", "issue_type_labels", "question", "variants", default=[]),
        }
    if not any(dict(cfg_map).values()):
        cfg_map = {}

    merged: dict[str, list[str]] = {}
    for issue_type in set(DEFAULT_ISSUE_TYPE_MAP) | set(dict(cfg_map)):
        merged[issue_type] = unique_lowered(
            list(DEFAULT_ISSUE_TYPE_MAP.get(issue_type, []))
            + list(dict(cfg_map).get(issue_type, []) or [])
        )
    return unique_lowered(list(wontfix_labels)), unique_lowered(list(invalid_labels)), merged


# -----------------------------------------------------------------------------
# Derivation and matching
# -----------------------------------------------------------------------------


def add_derived_columns(
    issues_df: pd.DataFrame,
    wontfix_labels: list[str],
    invalid_labels: list[str],
    issue_type_map: dict[str, list[str]],
    issue_pr_links_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = issues_df.copy()
    label_cols = find_label_columns(df)
    if not label_cols:
        raise KeyError("No label columns found. Expected label_names_json and/or label_payload_json.")

    colmap: dict[str, Any] = {
        "issue_id": find_column(df, ["issue_id", "id", "node_id", "number"], required=True),
        "repo": find_column(df, ["repo_full_name", "repo_name", "repository_name", "repo", "full_name"], required=True),
        "created_at": find_column(df, ["created_at", "issue_created_at", "created"], required=True),
        "closed_at": find_column(df, ["closed_at", "issue_closed_at", "closed"]),
        "state": find_column(df, ["state", "issue_state", "status"]),
        "labels": ",".join(label_cols),
        "label_columns": label_cols,
        "issue_type": label_cols,
        "issue_type_source": "label_columns_only",
        "comment_count": find_column(df, ["comments", "comment_count", "num_comments", "comments_count"]),
        "linked_pr": find_column(df, ["has_linked_pr", "linked_pr", "linked_pull_request", "has_pr", "pull_request"]),
        "issue_number": find_column(df, ["issue_number", "number"]),
        "is_wontfix_labeled": find_column(df, ["is_wontfix_labeled"]),
    }

    df["__issue_id"] = df[colmap["issue_id"]].apply(clean_text).astype(str)
    df["__repo"] = df[colmap["repo"]].apply(clean_text).astype(str)
    df["__issue_key"] = df["__repo"] + "::" + df["__issue_id"]
    df["__issue_number"] = pd.to_numeric(df[colmap["issue_number"]], errors="coerce") if colmap["issue_number"] else pd.NA
    df["__created_at"] = pd.to_datetime(df[colmap["created_at"]], errors="coerce", utc=True)
    df["__closed_at"] = pd.to_datetime(df[colmap["closed_at"]], errors="coerce", utc=True) if colmap["closed_at"] else pd.NaT
    df["__state"] = df[colmap["state"]].apply(lambda x: (clean_text(x) or "").lower()) if colmap["state"] else ""
    df["__comment_count"] = pd.to_numeric(df[colmap["comment_count"]], errors="coerce").fillna(0) if colmap["comment_count"] else 0

    df["__labels_list"] = df.apply(lambda row: parse_row_labels(row, label_cols), axis=1)
    df["__labels_json"] = df["__labels_list"].apply(lambda x: json.dumps(x, ensure_ascii=False))

    if colmap["is_wontfix_labeled"]:
        df["__is_wontfix"] = parse_boolish(df[colmap["is_wontfix_labeled"]])
    else:
        df["__is_wontfix"] = df["__labels_list"].apply(lambda labels: labels_contain_any(labels, wontfix_labels))
    df["__is_invalid"] = df["__labels_list"].apply(lambda labels: labels_contain_any(labels, invalid_labels))

    # Corrected issue-type logic: label-derived only. No scalar type/author_type path exists.
    df["__issue_types_list"] = df["__labels_list"].apply(lambda labels: infer_issue_types(labels, issue_type_map))
    df["__issue_type"] = df["__issue_types_list"].apply(primary_issue_type)
    df["__issue_types_json"] = df["__issue_types_list"].apply(lambda x: json.dumps(x, ensure_ascii=False))

    df["__has_linked_pr"] = parse_boolish(df[colmap["linked_pr"]]) if colmap["linked_pr"] else False

    if issue_pr_links_df is not None and not issue_pr_links_df.empty:
        links = issue_pr_links_df.copy()
        repo_col = find_column(links, ["repo_full_name", "repo_name", "repo", "full_name"])
        issue_num_col = find_column(links, ["issue_number", "number"])
        issue_id_col = find_column(links, ["issue_id", "id", "node_id"])

        if repo_col and issue_num_col and colmap["issue_number"]:
            link_repo = links[repo_col].apply(clean_text).astype(str)
            link_num = pd.to_numeric(links[issue_num_col], errors="coerce")
            valid = link_num.notna()
            linked_keys = set(zip(link_repo[valid], link_num[valid].astype(int)))
            issue_num = pd.to_numeric(df[colmap["issue_number"]], errors="coerce")
            df["__has_linked_pr"] = df["__has_linked_pr"] | pd.Series(
                [(repo, int(num)) in linked_keys if pd.notna(num) else False for repo, num in zip(df["__repo"], issue_num)],
                index=df.index,
            )
        elif repo_col and issue_id_col:
            linked_keys = set(zip(links[repo_col].apply(clean_text).astype(str), links[issue_id_col].apply(clean_text).astype(str)))
            df["__has_linked_pr"] = df["__has_linked_pr"] | pd.Series(
                [(repo, issue_id) in linked_keys for repo, issue_id in zip(df["__repo"], df["__issue_id"])],
                index=df.index,
            )

    df["__is_open"] = df["__state"].eq("open")
    df["__is_closed"] = df["__state"].eq("closed") | df["__closed_at"].notna()

    df["__comparison_bucket"] = "other"
    non_wontfix = ~df["__is_wontfix"]
    df.loc[df["__is_open"] & non_wontfix, "__comparison_bucket"] = "open"
    df.loc[df["__is_closed"] & non_wontfix, "__comparison_bucket"] = "closed_non_wontfix"
    df.loc[df["__is_invalid"] & non_wontfix, "__comparison_bucket"] = "invalid"
    df.loc[df["__is_closed"] & non_wontfix & df["__has_linked_pr"], "__comparison_bucket"] = "resolved_pr"

    return df, colmap


def candidate_score(wf: pd.Series, cand: pd.Series) -> float:
    score = 0.0
    if pd.notna(wf["__created_at"]) and pd.notna(cand["__created_at"]):
        score += min(abs((cand["__created_at"] - wf["__created_at"]).days), 3650) / 30.0
    else:
        score += 50
    score += min(abs(float(cand["__comment_count"]) - float(wf["__comment_count"])), 100)
    if type_overlap(wf["__issue_types_list"], cand["__issue_types_list"]):
        score -= 15
    if bool(cand["__has_linked_pr"]):
        score -= 8
    score += {"resolved_pr": -12, "closed_non_wontfix": -8, "invalid": -3, "open": 0, "other": 3}.get(cand["__comparison_bucket"], 3)
    return float(score)


def add_pair_diagnostics(cands: pd.DataFrame, wf: pd.Series) -> pd.DataFrame:
    cands = cands.copy()
    if pd.notna(wf["__created_at"]):
        cands["__created_at_day_diff"] = (cands["__created_at"] - wf["__created_at"]).abs().dt.days
    else:
        cands["__created_at_day_diff"] = pd.NA
    cands["__comment_count_abs_diff"] = (cands["__comment_count"].astype(float) - float(wf["__comment_count"])).abs()
    cands["__same_issue_type_flag"] = cands["__issue_types_list"].apply(lambda t: int(type_overlap(wf["__issue_types_list"], t)))
    return cands


def select_comparison_controls(df: pd.DataFrame, max_controls: int, time_window_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    wontfix_df = df[df["__is_wontfix"]].copy()
    non_wontfix_df = df[~df["__is_wontfix"]].copy()
    selected_rows: list[pd.Series] = []
    used_keys: set[str] = set()

    for _, wf in wontfix_df.iterrows():
        cands = non_wontfix_df[non_wontfix_df["__repo"] == wf["__repo"]].copy()
        if pd.notna(wf["__created_at"]):
            lo = wf["__created_at"] - pd.Timedelta(days=time_window_days)
            hi = wf["__created_at"] + pd.Timedelta(days=time_window_days)
            cands = cands[cands["__created_at"].between(lo, hi, inclusive="both")]

        if wf["__issue_types_list"]:
            same_type = cands[cands["__issue_types_list"].apply(lambda t: type_overlap(wf["__issue_types_list"], t))].copy()
            if not same_type.empty:
                cands = same_type

        cands = cands[~cands["__issue_key"].isin(used_keys)].copy()
        if cands.empty:
            continue

        cands = add_pair_diagnostics(cands, wf)
        cands["__match_score"] = cands.apply(lambda r: candidate_score(wf, r), axis=1)
        cands = cands.sort_values(
            ["__match_score", "__has_linked_pr", "__comment_count", "__created_at"],
            ascending=[True, False, True, True],
        )
        chosen = cands.head(max_controls).copy()
        if chosen.empty:
            continue

        chosen["__match_rank_for_wontfix"] = range(1, len(chosen) + 1)
        chosen["__matched_to_wontfix_issue_id"] = wf["__issue_id"]
        chosen["__matched_to_wontfix_issue_key"] = wf["__issue_key"]
        chosen["__matched_to_repo"] = wf["__repo"]
        chosen["__matched_to_issue_type"] = wf["__issue_type"]
        chosen["__matched_to_issue_types_json"] = json.dumps(wf["__issue_types_list"], ensure_ascii=False)
        chosen["__matched_to_wontfix_issue_number"] = wf["__issue_number"]
        chosen["__matched_to_wontfix_created_at"] = wf["__created_at"]
        chosen["__matched_to_wontfix_comment_count"] = wf["__comment_count"]
        selected_rows.extend([row for _, row in chosen.iterrows()])
        used_keys.update(chosen["__issue_key"].tolist())

    comparison_df = pd.DataFrame(selected_rows) if selected_rows else df.iloc[0:0].copy()
    return wontfix_df, comparison_df


# -----------------------------------------------------------------------------
# Outputs and QA
# -----------------------------------------------------------------------------


def build_pair_mapping(comparison_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "repo_full_name", "matched_set_id", "wontfix_issue_id", "wontfix_issue_number",
        "comparison_issue_id", "comparison_issue_number", "match_rank_for_wontfix",
        "match_score", "comparison_bucket", "wontfix_issue_type", "comparison_issue_type",
        "wontfix_issue_types_json", "comparison_issue_types_json", "same_issue_type_flag",
        "created_at_day_diff", "wontfix_created_at", "comparison_created_at",
        "wontfix_comment_count", "comparison_comment_count", "comment_count_abs_diff",
        "comparison_has_linked_pr", "comparison_is_invalid", "comparison_state",
    ]
    if comparison_df.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({
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
        "wontfix_issue_types_json": comparison_df["__matched_to_issue_types_json"],
        "comparison_issue_types_json": comparison_df["__issue_types_json"],
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
    out["matched_set_id"] = out["repo_full_name"].astype(str) + "::" + out["wontfix_issue_id"].astype(str)
    return out[cols].drop_duplicates().sort_values(
        ["repo_full_name", "wontfix_issue_number", "match_rank_for_wontfix"], kind="stable"
    ).reset_index(drop=True)


def build_issue_level_lookup(pair_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["repo_full_name", "issue_id", "issue_number", "analysis_set", "matched_set_id"]
    if pair_df.empty:
        return pd.DataFrame(columns=cols)
    wf = pair_df[["repo_full_name", "wontfix_issue_id", "wontfix_issue_number", "matched_set_id"]].copy()
    wf = wf.rename(columns={"wontfix_issue_id": "issue_id", "wontfix_issue_number": "issue_number"})
    wf["analysis_set"] = "wontfix"
    cmp = pair_df[["repo_full_name", "comparison_issue_id", "comparison_issue_number", "matched_set_id"]].copy()
    cmp = cmp.rename(columns={"comparison_issue_id": "issue_id", "comparison_issue_number": "issue_number"})
    cmp["analysis_set"] = "comparison"
    out = pd.concat([wf, cmp], ignore_index=True)
    out["issue_id"] = out["issue_id"].astype(str)
    return out[cols].drop_duplicates().reset_index(drop=True)


def controls_per_wontfix_counts(wontfix_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.Series:
    """Return one row per WONTFIX issue, with 0 for unmatched WONTFIX issues."""
    all_set_ids = wontfix_df["__issue_key"].astype(str) if not wontfix_df.empty else pd.Series(dtype="object")
    if pair_df.empty:
        return pd.Series(0, index=all_set_ids, dtype="int64")
    pair_counts = pair_df.groupby("matched_set_id").size().astype(int)
    return pair_counts.reindex(all_set_ids, fill_value=0).astype(int)


def controls_per_wontfix_distribution(counts: pd.Series, max_controls: int) -> dict[str, int]:
    vc = counts.value_counts().sort_index()
    upper = max(max_controls, int(vc.index.max()) if not vc.empty else max_controls)
    return {str(i): int(vc.get(i, 0)) for i in range(0, upper + 1)}


def write_controls_distribution_histogram(
    distribution: dict[str, int],
    project_root: Path,
    filename: str = "controls_per_wontfix_distribution.png",
) -> Path:
    """Create outputs/qa/matched_sets if needed and replace only the histogram PNG."""
    out_dir = project_root / "outputs" / "qa" / "matched_sets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_labels = list(distribution.keys())
    y_values = [distribution[k] for k in x_labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x_labels, y_values)
    ax.set_xlabel("Number of controls matched to WONTFIX issue")
    ax.set_ylabel("Number of WONTFIX issues")
    ax.set_title("Controls per WONTFIX matched set")
    for x, y in zip(x_labels, y_values):
        ax.text(x, y, str(y), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def build_qa_summary(
    full_df: pd.DataFrame,
    wontfix_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    max_controls: int,
    time_window_days: int,
    colmap: dict[str, Any],
    histogram_path: Path | None = None,
) -> pd.DataFrame:
    controls_per_set = pair_df.groupby("matched_set_id").size() if not pair_df.empty else pd.Series(dtype=int)
    controls_per_wontfix = controls_per_wontfix_counts(wontfix_df, pair_df)
    controls_distribution = controls_per_wontfix_distribution(controls_per_wontfix, max_controls)

    matched_keys = set(comparison_df["__matched_to_wontfix_issue_key"].astype(str)) if not comparison_df.empty else set()
    non_wontfix = full_df[~full_df["__is_wontfix"]].copy()
    wontfix_keys = set(wontfix_df["__issue_key"]) if not wontfix_df.empty else set()
    comparison_keys = set(comparison_df["__issue_key"]) if not comparison_df.empty else set()
    selected_keys = set(comparison_df["__issue_key"]) if not comparison_df.empty else set()
    issue_type_counts = full_df["__issue_type"].fillna("missing").value_counts(dropna=False).to_dict()
    selected_type_counts = comparison_df["__issue_type"].fillna("missing").value_counts(dropna=False).to_dict() if not comparison_df.empty else {}

    summary = {
        "total_issues": int(len(full_df)),
        "total_non_wontfix_pool_issues": int(len(non_wontfix)),
        "total_wontfix_issues": int(len(wontfix_df)),
        "total_selected_comparison_issues": int(len(comparison_df)),
        "pair_rows": int(len(pair_df)),
        "matched_sets": int(pair_df["matched_set_id"].nunique()) if not pair_df.empty else 0,
        "selected_resolved_pr": int((comparison_df["__comparison_bucket"] == "resolved_pr").sum()) if not comparison_df.empty else 0,
        "selected_closed_non_wontfix": int((comparison_df["__comparison_bucket"] == "closed_non_wontfix").sum()) if not comparison_df.empty else 0,
        "selected_invalid": int((comparison_df["__comparison_bucket"] == "invalid").sum()) if not comparison_df.empty else 0,
        "selected_open": int((comparison_df["__comparison_bucket"] == "open").sum()) if not comparison_df.empty else 0,
        "selected_other": int((comparison_df["__comparison_bucket"] == "other").sum()) if not comparison_df.empty else 0,
        "pool_resolved_pr": int((non_wontfix["__comparison_bucket"] == "resolved_pr").sum()) if not non_wontfix.empty else 0,
        "pool_closed_non_wontfix": int((non_wontfix["__comparison_bucket"] == "closed_non_wontfix").sum()) if not non_wontfix.empty else 0,
        "pool_invalid": int((non_wontfix["__comparison_bucket"] == "invalid").sum()) if not non_wontfix.empty else 0,
        "pool_open": int((non_wontfix["__comparison_bucket"] == "open").sum()) if not non_wontfix.empty else 0,
        "pool_other": int((non_wontfix["__comparison_bucket"] == "other").sum()) if not non_wontfix.empty else 0,
        "avg_controls_per_wontfix": float(round(len(comparison_df) / len(wontfix_df), 4)) if len(wontfix_df) else 0.0,
        "min_controls_per_matched_wontfix": int(controls_per_set.min()) if not controls_per_set.empty else 0,
        "max_controls_per_matched_wontfix": int(controls_per_set.max()) if not controls_per_set.empty else 0,
        "wontfix_issues_with_zero_matches": int((controls_per_wontfix == 0).sum()),
        "matched_sets_over_max_controls": int((controls_per_set > max_controls).sum()) if not controls_per_set.empty else 0,
        "max_controls_per_wontfix": int(max_controls),
        "time_window_days": int(time_window_days),
        "controls_per_wontfix_distribution_json": json.dumps(controls_distribution, sort_keys=True),
        "controls_per_wontfix_histogram_path": str(histogram_path) if histogram_path else "",
        "wontfix_duplicate_issue_keys": int(wontfix_df.duplicated(subset=["__issue_key"]).sum()) if not wontfix_df.empty else 0,
        "comparison_duplicate_issue_keys": int(comparison_df.duplicated(subset=["__issue_key"]).sum()) if not comparison_df.empty else 0,
        "wontfix_comparison_overlap_issue_keys": int(len(wontfix_keys & comparison_keys)),
        "comparison_rows_with_wontfix_label": int(comparison_df["__is_wontfix"].sum()) if not comparison_df.empty else 0,
        "comparison_issue_reused_count": int(pair_df.duplicated(subset=["repo_full_name", "comparison_issue_id"]).sum()) if not pair_df.empty else 0,
        "pair_duplicate_rows": int(pair_df.duplicated().sum()) if not pair_df.empty else 0,
        "pair_rows_cross_repo": int((comparison_df["__repo"] != comparison_df["__matched_to_repo"]).sum()) if not comparison_df.empty else 0,
        "pair_rows_outside_time_window": int((pd.to_numeric(comparison_df.get("__created_at_day_diff", pd.Series(dtype=float)), errors="coerce") > time_window_days).sum()) if not comparison_df.empty else 0,
        "pair_rows_same_issue_type_selected": int(pair_df["same_issue_type_flag"].sum()) if not pair_df.empty else 0,
        "pair_rows_issue_type_mismatch_selected": int((pair_df["same_issue_type_flag"] == 0).sum()) if not pair_df.empty else 0,
        "pair_rows_missing_wontfix_issue_type": int(pair_df["wontfix_issue_type"].isna().sum()) if not pair_df.empty else 0,
        "pair_rows_missing_comparison_issue_type": int(pair_df["comparison_issue_type"].isna().sum()) if not pair_df.empty else 0,
        "selected_comparison_keys_not_in_non_wontfix_pool": int(len(selected_keys - set(non_wontfix["__issue_key"]))),
        "label_columns_json": json.dumps(colmap.get("label_columns", [])),
        "issue_type_source": colmap.get("issue_type_source"),
        "issue_type_column_mapping_json": json.dumps(colmap.get("issue_type", [])),
        "issues_with_any_label": int(full_df["__labels_list"].apply(bool).sum()),
        "issue_type_from_labels_rows": int(full_df["__issue_type"].notna().sum()),
        "issue_type_missing_rows": int(full_df["__issue_type"].isna().sum()),
        "issue_type_multitype_rows": int(full_df["__issue_types_list"].apply(lambda x: len(x) > 1).sum()),
        "issue_type_value_counts_json": json.dumps(issue_type_counts, sort_keys=True),
        "selected_issue_type_value_counts_json": json.dumps(selected_type_counts, sort_keys=True),
    }

    flags: list[str] = []
    for condition, flag in [
        (summary["comparison_rows_with_wontfix_label"], "comparison_contains_wontfix_labeled_rows"),
        (summary["wontfix_comparison_overlap_issue_keys"], "wontfix_comparison_overlap_detected"),
        (summary["pair_rows_cross_repo"], "cross_repo_pairs_detected"),
        (summary["pair_rows_outside_time_window"], "pairs_outside_time_window_detected"),
        (summary["comparison_issue_reused_count"], "comparison_issue_reused_across_sets"),
        (summary["matched_sets_over_max_controls"], "matched_set_exceeds_max_controls"),
        (summary["wontfix_issues_with_zero_matches"], "some_wontfix_issues_unmatched"),
        ("user" in issue_type_counts, "issue_type_contains_user_value_check_label_parser"),
    ]:
        if condition:
            flags.append(flag)
    summary["quality_flags_json"] = json.dumps(flags)
    summary["quality_status"] = "ok" if not flags else "check"
    return pd.DataFrame([summary])


def drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if str(c).startswith("__")], errors="ignore")


def maybe_load_issue_pr_links(path: Path | None) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None
    return read_table(path)


def write_lookup(pair_df: pd.DataFrame, pair_output_path: Path) -> Path:
    lookup = build_issue_level_lookup(pair_df)
    lookup_path = pair_output_path.with_name(pair_output_path.stem + "_issue_lookup.csv")
    lookup.to_csv(lookup_path, index=False)
    return lookup_path


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WONTFIX and matched non-WONTFIX comparison issue sets.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_study_config(args.config)

    issues_path, wontfix_out, comparison_out, qa_path, pair_out, issue_pr_path = pick_paths(config)
    max_controls, time_window_days = resolve_matching_settings(config)
    wontfix_labels, invalid_labels, issue_type_map = resolve_label_settings(config)

    print(f"Script version: {SCRIPT_VERSION}")
    print("Loaded config successfully.")
    print(f"Issues input path: {issues_path}")
    print(f"Issue-PR links input path: {issue_pr_path}")
    print(f"WONTFIX output path: {wontfix_out}")
    print(f"Comparison output path: {comparison_out}")
    print(f"QA summary path: {qa_path}")
    print(f"Pair mapping output path: {pair_out}")
    print(f"Max controls per WONTFIX: {max_controls}")
    print(f"Time window (days): {time_window_days}")
    print("Issue type inference source: label JSON columns only; scalar type/author_type columns are ignored.")

    issues_df = read_table(issues_path)
    issue_pr_df = maybe_load_issue_pr_links(issue_pr_path)
    print(f"Loaded issues table with {len(issues_df)} rows.")
    print(f"Loaded issue_pr_links table with {len(issue_pr_df)} rows." if issue_pr_df is not None else "Issue-PR links table not found or not configured.")
    print("Columns:")
    print(list(issues_df.columns))

    derived, colmap = add_derived_columns(issues_df, wontfix_labels, invalid_labels, issue_type_map, issue_pr_df)
    print("Resolved column mapping:")
    print(colmap)
    print("Issue type value counts:")
    print(derived["__issue_type"].fillna("missing").value_counts(dropna=False).to_string())

    wontfix_df, comparison_df = select_comparison_controls(derived, max_controls, time_window_days)
    pair_df = build_pair_mapping(comparison_df)

    controls_distribution = controls_per_wontfix_distribution(
        controls_per_wontfix_counts(wontfix_df, pair_df),
        max_controls,
    )
    histogram_path = write_controls_distribution_histogram(controls_distribution, PROJECT_ROOT)
    print(f"Saved controls-per-WONTFIX histogram: {histogram_path}")
    print(f"Controls-per-WONTFIX distribution: {json.dumps(controls_distribution, sort_keys=True)}")

    qa_df = build_qa_summary(
        derived,
        wontfix_df,
        comparison_df,
        pair_df,
        max_controls,
        time_window_days,
        colmap,
        histogram_path=histogram_path,
    )

    write_table(drop_internal_columns(wontfix_df), wontfix_out)
    write_table(drop_internal_columns(comparison_df), comparison_out)
    ensure_parent_dir(qa_path)
    qa_df.to_csv(qa_path, index=False)
    write_table(pair_df, pair_out)
    lookup_path = write_lookup(pair_df, pair_out)

    print(f"Saved WONTFIX issue set: {wontfix_out} ({len(wontfix_df)} rows)")
    print(f"Saved comparison issue set: {comparison_out} ({len(comparison_df)} rows)")
    print(f"Saved QA summary CSV: {qa_path}")
    print(f"Saved pair mapping: {pair_out} ({len(pair_df)} rows)")
    print(f"Saved issue-level matched-set lookup CSV: {lookup_path}")
    print("QA status:")
    print(qa_df.to_string(index=False))


if __name__ == "__main__":
    main()
