from __future__ import annotations

from pathlib import Path
import ast
import json
import math
import re
from typing import Any

import pandas as pd

from config.study_config_loader import load_study_config


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
    "bug": ["bug", "type: bug", "kind/bug"],
    "feature": ["enhancement", "feature", "type: feature", "kind/feature"],
    "documentation": ["documentation", "docs", "type: docs", "kind/documentation"],
    "question": ["question", "type: question"],
}


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
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    for col in df.columns:
        col_l = col.lower()
        for cand in candidates:
            if cand.lower() in col_l:
                return col

    if required:
        raise KeyError(f"Could not find required column. Tried: {candidates}")
    return None


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

    # try JSON / Python literal list
    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, list):
                    return [str(x).strip().lower() for x in parsed if str(x).strip()]
            except Exception:
                pass

    # common separators
    parts = re.split(r"[|,;]", text)
    if len(parts) > 1:
        return [p.strip().lower() for p in parts if p.strip()]

    return [text.lower()]


def contains_any(labels: list[str], keywords: list[str]) -> bool:
    label_blob = " | ".join(labels)
    for kw in keywords:
        kw_l = kw.lower().strip()
        if kw_l and kw_l in label_blob:
            return True
    return False


def infer_issue_type(labels: list[str], issue_type_map: dict[str, list[str]]) -> str | None:
    for issue_type, aliases in issue_type_map.items():
        if contains_any(labels, aliases):
            return issue_type
    return None


def normalize_datetime(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    return pd.to_datetime(series, errors="coerce", utc=True)


def normalize_numeric(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    return pd.to_numeric(series, errors="coerce").fillna(0)


def pick_paths(config: Any) -> tuple[Path, Path, Path, Path]:
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

    return issues_path, wontfix_output_path, comparison_output_path, qa_summary_path


def resolve_matching_settings(config: Any) -> tuple[int, int]:
    max_controls = (
        get_cfg(config, "comparison_set", "max_controls_per_wontfix")
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
        get_cfg(config, "comparison_set", "wontfix_label_variants")
        or get_cfg(config, "labels", "wontfix")
        or DEFAULT_WONTFIX_LABELS
    )

    invalid_labels = (
        get_cfg(config, "comparison_set", "invalid_label_variants")
        or get_cfg(config, "labels", "invalid")
        or DEFAULT_INVALID_LABELS
    )

    issue_type_map = (
        get_cfg(config, "comparison_set", "issue_type_label_groups")
        or get_cfg(config, "issue_types")
        or DEFAULT_ISSUE_TYPE_MAP
    )

    return list(wontfix_labels), list(invalid_labels), dict(issue_type_map)


def add_derived_columns(
    issues_df: pd.DataFrame,
    wontfix_labels: list[str],
    invalid_labels: list[str],
    issue_type_map: dict[str, list[str]],
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    df = issues_df.copy()

    colmap = {
        "issue_id": find_column(df, ["issue_id", "id", "number"], required=True),
        "repo": find_column(df, ["repo_name", "repository_name", "repo", "full_name"], required=True),
        "created_at": find_column(df, ["created_at", "issue_created_at", "created"], required=True),
        "closed_at": find_column(df, ["closed_at", "issue_closed_at", "closed"]),
        "state": find_column(df, ["state", "issue_state", "status"]),
        "labels": find_column(df, ["labels", "label_names", "issue_labels"], required=True),
        "comment_count": find_column(df, ["comments", "comment_count", "num_comments", "comments_count"]),
        "linked_pr": find_column(df, ["has_linked_pr", "linked_pr", "linked_pull_request", "has_pr", "pull_request"]),
        "issue_type": find_column(df, ["issue_type", "type"]),
    }

    df["__issue_id"] = df[colmap["issue_id"]].astype(str)
    df["__repo"] = df[colmap["repo"]].astype(str)
    df["__created_at"] = normalize_datetime(df[colmap["created_at"]])
    df["__closed_at"] = normalize_datetime(df[colmap["closed_at"]]) if colmap["closed_at"] else pd.NaT
    df["__state"] = df[colmap["state"]].astype(str).str.lower() if colmap["state"] else ""
    df["__comment_count"] = (
        normalize_numeric(df[colmap["comment_count"]]) if colmap["comment_count"] else 0
    )

    df["__labels_list"] = df[colmap["labels"]].apply(parse_label_value)
    df["__is_wontfix"] = df["__labels_list"].apply(lambda x: contains_any(x, wontfix_labels))
    df["__is_invalid"] = df["__labels_list"].apply(lambda x: contains_any(x, invalid_labels))

    if colmap["issue_type"]:
        df["__issue_type"] = df[colmap["issue_type"]].astype(str).str.lower().replace({"nan": None})
    else:
        df["__issue_type"] = df["__labels_list"].apply(lambda x: infer_issue_type(x, issue_type_map))

    if colmap["linked_pr"]:
        raw = df[colmap["linked_pr"]]
        if pd.api.types.is_bool_dtype(raw):
            df["__has_linked_pr"] = raw.fillna(False)
        else:
            raw_num = pd.to_numeric(raw, errors="coerce")
            if raw_num.notna().any():
                df["__has_linked_pr"] = raw_num.fillna(0) > 0
            else:
                df["__has_linked_pr"] = raw.astype(str).str.lower().isin(
                    {"true", "1", "yes", "y"}
                )
    else:
        df["__has_linked_pr"] = False

    df["__is_open"] = df["__state"].eq("open")
    df["__is_closed"] = df["__state"].eq("closed") | df["__closed_at"].notna()

    df["__comparison_bucket"] = "other"
    df.loc[df["__is_closed"] & ~df["__is_wontfix"] & df["__has_linked_pr"], "__comparison_bucket"] = "resolved_pr"
    df.loc[df["__is_closed"] & ~df["__is_wontfix"], "__comparison_bucket"] = "closed_non_wontfix"
    df.loc[df["__is_invalid"] & ~df["__is_wontfix"], "__comparison_bucket"] = "invalid"
    df.loc[df["__is_open"] & ~df["__is_wontfix"], "__comparison_bucket"] = "open"

    return df, colmap


def candidate_score(wf_row: pd.Series, cand_row: pd.Series) -> float:
    score = 0.0

    # lower is better
    if pd.notna(wf_row["__created_at"]) and pd.notna(cand_row["__created_at"]):
        day_diff = abs((cand_row["__created_at"] - wf_row["__created_at"]).days)
        score += min(day_diff, 3650) / 30.0
    else:
        score += 50

    comment_diff = abs(float(cand_row["__comment_count"]) - float(wf_row["__comment_count"]))
    score += min(comment_diff, 100)

    # bonuses as negative score
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

    return score


def select_comparison_controls(
    derived_df: pd.DataFrame,
    max_controls_per_wontfix: int,
    time_window_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    wontfix_df = derived_df[derived_df["__is_wontfix"]].copy()
    non_wontfix_df = derived_df[~derived_df["__is_wontfix"]].copy()

    selected_rows: list[pd.Series] = []
    used_candidate_ids: set[str] = set()

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

        candidates = candidates[~candidates["__issue_id"].isin(used_candidate_ids)].copy()
        if candidates.empty:
            continue

        candidates["__match_score"] = candidates.apply(lambda r: candidate_score(wf_row, r), axis=1)
        candidates = candidates.sort_values(
            by=["__match_score", "__has_linked_pr", "__comment_count", "__created_at"],
            ascending=[True, False, True, True],
        )

        selected = candidates.head(max_controls_per_wontfix).copy()
        if selected.empty:
            continue

        selected["__matched_to_wontfix_issue_id"] = wf_row["__issue_id"]
        selected["__matched_to_repo"] = wf_row["__repo"]
        selected["__matched_to_issue_type"] = wf_row["__issue_type"]
        selected_rows.extend([row for _, row in selected.iterrows()])
        used_candidate_ids.update(selected["__issue_id"].tolist())

    comparison_df = pd.DataFrame(selected_rows) if selected_rows else derived_df.iloc[0:0].copy()
    return wontfix_df, comparison_df


def build_qa_summary(
    full_df: pd.DataFrame,
    wontfix_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    max_controls_per_wontfix: int,
    time_window_days: int,
) -> pd.DataFrame:
    matched_counts = (
        comparison_df.groupby("__matched_to_wontfix_issue_id").size()
        if not comparison_df.empty
        else pd.Series(dtype=int)
    )

    zero_match_count = 0
    if not wontfix_df.empty:
        zero_match_count = int(
            (~wontfix_df["__issue_id"].isin(set(matched_counts.index.astype(str)))).sum()
        )

    summary = {
        "total_issues": int(len(full_df)),
        "total_wontfix_issues": int(len(wontfix_df)),
        "total_selected_comparison_issues": int(len(comparison_df)),
        "selected_resolved_pr": int((comparison_df["__comparison_bucket"] == "resolved_pr").sum()) if not comparison_df.empty else 0,
        "selected_closed_non_wontfix": int((comparison_df["__comparison_bucket"] == "closed_non_wontfix").sum()) if not comparison_df.empty else 0,
        "selected_invalid": int((comparison_df["__comparison_bucket"] == "invalid").sum()) if not comparison_df.empty else 0,
        "selected_open": int((comparison_df["__comparison_bucket"] == "open").sum()) if not comparison_df.empty else 0,
        "avg_controls_per_wontfix": float(round(len(comparison_df) / len(wontfix_df), 4)) if len(wontfix_df) else 0.0,
        "wontfix_issues_with_zero_matches": zero_match_count,
        "max_controls_per_wontfix": int(max_controls_per_wontfix),
        "time_window_days": int(time_window_days),
    }
    return pd.DataFrame([summary])


def drop_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    internal_cols = [c for c in df.columns if c.startswith("__")]
    return df.drop(columns=internal_cols, errors="ignore")


def main():
    config = load_study_config("config/study_config.yaml")

    issues_path, wontfix_output_path, comparison_output_path, qa_summary_path = pick_paths(config)
    max_controls_per_wontfix, time_window_days = resolve_matching_settings(config)
    wontfix_labels, invalid_labels, issue_type_map = resolve_label_settings(config)

    print("Loaded config successfully.")
    print(f"Issues input path: {issues_path}")
    print(f"WONTFIX output path: {wontfix_output_path}")
    print(f"Comparison output path: {comparison_output_path}")
    print(f"QA summary path: {qa_summary_path}")
    print(f"Max controls per WONTFIX: {max_controls_per_wontfix}")
    print(f"Time window (days): {time_window_days}")

    if not issues_path.exists():
        raise FileNotFoundError(f"Issues table not found: {issues_path}")

    issues_df = pd.read_parquet(issues_path)
    print(f"Loaded issues table with {len(issues_df)} rows.")
    print("Columns:")
    print(list(issues_df.columns))

    derived_df, colmap = add_derived_columns(
        issues_df=issues_df,
        wontfix_labels=wontfix_labels,
        invalid_labels=invalid_labels,
        issue_type_map=issue_type_map,
    )

    print("Resolved column mapping:")
    print(colmap)

    wontfix_df, comparison_df = select_comparison_controls(
        derived_df=derived_df,
        max_controls_per_wontfix=max_controls_per_wontfix,
        time_window_days=time_window_days,
    )

    qa_df = build_qa_summary(
        full_df=derived_df,
        wontfix_df=wontfix_df,
        comparison_df=comparison_df,
        max_controls_per_wontfix=max_controls_per_wontfix,
        time_window_days=time_window_days,
    )

    ensure_parent_dir(wontfix_output_path)
    ensure_parent_dir(comparison_output_path)
    ensure_parent_dir(qa_summary_path)

    drop_internal_columns(wontfix_df).to_parquet(wontfix_output_path, index=False)
    drop_internal_columns(comparison_df).to_parquet(comparison_output_path, index=False)
    qa_df.to_csv(qa_summary_path, index=False)

    print(f"Saved WONTFIX issue set: {wontfix_output_path} ({len(wontfix_df)} rows)")
    print(f"Saved comparison issue set: {comparison_output_path} ({len(comparison_df)} rows)")
    print(f"Saved QA summary CSV: {qa_summary_path}")


if __name__ == "__main__":
    main()