import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.study_config_loader import load_study_config

def clean_text(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None

def to_datetime(series):
    if series is None:
        return pd.Series(dtype="datetime64[ns, UTC]")
    return pd.to_datetime(series, errors="coerce", utc=True)

def safe_divide(numer, denom, default=np.nan):
    try:
        if denom is None or pd.isna(denom) or float(denom) == 0.0:
            return default
        return float(numer) / float(denom)
    except Exception:
        return default

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def maybe_read_parquet(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

def find_col(df, candidates, required=False):
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for col in df.columns:
        col_l = str(col).lower()
        for cand in candidates:
            if cand.lower() in col_l:
                return str(col)
    if required:
        raise KeyError(f"Required column not found. Tried: {candidates}")
    return None

def normalize_issue_key_frame(df):
    if df.empty:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number"])

    repo_col = find_col(df, ["repo_full_name", "repo_name", "full_name", "repo"], required=True)
    issue_id_col = find_col(df, ["issue_id", "id"], required=False)
    issue_num_col = find_col(df, ["issue_number", "number"], required=False)

    out = pd.DataFrame()
    out["repo_full_name"] = df[repo_col].astype(str)
    out["issue_id"] = df[issue_id_col].astype(str) if issue_id_col else None
    out["issue_number"] = pd.to_numeric(df[issue_num_col], errors="coerce") if issue_num_col else np.nan
    out = out.drop_duplicates().reset_index(drop=True)
    return out

def build_issue_number_key(repo_full_name, issue_number):
    if repo_full_name is None or pd.isna(issue_number):
        return None
    return f"{str(repo_full_name)}::num::{int(issue_number)}"

def build_issue_id_key(repo_full_name, issue_id):
    issue_id = clean_text(issue_id)
    if repo_full_name is None or not issue_id:
        return None
    return f"{str(repo_full_name)}::id::{issue_id}"

def add_issue_key_columns(df):
    if df.empty:
        out = df.copy()
        out["issue_key_number"] = pd.Series(dtype="object")
        out["issue_key_id"] = pd.Series(dtype="object")
        return out

    out = df.copy()
    if "issue_number" in out.columns:
        out["issue_number"] = pd.to_numeric(out["issue_number"], errors="coerce")
    else:
        out["issue_number"] = np.nan

    if "issue_id" not in out.columns:
        out["issue_id"] = None

    out["issue_key_number"] = [
        build_issue_number_key(repo_name, issue_number)
        for repo_name, issue_number in zip(out["repo_full_name"], out["issue_number"])
    ]
    out["issue_key_id"] = [
        build_issue_id_key(repo_name, issue_id)
        for repo_name, issue_id in zip(out["repo_full_name"], out["issue_id"])
    ]
    return out

def normalize_pull_requests(df):
    if df.empty:
        return pd.DataFrame(columns=["repo_full_name", "pr_id", "pr_number", "merge_commit_sha", "head_sha"])

    out = pd.DataFrame()
    out["repo_full_name"] = df[find_col(df, ["repo_full_name"], required=True)].astype(str)

    pr_id_col = find_col(df, ["pr_id"])
    pr_number_col = find_col(df, ["pr_number", "number"])
    merge_sha_col = find_col(df, ["merge_commit_sha"])
    head_sha_col = find_col(df, ["head_sha"])

    out["pr_id"] = pd.to_numeric(df[pr_id_col], errors="coerce") if pr_id_col else np.nan
    out["pr_number"] = pd.to_numeric(df[pr_number_col], errors="coerce") if pr_number_col else np.nan
    out["merge_commit_sha"] = df[merge_sha_col].apply(clean_text) if merge_sha_col else None
    out["head_sha"] = df[head_sha_col].apply(clean_text) if head_sha_col else None

    return out.drop_duplicates().reset_index(drop=True)

def normalize_issue_pr_links(df):
    if df.empty:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number", "pr_id", "pr_number"])
    out = pd.DataFrame()
    out["repo_full_name"] = df[find_col(df, ["repo_full_name"], required=True)].astype(str)
    issue_id_col = find_col(df, ["issue_id", "id"])
    issue_num_col = find_col(df, ["issue_number", "number"])
    pr_id_col = find_col(df, ["pr_id"])
    pr_num_col = find_col(df, ["pr_number", "number"])
    out["issue_id"] = df[issue_id_col].astype(str) if issue_id_col else None
    out["issue_number"] = pd.to_numeric(df[issue_num_col], errors="coerce") if issue_num_col else np.nan
    out["pr_id"] = pd.to_numeric(df[pr_id_col], errors="coerce") if pr_id_col else np.nan
    out["pr_number"] = pd.to_numeric(df[pr_num_col], errors="coerce") if pr_num_col else np.nan
    return out.drop_duplicates().reset_index(drop=True)

def normalize_issue_comments(df):
    if df.empty:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number", "comment_author_contributor_key"])
    out = pd.DataFrame()
    out["repo_full_name"] = df[find_col(df, ["repo_full_name"], required=True)].astype(str)
    issue_id_col = find_col(df, ["issue_id", "id"])
    issue_num_col = find_col(df, ["issue_number", "number"], required=True)
    author_key_col = find_col(df, ["comment_author_contributor_key", "resolved_contributor_key"])
    author_login_col = find_col(df, ["author_login", "comment_author_login"])
    out["issue_id"] = df[issue_id_col].astype(str) if issue_id_col else None
    out["issue_number"] = pd.to_numeric(df[issue_num_col], errors="coerce")
    if author_key_col:
        out["comment_author_contributor_key"] = df[author_key_col].apply(clean_text)
    elif author_login_col:
        out["comment_author_contributor_key"] = df[author_login_col].apply(clean_text)
    else:
        out["comment_author_contributor_key"] = None
    return out

def normalize_issue_features(df):
    if df.empty:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number"])
    out = df.copy()
    if "issue_created_at" in out.columns:
        out["issue_created_at"] = to_datetime(out["issue_created_at"])
    return out

def normalize_evidence(df):
    if df.empty:
        return pd.DataFrame(columns=[
            "repo_full_name",
            "issue_id",
            "issue_number",
            "commit_sha",
            "commit_timestamp",
            "commit_author_contributor_key",
            "evidence_type",
            "evidence_selected_for_features",
            "selected_for_high_confidence_features",
            "selected_for_conservative_pre_issue_fallback",
            "selected_for_any_features",
            "ownership_time_bucket",
            "issue_file_confidence_level",
        ])

    out = df.copy()

    if "commit_timestamp" in out.columns:
        out["commit_timestamp"] = to_datetime(out["commit_timestamp"])
    if "issue_created_at" in out.columns:
        out["issue_created_at"] = to_datetime(out["issue_created_at"])

    for numeric_flag_col in [
        "evidence_selected_for_features",
        "selected_for_high_confidence_features",
        "selected_for_conservative_pre_issue_fallback",
        "selected_for_any_features",
    ]:
        if numeric_flag_col in out.columns:
            out[numeric_flag_col] = pd.to_numeric(out[numeric_flag_col], errors="coerce").fillna(0).astype(int)
        else:
            out[numeric_flag_col] = 0

    if "evidence_selected_for_features" not in df.columns and "selected_for_any_features" in out.columns:
        out["evidence_selected_for_features"] = out["selected_for_any_features"]

    for col in [
        "commit_author_contributor_key",
        "evidence_type",
        "ownership_time_bucket",
        "repo_full_name",
        "issue_id",
        "issue_file_confidence_level",
        "commit_sha",
        "file_path",
    ]:
        if col in out.columns:
            out[col] = out[col].apply(clean_text)

    if "issue_number" in out.columns:
        out["issue_number"] = pd.to_numeric(out["issue_number"], errors="coerce")
    else:
        out["issue_number"] = np.nan

    return out

def normalize_commits(df):
    if df.empty:
        return pd.DataFrame(columns=["repo_full_name", "commit_sha", "commit_timestamp"])
    out = df.copy()
    repo_col = find_col(out, ["repo_full_name"], required=True)
    sha_col = find_col(out, ["commit_sha"], required=True)
    ts_col = find_col(out, ["commit_timestamp", "timestamp"], required=True)
    author_key_col = find_col(out, ["commit_author_contributor_key", "resolved_contributor_key"])
    out2 = pd.DataFrame()
    out2["repo_full_name"] = out[repo_col].astype(str)
    out2["commit_sha"] = out[sha_col].astype(str)
    out2["commit_timestamp"] = to_datetime(out[ts_col])
    if author_key_col:
        out2["commit_author_contributor_key"] = out[author_key_col].apply(clean_text)
    else:
        out2["commit_author_contributor_key"] = None
    return out2.drop_duplicates(subset=["repo_full_name", "commit_sha"]).reset_index(drop=True)

def get_effective_git_window(cfg):
    git_cfg = cfg.git_history_extraction
    fast_mode = getattr(git_cfg, "fast_mode", False)
    include_full_history = getattr(git_cfg, "include_full_history", True)
    history_start = getattr(git_cfg, "history_start_date", None)
    history_end = getattr(git_cfg, "history_end_date", None)
    window_mode = getattr(git_cfg, "fast_mode_date_window", "participation_analysis")

    effective_include_full_history = include_full_history
    effective_start = history_start
    effective_end = history_end

    if fast_mode:
        effective_include_full_history = False
        if window_mode == "issue_collection":
            effective_start = cfg.study_windows.issue_collection.start_date
            effective_end = cfg.study_windows.issue_collection.end_date
        elif window_mode == "participation_analysis":
            effective_start = cfg.study_windows.participation_analysis.start_date
            effective_end = cfg.study_windows.participation_analysis.end_date
        elif window_mode == "explicit_history_dates":
            effective_start = history_start
            effective_end = history_end

    restriction_applies = (effective_start is not None) or (effective_end is not None)
    restriction_reason = None
    if fast_mode and include_full_history:
        restriction_reason = "fast_mode overrides include_full_history and applies the fast_mode_date_window"
    elif restriction_applies:
        restriction_reason = "explicit date bounds are applied to git log"
    else:
        restriction_reason = "no start/end date restriction is applied"

    return {
        "configured_include_full_history": include_full_history,
        "configured_fast_mode": fast_mode,
        "configured_fast_mode_date_window": window_mode,
        "configured_history_start_date": history_start,
        "configured_history_end_date": history_end,
        "effective_include_full_history": effective_include_full_history,
        "effective_history_start_date": effective_start,
        "effective_history_end_date": effective_end,
        "commit_window_restricted": bool(restriction_applies),
        "restriction_reason": restriction_reason,
    }

@dataclass
class Paths:
    issues_resolved: Path
    issue_comments_resolved: Path
    issue_pr_links: Path
    pull_requests: Path
    pr_commit_links: Path
    commits: Path
    commits_resolved: Path
    issue_ownership_features: Path
    issue_ownership_evidence: Path
    repo_list: Path

def resolve_paths(cfg):
    return Paths(
        issues_resolved=Path(cfg.outputs.issues_resolved_table),
        issue_comments_resolved=Path(cfg.outputs.issue_comments_resolved_table),
        issue_pr_links=Path(cfg.outputs.issue_pr_links_table),
        pull_requests=Path(cfg.outputs.pull_requests_table),
        pr_commit_links=Path(cfg.outputs.pr_commit_links_table),
        commits=Path(cfg.outputs.commits_table),
        commits_resolved=Path(cfg.outputs.commits_resolved_table),
        issue_ownership_features=Path(cfg.outputs.issue_ownership_features_table),
        issue_ownership_evidence=Path(cfg.outputs.issue_file_ownership_evidence_table),
        repo_list=Path(cfg.outputs.repo_included_list),
    )

def build_commit_ranges(commits_df, commits_resolved_df):
    rows = []
    overall = {
        "commits_min_timestamp": None,
        "commits_max_timestamp": None,
        "commits_resolved_min_timestamp": None,
        "commits_resolved_max_timestamp": None,
        "commits_count": int(len(commits_df)),
        "commits_resolved_count": int(len(commits_resolved_df)),
    }

    for label, df in [("commits", commits_df), ("commits_resolved", commits_resolved_df)]:
        if df.empty:
            continue
        min_ts = df["commit_timestamp"].min()
        max_ts = df["commit_timestamp"].max()
        overall[f"{label}_min_timestamp"] = None if pd.isna(min_ts) else min_ts.isoformat()
        overall[f"{label}_max_timestamp"] = None if pd.isna(max_ts) else max_ts.isoformat()

    for repo_name, repo_df in commits_df.groupby("repo_full_name", dropna=False):
        repo_resolved = commits_resolved_df[commits_resolved_df["repo_full_name"] == repo_name]
        min_ts = repo_df["commit_timestamp"].min() if not repo_df.empty else pd.NaT
        max_ts = repo_df["commit_timestamp"].max() if not repo_df.empty else pd.NaT
        min_res = repo_resolved["commit_timestamp"].min() if not repo_resolved.empty else pd.NaT
        max_res = repo_resolved["commit_timestamp"].max() if not repo_resolved.empty else pd.NaT
        rows.append({
            "repo_full_name": repo_name,
            "commits_count": int(len(repo_df)),
            "commits_min_timestamp": None if pd.isna(min_ts) else min_ts.isoformat(),
            "commits_max_timestamp": None if pd.isna(max_ts) else max_ts.isoformat(),
            "commits_resolved_count": int(len(repo_resolved)),
            "commits_resolved_min_timestamp": None if pd.isna(min_res) else min_res.isoformat(),
            "commits_resolved_max_timestamp": None if pd.isna(max_res) else max_res.isoformat(),
        })
    return pd.DataFrame(rows).sort_values("repo_full_name").reset_index(drop=True), overall

def build_pre_post_commit_distribution(evidence_df, issues_resolved_df):
    if evidence_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    issue_meta = normalize_issue_key_frame(issues_resolved_df)
    if "created_at" in issues_resolved_df.columns:
        issue_meta["issue_created_at"] = to_datetime(issues_resolved_df["created_at"])
    else:
        created_col = find_col(issues_resolved_df, ["created_at", "issue_created_at"])
        issue_meta["issue_created_at"] = to_datetime(issues_resolved_df[created_col]) if created_col else pd.NaT

    cols = ["repo_full_name", "issue_id", "issue_number", "issue_created_at"]
    issue_meta = issue_meta[cols].drop_duplicates(subset=["repo_full_name", "issue_id", "issue_number"])

    merged = evidence_df.merge(issue_meta, on=["repo_full_name", "issue_id", "issue_number"], how="left", suffixes=("", "_issue"))
    merged["days_from_issue_creation"] = (
        (merged["commit_timestamp"] - merged["issue_created_at"]).dt.total_seconds() / 86400.0
    )
    merged["derived_time_bucket"] = np.where(
        merged["days_from_issue_creation"].notna() & (merged["days_from_issue_creation"] <= 0),
        "pre_issue",
        np.where(merged["days_from_issue_creation"].notna(), "post_issue", "unknown"),
    )

    row_summary = (
        merged.groupby(["evidence_selected_for_features", "derived_time_bucket", "evidence_type"], dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(["evidence_selected_for_features", "derived_time_bucket", "evidence_type"])
        .reset_index(drop=True)
    )

    per_issue = (
        merged.groupby(["repo_full_name", "issue_id", "issue_number"], dropna=False)
        .agg(
            issue_created_at=("issue_created_at", "first"),
            selected_commit_rows=("evidence_selected_for_features", "sum"),
            pre_issue_rows=("derived_time_bucket", lambda s: int((s == "pre_issue").sum())),
            post_issue_rows=("derived_time_bucket", lambda s: int((s == "post_issue").sum())),
            min_days_from_issue_creation=("days_from_issue_creation", "min"),
            median_days_from_issue_creation=("days_from_issue_creation", "median"),
            max_days_from_issue_creation=("days_from_issue_creation", "max"),
        )
        .reset_index()
    )

    overall = {
        "all_evidence_pre_issue_rows": int((merged["derived_time_bucket"] == "pre_issue").sum()),
        "all_evidence_post_issue_rows": int((merged["derived_time_bucket"] == "post_issue").sum()),
        "selected_evidence_pre_issue_rows": int(((merged["evidence_selected_for_features"] == 1) & (merged["derived_time_bucket"] == "pre_issue")).sum()),
        "selected_evidence_post_issue_rows": int(((merged["evidence_selected_for_features"] == 1) & (merged["derived_time_bucket"] == "post_issue")).sum()),
        "selected_evidence_min_days_from_issue_creation": None if merged.loc[merged["evidence_selected_for_features"] == 1, "days_from_issue_creation"].dropna().empty else float(merged.loc[merged["evidence_selected_for_features"] == 1, "days_from_issue_creation"].min()),
        "selected_evidence_median_days_from_issue_creation": None if merged.loc[merged["evidence_selected_for_features"] == 1, "days_from_issue_creation"].dropna().empty else float(merged.loc[merged["evidence_selected_for_features"] == 1, "days_from_issue_creation"].median()),
        "selected_evidence_max_days_from_issue_creation": None if merged.loc[merged["evidence_selected_for_features"] == 1, "days_from_issue_creation"].dropna().empty else float(merged.loc[merged["evidence_selected_for_features"] == 1, "days_from_issue_creation"].max()),
    }
    return row_summary, per_issue, overall

def build_coverage_funnel(issues_resolved_df, issue_pr_links_df, pr_commit_links_df, commits_resolved_df, ownership_features_df, ownership_evidence_df):
    issues = normalize_issue_key_frame(issues_resolved_df)
    issues["issue_key"] = (
        issues["repo_full_name"].astype(str)
        + "::"
        + issues["issue_id"].astype(str)
        + "::"
        + issues["issue_number"].astype(str)
    )

    issue_pr = normalize_issue_pr_links(issue_pr_links_df)
    issue_pr["issue_key"] = (
        issue_pr["repo_full_name"].astype(str)
        + "::"
        + issue_pr["issue_id"].astype(str)
        + "::"
        + issue_pr["issue_number"].astype(str)
    )

    pr_commit_df = pr_commit_links_df.copy()
    if not pr_commit_df.empty:
        if "pr_id" in pr_commit_df.columns:
            pr_commit_df["pr_id"] = pd.to_numeric(pr_commit_df["pr_id"], errors="coerce")
        if "pr_number" in pr_commit_df.columns:
            pr_commit_df["pr_number"] = pd.to_numeric(pr_commit_df["pr_number"], errors="coerce")
        if "commit_sha" in pr_commit_df.columns:
            pr_commit_df["commit_sha"] = pr_commit_df["commit_sha"].astype(str)

    commits_resolved_df = normalize_commits(commits_resolved_df)
    features = normalize_issue_features(ownership_features_df)
    evidence = normalize_evidence(ownership_evidence_df)

    pr_links_with_issue = issue_pr[["repo_full_name", "issue_key", "pr_id", "pr_number"]].drop_duplicates()
    if not pr_links_with_issue.empty:
        tmp = pr_links_with_issue.merge(
            pr_commit_df[["repo_full_name", "pr_id", "pr_number", "commit_sha"]].drop_duplicates(),
            on=["repo_full_name", "pr_id", "pr_number"],
            how="left",
        )
        tmp["commit_exists_in_resolved"] = tmp["commit_sha"].isin(set(commits_resolved_df["commit_sha"].dropna().tolist()))
        issue_has_pr_link = set(pr_links_with_issue["issue_key"].tolist())
        issue_has_pr_commit = set(tmp.loc[tmp["commit_sha"].notna(), "issue_key"].tolist())
        issue_has_pr_commit_resolved = set(tmp.loc[tmp["commit_exists_in_resolved"], "issue_key"].tolist())
    else:
        issue_has_pr_link = set()
        issue_has_pr_commit = set()
        issue_has_pr_commit_resolved = set()

    if not features.empty:
        features["issue_key"] = (
            features["repo_full_name"].astype(str)
            + "::"
            + features["issue_id"].astype(str)
            + "::"
            + features["issue_number"].astype(str)
        )
    else:
        features["issue_key"] = pd.Series(dtype="object")

    if not evidence.empty:
        evidence["issue_key"] = (
            evidence["repo_full_name"].astype(str)
            + "::"
            + evidence["issue_id"].astype(str)
            + "::"
            + evidence["issue_number"].astype(str)
        )
    else:
        evidence["issue_key"] = pd.Series(dtype="object")

    issue_has_selected_high_conf = set(
        evidence.loc[evidence["selected_for_high_confidence_features"] == 1, "issue_key"].dropna().tolist()
    ) if not evidence.empty else set()
    issue_has_selected_any = set(
        evidence.loc[evidence["selected_for_any_features"] == 1, "issue_key"].dropna().tolist()
    ) if not evidence.empty else set()
    issue_has_selected_high_conf_pre = set(
        evidence.loc[
            (evidence["selected_for_high_confidence_features"] == 1)
            & (evidence["ownership_time_bucket"] == "pre_issue"),
            "issue_key",
        ].dropna().tolist()
    ) if not evidence.empty else set()
    issue_has_selected_any_pre = set(
        evidence.loc[
            (evidence["selected_for_any_features"] == 1)
            & (evidence["ownership_time_bucket"] == "pre_issue"),
            "issue_key",
        ].dropna().tolist()
    ) if not evidence.empty else set()
    issue_has_selected_any_post = set(
        evidence.loc[
            (evidence["selected_for_any_features"] == 1)
            & (evidence["ownership_time_bucket"] == "post_issue"),
            "issue_key",
        ].dropna().tolist()
    ) if not evidence.empty else set()
    issue_has_conservative_pre = set(
        evidence.loc[evidence["selected_for_conservative_pre_issue_fallback"] == 1, "issue_key"].dropna().tolist()
    ) if not evidence.empty else set()

    funnel_rows = []
    for repo_name, repo_issues in issues.groupby("repo_full_name", dropna=False):
        repo_issue_keys = set(repo_issues["issue_key"].tolist())
        repo_features = features[features["repo_full_name"] == repo_name] if not features.empty else pd.DataFrame()

        funnel_rows.append({
            "repo_full_name": repo_name,
            "target_issues": int(len(repo_issue_keys)),
            "issues_with_pr_links": int(len(repo_issue_keys.intersection(issue_has_pr_link))),
            "issues_with_pr_commit_links": int(len(repo_issue_keys.intersection(issue_has_pr_commit))),
            "issues_with_pr_commit_links_in_commits_resolved": int(len(repo_issue_keys.intersection(issue_has_pr_commit_resolved))),
            "issues_with_selected_high_confidence_evidence": int(len(repo_issue_keys.intersection(issue_has_selected_high_conf))),
            "issues_with_selected_any_evidence": int(len(repo_issue_keys.intersection(issue_has_selected_any))),
            "issues_with_selected_high_confidence_pre_issue_evidence": int(len(repo_issue_keys.intersection(issue_has_selected_high_conf_pre))),
            "issues_with_selected_any_pre_issue_evidence": int(len(repo_issue_keys.intersection(issue_has_selected_any_pre))),
            "issues_with_selected_any_post_issue_evidence": int(len(repo_issue_keys.intersection(issue_has_selected_any_post))),
            "issues_with_selected_conservative_pre_issue_fallback": int(len(repo_issue_keys.intersection(issue_has_conservative_pre))),
            "issues_with_high_confidence_ownership": int((repo_features.get("ownership_usable_high_confidence", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not repo_features.empty else 0,
            "issues_with_usable_any": int((repo_features.get("ownership_usable_any", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not repo_features.empty else 0,
            "issues_with_usable_any_including_conservative_pre_issue": int((repo_features.get("ownership_usable_any_including_conservative_pre_issue", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not repo_features.empty else 0,
            "issues_with_high_confidence_pre_issue_contributors": int((repo_features.get("ownership_pre_issue_high_confidence_contributor_count", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not repo_features.empty else 0,
            "issues_with_any_pre_issue_contributors": int((repo_features.get("ownership_pre_issue_contributor_count", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not repo_features.empty else 0,
            "issues_with_post_issue_contributors": int((repo_features.get("ownership_post_issue_contributor_count", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not repo_features.empty else 0,
            "issues_with_pre_issue_conservative_fallback": int((repo_features.get("ownership_has_selected_conservative_pre_issue_fallback", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not repo_features.empty else 0,
            "issues_with_usable_pre_issue_conservative_fallback": int((repo_features.get("ownership_usable_pre_issue_conservative_fallback", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not repo_features.empty else 0,
        })

    funnel_df = pd.DataFrame(funnel_rows).sort_values("repo_full_name").reset_index(drop=True)

    total_issue_keys = set(issues["issue_key"].tolist())
    overall = {
        "target_issues": int(len(total_issue_keys)),
        "issues_with_pr_links": int(len(total_issue_keys.intersection(issue_has_pr_link))),
        "issues_with_pr_commit_links": int(len(total_issue_keys.intersection(issue_has_pr_commit))),
        "issues_with_pr_commit_links_in_commits_resolved": int(len(total_issue_keys.intersection(issue_has_pr_commit_resolved))),
        "issues_with_selected_high_confidence_evidence": int(len(total_issue_keys.intersection(issue_has_selected_high_conf))),
        "issues_with_selected_any_evidence": int(len(total_issue_keys.intersection(issue_has_selected_any))),
        "issues_with_selected_high_confidence_pre_issue_evidence": int(len(total_issue_keys.intersection(issue_has_selected_high_conf_pre))),
        "issues_with_selected_any_pre_issue_evidence": int(len(total_issue_keys.intersection(issue_has_selected_any_pre))),
        "issues_with_selected_any_post_issue_evidence": int(len(total_issue_keys.intersection(issue_has_selected_any_post))),
        "issues_with_selected_conservative_pre_issue_fallback": int(len(total_issue_keys.intersection(issue_has_conservative_pre))),
        "issues_with_high_confidence_ownership": int((features.get("ownership_usable_high_confidence", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not features.empty else 0,
        "issues_with_usable_any": int((features.get("ownership_usable_any", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not features.empty else 0,
        "issues_with_usable_any_including_conservative_pre_issue": int((features.get("ownership_usable_any_including_conservative_pre_issue", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not features.empty else 0,
        "issues_with_high_confidence_pre_issue_contributors": int((features.get("ownership_pre_issue_high_confidence_contributor_count", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not features.empty else 0,
        "issues_with_any_pre_issue_contributors": int((features.get("ownership_pre_issue_contributor_count", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not features.empty else 0,
        "issues_with_post_issue_contributors": int((features.get("ownership_post_issue_contributor_count", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not features.empty else 0,
        "issues_with_pre_issue_conservative_fallback": int((features.get("ownership_has_selected_conservative_pre_issue_fallback", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not features.empty else 0,
        "issues_with_usable_pre_issue_conservative_fallback": int((features.get("ownership_usable_pre_issue_conservative_fallback", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not features.empty else 0,
    }
    return funnel_df, overall

def build_owner_commenter_overlap(ownership_features_df, ownership_evidence_df, issue_comments_resolved_df):
    features = add_issue_key_columns(normalize_issue_features(ownership_features_df))
    evidence = add_issue_key_columns(normalize_evidence(ownership_evidence_df))
    comments = add_issue_key_columns(normalize_issue_comments(issue_comments_resolved_df))

    if features.empty:
        return pd.DataFrame(), {}

    selected_high_conf = evidence[evidence["selected_for_high_confidence_features"] == 1].copy()
    selected_any = evidence[evidence["selected_for_any_features"] == 1].copy()

    commenter_sets_by_number = (
        comments.dropna(subset=["comment_author_contributor_key", "issue_key_number"])
        .groupby("issue_key_number")["comment_author_contributor_key"]
        .agg(lambda s: sorted(set([x for x in s.tolist() if clean_text(x)])))
        .to_dict()
    )
    commenter_sets_by_id = (
        comments.dropna(subset=["comment_author_contributor_key", "issue_key_id"])
        .groupby("issue_key_id")["comment_author_contributor_key"]
        .agg(lambda s: sorted(set([x for x in s.tolist() if clean_text(x)])))
        .to_dict()
    )

    owner_rows = []
    for _, feat_row in features.iterrows():
        issue_key_number = feat_row.get("issue_key_number")
        issue_key_id = feat_row.get("issue_key_id")

        commenter_keys = []
        matched_lookup_key = "none"
        if issue_key_number and issue_key_number in commenter_sets_by_number:
            commenter_keys = commenter_sets_by_number[issue_key_number]
            matched_lookup_key = "issue_number"
        elif issue_key_id and issue_key_id in commenter_sets_by_id:
            commenter_keys = commenter_sets_by_id[issue_key_id]
            matched_lookup_key = "issue_id"

        issue_high_conf = selected_high_conf[selected_high_conf["issue_key_number"] == issue_key_number].copy()
        issue_any = selected_any[selected_any["issue_key_number"] == issue_key_number].copy()

        owner_keys_high_conf_all = sorted(set([clean_text(x) for x in issue_high_conf["commit_author_contributor_key"].tolist() if clean_text(x)])) if not issue_high_conf.empty else []
        owner_keys_high_conf_pre = sorted(set([clean_text(x) for x in issue_high_conf.loc[issue_high_conf["ownership_time_bucket"] == "pre_issue", "commit_author_contributor_key"].tolist() if clean_text(x)])) if not issue_high_conf.empty else []
        owner_keys_any_all = sorted(set([clean_text(x) for x in issue_any["commit_author_contributor_key"].tolist() if clean_text(x)])) if not issue_any.empty else []
        owner_keys_any_pre = sorted(set([clean_text(x) for x in issue_any.loc[issue_any["ownership_time_bucket"] == "pre_issue", "commit_author_contributor_key"].tolist() if clean_text(x)])) if not issue_any.empty else []
        owner_keys_any_post = sorted(set([clean_text(x) for x in issue_any.loc[issue_any["ownership_time_bucket"] == "post_issue", "commit_author_contributor_key"].tolist() if clean_text(x)])) if not issue_any.empty else []
        owner_keys_conservative_pre = sorted(set([clean_text(x) for x in issue_any.loc[issue_any["selected_for_conservative_pre_issue_fallback"] == 1, "commit_author_contributor_key"].tolist() if clean_text(x)])) if not issue_any.empty else []

        overlap_high_conf_all = sorted(set(owner_keys_high_conf_all).intersection(set(commenter_keys)))
        overlap_high_conf_pre = sorted(set(owner_keys_high_conf_pre).intersection(set(commenter_keys)))
        overlap_any_all = sorted(set(owner_keys_any_all).intersection(set(commenter_keys)))
        overlap_any_pre = sorted(set(owner_keys_any_pre).intersection(set(commenter_keys)))
        overlap_any_post = sorted(set(owner_keys_any_post).intersection(set(commenter_keys)))
        overlap_conservative_pre = sorted(set(owner_keys_conservative_pre).intersection(set(commenter_keys)))

        owner_rows.append({
            "repo_full_name": feat_row.get("repo_full_name"),
            "issue_id": feat_row.get("issue_id"),
            "issue_number": feat_row.get("issue_number"),
            "matched_commenter_lookup_key": matched_lookup_key,
            "commenter_count": int(len(commenter_keys)),
            "owner_count_high_confidence_all": int(len(owner_keys_high_conf_all)),
            "owner_count_high_confidence_pre_issue": int(len(owner_keys_high_conf_pre)),
            "owner_count_any_all": int(len(owner_keys_any_all)),
            "owner_count_any_pre_issue": int(len(owner_keys_any_pre)),
            "owner_count_any_post_issue": int(len(owner_keys_any_post)),
            "owner_count_conservative_pre_issue": int(len(owner_keys_conservative_pre)),
            "owner_commenter_overlap_count_high_confidence_all": int(len(overlap_high_conf_all)),
            "owner_commenter_overlap_count_high_confidence_pre_issue": int(len(overlap_high_conf_pre)),
            "owner_commenter_overlap_count_any_all": int(len(overlap_any_all)),
            "owner_commenter_overlap_count_any_pre_issue": int(len(overlap_any_pre)),
            "owner_commenter_overlap_count_any_post_issue": int(len(overlap_any_post)),
            "owner_commenter_overlap_count_conservative_pre_issue": int(len(overlap_conservative_pre)),
            "owner_commenter_overlap_fraction_high_confidence_all": safe_divide(len(overlap_high_conf_all), len(owner_keys_high_conf_all), default=np.nan),
            "owner_commenter_overlap_fraction_high_confidence_pre_issue": safe_divide(len(overlap_high_conf_pre), len(owner_keys_high_conf_pre), default=np.nan),
            "owner_commenter_overlap_fraction_any_all": safe_divide(len(overlap_any_all), len(owner_keys_any_all), default=np.nan),
            "owner_commenter_overlap_fraction_any_pre_issue": safe_divide(len(overlap_any_pre), len(owner_keys_any_pre), default=np.nan),
            "owner_commenter_overlap_fraction_any_post_issue": safe_divide(len(overlap_any_post), len(owner_keys_any_post), default=np.nan),
            "owner_commenter_overlap_fraction_conservative_pre_issue": safe_divide(len(overlap_conservative_pre), len(owner_keys_conservative_pre), default=np.nan),
            "owner_keys_high_confidence_all_json": json.dumps(owner_keys_high_conf_all),
            "owner_keys_high_confidence_pre_issue_json": json.dumps(owner_keys_high_conf_pre),
            "owner_keys_any_all_json": json.dumps(owner_keys_any_all),
            "owner_keys_any_pre_issue_json": json.dumps(owner_keys_any_pre),
            "owner_keys_any_post_issue_json": json.dumps(owner_keys_any_post),
            "owner_keys_conservative_pre_issue_json": json.dumps(owner_keys_conservative_pre),
            "commenter_keys_json": json.dumps(commenter_keys),
            "overlap_keys_high_confidence_all_json": json.dumps(overlap_high_conf_all),
            "overlap_keys_high_confidence_pre_issue_json": json.dumps(overlap_high_conf_pre),
            "overlap_keys_any_all_json": json.dumps(overlap_any_all),
            "overlap_keys_any_pre_issue_json": json.dumps(overlap_any_pre),
            "overlap_keys_any_post_issue_json": json.dumps(overlap_any_post),
            "overlap_keys_conservative_pre_issue_json": json.dumps(overlap_conservative_pre),
        })

    overlap_df = pd.DataFrame(owner_rows).sort_values(["repo_full_name", "issue_number"]).reset_index(drop=True) if owner_rows else pd.DataFrame()

    overall = {
        "issues_with_selected_any_evidence": int(len(overlap_df)),
        "issues_matched_on_issue_number": int((overlap_df.get("matched_commenter_lookup_key", pd.Series(dtype="object")) == "issue_number").sum()) if not overlap_df.empty else 0,
        "issues_matched_on_issue_id": int((overlap_df.get("matched_commenter_lookup_key", pd.Series(dtype="object")) == "issue_id").sum()) if not overlap_df.empty else 0,
        "issues_with_no_comment_match_key": int((overlap_df.get("matched_commenter_lookup_key", pd.Series(dtype="object")) == "none").sum()) if not overlap_df.empty else 0,
        "issues_with_any_owner_commenter_overlap_high_confidence_all": int((overlap_df.get("owner_commenter_overlap_count_high_confidence_all", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not overlap_df.empty else 0,
        "issues_with_pre_issue_owner_commenter_overlap_high_confidence": int((overlap_df.get("owner_commenter_overlap_count_high_confidence_pre_issue", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not overlap_df.empty else 0,
        "issues_with_any_owner_commenter_overlap_any_all": int((overlap_df.get("owner_commenter_overlap_count_any_all", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not overlap_df.empty else 0,
        "issues_with_pre_issue_owner_commenter_overlap_any": int((overlap_df.get("owner_commenter_overlap_count_any_pre_issue", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not overlap_df.empty else 0,
        "issues_with_post_issue_owner_commenter_overlap_any": int((overlap_df.get("owner_commenter_overlap_count_any_post_issue", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not overlap_df.empty else 0,
        "issues_with_conservative_pre_issue_owner_commenter_overlap": int((overlap_df.get("owner_commenter_overlap_count_conservative_pre_issue", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not overlap_df.empty else 0,
        "mean_owner_commenter_overlap_fraction_high_confidence_all": None if overlap_df.empty else pd.to_numeric(overlap_df["owner_commenter_overlap_fraction_high_confidence_all"], errors="coerce").dropna().mean(),
        "mean_owner_commenter_overlap_fraction_high_confidence_pre_issue": None if overlap_df.empty else pd.to_numeric(overlap_df["owner_commenter_overlap_fraction_high_confidence_pre_issue"], errors="coerce").dropna().mean(),
        "mean_owner_commenter_overlap_fraction_any_all": None if overlap_df.empty else pd.to_numeric(overlap_df["owner_commenter_overlap_fraction_any_all"], errors="coerce").dropna().mean(),
        "mean_owner_commenter_overlap_fraction_any_pre_issue": None if overlap_df.empty else pd.to_numeric(overlap_df["owner_commenter_overlap_fraction_any_pre_issue"], errors="coerce").dropna().mean(),
        "mean_owner_commenter_overlap_fraction_any_post_issue": None if overlap_df.empty else pd.to_numeric(overlap_df["owner_commenter_overlap_fraction_any_post_issue"], errors="coerce").dropna().mean(),
        "mean_owner_commenter_overlap_fraction_conservative_pre_issue": None if overlap_df.empty else pd.to_numeric(overlap_df["owner_commenter_overlap_fraction_conservative_pre_issue"], errors="coerce").dropna().mean(),
    }

    print("DEBUG overlap | selected issues:", len(overlap_df))
    print("DEBUG overlap | matched on issue_number:", overall["issues_matched_on_issue_number"])
    print("DEBUG overlap | matched on issue_id:", overall["issues_matched_on_issue_id"])
    print("DEBUG overlap | no comment match key:", overall["issues_with_no_comment_match_key"])
    print("DEBUG overlap | any overlap high confidence:", overall["issues_with_any_owner_commenter_overlap_high_confidence_all"])
    print("DEBUG overlap | any overlap broadened:", overall["issues_with_any_owner_commenter_overlap_any_all"])
    print("DEBUG overlap | pre-issue overlap high confidence:", overall["issues_with_pre_issue_owner_commenter_overlap_high_confidence"])
    print("DEBUG overlap | pre-issue overlap broadened:", overall["issues_with_pre_issue_owner_commenter_overlap_any"])
    print("DEBUG overlap | conservative pre-issue overlap:", overall["issues_with_conservative_pre_issue_owner_commenter_overlap"])

    return overlap_df, overall

def build_selection_loss_diagnostic(ownership_evidence_df):
    evidence = add_issue_key_columns(normalize_evidence(ownership_evidence_df))
    if evidence.empty:
        return pd.DataFrame(), {}

    grouped_rows = []
    for issue_key_number, group in evidence.groupby("issue_key_number", dropna=False):
        rep_row = group.iloc[0]

        raw_total = int(len(group))
        selected_high_conf_total = int((group["selected_for_high_confidence_features"] == 1).sum())
        selected_any_total = int((group["selected_for_any_features"] == 1).sum())
        selected_conservative_total = int((group["selected_for_conservative_pre_issue_fallback"] == 1).sum())

        raw_pre = int((group["ownership_time_bucket"] == "pre_issue").sum())
        raw_post = int((group["ownership_time_bucket"] == "post_issue").sum())

        selected_high_conf_pre = int(((group["selected_for_high_confidence_features"] == 1) & (group["ownership_time_bucket"] == "pre_issue")).sum())
        selected_any_pre = int(((group["selected_for_any_features"] == 1) & (group["ownership_time_bucket"] == "pre_issue")).sum())
        selected_any_post = int(((group["selected_for_any_features"] == 1) & (group["ownership_time_bucket"] == "post_issue")).sum())

        raw_pr_merge = int((group["evidence_type"] == "pr_merge").sum())
        raw_pr_exact = int((group["evidence_type"] == "pr_exact_commit").sum())
        raw_pr_head = int((group["evidence_type"] == "pr_head").sum())
        raw_fallback = int((group["evidence_type"] == "file_fallback").sum())

        selected_high_conf_pr_merge = int(((group["selected_for_high_confidence_features"] == 1) & (group["evidence_type"] == "pr_merge")).sum())
        selected_high_conf_pr_exact = int(((group["selected_for_high_confidence_features"] == 1) & (group["evidence_type"] == "pr_exact_commit")).sum())
        selected_high_conf_pr_head = int(((group["selected_for_high_confidence_features"] == 1) & (group["evidence_type"] == "pr_head")).sum())
        selected_high_conf_fallback = int(((group["selected_for_high_confidence_features"] == 1) & (group["evidence_type"] == "file_fallback")).sum())

        selected_any_fallback = int(((group["selected_for_any_features"] == 1) & (group["evidence_type"] == "file_fallback")).sum())
        selected_conservative_fallback = int(((group["selected_for_conservative_pre_issue_fallback"] == 1) & (group["evidence_type"] == "file_fallback")).sum())

        raw_pre_fallback = int(((group["ownership_time_bucket"] == "pre_issue") & (group["evidence_type"] == "file_fallback")).sum())
        selected_high_conf_pre_fallback = int(((group["selected_for_high_confidence_features"] == 1) & (group["ownership_time_bucket"] == "pre_issue") & (group["evidence_type"] == "file_fallback")).sum())
        selected_any_pre_fallback = int(((group["selected_for_any_features"] == 1) & (group["ownership_time_bucket"] == "pre_issue") & (group["evidence_type"] == "file_fallback")).sum())
        selected_conservative_pre_fallback = int(((group["selected_for_conservative_pre_issue_fallback"] == 1) & (group["ownership_time_bucket"] == "pre_issue") & (group["evidence_type"] == "file_fallback")).sum())

        unresolved_author_rows = int(group["commit_author_contributor_key"].isna().sum()) if "commit_author_contributor_key" in group.columns else 0
        low_confidence_fallback_rows = int(((group["evidence_type"] == "file_fallback") & (~group["issue_file_confidence_level"].isin(["high"]))).sum()) if "issue_file_confidence_level" in group.columns else 0
        invalid_pre_conservative_rows = int(((group["selected_for_conservative_pre_issue_fallback"] == 0) & (group["ownership_time_bucket"] == "pre_issue") & (group["evidence_type"] == "file_fallback")).sum())

        grouped_rows.append({
            "repo_full_name": rep_row.get("repo_full_name"),
            "issue_id": rep_row.get("issue_id"),
            "issue_number": rep_row.get("issue_number"),
            "raw_evidence_rows": raw_total,
            "selected_high_confidence_rows": selected_high_conf_total,
            "selected_any_rows": selected_any_total,
            "selected_conservative_pre_issue_rows": selected_conservative_total,
            "high_confidence_selection_rate": safe_divide(selected_high_conf_total, raw_total, default=np.nan),
            "any_selection_rate": safe_divide(selected_any_total, raw_total, default=np.nan),
            "raw_pre_issue_rows": raw_pre,
            "selected_high_confidence_pre_issue_rows": selected_high_conf_pre,
            "selected_any_pre_issue_rows": selected_any_pre,
            "selected_any_post_issue_rows": selected_any_post,
            "high_confidence_pre_issue_selection_rate": safe_divide(selected_high_conf_pre, raw_pre, default=np.nan),
            "any_pre_issue_selection_rate": safe_divide(selected_any_pre, raw_pre, default=np.nan),
            "raw_post_issue_rows": raw_post,
            "any_post_issue_selection_rate": safe_divide(selected_any_post, raw_post, default=np.nan),
            "raw_pr_merge_rows": raw_pr_merge,
            "selected_high_confidence_pr_merge_rows": selected_high_conf_pr_merge,
            "raw_pr_exact_rows": raw_pr_exact,
            "selected_high_confidence_pr_exact_rows": selected_high_conf_pr_exact,
            "raw_pr_head_rows": raw_pr_head,
            "selected_high_confidence_pr_head_rows": selected_high_conf_pr_head,
            "raw_fallback_rows": raw_fallback,
            "selected_high_confidence_fallback_rows": selected_high_conf_fallback,
            "selected_any_fallback_rows": selected_any_fallback,
            "selected_conservative_fallback_rows": selected_conservative_fallback,
            "raw_pre_issue_fallback_rows": raw_pre_fallback,
            "selected_high_confidence_pre_issue_fallback_rows": selected_high_conf_pre_fallback,
            "selected_any_pre_issue_fallback_rows": selected_any_pre_fallback,
            "selected_conservative_pre_issue_fallback_rows": selected_conservative_pre_fallback,
            "lost_pre_issue_rows_high_confidence": int(raw_pre - selected_high_conf_pre),
            "lost_pre_issue_rows_any": int(raw_pre - selected_any_pre),
            "lost_post_issue_rows_any": int(raw_post - selected_any_post),
            "lost_pr_merge_rows_high_confidence": int(raw_pr_merge - selected_high_conf_pr_merge),
            "lost_pr_exact_rows_high_confidence": int(raw_pr_exact - selected_high_conf_pr_exact),
            "lost_pr_head_rows_high_confidence": int(raw_pr_head - selected_high_conf_pr_head),
            "lost_fallback_rows_high_confidence": int(raw_fallback - selected_high_conf_fallback),
            "lost_pre_issue_fallback_rows_before_broadening": int(raw_pre_fallback - selected_high_conf_pre_fallback),
            "lost_pre_issue_fallback_rows_after_broadening": int(raw_pre_fallback - selected_any_pre_fallback),
            "gained_pre_issue_rows_from_broadening": int(selected_any_pre - selected_high_conf_pre),
            "gained_pre_issue_fallback_rows_from_broadening": int(selected_any_pre_fallback - selected_high_conf_pre_fallback),
            "unresolved_author_rows": unresolved_author_rows,
            "low_confidence_fallback_rows": low_confidence_fallback_rows,
            "pre_issue_fallback_rows_not_selected_conservatively": invalid_pre_conservative_rows,
        })

    selection_loss_df = pd.DataFrame(grouped_rows).sort_values(["repo_full_name", "issue_number"]).reset_index(drop=True)

    overall = {
        "issues_with_any_raw_evidence": int(len(selection_loss_df)),
        "issues_with_any_selected_high_confidence_evidence": int((selection_loss_df["selected_high_confidence_rows"] > 0).sum()) if not selection_loss_df.empty else 0,
        "issues_with_any_selected_any_evidence": int((selection_loss_df["selected_any_rows"] > 0).sum()) if not selection_loss_df.empty else 0,
        "issues_with_any_selected_conservative_pre_issue_rows": int((selection_loss_df["selected_conservative_pre_issue_rows"] > 0).sum()) if not selection_loss_df.empty else 0,
        "mean_high_confidence_selection_rate": pd.to_numeric(selection_loss_df["high_confidence_selection_rate"], errors="coerce").dropna().mean() if not selection_loss_df.empty else None,
        "mean_any_selection_rate": pd.to_numeric(selection_loss_df["any_selection_rate"], errors="coerce").dropna().mean() if not selection_loss_df.empty else None,
        "mean_high_confidence_pre_issue_selection_rate": pd.to_numeric(selection_loss_df["high_confidence_pre_issue_selection_rate"], errors="coerce").dropna().mean() if not selection_loss_df.empty else None,
        "mean_any_pre_issue_selection_rate": pd.to_numeric(selection_loss_df["any_pre_issue_selection_rate"], errors="coerce").dropna().mean() if not selection_loss_df.empty else None,
        "mean_any_post_issue_selection_rate": pd.to_numeric(selection_loss_df["any_post_issue_selection_rate"], errors="coerce").dropna().mean() if not selection_loss_df.empty else None,
        "total_lost_pre_issue_rows_high_confidence": int(selection_loss_df["lost_pre_issue_rows_high_confidence"].sum()) if not selection_loss_df.empty else 0,
        "total_lost_pre_issue_rows_any": int(selection_loss_df["lost_pre_issue_rows_any"].sum()) if not selection_loss_df.empty else 0,
        "total_lost_post_issue_rows_any": int(selection_loss_df["lost_post_issue_rows_any"].sum()) if not selection_loss_df.empty else 0,
        "total_lost_pr_merge_rows_high_confidence": int(selection_loss_df["lost_pr_merge_rows_high_confidence"].sum()) if not selection_loss_df.empty else 0,
        "total_lost_pr_exact_rows_high_confidence": int(selection_loss_df["lost_pr_exact_rows_high_confidence"].sum()) if not selection_loss_df.empty else 0,
        "total_lost_pr_head_rows_high_confidence": int(selection_loss_df["lost_pr_head_rows_high_confidence"].sum()) if not selection_loss_df.empty else 0,
        "total_lost_fallback_rows_high_confidence": int(selection_loss_df["lost_fallback_rows_high_confidence"].sum()) if not selection_loss_df.empty else 0,
        "total_lost_pre_issue_fallback_rows_before_broadening": int(selection_loss_df["lost_pre_issue_fallback_rows_before_broadening"].sum()) if not selection_loss_df.empty else 0,
        "total_lost_pre_issue_fallback_rows_after_broadening": int(selection_loss_df["lost_pre_issue_fallback_rows_after_broadening"].sum()) if not selection_loss_df.empty else 0,
        "total_gained_pre_issue_rows_from_broadening": int(selection_loss_df["gained_pre_issue_rows_from_broadening"].sum()) if not selection_loss_df.empty else 0,
        "total_gained_pre_issue_fallback_rows_from_broadening": int(selection_loss_df["gained_pre_issue_fallback_rows_from_broadening"].sum()) if not selection_loss_df.empty else 0,
        "total_unresolved_author_rows": int(selection_loss_df["unresolved_author_rows"].sum()) if not selection_loss_df.empty else 0,
        "total_low_confidence_fallback_rows": int(selection_loss_df["low_confidence_fallback_rows"].sum()) if not selection_loss_df.empty else 0,
        "total_pre_issue_fallback_rows_not_selected_conservatively": int(selection_loss_df["pre_issue_fallback_rows_not_selected_conservatively"].sum()) if not selection_loss_df.empty else 0,
    }

    return selection_loss_df, overall

def write_markdown_summary(output_path, git_window_info, commit_overall, pre_post_overall, funnel_overall, overlap_overall, selection_loss_overall):
    lines = []
    lines.append("# Ownership Timing Diagnostic Summary")
    lines.append("")
    lines.append("## Git extraction window interpretation")
    lines.append("")
    for key, value in git_window_info.items():
        lines.append(f"- {key.replace('_', ' ')}: `{value}`")
    lines.append("")
    lines.append("## Commit timestamp ranges")
    lines.append("")
    lines.append(f"- commits count: {commit_overall.get('commits_count')}")
    lines.append(f"- commits min timestamp: {commit_overall.get('commits_min_timestamp')}")
    lines.append(f"- commits max timestamp: {commit_overall.get('commits_max_timestamp')}")
    lines.append(f"- commits_resolved count: {commit_overall.get('commits_resolved_count')}")
    lines.append(f"- commits_resolved min timestamp: {commit_overall.get('commits_resolved_min_timestamp')}")
    lines.append(f"- commits_resolved max timestamp: {commit_overall.get('commits_resolved_max_timestamp')}")
    lines.append("")
    lines.append("## Pre/post evidence")
    lines.append("")
    for key, value in pre_post_overall.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Ownership coverage funnel")
    lines.append("")
    for key, value in funnel_overall.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Owner vs commenter overlap")
    lines.append("")
    for key, value in overlap_overall.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Selection loss")
    lines.append("")
    for key, value in selection_loss_overall.items():
        lines.append(f"- {key}: {value}")
    output_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    project_root = Path.cwd()
    config_path = project_root / "config" / "study_config.yaml"
    cfg = load_study_config(config_path)
    paths = resolve_paths(cfg)

    output_dir = ensure_dir(project_root / "outputs" / "ownership_diagnostics")

    issues_resolved = maybe_read_parquet(paths.issues_resolved)
    issue_comments_resolved = maybe_read_parquet(paths.issue_comments_resolved)
    issue_pr_links = maybe_read_parquet(paths.issue_pr_links)
    pr_commit_links = maybe_read_parquet(paths.pr_commit_links)
    commits = normalize_commits(maybe_read_parquet(paths.commits))
    commits_resolved = normalize_commits(maybe_read_parquet(paths.commits_resolved))
    ownership_features = normalize_issue_features(maybe_read_parquet(paths.issue_ownership_features))
    ownership_evidence = normalize_evidence(maybe_read_parquet(paths.issue_ownership_evidence))

    git_window_info = get_effective_git_window(cfg)

    commit_ranges_by_repo, commit_overall = build_commit_ranges(commits, commits_resolved)
    pre_post_row_summary, pre_post_issue_summary, pre_post_overall = build_pre_post_commit_distribution(
        ownership_evidence,
        issues_resolved,
    )
    funnel_by_repo, funnel_overall = build_coverage_funnel(
        issues_resolved,
        issue_pr_links,
        pr_commit_links,
        commits_resolved,
        ownership_features,
        ownership_evidence,
    )
    owner_commenter_overlap_by_issue, overlap_overall = build_owner_commenter_overlap(
        ownership_features,
        ownership_evidence,
        issue_comments_resolved,
    )
    selection_loss_by_issue, selection_loss_overall = build_selection_loss_diagnostic(
        ownership_evidence,
    )

    print("DEBUG selection_loss | issues_with_any_raw_evidence:",
          selection_loss_overall["issues_with_any_raw_evidence"])
    print("DEBUG selection_loss | issues_with_any_selected_high_confidence_evidence:",
          selection_loss_overall["issues_with_any_selected_high_confidence_evidence"])
    print("DEBUG selection_loss | issues_with_any_selected_any_evidence:",
          selection_loss_overall["issues_with_any_selected_any_evidence"])
    print("DEBUG selection_loss | mean_high_confidence_selection_rate:",
          selection_loss_overall["mean_high_confidence_selection_rate"])
    print("DEBUG selection_loss | mean_any_selection_rate:", selection_loss_overall["mean_any_selection_rate"])
    print("DEBUG selection_loss | mean_high_confidence_pre_issue_selection_rate:",
          selection_loss_overall["mean_high_confidence_pre_issue_selection_rate"])
    print("DEBUG selection_loss | mean_any_pre_issue_selection_rate:",
          selection_loss_overall["mean_any_pre_issue_selection_rate"])
    print("DEBUG selection_loss | total_lost_pre_issue_rows_high_confidence:",
          selection_loss_overall["total_lost_pre_issue_rows_high_confidence"])
    print("DEBUG selection_loss | total_lost_pre_issue_rows_any:",
          selection_loss_overall["total_lost_pre_issue_rows_any"])
    print("DEBUG selection_loss | total_gained_pre_issue_rows_from_broadening:",
          selection_loss_overall["total_gained_pre_issue_rows_from_broadening"])
    print("DEBUG selection_loss | total_gained_pre_issue_fallback_rows_from_broadening:",
          selection_loss_overall["total_gained_pre_issue_fallback_rows_from_broadening"])

    commit_ranges_by_repo.to_csv(output_dir / "commit_ranges_by_repo.csv", index=False)
    pre_post_row_summary.to_csv(output_dir / "pre_post_commit_distribution_rows.csv", index=False)
    pre_post_issue_summary.to_csv(output_dir / "pre_post_commit_distribution_by_issue.csv", index=False)
    funnel_by_repo.to_csv(output_dir / "ownership_coverage_funnel_by_repo.csv", index=False)
    owner_commenter_overlap_by_issue.to_csv(output_dir / "owner_commenter_overlap_by_issue.csv", index=False)
    selection_loss_by_issue.to_csv(output_dir / "selection_loss_by_issue.csv", index=False)

    (output_dir / "git_window_interpretation.json").write_text(json.dumps(git_window_info, indent=2), encoding="utf-8")
    (output_dir / "overall_summary.json").write_text(
        json.dumps(
            {
                "git_window_info": git_window_info,
                "commit_overall": commit_overall,
                "pre_post_overall": pre_post_overall,
                "funnel_overall": funnel_overall,
                "overlap_overall": overlap_overall,
                "selection_loss_overall": selection_loss_overall,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_markdown_summary(
        output_dir / "ownership_diagnostic_summary.md",
        git_window_info,
        commit_overall,
        pre_post_overall,
        funnel_overall,
        overlap_overall,
        selection_loss_overall,
    )

    print(output_dir)


if __name__ == "__main__":
    main()
