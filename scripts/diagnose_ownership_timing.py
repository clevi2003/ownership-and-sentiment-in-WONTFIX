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
            "repo_full_name", "issue_id", "issue_number", "commit_sha", "commit_timestamp",
            "commit_author_contributor_key", "evidence_type", "evidence_selected_for_features",
            "ownership_time_bucket"
        ])
    out = df.copy()
    if "commit_timestamp" in out.columns:
        out["commit_timestamp"] = to_datetime(out["commit_timestamp"])
    if "issue_created_at" in out.columns:
        out["issue_created_at"] = to_datetime(out["issue_created_at"])
    if "evidence_selected_for_features" in out.columns:
        out["evidence_selected_for_features"] = pd.to_numeric(out["evidence_selected_for_features"], errors="coerce").fillna(0).astype(int)
    else:
        out["evidence_selected_for_features"] = 0
    for col in ["commit_author_contributor_key", "evidence_type", "ownership_time_bucket", "repo_full_name", "issue_id"]:
        if col in out.columns:
            out[col] = out[col].apply(clean_text)
    if "issue_number" in out.columns:
        out["issue_number"] = pd.to_numeric(out["issue_number"], errors="coerce")
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


def build_coverage_funnel(
    issues_resolved_df: pd.DataFrame,
    issue_pr_links_df: pd.DataFrame,
    pr_commit_links_df: pd.DataFrame,
    commits_resolved_df: pd.DataFrame,
    ownership_features_df: pd.DataFrame,
    ownership_evidence_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
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

    pr_commit = normalize_issue_pr_links(pr_commit_links_df.rename(columns={"pr_id": "issue_id", "pr_number": "issue_number"}))
    # The above frame is not used directly for issue matching; keep original pr_commit counts separately.
    pr_commit_df = pr_commit_links_df.copy()
    if not pr_commit_df.empty:
        if "pr_id" in pr_commit_df.columns:
            pr_commit_df["pr_id"] = pd.to_numeric(pr_commit_df["pr_id"], errors="coerce")
        if "pr_number" in pr_commit_df.columns:
            pr_commit_df["pr_number"] = pd.to_numeric(pr_commit_df["pr_number"], errors="coerce")
        if "commit_sha" in pr_commit_df.columns:
            pr_commit_df["commit_sha"] = pr_commit_df["commit_sha"].astype(str)

    commits_resolved_df = normalize_commits(commits_resolved_df)
    ownership_features_df = normalize_issue_features(ownership_features_df)
    ownership_evidence_df = normalize_evidence(ownership_evidence_df)

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

    features = normalize_issue_features(ownership_features_df)
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

    evidence = ownership_evidence_df.copy()
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

    issue_has_selected_evidence = set(evidence.loc[evidence["evidence_selected_for_features"] == 1, "issue_key"].dropna().tolist())
    issue_has_selected_pre = set(
        evidence.loc[(evidence["evidence_selected_for_features"] == 1) & (evidence["ownership_time_bucket"] == "pre_issue"), "issue_key"].dropna().tolist()
    )
    issue_has_selected_post = set(
        evidence.loc[(evidence["evidence_selected_for_features"] == 1) & (evidence["ownership_time_bucket"] == "post_issue"), "issue_key"].dropna().tolist()
    )

    features_map = features.set_index("issue_key", drop=False) if not features.empty else pd.DataFrame()

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
            "issues_with_selected_evidence": int(len(repo_issue_keys.intersection(issue_has_selected_evidence))),
            "issues_with_selected_pre_issue_evidence": int(len(repo_issue_keys.intersection(issue_has_selected_pre))),
            "issues_with_selected_post_issue_evidence": int(len(repo_issue_keys.intersection(issue_has_selected_post))),
            "issues_with_usable_any": int((repo_features.get("ownership_usable_any", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not repo_features.empty else 0,
            "issues_with_high_confidence_ownership": int((repo_features.get("ownership_usable_high_confidence", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not repo_features.empty else 0,
            "issues_with_pre_issue_contributors": int((repo_features.get("ownership_pre_issue_contributor_count", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not repo_features.empty else 0,
            "issues_with_post_issue_contributors": int((repo_features.get("ownership_post_issue_contributor_count", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not repo_features.empty else 0,
        })

    funnel_df = pd.DataFrame(funnel_rows).sort_values("repo_full_name").reset_index(drop=True)

    total_issue_keys = set(issues["issue_key"].tolist())
    overall = {
        "target_issues": int(len(total_issue_keys)),
        "issues_with_pr_links": int(len(total_issue_keys.intersection(issue_has_pr_link))),
        "issues_with_pr_commit_links": int(len(total_issue_keys.intersection(issue_has_pr_commit))),
        "issues_with_pr_commit_links_in_commits_resolved": int(len(total_issue_keys.intersection(issue_has_pr_commit_resolved))),
        "issues_with_selected_evidence": int(len(total_issue_keys.intersection(issue_has_selected_evidence))),
        "issues_with_selected_pre_issue_evidence": int(len(total_issue_keys.intersection(issue_has_selected_pre))),
        "issues_with_selected_post_issue_evidence": int(len(total_issue_keys.intersection(issue_has_selected_post))),
        "issues_with_usable_any": int((features.get("ownership_usable_any", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not features.empty else 0,
        "issues_with_high_confidence_ownership": int((features.get("ownership_usable_high_confidence", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not features.empty else 0,
        "issues_with_pre_issue_contributors": int((features.get("ownership_pre_issue_contributor_count", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not features.empty else 0,
        "issues_with_post_issue_contributors": int((features.get("ownership_post_issue_contributor_count", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not features.empty else 0,
    }
    return funnel_df, overall


def build_owner_commenter_overlap(
    ownership_features_df: pd.DataFrame,
    ownership_evidence_df: pd.DataFrame,
    issue_comments_resolved_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    features = normalize_issue_features(ownership_features_df)
    evidence = normalize_evidence(ownership_evidence_df)
    comments = normalize_issue_comments(issue_comments_resolved_df)

    if features.empty:
        return pd.DataFrame(), {}

    features["issue_key"] = (
        features["repo_full_name"].astype(str)
        + "::"
        + features["issue_id"].astype(str)
        + "::"
        + features["issue_number"].astype(str)
    )
    evidence["issue_key"] = (
        evidence["repo_full_name"].astype(str)
        + "::"
        + evidence["issue_id"].astype(str)
        + "::"
        + evidence["issue_number"].astype(str)
    )
    comments["issue_key"] = (
        comments["repo_full_name"].astype(str)
        + "::"
        + comments["issue_id"].astype(str)
        + "::"
        + comments["issue_number"].astype(str)
    )

    selected = evidence[evidence["evidence_selected_for_features"] == 1].copy()

    commenter_sets = (
        comments.dropna(subset=["comment_author_contributor_key"])
        .groupby("issue_key")["comment_author_contributor_key"]
        .agg(lambda s: sorted(set([x for x in s.tolist() if clean_text(x)])))
        .to_dict()
    )

    owner_rows = []
    for issue_key, group in selected.groupby("issue_key", dropna=False):
        owner_keys_all = sorted(set([clean_text(v) for v in group["commit_author_contributor_key"].tolist() if clean_text(v)]))
        owner_keys_pre = sorted(set([clean_text(v) for v in group.loc[group["ownership_time_bucket"] == "pre_issue", "commit_author_contributor_key"].tolist() if clean_text(v)]))
        owner_keys_post = sorted(set([clean_text(v) for v in group.loc[group["ownership_time_bucket"] == "post_issue", "commit_author_contributor_key"].tolist() if clean_text(v)]))
        commenter_keys = commenter_sets.get(issue_key, [])
        commenter_set = set(commenter_keys)

        overlap_all = sorted(set(owner_keys_all).intersection(commenter_set))
        overlap_pre = sorted(set(owner_keys_pre).intersection(commenter_set))
        overlap_post = sorted(set(owner_keys_post).intersection(commenter_set))

        rep_row = group.iloc[0]
        owner_rows.append({
            "repo_full_name": rep_row.get("repo_full_name"),
            "issue_id": rep_row.get("issue_id"),
            "issue_number": rep_row.get("issue_number"),
            "owner_count_all_selected": int(len(owner_keys_all)),
            "owner_count_pre_issue_selected": int(len(owner_keys_pre)),
            "owner_count_post_issue_selected": int(len(owner_keys_post)),
            "commenter_count": int(len(commenter_keys)),
            "owner_commenter_overlap_count_all": int(len(overlap_all)),
            "owner_commenter_overlap_count_pre_issue": int(len(overlap_pre)),
            "owner_commenter_overlap_count_post_issue": int(len(overlap_post)),
            "owner_commenter_overlap_fraction_all": safe_divide(len(overlap_all), len(owner_keys_all), default=np.nan),
            "owner_commenter_overlap_fraction_pre_issue": safe_divide(len(overlap_pre), len(owner_keys_pre), default=np.nan),
            "owner_commenter_overlap_fraction_post_issue": safe_divide(len(overlap_post), len(owner_keys_post), default=np.nan),
            "owner_keys_all_selected_json": json.dumps(owner_keys_all),
            "owner_keys_pre_issue_selected_json": json.dumps(owner_keys_pre),
            "owner_keys_post_issue_selected_json": json.dumps(owner_keys_post),
            "commenter_keys_json": json.dumps(commenter_keys),
            "overlap_keys_all_json": json.dumps(overlap_all),
            "overlap_keys_pre_issue_json": json.dumps(overlap_pre),
            "overlap_keys_post_issue_json": json.dumps(overlap_post),
        })

    overlap_df = pd.DataFrame(owner_rows).sort_values(["repo_full_name", "issue_number"]).reset_index(drop=True) if owner_rows else pd.DataFrame()

    overall = {
        "issues_with_selected_evidence": int(len(overlap_df)),
        "issues_with_any_owner_commenter_overlap": int((overlap_df.get("owner_commenter_overlap_count_all", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not overlap_df.empty else 0,
        "issues_with_pre_issue_owner_commenter_overlap": int((overlap_df.get("owner_commenter_overlap_count_pre_issue", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not overlap_df.empty else 0,
        "issues_with_post_issue_owner_commenter_overlap": int((overlap_df.get("owner_commenter_overlap_count_post_issue", pd.Series(dtype="float64")).fillna(0) > 0).sum()) if not overlap_df.empty else 0,
        "mean_owner_commenter_overlap_fraction_all": None if overlap_df.empty else pd.to_numeric(overlap_df["owner_commenter_overlap_fraction_all"], errors="coerce").dropna().mean(),
        "mean_owner_commenter_overlap_fraction_pre_issue": None if overlap_df.empty else pd.to_numeric(overlap_df["owner_commenter_overlap_fraction_pre_issue"], errors="coerce").dropna().mean(),
        "mean_owner_commenter_overlap_fraction_post_issue": None if overlap_df.empty else pd.to_numeric(overlap_df["owner_commenter_overlap_fraction_post_issue"], errors="coerce").dropna().mean(),
    }
    return overlap_df, overall


def write_markdown_summary(
    output_path: Path,
    git_window_info: dict,
    commit_overall: dict,
    pre_post_overall: dict,
    funnel_overall: dict,
    overlap_overall: dict,
) -> None:
    lines = []
    lines.append("# Ownership Timing Diagnostic Summary")
    lines.append("")
    lines.append("## Git extraction window interpretation")
    lines.append("")
    lines.append(f"- configured include_full_history: `{git_window_info['configured_include_full_history']}`")
    lines.append(f"- configured fast_mode: `{git_window_info['configured_fast_mode']}`")
    lines.append(f"- configured fast_mode_date_window: `{git_window_info['configured_fast_mode_date_window']}`")
    lines.append(f"- effective include_full_history: `{git_window_info['effective_include_full_history']}`")
    lines.append(f"- effective history start: `{git_window_info['effective_history_start_date']}`")
    lines.append(f"- effective history end: `{git_window_info['effective_history_end_date']}`")
    lines.append(f"- commit window restricted: `{git_window_info['commit_window_restricted']}`")
    lines.append(f"- restriction reason: {git_window_info['restriction_reason']}")
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

    commit_ranges_by_repo.to_csv(output_dir / "commit_ranges_by_repo.csv", index=False)
    pre_post_row_summary.to_csv(output_dir / "pre_post_commit_distribution_rows.csv", index=False)
    pre_post_issue_summary.to_csv(output_dir / "pre_post_commit_distribution_by_issue.csv", index=False)
    funnel_by_repo.to_csv(output_dir / "ownership_coverage_funnel_by_repo.csv", index=False)
    owner_commenter_overlap_by_issue.to_csv(output_dir / "owner_commenter_overlap_by_issue.csv", index=False)

    (output_dir / "git_window_interpretation.json").write_text(json.dumps(git_window_info, indent=2), encoding="utf-8")
    (output_dir / "overall_summary.json").write_text(
        json.dumps(
            {
                "git_window_info": git_window_info,
                "commit_overall": commit_overall,
                "pre_post_overall": pre_post_overall,
                "funnel_overall": funnel_overall,
                "overlap_overall": overlap_overall,
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
    )

    print(output_dir)


if __name__ == "__main__":
    main()
