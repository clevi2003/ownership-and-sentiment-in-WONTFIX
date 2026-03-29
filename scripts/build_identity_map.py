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
from utils.chunk_writers import IdentityResolutionRepoChunkWriter
from utils.io_helpers import collect_repo_part_files, load_repo_list, load_table, clean_text, normalize_value, has_real_value, repo_filter, write_merged_or_partitioned_output
from utils.regex_expressions import GITHUB_NOREPLY_EMAIL_PATTERN

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"
LOG_FILENAME = "06_build_identity_map.log"
CHECKPOINT_PREFIX = "06_build_identity_map"
BATCH_FOLDER_NAME = "identity_resolution"
RAW_FOLDER_NAME = "identity_resolution"


def setup_logger(config):
    logger = logging.getLogger("build_identity_map")
    logger.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if config.logging.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.logging.log_to_file:
        log_dir = Path(config.logging.linkage_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / LOG_FILENAME, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_identity_option(config, field_name, default_value):
    return get_stage_option(config, "identity_resolution", field_name, default_value)


def load_stage_inputs_for_repo(config, repo_full_name):
    merge_mode = get_identity_option(config, "input_merge_mode", None)

    issues_df = load_table(config.outputs.issues_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    comments_df = load_table(config.outputs.issue_comments_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    prs_df = load_table(config.outputs.pull_requests_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    commits_df = load_table(config.outputs.commits_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    pr_commit_df = load_table(
        config.outputs.pr_commit_links_table, repo_full_name=repo_full_name, merge_mode=merge_mode)

    return {
        "issues": repo_filter(issues_df, repo_full_name),
        "comments": repo_filter(comments_df, repo_full_name),
        "pull_requests": repo_filter(prs_df, repo_full_name),
        "commits": repo_filter(commits_df, repo_full_name),
        "pr_commit_links": repo_filter(pr_commit_df, repo_full_name),
    }

def normalize_name(value, config):
    value = clean_text(value)
    if not value:
        return None
    rules = config.identity_resolution.normalized_name_rules
    if getattr(rules, "lowercase", False):
        value = value.lower()
    if getattr(rules, "strip_whitespace", False):
        value = value.strip()
    if getattr(rules, "collapse_internal_spaces", False):
        value = " ".join(value.split())
    return value or None


def normalize_email(value, config):
    value = clean_text(value)
    if not value:
        return None
    rules = config.identity_resolution.email_rules
    if getattr(rules, "lowercase", False):
        value = value.lower()
    if getattr(rules, "strip_whitespace", False):
        value = value.strip()
    return value or None


def normalize_login_alias_for_name(value):
    value = clean_text(value)
    if not value:
        return None
    value = value.lower().strip()
    value = value.replace("_", " ")
    value = value.replace("-", " ")
    value = " ".join(value.split())
    return value or None


def extract_login_from_github_noreply_email(value):
    value = clean_text(value)
    if not has_real_value(value):
        return None
    match = GITHUB_NOREPLY_EMAIL_PATTERN.match(value)
    if not match:
        return None
    login_value = clean_text(match.group("login"))
    if not has_real_value(login_value):
        return None
    return normalize_value(login_value)


def choose_login_candidate(raw_login, raw_email):
    normalized_login = normalize_value(raw_login)
    if has_real_value(normalized_login):
        return normalized_login
    noreply_login = extract_login_from_github_noreply_email(raw_email)
    if has_real_value(noreply_login):
        return noreply_login
    return None


def is_probably_ambiguous_name(normalized_name):
    if not normalized_name:
        return True
    weak_values = {"root", "admin", "unknown", "noreply", "user", "github", "git", "ci", "bot", "actions", "renovate", "dependabot"}
    if normalized_name in weak_values:
        return True
    if len(normalized_name) <= 2:
        return True
    tokens = [token for token in normalized_name.split(" ") if token]
    if not tokens:
        return True
    if len(tokens) == 1 and len(tokens[0]) <= 3:
        return True
    return False


def detect_bot_flag(raw_login, raw_name, raw_email, config):
    if not getattr(config.bot_handling, "detect_bots", False):
        return False

    patterns = [str(value).lower() for value in getattr(config.bot_handling, "bot_name_patterns", [])]
    haystacks = [raw_login, raw_name, raw_email]
    for haystack in haystacks:
        if haystack is None:
            continue
        haystack_value = str(haystack).lower()
        for pattern in patterns:
            if pattern and pattern in haystack_value:
                return True
    return False


def choose_canonical_value(series, preferred_non_null=True):
    values = []
    for value in series:
        cleaned = clean_text(value)
        if cleaned:
            values.append(cleaned)
    if not values:
        return None
    if preferred_non_null:
        values = sorted(values, key=lambda item: (len(item), item.lower()))
    return values[0]


def new_repo_result(repo_full_name, repo_id=None):
    return {
        "repo_full_name": repo_full_name,
        "repo_id": repo_id,
        "status": "started",
        "issue_rows_seen": 0,
        "comment_rows_seen": 0,
        "pr_rows_seen": 0,
        "commit_rows_seen": 0,
        "candidate_rows_seen": 0,
        "candidate_rows_deduped": 0,
        "resolved_by_login": 0,
        "resolved_by_email": 0,
        "resolved_by_name": 0,
        "singleton_rows_created": 0,
        "bot_rows": 0,
        "unique_contributor_keys": 0,
        "identity_rows_written": 0,
        "cluster_rows_written": 0,
        "error_message": "",
        "name_alias_candidates_available": 0,
        "resolved_by_noreply_login_bridge": 0,
        "pr_login_email_bridge_pairs": 0,
        "resolved_by_pr_login_email_bridge": 0,
    }


def build_candidate_row(repo_id, repo_full_name, raw_source_type, raw_login=None, raw_name=None, raw_email=None):
    return {
        "repo_id": repo_id,
        "repo_full_name": repo_full_name,
        "raw_source_type": raw_source_type,
        "raw_login": clean_text(raw_login),
        "raw_name": clean_text(raw_name),
        "raw_email": clean_text(raw_email),
    }


def resolve_repo_id_from_stage_inputs(stage_inputs, fallback_repo_id=None):
    """
    try to use caller provided fallback_repo_id (from repo list)
    if that doesn't exist, try first not nan repo_id found in issues / comments / pull_requests / commits
    """
    if pd.notna(fallback_repo_id):
        return fallback_repo_id
    for table_name in ["issues", "comments", "pull_requests", "commits"]:
        df = stage_inputs.get(table_name)
        if df is None or df.empty or "repo_id" not in df.columns:
            continue
        non_null_ids = df["repo_id"].dropna()
        if not non_null_ids.empty:
            return non_null_ids.iloc[0]
    return None


def extract_identity_candidates(repo_full_name, stage_inputs, result, fallback_repo_id=None):
    issues_df = stage_inputs["issues"]
    comments_df = stage_inputs["comments"]
    prs_df = stage_inputs["pull_requests"]
    commits_df = stage_inputs["commits"]

    candidate_rows = []
    # resolve repo_id once for the whole repo before creating candidate rows
    repo_id = resolve_repo_id_from_stage_inputs(stage_inputs, fallback_repo_id=fallback_repo_id)

    if not issues_df.empty:
        result["issue_rows_seen"] = len(issues_df)
        issue_cols = [col for col in ["author_login", "closed_by_login"] if col in issues_df.columns]
        for row in issues_df[issue_cols].to_dict(orient="records"):
            author_login = row.get("author_login")
            if author_login:
                candidate_rows.append(build_candidate_row(repo_id, repo_full_name, "issue_author", raw_login=author_login))

            closed_by_login = row.get("closed_by_login")
            if closed_by_login:
                candidate_rows.append(build_candidate_row(repo_id, repo_full_name, "issue_closer", raw_login=closed_by_login))

    if not comments_df.empty:
        result["comment_rows_seen"] = len(comments_df)
        comment_cols = [col for col in ["author_login"] if col in comments_df.columns]
        for row in comments_df[comment_cols].to_dict(orient="records"):
            author_login = row.get("author_login")
            if author_login:
                candidate_rows.append(build_candidate_row(repo_id, repo_full_name, "issue_comment", raw_login=author_login))

    if not prs_df.empty:
        result["pr_rows_seen"] = len(prs_df)
        pr_cols = [col for col in ["author_login"] if col in prs_df.columns]
        for row in prs_df[pr_cols].to_dict(orient="records"):
            author_login = row.get("author_login")
            if author_login:
                candidate_rows.append(
                    build_candidate_row(repo_id, repo_full_name, "pr_author", raw_login=author_login))

    if not commits_df.empty:
        result["commit_rows_seen"] = len(commits_df)
        commit_cols = [col for col in ["author_name", "author_email"] if col in commits_df.columns]
        for row in commits_df[commit_cols].to_dict(orient="records"):
            if row.get("author_name") or row.get("author_email"):
                candidate_rows.append(
                    build_candidate_row(repo_id, repo_full_name, "commit_author", raw_name=row.get("author_name"), raw_email=row.get("author_email")))

    result["candidate_rows_seen"] = len(candidate_rows)
    if not candidate_rows:
        return pd.DataFrame(), repo_id
    candidates_df = pd.DataFrame(candidate_rows)
    candidates_df = candidates_df.drop_duplicates(
        subset=["repo_full_name", "raw_source_type", "raw_login", "raw_name", "raw_email"]
    ).reset_index(drop=True)
    result["candidate_rows_deduped"] = len(candidates_df)
    return candidates_df, repo_id


def build_commit_email_by_sha_map(commits_df, config):
    if commits_df is None or commits_df.empty:
        return {}
    required_cols = {"commit_sha", "author_email"}
    if not required_cols.issubset(commits_df.columns):
        return {}

    commit_slice = commits_df[["commit_sha", "author_email"]].copy()
    commit_slice["normalized_email"] = commit_slice["author_email"].apply(lambda value: normalize_email(value, config))

    commit_slice = commit_slice[commit_slice["commit_sha"].notna() & commit_slice["normalized_email"].notna()].copy()

    if commit_slice.empty:
        return {}

    commit_slice = commit_slice.drop_duplicates(subset=["commit_sha", "normalized_email"]).reset_index(drop=True)

    return dict(zip(commit_slice["commit_sha"], commit_slice["normalized_email"]))


def build_pr_login_email_bridge(prs_df, pr_commit_links_df, commits_df, config):
    if prs_df is None or prs_df.empty:
        return {"login_to_email": {}, "email_to_login": {}, "evidence_counts": {}}
    if pr_commit_links_df is None or pr_commit_links_df.empty:
        return {"login_to_email": {}, "email_to_login": {}, "evidence_counts": {}}
    if commits_df is None or commits_df.empty:
        return {"login_to_email": {}, "email_to_login": {}, "evidence_counts": {}}

    if "author_login" not in prs_df.columns:
        return {"login_to_email": {}, "email_to_login": {}, "evidence_counts": {}}
    if "commit_sha" not in pr_commit_links_df.columns:
        return {"login_to_email": {}, "email_to_login": {}, "evidence_counts": {}}
    if not ({"pr_id", "pr_number"} & set(pr_commit_links_df.columns)):
        return {"login_to_email": {}, "email_to_login": {}, "evidence_counts": {}}

    commit_email_by_sha = build_commit_email_by_sha_map(commits_df, config)
    if not commit_email_by_sha:
        return {"login_to_email": {}, "email_to_login": {}, "evidence_counts": {}}

    pr_cols = [col for col in ["pr_id", "pr_number", "author_login"] if col in prs_df.columns]
    pr_slice = prs_df[pr_cols].copy()

    if "pr_id" in pr_slice.columns:
        pr_slice["pr_id"] = pd.to_numeric(pr_slice["pr_id"], errors="coerce")
    if "pr_number" in pr_slice.columns:
        pr_slice["pr_number"] = pd.to_numeric(pr_slice["pr_number"], errors="coerce")

    pr_slice["normalized_login"] = pr_slice["author_login"].apply(normalize_value)
    pr_slice = pr_slice[pr_slice["normalized_login"].notna()].copy()

    if pr_slice.empty:
        return {"login_to_email": {}, "email_to_login": {}, "evidence_counts": {}}

    pr_commit_cols = [col for col in ["pr_id", "pr_number", "commit_sha"] if col in pr_commit_links_df.columns]
    pr_commit_slice = pr_commit_links_df[pr_commit_cols].copy()

    if "pr_id" in pr_commit_slice.columns:
        pr_commit_slice["pr_id"] = pd.to_numeric(pr_commit_slice["pr_id"], errors="coerce")
    if "pr_number" in pr_commit_slice.columns:
        pr_commit_slice["pr_number"] = pd.to_numeric(pr_commit_slice["pr_number"], errors="coerce")

    link_rows = []

    # use pr_id when present but can try pr_number if that doesn't work
    if "pr_id" in pr_slice.columns and "pr_id" in pr_commit_slice.columns:
        pr_id_slice = pr_slice[pr_slice["pr_id"].notna()][["pr_id", "normalized_login"]].drop_duplicates()
        pr_commit_id_slice = pr_commit_slice[pr_commit_slice["pr_id"].notna()][["pr_id", "commit_sha"]].drop_duplicates()

        if not pr_id_slice.empty and not pr_commit_id_slice.empty:
            merged_by_id = pr_id_slice.merge(pr_commit_id_slice, on="pr_id", how="inner")
            if not merged_by_id.empty:
                link_rows.append(merged_by_id[["normalized_login", "commit_sha"]])

    if "pr_number" in pr_slice.columns and "pr_number" in pr_commit_slice.columns:
        pr_num_slice = pr_slice[pr_slice["pr_number"].notna()][["pr_number", "normalized_login"]].drop_duplicates()
        pr_commit_num_slice = pr_commit_slice[pr_commit_slice["pr_number"].notna()][["pr_number", "commit_sha"]].drop_duplicates()

        if not pr_num_slice.empty and not pr_commit_num_slice.empty:
            merged_by_num = pr_num_slice.merge(pr_commit_num_slice, on="pr_number", how="inner")
            if not merged_by_num.empty:
                link_rows.append(merged_by_num[["normalized_login", "commit_sha"]])

    if not link_rows:
        return {"login_to_email": {}, "email_to_login": {}, "evidence_counts": {}}

    bridge_df = pd.concat(link_rows, ignore_index=True).drop_duplicates()
    if bridge_df.empty:
        return {"login_to_email": {}, "email_to_login": {}, "evidence_counts": {}}

    bridge_df["normalized_email"] = bridge_df["commit_sha"].map(commit_email_by_sha)
    bridge_df = bridge_df[bridge_df["normalized_login"].notna() & bridge_df["normalized_email"].notna()].copy()

    if bridge_df.empty:
        return {"login_to_email": {}, "email_to_login": {}, "evidence_counts": {}}

    evidence_counts = (bridge_df.groupby(["normalized_login", "normalized_email"]).size().to_dict())

    login_to_emails = (bridge_df.groupby("normalized_login")["normalized_email"]
                       .apply(lambda series: sorted(set(series.dropna().tolist()))).to_dict())

    email_to_logins = (bridge_df.groupby("normalized_email")["normalized_login"]
                       .apply(lambda series: sorted(set(series.dropna().tolist()))).to_dict())

    conservative_login_to_email = {}
    conservative_email_to_login = {}
    for login_value, emails in login_to_emails.items():
        if len(emails) != 1:
            continue
        email_value = emails[0]
        reverse_logins = email_to_logins.get(email_value, [])
        if len(reverse_logins) != 1:
            continue
        if reverse_logins[0] != login_value:
            continue
        conservative_login_to_email[login_value] = email_value
        conservative_email_to_login[email_value] = login_value

    return {
        "login_to_email": conservative_login_to_email,
        "email_to_login": conservative_email_to_login,
        "evidence_counts": evidence_counts,
    }


def assign_contributor_keys(df, repo_id, config):
    if df.empty:
        return df
    key_format = getattr(config.identity_resolution, "contributor_key_format", "{repo_id}::{resolved_identity}")
    safe_repo_id = repo_id if repo_id is not None else "missing_repo_id"
    df = df.copy()
    df["resolved_contributor_key"] = df["resolved_identity"].apply(lambda value: key_format.format(repo_id=safe_repo_id, resolved_identity=value))
    return df


def build_cluster_summary(identity_df, repo_id=None):
    if identity_df.empty:
        return pd.DataFrame()

    if repo_id is not None:
        identity_df = identity_df.copy()
        identity_df["repo_id"] = repo_id

    grouped = identity_df.groupby(["repo_id", "repo_full_name", "resolved_identity", "resolved_contributor_key"], dropna=False)
    rows = []
    for (repo_id, repo_full_name, resolved_identity, contributor_key), group in grouped:
        source_types = sorted({value for value in group["raw_source_type"].dropna().tolist()})
        confidence_values = sorted({value for value in group["identity_confidence"].dropna().tolist()})
        rows.append({"repo_id": repo_id,
                    "repo_full_name": repo_full_name,
                    "resolved_identity": resolved_identity,
                    "resolved_contributor_key": contributor_key,
                    "canonical_login": choose_canonical_value(group.get("raw_login", pd.Series(dtype=object))),
                    "canonical_name": choose_canonical_value(group.get("raw_name", pd.Series(dtype=object))),
                    "canonical_email": choose_canonical_value(group.get("raw_email", pd.Series(dtype=object))),
                    "bot_flag": bool(group["bot_flag"].any()) if "bot_flag" in group else False,
                    "raw_identity_row_count": int(len(group)),
                    "source_types_json": json.dumps(source_types),
                    "confidence_values_json": json.dumps(confidence_values)})
    return pd.DataFrame(rows)


def add_name_map_entry(name_map, name_value, resolved_identity, bot_flag):
    clean_name = clean_text(name_value)
    if not has_real_value(clean_name):
        return
    if is_probably_ambiguous_name(clean_name):
        return
    payload = name_map.setdefault(clean_name, {"clusters": set(), "bot_flags": set()})
    payload["clusters"].add(resolved_identity)
    payload["bot_flags"].add(bool(bot_flag))


def add_email_map_entry(email_map, email_value, resolved_identity, bot_flag):
    email_value = clean_text(email_value)
    if not has_real_value(email_value):
        return
    payload = email_map.setdefault(email_value, {"clusters": set(), "bot_flags": set()})
    payload["clusters"].add(resolved_identity)
    payload["bot_flags"].add(bool(bot_flag))


def resolve_identities(candidates_df, repo_id, config, result, stage_inputs=None):
    if candidates_df.empty:
        return candidates_df, pd.DataFrame()

    df = candidates_df.copy().reset_index(drop=True)
    # make only one repo_id exist for all rows in this repo specific df
    df["repo_id"] = repo_id
    df["normalized_login"] = df["raw_login"].apply(normalize_value)
    df["normalized_name"] = df["raw_name"].apply(lambda value: normalize_name(value, config))
    df["normalized_email"] = df["raw_email"].apply(lambda value: normalize_email(value, config))
    # ideal would be explicit login but still try a probable login from GitHub noreply emails when possible
    df["normalized_login_candidate"] = df.apply(lambda row: choose_login_candidate(row.get("raw_login"), row.get("raw_email")), axis=1)
    df["normalized_login_alias_name"] = df["normalized_login_candidate"].apply(normalize_login_alias_for_name)
    result["name_alias_candidates_available"] = int(df["normalized_login_alias_name"].notna().sum())
    bridge_payload = {"login_to_email": {}, "email_to_login": {}, "evidence_counts": {},}

    if stage_inputs is not None:
        bridge_payload = build_pr_login_email_bridge(stage_inputs.get("pull_requests"),
                                                     stage_inputs.get("pr_commit_links"),
                                                     stage_inputs.get("commits"),
                                                     config)
    result["pr_login_email_bridge_pairs"] = int(len(bridge_payload["login_to_email"]))
    df["bridged_email_from_pr_author_login"] = df["normalized_login_candidate"].map(bridge_payload["login_to_email"])
    df["bot_flag"] = df.apply(
        lambda row: detect_bot_flag(row.get("raw_login"), row.get("raw_name"), row.get("raw_email"), config),
        axis=1,
    )
    result["bot_rows"] = int(df["bot_flag"].sum())

    df["resolved_identity"] = None
    df["identity_confidence"] = None
    df["match_method"] = None

    # login exact, including high-precision recovery from GitHub noreply emails
    login_rows = df[df["normalized_login_candidate"].notna()].copy()
    if not login_rows.empty:
        df.loc[login_rows.index, "resolved_identity"] = login_rows["normalized_login_candidate"].apply(lambda value: f"login:{value}")

        explicit_login_mask = login_rows["normalized_login"].notna()
        inferred_login_mask = ~explicit_login_mask
        if explicit_login_mask.any():
            df.loc[login_rows.index[explicit_login_mask], "identity_confidence"] = "high"
            df.loc[login_rows.index[explicit_login_mask], "match_method"] = "github_login_exact"
        if inferred_login_mask.any():
            df.loc[login_rows.index[inferred_login_mask], "identity_confidence"] = "medium"
            df.loc[login_rows.index[inferred_login_mask], "match_method"] = "github_noreply_login_exact"
        result["resolved_by_login"] = int(explicit_login_mask.sum())
        result["resolved_by_noreply_login_bridge"] = int(inferred_login_mask.sum())

    matching_priority = list(getattr(config.identity_resolution, "matching_priority", [])) or ["github_login_exact", "email_exact", "normalized_name_exact"]

    for match_method in matching_priority:
        if match_method == "github_login_exact":
            continue
        unresolved = df[df["resolved_identity"].isna()].copy()
        if unresolved.empty:
            break

        if match_method == "email_exact":
            # get email map from esolved rows with normalized_email then try resolved login clusters with pr login to commit email bridge
            resolved_rows = df[df["resolved_identity"].notna()].copy()
            email_map = {}

            for _, resolved_row in resolved_rows.iterrows():
                resolved_identity = resolved_row.get("resolved_identity")
                bot_flag = resolved_row.get("bot_flag")

                # direct email
                add_email_map_entry(email_map, resolved_row.get("normalized_email"), resolved_identity, bot_flag)
                # bridged email from PR author login → commit email
                add_email_map_entry(email_map, resolved_row.get("bridged_email_from_pr_author_login"), resolved_identity, bot_flag)

            attach_count = 0
            bridge_attach_count = 0
            for index, row in unresolved.iterrows():
                email_value = row.get("normalized_email")
                if not has_real_value(email_value):
                    continue
                payload = email_map.get(email_value)
                if not payload:
                    continue
                if len(payload["clusters"]) != 1:
                    continue
                if row.get("bot_flag") and payload["bot_flags"] == {False}:
                    continue
                if (not row.get("bot_flag")) and payload["bot_flags"] == {True}:
                    continue
                resolved_identity = next(iter(payload["clusters"]))
                df.at[index, "resolved_identity"] = resolved_identity
                df.at[index, "identity_confidence"] = "medium"
                df.at[index, "match_method"] = "email_exact"
                attach_count += 1
                # track if this came from PR bridge
                if email_value in bridge_payload["email_to_login"]:
                    bridge_attach_count += 1

            result["resolved_by_email"] = attach_count
            result["resolved_by_pr_login_email_bridge"] = bridge_attach_count

        elif match_method == "normalized_name_exact":
            resolved_rows = df[df["resolved_identity"].notna()].copy()
            name_map = {}

            for _, resolved_row in resolved_rows.iterrows():
                resolved_identity = resolved_row.get("resolved_identity")
                bot_flag = resolved_row.get("bot_flag")
                # real name
                add_name_map_entry(name_map, resolved_row.get("normalized_name"), resolved_identity, bot_flag)
                # login alias
                add_name_map_entry(name_map, resolved_row.get("normalized_login_alias_name"), resolved_identity, bot_flag)

            attach_count = 0
            for index, row in unresolved.iterrows():
                name_value = clean_text(row.get("normalized_name"))
                if not has_real_value(name_value):
                    continue
                if is_probably_ambiguous_name(name_value):
                    continue
                payload = name_map.get(name_value)
                if not payload:
                    continue
                if len(payload["clusters"]) != 1:
                    continue
                if row.get("bot_flag") and payload["bot_flags"] == {False}:
                    continue
                if (not row.get("bot_flag")) and payload["bot_flags"] == {True}:
                    continue

                resolved_identity = next(iter(payload["clusters"]))
                df.at[index, "resolved_identity"] = resolved_identity
                df.at[index, "identity_confidence"] = "low"
                df.at[index, "match_method"] = "normalized_name_exact"
                attach_count += 1
            result["resolved_by_name"] = attach_count

    # indivisual fallbacks for everything unresolved
    unresolved = df[df["resolved_identity"].isna()].copy()
    if not unresolved.empty:
        singleton_count = 0
        for index, row in unresolved.iterrows():
            normalized_login_candidate = clean_text(row.get("normalized_login_candidate"))
            normalized_email = clean_text(row.get("normalized_email"))
            normalized_name = clean_text(row.get("normalized_name"))

            if has_real_value(normalized_login_candidate):
                resolved_identity = f"singleton_login:{normalized_login_candidate}"
            elif has_real_value(normalized_email):
                resolved_identity = f"singleton_email:{normalized_email}"
            elif has_real_value(normalized_name):
                resolved_identity = f"singleton_name:{normalized_name}"
            else:
                resolved_identity = f"singleton_row:{index}"

            df.at[index, "resolved_identity"] = resolved_identity
            df.at[index, "identity_confidence"] = "singleton"
            df.at[index, "match_method"] = "singleton"
            singleton_count += 1
        result["singleton_rows_created"] = singleton_count

    df = assign_contributor_keys(df, repo_id, config)
    result["unique_contributor_keys"] = int(df["resolved_contributor_key"].nunique())

    if not getattr(config.identity_resolution, "preserve_normalized_columns", True):
        df = df.drop(
            columns=["normalized_login", "normalized_name", "normalized_email", "normalized_login_alias_name", "normalized_login_candidate", "bridged_email_from_pr_author_login"],
            errors="ignore",
        )
    cluster_df = build_cluster_summary(df, repo_id=repo_id)
    return df, cluster_df


def merge_identity_batches(config, logger):
    batch_root = get_batch_root(config, BATCH_FOLDER_NAME)
    if not batch_root.exists():
        logger.warning("Identity batch root does not exist: %s", batch_root)
        return

    identity_repo_parts = collect_repo_part_files(batch_root,"contributor_identity_map_part_*.parquet")
    cluster_repo_parts = collect_repo_part_files(batch_root,"contributor_identity_clusters_part_*.parquet")

    if identity_repo_parts:
        mode_used = write_merged_or_partitioned_output(
            repo_part_map=identity_repo_parts,
            output_path=config.outputs.contributor_identity_table,
            config=config,
            table_name="contributor_identity_map",
            sort_columns=["repo_full_name", "resolved_contributor_key", "raw_source_type"],
            dedupe_subset=["repo_full_name", "raw_source_type", "raw_login", "raw_name", "raw_email", "resolved_contributor_key"],
        )
        logger.info("Wrote contributor identity map using %s mode to %s", mode_used, config.outputs.contributor_identity_table)
    else:
        logger.warning("No contributor identity parts found to merge.")

    cluster_output_path = getattr(config.outputs, "contributor_identity_clusters_table", None)
    if cluster_output_path and cluster_repo_parts:
        mode_used = write_merged_or_partitioned_output(
            repo_part_map=cluster_repo_parts,
            output_path=cluster_output_path,
            config=config,
            table_name="contributor_identity_clusters",
            sort_columns=["repo_full_name", "resolved_contributor_key"],
            dedupe_subset=["repo_full_name", "resolved_contributor_key"],
        )
        logger.info("Wrote contributor identity clusters using %s mode to %s", mode_used, cluster_output_path)


def write_summary_csv(summary_rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)


def write_run_manifest(config, repo_rows, summary_rows):
    manifest_path = Path(config.logging.linkage_log_dir) / "06_build_identity_map_run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "build_identity_map.py",
        "repo_count_requested": len(repo_rows),
        "repo_count_processed": len(summary_rows),
        "completed_repo_count": sum(1 for row in summary_rows if row.get("status") == "completed"),
        "failed_repo_count": sum(1 for row in summary_rows if row.get("status") == "failed"),
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "processed_merge_mode": getattr(config.storage, "processed_merge_mode", "single_parquet"),
        "matching_priority": list(getattr(config.identity_resolution, "matching_priority", [])),
        "summary_rows": summary_rows,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def process_repo(config, logger, repo_row):
    repo_full_name = repo_row["full_name"]
    result = new_repo_result(repo_full_name, repo_row.get("repo_id"))
    stage_inputs = load_stage_inputs_for_repo(config, repo_full_name)
    candidates_df, repo_id = extract_identity_candidates(repo_full_name, stage_inputs, result, fallback_repo_id=repo_row.get("repo_id"))
    result["repo_id"] = repo_id
    if candidates_df.empty:
        result["status"] = "completed"
        return result

    batch_size = get_identity_option(config, "write_batch_size", 5000)
    repo_dir = get_batch_root(config, BATCH_FOLDER_NAME) / sanitize_repo_name(repo_full_name)
    writer = IdentityResolutionRepoChunkWriter(config=config, repo_dir=repo_dir, batch_size=batch_size)

    identity_df, cluster_df = resolve_identities(candidates_df, repo_id, config, result, stage_inputs=stage_inputs)
    for row in identity_df.to_dict(orient="records"):
        writer.add_identity_row(row)
        result["identity_rows_written"] += 1
    if get_identity_option(config, "write_cluster_summary", True) and not cluster_df.empty:
        for row in cluster_df.to_dict(orient="records"):
            writer.add_cluster_row(row)
            result["cluster_rows_written"] += 1
    writer.finalize()
    result["status"] = "completed"
    return result


def main():
    config = load_study_config(DEFAULT_CONFIG_PATH)
    ensure_project_directories(config)
    logger = setup_logger(config)

    if not getattr(config.identity_resolution, "enabled", True):
        logger.warning("identity_resolution.enabled is false; nothing to do.")
        return

    if get_identity_option(config, "resume_mode", "fresh") == "fresh":
        reset_batch_root(config, BATCH_FOLDER_NAME)

    repo_rows = load_repo_list(config.outputs.repo_included_list)
    max_repos = get_identity_option(config, "max_repos_per_run", None)
    if max_repos:
        repo_rows = repo_rows[:max_repos]

    summary_rows = []
    for repo_row in repo_rows:
        repo_full_name = repo_row["full_name"]
        skip_repo, reason = should_skip_repo(
            config,
            repo_full_name,
            checkpoint_prefix=CHECKPOINT_PREFIX,
            raw_folder_name=RAW_FOLDER_NAME,
            section_name="identity_resolution",
            raw_source="linked",
        )
        if skip_repo:
            logger.info("Skipping %s due to %s", repo_full_name, reason)
            summary_rows.append(
                {
                    "repo_full_name": repo_full_name,
                    "repo_id": repo_row.get("repo_id"),
                    "status": f"skipped_{reason}",
                }
            )
            continue

        try:
            logger.info("Starting identity resolution for %s", repo_full_name)
            result = process_repo(config, logger, repo_row)
            summary_rows.append(result)
            write_repo_checkpoint(config, CHECKPOINT_PREFIX, repo_full_name, result)
        except Exception as exc:
            logger.exception("Identity resolution failed for %s", repo_full_name)
            error_row = new_repo_result(repo_full_name, repo_row.get("repo_id"))
            error_row["status"] = "failed"
            error_row["error_message"] = str(exc)
            summary_rows.append(error_row)
            write_repo_checkpoint(config, CHECKPOINT_PREFIX, repo_full_name, error_row)

    merge_identity_batches(config, logger)
    write_summary_csv(summary_rows, Path(config.logging.linkage_log_dir) / "06_build_identity_map_summary.csv")
    write_run_manifest(config, repo_rows, summary_rows)
    logger.info("Identity resolution complete.")


if __name__ == "__main__":
    main()
