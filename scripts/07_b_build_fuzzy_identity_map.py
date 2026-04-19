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
from utils.checkpoints import (
    get_batch_root,
    get_stage_option,
    reset_batch_root,
    sanitize_repo_name,
    should_skip_repo,
    write_repo_checkpoint,
)
from utils.chunk_writers import IdentityResolutionRepoChunkWriter
from utils.io_helpers import (
    clean_text,
    collect_repo_part_files,
    has_real_value,
    load_repo_list,
    load_table,
    normalize_value,
    repo_filter,
    write_merged_or_partitioned_output,
)
from utils.regex_expressions import GITHUB_NOREPLY_EMAIL_PATTERN

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"
LOG_FILENAME = "07_b_build_fuzzy_identity_map.log"
CHECKPOINT_PREFIX = "07_b_build_fuzzy_identity_map"
BATCH_FOLDER_NAME = "fuzzy_identity_resolution"
RAW_FOLDER_NAME = "fuzzy_identity_resolution"
PAIR_AUDIT_FILENAME = "fuzzy_identity_pair_audit.csv"
SUMMARY_FILENAME = "07_b_build_fuzzy_identity_map_summary.csv"
RUN_MANIFEST_FILENAME = "07_b_build_fuzzy_identity_map_run_manifest.json"


def setup_logger(config):
    logger = logging.getLogger("build_fuzzy_identity_map")
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

def get_fuzzy_option(config, field_name, default_value):
    return get_stage_option(config, "fuzzy_identity_resolution", field_name, default_value)

def load_stage_inputs_for_repo(config, repo_full_name):
    merge_mode = get_fuzzy_option(config, "input_merge_mode", None)

    issues_df = load_table(config.outputs.issues_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    comments_df = load_table(config.outputs.issue_comments_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    prs_df = load_table(config.outputs.pull_requests_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    commits_df = load_table(config.outputs.commits_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    pr_commit_df = load_table(config.outputs.pr_commit_links_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    issue_pr_df = load_table(config.outputs.issue_pr_links_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    strict_identity_df = load_table(config.outputs.contributor_identity_table, repo_full_name=repo_full_name, merge_mode=merge_mode)

    return {
        "issues": repo_filter(issues_df, repo_full_name),
        "comments": repo_filter(comments_df, repo_full_name),
        "pull_requests": repo_filter(prs_df, repo_full_name),
        "commits": repo_filter(commits_df, repo_full_name),
        "pr_commit_links": repo_filter(pr_commit_df, repo_full_name),
        "issue_pr_links": repo_filter(issue_pr_df, repo_full_name),
        "strict_identity_map": repo_filter(strict_identity_df, repo_full_name),
    }

def normalize_name(value, config):
    value = clean_text(value)
    if not value:
        return None

    rules = getattr(config.identity_resolution, "normalized_name_rules", None)
    if rules is not None:
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
    rules = getattr(config.identity_resolution, "email_rules", None)
    if rules is not None:
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

def jaro_winkler_similarity(s1, s2):
    s1 = clean_text(s1)
    s2 = clean_text(s2)
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    s1 = str(s1)
    s2 = str(s2)
    len1 = len(s1)
    len2 = len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    transpositions /= 2.0
    jaro = (
        (matches / len1)
        + (matches / len2)
        + ((matches - transpositions) / matches)
    ) / 3.0

    prefix = 0
    max_prefix = 4
    for i in range(min(len1, len2, max_prefix)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + (prefix * 0.1 * (1.0 - jaro))

class UnionFind:
    def __init__(self, elements):
        self.parent = {element: element for element in elements}
        self.rank = {element: 0 for element in elements}

    def find(self, item):
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left, right):
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        rank_left = self.rank[root_left]
        rank_right = self.rank[root_right]
        if rank_left < rank_right:
            self.parent[root_left] = root_right
        elif rank_left > rank_right:
            self.parent[root_right] = root_left
        else:
            self.parent[root_right] = root_left
            self.rank[root_left] += 1

def new_repo_result(repo_full_name, repo_id=None):
    return {
        "repo_full_name": repo_full_name,
        "repo_id": repo_id,
        "status": "started",
        "strict_identity_rows_seen": 0,
        "strict_cluster_keys_seen": 0,
        "candidate_pair_count": 0,
        "pair_edges_retained": 0,
        "strong_merge_edges_retained": 0,
        "moderate_merge_edges_retained": 0,
        "blocked_bot_pairs": 0,
        "rejected_low_score_pairs": 0,
        "rejected_insufficient_signal_pairs": 0,
        "fuzzy_clusters_written": 0,
        "fuzzy_identity_rows_written": 0,
        "singleton_fuzzy_clusters": 0,
        "merged_fuzzy_clusters": 0,
        "clusters_with_commit_author": 0,
        "clusters_with_issue_comment_and_commit": 0,
        "clusters_with_pr_author_and_commit": 0,
        "clusters_flagged_large": 0,
        "rejected_name_without_structure_pairs": 0,
        "rejected_bot_pairs": 0,
        "candidate_pairs_commit_discussion_scope": 0,
        "candidate_pairs_other_scope": 0,
        "candidate_pairs_from_anchor": 0,
        "candidate_pairs_from_name_similarity": 0,
        "candidate_pairs_from_email_localpart_similarity": 0,
        "candidate_pairs_from_pr_bridge": 0,
        "candidate_pairs_from_discussion_bridge": 0,
        "candidate_pairs_from_discussion_bridge_three_plus_shared_issues": 0,
        "candidate_pairs_from_discussion_bridge_two_shared_issues_plus_name_similarity": 0,
        "candidate_pairs_from_discussion_bridge_two_shared_issues_plus_email_localpart_similarity": 0,
        "candidate_pairs_from_discussion_bridge_two_shared_issues_plus_pr_bridge": 0,
        "candidate_pairs_from_discussion_bridge_plus_name_similarity": 0,
        "candidate_pairs_from_discussion_bridge_plus_email_localpart_similarity": 0,
        "candidate_pairs_from_discussion_bridge_plus_pr_bridge": 0,
        "error_message": "",
    }

def choose_canonical_value(values):
    cleaned_values = []
    for value in values:
        cleaned = clean_text(value)
        if cleaned:
            cleaned_values.append(cleaned)
    if not cleaned_values:
        return None
    cleaned_values = sorted(cleaned_values, key=lambda item: (len(item), item.lower()))
    return cleaned_values[0]

def build_strict_identity_candidates(repo_full_name, stage_inputs, result, config):
    strict_identity_df = stage_inputs["strict_identity_map"].copy()
    if strict_identity_df.empty:
        return pd.DataFrame()

    result["strict_identity_rows_seen"] = int(len(strict_identity_df))
    if "resolved_contributor_key" in strict_identity_df.columns:
        result["strict_cluster_keys_seen"] = int(strict_identity_df["resolved_contributor_key"].dropna().nunique())

    df = strict_identity_df.copy()
    df["raw_login"] = df.get("raw_login", pd.Series(dtype="object")).apply(clean_text)
    df["raw_name"] = df.get("raw_name", pd.Series(dtype="object")).apply(clean_text)
    df["raw_email"] = df.get("raw_email", pd.Series(dtype="object")).apply(clean_text)
    df["normalized_login"] = df["raw_login"].apply(normalize_value)
    df["normalized_name"] = df["raw_name"].apply(lambda value: normalize_name(value, config))
    df["normalized_email"] = df["raw_email"].apply(lambda value: normalize_email(value, config))
    df["login_alias_for_name"] = df["raw_login"].apply(normalize_login_alias_for_name)
    df["email_localpart"] = df["normalized_email"].apply(lambda value: value.split("@", 1)[0] if has_real_value(value) and "@" in value else None)
    df["noreply_login"] = df["normalized_email"].apply(extract_login_from_github_noreply_email)
    df["is_bot_flag"] = [
        1 if detect_bot_flag(row.get("raw_login"), row.get("raw_name"), row.get("raw_email"), config) else 0
        for row in df.to_dict(orient="records")
    ]

    activity_columns = [
        "raw_source_type",
        "resolved_contributor_key",
        "repo_full_name",
        "raw_login",
        "raw_name",
        "raw_email",
        "normalized_login",
        "normalized_name",
        "normalized_email",
        "login_alias_for_name",
        "email_localpart",
        "noreply_login",
        "is_bot_flag",
    ]
    for column_name in activity_columns:
        if column_name not in df.columns:
            df[column_name] = None

    return df[activity_columns].drop_duplicates().reset_index(drop=True)

def build_commit_activity_lookup(stage_inputs, config):
    commits_df = stage_inputs["commits"].copy()
    pr_commit_links_df = stage_inputs["pr_commit_links"].copy()
    issue_pr_links_df = stage_inputs["issue_pr_links"].copy()
    prs_df = stage_inputs["pull_requests"].copy()
    comments_df = stage_inputs["comments"].copy()
    issues_df = stage_inputs["issues"].copy()

    commit_lookup = {}
    if commits_df.empty:
        return commit_lookup

    commits_df["normalized_name"] = commits_df.get("author_name", pd.Series(dtype="object")).apply(
        lambda value: normalize_name(value, config)
    )
    commits_df["normalized_email"] = commits_df.get("author_email", pd.Series(dtype="object")).apply(
        lambda value: normalize_email(value, config)
    )
    commits_df["email_localpart"] = commits_df["normalized_email"].apply(
        lambda value: value.split("@", 1)[0] if has_real_value(value) and "@" in value else None
    )
    commits_df["commit_sha"] = commits_df.get("commit_sha", pd.Series(dtype="object")).apply(clean_text)

    pr_id_to_login = {}
    pr_number_to_login = {}
    if not prs_df.empty and "author_login" in prs_df.columns:
        for row in prs_df.to_dict(orient="records"):
            author_login = normalize_value(row.get("author_login"))
            if not has_real_value(author_login):
                continue
            if row.get("pr_id") is not None and pd.notna(row.get("pr_id")):
                pr_id_to_login[row.get("pr_id")] = author_login
            if row.get("pr_number") is not None and pd.notna(row.get("pr_number")):
                pr_number_to_login[int(row.get("pr_number"))] = author_login

    pr_id_to_issue_numbers = {}
    pr_number_to_issue_numbers = {}
    if not issue_pr_links_df.empty:
        issue_pr_links_df = issue_pr_links_df.copy()
        if "issue_number" in issue_pr_links_df.columns:
            issue_pr_links_df["issue_number"] = pd.to_numeric(issue_pr_links_df["issue_number"], errors="coerce")
        if "pr_number" in issue_pr_links_df.columns:
            issue_pr_links_df["pr_number"] = pd.to_numeric(issue_pr_links_df["pr_number"], errors="coerce")

        for row in issue_pr_links_df.to_dict(orient="records"):
            issue_number = row.get("issue_number")
            if issue_number is None or pd.isna(issue_number):
                continue
            issue_number = int(issue_number)

            pr_id = row.get("pr_id")
            if pr_id is not None and pd.notna(pr_id):
                pr_id_to_issue_numbers.setdefault(pr_id, set()).add(issue_number)

            pr_number = row.get("pr_number")
            if pr_number is not None and pd.notna(pr_number):
                pr_number_to_issue_numbers.setdefault(int(pr_number), set()).add(issue_number)

    issue_number_to_issue_author_login = {}
    if not issues_df.empty and "issue_number" in issues_df.columns and "author_login" in issues_df.columns:
        issue_slice = issues_df.copy()
        issue_slice["issue_number"] = pd.to_numeric(issue_slice["issue_number"], errors="coerce")
        issue_slice = issue_slice[issue_slice["issue_number"].notna()].copy()
        for row in issue_slice.to_dict(orient="records"):
            author_login = normalize_value(row.get("author_login"))
            if not has_real_value(author_login):
                continue
            issue_number_to_issue_author_login[int(row.get("issue_number"))] = author_login

    issue_number_to_comment_author_logins = {}
    if not comments_df.empty and "issue_number" in comments_df.columns and "author_login" in comments_df.columns:
        comment_slice = comments_df.copy()
        comment_slice["issue_number"] = pd.to_numeric(comment_slice["issue_number"], errors="coerce")
        comment_slice = comment_slice[comment_slice["issue_number"].notna()].copy()
        for row in comment_slice.to_dict(orient="records"):
            author_login = normalize_value(row.get("author_login"))
            if not has_real_value(author_login):
                continue
            issue_number = int(row.get("issue_number"))
            issue_number_to_comment_author_logins.setdefault(issue_number, set()).add(author_login)

    commit_to_pr_author_counts = {}
    commit_to_pr_numbers = {}
    commit_to_issue_numbers = {}
    commit_to_issue_author_counts = {}
    commit_to_comment_author_counts = {}

    if not pr_commit_links_df.empty and "commit_sha" in pr_commit_links_df.columns:
        for row in pr_commit_links_df.to_dict(orient="records"):
            commit_sha = clean_text(row.get("commit_sha"))
            if not commit_sha:
                continue

            pr_login = None
            pr_id = row.get("pr_id")
            pr_number = row.get("pr_number")

            if pr_id is not None and pd.notna(pr_id) and pr_id in pr_id_to_login:
                pr_login = pr_id_to_login.get(pr_id)
            elif pr_number is not None and pd.notna(pr_number):
                pr_login = pr_number_to_login.get(int(pr_number))

            if pr_login:
                login_counts = commit_to_pr_author_counts.setdefault(commit_sha, {})
                login_counts[pr_login] = login_counts.get(pr_login, 0) + 1

            issue_numbers = set()
            if pr_id is not None and pd.notna(pr_id):
                issue_numbers.update(pr_id_to_issue_numbers.get(pr_id, set()))
            if pr_number is not None and pd.notna(pr_number):
                pr_number_int = int(pr_number)
                commit_to_pr_numbers.setdefault(commit_sha, set()).add(pr_number_int)
                issue_numbers.update(pr_number_to_issue_numbers.get(pr_number_int, set()))

            for issue_number in sorted(issue_numbers):
                commit_to_issue_numbers.setdefault(commit_sha, set()).add(issue_number)

                issue_author_login = issue_number_to_issue_author_login.get(issue_number)
                if issue_author_login:
                    author_counts = commit_to_issue_author_counts.setdefault(commit_sha, {})
                    author_counts[issue_author_login] = author_counts.get(issue_author_login, 0) + 1

                for comment_login in issue_number_to_comment_author_logins.get(issue_number, set()):
                    comment_counts = commit_to_comment_author_counts.setdefault(commit_sha, {})
                    comment_counts[comment_login] = comment_counts.get(comment_login, 0) + 1

    for row in commits_df.to_dict(orient="records"):
        commit_sha = row.get("commit_sha")
        if not commit_sha:
            continue
        commit_lookup[commit_sha] = {
            "normalized_name": row.get("normalized_name"),
            "normalized_email": row.get("normalized_email"),
            "email_localpart": row.get("email_localpart"),
            "pr_author_login_counts": dict(commit_to_pr_author_counts.get(commit_sha, {})),
            "pr_numbers": sorted(commit_to_pr_numbers.get(commit_sha, set())),
            "issue_numbers": sorted(commit_to_issue_numbers.get(commit_sha, set())),
            "issue_author_login_counts": dict(commit_to_issue_author_counts.get(commit_sha, {})),
            "comment_author_login_counts": dict(commit_to_comment_author_counts.get(commit_sha, {})),
        }

    return commit_lookup

def build_activity_sets_for_identity(identity_df, commit_lookup):
    activity = {}
    if identity_df.empty:
        return activity

    for row in identity_df.to_dict(orient="records"):
        key = row.get("resolved_contributor_key")
        if not key:
            continue

        payload = activity.setdefault(
            key,
            {
                "source_types": set(),
                "logins": set(),
                "names": set(),
                "emails": set(),
                "name_aliases": set(),
                "email_localparts": set(),
                "noreply_logins": set(),
                "pr_bridge_logins": set(),
                "discussion_logins": set(),
                "bridge_pr_login_counts": {},
                "bridge_issue_login_counts": {},
                "bridge_comment_login_counts": {},
                "bridge_pr_numbers": set(),
                "bridge_issue_numbers": set(),
                "commit_count": 0,
                "is_bot_flag": False,
            },
        )

        raw_source_type = clean_text(row.get("raw_source_type"))
        if raw_source_type:
            payload["source_types"].add(raw_source_type)

        normalized_login = normalize_value(row.get("normalized_login"))
        normalized_name = clean_text(row.get("normalized_name"))
        normalized_email = clean_text(row.get("normalized_email"))
        login_alias_for_name = clean_text(row.get("login_alias_for_name"))
        email_localpart = clean_text(row.get("email_localpart"))
        noreply_login = normalize_value(row.get("noreply_login"))

        if normalized_login:
            payload["logins"].add(normalized_login)
        if normalized_name:
            payload["names"].add(normalized_name)
        if normalized_email:
            payload["emails"].add(normalized_email)
        if login_alias_for_name:
            payload["name_aliases"].add(login_alias_for_name)
        if email_localpart:
            payload["email_localparts"].add(email_localpart)
        if noreply_login:
            payload["noreply_logins"].add(noreply_login)
        if int(row.get("is_bot_flag") or 0) == 1:
            payload["is_bot_flag"] = True

        if raw_source_type == "commit_author":
            raw_commit_sha_value = row.get("raw_commit_sha")
            commit_shas = []

            if isinstance(raw_commit_sha_value, list):
                commit_shas = [clean_text(value) for value in raw_commit_sha_value if clean_text(value)]
            else:
                parsed = None
                text_value = clean_text(raw_commit_sha_value)
                if text_value:
                    try:
                        parsed = json.loads(text_value)
                    except Exception:
                        parsed = None

                if isinstance(parsed, list):
                    commit_shas = [clean_text(value) for value in parsed if clean_text(value)]
                else:
                    single_sha = clean_text(raw_commit_sha_value)
                    if single_sha:
                        commit_shas = [single_sha]

            unique_commit_shas = sorted(set(commit_shas))
            payload["commit_count"] += len(unique_commit_shas)

            for commit_sha in unique_commit_shas:
                if commit_sha not in commit_lookup:
                    continue
                lookup_payload = commit_lookup[commit_sha]

                for login_value, count_value in lookup_payload.get("pr_author_login_counts", {}).items():
                    payload["pr_bridge_logins"].add(login_value)
                    payload["bridge_pr_login_counts"][login_value] = (
                        payload["bridge_pr_login_counts"].get(login_value, 0) + int(count_value)
                    )

                for login_value, count_value in lookup_payload.get("issue_author_login_counts", {}).items():
                    payload["discussion_logins"].add(login_value)
                    payload["bridge_issue_login_counts"][login_value] = (
                        payload["bridge_issue_login_counts"].get(login_value, 0) + int(count_value)
                    )

                for login_value, count_value in lookup_payload.get("comment_author_login_counts", {}).items():
                    payload["discussion_logins"].add(login_value)
                    payload["bridge_comment_login_counts"][login_value] = (
                        payload["bridge_comment_login_counts"].get(login_value, 0) + int(count_value)
                    )

                for pr_number in lookup_payload.get("pr_numbers", []):
                    payload["bridge_pr_numbers"].add(int(pr_number))

                for issue_number in lookup_payload.get("issue_numbers", []):
                    payload["bridge_issue_numbers"].add(int(issue_number))

    return activity

def add_commit_sha_column(identity_df, stage_inputs, config):
    if identity_df.empty:
        return identity_df

    commits_df = stage_inputs["commits"].copy()
    out = identity_df.copy()
    out["raw_commit_sha"] = None

    if commits_df.empty:
        return out

    commits_df["normalized_name"] = commits_df.get("author_name", pd.Series(dtype="object")).apply(
        lambda value: normalize_name(value, config)
    )
    commits_df["normalized_email"] = commits_df.get("author_email", pd.Series(dtype="object")).apply(
        lambda value: normalize_email(value, config)
    )
    commits_df["commit_sha"] = commits_df.get("commit_sha", pd.Series(dtype="object")).apply(clean_text)
    commits_df = commits_df[commits_df["commit_sha"].notna()].copy()

    if commits_df.empty:
        return out

    pair_lookup = {}
    email_lookup = {}
    name_lookup = {}

    for row in commits_df.to_dict(orient="records"):
        commit_sha = row.get("commit_sha")
        normalized_name = row.get("normalized_name")
        normalized_email = row.get("normalized_email")

        if normalized_name or normalized_email:
            pair_key = (normalized_name, normalized_email)
            pair_lookup.setdefault(pair_key, set()).add(commit_sha)

        if normalized_email:
            email_lookup.setdefault(normalized_email, set()).add(commit_sha)

        if normalized_name:
            name_lookup.setdefault(normalized_name, set()).add(commit_sha)

    if "raw_source_type" not in out.columns:
        return out

    commit_identity_mask = out["raw_source_type"].astype(str) == "commit_author"
    if not commit_identity_mask.any():
        return out

    commit_side = out[commit_identity_mask].copy()
    raw_commit_sha_values = []

    for row in commit_side.to_dict(orient="records"):
        normalized_name = row.get("normalized_name")
        normalized_email = row.get("normalized_email")

        matched_shas = set()

        pair_key = (normalized_name, normalized_email)
        if pair_key in pair_lookup:
            matched_shas.update(pair_lookup[pair_key])

        if not matched_shas and normalized_email in email_lookup:
            matched_shas.update(email_lookup[normalized_email])

        if not matched_shas and normalized_name in name_lookup:
            matched_shas.update(name_lookup[normalized_name])

        if matched_shas:
            raw_commit_sha_values.append(json.dumps(sorted(matched_shas)))
        else:
            raw_commit_sha_values.append(None)

    out.loc[commit_identity_mask, "raw_commit_sha"] = raw_commit_sha_values
    return out

def get_shared_values(left_values, right_values):
    return sorted(set(left_values or set()).intersection(set(right_values or set())))

def should_consider_commit_discussion_pair(left_activity, right_activity):
    left_sources = set(left_activity.get("source_types", set()))
    right_sources = set(right_activity.get("source_types", set()))

    commit_left = "commit_author" in left_sources
    commit_right = "commit_author" in right_sources
    if commit_left == commit_right:
        return False

    discussion_or_pr_sources = {"issue_comment", "pr_author", "issue_author"}
    other_sources = right_sources if commit_left else left_sources
    return len(other_sources.intersection(discussion_or_pr_sources)) > 0

def get_repeated_pr_bridge_stats(left, right):
    left_logins = set(left.get("logins", set()))
    right_logins = set(right.get("logins", set()))

    left_bridge_counts = dict(left.get("bridge_pr_login_counts", {}))
    right_bridge_counts = dict(right.get("bridge_pr_login_counts", {}))

    left_overlap_logins = sorted(set(left_bridge_counts.keys()).intersection(right_logins))
    right_overlap_logins = sorted(set(right_bridge_counts.keys()).intersection(left_logins))

    overlap_logins = sorted(set(left_overlap_logins).union(set(right_overlap_logins)))

    commit_hits = 0
    pr_hits = 0
    bijective = 0

    for login_value in overlap_logins:
        left_count = int(left_bridge_counts.get(login_value, 0))
        right_count = int(right_bridge_counts.get(login_value, 0))
        overlap_count = max(left_count, right_count)
        commit_hits += overlap_count
        if overlap_count > 0:
            pr_hits += 1
        if overlap_count >= 2 and left_count <= 2 and right_count <= 2:
            bijective = 1

    return {
        "overlap_logins": overlap_logins,
        "commit_hits": int(commit_hits),
        "pr_hits": int(pr_hits),
        "bijective": int(bijective),
    }

def get_repeated_discussion_bridge_stats(left, right):
    left_logins = set(left.get("logins", set()))
    right_logins = set(right.get("logins", set()))

    left_issue_counts = dict(left.get("bridge_issue_login_counts", {}))
    left_comment_counts = dict(left.get("bridge_comment_login_counts", {}))
    right_issue_counts = dict(right.get("bridge_issue_login_counts", {}))
    right_comment_counts = dict(right.get("bridge_comment_login_counts", {}))

    overlap_logins = sorted(
        set(left_issue_counts.keys()).intersection(right_logins)
        | set(left_comment_counts.keys()).intersection(right_logins)
        | set(right_issue_counts.keys()).intersection(left_logins)
        | set(right_comment_counts.keys()).intersection(left_logins)
    )

    issue_hits = 0
    for login_value in overlap_logins:
        issue_hits += max(
            int(left_issue_counts.get(login_value, 0)) + int(left_comment_counts.get(login_value, 0)),
            int(right_issue_counts.get(login_value, 0)) + int(right_comment_counts.get(login_value, 0)),
        )

    shared_issue_numbers = sorted(
        set(left.get("bridge_issue_numbers", set())).intersection(set(right.get("bridge_issue_numbers", set())))
    )

    return {
        "overlap_logins": overlap_logins,
        "issue_hits": int(issue_hits),
        "shared_issue_numbers": shared_issue_numbers,
        "shared_issue_count": int(len(shared_issue_numbers)),
    }

def pair_has_name_similarity(left_activity, right_activity, config):
    left_names = set(left_activity.get("names", set()))
    right_names = set(right_activity.get("names", set()))
    if not left_names or not right_names:
        return False, 0.0

    min_name_similarity_score = float(get_fuzzy_option(config, "min_name_similarity_score", 0.94))
    best_similarity = 0.0
    for left_name in left_names:
        for right_name in right_names:
            best_similarity = max(best_similarity, jaro_winkler_similarity(left_name, right_name))

    return best_similarity >= min_name_similarity_score, best_similarity

def pair_has_email_localpart_similarity(left_activity, right_activity, config):
    if not bool(get_fuzzy_option(config, "allow_email_localpart_similarity_links", True)):
        return False, 0.0

    left_localparts = set(left_activity.get("email_localparts", set()))
    right_localparts = set(right_activity.get("email_localparts", set()))
    if not left_localparts or not right_localparts:
        return False, 0.0

    min_email_localpart_similarity_score = float(
        get_fuzzy_option(config, "min_email_localpart_similarity_score", 0.95)
    )
    best_similarity = 0.0
    for left_localpart in left_localparts:
        for right_localpart in right_localparts:
            best_similarity = max(best_similarity, jaro_winkler_similarity(left_localpart, right_localpart))

    return best_similarity >= min_email_localpart_similarity_score, best_similarity

def should_admit_discussion_bridge_candidate(left_activity, right_activity, pr_stats, discussion_stats, config):
    shared_issue_count = int(discussion_stats.get("shared_issue_count", 0))
    has_discussion_signal = bool(discussion_stats.get("overlap_logins")) or shared_issue_count >= 1

    if not has_discussion_signal:
        return False, None

    if shared_issue_count >= 3:
        return True, "discussion_three_plus_shared_issues"

    has_name_similarity, _ = pair_has_name_similarity(left_activity, right_activity, config)
    has_localpart_similarity, _ = pair_has_email_localpart_similarity(left_activity, right_activity, config)
    has_pr_bridge = bool(pr_stats.get("overlap_logins"))

    if shared_issue_count >= 2:
        if has_pr_bridge:
            return True, "discussion_two_shared_issues_plus_pr_bridge"
        if has_name_similarity:
            return True, "discussion_two_shared_issues_plus_name_similarity"
        if has_localpart_similarity:
            return True, "discussion_two_shared_issues_plus_email_localpart_similarity"
        return False, None

    if has_name_similarity:
        return True, "discussion_plus_name_similarity"

    if has_localpart_similarity:
        return True, "discussion_plus_email_localpart_similarity"

    if has_pr_bridge:
        return True, "discussion_plus_pr_bridge"

    return False, None

def build_candidate_pairs(identity_df, activity_sets, config, result):
    unique_keys = sorted([key for key in activity_sets.keys() if key])
    candidate_pairs = []
    seen_pairs = set()
    require_same_repo = bool(get_fuzzy_option(config, "require_same_repo", True))

    result["candidate_pairs_commit_discussion_scope"] = 0
    result["candidate_pairs_other_scope"] = 0
    result["candidate_pairs_from_anchor"] = 0
    result["candidate_pairs_from_name_similarity"] = 0
    result["candidate_pairs_from_email_localpart_similarity"] = 0
    result["candidate_pairs_from_pr_bridge"] = 0
    result["candidate_pairs_from_discussion_bridge"] = 0
    result["candidate_pairs_from_discussion_bridge_two_plus_shared_issues"] = 0
    result["candidate_pairs_from_discussion_bridge_plus_name_similarity"] = 0
    result["candidate_pairs_from_discussion_bridge_plus_email_localpart_similarity"] = 0
    result["candidate_pairs_from_discussion_bridge_plus_pr_bridge"] = 0

    key_to_repo = {}
    if not identity_df.empty and "resolved_contributor_key" in identity_df.columns and "repo_full_name" in identity_df.columns:
        key_repo_df = identity_df[["resolved_contributor_key", "repo_full_name"]].drop_duplicates().copy()
        key_to_repo = dict(zip(key_repo_df["resolved_contributor_key"], key_repo_df["repo_full_name"]))

    for i in range(len(unique_keys)):
        left_key = unique_keys[i]
        for j in range(i + 1, len(unique_keys)):
            right_key = unique_keys[j]
            if left_key == right_key:
                continue

            if require_same_repo and key_to_repo.get(left_key) != key_to_repo.get(right_key):
                continue

            pair_key = tuple(sorted([left_key, right_key]))
            if pair_key in seen_pairs:
                continue

            left_activity = activity_sets.get(left_key, {})
            right_activity = activity_sets.get(right_key, {})

            plausible = False
            plausible_reason_flags = {
                "anchor": 0,
                "name_similarity": 0,
                "email_localpart_similarity": 0,
                "pr_bridge": 0,
                "discussion_bridge": 0,
            }
            discussion_bridge_reason = None

            shared_emails = set(left_activity.get("emails", set())).intersection(set(right_activity.get("emails", set())))
            shared_logins = set(left_activity.get("logins", set())).intersection(set(right_activity.get("logins", set())))
            shared_noreply_left = set(left_activity.get("noreply_logins", set())).intersection(set(right_activity.get("logins", set())))
            shared_noreply_right = set(right_activity.get("noreply_logins", set())).intersection(set(left_activity.get("logins", set())))

            if shared_emails or shared_logins or shared_noreply_left or shared_noreply_right:
                plausible = True
                plausible_reason_flags["anchor"] = 1

            has_name_similarity = False
            if not plausible:
                has_name_similarity, _ = pair_has_name_similarity(left_activity, right_activity, config)
                if has_name_similarity:
                    plausible = True
                    plausible_reason_flags["name_similarity"] = 1

            has_localpart_similarity = False
            if not plausible:
                has_localpart_similarity, _ = pair_has_email_localpart_similarity(left_activity, right_activity, config)
                if has_localpart_similarity:
                    plausible = True
                    plausible_reason_flags["email_localpart_similarity"] = 1

            pr_stats = {"overlap_logins": [], "commit_hits": 0, "pr_hits": 0, "bijective": 0}
            discussion_stats = {"overlap_logins": [], "issue_hits": 0, "shared_issue_numbers": [], "shared_issue_count": 0}

            if bool(get_fuzzy_option(config, "allow_commit_discussion_fuzzy_links", True)):
                pr_stats = get_repeated_pr_bridge_stats(left_activity, right_activity)
                discussion_stats = get_repeated_discussion_bridge_stats(left_activity, right_activity)

                if pr_stats["overlap_logins"]:
                    plausible = True
                    plausible_reason_flags["pr_bridge"] = 1

                discussion_ok, discussion_bridge_reason = should_admit_discussion_bridge_candidate(
                    left_activity,
                    right_activity,
                    pr_stats,
                    discussion_stats,
                    config,
                )
                if discussion_ok:
                    plausible = True
                    plausible_reason_flags["discussion_bridge"] = 1

            if plausible:
                candidate_pairs.append((left_key, right_key))
                seen_pairs.add(pair_key)

                if should_consider_commit_discussion_pair(left_activity, right_activity):
                    result["candidate_pairs_commit_discussion_scope"] += 1
                else:
                    result["candidate_pairs_other_scope"] += 1

                result["candidate_pairs_from_anchor"] += plausible_reason_flags["anchor"]
                result["candidate_pairs_from_name_similarity"] += plausible_reason_flags["name_similarity"]
                result["candidate_pairs_from_email_localpart_similarity"] += plausible_reason_flags["email_localpart_similarity"]
                result["candidate_pairs_from_pr_bridge"] += plausible_reason_flags["pr_bridge"]
                result["candidate_pairs_from_discussion_bridge"] += plausible_reason_flags["discussion_bridge"]

                if discussion_bridge_reason == "discussion_three_plus_shared_issues":
                    result["candidate_pairs_from_discussion_bridge_three_plus_shared_issues"] += 1
                elif discussion_bridge_reason == "discussion_two_shared_issues_plus_name_similarity":
                    result["candidate_pairs_from_discussion_bridge_two_shared_issues_plus_name_similarity"] += 1
                elif discussion_bridge_reason == "discussion_two_shared_issues_plus_email_localpart_similarity":
                    result[
                        "candidate_pairs_from_discussion_bridge_two_shared_issues_plus_email_localpart_similarity"] += 1
                elif discussion_bridge_reason == "discussion_two_shared_issues_plus_pr_bridge":
                    result["candidate_pairs_from_discussion_bridge_two_shared_issues_plus_pr_bridge"] += 1
                elif discussion_bridge_reason == "discussion_plus_name_similarity":
                    result["candidate_pairs_from_discussion_bridge_plus_name_similarity"] += 1
                elif discussion_bridge_reason == "discussion_plus_email_localpart_similarity":
                    result["candidate_pairs_from_discussion_bridge_plus_email_localpart_similarity"] += 1
                elif discussion_bridge_reason == "discussion_plus_pr_bridge":
                    result["candidate_pairs_from_discussion_bridge_plus_pr_bridge"] += 1

    result["candidate_pair_count"] = int(len(candidate_pairs))
    return candidate_pairs

def score_identity_pair(left_key, right_key, activity_sets, config):
    left = activity_sets.get(left_key, {})
    right = activity_sets.get(right_key, {})

    component_scores = {
        "exact_email": 0.0,
        "exact_login": 0.0,
        "noreply_login": 0.0,
        "exact_name": 0.0,
        "name_similarity": 0.0,
        "email_localpart_similarity": 0.0,
        "pr_commit_bridge": 0.0,
        "discussion_bridge": 0.0,
        "repeated_pr_commit_bridge_bonus": 0.0,
        "repeated_discussion_bridge_bonus": 0.0,
        "cross_source_bonus": 0.0,
        "penalty_weak_only": 0.0,
    }
    signal_families = set()

    left_emails = set(left.get("emails", set()))
    right_emails = set(right.get("emails", set()))
    shared_emails = sorted(left_emails.intersection(right_emails))
    if shared_emails:
        component_scores["exact_email"] = 1.00
        signal_families.add("anchor")

    left_logins = set(left.get("logins", set()))
    right_logins = set(right.get("logins", set()))
    shared_logins = sorted(left_logins.intersection(right_logins))
    if shared_logins:
        component_scores["exact_login"] = 1.00
        signal_families.add("anchor")

    left_noreply = set(left.get("noreply_logins", set()))
    right_noreply = set(right.get("noreply_logins", set()))
    if left_noreply and right_logins and left_noreply.intersection(right_logins):
        component_scores["noreply_login"] = max(component_scores["noreply_login"], 0.90)
        signal_families.add("anchor")
    if right_noreply and left_logins and right_noreply.intersection(left_logins):
        component_scores["noreply_login"] = max(component_scores["noreply_login"], 0.90)
        signal_families.add("anchor")

    left_names = set(left.get("names", set()))
    right_names = set(right.get("names", set()))
    if left_names and right_names and left_names.intersection(right_names):
        component_scores["exact_name"] = 0.60
        signal_families.add("name")

    best_name_similarity = 0.0
    for left_name in left_names:
        for right_name in right_names:
            best_name_similarity = max(best_name_similarity, jaro_winkler_similarity(left_name, right_name))
    min_name_similarity_score = float(get_fuzzy_option(config, "min_name_similarity_score", 0.94))
    if best_name_similarity >= min_name_similarity_score:
        component_scores["name_similarity"] = 0.50 if best_name_similarity >= 0.98 else 0.35
        signal_families.add("name")

    best_localpart_similarity = 0.0
    if bool(get_fuzzy_option(config, "allow_email_localpart_similarity_links", True)):
        for left_localpart in set(left.get("email_localparts", set())):
            for right_localpart in set(right.get("email_localparts", set())):
                best_localpart_similarity = max(
                    best_localpart_similarity,
                    jaro_winkler_similarity(left_localpart, right_localpart),
                )
    min_email_localpart_similarity_score = float(
        get_fuzzy_option(config, "min_email_localpart_similarity_score", 0.95)
    )
    if best_localpart_similarity >= min_email_localpart_similarity_score:
        component_scores["email_localpart_similarity"] = 0.25
        signal_families.add("email_localpart")

    pr_stats = {
        "overlap_logins": [],
        "commit_hits": 0,
        "pr_hits": 0,
        "bijective": 0,
    }
    discussion_stats = {
        "overlap_logins": [],
        "issue_hits": 0,
        "shared_issue_numbers": [],
        "shared_issue_count": 0,
    }

    if bool(get_fuzzy_option(config, "allow_commit_discussion_fuzzy_links", True)):
        pr_stats = get_repeated_pr_bridge_stats(left, right)
        discussion_stats = get_repeated_discussion_bridge_stats(left, right)

        if pr_stats["overlap_logins"]:
            component_scores["pr_commit_bridge"] = 0.80
            signal_families.add("bridge")

        if discussion_stats["overlap_logins"] or discussion_stats["shared_issue_count"] >= 1:
            component_scores["discussion_bridge"] = 0.20
            signal_families.add("activity")

        if pr_stats["commit_hits"] >= 2:
            component_scores["repeated_pr_commit_bridge_bonus"] = max(
                component_scores["repeated_pr_commit_bridge_bonus"],
                0.15,
            )
        if pr_stats["pr_hits"] >= 2:
            component_scores["repeated_pr_commit_bridge_bonus"] = max(
                component_scores["repeated_pr_commit_bridge_bonus"],
                0.25,
            )
        if pr_stats["bijective"] == 1:
            component_scores["repeated_pr_commit_bridge_bonus"] = max(
                component_scores["repeated_pr_commit_bridge_bonus"],
                0.35,
            )
        if component_scores["repeated_pr_commit_bridge_bonus"] > 0.0:
            signal_families.add("repeated_bridge")

        if discussion_stats["shared_issue_count"] >= 2:
            component_scores["repeated_discussion_bridge_bonus"] = 0.10
            signal_families.add("repeated_activity")
        if discussion_stats["shared_issue_count"] >= 3:
            component_scores["repeated_discussion_bridge_bonus"] = 0.20
            signal_families.add("repeated_activity")

    if len(set(left.get("source_types", set())).union(set(right.get("source_types", set())))) >= 2:
        component_scores["cross_source_bonus"] = 0.10
        signal_families.add("cross_source")

    raw_score = float(sum(component_scores.values()))

    if (
        component_scores["exact_email"] == 0.0
        and component_scores["exact_login"] == 0.0
        and component_scores["noreply_login"] == 0.0
        and component_scores["pr_commit_bridge"] == 0.0
        and component_scores["discussion_bridge"] == 0.0
        and component_scores["repeated_pr_commit_bridge_bonus"] == 0.0
        and component_scores["repeated_discussion_bridge_bonus"] == 0.0
    ):
        if component_scores["name_similarity"] > 0.0 or component_scores["email_localpart_similarity"] > 0.0:
            component_scores["penalty_weak_only"] = -0.50
            raw_score += component_scores["penalty_weak_only"]

    return {
        "score": raw_score,
        "component_scores": component_scores,
        "signal_family_count": len(signal_families),
        "signal_families": sorted(signal_families),
        "best_name_similarity": best_name_similarity,
        "best_email_localpart_similarity": best_localpart_similarity,
        "shared_login_count": int(len(shared_logins)),
        "shared_email_count": int(len(shared_emails)),
        "bridge_pr_count": int(pr_stats["pr_hits"]),
        "bridge_issue_count": int(discussion_stats["shared_issue_count"]),
        "has_anchor": 1 if any(component_scores[key] > 0.0 for key in ["exact_email", "exact_login", "noreply_login"]) else 0,
        "has_structural": 1 if any(
            component_scores[key] > 0.0
            for key in [
                "pr_commit_bridge",
                "discussion_bridge",
                "repeated_pr_commit_bridge_bonus",
                "repeated_discussion_bridge_bonus",
            ]
        ) else 0,
    }

def accept_pair_for_fuzzy_merge(left_key, right_key, activity_sets, score_payload, config, result):
    left = activity_sets.get(left_key, {})
    right = activity_sets.get(right_key, {})

    if bool(get_fuzzy_option(config, "block_bot_merges", True)) and (left.get("is_bot_flag") or right.get("is_bot_flag")):
        result["blocked_bot_pairs"] += 1
        result["rejected_bot_pairs"] += 1
        return False, "rejected_bot", None

    min_pair_score_for_merge = float(get_fuzzy_option(config, "min_pair_score_for_merge", 0.90))
    min_pair_score_for_strong_merge = float(get_fuzzy_option(config, "min_pair_score_for_strong_merge", 1.20))
    require_multi_signal_for_non_exact_merge = bool(get_fuzzy_option(config, "require_multi_signal_for_non_exact_merge", True))
    require_structural_signal_for_name_based_merge = bool(get_fuzzy_option(config, "require_structural_signal_for_name_based_merge", True))

    score = float(score_payload["score"])
    component_scores = score_payload["component_scores"]
    has_anchor = bool(score_payload.get("has_anchor", 0))
    has_structural = bool(score_payload.get("has_structural", 0))
    name_based = component_scores["exact_name"] > 0.0 or component_scores["name_similarity"] > 0.0

    if require_structural_signal_for_name_based_merge and name_based and not has_anchor and not has_structural:
        result["rejected_insufficient_signal_pairs"] += 1
        result["rejected_name_without_structure_pairs"] += 1
        return False, "rejected_name_without_structure", None

    if require_multi_signal_for_non_exact_merge and not has_anchor and score_payload["signal_family_count"] < 2:
        result["rejected_insufficient_signal_pairs"] += 1
        return False, "rejected_insufficient_signal_families", None

    if score < min_pair_score_for_merge:
        result["rejected_low_score_pairs"] += 1
        return False, "rejected_low_score", None

    merge_tier = "strong" if score >= min_pair_score_for_strong_merge else "moderate"
    if merge_tier == "strong":
        result["strong_merge_edges_retained"] += 1
    else:
        result["moderate_merge_edges_retained"] += 1
    result["pair_edges_retained"] += 1
    return True, "accepted", merge_tier

def cluster_fuzzy_graph(identity_df, accepted_edges):
    contributor_keys = [value for value in identity_df.get("resolved_contributor_key", pd.Series(dtype="object")).dropna().unique().tolist()]
    if not contributor_keys:
        return {}

    union_find = UnionFind(contributor_keys)
    for edge in accepted_edges:
        union_find.union(edge["left_contributor_key"], edge["right_contributor_key"])

    clusters = {}
    for key in contributor_keys:
        root = union_find.find(key)
        clusters.setdefault(root, []).append(key)
    return clusters

def build_fuzzy_cluster_rows(repo_full_name, repo_id, clusters, identity_df, activity_sets, config, result):
    rows = []
    max_cluster_size_before_flag = int(get_fuzzy_option(config, "max_cluster_size_before_flag", 6))
    contributor_to_cluster_key = {}

    for cluster_index, contributor_keys in enumerate(sorted(clusters.values(), key=lambda values: sorted(values)), start=1):
        cluster_identity_rows = identity_df[identity_df["resolved_contributor_key"].isin(contributor_keys)].copy()
        cluster_size = int(len(contributor_keys))
        fuzzy_cluster_id = f"{sanitize_repo_name(repo_full_name)}::fuzzy_cluster::{cluster_index:05d}"
        canonical_repo_id = repo_id
        canonical_name = choose_canonical_value(cluster_identity_rows.get("raw_name", pd.Series(dtype="object")).tolist())
        canonical_email = choose_canonical_value(cluster_identity_rows.get("raw_email", pd.Series(dtype="object")).tolist())
        canonical_login = choose_canonical_value(cluster_identity_rows.get("raw_login", pd.Series(dtype="object")).tolist())
        source_types = sorted({clean_text(value) for value in cluster_identity_rows.get("raw_source_type", pd.Series(dtype="object")).tolist() if clean_text(value)})
        contains_commit_author = "commit_author" in source_types
        contains_issue_comment = "issue_comment" in source_types
        contains_pr_author = "pr_author" in source_types
        ambiguity_flag = 1 if cluster_size > max_cluster_size_before_flag else 0

        if cluster_size == 1:
            result["singleton_fuzzy_clusters"] += 1
        else:
            result["merged_fuzzy_clusters"] += 1
        if contains_commit_author:
            result["clusters_with_commit_author"] += 1
        if contains_commit_author and contains_issue_comment:
            result["clusters_with_issue_comment_and_commit"] += 1
        if contains_commit_author and contains_pr_author:
            result["clusters_with_pr_author_and_commit"] += 1
        if ambiguity_flag == 1:
            result["clusters_flagged_large"] += 1

        fuzzy_resolved_contributor_key = f"{canonical_repo_id}::fuzzy::{sanitize_repo_name(repo_full_name)}::{cluster_index:05d}"
        rows.append({
            "repo_id": canonical_repo_id,
            "repo_full_name": repo_full_name,
            "fuzzy_cluster_id": fuzzy_cluster_id,
            "resolved_contributor_key": fuzzy_resolved_contributor_key,
            "strict_contributor_keys_json": json.dumps(sorted(contributor_keys)),
            "cluster_size": cluster_size,
            "canonical_login": canonical_login,
            "canonical_name": canonical_name,
            "canonical_email": canonical_email,
            "source_types_json": json.dumps(source_types),
            "contains_commit_author": 1 if contains_commit_author else 0,
            "contains_issue_comment": 1 if contains_issue_comment else 0,
            "contains_pr_author": 1 if contains_pr_author else 0,
            "ambiguity_flag": ambiguity_flag,
        })

        for contributor_key in contributor_keys:
            contributor_to_cluster_key[contributor_key] = fuzzy_resolved_contributor_key

    return pd.DataFrame(rows), contributor_to_cluster_key

def build_fuzzy_identity_rows(identity_df, contributor_to_cluster_key, pair_edge_lookup, config):
    if identity_df.empty:
        return pd.DataFrame()

    rows = []
    for row in identity_df.to_dict(orient="records"):
        strict_key = row.get("resolved_contributor_key")
        fuzzy_key = contributor_to_cluster_key.get(strict_key, strict_key)
        edge_payloads = pair_edge_lookup.get(strict_key, [])
        best_score = max([payload["score"] for payload in edge_payloads], default=None)
        signal_summary = []
        for payload in edge_payloads:
            signal_summary.append({
                "other_strict_key": payload["other_strict_key"],
                "score": payload["score"],
                "merge_tier": payload["merge_tier"],
                "signal_families": payload["signal_families"],
            })
        out_row = dict(row)
        out_row["strict_resolved_contributor_key"] = strict_key
        out_row["resolved_contributor_key"] = fuzzy_key
        out_row["fuzzy_resolved_contributor_key"] = fuzzy_key
        out_row["fuzzy_resolution_method"] = "strict_only" if strict_key == fuzzy_key else "fuzzy_cluster_merge"
        out_row["fuzzy_score_to_cluster_anchor"] = best_score
        out_row["fuzzy_signal_summary_json"] = json.dumps(signal_summary)
        rows.append(out_row)
    return pd.DataFrame(rows)

def write_pair_audit_table(config, repo_full_name, audit_rows):
    if not bool(get_fuzzy_option(config, "write_pair_audit_table", True)):
        return
    output_dir = Path(config.logging.linkage_log_dir) / "fuzzy_identity_pair_audit"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{sanitize_repo_name(repo_full_name)}__{PAIR_AUDIT_FILENAME}"
    pd.DataFrame(audit_rows).to_csv(output_path, index=False)

def merge_identity_batches(config, logger):
    identity_output_path = getattr(config.outputs, "contributor_identity_table_fuzzy", None)
    if not identity_output_path:
        identity_output_path = str(Path(config.paths.linked_root) / "identity_resolution" / "contributor_identity_map_fuzzy.parquet")

    cluster_output_path = getattr(config.outputs, "contributor_identity_clusters_table_fuzzy", None)
    if not cluster_output_path:
        cluster_output_path = str(Path(config.paths.linked_root) / "identity_resolution" / "contributor_identity_clusters_fuzzy.parquet")

    batch_root = get_batch_root(config, BATCH_FOLDER_NAME)
    identity_repo_parts = collect_repo_part_files(batch_root, "contributor_identity_map_part_*.parquet")
    cluster_repo_parts = collect_repo_part_files(batch_root, "contributor_identity_clusters_part_*.parquet")

    if identity_repo_parts:
        mode_used = write_merged_or_partitioned_output(
            repo_part_map=identity_repo_parts,
            output_path=identity_output_path,
            config=config,
            table_name="contributor_identity_map",
            sort_columns=["repo_full_name", "resolved_contributor_key", "raw_source_type"],
            dedupe_subset=["repo_full_name", "resolved_contributor_key", "raw_source_type", "raw_login", "raw_name", "raw_email"],
        )
        logger.info("Wrote fuzzy contributor identity map using %s mode to %s", mode_used, identity_output_path)

    if cluster_repo_parts:
        mode_used = write_merged_or_partitioned_output(
            repo_part_map=cluster_repo_parts,
            output_path=cluster_output_path,
            config=config,
            table_name="contributor_identity_clusters",
            sort_columns=["repo_full_name", "resolved_contributor_key"],
            dedupe_subset=["repo_full_name", "resolved_contributor_key"],
        )
        logger.info("Wrote fuzzy contributor identity clusters using %s mode to %s", mode_used, cluster_output_path)

def write_summary_csv(summary_rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)

def write_run_manifest(config, repo_rows, summary_rows):
    manifest_path = Path(config.logging.linkage_log_dir) / RUN_MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "07_b_build_fuzzy_identity_map.py",
        "repo_count_requested": len(repo_rows),
        "repo_count_processed": len(summary_rows),
        "completed_repo_count": sum(1 for row in summary_rows if row.get("status") == "completed"),
        "failed_repo_count": sum(1 for row in summary_rows if row.get("status") == "failed"),
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "processed_merge_mode": getattr(config.storage, "processed_merge_mode", "single_parquet"),
        "summary_rows": summary_rows,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

def process_repo(config, logger, repo_row):
    repo_full_name = repo_row["full_name"]
    result = new_repo_result(repo_full_name, repo_row.get("repo_id"))
    stage_inputs = load_stage_inputs_for_repo(config, repo_full_name)
    strict_identity_df = build_strict_identity_candidates(repo_full_name, stage_inputs, result, config)
    if strict_identity_df.empty:
        result["status"] = "completed"
        return result

    strict_identity_df = add_commit_sha_column(strict_identity_df, stage_inputs, config)
    activity_sets = build_activity_sets_for_identity(strict_identity_df, build_commit_activity_lookup(stage_inputs, config))
    candidate_pairs = build_candidate_pairs(strict_identity_df, activity_sets, config, result)

    accepted_edges = []
    audit_rows = []
    pair_edge_lookup = {}
    for left_key, right_key in candidate_pairs:
        score_payload = score_identity_pair(left_key, right_key, activity_sets, config)
        accepted, decision_reason, merge_tier = accept_pair_for_fuzzy_merge(left_key, right_key, activity_sets, score_payload, config, result)
        left_activity = activity_sets.get(left_key, {})
        right_activity = activity_sets.get(right_key, {})
        pair_scope = "commit_discussion_or_pr" if should_consider_commit_discussion_pair(left_activity, right_activity) else "other"
        pr_stats_for_audit = get_repeated_pr_bridge_stats(left_activity, right_activity)
        discussion_stats_for_audit = get_repeated_discussion_bridge_stats(left_activity, right_activity)
        discussion_candidate_ok, discussion_candidate_reason = should_admit_discussion_bridge_candidate(
            left_activity,
            right_activity,
            pr_stats_for_audit,
            discussion_stats_for_audit,
            config,
        )
        min_pair_score_for_merge = float(get_fuzzy_option(config, "min_pair_score_for_merge", 0.90))
        audit_row = {
            "repo_full_name": repo_full_name,
            "left_contributor_key": left_key,
            "right_contributor_key": right_key,
            "pair_scope": pair_scope,
            "discussion_candidate_ok": 1 if discussion_candidate_ok else 0,
            "discussion_candidate_reason": discussion_candidate_reason,
            "score": score_payload["score"],
            "merge_tier": merge_tier,
            "accepted": 1 if accepted else 0,
            "decision_reason": decision_reason,
            "decision_bucket": decision_reason,
            "score_margin_to_merge_threshold": float(score_payload["score"]) - min_pair_score_for_merge,
            "has_anchor": int(score_payload.get("has_anchor", 0)),
            "has_structural": int(score_payload.get("has_structural", 0)),
            "bridge_pr_count": int(score_payload.get("bridge_pr_count", 0)),
            "bridge_issue_count": int(score_payload.get("bridge_issue_count", 0)),
            "shared_login_count": int(score_payload.get("shared_login_count", 0)),
            "shared_email_count": int(score_payload.get("shared_email_count", 0)),
            "signal_families_json": json.dumps(score_payload["signal_families"]),
            "component_scores_json": json.dumps(score_payload["component_scores"]),
            "best_name_similarity": score_payload["best_name_similarity"],
            "best_email_localpart_similarity": score_payload["best_email_localpart_similarity"],
        }
        audit_rows.append(audit_row)
        if accepted:
            accepted_edge = {
                "left_contributor_key": left_key,
                "right_contributor_key": right_key,
                "score": score_payload["score"],
                "merge_tier": merge_tier,
                "signal_families": score_payload["signal_families"],
            }
            accepted_edges.append(accepted_edge)
            pair_edge_lookup.setdefault(left_key, []).append({
                "other_strict_key": right_key,
                "score": score_payload["score"],
                "merge_tier": merge_tier,
                "signal_families": score_payload["signal_families"],
            })
            pair_edge_lookup.setdefault(right_key, []).append({
                "other_strict_key": left_key,
                "score": score_payload["score"],
                "merge_tier": merge_tier,
                "signal_families": score_payload["signal_families"],
            })

    write_pair_audit_table(config, repo_full_name, audit_rows)

    clusters = cluster_fuzzy_graph(strict_identity_df, accepted_edges)
    fuzzy_cluster_df, contributor_to_cluster_key = build_fuzzy_cluster_rows(
        repo_full_name,
        repo_row.get("repo_id"),
        clusters,
        strict_identity_df,
        activity_sets,
        config,
        result,
    )
    fuzzy_identity_df = build_fuzzy_identity_rows(strict_identity_df, contributor_to_cluster_key, pair_edge_lookup, config)

    batch_size = int(get_fuzzy_option(config, "write_batch_size", 5000))
    repo_dir = get_batch_root(config, BATCH_FOLDER_NAME) / sanitize_repo_name(repo_full_name)
    writer = IdentityResolutionRepoChunkWriter(config=config, repo_dir=repo_dir, batch_size=batch_size)

    for row in fuzzy_identity_df.to_dict(orient="records"):
        writer.add_identity_row(row)
        result["fuzzy_identity_rows_written"] += 1
    for row in fuzzy_cluster_df.to_dict(orient="records"):
        writer.add_cluster_row(row)
        result["fuzzy_clusters_written"] += 1
    writer.finalize()

    result["status"] = "completed"
    return result

def main():
    config = load_study_config(DEFAULT_CONFIG_PATH)
    ensure_project_directories(config)
    logger = setup_logger(config)

    fuzzy_cfg = getattr(config, "fuzzy_identity_resolution", None)
    if fuzzy_cfg is None or not getattr(fuzzy_cfg, "enabled", False):
        logger.warning("fuzzy_identity_resolution.enabled is false; nothing to do.")
        return

    if get_fuzzy_option(config, "resume_mode", "fresh") == "fresh":
        reset_batch_root(config, BATCH_FOLDER_NAME)

    repo_rows = load_repo_list(config.outputs.repo_included_list)
    max_repos = get_fuzzy_option(config, "max_repos_per_run", None)
    if max_repos:
        repo_rows = repo_rows[: int(max_repos)]

    summary_rows = []
    for repo_row in repo_rows:
        repo_full_name = repo_row["full_name"]
        skip_repo, reason = should_skip_repo(
            config,
            repo_full_name,
            checkpoint_prefix=CHECKPOINT_PREFIX,
            raw_folder_name=RAW_FOLDER_NAME,
            section_name="fuzzy_identity_resolution",
            raw_source="linked",
        )
        if skip_repo:
            logger.info("Skipping %s due to %s", repo_full_name, reason)
            summary_rows.append({
                "repo_full_name": repo_full_name,
                "repo_id": repo_row.get("repo_id"),
                "status": f"skipped_{reason}",
            })
            continue

        try:
            logger.info("Starting fuzzy identity resolution for %s", repo_full_name)
            result = process_repo(config, logger, repo_row)
            summary_rows.append(result)
            write_repo_checkpoint(config, CHECKPOINT_PREFIX, repo_full_name, result)
        except Exception as exc:
            logger.exception("Fuzzy identity resolution failed for %s", repo_full_name)
            error_row = new_repo_result(repo_full_name, repo_row.get("repo_id"))
            error_row["status"] = "failed"
            error_row["error_message"] = str(exc)
            summary_rows.append(error_row)
            write_repo_checkpoint(config, CHECKPOINT_PREFIX, repo_full_name, error_row)

    merge_identity_batches(config, logger)
    write_summary_csv(summary_rows, Path(config.logging.linkage_log_dir) / SUMMARY_FILENAME)
    write_run_manifest(config, repo_rows, summary_rows)
    logger.info("Fuzzy identity resolution complete.")


if __name__ == "__main__":
    main()
