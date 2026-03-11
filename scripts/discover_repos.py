import csv
import gzip
import json
import logging
import os
import time
from datetime import datetime
import pandas as pd
import requests

import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config.study_config_loader import load_study_config, ensure_project_directories, ConfigError


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"


def setup_logger(config):
    logger = logging.getLogger("discover_repos")
    logger.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if config.logging.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.logging.log_to_file:
        log_dir = Path(config.logging.extraction_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "01_discover_repos.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_github_headers(config):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": config.github.requests.user_agent,
    }

    if config.github.auth.use_token:
        token = os.getenv(config.github.auth.token_env_var)
        if not token:
            raise ConfigError(
                f"GitHub token environment variable '{config.github.auth.token_env_var}' is not set."
            )
        headers["Authorization"] = f"Bearer {token}"

    return headers


def save_raw_json(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if str(output_path).endswith(".gz"):
        with gzip.open(output_path, "wt", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    else:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def normalize_label_text(label, config):
    if label is None:
        return ""

    text = str(label)

    if config.label_normalization.normalize_unicode_quotes:
        text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')

    if config.label_normalization.normalize_hyphens_and_apostrophes:
        text = text.replace("‐", "-").replace("–", "-").replace("—", "-")

    if config.label_normalization.strip_whitespace:
        text = text.strip()

    if not config.label_normalization.case_sensitive:
        text = text.lower()

    return text


def get_normalized_wontfix_variants(config):
    variants = config.label_normalization.outcome_labels.wontfix.variants
    normalized = []
    seen = set()

    for variant in variants:
        clean = normalize_label_text(variant, config)
        if clean and clean not in seen:
            normalized.append(clean)
            seen.add(clean)

    return normalized


def repo_passes_allow_block_rules(repo_full_name, owner_login, config):
    filters = config.repo_selection.discovery_filters

    if owner_login in filters.owners_blocklist:
        return False

    if repo_full_name in filters.repos_blocklist:
        return False

    if filters.owners_allowlist and owner_login not in filters.owners_allowlist:
        return False

    if filters.repos_allowlist and repo_full_name not in filters.repos_allowlist:
        return False

    return True


def make_request(session, url, headers, params, config, logger):
    retries = 0

    while True:
        response = session.get(
            url,
            headers=headers,
            params=params,
            timeout=config.github.requests.timeout_seconds,
        )

        remaining = response.headers.get("X-RateLimit-Remaining")
        reset_time = response.headers.get("X-RateLimit-Reset")

        if response.status_code == 403 and remaining == "0":
            if config.github.rate_limit.respect_reset_header and reset_time:
                sleep_seconds = max(int(reset_time) - int(time.time()) + 5, 5)
            else:
                sleep_seconds = config.github.rate_limit.default_pause_seconds

            logger.warning(
                "Rate limit reached. Sleeping for %s seconds before retrying.",
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
            continue

        if response.status_code in (500, 502, 503, 504):
            retries += 1
            if retries > config.github.rate_limit.max_retries:
                response.raise_for_status()

            sleep_seconds = config.github.rate_limit.retry_backoff_seconds * retries
            logger.warning(
                "GitHub server error %s. Retry %s/%s after %s seconds.",
                response.status_code,
                retries,
                config.github.rate_limit.max_retries,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
            continue

        if response.status_code == 403:
            retries += 1
            if retries > config.github.rate_limit.max_retries:
                response.raise_for_status()

            sleep_seconds = config.github.rate_limit.retry_backoff_seconds * retries
            logger.warning(
                "Received 403 response. Retry %s/%s after %s seconds.",
                retries,
                config.github.rate_limit.max_retries,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
            continue

        response.raise_for_status()

        if remaining is not None:
            try:
                remaining_int = int(remaining)
                if remaining_int <= config.github.rate_limit.min_remaining_before_pause:
                    logger.info(
                        "Approaching rate limit (remaining=%s). Pausing for %s seconds.",
                        remaining_int,
                        config.github.rate_limit.default_pause_seconds,
                    )
                    time.sleep(config.github.rate_limit.default_pause_seconds)
            except ValueError:
                pass

        logger.info(
            "Rate limit headers | resource=%s | limit=%s | remaining=%s | reset=%s",
            response.headers.get("X-RateLimit-Resource"),
            response.headers.get("X-RateLimit-Limit"),
            response.headers.get("X-RateLimit-Remaining"),
            response.headers.get("X-RateLimit-Reset"),
        )
        return response


def flatten_repo_record(repo):
    owner = repo.get("owner") or {}

    return {
        "repo_id": repo.get("id"),
        "repo_name": repo.get("name"),
        "full_name": repo.get("full_name"),
        "owner_login": owner.get("login"),
        "html_url": repo.get("html_url"),
        "description": repo.get("description"),
        "language": repo.get("language"),
        "stargazers_count": repo.get("stargazers_count"),
        "fork": repo.get("fork"),
        "archived": repo.get("archived"),
        "disabled": repo.get("disabled"),
        "is_template": repo.get("is_template"),
        "created_at": repo.get("created_at"),
        "updated_at": repo.get("updated_at"),
        "pushed_at": repo.get("pushed_at"),
        "open_issues_count": repo.get("open_issues_count"),
        "size": repo.get("size"),
        "default_branch": repo.get("default_branch"),
        "visibility": repo.get("visibility"),
        "topics": json.dumps(repo.get("topics", [])),
    }


def build_repo_search_query(config, page):
    parts = []

    parts.append(f"stars:>={config.repo_selection.inclusion_criteria.min_stars}")

    if config.repo_selection.inclusion_criteria.exclude_forks:
        parts.append("fork:false")

    if config.repo_selection.inclusion_criteria.exclude_archived:
        parts.append("archived:false")

    if config.repo_selection.inclusion_criteria.require_recent_activity:
        cutoff = config.repo_selection.activity_definition.cutoff_date
        activity_field = config.repo_selection.activity_definition.field

        if activity_field == "pushed_at":
            parts.append(f"pushed:>={cutoff}")
        else:
            parts.append(f"pushed:>={cutoff}")

    discovery_filters = config.repo_selection.discovery_filters
    for language in discovery_filters.languages:
        parts.append(f"language:{language}")

    query = " ".join(parts)

    return {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": config.github.pagination.per_page,
        "page": page,
    }


def apply_row_limit(rows, limit_value, logger, label):
    if limit_value is None:
        return rows

    if limit_value <= 0:
        return rows

    if len(rows) <= limit_value:
        return rows

    logger.info(
        "Applying %s limit: keeping first %s of %s rows.",
        label,
        limit_value,
        len(rows),
    )
    return rows[:limit_value]


def discover_candidate_repos(config, logger):
    logger.info("Phase 1: Discovering structurally eligible candidate repositories.")

    raw_repo_dir = Path(config.paths.raw_root) / "github_api" / "repo_search"
    raw_repo_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    headers = get_github_headers(config)

    all_rows = []
    seen_repo_ids = set()

    url = f"{config.github.api_base_url}/search/repositories"

    page = 1
    max_pages = config.repo_discovery.max_repo_search_pages

    while True:
        params = build_repo_search_query(config, page)
        logger.info("Fetching candidate repository search page %s.", page)

        response = make_request(session, url, headers, params, config, logger)
        payload = response.json()

        save_raw_json(payload, raw_repo_dir / f"repo_search_page_{page:03d}.json.gz")

        items = payload.get("items", [])
        if not items:
            break

        for repo in items:
            repo_id = repo.get("id")
            if repo_id in seen_repo_ids:
                continue

            row = flatten_repo_record(repo)

            if not repo_passes_allow_block_rules(row["full_name"], row["owner_login"], config):
                continue

            row["meets_structural_filters"] = True
            all_rows.append(row)
            seen_repo_ids.add(repo_id)

            # early stopping if we have enough repos
            if (
                    config.repo_discovery.candidate_repo_limit
                    and len(all_rows) >= config.repo_discovery.candidate_repo_limit
            ):
                logger.info(
                    "Reached candidate_repo_limit (%s). Stopping candidate discovery early.",
                    config.repo_discovery.candidate_repo_limit,
                )
                return all_rows

        logger.info("Collected %s structurally eligible candidate repos so far.", len(all_rows))

        if len(items) < config.github.pagination.per_page:
            break

        page += 1
        if page > max_pages:
            logger.warning(
                "Reached repository search page cap (%s). Stop here for now.",
                max_pages,
            )
            break

    all_rows = apply_row_limit(
        all_rows,
        config.repo_discovery.candidate_repo_limit,
        logger,
        "candidate_repo_limit",
    )

    logger.info("Candidate repo discovery complete. Total candidates kept: %s", len(all_rows))
    return all_rows


def build_year_ranges(start_date, end_date):
    start_year = datetime.fromisoformat(start_date).year
    end_year = datetime.fromisoformat(end_date).year

    ranges = []
    for year in range(start_year, end_year + 1):
        ranges.append(
            {
                "year": year,
                "start": f"{year}-01-01",
                "end": f"{year}-12-31",
            }
        )

    return ranges


def build_search_ranges(config):
    issue_window = config.study_windows.issue_collection

    if config.repo_discovery.split_wontfix_issue_search_by_year:
        return build_year_ranges(issue_window.start_date, issue_window.end_date)

    return [
        {
            "year": "all",
            "start": issue_window.start_date,
            "end": issue_window.end_date,
        }
    ]


def extract_repo_full_name_from_issue_item(issue_item):
    repo_url = issue_item.get("repository_url")
    if not repo_url:
        return None

    parts = repo_url.rstrip("/").split("/")
    if len(parts) < 2:
        return None

    owner = parts[-2]
    repo = parts[-1]
    return f"{owner}/{repo}"


def search_wontfix_repos_globally(config, logger):
    logger.info("Phase 2: Discovering repos with WONTFIX issues globally.")

    raw_issue_dir = Path(config.paths.raw_root) / "github_api" / "global_wontfix_issue_search"
    raw_issue_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    headers = get_github_headers(config)
    url = f"{config.github.api_base_url}/search/issues"

    search_ranges = build_search_ranges(config)
    wontfix_variants = get_normalized_wontfix_variants(config)

    repos_with_wontfix = {}
    query_run_rows = []

    max_pages_per_query = config.repo_discovery.max_issue_search_pages_per_query

    for variant in wontfix_variants:
        for date_range in search_ranges:
            page = 1

            while True:
                query = (
                    f'is:issue archived:false label:"{variant}" '
                    f'created:{date_range["start"]}..{date_range["end"]}'
                )

                params = {
                    "q": query,
                    "sort": "created",
                    "order": "desc",
                    "per_page": config.github.pagination.per_page,
                    "page": page,
                }

                logger.info(
                    "Searching WONTFIX issues | variant='%s' | range=%s | page=%s",
                    variant,
                    date_range["year"],
                    page,
                )

                response = make_request(session, url, headers, params, config, logger)
                payload = response.json()

                raw_name = (
                    f'wontfix_search_{variant.replace(" ", "_").replace("/", "_")}_'
                    f'{date_range["year"]}_page_{page:03d}.json.gz'
                )
                save_raw_json(payload, raw_issue_dir / raw_name)

                items = payload.get("items", [])
                total_count = payload.get("total_count", 0)

                query_run_rows.append(
                    {
                        "label_variant": variant,
                        "range_label": date_range["year"],
                        "start_date": date_range["start"],
                        "end_date": date_range["end"],
                        "page": page,
                        "returned_items": len(items),
                        "reported_total_count": total_count,
                    }
                )

                if not items:
                    break

                for item in items:
                    repo_full_name = extract_repo_full_name_from_issue_item(item)
                    if not repo_full_name:
                        continue

                    if repo_full_name not in repos_with_wontfix:
                        repos_with_wontfix[repo_full_name] = {
                            "full_name": repo_full_name,
                            "matched_variants": set(),
                            "matched_years": set(),
                            "example_issue_urls": set(),
                            "issue_hit_count_observed": 0,
                        }

                    repos_with_wontfix[repo_full_name]["matched_variants"].add(variant)
                    repos_with_wontfix[repo_full_name]["matched_years"].add(str(date_range["year"]))

                    html_url = item.get("html_url")
                    if html_url:
                        repos_with_wontfix[repo_full_name]["example_issue_urls"].add(html_url)

                    repos_with_wontfix[repo_full_name]["issue_hit_count_observed"] += 1

                if len(items) < config.github.pagination.per_page:
                    break

                page += 1
                if page > max_pages_per_query:
                    logger.warning(
                        "Reached page cap for issue query | variant='%s' | range=%s",
                        variant,
                        date_range["year"],
                    )
                    break

    output_rows = []
    for repo_full_name, info in repos_with_wontfix.items():
        output_rows.append(
            {
                "full_name": repo_full_name,
                "matched_variants": json.dumps(sorted(info["matched_variants"])),
                "matched_years": json.dumps(sorted(info["matched_years"])),
                "example_issue_urls": json.dumps(sorted(list(info["example_issue_urls"]))[:5]),
                "issue_hit_count_observed": info["issue_hit_count_observed"],
            }
        )

    logger.info("Global WONTFIX repo discovery complete. Unique repos found: %s", len(output_rows))
    return output_rows, query_run_rows


def intersect_candidate_and_wontfix_repos(candidate_rows, wontfix_rows, config, logger):
    logger.info("Phase 3: Intersecting structural candidates with WONTFIX repos.")

    candidate_map = {}
    for row in candidate_rows:
        candidate_map[row["full_name"]] = row

    wontfix_map = {}
    for row in wontfix_rows:
        wontfix_map[row["full_name"]] = row

    final_rows = []

    for repo_full_name, candidate in candidate_map.items():
        if repo_full_name not in wontfix_map:
            continue

        merged = dict(candidate)
        merged["has_wontfix_issue"] = True
        merged["matched_wontfix_variants"] = wontfix_map[repo_full_name]["matched_variants"]
        merged["matched_wontfix_years"] = wontfix_map[repo_full_name]["matched_years"]
        merged["example_issue_urls"] = wontfix_map[repo_full_name]["example_issue_urls"]
        merged["issue_hit_count_observed"] = wontfix_map[repo_full_name]["issue_hit_count_observed"]
        merged["screening_status"] = "included_candidate"

        final_rows.append(merged)

    final_rows = apply_row_limit(
        final_rows,
        config.repo_discovery.final_repo_limit,
        logger,
        "final_repo_limit",
    )

    logger.info(
        "Intersection complete. Structural candidates=%s | WONTFIX repos=%s | Final included=%s",
        len(candidate_rows),
        len(wontfix_rows),
        len(final_rows),
    )

    return final_rows


def write_outputs(config, candidate_rows, wontfix_rows, final_rows, query_run_rows, logger):
    processed_root = Path(config.paths.processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    candidate_output = Path(config.outputs.repo_candidate_list)
    included_output = Path(config.outputs.repo_included_list)
    wontfix_repo_output = processed_root / "repos_with_wontfix_issue.csv"
    issue_query_summary_output = processed_root / "wontfix_issue_search_runs.csv"

    candidate_df = pd.DataFrame(candidate_rows)
    wontfix_df = pd.DataFrame(wontfix_rows)
    final_df = pd.DataFrame(final_rows)
    query_df = pd.DataFrame(query_run_rows)

    candidate_df.to_csv(candidate_output, index=False)
    wontfix_df.to_csv(wontfix_repo_output, index=False)
    final_df.to_csv(included_output, index=False)
    query_df.to_csv(issue_query_summary_output, index=False)

    logger.info("Wrote candidate repos to %s", candidate_output)
    logger.info("Wrote repos with WONTFIX issues to %s", wontfix_repo_output)
    logger.info("Wrote included repo list to %s", included_output)
    logger.info("Wrote WONTFIX issue search run summary to %s", issue_query_summary_output)


def write_summary_report(config, candidate_rows, wontfix_rows, final_rows, logger):
    summary_path = Path(config.logging.qa_log_dir) / "repo_discovery_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {"metric": "candidate_repo_rows", "value": len(candidate_rows)},
        {"metric": "repos_with_wontfix_issue", "value": len(wontfix_rows)},
        {"metric": "included_repo_rows", "value": len(final_rows)},
    ]

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote repo discovery summary report to %s", summary_path)


def main():
    config = load_study_config(DEFAULT_CONFIG_PATH)
    ensure_project_directories(config)
    logger = setup_logger(config)

    logger.info("Loaded config from %s", DEFAULT_CONFIG_PATH)

    candidate_rows = discover_candidate_repos(config, logger)
    wontfix_rows, query_run_rows = search_wontfix_repos_globally(config, logger)
    final_rows = intersect_candidate_and_wontfix_repos(candidate_rows, wontfix_rows, config, logger)
    write_outputs(config, candidate_rows, wontfix_rows, final_rows, query_run_rows, logger)
    write_summary_report(config, candidate_rows, wontfix_rows, final_rows, logger)

    logger.info("Repository discovery pipeline complete.")


if __name__ == "__main__":
    main()