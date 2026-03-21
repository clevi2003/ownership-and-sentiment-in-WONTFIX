import os
import time
from datetime import datetime, timezone
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.study_config_loader import ConfigError


DEFAULT_ACCEPT_HEADER = "application/vnd.github+json"


def build_session(config):
    session = requests.Session()

    retry = Retry(total=config.github.rate_limit.max_retries,
                  read=config.github.rate_limit.max_retries,
                  connect=config.github.rate_limit.max_retries,
                  status=config.github.rate_limit.max_retries,
                  backoff_factor=max(config.github.rate_limit.retry_backoff_seconds / 10.0, 0.1),
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"],
                  raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_github_headers(config, accept=DEFAULT_ACCEPT_HEADER, extra_headers=None):

    headers = {"Accept": accept, "User-Agent": config.github.requests.user_agent,}

    if config.github.auth.use_token:
        token = os.getenv(config.github.auth.token_env_var)
        if not token:
            raise ConfigError(
                f"GitHub token environment variable '{config.github.auth.token_env_var}' is not set."
            )
        headers["Authorization"] = f"Bearer {token}"

    if extra_headers:
        headers.update(extra_headers)

    return headers


def make_request(session,
                 url,
                 headers,
                 params,
                 config,
                 logger,
                 method="GET",
                 json_body=None,
                 data=None,):
    """
    Make a git api request with timeout from config, retry/backoff for transient/server/rate-limit responses,
    optional pause when rate limit remaining gets low
    """
    retries = 0
    method = method.upper()

    while True:
        response = session.request(method=method,
                                   url=url,
                                   headers=headers,
                                   params=params,
                                   json=json_body,
                                   data=data,
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
                "Rate limit reached for %s %s. Sleeping for %s seconds before retrying.",
                method,
                url,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)
            continue

        if response.status_code in (403, 429, 500, 502, 503, 504):
            retries += 1
            if retries > config.github.rate_limit.max_retries:
                response.raise_for_status()

            sleep_seconds = config.github.rate_limit.retry_backoff_seconds * retries
            logger.warning(
                "GitHub response %s for %s %s. Retry %s/%s after %s seconds.",
                response.status_code,
                method,
                url,
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

        logger.debug(
            "Rate limit headers | resource=%s | limit=%s | remaining=%s | reset=%s",
            response.headers.get("X-RateLimit-Resource"),
            response.headers.get("X-RateLimit-Limit"),
            response.headers.get("X-RateLimit-Remaining"),
            response.headers.get("X-RateLimit-Reset"),
        )
        return response


def fetch_repository_metadata(session, headers, config, logger, repo_full_name):
    url = f"{config.github.api_base_url}/repos/{repo_full_name}"
    response = make_request(session, url, headers, None, config, logger)
    return response.json()


def parse_github_datetime(value):
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def parse_link_header(link_header):
    if not link_header:
        return {}

    links = {}
    parts = [part.strip() for part in link_header.split(",") if part.strip()]
    for part in parts:
        if ";" not in part:
            continue

        url_part, rel_part = [x.strip() for x in part.split(";", 1)]
        if not (url_part.startswith("<") and url_part.endswith(">")):
            continue

        url = url_part[1:-1]
        rel_tokens = [token.strip() for token in rel_part.split(";")]
        rel_value = None

        for token in rel_tokens:
            if token.startswith("rel="):
                rel_value = token.split("=", 1)[1].strip('"')
                break

        if rel_value:
            links[rel_value] = url

    return links
