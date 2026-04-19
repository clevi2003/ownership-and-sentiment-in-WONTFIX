import time
import logging
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime


GITHUB_TOKEN = "token"
REPOS_TO_PROCESS = ["torvalds/linux", "pallets/flask"]
OUTPUT_DIR = Path("commit_data_output")
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("commit_extractor")

def get_github_headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

def make_request(url, params=None):
    headers = get_github_headers()
    for attempt in range(MAX_RETRIES):
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code in (403, 429) and 'X-RateLimit-Remaining' in response.headers:
            if int(response.headers['X-RateLimit-Remaining']) == 0:
                reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
                sleep_duration = max(reset_time - time.time(), 0) + 5
                logger.warning(f"Rate limit hit. Sleeping for {sleep_duration:.0f} seconds.")
                time.sleep(sleep_duration)
                continue
                
        response.raise_for_status()
        return response.json()
    raise Exception(f"Failed to fetch {url} after {MAX_RETRIES} attempts.")

def flatten_commit(detailed_commit_payload, repo_full_name):
    github_author = detailed_commit_payload.get("author") or {}
    git_commit_info = detailed_commit_payload.get("commit") or {}
    author_info = git_commit_info.get("author") or {}
    stats = detailed_commit_payload.get("stats") or {}
    files = detailed_commit_payload.get("files") or []

    return {
        "repo_full_name": repo_full_name,
        "commit_id": detailed_commit_payload.get("sha"),
        "author_login": github_author.get("login"),
        "author_id": github_author.get("id"),
        "timestamp": author_info.get("date"),
        "files_modified": len(files),
        "lines_added": stats.get("additions", 0),
        "lines_deleted": stats.get("deletions", 0),
        "total_changes": stats.get("total", 0),
    }

def extract_commits_for_repo(repo_full_name):
    logger.info(f"Starting extraction for {repo_full_name}")
    commits_data = []
    
    base_url = f"https://api.github.com/repos/{repo_full_name}/commits"
    page = 1
    per_page = 100 
    
    while True:
        logger.info(f"Fetching commit list page {page} for {repo_full_name}...")
        params = {"per_page": per_page, "page": page}
        
        try:
            commits_list = make_request(base_url, params=params)
        except Exception as e:
            logger.error(f"Error fetching commit list: {e}")
            break
            
        if not commits_list:
            break

        for short_commit in commits_list:
            commit_sha = short_commit.get("sha")
            detail_url = f"https://api.github.com/repos/{repo_full_name}/commits/{commit_sha}"
            
            try:
                detailed_payload = make_request(detail_url)
                flat_row = flatten_commit(detailed_payload, repo_full_name)
                commits_data.append(flat_row)
            except Exception as e:
                logger.error(f"Failed to fetch details for commit {commit_sha}: {e}")
                
        page += 1

    return commits_data

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_commits = []

    for repo in REPOS_TO_PROCESS:
        repo_commits = extract_commits_for_repo(repo)
        all_commits.extend(repo_commits)
        
        if repo_commits:
            df_repo = pd.DataFrame(repo_commits)
            safe_name = repo.replace("/", "__")
            out_path = OUTPUT_DIR / f"{safe_name}_commits.parquet"
            df_repo.to_parquet(out_path, index=False)
            logger.info(f"Saved {len(repo_commits)} commits to {out_path}")

    if all_commits:
        df_all = pd.DataFrame(all_commits)
        final_path = OUTPUT_DIR / "all_repositories_commits.csv"
        df_all.to_csv(final_path, index=False)
        logger.info(f"Extraction complete! Wrote {len(df_all)} total rows to {final_path}")
    else:
        logger.warning("No commits extracted.")

if __name__ == "__main__":
    main()
