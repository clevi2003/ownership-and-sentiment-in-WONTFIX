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
from utils.checkpoints import get_batch_root, get_stage_option, reset_batch_root, sanitize_repo_name, should_skip_repo, write_repo_checkpoint
from utils.io_helpers import collect_repo_part_files, load_repo_list, load_table, repo_filter, write_merged_or_partitioned_output
from utils.sentiment_utils import (
    SentimentAnalyzerWrapper,
    SentimentFeatureRepoChunkWriter,
    clean_text,
    compute_series_slope,
    safe_divide,
    safe_to_datetime,
    score_text_features,
    split_early_late,
    take_mean,
    take_median,
    take_std,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "study_config.yaml"
LOG_FILENAME = "08_build_sentiment_features.log"
CHECKPOINT_PREFIX = "08_build_sentiment_features"
BATCH_FOLDER_NAME = "sentiment_features"
RAW_FOLDER_NAME = "sentiment_features"


def setup_logger(config):
    logger = logging.getLogger("build_sentiment_features")
    logger.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if config.logging.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.logging.log_to_file:
        log_dir = Path(config.logging.qa_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / LOG_FILENAME, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_sentiment_option(config, field_name, default_value):
    return get_stage_option(config, "sentiment_features", field_name, default_value)


def get_stage_paths(config):
    outputs = getattr(config, "outputs", None)

    issue_output_path = getattr(outputs, "issue_sentiment_features_table", None)
    if not issue_output_path:
        issue_output_path = "./data/features/sentiment/issue_sentiment_features.parquet"

    comment_output_path = getattr(outputs, "comment_sentiment_features_table", None)
    if not comment_output_path:
        comment_output_path = "./data/features/sentiment/comment_sentiment_features.parquet"

    qa_summary_path = getattr(outputs, "sentiment_feature_qa_summary_csv", None)
    if not qa_summary_path:
        qa_summary_path = "./logs/qa/sentiment_feature_qa_summary.csv"

    return {
        "issue_output_path": Path(issue_output_path),
        "comment_output_path": Path(comment_output_path),
        "qa_summary_path": Path(qa_summary_path),
        "run_manifest_path": Path(config.logging.qa_log_dir) / "08_build_sentiment_features_run_manifest.json",
    }


def normalize_issue_set_columns(df):
    if df.empty:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number", "analysis_set"])

    normalized = df.copy()
    repo_col = find_first_present_column(normalized, ["repo_full_name", "repo_name", "full_name", "repo"])
    issue_id_col = find_first_present_column(normalized, ["issue_id", "id"])
    issue_number_col = find_first_present_column(normalized, ["issue_number", "number"])

    if repo_col is None:
        return pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number", "analysis_set"])

    out_df = pd.DataFrame()
    out_df["repo_full_name"] = normalized[repo_col].astype(str)
    out_df["issue_id"] = normalized[issue_id_col].astype(str) if issue_id_col else None
    out_df["issue_number"] = pd.to_numeric(normalized[issue_number_col], errors="coerce") if issue_number_col else None

    if "analysis_set" in normalized.columns:
        out_df["analysis_set"] = normalized["analysis_set"].astype(str)
    else:
        out_df["analysis_set"] = None

    out_df = out_df.drop_duplicates().reset_index(drop=True)
    return out_df


def find_first_present_column(df, candidates):
    lower_map = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def build_target_issue_lookup(config):
    merge_mode = getattr(config.storage, "processed_merge_mode", "single_parquet")
    wontfix_df = load_table(config.outputs.wontfix_issue_set_table, merge_mode=merge_mode)
    comparison_df = load_table(config.outputs.comparison_issue_set_table, merge_mode=merge_mode)

    wontfix_df = normalize_issue_set_columns(wontfix_df)
    comparison_df = normalize_issue_set_columns(comparison_df)

    if not wontfix_df.empty:
        wontfix_df["analysis_set"] = "wontfix"
    if not comparison_df.empty:
        comparison_df["analysis_set"] = "comparison"

    combined = pd.concat([wontfix_df, comparison_df], ignore_index=True) if (not wontfix_df.empty or not comparison_df.empty) else pd.DataFrame(columns=["repo_full_name", "issue_id", "issue_number", "analysis_set"])
    if combined.empty:
        return {}

    combined = combined.sort_values(["repo_full_name", "analysis_set"], kind="stable").reset_index(drop=True)
    lookup = {}
    for row in combined.to_dict(orient="records"):
        repo_full_name = row.get("repo_full_name")
        if not repo_full_name:
            continue
        repo_payload = lookup.setdefault(repo_full_name, {"by_issue_id": {}, "by_issue_number": {}})
        issue_id = clean_text(row.get("issue_id"))
        issue_number = row.get("issue_number")
        analysis_set = row.get("analysis_set")

        if issue_id:
            repo_payload["by_issue_id"][issue_id] = analysis_set
        if pd.notna(issue_number):
            repo_payload["by_issue_number"][int(issue_number)] = analysis_set

    return lookup


def load_stage_inputs_for_repo(config, repo_full_name):
    merge_mode = getattr(config.storage, "processed_merge_mode", "single_parquet")

    issues_df = load_table(config.outputs.issues_resolved_table, repo_full_name=repo_full_name, merge_mode=merge_mode)
    comments_df = load_table(config.outputs.issue_comments_resolved_table, repo_full_name=repo_full_name, merge_mode=merge_mode)

    return {
        "issues_resolved": repo_filter(issues_df, repo_full_name),
        "issue_comments_resolved": repo_filter(comments_df, repo_full_name),
    }


def new_repo_result(repo_full_name, repo_id=None):
    return {
        "repo_full_name": repo_full_name,
        "repo_id": repo_id,
        "status": "started",
        "target_issues_requested": 0,
        "issues_resolved_rows_seen": 0,
        "issue_comments_resolved_rows_seen": 0,
        "target_issues_kept": 0,
        "target_comments_kept": 0,
        "issues_with_zero_comments": 0,
        "issues_with_missing_issue_text": 0,
        "comments_with_missing_text": 0,
        "comment_rows_written": 0,
        "issue_rows_written": 0,
        "sentiment_backend": None,
        "error_message": "",
    }


def attach_analysis_set(issues_df, repo_lookup):
    if issues_df.empty:
        return issues_df

    df = issues_df.copy()
    df["analysis_set"] = None

    for index, row in df.iterrows():
        issue_id = clean_text(row.get("issue_id"))
        issue_number = row.get("issue_number")
        analysis_set = None

        if issue_id and issue_id in repo_lookup.get("by_issue_id", {}):
            analysis_set = repo_lookup["by_issue_id"].get(issue_id)
        elif pd.notna(issue_number):
            analysis_set = repo_lookup.get("by_issue_number", {}).get(int(issue_number))

        df.at[index, "analysis_set"] = analysis_set

    df = df[df["analysis_set"].notna()].copy()
    if df.empty:
        return df

    return df.reset_index(drop=True)


def prepare_issue_frame(issues_df):
    if issues_df.empty:
        return issues_df

    df = issues_df.copy()
    needed_columns = [
        "repo_id",
        "repo_full_name",
        "issue_id",
        "issue_number",
        "state",
        "created_at",
        "closed_at",
        "title",
        "body",
        "author_login",
        "issue_author_contributor_key",
        "analysis_set",
    ]
    existing_columns = [column for column in needed_columns if column in df.columns]
    df = df[existing_columns].copy()
    if "issue_number" in df.columns:
        df["issue_number"] = pd.to_numeric(df["issue_number"], errors="coerce")
    df["created_at"] = safe_to_datetime(df.get("created_at"))
    df["closed_at"] = safe_to_datetime(df.get("closed_at"))
    return df


def prepare_comment_frame(comments_df, target_issue_numbers):
    if comments_df.empty:
        return comments_df

    df = comments_df.copy()
    needed_columns = [
        "repo_full_name",
        "issue_number",
        "comment_id",
        "author_login",
        "comment_author_contributor_key",
        "created_at",
        "body",
    ]
    existing_columns = [column for column in needed_columns if column in df.columns]
    df = df[existing_columns].copy()
    if "issue_number" in df.columns:
        df["issue_number"] = pd.to_numeric(df["issue_number"], errors="coerce")
        df = df[df["issue_number"].notna()].copy()
        df["issue_number"] = df["issue_number"].astype(int)
    df = df[df["issue_number"].isin(target_issue_numbers)].copy()
    if "created_at" in df.columns:
        df["created_at"] = safe_to_datetime(df["created_at"])
    return df.reset_index(drop=True)


def build_comment_rows_for_issue(issue_row, issue_comments_df, analyzer):
    if issue_comments_df.empty:
        return []

    rows = []
    sorted_comments = issue_comments_df.sort_values(["created_at", "comment_id"], kind="stable", na_position="last").reset_index(drop=True)

    for sequence_index, comment_row in enumerate(sorted_comments.to_dict(orient="records"), start=1):
        score_payload = score_text_features(comment_row.get("body"), analyzer)
        author_key = clean_text(comment_row.get("comment_author_contributor_key")) or clean_text(comment_row.get("author_login"))
        compound = float(score_payload["sentiment_compound"])

        rows.append({
            "repo_id": issue_row.get("repo_id"),
            "repo_full_name": issue_row.get("repo_full_name"),
            "issue_id": issue_row.get("issue_id"),
            "issue_number": issue_row.get("issue_number"),
            "comment_id": comment_row.get("comment_id"),
            "analysis_set": issue_row.get("analysis_set"),
            "created_at": comment_row.get("created_at"),
            "comment_sequence_index": sequence_index,
            "comment_author_login": comment_row.get("author_login"),
            "comment_author_contributor_key": comment_row.get("comment_author_contributor_key"),
            "effective_commenter_key": author_key,
            "has_text": score_payload["has_text"],
            "text_length_chars": score_payload["text_length_chars"],
            "token_count": score_payload["token_count"],
            "question_mark_count": score_payload["question_mark_count"],
            "exclamation_mark_count": score_payload["exclamation_mark_count"],
            "uppercase_ratio": score_payload["uppercase_ratio"],
            "has_code_block": score_payload["has_code_block"],
            "has_url": score_payload["has_url"],
            "has_path_reference": score_payload["has_path_reference"],
            "sentiment_neg": score_payload["sentiment_neg"],
            "sentiment_neu": score_payload["sentiment_neu"],
            "sentiment_pos": score_payload["sentiment_pos"],
            "sentiment_compound": compound,
            "is_positive_comment": 1 if compound > 0.05 else 0,
            "is_negative_comment": 1 if compound < -0.05 else 0,
            "is_neutral_comment": 1 if -0.05 <= compound <= 0.05 else 0,
        })

    return rows


def build_issue_feature_row(issue_row, comment_rows, analyzer):
    title_scores = score_text_features(issue_row.get("title"), analyzer)
    body_scores = score_text_features(issue_row.get("body"), analyzer)
    combined_scores = score_text_features(
        "\n\n".join([value for value in [clean_text(issue_row.get("title")), clean_text(issue_row.get("body"))] if value]),
        analyzer,
    )

    issue_author_key = clean_text(issue_row.get("issue_author_contributor_key")) or clean_text(issue_row.get("author_login"))

    total_comments = len(comment_rows)
    comments_with_text = [row for row in comment_rows if row.get("has_text") == 1]
    comment_compounds = [row.get("sentiment_compound") for row in comments_with_text]
    early_values, late_values = split_early_late(comment_compounds)

    commenter_counts = {}
    for row in comment_rows:
        commenter_key = clean_text(row.get("effective_commenter_key"))
        if not commenter_key:
            continue
        commenter_counts[commenter_key] = commenter_counts.get(commenter_key, 0) + 1

    unique_commenter_count = len(commenter_counts)
    top_commenter_share = safe_divide(max(commenter_counts.values()) if commenter_counts else 0, total_comments, default_value=0.0)
    concentration_ratio = 0.0
    if total_comments > 0:
        concentration_ratio = sum((count / float(total_comments)) ** 2 for count in commenter_counts.values())

    issue_author_commented_flag = 1 if issue_author_key and issue_author_key in commenter_counts else 0
    num_non_author_commenters = len([key for key in commenter_counts if key != issue_author_key])

    missing_comment_text_count = sum(1 for row in comment_rows if row.get("has_text") != 1)
    positive_comment_count = sum(row.get("is_positive_comment", 0) for row in comment_rows)
    negative_comment_count = sum(row.get("is_negative_comment", 0) for row in comment_rows)
    neutral_comment_count = sum(row.get("is_neutral_comment", 0) for row in comment_rows)

    text_lengths = [row.get("text_length_chars") for row in comments_with_text]
    uppercase_ratios = [row.get("uppercase_ratio") for row in comments_with_text]
    question_counts = [row.get("question_mark_count") for row in comments_with_text]
    exclamation_counts = [row.get("exclamation_mark_count") for row in comments_with_text]

    return {
        "repo_id": issue_row.get("repo_id"),
        "repo_full_name": issue_row.get("repo_full_name"),
        "issue_id": issue_row.get("issue_id"),
        "issue_number": issue_row.get("issue_number"),
        "analysis_set": issue_row.get("analysis_set"),
        "state": issue_row.get("state"),
        "created_at": issue_row.get("created_at"),
        "closed_at": issue_row.get("closed_at"),
        "issue_author_login": issue_row.get("author_login"),
        "issue_author_contributor_key": issue_row.get("issue_author_contributor_key"),
        "issue_title_has_text": title_scores["has_text"],
        "issue_title_length_chars": title_scores["text_length_chars"],
        "issue_title_sentiment_neg": title_scores["sentiment_neg"],
        "issue_title_sentiment_neu": title_scores["sentiment_neu"],
        "issue_title_sentiment_pos": title_scores["sentiment_pos"],
        "issue_title_sentiment_compound": title_scores["sentiment_compound"],
        "issue_body_has_text": body_scores["has_text"],
        "issue_body_length_chars": body_scores["text_length_chars"],
        "issue_body_sentiment_neg": body_scores["sentiment_neg"],
        "issue_body_sentiment_neu": body_scores["sentiment_neu"],
        "issue_body_sentiment_pos": body_scores["sentiment_pos"],
        "issue_body_sentiment_compound": body_scores["sentiment_compound"],
        "issue_text_sentiment_compound": combined_scores["sentiment_compound"],
        "comment_count": total_comments,
        "comments_with_text_count": len(comments_with_text),
        "missing_comment_text_count": missing_comment_text_count,
        "zero_comment_flag": 1 if total_comments == 0 else 0,
        "unique_commenter_count": unique_commenter_count,
        "issue_author_commented_flag": issue_author_commented_flag,
        "num_distinct_non_author_commenters": num_non_author_commenters,
        "top_commenter_share": top_commenter_share,
        "comment_concentration_ratio": concentration_ratio,
        "positive_comment_count": positive_comment_count,
        "negative_comment_count": negative_comment_count,
        "neutral_comment_count": neutral_comment_count,
        "positive_comment_share": safe_divide(positive_comment_count, total_comments, default_value=0.0),
        "negative_comment_share": safe_divide(negative_comment_count, total_comments, default_value=0.0),
        "neutral_comment_share": safe_divide(neutral_comment_count, total_comments, default_value=0.0),
        "mean_comment_sentiment": take_mean(comment_compounds),
        "median_comment_sentiment": take_median(comment_compounds),
        "min_comment_sentiment": min(comment_compounds) if comment_compounds else 0.0,
        "max_comment_sentiment": max(comment_compounds) if comment_compounds else 0.0,
        "std_comment_sentiment": take_std(comment_compounds),
        "early_mean_comment_sentiment": take_mean(early_values),
        "late_mean_comment_sentiment": take_mean(late_values),
        "comment_sentiment_change_late_minus_early": take_mean(late_values) - take_mean(early_values),
        "comment_sentiment_slope": compute_series_slope(comment_compounds),
        "mean_comment_length_chars": take_mean(text_lengths),
        "max_comment_length_chars": max(text_lengths) if text_lengths else 0,
        "total_comment_text_chars": int(sum(text_lengths)) if text_lengths else 0,
        "mean_comment_uppercase_ratio": take_mean(uppercase_ratios),
        "total_question_mark_count": int(sum(question_counts)) if question_counts else 0,
        "total_exclamation_mark_count": int(sum(exclamation_counts)) if exclamation_counts else 0,
        "comments_with_code_block_count": int(sum(row.get("has_code_block", 0) for row in comment_rows)),
        "comments_with_url_count": int(sum(row.get("has_url", 0) for row in comment_rows)),
        "comments_with_path_reference_count": int(sum(row.get("has_path_reference", 0) for row in comment_rows)),
    }


def process_repo(config, logger, repo_row, target_issue_lookup, analyzer):
    repo_full_name = repo_row["full_name"]
    repo_lookup = target_issue_lookup.get(repo_full_name)
    result = new_repo_result(repo_full_name, repo_row.get("repo_id"))
    result["sentiment_backend"] = analyzer.backend_name

    if not repo_lookup:
        result["status"] = "skipped_no_target_issues"
        return result

    requested_issue_count = len(repo_lookup.get("by_issue_id", {})) + len(repo_lookup.get("by_issue_number", {}))
    result["target_issues_requested"] = requested_issue_count

    stage_inputs = load_stage_inputs_for_repo(config, repo_full_name)
    issues_df = stage_inputs["issues_resolved"]
    comments_df = stage_inputs["issue_comments_resolved"]
    result["issues_resolved_rows_seen"] = len(issues_df)
    result["issue_comments_resolved_rows_seen"] = len(comments_df)

    if issues_df.empty:
        result["status"] = "completed"
        return result

    issues_df = attach_analysis_set(issues_df, repo_lookup)
    issues_df = prepare_issue_frame(issues_df)
    if issues_df.empty:
        result["status"] = "completed"
        return result

    result["target_issues_kept"] = len(issues_df)
    target_issue_numbers = set(issues_df["issue_number"].dropna().astype(int).tolist()) if "issue_number" in issues_df.columns else set()
    comments_df = prepare_comment_frame(comments_df, target_issue_numbers)
    result["target_comments_kept"] = len(comments_df)

    batch_size = get_sentiment_option(config, "write_batch_size", 5000)
    repo_dir = get_batch_root(config, BATCH_FOLDER_NAME) / sanitize_repo_name(repo_full_name)
    writer = SentimentFeatureRepoChunkWriter(config=config, repo_dir=repo_dir, batch_size=batch_size)

    comments_by_issue_number = {}
    if not comments_df.empty:
        for row in comments_df.to_dict(orient="records"):
            issue_number = row.get("issue_number")
            if pd.isna(issue_number):
                continue
            comments_by_issue_number.setdefault(int(issue_number), []).append(row)

    for issue_row in issues_df.to_dict(orient="records"):
        issue_title = clean_text(issue_row.get("title"))
        issue_body = clean_text(issue_row.get("body"))
        if not issue_title and not issue_body:
            result["issues_with_missing_issue_text"] += 1

        issue_number = int(issue_row.get("issue_number")) if pd.notna(issue_row.get("issue_number")) else None
        issue_comments = comments_by_issue_number.get(issue_number, [])
        comment_rows = build_comment_rows_for_issue(issue_row, pd.DataFrame(issue_comments), analyzer)
        if not comment_rows:
            result["issues_with_zero_comments"] += 1
        result["comments_with_missing_text"] += sum(1 for row in comment_rows if row.get("has_text") != 1)

        for comment_feature_row in comment_rows:
            writer.add_comment_row(comment_feature_row)
            result["comment_rows_written"] += 1

        issue_feature_row = build_issue_feature_row(issue_row, comment_rows, analyzer)
        writer.add_issue_row(issue_feature_row)
        result["issue_rows_written"] += 1

    writer.finalize()
    result["status"] = "completed"
    return result


def merge_sentiment_feature_batches(config, logger, stage_paths):
    batch_root = get_batch_root(config, BATCH_FOLDER_NAME)
    if not batch_root.exists():
        logger.warning("Sentiment feature batch root does not exist: %s", batch_root)
        return

    comment_repo_parts = collect_repo_part_files(batch_root, "comment_sentiment_features_part_*.parquet")
    issue_repo_parts = collect_repo_part_files(batch_root, "issue_sentiment_features_part_*.parquet")

    if comment_repo_parts:
        mode_used = write_merged_or_partitioned_output(
            repo_part_map=comment_repo_parts,
            output_path=stage_paths["comment_output_path"],
            config=config,
            table_name="comment_sentiment_features",
            sort_columns=["repo_full_name", "issue_number", "comment_sequence_index", "comment_id"],
            dedupe_subset=["repo_full_name", "comment_id"],
        )
        logger.info("Wrote comment sentiment features using %s mode to %s", mode_used, stage_paths["comment_output_path"])
    else:
        logger.warning("No comment sentiment feature parts found to merge.")

    if issue_repo_parts:
        mode_used = write_merged_or_partitioned_output(
            repo_part_map=issue_repo_parts,
            output_path=stage_paths["issue_output_path"],
            config=config,
            table_name="issue_sentiment_features",
            sort_columns=["repo_full_name", "issue_number"],
            dedupe_subset=["repo_full_name", "issue_id", "issue_number"],
        )
        logger.info("Wrote issue sentiment features using %s mode to %s", mode_used, stage_paths["issue_output_path"])
    else:
        logger.warning("No issue sentiment feature parts found to merge.")


def write_summary_csv(summary_rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)


def write_run_manifest(repo_rows, summary_rows, stage_paths, analyzer_backend):
    manifest_path = Path(stage_paths["run_manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": "build_sentiment_features.py",
        "repo_count_requested": len(repo_rows),
        "repo_count_processed": len(summary_rows),
        "completed_repo_count": sum(1 for row in summary_rows if row.get("status") == "completed"),
        "failed_repo_count": sum(1 for row in summary_rows if row.get("status") == "failed"),
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sentiment_backend": analyzer_backend,
        "summary_rows": summary_rows,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main(config_path=None):
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = load_study_config(config_path)
    ensure_project_directories(config)
    logger = setup_logger(config)
    stage_paths = get_stage_paths(config)

    repo_rows = load_repo_list(config.outputs.repo_included_list)
    target_issue_lookup = build_target_issue_lookup(config)
    analyzer = SentimentAnalyzerWrapper()
    logger.info("Using sentiment backend: %s", analyzer.backend_name)

    batch_root = reset_batch_root(config, BATCH_FOLDER_NAME)
    logger.info("Reset batch root: %s", batch_root)

    max_repos_per_run = get_sentiment_option(config, "max_repos_per_run", None)
    if max_repos_per_run and max_repos_per_run > 0:
        repo_rows = repo_rows[:max_repos_per_run]

    summary_rows = []

    for repo_row in repo_rows:
        repo_full_name = repo_row["full_name"]
        should_skip, reason = should_skip_repo(
            config,
            repo_full_name,
            checkpoint_prefix=CHECKPOINT_PREFIX,
            raw_folder_name=RAW_FOLDER_NAME,
            section_name="sentiment_features",
            raw_source="features",
        )
        if should_skip:
            logger.info("Skipping repo %s due to %s.", repo_full_name, reason)
            summary_rows.append({
                "repo_full_name": repo_full_name,
                "repo_id": repo_row.get("repo_id"),
                "status": f"skipped_{reason}",
            })
            continue

        logger.info("Processing repo %s", repo_full_name)
        try:
            result = process_repo(config, logger, repo_row, target_issue_lookup, analyzer)
        except Exception as exc:
            logger.exception("Failed while building sentiment features for %s", repo_full_name)
            result = new_repo_result(repo_full_name, repo_row.get("repo_id"))
            result["status"] = "failed"
            result["error_message"] = str(exc)

        write_repo_checkpoint(config, CHECKPOINT_PREFIX, repo_full_name, result)
        summary_rows.append(result)

    merge_sentiment_feature_batches(config, logger, stage_paths)
    write_summary_csv(summary_rows, stage_paths["qa_summary_path"])
    write_run_manifest(repo_rows, summary_rows, stage_paths, analyzer.backend_name)
    logger.info("Sentiment feature building complete. Repos processed: %s", len(summary_rows))


if __name__ == "__main__":
    main()
