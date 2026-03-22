import json
import shutil
from pathlib import Path


def get_stage_option(config, section_name, field_name, default_value):
    if not hasattr(config, section_name):
        return default_value

    section = getattr(config, section_name)
    if not hasattr(section, field_name):
        return default_value

    value = getattr(section, field_name)
    if value is None:
        return default_value

    return value


def sanitize_repo_name(repo_full_name):
    return str(repo_full_name).replace("/", "__")


def get_batch_root(config, batch_folder_name):
    return Path(config.paths.processed_root) / "_batches" / batch_folder_name


def reset_batch_root(config, batch_folder_name):
    batch_root = get_batch_root(config, batch_folder_name)
    if batch_root.exists():
        shutil.rmtree(batch_root)
    batch_root.mkdir(parents=True, exist_ok=True)
    return batch_root


def get_repo_output_root(config, raw_folder_name, repo_full_name, raw_source="github_api"):
    safe_repo_name = sanitize_repo_name(repo_full_name)
    if raw_source == "github_api":
        return Path(config.paths.raw_root) / "github_api" / raw_folder_name / safe_repo_name
    if raw_source == "git_logs":
        return Path(config.paths.raw_root) / "git_logs" / safe_repo_name
    return Path(config.paths.raw_root) / raw_source / raw_folder_name / safe_repo_name


def get_checkpoint_path(config, checkpoint_prefix, repo_full_name):
    safe_repo_name = sanitize_repo_name(repo_full_name)
    return Path(config.checkpointing.checkpoint_dir) / f"{checkpoint_prefix}__{safe_repo_name}.json"


def read_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    with checkpoint_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_repo_checkpoint(config, checkpoint_prefix, repo_full_name, payload):
    if not config.checkpointing.enabled:
        return None

    checkpoint_path = get_checkpoint_path(config, checkpoint_prefix, repo_full_name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with checkpoint_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return checkpoint_path


def should_skip_repo(config,
                     repo_full_name,
                     *,
                     checkpoint_prefix,
                     raw_folder_name,
                     section_name="issue_extraction",
                     raw_source="github_api"):
    resume_mode = get_stage_option(config, section_name, "resume_mode", "checkpoint_only")

    if resume_mode in {"checkpoint_only", "raw_or_checkpoint"}:
        if config.checkpointing.enabled and config.checkpointing.resume_from_checkpoints:
            checkpoint_path = get_checkpoint_path(config, checkpoint_prefix, repo_full_name)
            payload = read_checkpoint(checkpoint_path)
            if payload and payload.get("status") == "completed":
                return True, "completed_checkpoint"

    if resume_mode == "raw_or_checkpoint":
        raw_root = get_repo_output_root(config, raw_folder_name, repo_full_name, raw_source=raw_source)
        if raw_root.exists() and get_stage_option(config, section_name, "skip_repo_if_raw_exists", False):
            return True, "existing_raw_output"

    return False, ""