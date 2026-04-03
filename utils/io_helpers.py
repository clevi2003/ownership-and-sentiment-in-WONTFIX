import csv
import gzip
import json
from pathlib import Path
import pandas as pd
import math
from utils.checkpoints import sanitize_repo_name


def save_json(data, output_path, use_gzip=False, indent=2):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_gzip and output_path.suffix != ".gz":
        output_path = output_path.with_suffix(output_path.suffix + ".gz")

    if output_path.suffix == ".gz":
        with gzip.open(output_path, "wt", encoding="utf-8") as handle:
            json.dump(data, handle, indent=indent)
    else:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=indent)

    # return 1 so it can be used as a counter
    return 1


def append_jsonl_row(row, output_path, use_gzip=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_gzip and output_path.suffix != ".gz":
        output_path = output_path.with_suffix(output_path.suffix + ".gz")

    line = json.dumps(row, ensure_ascii=False) + "\n"

    if output_path.suffix == ".gz":
        with gzip.open(output_path, "at", encoding="utf-8") as handle:
            handle.write(line)
    else:
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    return 1


def reset_output_file(output_path, use_gzip=False):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_gzip and output_path.suffix != ".gz":
        output_path = output_path.with_suffix(output_path.suffix + ".gz")

    if output_path.exists():
        output_path.unlink()

    return output_path


def load_repo_list(repo_list_path):
    repo_list_path = Path(repo_list_path)
    if not repo_list_path.exists():
        raise FileNotFoundError(f"Repo list does not exist: {repo_list_path}")

    repo_df = pd.read_csv(repo_list_path)
    if repo_df.empty:
        return []

    required_columns = {"full_name"}
    missing = required_columns - set(repo_df.columns)
    if missing:
        raise ValueError(
            "Repo list is missing required columns: " + ", ".join(sorted(missing))
        )

    return repo_df.to_dict(orient="records")


def write_processed_table(df, output_path, config):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if config.storage.processed_format == "parquet":
        df.to_parquet(output_path,
                      index=False,
                      compression=config.storage.compression.parquet_compression)
        return output_path

    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def write_csv_rows(rows, output_path, fieldnames=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(rows)
    if not rows:
        if fieldnames is None:
            fieldnames = []
    elif fieldnames is None:
        fieldnames = list(rows[0].keys())

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)

    return output_path

def get_partitioned_output_root(output_path):
    output_path = Path(output_path)
    return output_path.with_suffix("").with_name(output_path.stem + "_dataset")


def read_parquet_if_exists(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def read_repo_partitioned_dataset(output_path, repo_full_name):
    dataset_root = get_partitioned_output_root(output_path)
    partition_dir = dataset_root / f"repo_full_name={sanitize_repo_name(repo_full_name)}"

    if not partition_dir.exists():
        return pd.DataFrame()

    part_paths = sorted(partition_dir.glob("*.parquet"))
    if not part_paths:
        return pd.DataFrame()

    frames = []
    for path in part_paths:
        df = pd.read_parquet(path)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def load_table(path, repo_full_name=None, merge_mode=None):
    if merge_mode == "partitioned_dataset" and repo_full_name is not None:
        return read_repo_partitioned_dataset(path, repo_full_name)
    return read_parquet_if_exists(path)


def repo_filter(df, repo_full_name):
    if df.empty:
        return df

    if "repo_full_name" in df.columns:
        return df[df["repo_full_name"] == repo_full_name].copy()

    if "full_name" in df.columns:
        return df[df["full_name"] == repo_full_name].copy()

    return df.iloc[0:0].copy()


def merge_part_files(part_paths, sort_columns=None):
    frames = []
    for path in sorted(part_paths):
        df = pd.read_parquet(path)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)

    if sort_columns:
        existing = [col for col in sort_columns if col in merged.columns]
        if existing:
            merged = merged.sort_values(existing, kind="stable").reset_index(drop=True)

    return merged


def collect_repo_part_files(batch_root, part_glob):
    batch_root = Path(batch_root)
    repo_part_map = {}

    if not batch_root.exists():
        return repo_part_map

    for repo_dir in sorted(batch_root.iterdir()):
        if not repo_dir.is_dir():
            continue

        # repo_dir.name is already sanitized because _batches uses sanitized repo dirs
        repo_key = repo_dir.name
        part_paths = sorted(repo_dir.glob(part_glob))
        if part_paths:
            repo_part_map[repo_key] = part_paths

    return repo_part_map


def write_partitioned_dataset_from_repo_parts(
    *,
    repo_part_map,
    output_path,
    config,
    table_name,
    sort_columns=None,
    dedupe_subset=None,
):
    dataset_root = get_partitioned_output_root(output_path)
    dataset_root.mkdir(parents=True, exist_ok=True)

    for repo_key, part_paths in sorted(repo_part_map.items()):
        repo_df = merge_part_files(part_paths, sort_columns=sort_columns)
        if repo_df.empty:
            continue

        if dedupe_subset:
            existing_subset = [col for col in dedupe_subset if col in repo_df.columns]
            if existing_subset:
                repo_df = repo_df.drop_duplicates(subset=existing_subset).reset_index(drop=True)

        partition_dir = dataset_root / f"repo_full_name={repo_key}"
        partition_dir.mkdir(parents=True, exist_ok=True)

        output_file = partition_dir / f"{table_name}_part_00001.parquet"
        repo_df.to_parquet(
            output_file,
            index=False,
            compression=config.storage.compression.parquet_compression,
        )


def write_merged_or_partitioned_output(
    *,
    repo_part_map,
    output_path,
    config,
    table_name,
    sort_columns=None,
    dedupe_subset=None,
):
    merge_mode = getattr(config.storage, "processed_merge_mode", "single_parquet")

    if merge_mode == "partitioned_dataset":
        write_partitioned_dataset_from_repo_parts(
            repo_part_map=repo_part_map,
            output_path=output_path,
            config=config,
            table_name=table_name,
            sort_columns=sort_columns,
            dedupe_subset=dedupe_subset,
        )
        return "partitioned_dataset"

    all_part_paths = [path for paths in repo_part_map.values() for path in paths]
    merged_df = merge_part_files(all_part_paths, sort_columns=sort_columns)

    if merged_df.empty:
        return "single_parquet_empty"

    if dedupe_subset:
        existing_subset = [col for col in dedupe_subset if col in merged_df.columns]
        if existing_subset:
            merged_df = merged_df.drop_duplicates(subset=existing_subset).reset_index(drop=True)

    write_processed_table(merged_df, Path(output_path), config)
    return "single_parquet"

def clean_text(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def normalize_value(value):
    value = clean_text(value)
    if not value:
        return None
    return value.lower()

def has_real_value(value):
    cleaned = clean_text(value)
    if cleaned is None:
        return False
    lowered = cleaned.lower()
    if lowered in {"nan", "none", "null", "nat", "<na>"}:
        return False
    return True

def write_summary_csv(summary_rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)

def safe_to_datetime(value):
    if value is None:
        return pd.NaT
    try:
        return pd.to_datetime(value, errors="coerce", utc=True)
    except Exception:
        # preserve shape for common container inputs if conversion blows up
        if isinstance(value, pd.Series):
            return pd.Series(pd.NaT, index=value.index, dtype="datetime64[ns, UTC]")
        try:
            length = len(value)
            return pd.Series([pd.NaT] * length, dtype="datetime64[ns, UTC]")
        except Exception:
            return pd.NaT

def safe_divide(numerator, denominator, default_value=0.0):
    if denominator in {0, 0.0, None}:
        return default_value
    try:
        return float(numerator) / float(denominator)
    except Exception:
        return default_value

def take_mean(values):
    numeric_values = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not numeric_values:
        return 0.0
    return float(sum(numeric_values)) / float(len(numeric_values))


def take_std(values):
    numeric_values = [float(value) for value in values if value is not None and not pd.isna(value)]
    if len(numeric_values) < 2:
        return 0.0
    mean_value = take_mean(numeric_values)
    variance = sum((value - mean_value) ** 2 for value in numeric_values) / float(len(numeric_values) - 1)
    return math.sqrt(max(variance, 0.0))


def take_median(values):
    numeric_values = sorted(float(value) for value in values if value is not None and not pd.isna(value))
    if not numeric_values:
        return 0.0
    count = len(numeric_values)
    midpoint = count // 2
    if count % 2 == 1:
        return numeric_values[midpoint]
    return (numeric_values[midpoint - 1] + numeric_values[midpoint]) / 2.0