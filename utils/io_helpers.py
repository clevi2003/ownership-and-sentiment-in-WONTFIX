import csv
import gzip
import json
from pathlib import Path

import pandas as pd


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