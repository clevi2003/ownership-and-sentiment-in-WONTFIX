from pathlib import Path
import pandas as pd
import argparse


def parquet_to_csv(folder_path: Path, recursive: bool = False):
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"Skipping invalid folder: {folder_path}")
        return
    if recursive:
        parquet_files = list(folder_path.rglob("*.parquet"))
    else:
        parquet_files = list(folder_path.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {folder_path}")
        return

    for pq_file in parquet_files:
        csv_file = pq_file.with_suffix(".csv")
        print(f"Converting: {pq_file} -> {csv_file}")
        try:
            df = pd.read_parquet(pq_file)
            df.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"Failed to convert {pq_file}: {e}")
    print(f"Done with {folder_path}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Parquet files to CSV.")
    # nargs="+" for one or more folder paths
    parser.add_argument(
        "folder_paths",
        type=Path,
        nargs="+",
        help="One or more folders containing parquet files"
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search subfolders"
    )
    args = parser.parse_args()
    # loop over all provided directories
    for folder in args.folder_paths:
        parquet_to_csv(folder, recursive=args.recursive)