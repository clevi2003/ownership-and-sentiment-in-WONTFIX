import sys
from pathlib import Path
import pandas as pd

def parquet_to_csv(folder_path: Path):
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Invalid folder: {folder_path}")

    parquet_files = list(folder_path.glob("*.parquet"))

    if not parquet_files:
        print("No parquet files found.")
        return

    for pq_file in parquet_files:
        csv_file = pq_file.with_suffix(".csv")
        print(f"Converting: {pq_file.name} -> {csv_file.name}")

        try:
            df = pd.read_parquet(pq_file)
            df.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"Failed to convert {pq_file.name}: {e}")

    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parquet_to_csv.py <folder_path>")
        sys.exit(1)

    folder = Path(sys.argv[1])
    parquet_to_csv(folder)