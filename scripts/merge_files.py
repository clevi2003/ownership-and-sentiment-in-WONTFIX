#!/usr/bin/env python3
"""
Merge batch files from data/processed/_batches into data/processed/merged.

Expected layout:
    data/processed/_batches/<group>/<repo>/<file_prefix>_0001.parquet
    data/processed/_batches/<group>/<repo>/<file_prefix>_0002.parquet
    ...

Example:
    data/processed/_batches/commit_histories/repo_a/commit_files_0001.parquet
    data/processed/_batches/commit_histories/repo_a/commit_files_0002.parquet
    data/processed/_batches/commit_histories/repo_b/commit_files_0001.parquet

becomes:
    data/processed/merged/commit_files.parquet

Behavior:
- Recursively scans under data/processed/_batches
- Groups files by basename with trailing _NNNN removed
- Merges parquet files row-wise with pandas
- Writes outputs to data/processed/merged
- Does NOT delete or modify source files
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BATCHES_ROOT = PROJECT_ROOT / "data" / "processed" / "_batches"
MERGED_ROOT = PROJECT_ROOT / "data" / "processed" / "merged"

# Matches names like:
#   commit_files_0001.parquet -> commit_files.parquet
#   commits_0123.parquet      -> commits.parquet
BATCH_FILE_PATTERN = re.compile(
    r"^(?P<prefix>.+?)_part_\d+(?P<suffix>\.parquet)$"
)

def collect_parquet_groups(batches_root: Path) -> dict[str, list[Path]]:
    """
    Collect parquet files under _batches and group them by shared prefix,
    stripping the trailing batch number.
    """
    groups: dict[str, list[Path]] = defaultdict(list)

    for path in batches_root.rglob("*.parquet"):
        match = BATCH_FILE_PATTERN.match(path.name)
        if not match:
            # Skip files that do not follow the expected batch naming convention
            continue

        merged_name = f"{match.group('prefix')}{match.group('suffix')}"
        groups[merged_name].append(path)

    return dict(groups)


def merge_parquet_files(files: list[Path], output_path: Path) -> None:
    """
    Merge parquet files row-wise and write a single parquet file.
    """
    frames = []
    for path in sorted(files):
        df = pd.read_parquet(path)
        if not df.empty:
            frames.append(df)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)


def main() -> int:
    if not BATCHES_ROOT.exists():
        raise FileNotFoundError(f"_batches directory not found: {BATCHES_ROOT}")

    groups = collect_parquet_groups(BATCHES_ROOT)

    if not groups:
        print(f"No matching parquet batch files found under: {BATCHES_ROOT}")
        return 0

    MERGED_ROOT.mkdir(parents=True, exist_ok=True)

    written = 0
    for merged_name in sorted(groups):
        files = sorted(groups[merged_name])
        output_path = MERGED_ROOT / merged_name

        print(f"Merging {len(files)} files -> {output_path}")
        for file_path in files:
            print(f"  - {file_path}")

        merge_parquet_files(files, output_path)
        written += 1

    print(f"Done. Wrote {written} merged parquet file(s) to {MERGED_ROOT}")
    print("Source files in _batches were not deleted or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())