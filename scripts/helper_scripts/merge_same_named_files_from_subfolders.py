#!/usr/bin/env python3
"""
Merge same-named files found across immediate subfolders of a root directory.

Behavior:
- Scans every immediate subfolder under a given root directory.
- Groups files by relative path within each subfolder.
- For every relative path found in any subfolder, writes one combined file
  at that same relative path under the root directory.
- Example:
    root/
      run_a/results/issues.csv
      run_b/results/issues.csv
    -> writes:
      root/results/issues.csv

Default merge behavior is byte concatenation with a separator for text files,
and raw byte concatenation for binary files. This makes the script generic.

Optional special handling:
- --parquet: merge parquet files row-wise with pandas
- --csv: merge csv files row-wise with pandas
- --jsonl: merge jsonl files by line concatenation

Notes:
- Existing output files in the root are overwritten unless --skip-existing is set.
- Files already located directly in the root directory are ignored as inputs.
- By default, only immediate subfolders are scanned.
"""

from __future__ import annotations

import argparse
import mimetypes
from collections import defaultdict
from pathlib import Path
from typing import Iterable


TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".tsv", ".jsonl", ".log", ".yaml", ".yml",
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".xml",
    ".html", ".css", ".sql", ".sh", ".bat", ".ps1", ".r", ".go", ".rs", ".parquet"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine same-named files from subfolders into merged files written to the root directory."
        )
    )
    parser.add_argument(
        "root_dir",
        help="Root directory whose immediate subfolders will be scanned.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include nested files inside each subfolder and preserve their relative paths.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "text", "binary", "csv", "jsonl", "parquet"],
        default="auto",
        help=(
            "Merge mode. 'auto' uses parquet/csv/jsonl handlers by extension when possible, "
            "otherwise text for likely text files and binary for everything else."
        ),
    )
    parser.add_argument(
        "--separator",
        default="\n",
        help="Separator inserted between concatenated text file contents in text mode.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing an output file if it already exists in the root.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden subfolders and files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress.",
    )
    return parser.parse_args()


def is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def iter_subfolders(root_dir: Path, include_hidden: bool) -> Iterable[Path]:
    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        if not include_hidden and child.name.startswith("."):
            continue
        yield child


def iter_files_in_subfolder(subfolder: Path, recursive: bool, include_hidden: bool) -> Iterable[Path]:
    iterator = subfolder.rglob("*") if recursive else subfolder.glob("*")
    for path in sorted(iterator):
        if not path.is_file():
            continue
        if not include_hidden and is_hidden(path.relative_to(subfolder)):
            continue
        yield path


def collect_groups(root_dir: Path, recursive: bool, include_hidden: bool) -> dict[Path, list[Path]]:
    groups: dict[Path, list[Path]] = defaultdict(list)

    for subfolder in iter_subfolders(root_dir, include_hidden=include_hidden):
        for file_path in iter_files_in_subfolder(subfolder, recursive=recursive, include_hidden=include_hidden):
            rel_path = file_path.relative_to(subfolder)
            groups[rel_path].append(file_path)

    return dict(groups)


def looks_like_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    mime_type, _ = mimetypes.guess_type(str(path))
    return bool(mime_type and mime_type.startswith("text/"))


def detect_mode(rel_path: Path, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode

    suffix = rel_path.suffix.lower()
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".csv":
        return "csv"
    if suffix == ".jsonl":
        return "jsonl"
    if looks_like_text(rel_path):
        return "text"
    return "binary"


def merge_text(files: list[Path], output_path: Path, separator: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as out:
        for index, path in enumerate(files):
            content = path.read_text(encoding="utf-8")
            if index > 0:
                out.write(separator)
            out.write(content)


def merge_binary(files: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as out:
        for path in files:
            out.write(path.read_bytes())


def merge_jsonl(files: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for path in files:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    out.write(line if line.endswith("\n") else line + "\n")


def merge_csv(files: list[Path], output_path: Path) -> None:
    import pandas as pd

    frames = []
    for path in files:
        df = pd.read_csv(path)
        if not df.empty:
            frames.append(df)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)


def merge_parquet(files: list[Path], output_path: Path) -> None:
    import pandas as pd

    frames = []
    for path in files:
        df = pd.read_parquet(path)
        if not df.empty:
            frames.append(df)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)


def merge_group(files: list[Path], output_path: Path, mode: str, separator: str) -> None:
    if mode == "text":
        merge_text(files, output_path, separator)
    elif mode == "binary":
        merge_binary(files, output_path)
    elif mode == "jsonl":
        merge_jsonl(files, output_path)
    elif mode == "csv":
        merge_csv(files, output_path)
    elif mode == "parquet":
        merge_parquet(files, output_path)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def main() -> int:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_dir}")

    groups = collect_groups(
        root_dir=root_dir,
        recursive=args.recursive,
        include_hidden=args.include_hidden,
    )

    if not groups:
        print("No files found in subfolders.")
        return 0

    merged_count = 0
    skipped_count = 0

    for rel_path in sorted(groups.keys()):
        files = sorted(groups[rel_path])
        output_path = root_dir / rel_path

        if args.skip_existing and output_path.exists():
            skipped_count += 1
            if args.verbose:
                print(f"Skipping existing output: {output_path}")
            continue

        mode = detect_mode(rel_path, args.mode)
        if args.verbose:
            print(f"Merging {len(files)} files -> {output_path} [{mode}]")
            for file_path in files:
                print(f"  - {file_path}")

        merge_group(files, output_path, mode=mode, separator=args.separator)
        merged_count += 1

    print(f"Done. Wrote {merged_count} merged file(s). Skipped {skipped_count} existing file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
