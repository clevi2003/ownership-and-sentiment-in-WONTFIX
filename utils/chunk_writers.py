from pathlib import Path
import pandas as pd
from utils.checkpoints import sanitize_repo_name


class BaseRepoChunkWriter:
    """ general chunk writer for repositories, works for both single row tables and chunked tables"""

    def __init__(self, *, config, repo_dir, batch_size=5000):
        self.config = config
        self.repo_dir = Path(repo_dir)
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

        self._single_tables_written = set()
        self._buffers = {}
        self._part_indices = {}
        self._file_prefixes = {}

    def register_chunked_table(self, table_name):
        self._buffers[table_name] = []
        self._part_indices[table_name] = 1
        self._file_prefixes[table_name] = table_name

    def write_single_row_table(self, table_name, row, filename=None):
        if table_name in self._single_tables_written:
            return

        output_name = filename or f"{table_name}.parquet"
        output_path = self.repo_dir / output_name
        pd.DataFrame([row]).to_parquet(
            output_path,
            index=False,
            compression=self.config.storage.compression.parquet_compression,
        )
        self._single_tables_written.add(table_name)

    @staticmethod
    def dedupe_rows(rows, key_fields):
        seen = set()
        deduped = []
        for row in rows:
            key = tuple(row[field] for field in key_fields)
            if key not in seen:
                seen.add(key)
                deduped.append(row)
        return deduped

    @staticmethod
    def build_repo_dir(processed_root, stage_name, repo_full_name):
        safe_name = sanitize_repo_name(repo_full_name)
        return Path(processed_root) / "_batches" / stage_name / safe_name

    def add_row(self, table_name, row):
        if table_name not in self._buffers:
            raise ValueError(f"Table '{table_name}' is not registered as a chunked table.")

        self._buffers[table_name].append(row)
        if len(self._buffers[table_name]) >= self.batch_size:
            self.flush_table(table_name)

    def flush_table(self, table_name):
        if table_name not in self._buffers:
            raise ValueError(f"Table '{table_name}' is not registered as a chunked table.")

        rows = self._buffers[table_name]
        if not rows:
            return

        file_prefix = self._file_prefixes[table_name]
        part_index = self._part_indices[table_name]
        output_path = self.repo_dir / f"{file_prefix}_part_{part_index:05d}.parquet"

        pd.DataFrame(rows).to_parquet(
            output_path,
            index=False,
            compression=self.config.storage.compression.parquet_compression,
        )

        self._buffers[table_name] = []
        self._part_indices[table_name] += 1

    def flush_all(self):
        for table_name in self._buffers:
            self.flush_table(table_name)

    def finalize(self):
        self.flush_all()


class IssueCommentRepoChunkWriter(BaseRepoChunkWriter):
    """ Issue/Comment specific writer"""
    def __init__(self, *, config, repo_dir, batch_size=5000):
        super().__init__(config=config, repo_dir=repo_dir, batch_size=batch_size)
        self.register_chunked_table("issues")
        self.register_chunked_table("issue_comments")

    def write_repository_row(self, row):
        #self.write_single_row_table("repositories", row, filename="repositories.parquet")
        return row

    def add_issue_row(self, row):
        self.add_row("issues", row)

    def add_comment_row(self, row):
        self.add_row("issue_comments", row)


class PullRequestRepoChunkWriter(BaseRepoChunkWriter):
    """ PR specific writer"""
    def __init__(self, *, config, repo_dir, batch_size=5000):
        super().__init__(config=config, repo_dir=repo_dir, batch_size=batch_size)
        self.register_chunked_table("pull_requests")
        self.register_chunked_table("issue_pr_links")
        self.register_chunked_table("pr_commit_links")

    def add_pr_row(self, row):
        self.add_row("pull_requests", row)

    def add_issue_pr_row(self, row):
        self.add_row("issue_pr_links", row)

    def add_pr_commit_row(self, row):
        self.add_row("pr_commit_links", row)


class CommitHistoryRepoChunkWriter(BaseRepoChunkWriter):
    """ Commit history specific writer"""
    def __init__(self, *, config, repo_dir, batch_size=5000):
        super().__init__(config=config, repo_dir=repo_dir, batch_size=batch_size)
        self.register_chunked_table("commits")
        self.register_chunked_table("commit_files")

    def add_commit_row(self, row):
        self.add_row("commits", row)

    def add_commit_file_row(self, row):
        self.add_row("commit_files", row)


class IssueFileLinkRepoChunkWriter(BaseRepoChunkWriter):
    """Issue file linking specific writer"""

    def __init__(self, *, config, repo_dir, batch_size=5000):
        super().__init__(config=config, repo_dir=repo_dir, batch_size=batch_size)
        self.register_chunked_table("issue_file_links")

    def add_issue_file_link_row(self, row):
        self.add_row("issue_file_links", row)