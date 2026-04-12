from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import copy
import yaml


class ConfigError(Exception):
    """Raised when the study config is missing required fields or is malformed."""
    pass

def _require_dict(value, section_name):
    if not isinstance(value, dict):
        raise ConfigError(f"Expected '{section_name}' to be a dictionary, got {type(value).__name__}.")
    return value

def _require_list(value, section_name):
    if not isinstance(value, list):
        raise ConfigError(f"Expected '{section_name}' to be a list, got {type(value).__name__}.")
    return value

def _get_required(data, key, section_name):
    if key not in data:
        raise ConfigError(f"Missing required key '{key}' in section '{section_name}'.")
    return data[key]

def _get_optional(data, key, default=None):
    return data.get(key, default)

def _ensure_path_str(value, section_name, key):
    if not isinstance(value, str):
        raise ConfigError(f"Expected '{key}' in section '{section_name}' to be a string path.")
    return value

@dataclass
class StudyConfigSection:
    name: str
    description: str
    version: str
    semester: str
    date: str
    notes: list = field(default_factory=list)

@dataclass
class PathsConfig:
    project_root: str
    data_root: str
    raw_root: str
    processed_root: str
    linked_root: str
    features_root: str
    final_root: str
    logs_root: str
    config_root: str

@dataclass
class AuthConfig:
    use_token: bool
    token_env_var: str

@dataclass
class PaginationConfig:
    per_page: int

@dataclass
class RateLimitConfig:
    enable_backoff: bool
    min_remaining_before_pause: int
    default_pause_seconds: int
    respect_reset_header: bool
    max_retries: int
    retry_backoff_seconds: int

@dataclass
class RequestsConfig:
    timeout_seconds: int
    user_agent: str

@dataclass
class GitHubConfig:
    api_base_url: str
    use_rest_api: bool
    use_graphql_api: bool
    auth: AuthConfig
    pagination: PaginationConfig
    rate_limit: RateLimitConfig
    requests: RequestsConfig

@dataclass
class RepoActivityWindowConfig:
    activity_cutoff_date: str
    use_last_push_date: bool

@dataclass
class IssueCollectionWindowConfig:
    start_date: str
    end_date: str
    include_open_issues_after_end_date: bool

@dataclass
class ParticipationAnalysisWindowConfig:
    start_date: str
    end_date: str
    time_window_unit: str

@dataclass
class StudyWindowsConfig:
    repo_activity: RepoActivityWindowConfig
    issue_collection: IssueCollectionWindowConfig
    participation_analysis: ParticipationAnalysisWindowConfig

@dataclass
class InclusionCriteriaConfig:
    visibility: str
    min_stars: int
    exclude_forks: bool
    exclude_archived: bool
    require_recent_activity: bool
    require_at_least_one_wontfix_issue: bool

@dataclass
class DiscoveryFiltersConfig:
    languages: list = field(default_factory=list)
    owners_allowlist: list = field(default_factory=list)
    owners_blocklist: list = field(default_factory=list)
    repos_allowlist: list = field(default_factory=list)
    repos_blocklist: list = field(default_factory=list)

@dataclass
class ActivityDefinitionConfig:
    field: str
    cutoff_date: str

@dataclass
class ArchivalPolicyConfig:
    if_archive_status_unavailable: str

@dataclass
class ForkPolicyConfig:
    include_mirrors: bool

@dataclass
class RepoStatusFlagsConfig:
    track_disabled: bool
    track_archived: bool
    track_fork: bool
    track_template: bool

@dataclass
class RepoSelectionConfig:
    discover_repositories: bool
    inclusion_criteria: InclusionCriteriaConfig
    discovery_filters: DiscoveryFiltersConfig
    activity_definition: ActivityDefinitionConfig
    archival_policy: ArchivalPolicyConfig
    fork_policy: ForkPolicyConfig
    repo_status_flags: RepoStatusFlagsConfig

@dataclass
class FinalWontfixCountScreenConfig:
    enabled: bool
    count_mode: str
    min_approx_wontfix_issue_count: int
    sort_descending_before_final_limit: bool

@dataclass
class RepoDiscoveryConfig:
    max_repo_search_pages: int
    max_issue_search_pages_per_query: int
    split_wontfix_issue_search_by_year: bool
    candidate_repo_limit: int
    final_repo_limit: int
    final_wontfix_count_screen: FinalWontfixCountScreenConfig

@dataclass
class CanonicalLabelConfig:
    canonical_name: str
    variants: list
    partial_match_allowed: bool = False

@dataclass
class LabelTypeConfig:
    canonical_name: str
    variants: list

@dataclass
class OutcomeLabelsConfig:
    wontfix: CanonicalLabelConfig
    invalid: CanonicalLabelConfig

@dataclass
class IssueTypeLabelsConfig:
    bug: LabelTypeConfig
    feature_request: LabelTypeConfig
    documentation: LabelTypeConfig
    question: LabelTypeConfig

@dataclass
class HelperLabelsConfig:
    help_wanted: LabelTypeConfig
    good_first_issue: LabelTypeConfig

@dataclass
class LabelNormalizationConfig:
    case_sensitive: bool
    strip_whitespace: bool
    normalize_hyphens_and_apostrophes: bool
    normalize_unicode_quotes: bool
    outcome_labels: OutcomeLabelsConfig
    issue_type_labels: IssueTypeLabelsConfig
    helper_labels: HelperLabelsConfig

@dataclass
class IssueCommentsConfig:
    include_comments: bool
    include_comment_bodies: bool
    include_comment_authors: bool
    include_comment_timestamps: bool
    include_comment_reactions: bool
    preserve_comment_order: bool

@dataclass
class IssueSelectionConfig:
    include_issues: bool
    include_pull_requests_from_issues_endpoint: bool
    states: list
    require_created_within_window: bool
    require_labels_loaded: bool
    store_issue_title: bool
    store_issue_body: bool
    store_issue_author: bool
    store_assignees: bool
    store_milestones: bool
    store_reactions: bool
    store_timeline_events: bool
    comments: IssueCommentsConfig

@dataclass
class IssueExtractionConfig:
    enabled: bool
    max_issue_pages_per_repo_per_state: int
    max_comment_pages_per_issue: int
    max_repos_per_run: int
    resume_mode: str
    write_repo_manifest: bool
    fail_on_missing_repo_id: bool
    write_batch_size: int
    sort_before_write: bool
    request_pause_seconds_between_repos: int
    skip_repo_if_raw_exists: bool
    search_max_results_per_shard: int
    search_max_shard_splits: int
    max_search_pages_per_shard: int

@dataclass
class PullRequestSelectionConfig:
    include_pull_requests: bool
    include_pr_body: bool
    include_pr_author: bool
    include_pr_state: bool
    include_pr_created_closed_merged_dates: bool
    include_pr_commits: bool
    include_pr_files: bool
    include_review_comments: bool

@dataclass
class GitHistoryExtractFieldsConfig:
    commits: bool
    commit_message: bool
    commit_author_name: bool
    commit_author_email: bool
    commit_timestamp: bool
    parent_shas: bool
    modified_files: bool
    additions_deletions: bool
    file_change_type: bool
    renames_when_detectable: bool

@dataclass
class GitHistoryExtractionConfig:
    enabled: bool
    clone_root: str
    clone_depth: str
    use_local_git: bool
    include_full_history: bool
    history_start_date: str
    history_end_date: str
    fast_mode: bool
    fast_mode_date_window: str
    commit_message_mode: str
    max_commit_message_chars: int
    max_repos_per_run: int
    resume_mode: str
    write_batch_size: int
    request_pause_seconds_between_repos: int
    fail_on_missing_repo_id: bool
    skip_repo_if_raw_exists: bool
    extract: GitHistoryExtractFieldsConfig

@dataclass
class MatchingRulesConfig:
    same_repository: bool
    same_broad_time_window: bool
    same_issue_type_if_available: bool
    max_controls_per_wontfix: int

@dataclass
class ComparisonSetConfig:
    enabled: bool
    build_after_extraction: bool
    strategy: str
    include_non_wontfix_issues: bool
    include_invalid_as_flagged_subgroup: bool
    include_open_issues: bool
    include_closed_non_wontfix_issues: bool
    include_pr_resolved_issues: bool
    matching_rules: MatchingRulesConfig

@dataclass
class IssuePrLinkConfig:
    enabled: bool
    allowed_link_sources: list
    confidence_levels: dict
    keep_low_confidence_links: bool

@dataclass
class PrCommitLinkConfig:
    enabled: bool
    source: str

@dataclass
class IssueFileLinkConfig:
    enabled: bool
    allowed_link_sources: list
    confidence_levels: dict
    allow_rq_specific_missing_links: bool
    notes: list = field(default_factory=list)

@dataclass
class IssueFileLinkingRuntimeConfig:
    enabled: bool
    max_repos_per_run: int
    resume_mode: str
    write_batch_size: int
    include_comment_text_fallback: bool
    require_repo_file_match_for_text_links: bool
    allow_unique_basename_match: bool

@dataclass
class LinkageConfig:
    issue_pr: IssuePrLinkConfig
    pr_commit: PrCommitLinkConfig
    issue_file: IssueFileLinkConfig

@dataclass
class NormalizedNameRulesConfig:
    lowercase: bool
    strip_whitespace: bool
    collapse_internal_spaces: bool

@dataclass
class EmailRulesConfig:
    lowercase: bool
    strip_whitespace: bool
    allow_email_for_internal_mapping_only: bool

@dataclass
class IdentityResolutionConfig:
    enabled: bool
    scope: str
    create_contributor_key: bool
    contributor_key_format: str
    matching_priority: list
    max_repos_per_run: int
    resume_mode: str
    write_batch_size: int
    write_cluster_summary: bool
    preserve_normalized_columns: bool
    normalized_name_rules: NormalizedNameRulesConfig
    email_rules: EmailRulesConfig
    aggressive_fuzzy_merge: bool
    keep_unresolved_identities: bool
    store_identity_confidence: bool
    enable_pr_commit_login_email_bridge: bool
    pr_commit_bridge_min_pair_evidence: int
    pr_commit_bridge_min_distinct_prs: int
    pr_commit_bridge_require_bijective: bool

@dataclass
class BotHandlingConfig:
    detect_bots: bool
    bot_name_patterns: list
    exclude_bots_from_sentiment_analysis: bool
    exclude_bots_from_participation_counts: bool
    exclude_bots_from_ownership_metrics: bool
    keep_bot_rows_with_flag: bool

@dataclass
class CompressionConfig:
    raw_json_gzip: bool
    parquet_compression: str

@dataclass
class StorageConfig:
    raw_format: str
    processed_format: str
    summary_format: str
    processed_merge_mode: str
    compression: CompressionConfig
    overwrite_raw: bool
    overwrite_processed: bool
    save_per_repo_raw_files: bool
    save_checkpoint_files: bool
    save_intermediate_tables: bool
    append_processed_batches: bool

@dataclass
class OutputsConfig:
    repo_candidate_list: str
    repo_included_list: str
    repositories_table: str
    issues_table: str
    issue_comments_table: str
    pull_requests_table: str
    commits_table: str
    commit_files_table: str
    comparison_issue_set_table: str
    wontfix_issue_set_table: str
    issue_pr_links_table: str
    pr_commit_links_table: str
    issue_file_links_table: str
    contributor_identity_table: str
    contributor_identity_clusters_table: str
    issues_resolved_table: str
    issue_comments_resolved_table: str
    pull_requests_resolved_table: str
    commits_resolved_table: str
    extraction_summary_csv: str
    run_manifest_json: str
    resolved_config_snapshot_yaml: str
    comparison_issue_qa_summary_csv: str

@dataclass
class CheckpointingConfig:
    enabled: bool
    granularity: str
    checkpoint_dir: str
    resume_from_checkpoints: bool
    write_status_after_each_page: bool
    write_status_after_each_repo: bool

@dataclass
class LoggingConfig:
    level: str
    log_to_file: bool
    log_to_console: bool
    extraction_log_dir: str
    normalization_log_dir: str
    linkage_log_dir: str
    qa_log_dir: str

@dataclass
class QualityAssuranceConfig:
    enabled: bool
    checks: list
    fail_on_critical_errors: bool
    write_summary_report: bool
    summary_report_path: str

@dataclass
class RqScopeItemConfig:
    requires_issue_comments: bool = False
    requires_sentiment_text: bool = False
    allows_missing_issue_file_links: bool = False
    requires_git_history: bool = False
    requires_issue_file_links: bool = False
    allowed_issue_file_confidence: list = field(default_factory=list)
    requires_repo_activity_timeseries: bool = False

@dataclass
class RqScopingConfig:
    rq1: RqScopeItemConfig
    rq2: RqScopeItemConfig
    rq3: RqScopeItemConfig

@dataclass
class StudyConfig:
    study: StudyConfigSection
    paths: PathsConfig
    github: GitHubConfig
    study_windows: StudyWindowsConfig
    repo_selection: RepoSelectionConfig
    repo_discovery: RepoDiscoveryConfig
    label_normalization: LabelNormalizationConfig
    issue_selection: IssueSelectionConfig
    issue_extraction: IssueExtractionConfig
    pull_request_selection: PullRequestSelectionConfig
    git_history_extraction: GitHistoryExtractionConfig
    comparison_set: ComparisonSetConfig
    linkage: LinkageConfig
    issue_file_linking: IssueFileLinkingRuntimeConfig
    identity_resolution: IdentityResolutionConfig
    bot_handling: BotHandlingConfig
    storage: StorageConfig
    outputs: OutputsConfig
    checkpointing: CheckpointingConfig
    logging: LoggingConfig
    quality_assurance: QualityAssuranceConfig
    rq_scoping: RqScopingConfig

def _parse_study(data):
    section = "study"
    d = _require_dict(_get_required(data, "study", "root"), section)
    return StudyConfigSection(
        name=_get_required(d, "name", section),
        description=_get_required(d, "description", section),
        version=_get_required(d, "version", section),
        semester=_get_required(d, "semester", section),
        date=_get_required(d, "date", section),
        notes=_get_optional(d, "notes", []),
    )

def _parse_paths(data):
    section = "paths"
    d = _require_dict(_get_required(data, "paths", "root"), section)
    return PathsConfig(
        project_root=_ensure_path_str(_get_required(d, "project_root", section), section, "project_root"),
        data_root=_ensure_path_str(_get_required(d, "data_root", section), section, "data_root"),
        raw_root=_ensure_path_str(_get_required(d, "raw_root", section), section, "raw_root"),
        processed_root=_ensure_path_str(_get_required(d, "processed_root", section), section, "processed_root"),
        linked_root=_ensure_path_str(_get_required(d, "linked_root", section), section, "linked_root"),
        features_root=_ensure_path_str(_get_required(d, "features_root", section), section, "features_root"),
        final_root=_ensure_path_str(_get_required(d, "final_root", section), section, "final_root"),
        logs_root=_ensure_path_str(_get_required(d, "logs_root", section), section, "logs_root"),
        config_root=_ensure_path_str(_get_required(d, "config_root", section), section, "config_root"),
    )

def _parse_github(data):
    section = "github"
    d = _require_dict(_get_required(data, "github", "root"), section)

    auth_d = _require_dict(_get_required(d, "auth", section), "github.auth")
    pagination_d = _require_dict(_get_required(d, "pagination", section), "github.pagination")
    rate_limit_d = _require_dict(_get_required(d, "rate_limit", section), "github.rate_limit")
    requests_d = _require_dict(_get_required(d, "requests", section), "github.requests")

    return GitHubConfig(
        api_base_url=_get_required(d, "api_base_url", section),
        use_rest_api=_get_required(d, "use_rest_api", section),
        use_graphql_api=_get_required(d, "use_graphql_api", section),
        auth=AuthConfig(
            use_token=_get_required(auth_d, "use_token", "github.auth"),
            token_env_var=_get_required(auth_d, "token_env_var", "github.auth"),
        ),
        pagination=PaginationConfig(
            per_page=_get_required(pagination_d, "per_page", "github.pagination"),
        ),
        rate_limit=RateLimitConfig(
            enable_backoff=_get_required(rate_limit_d, "enable_backoff", "github.rate_limit"),
            min_remaining_before_pause=_get_required(rate_limit_d, "min_remaining_before_pause", "github.rate_limit"),
            default_pause_seconds=_get_required(rate_limit_d, "default_pause_seconds", "github.rate_limit"),
            respect_reset_header=_get_required(rate_limit_d, "respect_reset_header", "github.rate_limit"),
            max_retries=_get_required(rate_limit_d, "max_retries", "github.rate_limit"),
            retry_backoff_seconds=_get_required(rate_limit_d, "retry_backoff_seconds", "github.rate_limit"),
        ),
        requests=RequestsConfig(
            timeout_seconds=_get_required(requests_d, "timeout_seconds", "github.requests"),
            user_agent=_get_required(requests_d, "user_agent", "github.requests"),
        ),
    )

def _parse_study_windows(data):
    section = "study_windows"
    d = _require_dict(_get_required(data, "study_windows", "root"), section)

    repo_activity_d = _require_dict(_get_required(d, "repo_activity", section), "study_windows.repo_activity")
    issue_collection_d = _require_dict(_get_required(d, "issue_collection", section), "study_windows.issue_collection")
    participation_d = _require_dict(_get_required(d, "participation_analysis", section), "study_windows.participation_analysis")

    return StudyWindowsConfig(
        repo_activity=RepoActivityWindowConfig(
            activity_cutoff_date=_get_required(repo_activity_d, "activity_cutoff_date", "study_windows.repo_activity"),
            use_last_push_date=_get_required(repo_activity_d, "use_last_push_date", "study_windows.repo_activity"),
        ),
        issue_collection=IssueCollectionWindowConfig(
            start_date=_get_required(issue_collection_d, "start_date", "study_windows.issue_collection"),
            end_date=_get_required(issue_collection_d, "end_date", "study_windows.issue_collection"),
            include_open_issues_after_end_date=_get_required(issue_collection_d, "include_open_issues_after_end_date", "study_windows.issue_collection"),
        ),
        participation_analysis=ParticipationAnalysisWindowConfig(
            start_date=_get_required(participation_d, "start_date", "study_windows.participation_analysis"),
            end_date=_get_required(participation_d, "end_date", "study_windows.participation_analysis"),
            time_window_unit=_get_required(participation_d, "time_window_unit", "study_windows.participation_analysis"),
        ),
    )

def _parse_repo_selection(data):
    section = "repo_selection"
    d = _require_dict(_get_required(data, "repo_selection", "root"), section)

    inclusion_d = _require_dict(_get_required(d, "inclusion_criteria", section), "repo_selection.inclusion_criteria")
    filters_d = _require_dict(_get_required(d, "discovery_filters", section), "repo_selection.discovery_filters")
    activity_d = _require_dict(_get_required(d, "activity_definition", section), "repo_selection.activity_definition")
    archival_d = _require_dict(_get_required(d, "archival_policy", section), "repo_selection.archival_policy")
    fork_d = _require_dict(_get_required(d, "fork_policy", section), "repo_selection.fork_policy")
    flags_d = _require_dict(_get_required(d, "repo_status_flags", section), "repo_selection.repo_status_flags")

    return RepoSelectionConfig(
        discover_repositories=_get_required(d, "discover_repositories", section),
        inclusion_criteria=InclusionCriteriaConfig(
            visibility=_get_required(inclusion_d, "visibility", "repo_selection.inclusion_criteria"),
            min_stars=_get_required(inclusion_d, "min_stars", "repo_selection.inclusion_criteria"),
            exclude_forks=_get_required(inclusion_d, "exclude_forks", "repo_selection.inclusion_criteria"),
            exclude_archived=_get_required(inclusion_d, "exclude_archived", "repo_selection.inclusion_criteria"),
            require_recent_activity=_get_required(inclusion_d, "require_recent_activity", "repo_selection.inclusion_criteria"),
            require_at_least_one_wontfix_issue=_get_required(inclusion_d, "require_at_least_one_wontfix_issue", "repo_selection.inclusion_criteria"),
        ),
        discovery_filters=DiscoveryFiltersConfig(
            languages=_get_optional(filters_d, "languages", []),
            owners_allowlist=_get_optional(filters_d, "owners_allowlist", []),
            owners_blocklist=_get_optional(filters_d, "owners_blocklist", []),
            repos_allowlist=_get_optional(filters_d, "repos_allowlist", []),
            repos_blocklist=_get_optional(filters_d, "repos_blocklist", []),
        ),
        activity_definition=ActivityDefinitionConfig(
            field=_get_required(activity_d, "field", "repo_selection.activity_definition"),
            cutoff_date=_get_required(activity_d, "cutoff_date", "repo_selection.activity_definition"),
        ),
        archival_policy=ArchivalPolicyConfig(
            if_archive_status_unavailable=_get_required(archival_d, "if_archive_status_unavailable", "repo_selection.archival_policy"),
        ),
        fork_policy=ForkPolicyConfig(
            include_mirrors=_get_required(fork_d, "include_mirrors", "repo_selection.fork_policy"),
        ),
        repo_status_flags=RepoStatusFlagsConfig(
            track_disabled=_get_required(flags_d, "track_disabled", "repo_selection.repo_status_flags"),
            track_archived=_get_required(flags_d, "track_archived", "repo_selection.repo_status_flags"),
            track_fork=_get_required(flags_d, "track_fork", "repo_selection.repo_status_flags"),
            track_template=_get_required(flags_d, "track_template", "repo_selection.repo_status_flags"),
        ),
    )

def _parse_repo_discovery(data):
    section = "repo_discovery"
    d = _require_dict(_get_required(data, "repo_discovery", "root"), section)
    final_count_d = _require_dict(
        _get_required(d, "final_wontfix_count_screen", section),
        "repo_discovery.final_wontfix_count_screen",
    )

    return RepoDiscoveryConfig(
        max_repo_search_pages=_get_required(d, "max_repo_search_pages", section),
        max_issue_search_pages_per_query=_get_required(d, "max_issue_search_pages_per_query", section),
        split_wontfix_issue_search_by_year=_get_required(d, "split_wontfix_issue_search_by_year", section),
        candidate_repo_limit=_get_required(d, "candidate_repo_limit", section),
        final_repo_limit=_get_required(d, "final_repo_limit", section),
        final_wontfix_count_screen=FinalWontfixCountScreenConfig(
            enabled=_get_required(
                final_count_d,
                "enabled",
                "repo_discovery.final_wontfix_count_screen",
            ),
            count_mode=_get_required(
                final_count_d,
                "count_mode",
                "repo_discovery.final_wontfix_count_screen",
            ),
            min_approx_wontfix_issue_count=_get_required(
                final_count_d,
                "min_approx_wontfix_issue_count",
                "repo_discovery.final_wontfix_count_screen",
            ),
            sort_descending_before_final_limit=_get_required(
                final_count_d,
                "sort_descending_before_final_limit",
                "repo_discovery.final_wontfix_count_screen",
            ),
        ),
    )

def _parse_canonical_label(data, section_name):
    d = _require_dict(data, section_name)
    return CanonicalLabelConfig(
        canonical_name=_get_required(d, "canonical_name", section_name),
        variants=_require_list(_get_required(d, "variants", section_name), section_name + ".variants"),
        partial_match_allowed=_get_optional(d, "partial_match_allowed", False),
    )

def _parse_label_type(data, section_name):
    d = _require_dict(data, section_name)
    return LabelTypeConfig(
        canonical_name=_get_required(d, "canonical_name", section_name),
        variants=_require_list(_get_required(d, "variants", section_name), section_name + ".variants"),
    )

def _parse_label_normalization(data):
    section = "label_normalization"
    d = _require_dict(_get_required(data, "label_normalization", "root"), section)

    outcome_d = _require_dict(_get_required(d, "outcome_labels", section), "label_normalization.outcome_labels")
    issue_types_d = _require_dict(_get_required(d, "issue_type_labels", section), "label_normalization.issue_type_labels")
    helper_d = _require_dict(_get_required(d, "helper_labels", section), "label_normalization.helper_labels")

    return LabelNormalizationConfig(
        case_sensitive=_get_required(d, "case_sensitive", section),
        strip_whitespace=_get_required(d, "strip_whitespace", section),
        normalize_hyphens_and_apostrophes=_get_required(d, "normalize_hyphens_and_apostrophes", section),
        normalize_unicode_quotes=_get_required(d, "normalize_unicode_quotes", section),
        outcome_labels=OutcomeLabelsConfig(
            wontfix=_parse_canonical_label(_get_required(outcome_d, "wontfix", "label_normalization.outcome_labels"), "label_normalization.outcome_labels.wontfix"),
            invalid=_parse_canonical_label(_get_required(outcome_d, "invalid", "label_normalization.outcome_labels"), "label_normalization.outcome_labels.invalid"),
        ),
        issue_type_labels=IssueTypeLabelsConfig(
            bug=_parse_label_type(_get_required(issue_types_d, "bug", "label_normalization.issue_type_labels"), "label_normalization.issue_type_labels.bug"),
            feature_request=_parse_label_type(_get_required(issue_types_d, "feature_request", "label_normalization.issue_type_labels"), "label_normalization.issue_type_labels.feature_request"),
            documentation=_parse_label_type(_get_required(issue_types_d, "documentation", "label_normalization.issue_type_labels"), "label_normalization.issue_type_labels.documentation"),
            question=_parse_label_type(_get_required(issue_types_d, "question", "label_normalization.issue_type_labels"), "label_normalization.issue_type_labels.question"),
        ),
        helper_labels=HelperLabelsConfig(
            help_wanted=_parse_label_type(_get_required(helper_d, "help_wanted", "label_normalization.helper_labels"), "label_normalization.helper_labels.help_wanted"),
            good_first_issue=_parse_label_type(_get_required(helper_d, "good_first_issue", "label_normalization.helper_labels"), "label_normalization.helper_labels.good_first_issue"),
        ),
    )

def _parse_issue_selection(data):
    section = "issue_selection"
    d = _require_dict(_get_required(data, "issue_selection", "root"), section)
    comments_d = _require_dict(_get_required(d, "comments", section), "issue_selection.comments")

    return IssueSelectionConfig(
        include_issues=_get_required(d, "include_issues", section),
        include_pull_requests_from_issues_endpoint=_get_required(d, "include_pull_requests_from_issues_endpoint", section),
        states=_require_list(_get_required(d, "states", section), "issue_selection.states"),
        require_created_within_window=_get_required(d, "require_created_within_window", section),
        require_labels_loaded=_get_required(d, "require_labels_loaded", section),
        store_issue_title=_get_required(d, "store_issue_title", section),
        store_issue_body=_get_required(d, "store_issue_body", section),
        store_issue_author=_get_required(d, "store_issue_author", section),
        store_assignees=_get_required(d, "store_assignees", section),
        store_milestones=_get_required(d, "store_milestones", section),
        store_reactions=_get_required(d, "store_reactions", section),
        store_timeline_events=_get_required(d, "store_timeline_events", section),
        comments=IssueCommentsConfig(
            include_comments=_get_required(comments_d, "include_comments", "issue_selection.comments"),
            include_comment_bodies=_get_required(comments_d, "include_comment_bodies", "issue_selection.comments"),
            include_comment_authors=_get_required(comments_d, "include_comment_authors", "issue_selection.comments"),
            include_comment_timestamps=_get_required(comments_d, "include_comment_timestamps", "issue_selection.comments"),
            include_comment_reactions=_get_required(comments_d, "include_comment_reactions", "issue_selection.comments"),
            preserve_comment_order=_get_required(comments_d, "preserve_comment_order", "issue_selection.comments"),
        ),
    )

def _parse_issue_extraction(data):
    section = "issue_extraction"
    d = _require_dict(_get_required(data, "issue_extraction", "root"), section)
    return IssueExtractionConfig(
        enabled=_get_required(d, "enabled", section),
        max_issue_pages_per_repo_per_state=_get_required(d, "max_issue_pages_per_repo_per_state", section),
        max_comment_pages_per_issue=_get_required(d, "max_comment_pages_per_issue", section),
        max_repos_per_run=_get_optional(d, "max_repos_per_run", None),
        resume_mode=_get_required(d, "resume_mode", section),
        write_repo_manifest=_get_required(d, "write_repo_manifest", section),
        fail_on_missing_repo_id=_get_required(d, "fail_on_missing_repo_id", section),
        write_batch_size=_get_required(d, "write_batch_size", section),
        sort_before_write=_get_required(d, "sort_before_write", section),
        request_pause_seconds_between_repos=_get_required(d, "request_pause_seconds_between_repos", section),
        skip_repo_if_raw_exists=_get_required(d, "skip_repo_if_raw_exists", section),
        search_max_results_per_shard=_get_optional(d, "search_max_results_per_shard", 900),
        search_max_shard_splits=_get_optional(d, "search_max_shard_splits", 1000),
        max_search_pages_per_shard=_get_optional(d, "max_search_pages_per_shard", None),
    )

def _parse_pull_request_selection(data):
    section = "pull_request_selection"
    d = _require_dict(_get_required(data, "pull_request_selection", "root"), section)
    return PullRequestSelectionConfig(
        include_pull_requests=_get_required(d, "include_pull_requests", section),
        include_pr_body=_get_required(d, "include_pr_body", section),
        include_pr_author=_get_required(d, "include_pr_author", section),
        include_pr_state=_get_required(d, "include_pr_state", section),
        include_pr_created_closed_merged_dates=_get_required(d, "include_pr_created_closed_merged_dates", section),
        include_pr_commits=_get_required(d, "include_pr_commits", section),
        include_pr_files=_get_required(d, "include_pr_files", section),
        include_review_comments=_get_required(d, "include_review_comments", section),
    )

def _parse_git_history_extraction(data):
    section = "git_history_extraction"
    d = _require_dict(_get_required(data, "git_history_extraction", "root"), section)
    extract_d = _require_dict(_get_required(d, "extract", section), "git_history_extraction.extract")

    return GitHistoryExtractionConfig(
        enabled=_get_required(d, "enabled", section),
        clone_root=_get_required(d, "clone_root", section),
        clone_depth=_get_required(d, "clone_depth", section),
        use_local_git=_get_required(d, "use_local_git", section),
        include_full_history=_get_required(d, "include_full_history", section),
        history_start_date=_get_optional(d, "history_start_date", None),
        history_end_date=_get_optional(d, "history_end_date", None),
        fast_mode=_get_optional(d, "fast_mode", False),
        fast_mode_date_window=_get_optional(d, "fast_mode_date_window", "participation_analysis"),
        commit_message_mode=_get_optional(d, "commit_message_mode", "full"),
        max_commit_message_chars=_get_optional(d, "max_commit_message_chars", None),
        max_repos_per_run=_get_optional(d, "max_repos_per_run", None),
        resume_mode=_get_required(d, "resume_mode", section),
        write_batch_size=_get_required(d, "write_batch_size", section),
        request_pause_seconds_between_repos=_get_required(d, "request_pause_seconds_between_repos", section),
        fail_on_missing_repo_id=_get_required(d, "fail_on_missing_repo_id", section),
        skip_repo_if_raw_exists=_get_required(d, "skip_repo_if_raw_exists", section),
        extract=GitHistoryExtractFieldsConfig(
            commits=_get_required(extract_d, "commits", "git_history_extraction.extract"),
            commit_message=_get_required(extract_d, "commit_message", "git_history_extraction.extract"),
            commit_author_name=_get_required(extract_d, "commit_author_name", "git_history_extraction.extract"),
            commit_author_email=_get_required(extract_d, "commit_author_email", "git_history_extraction.extract"),
            commit_timestamp=_get_required(extract_d, "commit_timestamp", "git_history_extraction.extract"),
            parent_shas=_get_required(extract_d, "parent_shas", "git_history_extraction.extract"),
            modified_files=_get_required(extract_d, "modified_files", "git_history_extraction.extract"),
            additions_deletions=_get_required(extract_d, "additions_deletions", "git_history_extraction.extract"),
            file_change_type=_get_required(extract_d, "file_change_type", "git_history_extraction.extract"),
            renames_when_detectable=_get_required(extract_d, "renames_when_detectable",
                                                  "git_history_extraction.extract"),
        ),
    )

def _parse_comparison_set(data):
    section = "comparison_set"
    d = _require_dict(_get_required(data, "comparison_set", "root"), section)
    matching_d = _require_dict(_get_required(d, "matching_rules", section), "comparison_set.matching_rules")

    return ComparisonSetConfig(
        enabled=_get_required(d, "enabled", section),
        build_after_extraction=_get_required(d, "build_after_extraction", section),
        strategy=_get_required(d, "strategy", section),
        include_non_wontfix_issues=_get_required(d, "include_non_wontfix_issues", section),
        include_invalid_as_flagged_subgroup=_get_required(d, "include_invalid_as_flagged_subgroup", section),
        include_open_issues=_get_required(d, "include_open_issues", section),
        include_closed_non_wontfix_issues=_get_required(d, "include_closed_non_wontfix_issues", section),
        include_pr_resolved_issues=_get_required(d, "include_pr_resolved_issues", section),
        matching_rules=MatchingRulesConfig(
            same_repository=_get_required(matching_d, "same_repository", "comparison_set.matching_rules"),
            same_broad_time_window=_get_required(matching_d, "same_broad_time_window", "comparison_set.matching_rules"),
            same_issue_type_if_available=_get_required(matching_d, "same_issue_type_if_available", "comparison_set.matching_rules"),
            max_controls_per_wontfix=_get_required(matching_d, "max_controls_per_wontfix", "comparison_set.matching_rules"),
        ),
    )

def _parse_linkage(data):
    section = "linkage"
    d = _require_dict(_get_required(data, "linkage", "root"), section)
    issue_pr_d = _require_dict(_get_required(d, "issue_pr", section), "linkage.issue_pr")
    pr_commit_d = _require_dict(_get_required(d, "pr_commit", section), "linkage.pr_commit")
    issue_file_d = _require_dict(_get_required(d, "issue_file", section), "linkage.issue_file")

    return LinkageConfig(
        issue_pr=IssuePrLinkConfig(
            enabled=_get_required(issue_pr_d, "enabled", "linkage.issue_pr"),
            allowed_link_sources=_require_list(
                _get_required(issue_pr_d, "allowed_link_sources", "linkage.issue_pr"),
                "linkage.issue_pr.allowed_link_sources",
            ),
            confidence_levels=_require_dict(
                _get_required(issue_pr_d, "confidence_levels", "linkage.issue_pr"),
                "linkage.issue_pr.confidence_levels",
            ),
            keep_low_confidence_links=_get_required(
                issue_pr_d,
                "keep_low_confidence_links",
                "linkage.issue_pr",
            ),
        ),
        pr_commit=PrCommitLinkConfig(
            enabled=_get_required(pr_commit_d, "enabled", "linkage.pr_commit"),
            source=_get_required(pr_commit_d, "source", "linkage.pr_commit"),
        ),
        issue_file=IssueFileLinkConfig(
            enabled=_get_required(issue_file_d, "enabled", "linkage.issue_file"),
            allowed_link_sources=_require_list(
                _get_required(issue_file_d, "allowed_link_sources", "linkage.issue_file"),
                "linkage.issue_file.allowed_link_sources",
            ),
            confidence_levels=_require_dict(
                _get_required(issue_file_d, "confidence_levels", "linkage.issue_file"),
                "linkage.issue_file.confidence_levels",
            ),
            allow_rq_specific_missing_links=_get_required(
                issue_file_d,
                "allow_rq_specific_missing_links",
                "linkage.issue_file",
            ),
            notes=_get_optional(issue_file_d, "notes", []),
        ),
    )

def _parse_issue_file_linking_runtime(data):
    section = "issue_file_linking"
    d = _require_dict(_get_required(data, "issue_file_linking", "root"), section)
    return IssueFileLinkingRuntimeConfig(
        enabled=_get_required(d, "enabled", section),
        max_repos_per_run=_get_required(d, "max_repos_per_run", section),
        resume_mode=_get_required(d, "resume_mode", section),
        write_batch_size=_get_required(d, "write_batch_size", section),
        include_comment_text_fallback=_get_required(d, "include_comment_text_fallback", section),
        require_repo_file_match_for_text_links=_get_required(d, "require_repo_file_match_for_text_links", section),
        allow_unique_basename_match=_get_required(d, "allow_unique_basename_match", section),
    )

def _parse_identity_resolution(data):
    section = "identity_resolution"
    d = _require_dict(_get_required(data, "identity_resolution", "root"), section)
    name_rules_d = _require_dict(_get_required(d, "normalized_name_rules", section), "identity_resolution.normalized_name_rules")
    email_rules_d = _require_dict(_get_required(d, "email_rules", section), "identity_resolution.email_rules")

    return IdentityResolutionConfig(
        enabled=_get_required(d, "enabled", section),
        scope=_get_required(d, "scope", section),
        create_contributor_key=_get_required(d, "create_contributor_key", section),
        contributor_key_format=_get_required(d, "contributor_key_format", section),
        matching_priority=_require_list(_get_required(d, "matching_priority", section),
                                        "identity_resolution.matching_priority"),
        max_repos_per_run=_get_optional(d, "max_repos_per_run", None),
        resume_mode=_get_optional(d, "resume_mode", "fresh"),
        write_batch_size=_get_optional(d, "write_batch_size", 5000),
        write_cluster_summary=_get_optional(d, "write_cluster_summary", True),
        preserve_normalized_columns=_get_optional(d, "preserve_normalized_columns", True),
        normalized_name_rules=NormalizedNameRulesConfig(
            lowercase=_get_required(name_rules_d, "lowercase", "identity_resolution.normalized_name_rules"),
            strip_whitespace=_get_required(name_rules_d, "strip_whitespace",
                                           "identity_resolution.normalized_name_rules"),
            collapse_internal_spaces=_get_required(name_rules_d, "collapse_internal_spaces",
                                                   "identity_resolution.normalized_name_rules"),
        ),
        email_rules=EmailRulesConfig(
            lowercase=_get_required(email_rules_d, "lowercase", "identity_resolution.email_rules"),
            strip_whitespace=_get_required(email_rules_d, "strip_whitespace", "identity_resolution.email_rules"),
            allow_email_for_internal_mapping_only=_get_required(email_rules_d, "allow_email_for_internal_mapping_only",
                                                                "identity_resolution.email_rules"),
        ),
        aggressive_fuzzy_merge=_get_required(d, "aggressive_fuzzy_merge", section),
        keep_unresolved_identities=_get_required(d, "keep_unresolved_identities", section),
        store_identity_confidence=_get_required(d, "store_identity_confidence", section),
        enable_pr_commit_login_email_bridge=_get_optional(d, "enable_pr_commit_login_email_bridge", True),
        pr_commit_bridge_min_pair_evidence=_get_optional(d, "pr_commit_bridge_min_pair_evidence", 2),
        pr_commit_bridge_min_distinct_prs=_get_optional(d, "pr_commit_bridge_min_distinct_prs", 1),
        pr_commit_bridge_require_bijective=_get_optional(d, "pr_commit_bridge_require_bijective", True),
    )

def _parse_bot_handling(data):
    section = "bot_handling"
    d = _require_dict(_get_required(data, "bot_handling", "root"), section)
    return BotHandlingConfig(
        detect_bots=_get_required(d, "detect_bots", section),
        bot_name_patterns=_require_list(_get_required(d, "bot_name_patterns", section), "bot_handling.bot_name_patterns"),
        exclude_bots_from_sentiment_analysis=_get_required(d, "exclude_bots_from_sentiment_analysis", section),
        exclude_bots_from_participation_counts=_get_required(d, "exclude_bots_from_participation_counts", section),
        exclude_bots_from_ownership_metrics=_get_required(d, "exclude_bots_from_ownership_metrics", section),
        keep_bot_rows_with_flag=_get_required(d, "keep_bot_rows_with_flag", section),
    )

def _parse_storage(data):
    section = "storage"
    d = _require_dict(_get_required(data, "storage", "root"), section)
    compression_d = _require_dict(_get_required(d, "compression", section), "storage.compression")

    return StorageConfig(
        raw_format=_get_required(d, "raw_format", section),
        processed_format=_get_required(d, "processed_format", section),
        summary_format=_get_required(d, "summary_format", section),
        processed_merge_mode=_get_optional(d, "processed_merge_mode", "single_parquet"),
        compression=CompressionConfig(
            raw_json_gzip=_get_required(compression_d, "raw_json_gzip", "storage.compression"),
            parquet_compression=_get_required(compression_d, "parquet_compression", "storage.compression"),
        ),
        overwrite_raw=_get_required(d, "overwrite_raw", section),
        overwrite_processed=_get_required(d, "overwrite_processed", section),
        save_per_repo_raw_files=_get_required(d, "save_per_repo_raw_files", section),
        save_checkpoint_files=_get_required(d, "save_checkpoint_files", section),
        save_intermediate_tables=_get_required(d, "save_intermediate_tables", section),
        append_processed_batches=_get_required(d, "append_processed_batches", section),
    )

def _parse_outputs(raw_outputs):
    raw_outputs = raw_outputs or {}
    return OutputsConfig(
        repo_candidate_list=raw_outputs.get("repo_candidate_list", "./data/processed/repo_candidates.csv"),
        repo_included_list=raw_outputs.get("repo_included_list", "./data/processed/repo_list.csv"),
        repositories_table=raw_outputs.get("repositories_table", "./data/processed/repositories.parquet"),
        issues_table=raw_outputs.get("issues_table", "./data/processed/issues.parquet"),
        issue_comments_table=raw_outputs.get("issue_comments_table", "./data/processed/issue_comments.parquet"),
        pull_requests_table=raw_outputs.get("pull_requests_table", "./data/processed/pull_requests.parquet"),
        commits_table=raw_outputs.get("commits_table", "./data/processed/commits.parquet"),
        commit_files_table=raw_outputs.get("commit_files_table", "./data/processed/commit_files.parquet"),

        comparison_issue_set_table=raw_outputs.get("comparison_issue_set_table", "./data/final/issue_sets/comparison_issue_set.parquet"),
        wontfix_issue_set_table=raw_outputs.get("wontfix_issue_set_table", "./data/final/issue_sets/wontfix_issue_set.parquet"),

        issue_pr_links_table=raw_outputs.get("issue_pr_links_table", "./data/linked/entity_links/issue_pr_links.parquet"),
        pr_commit_links_table=raw_outputs.get("pr_commit_links_table", "./data/linked/entity_links/pr_commit_links.parquet"),
        issue_file_links_table=raw_outputs.get("issue_file_links_table", "./data/linked/entity_links/issue_file_links.parquet"),
        contributor_identity_table=raw_outputs.get("contributor_identity_table", "./data/linked/identity/contributor_identity_map.parquet"),

        contributor_identity_clusters_table=raw_outputs.get("contributor_identity_clusters_table", "./data/linked/identity/contributor_identity_clusters.parquet"),
        issues_resolved_table=raw_outputs.get("issues_resolved_table", "./data/linked/resolved_entities/issues_resolved.parquet"),
        issue_comments_resolved_table=raw_outputs.get("issue_comments_resolved_table", "./data/linked/resolved_entities/issue_comments_resolved.parquet" ),
        pull_requests_resolved_table=raw_outputs.get("pull_requests_resolved_table", "./data/linked/resolved_entities/pull_requests_resolved.parquet"),
        commits_resolved_table=raw_outputs.get("commits_resolved_table", "./data/linked/resolved_entities/commits_resolved.parquet"),

        extraction_summary_csv=raw_outputs.get("extraction_summary_csv", "./logs/extraction/issues_comments_extraction_summary.csv"),
        run_manifest_json=raw_outputs.get("run_manifest_json", "./logs/extraction/issues_comments_run_manifest.json"),
        resolved_config_snapshot_yaml=raw_outputs.get("resolved_config_snapshot_yaml", "./logs/extraction/resolved_study_config.yaml"),
        comparison_issue_qa_summary_csv=raw_outputs.get("comparison_issue_qa_summary_csv", "./logs/qa/comparison_issue_set_summary.csv"),
    )

def _parse_checkpointing(data):
    section = "checkpointing"
    d = _require_dict(_get_required(data, "checkpointing", "root"), section)
    return CheckpointingConfig(
        enabled=_get_required(d, "enabled", section),
        granularity=_get_required(d, "granularity", section),
        checkpoint_dir=_get_required(d, "checkpoint_dir", section),
        resume_from_checkpoints=_get_required(d, "resume_from_checkpoints", section),
        write_status_after_each_page=_get_required(d, "write_status_after_each_page", section),
        write_status_after_each_repo=_get_required(d, "write_status_after_each_repo", section),
    )

def _parse_logging(data):
    section = "logging"
    d = _require_dict(_get_required(data, "logging", "root"), section)
    return LoggingConfig(
        level=_get_required(d, "level", section),
        log_to_file=_get_required(d, "log_to_file", section),
        log_to_console=_get_required(d, "log_to_console", section),
        extraction_log_dir=_get_required(d, "extraction_log_dir", section),
        normalization_log_dir=_get_required(d, "normalization_log_dir", section),
        linkage_log_dir=_get_required(d, "linkage_log_dir", section),
        qa_log_dir=_get_required(d, "qa_log_dir", section),
    )

def _parse_quality_assurance(data):
    section = "quality_assurance"
    d = _require_dict(_get_required(data, "quality_assurance", "root"), section)
    return QualityAssuranceConfig(
        enabled=_get_required(d, "enabled", section),
        checks=_require_list(_get_required(d, "checks", section), "quality_assurance.checks"),
        fail_on_critical_errors=_get_required(d, "fail_on_critical_errors", section),
        write_summary_report=_get_required(d, "write_summary_report", section),
        summary_report_path=_get_required(d, "summary_report_path", section),
    )

def _parse_rq_scope_item(data):
    return RqScopeItemConfig(
        requires_issue_comments=_get_optional(data, "requires_issue_comments", False),
        requires_sentiment_text=_get_optional(data, "requires_sentiment_text", False),
        allows_missing_issue_file_links=_get_optional(data, "allows_missing_issue_file_links", False),
        requires_git_history=_get_optional(data, "requires_git_history", False),
        requires_issue_file_links=_get_optional(data, "requires_issue_file_links", False),
        allowed_issue_file_confidence=_get_optional(data, "allowed_issue_file_confidence", []),
        requires_repo_activity_timeseries=_get_optional(data, "requires_repo_activity_timeseries", False),
    )

def _parse_rq_scoping(data):
    section = "rq_scoping"
    d = _require_dict(_get_required(data, "rq_scoping", "root"), section)
    return RqScopingConfig(
        rq1=_parse_rq_scope_item(_require_dict(_get_required(d, "rq1", section), "rq_scoping.rq1")),
        rq2=_parse_rq_scope_item(_require_dict(_get_required(d, "rq2", section), "rq_scoping.rq2")),
        rq3=_parse_rq_scope_item(_require_dict(_get_required(d, "rq3", section), "rq_scoping.rq3")),
    )

def load_study_config(config_path):
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if raw_data is None:
        raise ConfigError(f"Config file is empty: {config_path}")

    raw_data = _require_dict(raw_data, "root")

    config = StudyConfig(
        study=_parse_study(raw_data),
        paths=_parse_paths(raw_data),
        github=_parse_github(raw_data),
        study_windows=_parse_study_windows(raw_data),
        repo_selection=_parse_repo_selection(raw_data),
        repo_discovery=_parse_repo_discovery(raw_data),
        label_normalization=_parse_label_normalization(raw_data),
        issue_selection=_parse_issue_selection(raw_data),
        issue_extraction=_parse_issue_extraction(raw_data),
        pull_request_selection=_parse_pull_request_selection(raw_data),
        git_history_extraction=_parse_git_history_extraction(raw_data),
        comparison_set=_parse_comparison_set(raw_data),
        linkage=_parse_linkage(raw_data),
        issue_file_linking=_parse_issue_file_linking_runtime(raw_data),
        identity_resolution=_parse_identity_resolution(raw_data),
        bot_handling=_parse_bot_handling(raw_data),
        storage=_parse_storage(raw_data),
        outputs=_parse_outputs(raw_data),
        checkpointing=_parse_checkpointing(raw_data),
        logging=_parse_logging(raw_data),
        quality_assurance=_parse_quality_assurance(raw_data),
        rq_scoping=_parse_rq_scoping(raw_data),
    )
    validate_study_config(config)
    config = resolve_config_paths(config)
    return config

def _validate_date_string(value, field_name):
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ConfigError(f"{field_name} must be YYYY-MM-DD.") from exc

def _resolve_path(base_path, value):
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(base_path) / path).resolve())

def validate_study_config(config):
    if config.repo_selection.inclusion_criteria.min_stars < 0:
        raise ConfigError("repo_selection.inclusion_criteria.min_stars must be >= 0.")

    if config.repo_discovery.max_repo_search_pages <= 0:
        raise ConfigError("repo_discovery.max_repo_search_pages must be > 0.")
    if config.repo_discovery.max_issue_search_pages_per_query <= 0:
        raise ConfigError("repo_discovery.max_issue_search_pages_per_query must be > 0.")
    if config.repo_discovery.candidate_repo_limit <= 0:
        raise ConfigError("repo_discovery.candidate_repo_limit must be > 0.")
    if config.repo_discovery.final_repo_limit <= 0:
        raise ConfigError("repo_discovery.final_repo_limit must be > 0.")
    if config.repo_discovery.final_repo_limit > config.repo_discovery.candidate_repo_limit:
        raise ConfigError("repo_discovery.final_repo_limit should be less than or equal to repo_discovery.candidate_repo_limit.")
    allowed_wontfix_count_modes = {"repo_search_total_count"}
    if config.repo_discovery.final_wontfix_count_screen.count_mode not in allowed_wontfix_count_modes:
        raise ConfigError(f"repo_discovery.final_wontfix_count_screen.count_mode must be one of {sorted(allowed_wontfix_count_modes)}.")
    if config.repo_discovery.final_wontfix_count_screen.min_approx_wontfix_issue_count < 0:
        raise ConfigError("repo_discovery.final_wontfix_count_screen.min_approx_wontfix_issue_count must be >= 0.")

    if config.github.pagination.per_page <= 0:
        raise ConfigError("github.pagination.per_page must be > 0.")
    if config.github.rate_limit.max_retries < 0:
        raise ConfigError("github.rate_limit.max_retries must be >= 0.")

    if config.comparison_set.matching_rules.max_controls_per_wontfix <= 0:
        raise ConfigError("comparison_set.matching_rules.max_controls_per_wontfix must be > 0.")

    if config.storage.raw_format not in {"json"}:
        raise ConfigError("storage.raw_format must currently be 'json'.")
    if config.storage.processed_format not in {"parquet", "csv"}:
        raise ConfigError("storage.processed_format must be 'parquet' or 'csv'.")
    if config.storage.processed_merge_mode not in {"single_parquet", "partitioned_dataset"}:
        raise ConfigError("storage.processed_merge_mode must be 'single_parquet' or 'partitioned_dataset'.")

    if config.issue_selection.include_issues and not config.issue_selection.states:
        raise ConfigError("issue_selection.states cannot be empty when include_issues is true.")

    _validate_date_string(config.study.date, "study.date")
    _validate_date_string(config.study_windows.issue_collection.start_date, "study_windows.issue_collection.start_date")
    _validate_date_string(config.study_windows.issue_collection.end_date, "study_windows.issue_collection.end_date")
    _validate_date_string(config.study_windows.participation_analysis.start_date, "study_windows.participation_analysis.start_date")
    _validate_date_string(config.study_windows.participation_analysis.end_date, "study_windows.participation_analysis.end_date")
    _validate_date_string(config.repo_selection.activity_definition.cutoff_date, "repo_selection.activity_definition.cutoff_date")

    if config.study_windows.issue_collection.start_date > config.study_windows.issue_collection.end_date:
        raise ConfigError("study_windows.issue_collection.start_date must be <= end_date.")

    if config.issue_extraction.max_issue_pages_per_repo_per_state <= 0:
        raise ConfigError("issue_extraction.max_issue_pages_per_repo_per_state must be > 0.")
    if config.issue_extraction.max_comment_pages_per_issue <= 0:
        raise ConfigError("issue_extraction.max_comment_pages_per_issue must be > 0.")
    if config.issue_extraction.write_batch_size <= 0:
        raise ConfigError("issue_extraction.write_batch_size must be > 0.")
    if config.issue_extraction.resume_mode not in {"checkpoint_only", "raw_or_checkpoint", "fresh"}:
        raise ConfigError("issue_extraction.resume_mode must be one of checkpoint_only, raw_or_checkpoint, fresh.")
    if config.issue_extraction.search_max_results_per_shard <= 0:
        raise ConfigError("issue_extraction.search_max_results_per_shard must be > 0.")

    if config.issue_extraction.search_max_shard_splits <= 0:
        raise ConfigError("issue_extraction.search_max_shard_splits must be > 0.")

    if (config.issue_extraction.max_search_pages_per_shard is not None and config.issue_extraction.max_search_pages_per_shard <= 0):
        raise ConfigError("issue_extraction.max_search_pages_per_shard must be > 0 when provided.")

    allowed_visibility = {"public"}
    if config.repo_selection.inclusion_criteria.visibility not in allowed_visibility:
        raise ConfigError(
            f"repo_selection.inclusion_criteria.visibility must be one of {sorted(allowed_visibility)}."
        )

    allowed_time_units = {"month", "quarter", "year"}
    if config.study_windows.participation_analysis.time_window_unit not in allowed_time_units:
        raise ConfigError(f"study_windows.participation_analysis.time_window_unit must be one of {sorted(allowed_time_units)}.")

    allowed_identity_scopes = {"repository_local"}
    if config.identity_resolution.scope not in allowed_identity_scopes:
        raise ConfigError(f"identity_resolution.scope must be one of {sorted(allowed_identity_scopes)}.")
    allowed_identity_match_methods = {"github_login_exact", "email_exact", "normalized_name_exact"}
    for value in config.identity_resolution.matching_priority:
        if value not in allowed_identity_match_methods:
            raise ConfigError(f"identity_resolution.matching_priority values must be in {sorted(allowed_identity_match_methods)}.")
    if getattr(config.identity_resolution, "write_batch_size", 1) <= 0:
        raise ConfigError("identity_resolution.write_batch_size must be > 0.")
    if getattr(config.identity_resolution, "max_repos_per_run",
               None) is not None and config.identity_resolution.max_repos_per_run <= 0:
        raise ConfigError("identity_resolution.max_repos_per_run must be > 0 when provided.")
    if getattr(config.identity_resolution, "resume_mode", "fresh") not in {"fresh", "checkpoint_only", "raw_or_checkpoint"}:
        raise ConfigError("identity_resolution.resume_mode must be 'fresh', 'checkpoint_only', or 'raw_or_checkpoint'.")

    if not config.label_normalization.outcome_labels.wontfix.variants:
        raise ConfigError("At least one WONTFIX label variant must be provided.")
    if not config.label_normalization.outcome_labels.invalid.variants:
        raise ConfigError("At least one INVALID label variant must be provided.")

    if config.git_history_extraction.enabled and not config.git_history_extraction.clone_root:
        raise ConfigError("git_history_extraction.clone_root is required when git history extraction is enabled.")
    allowed_commit_message_modes = {"full", "subject_only", "none"}
    if config.git_history_extraction.commit_message_mode not in allowed_commit_message_modes:
        raise ConfigError(f"git_history_extraction.commit_message_mode must be one of {sorted(allowed_commit_message_modes)}.")
    allowed_fast_mode_windows = {
        "issue_collection",
        "participation_analysis",
        "explicit_history_dates",
    }
    if config.git_history_extraction.fast_mode_date_window not in allowed_fast_mode_windows:
        raise ConfigError(f"git_history_extraction.fast_mode_date_window must be one of {sorted(allowed_fast_mode_windows)}.")
    if (config.git_history_extraction.max_commit_message_chars is not None
        and config.git_history_extraction.max_commit_message_chars <= 0):
        raise ConfigError("git_history_extraction.max_commit_message_chars must be > 0 when provided.")

def resolve_config_paths(config):
    resolved = copy.deepcopy(config)
    base = Path(resolved.paths.project_root).resolve()

    resolved.paths.project_root = str(base)
    resolved.paths.data_root = _resolve_path(base, resolved.paths.data_root)
    resolved.paths.raw_root = _resolve_path(base, resolved.paths.raw_root)
    resolved.paths.processed_root = _resolve_path(base, resolved.paths.processed_root)
    resolved.paths.linked_root = _resolve_path(base, resolved.paths.linked_root)
    resolved.paths.features_root = _resolve_path(base, resolved.paths.features_root)
    resolved.paths.final_root = _resolve_path(base, resolved.paths.final_root)
    resolved.paths.logs_root = _resolve_path(base, resolved.paths.logs_root)
    resolved.paths.config_root = _resolve_path(base, resolved.paths.config_root)

    resolved.git_history_extraction.clone_root = _resolve_path(base, resolved.git_history_extraction.clone_root)

    resolved.outputs.repo_candidate_list = _resolve_path(base, resolved.outputs.repo_candidate_list)
    resolved.outputs.repo_included_list = _resolve_path(base, resolved.outputs.repo_included_list)
    resolved.outputs.repositories_table = _resolve_path(base, resolved.outputs.repositories_table)
    resolved.outputs.issues_table = _resolve_path(base, resolved.outputs.issues_table)
    resolved.outputs.issue_comments_table = _resolve_path(base, resolved.outputs.issue_comments_table)
    resolved.outputs.pull_requests_table = _resolve_path(base, resolved.outputs.pull_requests_table)
    resolved.outputs.commits_table = _resolve_path(base, resolved.outputs.commits_table)
    resolved.outputs.commit_files_table = _resolve_path(base, resolved.outputs.commit_files_table)
    resolved.outputs.comparison_issue_set_table = _resolve_path(base, resolved.outputs.comparison_issue_set_table)
    resolved.outputs.wontfix_issue_set_table = _resolve_path(base, resolved.outputs.wontfix_issue_set_table)
    resolved.outputs.contributor_identity_table = _resolve_path(base, resolved.outputs.contributor_identity_table)
    resolved.outputs.contributor_identity_clusters_table = _resolve_path(base, resolved.outputs.contributor_identity_clusters_table)
    resolved.outputs.issue_pr_links_table = _resolve_path(base, resolved.outputs.issue_pr_links_table)
    resolved.outputs.pr_commit_links_table = _resolve_path(base, resolved.outputs.pr_commit_links_table)
    resolved.outputs.issue_file_links_table = _resolve_path(base, resolved.outputs.issue_file_links_table)
    resolved.outputs.extraction_summary_csv = _resolve_path(base, resolved.outputs.extraction_summary_csv)
    resolved.outputs.run_manifest_json = _resolve_path(base, resolved.outputs.run_manifest_json)
    resolved.outputs.resolved_config_snapshot_yaml = _resolve_path(base, resolved.outputs.resolved_config_snapshot_yaml)
    resolved.outputs.comparison_issue_qa_summary_csv = _resolve_path(base, resolved.outputs.comparison_issue_qa_summary_csv)
    resolved.outputs.issues_resolved_table = _resolve_path(base, resolved.outputs.issues_resolved_table)
    resolved.outputs.issue_comments_resolved_table = _resolve_path(base, resolved.outputs.issue_comments_resolved_table)
    resolved.outputs.pull_requests_resolved_table = _resolve_path(base, resolved.outputs.pull_requests_resolved_table)
    resolved.outputs.commits_resolved_table = _resolve_path(base, resolved.outputs.commits_resolved_table)

    resolved.checkpointing.checkpoint_dir = _resolve_path(base, resolved.checkpointing.checkpoint_dir)
    resolved.logging.extraction_log_dir = _resolve_path(base, resolved.logging.extraction_log_dir)
    resolved.logging.normalization_log_dir = _resolve_path(base, resolved.logging.normalization_log_dir)
    resolved.logging.linkage_log_dir = _resolve_path(base, resolved.logging.linkage_log_dir)
    resolved.logging.qa_log_dir = _resolve_path(base, resolved.logging.qa_log_dir)
    resolved.quality_assurance.summary_report_path = _resolve_path(base, resolved.quality_assurance.summary_report_path)

    return resolved

from pathlib import Path

def _ensure_directory(path_str):
    path = Path(path_str)

    # If path has a suffix, treat it as a file → create parent dir
    if path.suffix:
        path = path.parent

    path.mkdir(parents=True, exist_ok=True)


def ensure_project_directories(config):
    # required path roots must be made
    for field_name in vars(config.paths):
        path_str = getattr(config.paths, field_name)
        if path_str:
            _ensure_directory(path_str)

    # output paths can include files, so only ensure dir if not a filename
    for field_name in vars(config.outputs):
        path_str = getattr(config.outputs, field_name)
        if path_str:
            _ensure_directory(path_str)

    # logging directories
    for field_name in vars(config.logging):
        path_str = getattr(config.logging, field_name)
        if isinstance(path_str, str) and ("log" in field_name or "dir" in field_name):
            _ensure_directory(path_str)

    # checkpoint directory
    if config.checkpointing.checkpoint_dir:
        _ensure_directory(config.checkpointing.checkpoint_dir)

def config_to_dict(config):
    """
    little helper for debugging. converts the dataclass tree into plain dicts.
    """
    if hasattr(config, "__dataclass_fields__"):
        result = {}
        for field_name in config.__dataclass_fields__:
            result[field_name] = config_to_dict(getattr(config, field_name))
        return result

    if isinstance(config, list):
        return [config_to_dict(item) for item in config]

    if isinstance(config, dict):
        return {key: config_to_dict(value) for key, value in config.items()}

    return config

def write_resolved_config_snapshot(config):
    output_path = Path(config.outputs.resolved_config_snapshot_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False)

if __name__ == "__main__":
    config = load_study_config("config/study_config.yaml")
    ensure_project_directories(config)

    print("Loaded study config successfully.")
    print(f"Study name: {config.study.name}")
    print(f"Version: {config.study.version}")
    print(f"Repo candidate output: {config.outputs.repo_candidate_list}")