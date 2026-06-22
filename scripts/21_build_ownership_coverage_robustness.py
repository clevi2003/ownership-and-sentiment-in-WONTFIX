#!/usr/bin/env python3
"""
Build ownership coverage and high-bilateral-coverage robustness outputs for the
WONTFIX matched-set FE revision.

This script is intentionally narrow: it focuses on RQ2 ownership coverage and
repo-level participant-role robustness. It can be run after scripts/13 and 20,
or independently after scripts/13 if you only need ownership robustness outputs.

Default usage:
    python scripts/21_build_ownership_coverage_robustness.py

Typical outputs:
    outputs/ownership_coverage_robustness/ownership_coverage_by_analysis_set.csv
    outputs/ownership_coverage_robustness/ownership_coverage_by_repo_and_analysis_set.csv
    outputs/ownership_coverage_robustness/ownership_bilateral_coverage_repo_selection.csv
    data/final/analysis_dataset_rq2_high_repo_participant_coverage.parquet
    outputs/ownership_coverage_robustness/ownership_repo_primary_full_results.csv
    outputs/ownership_coverage_robustness/ownership_repo_primary_high_coverage_results.csv
    outputs/ownership_coverage_robustness/ownership_full_vs_high_bilateral_coverage_results.csv
    outputs/ownership_coverage_robustness/ownership_coverage_robustness_report.md
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
except Exception:  # pragma: no cover
    smf = None

try:
    from statsmodels.stats.multitest import multipletests
except Exception:  # pragma: no cover
    multipletests = None

SCRIPT_VERSION = "ownership_coverage_robustness_v2026_06_21"

DEFAULT_RQ2_DATASET = "data/final/analysis_dataset_rq2.parquet"
DEFAULT_OUTPUT_DIR = "outputs/ownership_coverage_robustness"
DEFAULT_SUBSET_DATASET_OUT = "data/final/analysis_dataset_rq2_high_repo_participant_coverage.parquet"
DEFAULT_COVERAGE_COLUMN = "has_repo_participant_role_signal"
DEFAULT_THRESHOLDS = "0.20,0.30,0.40"
DEFAULT_MAIN_THRESHOLD = 0.30

MIN_MODEL_N = 40
MIN_GROUP_N = 10
MIN_EVENT_COUNT = 10


@dataclass(frozen=True)
class OutcomeSpec:
    outcome: str
    label: str
    outcome_type: str
    denominator_filter: str | None = None


OWNERSHIP_REPO_PRIMARY_OUTCOMES = [
    OutcomeSpec(
        "issue_author_is_pre_issue_repo_contributor",
        "Issue author was prior repo contributor",
        "binary",
    ),
    OutcomeSpec(
        "issue_author_is_pre_issue_major_repo_contributor",
        "Issue author was major prior repo contributor",
        "binary",
    ),
    OutcomeSpec(
        "any_commenter_is_pre_issue_repo_contributor",
        "Any commenter was prior repo contributor",
        "binary",
    ),
    OutcomeSpec(
        "any_commenter_is_pre_issue_major_repo_contributor",
        "Any commenter was major prior repo contributor",
        "binary",
    ),
    OutcomeSpec(
        "share_commenters_pre_issue_repo_contributors",
        "Share of commenters who were prior repo contributors",
        "proportion",
    ),
    OutcomeSpec(
        "share_commenters_pre_issue_major_repo_contributors",
        "Share of commenters who were major prior repo contributors",
        "proportion",
    ),
    OutcomeSpec(
        "top_commenter_is_pre_issue_repo_contributor",
        "Top commenter was prior repo contributor",
        "binary",
    ),
    OutcomeSpec(
        "top_commenter_is_pre_issue_major_repo_contributor",
        "Top commenter was major prior repo contributor",
        "binary",
    ),
    OutcomeSpec(
        "commenter_count_pre_issue_repo_contributors",
        "Prior repo contributor commenters",
        "count",
    ),
]

COVERAGE_FAMILIES = {
    "repo_participant_roles": "has_repo_participant_role_signal",
    "file_participant_roles": "participant_role_file_features_applicable",
    "direct_issue_linked_ownership": "has_direct_issue_linked_ownership_features",
    "continuity": "has_continuity_signal",
    "issue_author_resolved": "issue_author_has_resolved_key",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ownership coverage tables and high-bilateral-coverage matched-set robustness outputs."
    )
    parser.add_argument("--rq2-dataset", default=DEFAULT_RQ2_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--subset-dataset-out", default=DEFAULT_SUBSET_DATASET_OUT)
    parser.add_argument("--coverage-column", default=DEFAULT_COVERAGE_COLUMN)
    parser.add_argument("--main-threshold", type=float, default=DEFAULT_MAIN_THRESHOLD)
    parser.add_argument(
        "--thresholds",
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated bilateral coverage thresholds to summarize, e.g. 0.20,0.30,0.40.",
    )
    parser.add_argument("--min-model-n", type=int, default=MIN_MODEL_N)
    parser.add_argument("--min-group-n", type=int, default=MIN_GROUP_N)
    parser.add_argument("--min-event-count", type=int, default=MIN_EVENT_COUNT)
    parser.add_argument(
        "--write-subset-only",
        action="store_true",
        help="Only write coverage tables and high-coverage subset; skip robustness model fitting.",
    )
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input table does not exist: {p}")
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix == ".json":
        return pd.read_json(p)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(p, lines=True)
    raise ValueError(f"Unsupported table format for {p}")


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


def write_markdown(text: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    else:
        df.to_parquet(p, index=False)
    return p


def parse_thresholds(value: str) -> list[float]:
    out = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return sorted(set(out))


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>", "nat"}:
        return None
    return text


def normalize_analysis_set(value: Any) -> str:
    text = clean_text(value)
    if text is None:
        return "missing"
    lowered = text.lower().replace("-", "_").replace(" ", "_")
    lowered = lowered.replace("'", "").replace("’", "")
    if lowered in {"wontfix", "wont_fix", "won_t_fix"}:
        return "wontfix"
    if lowered in {"comparison", "control", "controls", "non_wontfix", "nonwontfix", "matched_control"}:
        return "comparison"
    return lowered


def to_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def normalize_rq2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "repo_full_name" not in out.columns:
        raise KeyError("RQ2 dataset must include repo_full_name")
    if "analysis_set" not in out.columns:
        raise KeyError("RQ2 dataset must include analysis_set")
    if "matched_set_id" not in out.columns:
        raise KeyError("RQ2 dataset must include matched_set_id")

    out["repo_full_name"] = out["repo_full_name"].astype(str)
    out["analysis_set"] = out["analysis_set"].apply(normalize_analysis_set)
    out["is_wontfix"] = (out["analysis_set"] == "wontfix").astype(int)
    out["matched_set_id"] = out["matched_set_id"].apply(clean_text)
    out["matched_set_fe_id"] = out["repo_full_name"].astype(str) + "::" + out["matched_set_id"].astype(str)
    out.loc[out["matched_set_id"].isna(), "matched_set_fe_id"] = pd.NA
    return out


def flag_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0, index=df.index, dtype="int64")
    return to_numeric(df[column]).fillna(0).astype(int)


def build_coverage_tables(rq2: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    repo_rows = []

    for family, column in COVERAGE_FAMILIES.items():
        if column not in rq2.columns:
            overall_rows.append({
                "analysis_set": "missing",
                "total_issues": 0,
                "covered_issues": 0,
                "coverage_rate": np.nan,
                "coverage_family": family,
                "coverage_column": column,
                "status": "missing_column",
            })
            continue

        x = rq2.copy()
        x["_covered"] = flag_series(x, column)

        overall = (
            x.groupby("analysis_set", dropna=False)["_covered"]
            .agg(total_issues="count", covered_issues="sum", coverage_rate="mean")
            .reset_index()
        )
        overall["coverage_family"] = family
        overall["coverage_column"] = column
        overall["status"] = "ok"
        overall_rows.append(overall)

        repo = (
            x.groupby(["repo_full_name", "analysis_set"], dropna=False)["_covered"]
            .agg(total_issues="count", covered_issues="sum", coverage_rate="mean")
            .reset_index()
        )
        repo["coverage_family"] = family
        repo["coverage_column"] = column
        repo["status"] = "ok"
        repo_rows.append(repo)

    overall_df = pd.concat(overall_rows, ignore_index=True) if overall_rows else pd.DataFrame()
    repo_df = pd.concat(repo_rows, ignore_index=True) if repo_rows else pd.DataFrame()

    write_csv(overall_df, out_dir / "ownership_coverage_by_analysis_set.csv")
    write_csv(repo_df, out_dir / "ownership_coverage_by_repo_and_analysis_set.csv")
    return overall_df, repo_df


def build_bilateral_repo_selection(
    rq2: pd.DataFrame,
    coverage_column: str,
    thresholds: list[float],
    main_threshold: float,
    out_dir: Path,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    if coverage_column not in rq2.columns:
        raise KeyError(f"Coverage column not found in RQ2 dataset: {coverage_column}")

    x = rq2.copy()
    x["_covered"] = flag_series(x, coverage_column)

    repo_cov = (
        x.groupby(["repo_full_name", "analysis_set"], dropna=False)["_covered"]
        .agg(total_issues="count", covered_issues="sum", coverage_rate="mean")
        .reset_index()
    )

    rate_wide = repo_cov.pivot(index="repo_full_name", columns="analysis_set", values="coverage_rate")
    n_wide = repo_cov.pivot(index="repo_full_name", columns="analysis_set", values="total_issues")
    covered_wide = repo_cov.pivot(index="repo_full_name", columns="analysis_set", values="covered_issues")

    rows = []
    for repo in rate_wide.index:
        row = {
            "repo_full_name": repo,
            "coverage_column": coverage_column,
            "wontfix_coverage_rate": rate_wide.loc[repo].get("wontfix", np.nan),
            "comparison_coverage_rate": rate_wide.loc[repo].get("comparison", np.nan),
            "wontfix_issues": n_wide.loc[repo].get("wontfix", 0) if repo in n_wide.index else 0,
            "comparison_issues": n_wide.loc[repo].get("comparison", 0) if repo in n_wide.index else 0,
            "wontfix_covered_issues": covered_wide.loc[repo].get("wontfix", 0) if repo in covered_wide.index else 0,
            "comparison_covered_issues": covered_wide.loc[repo].get("comparison", 0) if repo in covered_wide.index else 0,
        }
        for threshold in thresholds:
            flag = f"passes_{int(round(threshold * 100))}pct_bilateral"
            row[flag] = (
                pd.notna(row["wontfix_coverage_rate"])
                and pd.notna(row["comparison_coverage_rate"])
                and float(row["wontfix_coverage_rate"]) >= threshold
                and float(row["comparison_coverage_rate"]) >= threshold
            )
        rows.append(row)

    selection = pd.DataFrame(rows).sort_values(
        ["wontfix_coverage_rate", "comparison_coverage_rate", "repo_full_name"],
        ascending=[False, False, True],
    )

    for threshold in thresholds:
        flag = f"passes_{int(round(threshold * 100))}pct_bilateral"
        passing_repos = selection.loc[selection[flag], "repo_full_name"].tolist()
        subset = rq2[rq2["repo_full_name"].isin(passing_repos)]
        selection.loc[:, f"threshold_{int(round(threshold * 100))}pct_repos"] = int(len(passing_repos))
        selection.loc[:, f"threshold_{int(round(threshold * 100))}pct_rows"] = int(len(subset))
        selection.loc[:, f"threshold_{int(round(threshold * 100))}pct_matched_sets"] = int(subset["matched_set_fe_id"].dropna().nunique())

    main_flag = f"passes_{int(round(main_threshold * 100))}pct_bilateral"
    if main_flag not in selection.columns:
        raise ValueError(f"Main threshold {main_threshold} not present in thresholds {thresholds}")

    main_repos = selection.loc[selection[main_flag], "repo_full_name"].tolist()
    subset = rq2[rq2["repo_full_name"].isin(main_repos)].copy()

    write_csv(selection, out_dir / "ownership_bilateral_coverage_repo_selection.csv")
    return selection, main_repos, subset


def valid_matched_set_ids(df: pd.DataFrame) -> set[str]:
    if df.empty:
        return set()
    grouped = df.groupby("matched_set_fe_id", dropna=True)["analysis_set"].agg(
        has_wontfix=lambda s: bool((s == "wontfix").any()),
        has_comparison=lambda s: bool((s == "comparison").any()),
    )
    valid = grouped[grouped["has_wontfix"] & grouped["has_comparison"]]
    return set(valid.index.astype(str))


def prepare_model_data(
    df: pd.DataFrame,
    spec: OutcomeSpec,
    min_model_n: int,
    min_group_n: int,
    min_event_count: int,
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    readiness = {
        "family": "ownership_repo_primary",
        "tier": "primary",
        "outcome": spec.outcome,
        "outcome_label": spec.label,
        "outcome_type": spec.outcome_type,
        "rows_before_filter": int(len(df)),
        "rows_nonmissing_outcome": 0,
        "rows_after_valid_set_filter": 0,
        "n_wontfix": 0,
        "n_comparison": 0,
        "n_matched_sets": 0,
        "status": "ok",
        "skip_reason": "",
    }

    if df.empty:
        readiness.update(status="empty_dataset", skip_reason="dataset_has_no_rows")
        return None, readiness
    if spec.outcome not in df.columns:
        readiness.update(status="missing_outcome", skip_reason="outcome_column_missing")
        return None, readiness

    model_df = df.copy()
    model_df["_outcome"] = to_numeric(model_df[spec.outcome])
    model_df = model_df[model_df["_outcome"].notna()].copy()
    readiness["rows_nonmissing_outcome"] = int(len(model_df))

    model_df = model_df[model_df["matched_set_fe_id"].notna()].copy()
    valid_ids = valid_matched_set_ids(model_df)
    model_df = model_df[model_df["matched_set_fe_id"].astype(str).isin(valid_ids)].copy()
    readiness["rows_after_valid_set_filter"] = int(len(model_df))
    readiness["n_matched_sets"] = int(model_df["matched_set_fe_id"].nunique()) if not model_df.empty else 0
    readiness["n_wontfix"] = int((model_df["analysis_set"] == "wontfix").sum()) if not model_df.empty else 0
    readiness["n_comparison"] = int((model_df["analysis_set"] == "comparison").sum()) if not model_df.empty else 0

    if len(model_df) < min_model_n:
        readiness.update(status="insufficient_data", skip_reason="n_below_min_model_n")
        return None, readiness
    if readiness["n_wontfix"] < min_group_n or readiness["n_comparison"] < min_group_n:
        readiness.update(status="insufficient_group_n", skip_reason="group_n_below_min_group_n")
        return None, readiness

    if spec.outcome_type in {"binary", "proportion"}:
        w_events = float(model_df.loc[model_df["analysis_set"] == "wontfix", "_outcome"].sum())
        c_events = float(model_df.loc[model_df["analysis_set"] == "comparison", "_outcome"].sum())
        readiness["event_count_wontfix"] = w_events
        readiness["event_count_comparison"] = c_events
        if spec.outcome_type == "binary" and (w_events < min_event_count or c_events < min_event_count):
            readiness.update(status="low_event_count", skip_reason="binary_event_count_below_min_event_count")
            return None, readiness

    if model_df["_outcome"].nunique(dropna=True) < 2:
        readiness.update(status="constant_outcome", skip_reason="outcome_has_no_variation")
        return None, readiness

    return model_df, readiness


def compute_descriptive_effects(model_df: pd.DataFrame) -> dict[str, float]:
    w = model_df.loc[model_df["analysis_set"] == "wontfix", "_outcome"].astype(float)
    c = model_df.loc[model_df["analysis_set"] == "comparison", "_outcome"].astype(float)
    w_mean = float(w.mean()) if len(w) else np.nan
    c_mean = float(c.mean()) if len(c) else np.nan
    w_sd = float(w.std(ddof=1)) if len(w) > 1 else np.nan
    c_sd = float(c.std(ddof=1)) if len(c) > 1 else np.nan
    pooled = np.nan
    if len(w) > 1 and len(c) > 1 and np.isfinite(w_sd) and np.isfinite(c_sd):
        denom = max(len(w) + len(c) - 2, 1)
        pooled = math.sqrt(((len(w) - 1) * w_sd**2 + (len(c) - 1) * c_sd**2) / denom)
    d = (w_mean - c_mean) / pooled if pooled and np.isfinite(pooled) and pooled != 0 else np.nan
    return {
        "raw_wontfix_mean": w_mean,
        "raw_comparison_mean": c_mean,
        "raw_difference": w_mean - c_mean if np.isfinite(w_mean) and np.isfinite(c_mean) else np.nan,
        "raw_wontfix_sd": w_sd,
        "raw_comparison_sd": c_sd,
        "standardized_effect_raw_d": d,
    }


def fit_matched_set_fe(model_df: pd.DataFrame, spec: OutcomeSpec, sample_label: str) -> dict[str, Any]:
    base = {
        "sample": sample_label,
        "family": "ownership_repo_primary",
        "tier": "primary",
        "outcome": spec.outcome,
        "outcome_label": spec.label,
        "outcome_type": spec.outcome_type,
        "model_type": "matched_set_fe",
        "covariance": "cluster_matched_set_fe_id",
        "model_status": "ok",
        "error_message": "",
        "n_issues": int(len(model_df)),
        "n_matched_sets": int(model_df["matched_set_fe_id"].nunique()),
        "n_repos": int(model_df["repo_full_name"].nunique()),
        "n_wontfix": int((model_df["analysis_set"] == "wontfix").sum()),
        "n_comparison": int((model_df["analysis_set"] == "comparison").sum()),
    }
    try:
        if smf is None:
            raise RuntimeError("statsmodels is not installed; cannot fit matched-set FE models")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.ols("_outcome ~ is_wontfix + C(matched_set_fe_id)", data=model_df)
            fit = model.fit(cov_type="cluster", cov_kwds={"groups": model_df["matched_set_fe_id"]})
        if "is_wontfix" not in fit.params.index:
            raise RuntimeError("Model did not estimate is_wontfix coefficient")
        conf = fit.conf_int()
        coef = float(fit.params["is_wontfix"])
        out = dict(base)
        out.update({
            "coef_wontfix": coef,
            "std_error": float(fit.bse["is_wontfix"]),
            "ci_low": float(conf.loc["is_wontfix", 0]),
            "ci_high": float(conf.loc["is_wontfix", 1]),
            "p_value": float(fit.pvalues["is_wontfix"]),
            "df_resid": float(fit.df_resid),
            "rsquared": float(getattr(fit, "rsquared", np.nan)),
            "percentage_point_effect": 100.0 * coef if spec.outcome_type in {"binary", "proportion"} else np.nan,
        })
        out.update(compute_descriptive_effects(model_df))
        return out
    except Exception as exc:
        out = dict(base)
        out.update({
            "model_status": "model_failed",
            "error_message": str(exc),
            "coef_wontfix": np.nan,
            "std_error": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "df_resid": np.nan,
            "rsquared": np.nan,
            "percentage_point_effect": np.nan,
        })
        out.update(compute_descriptive_effects(model_df) if model_df is not None and not model_df.empty else {})
        return out


def failed_model_row(spec: OutcomeSpec, readiness: dict[str, Any], sample_label: str) -> dict[str, Any]:
    return {
        "sample": sample_label,
        "family": "ownership_repo_primary",
        "tier": "primary",
        "outcome": spec.outcome,
        "outcome_label": spec.label,
        "outcome_type": spec.outcome_type,
        "model_type": "matched_set_fe",
        "covariance": "cluster_matched_set_fe_id",
        "model_status": readiness.get("status", "skipped"),
        "error_message": readiness.get("skip_reason", ""),
        "n_issues": readiness.get("rows_after_valid_set_filter", 0),
        "n_matched_sets": readiness.get("n_matched_sets", 0),
        "n_repos": np.nan,
        "n_wontfix": readiness.get("n_wontfix", 0),
        "n_comparison": readiness.get("n_comparison", 0),
        "coef_wontfix": np.nan,
        "std_error": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": np.nan,
        "df_resid": np.nan,
        "rsquared": np.nan,
        "percentage_point_effect": np.nan,
    }


def add_fdr(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    out["p_value_fdr_bh"] = np.nan
    out["reject_fdr_bh_05"] = False
    if out.empty or multipletests is None or "p_value" not in out.columns:
        return out
    valid = out["p_value"].notna() & out["model_status"].eq("ok")
    if int(valid.sum()) == 0:
        return out
    reject, p_adj, _, _ = multipletests(out.loc[valid, "p_value"].astype(float), method="fdr_bh")
    out.loc[valid, "p_value_fdr_bh"] = p_adj
    out.loc[valid, "reject_fdr_bh_05"] = reject
    return out


def run_ownership_models(
    df: pd.DataFrame,
    sample_label: str,
    min_model_n: int,
    min_group_n: int,
    min_event_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    readiness_rows = []
    result_rows = []
    for spec in OWNERSHIP_REPO_PRIMARY_OUTCOMES:
        model_df, readiness = prepare_model_data(
            df,
            spec,
            min_model_n=min_model_n,
            min_group_n=min_group_n,
            min_event_count=min_event_count,
        )
        readiness["sample"] = sample_label
        readiness_rows.append(readiness)
        if model_df is None or readiness.get("status") != "ok":
            result_rows.append(failed_model_row(spec, readiness, sample_label))
        else:
            result_rows.append(fit_matched_set_fe(model_df, spec, sample_label))
    return pd.DataFrame(readiness_rows), add_fdr(pd.DataFrame(result_rows))


def compare_full_vs_subset(full_results: pd.DataFrame, subset_results: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["family", "tier", "outcome", "outcome_label", "outcome_type"]
    full = full_results.copy()
    subset = subset_results.copy()
    compare = full.merge(subset, on=key_cols, how="outer", suffixes=("_full", "_high_coverage"))
    compare["same_direction_flag"] = (
        np.sign(to_numeric(compare.get("coef_wontfix_full")))
        == np.sign(to_numeric(compare.get("coef_wontfix_high_coverage")))
    )
    compare.loc[
        compare["coef_wontfix_full"].isna() | compare["coef_wontfix_high_coverage"].isna(),
        "same_direction_flag",
    ] = False
    compare["abs_coef_difference"] = (
        to_numeric(compare.get("coef_wontfix_high_coverage"))
        - to_numeric(compare.get("coef_wontfix_full"))
    ).abs()
    compare["relative_coef_ratio"] = (
        to_numeric(compare.get("coef_wontfix_high_coverage"))
        / to_numeric(compare.get("coef_wontfix_full"))
    )
    return compare


def build_threshold_summary(selection: pd.DataFrame, rq2: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        flag = f"passes_{int(round(threshold * 100))}pct_bilateral"
        repos = selection.loc[selection[flag], "repo_full_name"].tolist() if flag in selection.columns else []
        subset = rq2[rq2["repo_full_name"].isin(repos)]
        rows.append({
            "threshold": threshold,
            "repos": int(len(repos)),
            "rows": int(len(subset)),
            "wontfix_rows": int((subset["analysis_set"] == "wontfix").sum()) if not subset.empty else 0,
            "comparison_rows": int((subset["analysis_set"] == "comparison").sum()) if not subset.empty else 0,
            "matched_sets": int(subset["matched_set_fe_id"].dropna().nunique()) if not subset.empty else 0,
        })
    return pd.DataFrame(rows)


def render_report(
    overall: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    main_threshold: float,
    main_repos: list[str],
    subset: pd.DataFrame,
    comparison: pd.DataFrame,
) -> str:
    def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
        if df is None or df.empty:
            return "_No data available._"
        out = df.head(max_rows).copy()
        for col in out.columns:
            if pd.api.types.is_float_dtype(out[col]):
                out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        return out.to_markdown(index=False)

    repo_cov = overall[overall["coverage_family"].eq("repo_participant_roles")].copy()
    direct = overall[overall["coverage_family"].eq("direct_issue_linked_ownership")].copy()
    continuity = overall[overall["coverage_family"].eq("continuity")].copy()

    ok_compare = comparison[
        comparison.get("model_status_full", pd.Series(index=comparison.index, dtype="object")).eq("ok")
        & comparison.get("model_status_high_coverage", pd.Series(index=comparison.index, dtype="object")).eq("ok")
    ] if not comparison.empty else pd.DataFrame()
    same_direction_n = int(ok_compare["same_direction_flag"].sum()) if not ok_compare.empty else 0
    ok_n = int(len(ok_compare)) if not ok_compare.empty else 0

    lines = [
        "# Ownership coverage robustness report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Script version: `{SCRIPT_VERSION}`",
        "",
        "## Coverage overview",
        "",
        md_table(overall[[c for c in ["coverage_family", "analysis_set", "total_issues", "covered_issues", "coverage_rate", "coverage_column"] if c in overall.columns]]),
        "",
        "## Bilateral repo threshold summary",
        "",
        md_table(threshold_summary),
        "",
        f"Main robustness threshold: **{main_threshold:.0%} bilateral repo participant-role coverage**.",
        f"Repos retained: **{len(main_repos)}**.",
        f"Subset rows: **{len(subset)}**.",
        f"Subset matched sets: **{int(subset['matched_set_fe_id'].dropna().nunique()) if not subset.empty else 0}**.",
        "",
        "## Full vs high-coverage ownership results",
        "",
        f"Same-direction matched-set FE ownership results among successfully estimated paired models: **{same_direction_n}/{ok_n}**.",
        "",
        md_table(comparison[[c for c in [
            "outcome_label",
            "coef_wontfix_full",
            "p_value_full",
            "n_issues_full",
            "n_matched_sets_full",
            "coef_wontfix_high_coverage",
            "p_value_high_coverage",
            "n_issues_high_coverage",
            "n_matched_sets_high_coverage",
            "same_direction_flag",
        ] if c in comparison.columns]], max_rows=40),
        "",
        "## Interpretation guardrails",
        "",
        "- Repo-level participant-role features are the primary ownership-adjacent family because they have the broadest coverage.",
        "- File-level participant-role features are secondary because they depend on linked-file applicability.",
        "- Direct issue-linked ownership and continuity are descriptive/conditional because they depend on PR/commit/file evidence that is structurally sparse for WONTFIX issues.",
        "- The high-coverage subset is a robustness check for repo-level participant-role ownership results, not a replacement for the full matched-set FE model.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    thresholds = parse_thresholds(args.thresholds)
    if args.main_threshold not in thresholds:
        thresholds = sorted(set(thresholds + [args.main_threshold]))

    print(f"Running {SCRIPT_VERSION}")
    print(f"RQ2 dataset: {args.rq2_dataset}")
    print(f"Output directory: {out_dir}")

    rq2 = normalize_rq2(read_table(args.rq2_dataset))

    overall, repo_coverage = build_coverage_tables(rq2, out_dir)
    selection, main_repos, subset = build_bilateral_repo_selection(
        rq2,
        coverage_column=args.coverage_column,
        thresholds=thresholds,
        main_threshold=args.main_threshold,
        out_dir=out_dir,
    )
    threshold_summary = build_threshold_summary(selection, rq2, thresholds)
    write_csv(threshold_summary, out_dir / "ownership_bilateral_coverage_threshold_summary.csv")

    subset_path = write_table(subset, args.subset_dataset_out)
    print(f"Wrote high-coverage subset: {subset_path}")

    if args.write_subset_only:
        report = render_report(overall, threshold_summary, args.main_threshold, main_repos, subset, pd.DataFrame())
        write_markdown(report, out_dir / "ownership_coverage_robustness_report.md")
        return

    full_readiness, full_results = run_ownership_models(
        rq2,
        sample_label="full",
        min_model_n=args.min_model_n,
        min_group_n=args.min_group_n,
        min_event_count=args.min_event_count,
    )
    subset_readiness, subset_results = run_ownership_models(
        subset,
        sample_label=f"high_bilateral_coverage_{int(round(args.main_threshold * 100))}pct",
        min_model_n=args.min_model_n,
        min_group_n=args.min_group_n,
        min_event_count=args.min_event_count,
    )

    write_csv(full_readiness, out_dir / "ownership_repo_primary_full_model_readiness.csv")
    write_csv(subset_readiness, out_dir / "ownership_repo_primary_high_coverage_model_readiness.csv")
    write_csv(full_results, out_dir / "ownership_repo_primary_full_results.csv")
    write_csv(subset_results, out_dir / "ownership_repo_primary_high_coverage_results.csv")

    comparison = compare_full_vs_subset(full_results, subset_results)
    write_csv(comparison, out_dir / "ownership_full_vs_high_bilateral_coverage_results.csv")

    report = render_report(overall, threshold_summary, args.main_threshold, main_repos, subset, comparison)
    write_markdown(report, out_dir / "ownership_coverage_robustness_report.md")

    ok_full = int(full_results["model_status"].eq("ok").sum()) if not full_results.empty else 0
    ok_subset = int(subset_results["model_status"].eq("ok").sum()) if not subset_results.empty else 0
    same_direction = int(comparison["same_direction_flag"].sum()) if not comparison.empty and "same_direction_flag" in comparison.columns else 0
    print(f"Full ownership models ok: {ok_full}/{len(full_results)}")
    print(f"High-coverage ownership models ok: {ok_subset}/{len(subset_results)}")
    print(f"Same-direction full vs high-coverage rows: {same_direction}/{len(comparison)}")
    print(f"Wrote report: {out_dir / 'ownership_coverage_robustness_report.md'}")


if __name__ == "__main__":
    main()
