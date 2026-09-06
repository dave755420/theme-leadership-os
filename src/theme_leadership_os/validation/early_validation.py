"""M0-vs-early-radar validation pack.

This module turns the existing early-rise watchlist into an auditable evaluation
surface without promoting exploratory data into an investment claim. It keeps
three states separate:

* ``PASS``: every implemented release gate is measurable and green;
* ``FAILED``: required evidence is present but one or more gates are red;
* ``NOT_READY``: required evidence (PIT data manifest, matured outcomes, formal
  episode file, or portfolio backtest sign-off) is still missing.

The evaluator never fetches market data and never tunes thresholds from outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import log
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..signals.core import SignalConfig, _prepare
from .annual import early_rise_signals
from .episodes import future_panel_labels
from .protocol import (
    early_signal_config_from_protocol,
    load_protocol,
    protocol_alignment_mismatches,
)

REQUIRED_COLUMNS = frozenset({"date", "theme", "security", "value"})
AVAILABILITY_COLUMNS = (
    "feature_available_at",
    "evidence_available_at",
    "available_at",
)
REQUIRED_MANIFEST_FLAGS = (
    "total_return_adjusted",
    "point_in_time_membership_complete",
    "delisting_complete",
    "source_registry_frozen",
    "availability_semantics_frozen",
)


@dataclass(frozen=True)
class EarlyValidationResult:
    """Serializable validation result plus auditable row-level panels."""

    status: str
    summary: dict[str, Any]
    signal_panel: pd.DataFrame
    weekly_metrics: pd.DataFrame
    episode_metrics: pd.DataFrame

    def to_dict(self, *, include_rows: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "summary": _jsonable(self.summary),
        }
        if include_rows:
            payload["signal_panel"] = _frame_records(self.signal_panel)
            payload["weekly_metrics"] = _frame_records(self.weekly_metrics)
            payload["episode_metrics"] = _frame_records(self.episode_metrics)
        return payload

    def write_json(self, path: str | Path, *, include_rows: bool = False) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(include_rows=include_rows), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if value is None or isinstance(value, (str, bytes, bool, int)):
        return value
    try:
        return None if bool(pd.isna(value)) else value
    except (TypeError, ValueError):
        return value


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [_jsonable(row) for row in frame.to_dict(orient="records")]


def _input_fingerprint(data: pd.DataFrame) -> str:
    columns = sorted(
        REQUIRED_COLUMNS.union(set(AVAILABILITY_COLUMNS).intersection(data.columns))
    )
    canonical = data.loc[:, columns].copy()
    canonical["date"] = pd.to_datetime(canonical["date"], errors="coerce")
    canonical = canonical.sort_values(["date", "theme", "security"], kind="mergesort")
    digest = pd.util.hash_pandas_object(canonical, index=False).to_numpy().tobytes()
    return sha256(digest).hexdigest()


def _protocol_hash(protocol: Mapping[str, Any]) -> str:
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def _availability_audit(data: pd.DataFrame) -> tuple[str | None, int | None]:
    availability_column = next((name for name in AVAILABILITY_COLUMNS if name in data), None)
    if availability_column is None:
        return None, None
    available = pd.to_datetime(data[availability_column], errors="coerce", utc=True)
    observed = pd.to_datetime(data["date"], errors="coerce", utc=True)
    observation_end = observed.dt.normalize() + pd.Timedelta(days=1)
    bad = available.isna() | observed.isna() | available.ge(observation_end)
    return availability_column, int(bad.sum())


def _integrity_report(
    data: pd.DataFrame,
    *,
    data_manifest: Mapping[str, Any] | None,
    coverage: float,
    certification_min_coverage: float,
) -> dict[str, Any]:
    missing_columns = sorted(REQUIRED_COLUMNS.difference(data.columns))
    duplicate_count = 0
    invalid_date_count = 0
    invalid_value_count = 0
    availability_column: str | None = None
    availability_violations: int | None = None
    if not missing_columns:
        dates = pd.to_datetime(data["date"], errors="coerce")
        values = pd.to_numeric(data["value"], errors="coerce")
        invalid_date_count = int(dates.isna().sum())
        invalid_value_count = int((values.isna() | values.le(0)).sum())
        duplicate_count = int(data.duplicated(["date", "theme", "security"], keep=False).sum())
        availability_column, availability_violations = _availability_audit(data)

    manifest = dict(data_manifest or {})
    manifest_flags = {name: bool(manifest.get(name, False)) for name in REQUIRED_MANIFEST_FLAGS}
    explicit_availability = availability_column is not None and availability_violations == 0
    passed = (
        not missing_columns
        and invalid_date_count == 0
        and invalid_value_count == 0
        and duplicate_count == 0
        and explicit_availability
        and coverage >= certification_min_coverage
        and all(manifest_flags.values())
    )
    return {
        "passed": passed,
        "missing_columns": missing_columns,
        "invalid_dates": invalid_date_count,
        "invalid_values": invalid_value_count,
        "duplicate_rows": duplicate_count,
        "availability_column": availability_column,
        "availability_violations": availability_violations,
        "availability_auditable": availability_column is not None,
        "overall_oos_coverage": coverage,
        "required_coverage": certification_min_coverage,
        "manifest_flags": manifest_flags,
        "manifest_missing": [name for name, value in manifest_flags.items() if not value],
    }


def _score_panel(signals: pd.DataFrame, weights: Mapping[str, Any]) -> pd.DataFrame:
    out = signals.copy()
    components = {
        "acceleration_rank": pd.to_numeric(out["acceleration_rank"], errors="coerce"),
        "rs_rank_4w": pd.to_numeric(out["rs_rank_4w"], errors="coerce"),
        "rank_improvement_4v13": pd.to_numeric(
            out["rank_improvement_4v13"], errors="coerce"
        ).clip(lower=0.0, upper=1.0),
        "breadth_4w": pd.to_numeric(out["breadth_4w"], errors="coerce"),
        "participation": pd.to_numeric(out["participation"], errors="coerce"),
    }
    numeric_weights = {name: float(weights.get(name, 0.0)) for name in components}
    total = sum(numeric_weights.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"early_score_weights must sum to 1.0, got {total:.6f}")
    valid = pd.Series(True, index=out.index)
    score = pd.Series(0.0, index=out.index, dtype=float)
    for name, values in components.items():
        valid &= values.notna()
        score += numeric_weights[name] * values.fillna(0.0)
    out["early_score"] = score.where(valid)
    out["early_score_rank"] = out.groupby("decision_date")["early_score"].rank(
        pct=True, method="average"
    )
    out["m0_rank"] = out.groupby("decision_date")["m0_26w_spy_relative"].rank(
        pct=True, method="average"
    )
    return out


def _weekly_rank_metrics(
    panel: pd.DataFrame,
    score_column: str,
    outcome_column: str,
    *,
    min_assets: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, group in panel.groupby("decision_date", sort=True):
        valid = group[[score_column, outcome_column]].dropna()
        if len(valid) < min_assets:
            continue
        ic = valid[score_column].corr(valid[outcome_column], method="spearman")
        score_rank = valid[score_column].rank(pct=True, method="average")
        top = valid.loc[score_rank.ge(0.80), outcome_column]
        universe = valid[outcome_column]
        rows.append(
            {
                "decision_date": pd.Timestamp(date),
                "model": score_column,
                "assets": int(len(valid)),
                "rank_ic": float(ic) if pd.notna(ic) else np.nan,
                "top_bucket_excess": float(top.mean()) if not top.empty else np.nan,
                "top_minus_universe_excess": (
                    float(top.mean() - universe.mean()) if not top.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _block_bootstrap_lower_bound(
    values: pd.Series,
    *,
    block: int = 52,
    repetitions: int = 1000,
    seed: int = 755420,
) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) < max(10, block):
        return None
    rng = np.random.default_rng(seed)
    n = len(clean)
    block = min(block, n)
    means = np.empty(repetitions, dtype=float)
    starts = np.arange(0, n - block + 1)
    blocks_needed = int(np.ceil(n / block))
    for i in range(repetitions):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([clean[start : start + block] for start in chosen])[:n]
        means[i] = float(np.mean(sample))
    return float(np.quantile(means, 0.025))


def _summary_for_model(weekly: pd.DataFrame) -> dict[str, Any]:
    if weekly.empty:
        return {
            "weeks": 0,
            "mean_rank_ic": None,
            "median_rank_ic": None,
            "block_bootstrap_95_lower": None,
            "mean_top_bucket_excess": None,
            "mean_top_minus_universe_excess": None,
        }
    return {
        "weeks": int(len(weekly)),
        "mean_rank_ic": float(weekly["rank_ic"].mean()),
        "median_rank_ic": float(weekly["rank_ic"].median()),
        "block_bootstrap_95_lower": _block_bootstrap_lower_bound(weekly["rank_ic"]),
        "mean_top_bucket_excess": float(weekly["top_bucket_excess"].mean()),
        "mean_top_minus_universe_excess": float(
            weekly["top_minus_universe_excess"].mean()
        ),
    }


def _episode_validation(
    panel: pd.DataFrame,
    prepared: Any,
    episodes: pd.DataFrame | None,
    *,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    acceptance = dict(protocol.get("early_discovery_acceptance", {}))
    early_hit = dict(protocol.get("early_hit", {}))
    if episodes is None or episodes.empty:
        return {
            "ready": False,
            "reason": "formal episode file not supplied",
            "episode_count": 0,
            "theme_count": 0,
            "recall": None,
            "false_discovery_rate": None,
            "median_signal_lag_weeks": None,
            "median_remaining_move": None,
            "passed": False,
        }, pd.DataFrame()

    required = {"theme", "onset", "peak"}
    missing = required.difference(episodes.columns)
    if missing:
        return {
            "ready": False,
            "reason": "episode file missing columns: " + ", ".join(sorted(missing)),
            "episode_count": 0,
            "theme_count": 0,
            "recall": None,
            "false_discovery_rate": None,
            "median_signal_lag_weeks": None,
            "median_remaining_move": None,
            "passed": False,
        }, pd.DataFrame()

    eps = episodes.copy()
    eps["onset"] = pd.to_datetime(eps["onset"], errors="coerce")
    eps["peak"] = pd.to_datetime(eps["peak"], errors="coerce")
    if "qualifies" in eps:
        eps = eps.loc[eps["qualifies"].fillna(False).astype(bool)]
    eps = eps.loc[
        eps["onset"].between(oos_start, oos_end)
        & eps["peak"].le(oos_end)
        & eps["onset"].notna()
        & eps["peak"].notna()
    ].copy()
    if eps.empty:
        return {
            "ready": False,
            "reason": "no matured OOS episodes",
            "episode_count": 0,
            "theme_count": 0,
            "recall": None,
            "false_discovery_rate": None,
            "median_signal_lag_weeks": None,
            "median_remaining_move": None,
            "passed": False,
        }, pd.DataFrame()

    benchmark = pd.to_numeric(prepared.benchmark, errors="coerce")
    theme_index = prepared.theme_index
    confirmed = panel.loc[
        panel["early_confirmed"].fillna(False)
        & panel["decision_date"].between(oos_start, oos_end)
    ].copy()
    before = int(early_hit.get("onset_before_weeks", 4))
    after = int(early_hit.get("onset_after_weeks", 8))
    max_realized = float(early_hit.get("max_realized_log_fraction", 0.25))
    episode_rows: list[dict[str, Any]] = []
    matched_signal_indices: set[int] = set()

    for episode_id, (_, ep) in enumerate(
        eps.sort_values(["onset", "theme"]).iterrows(), start=1
    ):
        theme = str(ep["theme"])
        onset = pd.Timestamp(ep["onset"])
        peak = pd.Timestamp(ep["peak"])
        candidates = confirmed.loc[
            confirmed["theme"].eq(theme)
            & confirmed["decision_date"].between(
                onset - pd.Timedelta(weeks=before), onset + pd.Timedelta(weeks=after)
            )
        ].sort_values("decision_date")
        hit = False
        signal_date: pd.Timestamp | None = None
        realized_fraction: float | None = None
        remaining_fraction: float | None = None
        if theme in theme_index.columns and onset in theme_index.index and peak in theme_index.index:
            rel = pd.to_numeric(theme_index[theme], errors="coerce").div(benchmark)
            onset_rel = rel.get(onset)
            peak_rel = rel.get(peak)
            if pd.notna(onset_rel) and pd.notna(peak_rel) and onset_rel > 0 and peak_rel > 0:
                full_log = log(float(peak_rel) / float(onset_rel))
                if full_log > 0:
                    for candidate_index, candidate in candidates.iterrows():
                        candidate_date = pd.Timestamp(candidate["decision_date"])
                        signal_rel = rel.get(candidate_date)
                        if pd.isna(signal_rel) or signal_rel <= 0:
                            continue
                        realized = log(float(signal_rel) / float(onset_rel)) / full_log
                        if realized <= max_realized:
                            hit = True
                            signal_date = candidate_date
                            realized_fraction = float(realized)
                            remaining_fraction = float(max(0.0, 1.0 - realized))
                            matched_signal_indices.add(int(candidate_index))
                            break
        episode_rows.append(
            {
                "episode_id": episode_id,
                "theme": theme,
                "onset": onset,
                "peak": peak,
                "hit": hit,
                "signal_date": signal_date,
                "signal_lag_weeks": (
                    None if signal_date is None else float((signal_date - onset).days / 7.0)
                ),
                "realized_log_fraction": realized_fraction,
                "remaining_move_fraction": remaining_fraction,
            }
        )

    table = pd.DataFrame(episode_rows)
    hits = int(table["hit"].sum()) if not table.empty else 0
    episode_count = int(len(table))
    recall = hits / episode_count if episode_count else None
    confirmed_count = int(len(confirmed))
    false_signals = confirmed_count - len(matched_signal_indices)
    fdr = false_signals / confirmed_count if confirmed_count else None
    hit_rows = table.loc[table["hit"]]
    median_lag = (
        None if hit_rows.empty else float(pd.to_numeric(hit_rows["signal_lag_weeks"]).median())
    )
    median_remaining = (
        None
        if hit_rows.empty
        else float(pd.to_numeric(hit_rows["remaining_move_fraction"]).median())
    )
    min_recall = float(acceptance.get("min_episode_recall", 0.70))
    max_fdr = float(acceptance.get("max_false_discovery_rate", 0.35))
    max_lag = float(acceptance.get("max_median_signal_lag_weeks", 2.0))
    min_remaining = float(acceptance.get("min_median_remaining_move", 0.60))
    min_episodes = int(acceptance.get("min_independent_episodes", 30))
    min_themes = int(acceptance.get("min_episode_themes", 10))
    theme_count = int(table["theme"].nunique()) if not table.empty else 0
    ready = episode_count >= min_episodes and theme_count >= min_themes and confirmed_count > 0
    passed = bool(
        ready
        and recall is not None
        and recall >= min_recall
        and fdr is not None
        and fdr <= max_fdr
        and median_lag is not None
        and median_lag <= max_lag
        and median_remaining is not None
        and median_remaining >= min_remaining
    )
    return {
        "ready": ready,
        "episode_count": episode_count,
        "theme_count": theme_count,
        "confirmed_events": confirmed_count,
        "matched_confirmed_events": len(matched_signal_indices),
        "recall": recall,
        "false_discovery_rate": fdr,
        "median_signal_lag_weeks": median_lag,
        "median_remaining_move": median_remaining,
        "minimums": {
            "episode_count": min_episodes,
            "theme_count": min_themes,
            "recall": min_recall,
            "max_false_discovery_rate": max_fdr,
            "max_median_signal_lag_weeks": max_lag,
            "min_median_remaining_move": min_remaining,
        },
        "passed": passed,
    }, table


def validate_early_radar(
    data: pd.DataFrame,
    *,
    episodes: pd.DataFrame | None = None,
    data_manifest: Mapping[str, Any] | None = None,
    protocol: Mapping[str, Any] | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> EarlyValidationResult:
    """Evaluate the frozen early radar against M0 on matured OOS observations."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError("input is missing columns: " + ", ".join(sorted(missing)))

    rules = dict(protocol or load_protocol())
    validation = dict(rules.get("validation", {}))
    weights = dict(rules.get("early_score_weights", {}))
    cfg = early_signal_config_from_protocol(rules)
    drift = protocol_alignment_mismatches(rules)
    if drift:
        raise ValueError("protocol/default drift detected: " + "; ".join(drift))

    cutoff = pd.Timestamp(as_of) if as_of is not None else pd.to_datetime(data["date"]).max()
    cutoff = cutoff.tz_localize(None) if cutoff.tzinfo is not None else cutoff
    oos_start = pd.Timestamp(validation.get("oos_start", "2019-01-01"))
    frozen_oos_end = pd.Timestamp(validation.get("oos_end", "2025-12-31"))
    oos_end = min(frozen_oos_end, pd.Timestamp(cutoff).normalize())
    min_assets = int(validation.get("min_cross_section_themes", 15))
    certification_coverage = float(validation.get("certification_min_coverage", 0.95))

    availability_column, availability_violations = _availability_audit(data)
    if availability_column is not None and availability_violations:
        raise ValueError(
            f"{availability_column} contains {availability_violations} timing violation(s); "
            "validation metrics are not computed on temporally invalid rows"
        )

    signals = early_rise_signals(data, cfg, as_of=cutoff)
    panel = _score_panel(signals, weights)
    prepared = _prepare(
        data.loc[pd.to_datetime(data["date"], errors="coerce") <= cutoff].copy(),
        SignalConfig(
            benchmark_security=cfg.benchmark_security,
            min_participation=cfg.min_participation,
            min_history_weeks=cfg.signal_history_weeks,
        ),
    )
    futures = future_panel_labels(
        prepared.theme_index,
        prepared.benchmark,
        horizons=(13, 26, 52),
        next_session=True,
    ).reset_index()
    futures = futures.rename(columns={"decision_at": "decision_date"})
    panel = panel.merge(futures, on=["decision_date", "theme"], how="left", validate="one_to_one")
    panel["decision_date"] = pd.to_datetime(panel["decision_date"])
    oos = panel.loc[panel["decision_date"].between(oos_start, oos_end)].copy()

    oos_returns = prepared.theme_returns.loc[oos_start:oos_end]
    expected_cells = int(len(oos_returns) * len(prepared.themes))
    observed_cells = int(oos_returns.notna().sum().sum())
    coverage = observed_cells / expected_cells if expected_cells else 0.0
    integrity = _integrity_report(
        data,
        data_manifest=data_manifest,
        coverage=coverage,
        certification_min_coverage=certification_coverage,
    )

    available_scores = oos[["decision_date", "theme", "early_score", "m0_rank"]].copy()
    counts = available_scores.groupby("decision_date")["early_score"].count()
    eligible_dates = counts.loc[counts.ge(min_assets)].index
    oos["cross_section_count"] = oos["decision_date"].map(counts).fillna(0).astype(int)
    oos["cross_section_eligible"] = oos["decision_date"].isin(eligible_dates)
    matured26 = oos.loc[
        oos["cross_section_eligible"] & oos["future_26w_excess_return"].notna()
    ].copy()
    matured52 = oos.loc[
        oos["cross_section_eligible"] & oos["future_52w_excess_return"].notna()
    ].copy()

    early26 = _weekly_rank_metrics(
        matured26, "early_score", "future_26w_excess_return", min_assets=min_assets
    )
    m026 = _weekly_rank_metrics(
        matured26, "m0_rank", "future_26w_excess_return", min_assets=min_assets
    )
    early52 = _weekly_rank_metrics(
        matured52, "early_score", "future_52w_excess_return", min_assets=min_assets
    )
    weekly = pd.concat([early26, m026, early52], ignore_index=True)
    early_summary = _summary_for_model(early26)
    m0_summary = _summary_for_model(m026)
    early52_summary = _summary_for_model(early52)

    early_ic = early_summary["mean_rank_ic"]
    m0_ic = m0_summary["mean_rank_ic"]
    early_top = early_summary["mean_top_bucket_excess"]
    m0_top = m0_summary["mean_top_bucket_excess"]
    ic_delta = None if early_ic is None or m0_ic is None else float(early_ic - m0_ic)
    top_delta = None if early_top is None or m0_top is None else float(early_top - m0_top)
    min_ic = float(validation.get("min_rank_ic_26w", 0.05))
    min_ic_delta = float(validation.get("min_ic_improvement_vs_m0", 0.02))
    min_top_delta = float(validation.get("min_top_bucket_excess_improvement", 0.02))
    statistical_ready = bool(len(early26) > 0 and len(m026) > 0 and len(early52) > 0)
    statistical_passed = bool(
        statistical_ready
        and early_ic is not None
        and early_ic >= min_ic
        and ic_delta is not None
        and ic_delta >= min_ic_delta
        and top_delta is not None
        and top_delta >= min_top_delta
        and early52_summary["mean_rank_ic"] is not None
        and early52_summary["mean_rank_ic"] > 0.0
        and early_summary["block_bootstrap_95_lower"] is not None
        and early_summary["block_bootstrap_95_lower"] > 0.0
    )

    episode_summary, episode_table = _episode_validation(
        oos,
        prepared,
        episodes,
        oos_start=oos_start,
        oos_end=oos_end,
        protocol=rules,
    )
    manifest = dict(data_manifest or {})
    portfolio_ready = bool(manifest.get("portfolio_backtest_complete", False))
    portfolio_passed = bool(manifest.get("portfolio_backtest_passed", False))
    if integrity["passed"] and statistical_ready and episode_summary["ready"] and portfolio_ready:
        status = (
            "PASS"
            if statistical_passed and episode_summary["passed"] and portfolio_passed
            else "FAILED"
        )
    else:
        status = "NOT_READY"

    summary = {
        "schema": "theme_leadership_early_validation.v1",
        "status": status,
        "as_of": pd.Timestamp(cutoff).normalize(),
        "oos_window": {"start": oos_start, "end": oos_end},
        "input_fingerprint": _input_fingerprint(data),
        "protocol_hash": _protocol_hash(rules),
        "protocol_alignment": {"passed": not drift, "mismatches": drift},
        "data_integrity": integrity,
        "cross_section": {
            "minimum_themes": min_assets,
            "eligible_weeks": int(len(eligible_dates)),
            "total_oos_weeks": int(oos["decision_date"].nunique()),
        },
        "m0_26w": m0_summary,
        "early_score_26w": early_summary,
        "early_score_52w": early52_summary,
        "comparison_vs_m0": {
            "rank_ic_delta": ic_delta,
            "required_rank_ic_delta": min_ic_delta,
            "top_bucket_excess_delta": top_delta,
            "required_top_bucket_excess_delta": min_top_delta,
        },
        "statistical_gate": {
            "ready": statistical_ready,
            "passed": statistical_passed,
            "minimum_26w_rank_ic": min_ic,
            "minimum_improvement_vs_m0": min_ic_delta,
            "minimum_top_bucket_improvement": min_top_delta,
            "requires_positive_52w_ic": True,
            "requires_positive_26w_block_bootstrap_lower_bound": True,
        },
        "early_discovery_gate": episode_summary,
        "economic_gate": {
            "ready": portfolio_ready,
            "passed": None if not portfolio_ready else portfolio_passed,
            "reason": (
                None
                if portfolio_ready
                else "cost/turnover portfolio backtest pack not supplied"
            ),
        },
        "release_gate": {
            "status": status,
            "note": (
                "PASS is reserved for PIT-complete data, measurable OOS statistical and "
                "early-discovery gates, and a separately completed cost/turnover portfolio backtest."
            ),
        },
    }
    return EarlyValidationResult(
        status=status,
        summary=summary,
        signal_panel=oos.reset_index(drop=True),
        weekly_metrics=weekly.reset_index(drop=True),
        episode_metrics=episode_table.reset_index(drop=True),
    )


__all__ = ["EarlyValidationResult", "validate_early_radar"]
