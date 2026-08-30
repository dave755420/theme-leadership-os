"""End-to-end orchestration for the three independent signal lanes.

The pipeline is intentionally thin: it validates a local tidy price frame,
applies an optional point-in-time cutoff, and runs each transparent signal
engine.  It does not fetch data, place orders, or collapse the three outputs
into a single recommendation score.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import pandas as pd

from .signals import (
    CurrentLeaderConfig,
    EmergingRadarConfig,
    HoldCandidate12MConfig,
    current_leader,
    emerging_radar,
    hold_candidate_12m,
)

RESEARCH_DISCLAIMER = (
    "Experimental research/watchlist output; not an investment recommendation."
)
REQUIRED_PRICE_COLUMNS = frozenset({"date", "theme", "security", "value"})
OPTIONAL_FINGERPRINT_COLUMNS = frozenset(
    {"feature_available_at", "evidence_available_at", "available_at"}
)


class PipelineIntegrityError(ValueError):
    """Raised when an input would make the signal run temporally ambiguous."""


@dataclass(frozen=True)
class SignalRun:
    """One reproducible run containing three deliberately separate outputs."""

    as_of: pd.Timestamp
    input_fingerprint: str
    current_leaders: pd.DataFrame
    emerging_radar: pd.DataFrame
    hold_candidates_12m: pd.DataFrame
    disclaimer: str = RESEARCH_DISCLAIMER

    def latest(self) -> dict[str, pd.DataFrame]:
        """Return the latest available cross-section for each output lane."""

        return {
            "current_leaders": _latest_cross_section(self.current_leaders),
            "emerging_radar": _latest_cross_section(self.emerging_radar),
            "hold_candidates_12m": _latest_cross_section(self.hold_candidates_12m),
        }

    def manifest(self) -> dict[str, Any]:
        """Return a non-sensitive run manifest suitable for a report."""

        latest = self.latest()
        return {
            "schema": "theme_leadership_signal_run.v1",
            "as_of": self.as_of.isoformat(),
            "input_fingerprint": self.input_fingerprint,
            "disclaimer": self.disclaimer,
            "lanes": {
                name: {
                    "rows": int(len(frame)),
                    "decision_date": (
                        None
                        if frame.empty
                        else pd.Timestamp(frame["decision_date"].max()).isoformat()
                    ),
                    "status_counts": (
                        {}
                        if frame.empty or "status" not in frame
                        else {
                            str(key): int(value)
                            for key, value in frame["status"].value_counts(dropna=False).items()
                        }
                    ),
                }
                for name, frame in latest.items()
            },
        }


def _latest_cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "decision_date" not in frame:
        return frame.copy()
    decision_dates = pd.to_datetime(frame["decision_date"], errors="coerce")
    if decision_dates.notna().sum() == 0:
        return frame.iloc[0:0].copy()
    latest = decision_dates.max()
    return frame.loc[decision_dates.eq(latest)].reset_index(drop=True)


def _normalise_as_of(value: str | pd.Timestamp | None, dates: pd.Series) -> pd.Timestamp:
    if value is None:
        parsed = pd.to_datetime(dates, errors="coerce")
        if parsed.notna().sum() == 0:
            raise PipelineIntegrityError("input has no valid dates")
        return pd.Timestamp(parsed.max()).tz_localize(None).normalize()
    result = pd.Timestamp(value)
    if result.tzinfo is not None:
        result = result.tz_convert("UTC").tz_localize(None)
    return result.normalize()


def _prepare_input(prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    missing = REQUIRED_PRICE_COLUMNS.difference(prices.columns)
    if missing:
        raise PipelineIntegrityError(
            "tidy price input is missing columns: " + ", ".join(sorted(missing))
        )
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise PipelineIntegrityError("price input contains invalid dates")
    if isinstance(frame["date"].dtype, pd.DatetimeTZDtype):
        frame["date"] = frame["date"].dt.tz_convert("UTC").dt.tz_localize(None)

    available_column = next(
        (
            name
            for name in ("feature_available_at", "evidence_available_at", "available_at")
            if name in frame
        ),
        None,
    )
    if available_column is not None:
        available = pd.to_datetime(frame[available_column], errors="coerce", utc=True)
        if available.isna().any():
            raise PipelineIntegrityError(f"{available_column} contains invalid timestamps")
        available_naive = available.dt.tz_convert("UTC").dt.tz_localize(None)
        observation_end = frame["date"].dt.normalize() + np.timedelta64(1, "D")
        violations = available_naive.ge(observation_end)
        if violations.any():
            raise PipelineIntegrityError(
                f"{available_column} is later than its observation session in "
                f"{int(violations.sum())} row(s)"
            )

    frame = frame.loc[frame["date"].le(as_of)].copy()
    if frame.empty:
        raise PipelineIntegrityError("no price observations are available at the as-of date")
    return frame


def _fingerprint(frame: pd.DataFrame, configs: dict[str, Any], as_of: pd.Timestamp) -> str:
    columns = sorted(
        REQUIRED_PRICE_COLUMNS.union(OPTIONAL_FINGERPRINT_COLUMNS.intersection(frame.columns))
    )
    canonical = frame.loc[:, columns].copy()
    canonical = canonical.sort_values(["date", "theme", "security"], kind="mergesort")
    row_hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy().tobytes()
    config_json = json.dumps(
        {"as_of": as_of.isoformat(), "configs": configs},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(row_hashes + config_json).hexdigest()


def run_signal_pipeline(
    prices: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp | None = None,
    current_config: CurrentLeaderConfig | None = None,
    radar_config: EmergingRadarConfig | None = None,
    hold_config: HoldCandidate12MConfig | None = None,
) -> SignalRun:
    """Run all three lanes without combining their scores or statuses."""

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    missing = REQUIRED_PRICE_COLUMNS.difference(prices.columns)
    if missing:
        raise PipelineIntegrityError(
            "tidy price input is missing columns: " + ", ".join(sorted(missing))
        )
    as_of_timestamp = _normalise_as_of(as_of, prices.get("date", pd.Series(dtype=object)))
    prepared = _prepare_input(prices, as_of_timestamp)
    current_cfg = current_config or CurrentLeaderConfig()
    radar_cfg = radar_config or EmergingRadarConfig()
    hold_cfg = hold_config or HoldCandidate12MConfig()
    configs = {
        "current": asdict(current_cfg),
        "radar": asdict(radar_cfg),
        "hold_12m": asdict(hold_cfg),
    }
    fingerprint = _fingerprint(prepared, configs, as_of_timestamp)
    return SignalRun(
        as_of=as_of_timestamp,
        input_fingerprint=fingerprint,
        current_leaders=current_leader(
            prepared,
            config=current_cfg,
            as_of=as_of_timestamp,
        ),
        emerging_radar=emerging_radar(
            prepared,
            config=radar_cfg,
            as_of=as_of_timestamp,
        ),
        hold_candidates_12m=hold_candidate_12m(
            prepared,
            config=hold_cfg,
            as_of=as_of_timestamp,
        ),
    )


__all__ = [
    "PipelineIntegrityError",
    "RESEARCH_DISCLAIMER",
    "SignalRun",
    "run_signal_pipeline",
]
