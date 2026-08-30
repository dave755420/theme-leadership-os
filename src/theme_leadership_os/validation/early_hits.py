"""Leakage-safe evaluation of persistent early signals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .episodes import (
    DEFAULT_EPISODE_HORIZON_WEEKS,
    DEFAULT_MAJOR_EXCESS,
    DEFAULT_MAJOR_RETURN,
    _fraction,
    _series,
    future_labels,
)

DEFAULT_EARLY_WINDOW_BEFORE_WEEKS = 4
DEFAULT_EARLY_WINDOW_AFTER_WEEKS = 8
DEFAULT_MAX_REALIZED_LOG_FRACTION = 0.25


def _signal_series(signals: Any, *, signal_column: str | None = None) -> pd.Series:
    if isinstance(signals, pd.DataFrame):
        if signal_column is not None:
            if signal_column not in signals.columns:
                raise KeyError(f"signal column not found: {signal_column}")
            result = signals[signal_column]
        else:
            candidates = ("signal", "confirmed", "watch", "entry", "prediction")
            selected = next((c for c in candidates if c in signals.columns), None)
            if selected is None:
                if signals.shape[1] != 1:
                    raise ValueError("provide signal_column for a multi-column signal frame")
                selected = signals.columns[0]
            result = signals[selected]
    elif isinstance(signals, pd.Series):
        result = signals
    elif isinstance(signals, Mapping):
        result = pd.Series(signals)
    else:
        result = pd.Series(signals)
    result = result.copy()
    if not isinstance(result.index, pd.DatetimeIndex) and not pd.api.types.is_numeric_dtype(
        result.index
    ):
        result.index = pd.to_datetime(result.index)
    if isinstance(result.index, pd.DatetimeIndex):
        result = result.sort_index()
        result = result[~result.index.duplicated(keep="last")]
    return result


def _persistent(signal: pd.Series, position: int, persistence: int, window: int | None) -> bool:
    if persistence <= 1:
        return bool(signal.iloc[position])
    if window is None:
        window = persistence
    if window < persistence or position + 1 < persistence:
        return False
    values = signal.iloc[max(0, position - window + 1) : position + 1].fillna(False).astype(bool)
    # A persistence requirement means consecutive confirmations at the end of
    # the window.  ``window`` additionally lets callers tolerate a false day in
    # an n-of-m implementation without changing the default strict behavior.
    consecutive = 0
    for value in values.iloc[::-1]:
        if bool(value):
            consecutive += 1
        else:
            break
    if consecutive >= persistence:
        return True
    return bool(values.sum() >= persistence and window > persistence)


def _lookup_start(event_starts: Any, decision: Any) -> Any:
    if event_starts is None:
        return None
    if isinstance(event_starts, Mapping):
        return event_starts.get(decision)
    if isinstance(event_starts, pd.Series):
        if decision in event_starts.index:
            return event_starts.loc[decision]
        return None
    if isinstance(event_starts, pd.DataFrame):
        for column in ("onset", "episode_start", "start", "event_start"):
            if column in event_starts.columns and decision in event_starts.index:
                return event_starts.loc[decision, column]
        onset_column = next(
            (
                column
                for column in ("onset", "episode_start", "start", "event_start")
                if column in event_starts.columns
            ),
            None,
        )
        if onset_column is not None:
            starts = pd.to_datetime(event_starts[onset_column], errors="coerce").dropna()
            if len(starts):
                prior = starts[starts <= decision]
                return prior.iloc[-1] if len(prior) else None
    return event_starts


def _lookup_event(event_data: Any, decision: Any) -> dict[str, Any]:
    """Return the nearest event metadata row at/before a signal date."""

    if event_data is None:
        return {}
    if isinstance(event_data, pd.DataFrame):
        frame = event_data
        if decision in frame.index:
            value = frame.loc[decision]
            return value.iloc[0].to_dict() if isinstance(value, pd.DataFrame) else value.to_dict()
        onset_column = next(
            (c for c in ("onset", "episode_start", "start", "event_start") if c in frame.columns),
            None,
        )
        if onset_column is not None:
            onsets = pd.to_datetime(frame[onset_column], errors="coerce")
            if isinstance(decision, (pd.Timestamp, np.datetime64)):
                valid = frame.loc[onsets <= pd.Timestamp(decision)]
                if not valid.empty:
                    return valid.iloc[-1].to_dict()
        return {}
    if isinstance(event_data, Mapping):
        value = event_data.get(decision, {})
        return value if isinstance(value, Mapping) else {"onset": value}
    return {}


def _date_window_ok(decision: Any, onset: Any, before_weeks: int, after_weeks: int) -> bool:
    if onset is None or pd.isna(onset):
        return True
    try:
        if isinstance(decision, (pd.Timestamp, np.datetime64)) or isinstance(
            onset, (pd.Timestamp, np.datetime64)
        ):
            delta = pd.Timestamp(decision) - pd.Timestamp(onset)
            before = np.timedelta64(7 * before_weeks, "D")
            after = np.timedelta64(7 * after_weeks, "D")
            return -before <= delta <= after
        return -before_weeks <= float(decision) - float(onset) <= after_weeks
    except (TypeError, ValueError):
        return False


def _price_on_or_before(prices: pd.Series, timestamp: Any) -> float | None:
    numeric = pd.to_numeric(prices, errors="coerce").dropna()
    if numeric.empty:
        return None
    try:
        if isinstance(numeric.index, pd.DatetimeIndex):
            values = numeric.loc[numeric.index <= pd.Timestamp(timestamp)]
        else:
            values = numeric.loc[numeric.index <= timestamp]
        return None if values.empty else float(values.iloc[-1])
    except (TypeError, ValueError):
        return None


def _missing_scalar(value: Any) -> bool:
    try:
        result = pd.isna(value)
        return bool(result) if np.isscalar(result) else False
    except (TypeError, ValueError):
        return False


@dataclass
class EarlyHitReport:
    """Rows and aggregate metrics produced by :func:`evaluate_early_hits`."""

    observations: pd.DataFrame
    eligible: int
    hits: int
    hit_rate: float | None
    persistent_signals: int
    matured: int
    first_hits: int = 0
    window_available: int = 0
    fraction_available: int = 0
    false_alerts: int = 0
    median_lead_lag_weeks: float | None = None
    median_remaining_move_fraction: float | None = None

    @property
    def precision(self) -> float | None:
        return self.hit_rate

    @property
    def summary(self) -> dict[str, Any]:
        return self.as_dict()

    def as_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "hits": self.hits,
            "hit_rate": self.hit_rate,
            "precision": self.precision,
            "persistent_signals": self.persistent_signals,
            "matured": self.matured,
            "first_hits": self.first_hits,
            "window_available": self.window_available,
            "fraction_available": self.fraction_available,
            "false_alerts": self.false_alerts,
            "median_lead_lag_weeks": self.median_lead_lag_weeks,
            "median_remaining_move_fraction": self.median_remaining_move_fraction,
        }

    def to_frame(self) -> pd.DataFrame:
        return self.observations.copy()

    def annual_false_alerts(self) -> pd.Series:
        """Count matured persistent non-hits by calendar year."""

        if self.observations.empty:
            return pd.Series(dtype=int, name="false_alerts")
        mask = (
            self.observations.get(
                "eligible_for_early_hit",
                self.observations["signal"]
                & self.observations["persistent"]
                & self.observations["matured"],
            )
            & ~self.observations["hit"]
        )
        dates = pd.to_datetime(self.observations.index, errors="coerce")
        values = self.observations.loc[mask].copy()
        values["year"] = dates[mask].year
        return values.groupby("year").size().astype(int).rename("false_alerts")

    def __getitem__(self, key: Any) -> Any:
        if key in self.as_dict():
            return self.as_dict()[key]
        return self.observations[key]

    def __len__(self) -> int:
        return len(self.observations)


def evaluate_early_hits(
    signals: pd.Series | pd.DataFrame | Mapping[Any, Any],
    theme_prices: pd.Series | Mapping[Any, float],
    benchmark_prices: pd.Series | Mapping[Any, float] | None = None,
    *,
    horizon_weeks: int = DEFAULT_EPISODE_HORIZON_WEEKS,
    persistence: int = 2,
    persistence_window: int | None = 3,
    absolute_threshold: float | None = DEFAULT_MAJOR_RETURN,
    excess_threshold: float | None = DEFAULT_MAJOR_EXCESS,
    max_realized_move: float | None = DEFAULT_MAX_REALIZED_LOG_FRACTION,
    event_starts: Any = None,
    event_onsets: Any = None,
    event_peaks: Any = None,
    episodes: pd.DataFrame | None = None,
    onset_before_weeks: int = DEFAULT_EARLY_WINDOW_BEFORE_WEEKS,
    onset_after_weeks: int = DEFAULT_EARLY_WINDOW_AFTER_WEEKS,
    signal_feature_available_at: Any = None,
    require_point_in_time: bool = False,
    signal_column: str | None = None,
    next_session: bool = True,
) -> EarlyHitReport:
    """Evaluate signals without allowing a same-session close to count.

    A hit is the first persistent signal (at least two of three observations by
    default) in the window four weeks before through eight weeks after onset,
    with less than 25% of the onset-to-peak *log* move realised.  Retrospective
    episode labels may provide the peak because this function is an evaluator,
    not a signal generator.  If event metadata is absent, timing and realized
    fraction are reported as unavailable and the row cannot silently count as
    an early hit.  Signals with immature labels remain in the output but are
    excluded from the denominator.
    """

    if persistence < 1:
        raise ValueError("persistence must be at least one")
    if persistence_window is not None and persistence_window < persistence:
        raise ValueError("persistence_window must be >= persistence")
    if onset_before_weeks < 0 or onset_after_weeks < 0:
        raise ValueError("onset window widths cannot be negative")
    signal = _signal_series(signals, signal_column=signal_column)
    prices = _series(theme_prices, name="theme")
    benchmark = None if benchmark_prices is None else _series(benchmark_prices, name="benchmark")
    labels = future_labels(
        prices,
        benchmark,
        horizons=(int(horizon_weeks),),
        next_session=next_session,
        absolute_threshold=absolute_threshold,
        excess_threshold=excess_threshold,
    )
    hold_labels = future_labels(
        prices,
        benchmark,
        horizons=(52,),
        next_session=next_session,
        absolute_threshold=None,
        excess_threshold=None,
    )
    event_data: Any = episodes
    rows: list[dict[str, Any]] = []
    first_signal_by_event: dict[Any, Any] = {}
    signal_values = signal.astype("boolean")
    for position, decision in enumerate(signal.index):
        raw = signal.iloc[position]
        is_signal = bool(raw) if pd.notna(raw) else False
        # labels has the price index after alignment.  A missing row is an
        # unavailable observation, never a failed prediction.
        if decision in labels.index:
            row = labels.loc[decision]
            matured = bool(pd.notna(row.get(f"label_available_at_{horizon_weeks}w")))
            label_value = row.get(f"label_{horizon_weeks}w", np.nan)
            future_return = row.get(f"future_{horizon_weeks}w_return", np.nan)
            future_excess = row.get(f"future_{horizon_weeks}w_excess_return", np.nan)
            entry_at = row.get(f"entry_at_{horizon_weeks}w")
            exit_at = row.get(f"exit_at_{horizon_weeks}w")
        else:
            matured = False
            label_value = np.nan
            future_return = np.nan
            future_excess = np.nan
            entry_at = None
            exit_at = None
        persistent = is_signal and _persistent(
            signal_values, position, persistence, persistence_window
        )
        event_row = _lookup_event(event_data, decision)
        event_start = _lookup_start(event_onsets, decision)
        if event_start is None:
            event_start = _lookup_start(event_starts, decision)
        if event_start is None:
            event_start = event_row.get(
                "onset", event_row.get("episode_start", event_row.get("start"))
            )
        event_peak = _lookup_start(event_peaks, decision)
        if event_peak is None:
            event_peak = event_row.get("peak", event_row.get("end", event_row.get("episode_end")))
        metadata_available = (
            event_start is not None
            and event_peak is not None
            and not _missing_scalar(event_start)
            and not _missing_scalar(event_peak)
        )
        within_window: bool | None = (
            _date_window_ok(decision, event_start, onset_before_weeks, onset_after_weeks)
            if event_start is not None
            else None
        )
        realized_move = np.nan
        realized_log_fraction = np.nan
        remaining_move_fraction = np.nan
        remaining_upside = np.nan
        within_limit: bool | None = None
        if metadata_available:
            max_move = None if max_realized_move is None else float(_fraction(max_realized_move))
            baseline = _price_on_or_before(prices, event_start)
            observed = _price_on_or_before(prices, decision)
            peak_value = _price_on_or_before(prices, event_peak)
            if baseline is not None and observed is not None and baseline > 0:
                realized_move = observed / baseline - 1.0
                if peak_value is not None and peak_value > 0:
                    total_log = float(np.log(peak_value / baseline))
                    realized_log = float(np.log(observed / baseline))
                    if total_log > 0:
                        realized_log_fraction = realized_log / total_log
                        remaining_move_fraction = (total_log - realized_log) / total_log
                        remaining_upside = peak_value / observed - 1.0
                        within_limit = (
                            True if max_move is None else bool(realized_log_fraction < max_move)
                        )
                    else:
                        within_limit = False
        event_qualifies: bool | None = None
        if event_row:
            event_qualifies = bool(event_row.get("qualifies", event_row.get("major_surge", False)))
        event_key = event_start if event_start is not None else "__no_event__"
        first_signal = False
        if persistent and event_key not in first_signal_by_event:
            first_signal_by_event[event_key] = decision
            first_signal = True
        point_in_time_ok = True
        availability = signal_feature_available_at
        if availability is None and isinstance(signals, pd.DataFrame):
            if "feature_available_at" in signals.columns:
                availability = signals["feature_available_at"]
        if availability is not None:
            if isinstance(availability, pd.Series):
                available_at = availability.get(decision, pd.NaT)
            elif isinstance(availability, Mapping):
                available_at = availability.get(decision, pd.NaT)
            else:
                available_at = availability
            try:
                point_in_time_ok = bool(pd.Timestamp(available_at) <= pd.Timestamp(decision))
            except (TypeError, ValueError):
                point_in_time_ok = False
        if require_point_in_time and not point_in_time_ok:
            within_limit = False
        hit = bool(
            first_signal
            and persistent
            and matured
            and (event_qualifies if event_qualifies is not None else bool(label_value))
            and within_window is True
            and within_limit is True
            and point_in_time_ok
        )
        eligible_for_early_hit = bool(
            is_signal
            and persistent
            and matured
            and metadata_available
            and within_window is not None
            and within_limit is not None
        )
        hold_52w_return = np.nan
        hold_52w_excess_return = np.nan
        hold_52w_exit_at = None
        if decision in hold_labels.index:
            hold_row = hold_labels.loc[decision]
            hold_52w_return = hold_row.get("future_52w_return", np.nan)
            hold_52w_excess_return = hold_row.get("future_52w_excess_return", np.nan)
            hold_52w_exit_at = hold_row.get("exit_at_52w")
        rows.append(
            {
                "decision_at": decision,
                "signal": is_signal,
                "persistent": persistent,
                "matured": matured,
                "label": label_value,
                "future_return": future_return,
                "future_excess_return": future_excess,
                "entry_at": entry_at,
                "exit_at": exit_at,
                "event_start": event_start,
                "event_peak": event_peak,
                "event_metadata_available": metadata_available,
                "within_onset_window": within_window,
                "realized_move": realized_move,
                "realized_log_fraction": realized_log_fraction,
                "remaining_move_fraction": remaining_move_fraction,
                "remaining_upside_to_peak": remaining_upside,
                "within_realized_limit": within_limit,
                "point_in_time_ok": point_in_time_ok,
                "first_signal": first_signal,
                "eligible_for_early_hit": eligible_for_early_hit,
                "hold_52w_return": hold_52w_return,
                "hold_52w_excess_return": hold_52w_excess_return,
                "hold_52w_exit_at": hold_52w_exit_at,
                "hit": hit,
            }
        )
    observations = pd.DataFrame(rows).set_index("decision_at") if rows else pd.DataFrame()
    if not observations.empty:
        eligible_mask = observations["eligible_for_early_hit"]
        eligible = int(eligible_mask.sum())
        hits = int(observations.loc[eligible_mask, "hit"].sum())
        persistent_count = int((observations["signal"] & observations["persistent"]).sum())
        matured_count = int((observations["signal"] & observations["matured"]).sum())
        first_hits = int(observations.loc[eligible_mask, "hit"].sum())
        signal_rows = observations["signal"]
        window_available = int((signal_rows & observations["within_onset_window"].notna()).sum())
        fraction_available = int(
            (signal_rows & observations["realized_log_fraction"].notna()).sum()
        )
        false_alerts = int((eligible_mask & ~observations["hit"]).sum())
        lead_lag_values: list[float] = []
        for timestamp, event_start in observations.loc[eligible_mask, "event_start"].items():
            if event_start is None or pd.isna(event_start):
                continue
            try:
                lead_lag_values.append(
                    (pd.Timestamp(timestamp) - pd.Timestamp(event_start)).days / 7.0
                )
            except (TypeError, ValueError):
                continue
        median_lead_lag = float(np.median(lead_lag_values)) if lead_lag_values else None
        remaining_values = observations.loc[eligible_mask, "remaining_move_fraction"].dropna()
        median_remaining = float(remaining_values.median()) if not remaining_values.empty else None
    else:
        eligible = hits = persistent_count = matured_count = first_hits = 0
        window_available = fraction_available = 0
        false_alerts = 0
        median_lead_lag = median_remaining = None
    return EarlyHitReport(
        observations=observations,
        eligible=eligible,
        hits=hits,
        hit_rate=(hits / eligible if eligible else None),
        persistent_signals=persistent_count,
        matured=matured_count,
        first_hits=first_hits,
        window_available=window_available,
        fraction_available=fraction_available,
        false_alerts=false_alerts,
        median_lead_lag_weeks=median_lead_lag,
        median_remaining_move_fraction=median_remaining,
    )


def early_hit_metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return only the aggregate early-hit metrics."""

    return evaluate_early_hits(*args, **kwargs).as_dict()


evaluate_early_hit = evaluate_early_hits


__all__ = [
    "DEFAULT_EARLY_WINDOW_BEFORE_WEEKS",
    "DEFAULT_EARLY_WINDOW_AFTER_WEEKS",
    "DEFAULT_MAX_REALIZED_LOG_FRACTION",
    "EarlyHitReport",
    "evaluate_early_hits",
    "evaluate_early_hit",
    "early_hit_metrics",
]
