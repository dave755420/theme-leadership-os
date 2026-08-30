"""Deterministic episode and forward-outcome labels.

The validation code deliberately keeps *decision time* separate from the first
tradable observation after that time.  This is a small detail, but it prevents
same-close look-ahead from leaking into an otherwise honest backtest.  All
functions in this module operate on synthetic or caller-provided series; they
do not fetch market data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_MAJOR_RETURN = 0.80
DEFAULT_MAJOR_EXCESS = 0.40
DEFAULT_EXPLOSIVE_RETURN = 1.00
DEFAULT_BREADTH = 0.60
DEFAULT_BREADTH_BEAT_SPY = 0.50
DEFAULT_MIN_NAMES = 4
DEFAULT_EPISODE_HORIZON_WEEKS = 52
DEFAULT_ONSET_LOOKBACK_WEEKS = 26
DEFAULT_EPISODE_MERGE_GAP_WEEKS = 13
DEFAULT_MIN_THEME_RANK = 0.90
DEFAULT_FUTURE_RETURN = 0.30
DEFAULT_FUTURE_EXCESS = 0.20


def _fraction(value: float | int | None) -> float | None:
    """Validate a decimal return/threshold.

    Public validation APIs use decimal returns (``0.80`` means 80%).  We do
    not guess whether ``80`` means 80% or an 80x return: callers must convert
    percentage inputs explicitly before invoking these functions.
    """

    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("return/threshold must be finite")
    return number


def _series(value: Any, *, name: str | None = None) -> pd.Series:
    """Coerce common array-like inputs to a sorted, de-duplicated Series."""

    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError("expected a Series or a one-column DataFrame")
        value = value.iloc[:, 0]
    if isinstance(value, pd.Series):
        out = value.copy()
    elif isinstance(value, Mapping):
        out = pd.Series(value, name=name)
    else:
        out = pd.Series(value, name=name)
    if name is not None:
        out.name = name
    if not isinstance(out.index, pd.DatetimeIndex):
        # Preserve integer/RangeIndex inputs: those are intentionally
        # positional weekly fixtures, not nanoseconds after ``to_datetime``.
        is_numeric_index = pd.api.types.is_numeric_dtype(out.index)
        if not is_numeric_index:
            try:
                out.index = pd.to_datetime(out.index)
            except (TypeError, ValueError):
                # Positional/non-date labels are valid; horizon offsets then
                # use rows rather than calendar arithmetic.
                pass
    if isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
    return out


def _date_index(series: pd.Series) -> pd.DatetimeIndex | None:
    return series.index if isinstance(series.index, pd.DatetimeIndex) else None


def _valid_pair(
    theme: pd.Series, benchmark: pd.Series | None
) -> tuple[pd.Series, pd.Series | None]:
    """Align valid price observations without forward filling."""

    t = pd.to_numeric(theme, errors="coerce")
    if benchmark is None:
        return t, None
    b = pd.to_numeric(benchmark, errors="coerce")
    aligned = pd.concat([t.rename("theme"), b.rename("benchmark")], axis=1).dropna()
    return aligned["theme"], aligned["benchmark"]


def _future_position(index: pd.Index, entry_position: int, weeks: int) -> int | None:
    """Find the first observation at or after ``entry + weeks``.

    Date indexes use calendar weeks and therefore naturally account for
    weekends and holidays.  For positional indexes, one row represents one
    week, which is the convention used by weekly synthetic fixtures.
    """

    if entry_position >= len(index):
        return None
    if isinstance(index, pd.DatetimeIndex):
        target = index[entry_position] + np.timedelta64(7 * int(weeks), "D")
        position = int(index.searchsorted(target, side="left"))
    else:
        position = entry_position + int(weeks)
    return position if position < len(index) else None


def _entry_position(index: pd.Index, decision_position: int, next_session: bool) -> int | None:
    position = decision_position + (1 if next_session else 0)
    return position if position < len(index) else None


def _date_or_position(index: pd.Index, position: int | None) -> Any:
    return None if position is None else index[position]


def _lookahead_return(
    prices: pd.Series,
    decision_position: int,
    weeks: int,
    *,
    next_session: bool,
) -> tuple[float, Any, Any] | None:
    """Return forward return and entry/exit labels for one decision row."""

    valid = pd.to_numeric(prices, errors="coerce").dropna()
    index = valid.index
    if isinstance(index, pd.DatetimeIndex):
        # decision_position is in the original aligned index; this helper is
        # only called on pair-aligned data, so positions refer to valid rows.
        pass
    entry = _entry_position(index, decision_position, next_session)
    if entry is None:
        return None
    exit_position = _future_position(index, entry, weeks)
    if exit_position is None:
        return None
    start = float(valid.iloc[entry])
    end = float(valid.iloc[exit_position])
    if not np.isfinite(start) or not np.isfinite(end) or start == 0:
        return None
    return (
        end / start - 1.0,
        _date_or_position(index, entry),
        _date_or_position(index, exit_position),
    )


@dataclass(frozen=True)
class EpisodeLabelConfig:
    """Thresholds for labels, kept independent from challenge-case metadata."""

    horizon_weeks: int = DEFAULT_EPISODE_HORIZON_WEEKS
    major_return: float = DEFAULT_MAJOR_RETURN
    major_excess: float = DEFAULT_MAJOR_EXCESS
    explosive_return: float = DEFAULT_EXPLOSIVE_RETURN
    min_breadth: float = DEFAULT_BREADTH
    min_breadth_beat_spy: float = DEFAULT_BREADTH_BEAT_SPY
    min_names: int = DEFAULT_MIN_NAMES
    require_breadth: bool = True
    require_min_names: bool = True
    max_move_before_signal: float | None = None
    onset_lookback_weeks: int = DEFAULT_ONSET_LOOKBACK_WEEKS
    merge_gap_weeks: int = DEFAULT_EPISODE_MERGE_GAP_WEEKS
    min_theme_rank: float = DEFAULT_MIN_THEME_RANK

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> EpisodeLabelConfig:
        """Construct from the nested keys used by ``config/episodes.yml``."""

        major = values.get("major_surge", values)
        explosive = values.get("explosive", {})
        onset = values.get("onset", {})
        return cls(
            horizon_weeks=int(
                major.get("horizon_weeks", values.get("horizon_weeks", cls.horizon_weeks))
            ),
            major_return=float(
                major.get("min_theme_gain", major.get("major_return", cls.major_return))
            ),
            major_excess=float(
                major.get("min_spy_excess", major.get("major_excess", cls.major_excess))
            ),
            explosive_return=float(
                explosive.get(
                    "min_theme_gain", explosive.get("explosive_return", cls.explosive_return)
                )
            ),
            min_breadth=float(
                major.get("min_breadth_up", major.get("min_breadth", cls.min_breadth))
            ),
            min_breadth_beat_spy=float(major.get("min_breadth_beat_spy", cls.min_breadth_beat_spy)),
            min_names=int(major.get("min_valid_names", major.get("min_names", cls.min_names))),
            onset_lookback_weeks=int(onset.get("lookback_weeks", cls.onset_lookback_weeks)),
            merge_gap_weeks=int(onset.get("merge_gap_weeks", cls.merge_gap_weeks)),
            min_theme_rank=float(
                major.get(
                    "min_theme_rank_percentile", major.get("min_theme_rank", cls.min_theme_rank)
                )
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "horizon_weeks": self.horizon_weeks,
            "major_return": self.major_return,
            "major_excess": self.major_excess,
            "explosive_return": self.explosive_return,
            "min_breadth": self.min_breadth,
            "min_breadth_beat_spy": self.min_breadth_beat_spy,
            "min_names": self.min_names,
            "onset_lookback_weeks": self.onset_lookback_weeks,
            "merge_gap_weeks": self.merge_gap_weeks,
            "min_theme_rank": self.min_theme_rank,
        }

    from_dict = from_mapping

    def __post_init__(self) -> None:
        for field in (
            "major_return",
            "major_excess",
            "explosive_return",
            "min_breadth",
            "max_move_before_signal",
        ):
            value = getattr(self, field)
            if value is not None and field != "min_breadth":
                object.__setattr__(self, field, _fraction(value))
        for field in ("min_breadth", "min_breadth_beat_spy"):
            value = float(getattr(self, field))
            if not 0 <= value <= 1:
                raise ValueError(f"{field} must be between 0 and 1")
            object.__setattr__(self, field, value)
        if self.horizon_weeks <= 0:
            raise ValueError("horizon_weeks must be positive")
        if self.min_names < 0:
            raise ValueError("min_names cannot be negative")
        if self.onset_lookback_weeks <= 0 or self.merge_gap_weeks < 0:
            raise ValueError("onset lookback must be positive and merge gap non-negative")
        rank = float(self.min_theme_rank)
        if not 0 <= rank <= 1:
            raise ValueError("min_theme_rank must be between 0 and 1")
        object.__setattr__(self, "min_theme_rank", rank)


@dataclass(frozen=True)
class EpisodeLabel:
    """One deterministic episode classification."""

    start: Any
    end: Any | None
    theme_return: float
    spy_return: float | None
    excess_return: float | None
    breadth: float | None
    breadth_beat_spy: float | None
    active_names: int | None
    breadth_ok: bool
    breadth_up_ok: bool
    breadth_beat_spy_ok: bool
    min_names_ok: bool
    top_decile_ok: bool
    major_surge: bool
    explosive: bool
    label: str

    @property
    def qualifies(self) -> bool:
        return self.label != "none"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"qualifies": self.qualifies}


def classify_episode(
    theme_return: float,
    spy_return: float | None = None,
    *,
    breadth: float | Mapping[str, float] | None = None,
    active_names: int | None = None,
    config: EpisodeLabelConfig | None = None,
    start: Any = None,
    end: Any = None,
    major_return: float | None = None,
    major_excess: float | None = None,
    explosive_return: float | None = None,
    min_breadth: float | None = None,
    breadth_up: float | None = None,
    breadth_beat_spy: float | None = None,
    min_breadth_beat_spy: float | None = None,
    min_names: int | None = None,
    theme_rank_percentile: float | None = None,
    top_decile: bool | None = None,
    min_theme_rank: float | None = None,
) -> EpisodeLabel:
    """Classify an already measured move.

    A major surge is at least 80% absolute and 40 percentage points ahead of
    SPY, with the configured breadth and minimum-name gates.  An explosive move
    is at least 100% absolute, subject to the same breadth/name gates.  The
    thresholds are deliberately plain constants/configuration; challenge cases
    never alter them.
    """

    cfg = config or EpisodeLabelConfig()
    abs_return = _fraction(theme_return)
    spy = _fraction(spy_return)
    excess = None if spy is None else abs_return - spy
    # ``breadth`` is the historical shorthand for upward breadth.  New
    # callers can provide separate upward and SPY-relative breadth values.
    breadth_up_value = breadth_up if breadth_up is not None else breadth
    breadth_beat_value = breadth_beat_spy
    if isinstance(breadth, Mapping):
        breadth_up_value = breadth.get(
            "up", breadth.get("breadth_up", breadth.get("above_ma", breadth_up_value))
        )
        breadth_beat_value = breadth.get(
            "beat_spy", breadth.get("breadth_beat_spy", breadth_beat_value)
        )
    br_threshold = cfg.min_breadth if min_breadth is None else float(min_breadth)
    if not 0 <= br_threshold <= 1:
        raise ValueError("min_breadth must be a decimal fraction in [0, 1]")
    beat_threshold = (
        cfg.min_breadth_beat_spy if min_breadth_beat_spy is None else float(min_breadth_beat_spy)
    )
    if not 0 <= beat_threshold <= 1:
        raise ValueError("min_breadth_beat_spy must be a decimal fraction in [0, 1]")
    names_threshold = cfg.min_names if min_names is None else int(min_names)
    major_abs_threshold = cfg.major_return if major_return is None else _fraction(major_return)
    major_excess_threshold = cfg.major_excess if major_excess is None else _fraction(major_excess)
    explosive_threshold = (
        cfg.explosive_return if explosive_return is None else _fraction(explosive_return)
    )
    rank_threshold = cfg.min_theme_rank if min_theme_rank is None else float(min_theme_rank)
    if not 0 <= rank_threshold <= 1:
        raise ValueError("min_theme_rank must be a decimal fraction in [0, 1]")
    if top_decile is not None:
        top_rank_ok = bool(top_decile)
    elif theme_rank_percentile is None:
        # A scalar move classifier has no cross-sectional universe and leaves
        # this gate unevaluated.  ``label_episodes`` supplies the percentile
        # whenever it is available, where it is enforced deterministically.
        top_rank_ok = True
    else:
        top_rank_ok = float(theme_rank_percentile) >= rank_threshold

    breadth_up_ok = (
        True
        if not cfg.require_breadth
        else breadth_up_value is not None and float(breadth_up_value) >= br_threshold
    )
    # Relative breadth is required only when it is available.  This follows
    # the frozen protocol's "where available" rule while still requiring the
    # upward breadth gate whenever breadth metadata is present.
    breadth_beat_spy_ok = (
        True
        if breadth_beat_value is None or not cfg.require_breadth
        else float(breadth_beat_value) >= beat_threshold
    )
    breadth_ok = breadth_up_ok and breadth_beat_spy_ok
    min_names_ok = (
        True
        if not cfg.require_min_names
        else active_names is not None and int(active_names) >= names_threshold
    )
    excess_ok = excess is not None and excess >= float(major_excess_threshold)
    major = (
        abs_return >= float(major_abs_threshold)
        and excess_ok
        and breadth_ok
        and min_names_ok
        and top_rank_ok
    )
    # Explosive is intentionally a strict subset of Major Surge.  In
    # particular, a 100% move without the SPY-excess gate is not explosive.
    explosive = major and abs_return >= float(explosive_threshold)
    label = "explosive" if explosive else ("major_surge" if major else "none")
    return EpisodeLabel(
        start=start,
        end=end,
        theme_return=float(abs_return),
        spy_return=None if spy is None else float(spy),
        excess_return=None if excess is None else float(excess),
        breadth=None if breadth_up_value is None else float(breadth_up_value),
        breadth_beat_spy=None if breadth_beat_value is None else float(breadth_beat_value),
        active_names=None if active_names is None else int(active_names),
        breadth_ok=bool(breadth_ok),
        breadth_up_ok=bool(breadth_up_ok),
        breadth_beat_spy_ok=bool(breadth_beat_spy_ok),
        min_names_ok=bool(min_names_ok),
        top_decile_ok=bool(top_rank_ok),
        major_surge=bool(major),
        explosive=bool(explosive),
        label=label,
    )


def future_labels(
    theme_prices: pd.Series | pd.DataFrame | Mapping[Any, float],
    benchmark_prices: pd.Series | Mapping[Any, float] | None = None,
    *,
    horizons: Iterable[int] = (13, 26, 52),
    next_session: bool = True,
    absolute_threshold: float | None = DEFAULT_FUTURE_RETURN,
    excess_threshold: float | None = DEFAULT_FUTURE_EXCESS,
    label_name: str = "label",
    parent_sector_prices: pd.Series | pd.DataFrame | Mapping[Any, float] | None = None,
) -> pd.DataFrame:
    """Build 13/26/52-week labels indexed by decision date.

    For a decision at ``t``, the entry is the first available observation
    strictly after ``t`` when ``next_session=True``.  The exit is the first
    available observation at or after ``entry + horizon weeks``.  Consequently
    ``label_available_at_*`` is always after the decision and can be used as a
    label-maturity timestamp in split generation.

    Returns include raw and benchmark-relative returns even when thresholds
    are disabled.  ``label_*`` and ``future_*_label`` are boolean with NA when
    the horizon has not matured or a price is unavailable.
    """

    if isinstance(theme_prices, pd.DataFrame):
        return future_panel_labels(
            theme_prices,
            benchmark_prices,
            horizons=horizons,
            next_session=next_session,
            parent_sector_prices=parent_sector_prices,
        )
    t, b = _valid_pair(
        _series(theme_prices, name="theme"),
        None if benchmark_prices is None else _series(benchmark_prices, name="benchmark"),
    )
    if b is not None:
        aligned = pd.concat([t.rename("theme"), b.rename("benchmark")], axis=1).dropna()
        t, b = aligned["theme"], aligned["benchmark"]
    else:
        t = t.dropna()
    if t.empty:
        return pd.DataFrame(index=t.index)
    horizons_tuple = tuple(int(h) for h in horizons)
    if any(h <= 0 for h in horizons_tuple):
        raise ValueError("horizons must contain positive week counts")
    abs_threshold = _fraction(absolute_threshold)
    ex_threshold = _fraction(excess_threshold)
    rows: list[dict[str, Any]] = []
    index = t.index
    for i, decision in enumerate(index):
        row: dict[str, Any] = {"decision_at": decision}
        for h in horizons_tuple:
            entry_position = _entry_position(index, i, next_session)
            exit_position = (
                None if entry_position is None else _future_position(index, entry_position, h)
            )
            entry_at = _date_or_position(index, entry_position)
            exit_at = _date_or_position(index, exit_position)
            theme_return = np.nan
            bench_return = np.nan
            excess_return = np.nan
            max_relative_drawdown = np.nan
            parent_excess_return = np.nan
            if entry_position is not None and exit_position is not None:
                entry_value = float(t.iloc[entry_position])
                exit_value = float(t.iloc[exit_position])
                if np.isfinite(entry_value) and np.isfinite(exit_value) and entry_value != 0:
                    theme_return = exit_value / entry_value - 1.0
                if b is not None:
                    b_entry = float(b.iloc[entry_position])
                    b_exit = float(b.iloc[exit_position])
                    if np.isfinite(b_entry) and np.isfinite(b_exit) and b_entry != 0:
                        bench_return = b_exit / b_entry - 1.0
                        if np.isfinite(theme_return):
                            excess_return = theme_return - bench_return
                else:
                    excess_return = theme_return
                path = pd.to_numeric(
                    t.iloc[entry_position : exit_position + 1], errors="coerce"
                ).dropna()
                if not path.empty and path.iloc[0] != 0:
                    path_relative = path / float(path.iloc[0])
                    max_relative_drawdown = float(
                        (path_relative / path_relative.cummax() - 1.0).min()
                    )
                if parent_sector_prices is not None:
                    parent = _series(parent_sector_prices, name="parent_sector")
                    parent = parent.reindex(index)
                    parent_entry = parent.iloc[entry_position]
                    parent_exit = parent.iloc[exit_position]
                    if pd.notna(parent_entry) and pd.notna(parent_exit) and parent_entry != 0:
                        parent_excess_return = theme_return - (
                            float(parent_exit) / float(parent_entry) - 1.0
                        )
            valid = np.isfinite(theme_return) and (
                ex_threshold is None or np.isfinite(excess_return)
            )
            label_value: bool | float = np.nan
            if valid:
                label_value = bool(
                    (abs_threshold is None or theme_return >= abs_threshold)
                    and (ex_threshold is None or excess_return >= ex_threshold)
                )
            # Both spellings are intentional: older callers use label_13w,
            # while reports tend to use future_13w_label.
            suffix = f"{h}w"
            row[f"entry_at_{suffix}"] = entry_at
            row[f"exit_at_{suffix}"] = exit_at
            row[f"label_available_at_{suffix}"] = exit_at
            row[f"maturity_at_{suffix}"] = exit_at
            row[f"label_end_{suffix}"] = exit_at
            row[f"future_{suffix}_return"] = theme_return
            row[f"future_{suffix}_benchmark_return"] = bench_return
            row[f"future_{suffix}_excess_return"] = excess_return
            row[f"future_{suffix}_max_relative_drawdown"] = max_relative_drawdown
            row[f"future_{suffix}_parent_sector_excess_return"] = parent_excess_return
            row[f"future_return_{suffix}"] = theme_return
            row[f"future_excess_return_{suffix}"] = excess_return
            row[f"{label_name}_{suffix}"] = label_value
            row[f"future_{suffix}_label"] = label_value
            row[f"{label_name}_{h}"] = label_value
            row[f"future_{suffix}"] = label_value
            if h == 52:
                row["exact_52w_hold_return"] = theme_return
        rows.append(row)
    out = pd.DataFrame(rows).set_index("decision_at")
    out.index.name = index.name or "decision_at"
    return out


def future_panel_labels(
    theme_prices: pd.DataFrame,
    benchmark_prices: pd.Series | Mapping[Any, float] | None = None,
    *,
    horizons: Iterable[int] = (13, 26, 52),
    next_session: bool = True,
    parent_sector_prices: pd.DataFrame | Mapping[Any, Any] | None = None,
) -> pd.DataFrame:
    """Build long-form future labels and cross-sectional ``Leader_h`` flags.

    The panel is evaluated independently per theme, then future-return ranks
    are computed across themes at each decision date.  ``Leader_h`` requires
    positive SPY-relative return and a top-20% future-return rank, exactly as
    frozen in the validation protocol.  Parent-sector excess and drawdown are
    retained as outcome columns when the corresponding prices are supplied.
    """

    if not isinstance(theme_prices, pd.DataFrame):
        raise TypeError("future_panel_labels expects a DataFrame of theme prices")
    benchmark = None if benchmark_prices is None else _series(benchmark_prices, name="benchmark")
    pieces: list[pd.DataFrame] = []
    for theme in theme_prices.columns:
        parent = None
        if parent_sector_prices is not None:
            if (
                isinstance(parent_sector_prices, pd.DataFrame)
                and theme in parent_sector_prices.columns
            ):
                parent = parent_sector_prices[theme]
            elif isinstance(parent_sector_prices, Mapping) and theme in parent_sector_prices:
                parent = parent_sector_prices[theme]
        one = future_labels(
            theme_prices[theme],
            benchmark,
            horizons=horizons,
            next_session=next_session,
            absolute_threshold=None,
            excess_threshold=None,
            parent_sector_prices=parent,
        ).reset_index()
        one.insert(1, "theme", theme)
        pieces.append(one)
    if not pieces:
        return pd.DataFrame(columns=["decision_at", "theme"])
    out = pd.concat(pieces, ignore_index=True)
    for h in tuple(int(value) for value in horizons):
        suffix = f"{h}w"
        return_column = f"future_{suffix}_return"
        rank_column = f"future_{suffix}_return_rank"
        out[rank_column] = out.groupby("decision_at")[return_column].rank(
            pct=True, method="average"
        )
        out[f"Leader_{suffix}"] = out[f"future_{suffix}_excess_return"].gt(0) & out[rank_column].ge(
            0.80
        )
        # Lower-case alias is convenient for column-oriented clients.
        out[f"leader_{suffix}"] = out[f"Leader_{suffix}"]
    return out.set_index(["decision_at", "theme"]).sort_index()


def label_future_returns(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Backward-compatible descriptive alias for :func:`future_labels`."""

    return future_labels(*args, **kwargs)


def _value_at(value: Any, timestamp: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return value.get(timestamp, default)
    if isinstance(value, pd.Series):
        if timestamp in value.index:
            result = value.loc[timestamp]
            return None if pd.isna(result) else result
        return default
    if isinstance(value, pd.DataFrame):
        if timestamp in value.index:
            result = value.loc[timestamp]
            return result.to_dict() if isinstance(result, pd.Series) else result
        return default
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return default
    return value


def _episode_measurement(
    prices: pd.Series,
    benchmark: pd.Series | None,
    index: pd.Index,
    decision_position: int,
    config: EpisodeLabelConfig,
    *,
    max_forward: bool,
    next_session: bool,
) -> dict[str, Any] | None:
    """Find onset/crossing/peak positions for one decision row."""

    entry_position = _entry_position(index, decision_position, next_session)
    terminal_position = (
        None
        if entry_position is None
        else _future_position(index, entry_position, config.horizon_weeks)
    )
    if entry_position is None or terminal_position is None:
        return None
    future = pd.to_numeric(prices.iloc[entry_position : terminal_position + 1], errors="coerce")
    if future.empty or pd.isna(future.iloc[0]):
        return None
    entry_value = float(future.iloc[0])
    relative = future / entry_value - 1.0
    if relative.dropna().empty:
        return None
    crossing_position = None
    onset_position = None
    # A crossing is measured from the lowest weekly close in the preceding
    # 26-week lookback, rather than from an arbitrary decision close.  This is
    # the frozen onset definition and lets the same episode be found from
    # multiple decision dates without changing its threshold.
    for local_position, timestamp in enumerate(future.index):
        absolute_position = entry_position + local_position
        onset_low = max(0, absolute_position - config.onset_lookback_weeks)
        candidate_window = pd.to_numeric(
            prices.iloc[onset_low : absolute_position + 1], errors="coerce"
        )
        if candidate_window.dropna().empty:
            continue
        candidate_onset = int(index.get_loc(candidate_window.idxmin()))
        candidate_base = float(prices.iloc[candidate_onset])
        if (
            candidate_base > 0
            and float(prices.loc[timestamp]) / candidate_base - 1.0 >= config.major_return
        ):
            crossing_position = absolute_position
            onset_position = candidate_onset
            break
    peak_terminal_position = terminal_position
    if crossing_position is not None:
        subsequent = _future_position(index, crossing_position, config.horizon_weeks)
        if subsequent is not None:
            peak_terminal_position = subsequent
            if peak_terminal_position > terminal_position:
                future = pd.to_numeric(
                    prices.iloc[entry_position : peak_terminal_position + 1], errors="coerce"
                )
                relative = future / entry_value - 1.0
    end_position = terminal_position
    if max_forward:
        end_position = int(index.get_loc(relative.idxmax()))
    start_position = entry_position
    if crossing_position is not None and onset_position is not None:
        # The onset is the lowest weekly close in the 26 weeks before the
        # first 80% crossing, inclusive of the crossing week's history.
        start_position = onset_position
        # The peak is the maximum close subsequent to the crossing through the
        # fixed 52-week outcome window.
        peak_window = pd.to_numeric(
            prices.iloc[crossing_position : peak_terminal_position + 1], errors="coerce"
        )
        if peak_window.notna().any():
            end_position = int(index.get_loc(peak_window.idxmax()))
    return {
        "entry_position": entry_position,
        "terminal_position": terminal_position,
        "start_position": start_position,
        "end_position": end_position,
        "crossing_position": crossing_position,
        "entry_return": float(prices.iloc[end_position] / prices.iloc[entry_position] - 1.0),
        "theme_return": float(prices.iloc[end_position] / prices.iloc[start_position] - 1.0),
        "spy_return": (
            None
            if benchmark is None
            else float(benchmark.iloc[end_position] / benchmark.iloc[start_position] - 1.0)
        ),
    }


def label_episodes(
    theme_prices: pd.Series | Mapping[Any, float],
    benchmark_prices: pd.Series | Mapping[Any, float] | None = None,
    *,
    breadth: pd.Series | pd.DataFrame | Mapping[Any, float] | float | None = None,
    active_names: pd.Series | Mapping[Any, int] | int | None = None,
    theme_rank_percentile: pd.Series | Mapping[Any, float] | float | None = None,
    top_decile: pd.Series | Mapping[Any, bool] | bool | None = None,
    config: EpisodeLabelConfig | None = None,
    horizon_weeks: int | None = None,
    next_session: bool = True,
    max_forward: bool = True,
) -> pd.DataFrame:
    """Label every decision date using deterministic episode thresholds.

    ``max_forward=False`` labels the return at the horizon endpoint.  With
    ``max_forward=True`` the best endpoint inside the horizon is selected (and
    reported as ``end``); this is useful when an episode's peak occurs before
    the fixed evaluation horizon.  No threshold is inferred from the data.
    """

    cfg = config or EpisodeLabelConfig()
    h = int(cfg.horizon_weeks if horizon_weeks is None else horizon_weeks)
    # A caller-specified horizon is a label setting, not a way to change the
    # frozen onset lookback/merge rules.  Build a local immutable config so the
    # helper uses the requested horizon while retaining all other gates.
    if h != cfg.horizon_weeks:
        cfg = EpisodeLabelConfig(
            horizon_weeks=h,
            major_return=cfg.major_return,
            major_excess=cfg.major_excess,
            explosive_return=cfg.explosive_return,
            min_breadth=cfg.min_breadth,
            min_breadth_beat_spy=cfg.min_breadth_beat_spy,
            min_names=cfg.min_names,
            require_breadth=cfg.require_breadth,
            require_min_names=cfg.require_min_names,
            max_move_before_signal=cfg.max_move_before_signal,
            onset_lookback_weeks=cfg.onset_lookback_weeks,
            merge_gap_weeks=cfg.merge_gap_weeks,
            min_theme_rank=cfg.min_theme_rank,
        )
    t, b = _valid_pair(
        _series(theme_prices, name="theme"),
        None if benchmark_prices is None else _series(benchmark_prices, name="benchmark"),
    )
    if b is not None:
        aligned = pd.concat([t.rename("theme"), b.rename("benchmark")], axis=1).dropna()
        t, b = aligned["theme"], aligned["benchmark"]
    else:
        t = t.dropna()
    idx = t.index
    rows: list[dict[str, Any]] = []
    for i, decision in enumerate(idx):
        measurement = _episode_measurement(
            t,
            b,
            idx,
            i,
            cfg,
            max_forward=max_forward,
            next_session=next_session,
        )
        entry_position = None if measurement is None else measurement["entry_position"]
        if measurement is None:
            row = {
                "decision_at": decision,
                "entry_at": _date_or_position(idx, entry_position),
                "onset": None,
                "first_80_crossing": None,
                "end": None,
                "peak": None,
                "label_available_at": None,
                "entry_to_peak_return": np.nan,
                "theme_return": np.nan,
                "spy_return": np.nan,
                "excess_return": np.nan,
                "breadth": _value_at(breadth, decision),
                "breadth_up": _value_at(breadth, decision),
                "breadth_beat_spy": np.nan,
                "active_names": _value_at(active_names, decision),
                "breadth_ok": False,
                "breadth_up_ok": False,
                "breadth_beat_spy_ok": False,
                "min_names_ok": False,
                "top_decile_ok": False,
                "label": None,
                "major_surge": False,
                "explosive": False,
                "qualifies": False,
                "matured": False,
            }
            rows.append(row)
            continue
        start_position = measurement["start_position"]
        end_position = measurement["end_position"]
        theme_return = measurement["theme_return"]
        spy_return = measurement["spy_return"]
        breadth_value = _value_at(breadth, decision)
        breadth_up_value = breadth_value
        breadth_beat_value = None
        if isinstance(breadth_value, Mapping):
            breadth_up_value = breadth_value.get(
                "up", breadth_value.get("breadth_up", breadth_value.get("above_ma"))
            )
            breadth_beat_value = breadth_value.get(
                "beat_spy", breadth_value.get("breadth_beat_spy")
            )
        episode = classify_episode(
            theme_return,
            spy_return,
            breadth=breadth_value,
            breadth_up=breadth_up_value,
            breadth_beat_spy=breadth_beat_value,
            active_names=_value_at(active_names, decision),
            theme_rank_percentile=_value_at(theme_rank_percentile, decision),
            top_decile=_value_at(top_decile, decision),
            config=cfg,
            start=_date_or_position(idx, start_position),
            end=_date_or_position(idx, end_position),
        )
        if measurement["crossing_position"] is None:
            # A terminal 100% return from the decision close is not enough to
            # call an episode: the protocol requires a first 80% crossing
            # measured from the preceding 26-week onset low.
            episode = replace(episode, major_surge=False, explosive=False, label="none")
        rows.append(
            {
                "decision_at": decision,
                "entry_at": _date_or_position(idx, entry_position),
                "onset": _date_or_position(idx, start_position),
                "first_80_crossing": (
                    None
                    if measurement["crossing_position"] is None
                    else _date_or_position(idx, measurement["crossing_position"])
                ),
                "end": _date_or_position(idx, end_position),
                "peak": _date_or_position(idx, end_position),
                "entry_to_peak_return": measurement["entry_return"],
                "label_available_at": _date_or_position(idx, end_position),
                "theme_return": episode.theme_return,
                "spy_return": episode.spy_return,
                "excess_return": episode.excess_return,
                "breadth": episode.breadth,
                "breadth_up": episode.breadth,
                "breadth_beat_spy": breadth_beat_value,
                "active_names": episode.active_names,
                "theme_rank_percentile": _value_at(theme_rank_percentile, decision),
                "breadth_ok": episode.breadth_ok,
                "breadth_up_ok": episode.breadth_up_ok,
                "breadth_beat_spy_ok": episode.breadth_beat_spy_ok,
                "min_names_ok": episode.min_names_ok,
                "top_decile_ok": episode.top_decile_ok,
                "label": episode.label,
                "major_surge": episode.major_surge,
                "explosive": episode.explosive,
                "qualifies": episode.qualifies,
                "matured": True,
            }
        )
    return pd.DataFrame(rows).set_index("decision_at")


def detect_episodes(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Return one row per non-overlapping qualifying episode start.

    This convenience view is built from :func:`label_episodes`; the full
    decision-date table remains available from that function for auditability.
    Consecutive qualifying dates are collapsed until the previous episode's
    reported end, making the result deterministic and easy to inspect.
    """

    labels = label_episodes(*args, **kwargs)
    if labels.empty:
        return labels.assign(episode_id=pd.Series(dtype=int))
    qualifying = labels[labels["qualifies"].fillna(False)].copy()
    if qualifying.empty:
        return qualifying.assign(episode_id=pd.Series(dtype=int))
    config = kwargs.get("config") or EpisodeLabelConfig()
    merge_gap = config.merge_gap_weeks
    starts: list[Any] = []
    last_start: Any = None
    last_end: Any = None
    for timestamp, row in qualifying.iterrows():
        if last_start is not None:
            if isinstance(qualifying.index, pd.DatetimeIndex):
                too_close = timestamp - last_start < np.timedelta64(7 * merge_gap, "D")
            else:
                too_close = timestamp - last_start < merge_gap
            if too_close:
                continue
        if last_end is not None and timestamp <= last_end:
            continue
        starts.append(timestamp)
        last_start = timestamp
        last_end = row.get("end")
    result = qualifying.loc[starts].copy()
    result.insert(0, "episode_id", np.arange(1, len(result) + 1, dtype=int))
    result.index.name = "start"
    return result


# Names used by early prototypes and external notebooks.
build_episode_labels = label_episodes
classify_move = classify_episode
label_episode = classify_episode
episode_labels = label_episodes
make_future_labels = future_labels
future_13_26_52_labels = future_labels


__all__ = [
    "DEFAULT_MAJOR_RETURN",
    "DEFAULT_MAJOR_EXCESS",
    "DEFAULT_EXPLOSIVE_RETURN",
    "DEFAULT_BREADTH_BEAT_SPY",
    "DEFAULT_EPISODE_HORIZON_WEEKS",
    "DEFAULT_ONSET_LOOKBACK_WEEKS",
    "DEFAULT_EPISODE_MERGE_GAP_WEEKS",
    "DEFAULT_MIN_THEME_RANK",
    "EpisodeLabelConfig",
    "EpisodeLabel",
    "classify_episode",
    "classify_move",
    "label_episode",
    "future_labels",
    "future_panel_labels",
    "label_future_returns",
    "label_episodes",
    "build_episode_labels",
    "episode_labels",
    "make_future_labels",
    "future_13_26_52_labels",
    "detect_episodes",
]
