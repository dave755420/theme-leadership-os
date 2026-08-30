"""Deterministic, no-lookahead theme leadership signals.

The module intentionally keeps the calculations in ordinary pandas objects.
Every output is a tidy frame with one row per ``decision_date``/``theme`` and
exposes the component values used by its gate.  There is no hidden model or
learned threshold: configuration dataclasses contain all thresholds and
weights.

Input
-----
``date`` is an observation/session date, ``theme`` is the theme membership,
``security`` identifies a member, and ``value`` is a positive price/index
level.  ``SPY`` (or ``SignalConfig.benchmark_security``) is extracted as the
benchmark and is not treated as a theme member.  Daily observations are
reduced to Friday-ending weeks using only observations on or before that
Friday.  Metrics at a Friday therefore cannot use a later session.  The
``effective_date`` column is the next observed session after the decision
Friday (or the following business day for a weekly-only extract) and is a
label only; it is never used in a calculation.

M0 is deliberately explicit.  ``m0_26w_spy_relative`` is the 26-week theme
return minus the 26-week SPY return, known at the decision Friday.  It is
repeated in every signal output together with ``m0_baseline_pass`` and its
cross-sectional percentile rank.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = frozenset({"date", "theme", "security", "value"})
WEEKLY_FREQUENCY = "W-FRI"
NOT_A_BUY_RECOMMENDATION = "WATCH_ONLY_NOT_A_BUY_RECOMMENDATION"


@dataclass(frozen=True)
class SignalConfig:
    """Shared data and calendar configuration.

    ``min_participation`` is a fraction of a theme's known members that must
    have a valid weekly return before that week's theme return is used.  The
    benchmark is identified by security, not by theme, because source files
    commonly place SPY in a separate pseudo-theme.
    """

    benchmark_security: str = "SPY"
    frequency: str = WEEKLY_FREQUENCY
    min_participation: float = 0.50
    min_value: float = 0.0
    m0_window: int = 26
    min_history_weeks: int = 52

    def __post_init__(self) -> None:
        if not self.benchmark_security:
            raise ValueError("benchmark_security must be non-empty")
        if self.frequency != WEEKLY_FREQUENCY:
            raise ValueError("signals currently require Friday-ending W-FRI weeks")
        if not 0.0 < self.min_participation <= 1.0:
            raise ValueError("min_participation must be in (0, 1]")
        if not np.isfinite(self.min_value) or self.min_value < 0.0:
            raise ValueError("min_value must be a finite, non-negative number")
        if self.m0_window <= 0:
            raise ValueError("m0_window must be positive")
        if self.min_history_weeks <= 0:
            raise ValueError("min_history_weeks must be positive")


@dataclass(frozen=True)
class CurrentLeaderConfig(SignalConfig):
    """Thresholds for the confirmed current-leadership screen."""

    windows: tuple[int, int, int] = (13, 26, 52)
    min_breadth: float = 0.50
    min_relative_strength: float = 0.0
    min_score: float = 0.60
    leader_count: int = 1
    score_weights: tuple[float, float, float, float] = (0.30, 0.35, 0.20, 0.15)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.windows != (13, 26, 52):
            raise ValueError("CurrentLeader windows are fixed at (13, 26, 52)")
        if not 0.0 <= self.min_breadth <= 1.0:
            raise ValueError("min_breadth must be in [0, 1]")
        if self.leader_count <= 0:
            raise ValueError("leader_count must be positive")
        if len(self.score_weights) != 4 or not np.isclose(sum(self.score_weights), 1.0):
            raise ValueError("score_weights must contain four values summing to one")


@dataclass(frozen=True)
class EmergingRadarConfig(SignalConfig):
    """High-sensitivity watchlist thresholds.

    Radar intentionally permits a weak 13-week trend.  It still records the
    M0 baseline and uses breadth/participation/concentration warnings to make
    a single-security rally visible rather than presenting it as a clean
    signal.  ``recommendation`` is descriptive and cannot be configured into
    an order instruction.
    """

    windows: tuple[int, int, int] = (4, 8, 13)
    # Radar can work with a shorter history than the 12M hold screen.  M0 is
    # still shown (and warned when unavailable), but 26 valid weekly returns
    # are required so its named 26-week baseline is available to the screen.
    min_history_weeks: int = 26
    min_acceleration_rank: float = 0.60
    min_rank_improvement: float = 0.05
    min_fast_breadth: float = 0.35
    min_participation_for_watch: float = 0.50
    max_concentration: float = 0.65
    min_fast_relative_strength: float = -0.02
    score_threshold: float = 0.55

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.windows != (4, 8, 13):
            raise ValueError("EmergingRadar windows are fixed at (4, 8, 13)")
        for name, value in (
            ("min_acceleration_rank", self.min_acceleration_rank),
            ("min_fast_breadth", self.min_fast_breadth),
            ("min_participation_for_watch", self.min_participation_for_watch),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= self.max_concentration <= 1.0:
            raise ValueError("max_concentration must be in [0, 1]")


@dataclass(frozen=True)
class HoldCandidate12MConfig(SignalConfig):
    """Confirmed relative-strength, durability and risk gates for a 12M hold."""

    windows: tuple[int, int, int] = (13, 26, 52)
    min_relative_strength: float = 0.0
    min_breadth: float = 0.50
    min_durability_score: float = 0.60
    confirmation_window: int = 3
    min_confirmed_weeks: int = 2
    max_concentration: float = 0.65
    max_annualized_volatility: float = 0.65
    max_drawdown: float = 0.45
    min_score: float = 0.60

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.windows != (13, 26, 52):
            raise ValueError("HoldCandidate12M windows are fixed at (13, 26, 52)")
        if not 0.0 <= self.min_breadth <= 1.0:
            raise ValueError("min_breadth must be in [0, 1]")
        if not 0.0 <= self.min_durability_score <= 1.0:
            raise ValueError("min_durability_score must be in [0, 1]")
        if self.confirmation_window <= 0 or self.min_confirmed_weeks <= 0:
            raise ValueError("confirmation settings must be positive")
        if self.min_confirmed_weeks > self.confirmation_window:
            raise ValueError("min_confirmed_weeks cannot exceed confirmation_window")
        if not 0.0 <= self.max_concentration <= 1.0:
            raise ValueError("max_concentration must be in [0, 1]")
        if self.max_annualized_volatility <= 0.0:
            raise ValueError("max_annualized_volatility must be positive")
        if not 0.0 < self.max_drawdown <= 1.0:
            raise ValueError("max_drawdown must be in (0, 1]")


@dataclass
class _Prepared:
    """Internal weekly panels shared by all three public screens."""

    weeks: pd.DatetimeIndex
    themes: tuple[str, ...]
    members: Mapping[str, tuple[str, ...]]
    benchmark: pd.Series
    benchmark_returns: pd.Series
    security_prices: pd.DataFrame
    security_returns: pd.DataFrame
    theme_returns: pd.DataFrame
    theme_index: pd.DataFrame
    sessions: pd.DatetimeIndex


def _empty_output() -> pd.DataFrame:
    """Return a stable empty schema for malformed-but-empty input."""

    return pd.DataFrame(
        columns=[
            "decision_date",
            "effective_date",
            "theme",
            "status",
            "warning",
            "warnings",
            "recommendation",
            "m0_26w_spy_relative",
            "m0_baseline_pass",
        ]
    )


def _coerce_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    # A mixed timezone column can become object dtype.  Convert each value to
    # a timezone-naive timestamp without introducing a future observation.
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        return dates.dt.tz_convert(None)
    if dates.dtype == object:
        return dates.map(
            lambda value: value.tz_convert(None)
            if isinstance(value, pd.Timestamp) and value.tzinfo is not None
            else value
        )
    return dates


def _normalise_input(
    data: pd.DataFrame, benchmark_security: str
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"tidy signal input is missing required columns: {missing_text}")

    frame = data.loc[:, ["date", "theme", "security", "value"]].copy()
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    frame["date"] = _coerce_dates(frame["date"])
    frame["theme"] = frame["theme"].astype("string").str.strip()
    # Ticker case is presentation-only; treating ``spy`` and ``SPY`` as
    # different instruments would silently remove the benchmark and produce a
    # non-actionable signal panel.
    frame["security"] = frame["security"].astype("string").str.strip().str.upper()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    valid_dates = frame["date"].notna()
    valid_names = (
        frame["theme"].notna()
        & frame["security"].notna()
        & frame["theme"].ne("")
        & frame["security"].ne("")
    )
    sessions = pd.DatetimeIndex(frame.loc[valid_dates, "date"].drop_duplicates().sort_values())
    frame = frame.loc[valid_dates & valid_names].copy()
    frame["week"] = frame["date"].dt.to_period("W-FRI").dt.end_time.dt.normalize()
    # Sort on the source order so duplicate observations resolve predictably.
    # Groupby.last(skipna) retains the latest valid value within a week.
    frame = frame.sort_values(["date", "theme", "security", "_row_order"], kind="mergesort")
    return frame, sessions


def _weekly_price_panels(
    frame: pd.DataFrame,
    benchmark_security: str,
    min_value: float = 0.0,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    tuple[str, ...],
    Mapping[str, tuple[str, ...]],
    pd.DatetimeIndex,
]:
    if frame.empty:
        return (
            pd.DataFrame(index=pd.DatetimeIndex([], name="decision_date")),
            pd.Series(dtype=float, name=benchmark_security),
            (),
            {},
            pd.DatetimeIndex([], name="decision_date"),
        )

    all_weeks = pd.date_range(frame["week"].min(), frame["week"].max(), freq=WEEKLY_FREQUENCY)
    # ``last`` is selected after sorting by date/source row.  Therefore a
    # Friday observation is used for Friday and a Wednesday observation is
    # used only when Friday is absent, never a later week's observation.
    values = frame["value"].where(frame["value"] > min_value)
    grouped = (
        frame.assign(_positive_value=values)
        .groupby(["week", "theme", "security"], sort=True)["_positive_value"]
        .last()
    )
    tidy = grouped.rename("value").reset_index()
    tidy["week"] = pd.to_datetime(tidy["week"])
    benchmark_rows = tidy.loc[tidy["security"] == benchmark_security]
    if benchmark_rows.empty:
        benchmark = pd.Series(index=all_weeks, dtype=float, name=benchmark_security)
    else:
        # SPY may be repeated in more than one source theme.  Median is
        # deterministic and makes duplicate copies harmless.
        benchmark = benchmark_rows.groupby("week", sort=True)["value"].median().reindex(all_weeks)
        benchmark.name = benchmark_security

    members_by_theme: dict[str, tuple[str, ...]] = {}
    for theme, group in tidy.loc[tidy["security"] != benchmark_security].groupby(
        "theme", sort=True
    ):
        names = tuple(sorted(str(item) for item in group["security"].dropna().unique()))
        if names:
            members_by_theme[str(theme)] = names
    themes = tuple(sorted(members_by_theme))
    theme_rows = tidy.loc[tidy["security"] != benchmark_security]
    if theme_rows.empty:
        prices = pd.DataFrame(index=all_weeks)
    else:
        prices = (
            theme_rows.pivot_table(
                index="week",
                columns=["theme", "security"],
                values="value",
                aggfunc="last",
                sort=True,
            )
            .reindex(all_weeks)
        )
        if not isinstance(prices.columns, pd.MultiIndex):
            prices.columns = pd.MultiIndex.from_tuples([], names=["theme", "security"])
        prices.columns = pd.MultiIndex.from_tuples(
            [(str(theme), str(security)) for theme, security in prices.columns],
            names=["theme", "security"],
        )
        prices = prices.sort_index(axis=1)
    all_sessions = pd.DatetimeIndex(frame["date"].drop_duplicates().sort_values())
    return prices, benchmark, themes, members_by_theme, all_sessions


def _rolling_compounded_returns(
    returns: pd.DataFrame | pd.Series, window: int
) -> pd.DataFrame | pd.Series:
    """Compound exactly ``window`` observed weekly returns, without filling."""

    def compound(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        return float(np.prod(1.0 + values) - 1.0)

    return returns.rolling(window=window, min_periods=window).apply(compound, raw=True)


def _security_return_panels(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()
    return prices.pct_change(fill_method=None)


def _theme_return_panels(
    security_returns: pd.DataFrame,
    security_prices: pd.DataFrame,
    themes: Sequence[str],
    members: Mapping[str, Sequence[str]],
    min_participation: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build equal-weight weekly returns and participation fractions."""

    out = pd.DataFrame(index=security_returns.index, columns=list(themes), dtype=float)
    participation = out.copy()
    for theme in themes:
        columns = [column for column in security_returns.columns if column[0] == theme]
        if not columns:
            continue
        valid = security_returns.loc[:, columns].notna()
        count = valid.sum(axis=1)
        # Infer the point-in-time universe from first observed prices.  A
        # security added later cannot reduce an earlier week's participation or
        # change an earlier theme return (a subtle source of lookahead).
        known = security_prices.loc[:, columns].notna().expanding(min_periods=1).max()
        known_count = known.sum(axis=1).replace(0, np.nan)
        participation[theme] = count / known_count
        min_count = np.maximum(1.0, np.ceil(known_count * min_participation))
        out[theme] = security_returns.loc[:, columns].mean(axis=1, skipna=True).where(
            count >= min_count
        )
    return out, participation


def _prepare(data: pd.DataFrame, config: SignalConfig) -> _Prepared:
    benchmark_security = str(config.benchmark_security).strip().upper()
    frame, sessions = _normalise_input(data, benchmark_security)
    prices, benchmark, themes, members, all_sessions = _weekly_price_panels(
        frame, benchmark_security, config.min_value
    )
    weeks = prices.index if not prices.empty else benchmark.index
    if len(weeks) == 0:
        weeks = pd.DatetimeIndex([], name="decision_date")
    security_returns = _security_return_panels(prices)
    theme_returns, _participation = _theme_return_panels(
        security_returns,
        prices,
        themes,
        members,
        config.min_participation,
    )
    benchmark_returns = benchmark.pct_change(fill_method=None)
    theme_index = (1.0 + theme_returns).cumprod(skipna=True) * 100.0
    theme_index = theme_index.where(theme_returns.notna().cumsum() > 0)
    # all_sessions includes invalid-value rows, which is useful for the
    # next-session label while no invalid value participates in metrics.
    if len(sessions) == 0:
        all_sessions = sessions
    return _Prepared(
        weeks=pd.DatetimeIndex(weeks, name="decision_date"),
        themes=themes,
        members=members,
        benchmark=benchmark,
        benchmark_returns=benchmark_returns,
        security_prices=prices,
        security_returns=security_returns,
        theme_returns=theme_returns,
        theme_index=theme_index,
        sessions=all_sessions,
    )


def _percentile_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Stable 0..1 cross-sectional percentile ranks for each decision date.

    Columns are sorted before ranking and ties use their average rank.  The
    percentile convention is the standard pandas ``rank(pct=True)`` rule
    (rank divided by the number of valid observations); this keeps M0
    compatible with ordinary cross-sectional research implementations while
    still making ties and missing values explicit.
    """

    if frame.empty:
        return frame.copy()
    ordered = frame.reindex(sorted(frame.columns), axis=1)
    result = pd.DataFrame(index=ordered.index, columns=ordered.columns, dtype=float)
    for date, row in ordered.iterrows():
        valid = row.dropna()
        if valid.empty:
            continue
        ranks = valid.rank(method="average", ascending=True, pct=True)
        result.loc[date, valid.index] = ranks
    return result.reindex(columns=frame.columns)


# Public aliases make the tie/missingness rule reusable by validation code and
# keep the ranking primitive independently testable.
percentile_rank = _percentile_rank
cross_sectional_percentile_rank = _percentile_rank


def _nanmax_frames(
    frames: Sequence[pd.DataFrame],
    index: pd.DatetimeIndex,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Elementwise maximum preserving all-NaN cells as NaN."""

    aligned = [
        frame.reindex(index=index, columns=columns).to_numpy(dtype=float)
        for frame in frames
    ]
    if not aligned:
        return pd.DataFrame(index=index, columns=columns, dtype=float)
    values = np.stack(aligned, axis=0)
    all_missing = np.all(np.isnan(values), axis=0)
    # Replace missing cells before max so an all-missing slice does not emit
    # NumPy's RuntimeWarning; restore those cells to NaN below.
    result = np.max(np.where(np.isnan(values), -np.inf, values), axis=0)
    result[all_missing] = np.nan
    return pd.DataFrame(result, index=index, columns=columns, dtype=float)


def _next_session_labels(
    decision_dates: Iterable[pd.Timestamp], sessions: pd.DatetimeIndex
) -> pd.Series:
    dates = pd.DatetimeIndex(decision_dates)
    if len(dates) == 0:
        return pd.Series(index=dates, dtype="datetime64[ns]")
    sessions = pd.DatetimeIndex(sessions).sort_values().drop_duplicates()
    if len(sessions) == 0:
        return pd.Series(pd.NaT, index=dates, dtype="datetime64[ns]")
    positions = sessions.searchsorted(dates, side="right")
    # With a weekly-only extract there may be no observed row after the latest
    # Friday.  A following business day is the conservative label fallback;
    # when a real session exists (including holiday-aware gaps), it wins.
    values = [
        sessions[pos] if pos < len(sessions) else date + pd.offsets.BDay(1)
        for date, pos in zip(dates, positions, strict=True)
    ]
    return pd.Series(values, index=dates, dtype="datetime64[ns]")


def _base_metrics(prepared: _Prepared, config: SignalConfig) -> dict[str, object]:
    """Compute shared transparent metrics; no output screen logic here."""

    dates = prepared.weeks
    themes = prepared.themes
    theme_returns = prepared.theme_returns.reindex(index=dates, columns=themes)
    participation = pd.DataFrame(index=dates, columns=themes, dtype=float)
    for theme in themes:
        cols = [column for column in prepared.security_returns.columns if column[0] == theme]
        known = prepared.security_prices.loc[:, cols].notna().expanding(min_periods=1).max()
        known_count = known.sum(axis=1).replace(0, np.nan)
        participation[theme] = prepared.security_returns.loc[:, cols].notna().sum(axis=1).div(
            known_count
        )

    benchmark_returns = prepared.benchmark_returns.reindex(dates)
    benchmark_window: dict[int, pd.Series] = {}
    theme_window: dict[int, pd.DataFrame] = {}
    security_window: dict[int, pd.DataFrame] = {}
    breadth: dict[int, pd.DataFrame] = {}
    for window in sorted({4, 8, 13, 26, 52}):
        benchmark_window[window] = _rolling_compounded_returns(benchmark_returns, window)
        theme_window[window] = _rolling_compounded_returns(theme_returns, window)
        security_window[window] = _rolling_compounded_returns(prepared.security_returns, window)
        valid_benchmark = benchmark_window[window].notna()
        breadth_frame = pd.DataFrame(index=dates, columns=themes, dtype=float)
        for theme in themes:
            columns = [column for column in security_window[window].columns if column[0] == theme]
            if not columns:
                continue
            sec = security_window[window].loc[:, columns]
            valid = sec.notna() & valid_benchmark.to_numpy()[:, None]
            denominator = valid.sum(axis=1)
            winner = sec.gt(benchmark_window[window], axis=0) & valid
            breadth_frame[theme] = (
                winner.sum(axis=1) / denominator.replace(0, np.nan)
            ).where(denominator > 0)
        breadth[window] = breadth_frame

    rs: dict[int, pd.DataFrame] = {}
    relative_ratio: dict[int, pd.DataFrame] = {}
    for window in sorted({4, 8, 13, 26, 52}):
        rs[window] = theme_window[window].sub(benchmark_window[window], axis=0)
        relative_ratio[window] = (1.0 + theme_window[window]).div(
            1.0 + benchmark_window[window], axis=0
        ) - 1.0

    # M0 is kept as its own named baseline even though it equals rs[26].
    m0 = rs[config.m0_window].copy()
    m0_rank = _percentile_rank(m0)
    m0_available = m0.notna()
    benchmark_available = benchmark_window[config.m0_window].notna()

    # Security-level total returns are used only for concentration.  Positive
    # contributions make a single winner visible while not treating a single
    # large loser as a rally concentration.
    concentration: dict[int, pd.DataFrame] = {}
    top_share: dict[int, pd.DataFrame] = {}
    for window in (4, 8, 13, 26, 52):
        concentration[window] = pd.DataFrame(index=dates, columns=themes, dtype=float)
        top_share[window] = pd.DataFrame(index=dates, columns=themes, dtype=float)
        sec_panel = security_window[window]
        for theme in themes:
            columns = [column for column in sec_panel.columns if column[0] == theme]
            if not columns:
                continue
            for date in dates:
                values = sec_panel.loc[date, columns].dropna()
                if values.empty:
                    continue
                positive = values.clip(lower=0.0)
                total = float(positive.sum())
                if total <= 0.0:
                    # No positive return is not a concentrated rally.  A
                    # neutral value makes the component transparent.
                    concentration[window].loc[date, theme] = 0.0
                    top_share[window].loc[date, theme] = 0.0
                else:
                    weights = positive / total
                    concentration[window].loc[date, theme] = float((weights**2).sum())
                    top_share[window].loc[date, theme] = float(weights.max())

    history = pd.DataFrame(index=dates, columns=themes, dtype=float)
    for theme in themes:
        history[theme] = theme_returns[theme].notna().cumsum().astype(float)

    next_sessions = _next_session_labels(dates, prepared.sessions)
    return {
        "dates": dates,
        "themes": themes,
        "theme_returns": theme_returns,
        "participation": participation,
        "benchmark_window": benchmark_window,
        "theme_window": theme_window,
        "security_window": security_window,
        "breadth": breadth,
        "rs": rs,
        "relative_ratio": relative_ratio,
        "m0": m0,
        "m0_rank": m0_rank,
        "m0_available": m0_available,
        "benchmark_available": benchmark_available,
        "concentration": concentration,
        "top_share": top_share,
        "history": history,
        "next_sessions": next_sessions,
        "theme_index": prepared.theme_index.reindex(index=dates, columns=themes),
    }


def _frame_from_columns(
    metrics: Mapping[str, object],
    columns: Mapping[str, pd.DataFrame | pd.Series | object],
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(metrics["dates"])
    themes: tuple[str, ...] = tuple(metrics["themes"])
    if not themes or len(dates) == 0:
        return _empty_output()
    index = pd.MultiIndex.from_product([dates, themes], names=["decision_date", "theme"])
    output = pd.DataFrame(index=index)
    for name, value in columns.items():
        if isinstance(value, pd.DataFrame):
            # MultiIndex product order is date-major/theme-minor, matching
            # DataFrame.to_numpy().reshape(-1) and avoiding stack's changing
            # dropna/future_stack semantics across pandas releases.
            matrix = value.reindex(index=dates, columns=themes)
            output[name] = matrix.to_numpy().reshape(-1)
        elif isinstance(value, pd.Series):
            output[name] = value.reindex(dates).repeat(len(themes)).to_numpy()
        else:
            output[name] = value
    output = output.reset_index()
    next_sessions = metrics["next_sessions"]
    output["effective_date"] = output["decision_date"].map(next_sessions)
    # Both names are useful to consumers: decision_date is the canonical
    # field; signal_date makes next-session semantics obvious in APIs.
    output["signal_date"] = output["decision_date"]
    output["decision_weekday"] = output["decision_date"].dt.day_name()
    output["effective_is_next_session"] = (
        output["effective_date"].notna() & (output["effective_date"] > output["decision_date"])
    )
    output = output.sort_values(["decision_date", "theme"], kind="mergesort").reset_index(drop=True)
    return output


def _add_common_aliases(output: pd.DataFrame) -> pd.DataFrame:
    if output.empty:
        return output
    # M0 aliases are intentionally retained to make the baseline hard to
    # miss in consumers with either snake_case or display-oriented naming.
    output["M0_26W_SPY_RELATIVE"] = output["m0_26w_spy_relative"]
    output["m0_26w_spy_relative_rank"] = output["m0_rank"]
    # Preserve a lane's configured M0 gate (which may be stricter than zero).
    # The fallback is for empty/legacy frames that only expose the raw alias.
    if "m0_baseline_pass" not in output:
        output["m0_baseline_pass"] = output["m0_26w_spy_relative"].ge(0.0)
    output["m0_baseline_available"] = output["m0_26w_spy_relative"].notna()
    return output


def _warning_text(flags: Sequence[str]) -> str:
    return "; ".join(dict.fromkeys(flag for flag in flags if flag))


def _append_status_warnings(
    output: pd.DataFrame,
    status: pd.Series,
    warning_lists: Sequence[Sequence[str]],
    *,
    recommendation: str | None = None,
) -> pd.DataFrame:
    output["status"] = status.to_numpy()
    output["warnings"] = [_warning_text(flags) for flags in warning_lists]
    output["warning"] = output["warnings"]
    if recommendation is not None:
        output["recommendation"] = recommendation
        output["not_a_buy_recommendation"] = True
    return output


def _empty_or_prepared(
    data: pd.DataFrame, config: SignalConfig
) -> tuple[_Prepared, dict[str, object]]:
    prepared = _prepare(data, config)
    if not prepared.themes or len(prepared.weeks) == 0:
        return prepared, {}
    return prepared, _base_metrics(prepared, config)


def current_leader(
    data: pd.DataFrame,
    config: CurrentLeaderConfig | None = None,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return current-leadership components for every Friday/theme.

    A theme is selected only when all 13/26/52-week RS and breadth gates pass,
    the explicit M0 26-week SPY-relative baseline is positive, and its
    transparent score ranks within ``leader_count`` eligible themes.  The
    complete panel is returned so a consumer can inspect rejected themes too.
    """

    cfg = config or CurrentLeaderConfig()
    if not isinstance(cfg, CurrentLeaderConfig):
        raise TypeError("config must be CurrentLeaderConfig")
    prepared, metrics = _empty_or_prepared(data, cfg)
    if not metrics:
        return _empty_output()
    dates: pd.DatetimeIndex = metrics["dates"]
    themes: tuple[str, ...] = metrics["themes"]
    if as_of is not None:
        as_of_date = _coerce_dates(pd.Series([as_of])).iloc[0]
        if pd.notna(as_of_date):
            keep = dates <= pd.Timestamp(as_of_date).normalize()
            metrics = {**metrics, "dates": dates[keep]}
            # Reindex every panel in a single helper-free pass below.
            for key in (
                "theme_returns",
                "participation",
                "m0",
                "m0_rank",
                "history",
                "theme_index",
            ):
                value = metrics[key]
                metrics[key] = value.loc[keep]
            for key in ("breadth", "rs", "relative_ratio", "concentration", "top_share"):
                metrics[key] = {window: value.loc[keep] for window, value in metrics[key].items()}
            metrics["next_sessions"] = metrics["next_sessions"].loc[keep]
            dates = metrics["dates"]
    rs: Mapping[int, pd.DataFrame] = metrics["rs"]
    breadth: Mapping[int, pd.DataFrame] = metrics["breadth"]
    m0: pd.DataFrame = metrics["m0"]
    m0_rank: pd.DataFrame = metrics["m0_rank"]
    participation: pd.DataFrame = metrics["participation"]
    history: pd.DataFrame = metrics["history"]
    rs_rank = {window: _percentile_rank(rs[window]) for window in (13, 26, 52)}
    breadth_avg = (breadth[13] + breadth[26] + breadth[52]) / 3.0
    score = (
        cfg.score_weights[0] * rs_rank[13]
        + cfg.score_weights[1] * rs_rank[26]
        + cfg.score_weights[2] * rs_rank[52]
        + cfg.score_weights[3] * breadth_avg
    )
    # History uses valid weekly returns; 52 points are required for a 52w RS
    # observation, and this explicit count catches missing-data gaps.
    history_ok = history >= float(cfg.min_history_weeks)
    finite = rs[13].notna() & rs[26].notna() & rs[52].notna()
    baseline_pass = m0.ge(cfg.min_relative_strength)
    breadth_pass = (
        breadth[13].ge(cfg.min_breadth)
        & breadth[26].ge(cfg.min_breadth)
        & breadth[52].ge(cfg.min_breadth)
    )
    participation_pass = participation.ge(cfg.min_participation)
    gate = finite & history_ok & baseline_pass & breadth_pass & participation_pass
    eligible_score = score.where(gate)
    leader_rank = eligible_score.rank(axis=1, method="min", ascending=False, na_option="bottom")
    is_leader = gate & eligible_score.ge(cfg.min_score) & leader_rank.le(cfg.leader_count)
    status = pd.DataFrame("not_leader", index=dates, columns=themes, dtype=object)
    status = status.mask(~history_ok, "insufficient_history")
    status = status.mask(history_ok & ~finite, "insufficient_history")
    status = status.mask(history_ok & finite & ~participation_pass, "warning")
    status = status.mask(history_ok & finite & participation_pass & ~breadth_pass, "warning")
    status = status.mask(gate & ~is_leader, "eligible_not_top")
    status = status.mask(is_leader, "current_leader")

    columns: dict[str, pd.DataFrame | pd.Series | object] = {
        "m0_26w_spy_relative": m0,
        "m0_rank": m0_rank,
        "m0_available": metrics["m0_available"],
        "benchmark_26w_available": metrics["benchmark_available"],
        "rs_13w": rs[13],
        "rs_26w": rs[26],
        "rs_52w": rs[52],
        "rs_rank_13w": rs_rank[13],
        "rs_rank_26w": rs_rank[26],
        "rs_rank_52w": rs_rank[52],
        "breadth_13w": breadth[13],
        "breadth_26w": breadth[26],
        "breadth_52w": breadth[52],
        "breadth_average": breadth_avg,
        "participation": participation,
        "history_weeks": history,
        "leader_score": score,
        "leader_rank": leader_rank,
        "m0_baseline_pass": baseline_pass,
        "breadth_gate_pass": breadth_pass,
        "participation_gate_pass": participation_pass,
        "is_current_leader": is_leader,
        "status": status,
    }
    output = _frame_from_columns(metrics, columns)
    output = _add_common_aliases(output)
    # Rebuild status/warnings after the common M0 aliases.  Each warning names
    # the failed transparent gate and never hides a missing-data reason.
    status_flat = output["status"].astype(str)
    warning_lists: list[list[str]] = []
    for _, row in output.iterrows():
        flags: list[str] = []
        if row["status"] == "insufficient_history":
            flags.append("insufficient_history")
        if not bool(row["benchmark_26w_available"]):
            flags.append("missing_spy_26w_baseline")
        if not bool(row["m0_baseline_pass"]) and pd.notna(row["m0_26w_spy_relative"]):
            flags.append("negative_m0_26w_spy_relative")
        if not bool(row["breadth_gate_pass"]):
            flags.append("breadth_below_threshold")
        if not bool(row["participation_gate_pass"]):
            flags.append("low_participation")
        if row["status"] == "eligible_not_top":
            flags.append("ranked_below_leader_cut")
        warning_lists.append(flags)
    output = _append_status_warnings(output, status_flat, warning_lists)
    output["is_current_leader"] = output["status"].eq("current_leader")
    return output


def emerging_radar(
    data: pd.DataFrame,
    config: EmergingRadarConfig | None = None,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return a high-sensitivity emerging-theme watchlist.

    Radar is explicitly a watchlist, never a buy recommendation.  It surfaces
    fast acceleration and rank improvement but adds warnings for weak breadth,
    low participation, a negative M0 baseline, and concentrated rallies.
    """

    cfg = config or EmergingRadarConfig()
    if not isinstance(cfg, EmergingRadarConfig):
        raise TypeError("config must be EmergingRadarConfig")
    prepared, metrics = _empty_or_prepared(data, cfg)
    if not metrics:
        return _empty_output()
    dates: pd.DatetimeIndex = metrics["dates"]
    themes: tuple[str, ...] = metrics["themes"]
    if as_of is not None:
        as_of_date = _coerce_dates(pd.Series([as_of])).iloc[0]
        if pd.notna(as_of_date):
            keep = dates <= pd.Timestamp(as_of_date).normalize()
            metrics = {**metrics, "dates": dates[keep]}
            for key in (
                "theme_returns",
                "participation",
                "m0",
                "m0_rank",
                "history",
                "theme_index",
            ):
                metrics[key] = metrics[key].loc[keep]
            for key in ("breadth", "rs", "relative_ratio", "concentration", "top_share"):
                metrics[key] = {window: value.loc[keep] for window, value in metrics[key].items()}
            metrics["next_sessions"] = metrics["next_sessions"].loc[keep]
            dates = metrics["dates"]
    rs: Mapping[int, pd.DataFrame] = metrics["rs"]
    breadth: Mapping[int, pd.DataFrame] = metrics["breadth"]
    m0: pd.DataFrame = metrics["m0"]
    m0_rank: pd.DataFrame = metrics["m0_rank"]
    participation: pd.DataFrame = metrics["participation"]
    concentration: Mapping[int, pd.DataFrame] = metrics["concentration"]
    top_share: Mapping[int, pd.DataFrame] = metrics["top_share"]
    history: pd.DataFrame = metrics["history"]

    rs_rank = {window: _percentile_rank(rs[window]) for window in (4, 8, 13)}
    speed4 = rs[4] / 4.0
    speed8 = rs[8] / 8.0
    speed13 = rs[13] / 13.0
    accel_4v8 = speed4 - speed8
    accel_8v13 = speed8 - speed13
    acceleration = 0.60 * accel_4v8 + 0.40 * accel_8v13
    acceleration_rank = _percentile_rank(acceleration)
    rank_improvement_4v13 = rs_rank[4] - rs_rank[13]
    rank_improvement_4v8 = rs_rank[4] - rs_rank[8]
    breadth_average = 0.50 * breadth[4] + 0.30 * breadth[8] + 0.20 * breadth[13]
    concentration_max_frame = _nanmax_frames(
        [concentration[4], concentration[8], concentration[13]], dates, themes
    )

    score = (
        0.35 * acceleration_rank
        + 0.25 * rs_rank[4]
        + 0.20 * rank_improvement_4v13.clip(lower=0.0, upper=1.0)
        + 0.10 * breadth_average
        + 0.10 * participation
    )
    history_ok = history >= float(cfg.min_history_weeks)
    finite_fast = rs[4].notna() & acceleration.notna()
    participation_pass = participation.ge(cfg.min_participation_for_watch)
    breadth_pass = breadth[4].ge(cfg.min_fast_breadth)
    concentration_pass = concentration_max_frame.le(cfg.max_concentration)
    rank_pass = rank_improvement_4v13.ge(cfg.min_rank_improvement)
    acceleration_pass = acceleration_rank.ge(cfg.min_acceleration_rank)
    fast_rs_pass = rs[4].ge(cfg.min_fast_relative_strength)
    # Radar is deliberately high sensitivity: breadth and concentration are
    # quality diagnostics, not hard exclusions.  A one-name rally remains a
    # visible watch item with ``low_breadth``/``concentrated_rally`` warnings;
    # strict breadth/concentration gates belong to CurrentLeader/Hold.
    watch = (
        history_ok
        & finite_fast
        & participation_pass
        & rank_pass
        & acceleration_pass
        & fast_rs_pass
        & score.ge(cfg.score_threshold)
    )
    status = pd.DataFrame("not_watch", index=dates, columns=themes, dtype=object)
    status = status.mask(~history_ok | ~finite_fast, "insufficient_history")
    status = status.mask(
        history_ok
        & finite_fast
        & (~participation_pass | ~breadth_pass | ~concentration_pass),
        "warning",
    )
    status = status.mask(watch, "watch")

    columns: dict[str, pd.DataFrame | pd.Series | object] = {
        "m0_26w_spy_relative": m0,
        "m0_rank": m0_rank,
        "m0_available": metrics["m0_available"],
        "benchmark_26w_available": metrics["benchmark_available"],
        "rs_4w": rs[4],
        "rs_8w": rs[8],
        "rs_13w": rs[13],
        "rs_rank_4w": rs_rank[4],
        "rs_rank_8w": rs_rank[8],
        "rs_rank_13w": rs_rank[13],
        "accel_4v8": accel_4v8,
        "accel_8v13": accel_8v13,
        "acceleration": acceleration,
        "acceleration_rank": acceleration_rank,
        "rank_improvement_4v13": rank_improvement_4v13,
        "rank_improvement_4v8": rank_improvement_4v8,
        "breadth_4w": breadth[4],
        "breadth_8w": breadth[8],
        "breadth_13w": breadth[13],
        "breadth_average": breadth_average,
        "participation": participation,
        "concentration_4w": concentration[4],
        "concentration_8w": concentration[8],
        "concentration_13w": concentration[13],
        "concentration_max": concentration_max_frame,
        "top_security_share_4w": top_share[4],
        "history_weeks": history,
        "radar_score": score,
        "m0_baseline_pass": m0.ge(0.0),
        "acceleration_gate_pass": acceleration_pass,
        "rank_improvement_gate_pass": rank_pass,
        "breadth_gate_pass": breadth_pass,
        "participation_gate_pass": participation_pass,
        "concentration_gate_pass": concentration_pass,
        "is_watchlist": watch,
        "status": status,
    }
    output = _frame_from_columns(metrics, columns)
    output = _add_common_aliases(output)
    warning_lists: list[list[str]] = []
    for _, row in output.iterrows():
        flags: list[str] = ["watchlist_only_not_a_buy_recommendation"]
        if row["status"] == "insufficient_history":
            flags.append("insufficient_history")
        if not bool(row["benchmark_26w_available"]):
            flags.append("missing_spy_26w_baseline")
        if pd.notna(row["m0_26w_spy_relative"]) and not bool(row["m0_baseline_pass"]):
            flags.append("negative_m0_26w_spy_relative")
        if pd.notna(row["breadth_4w"]) and not bool(row["breadth_gate_pass"]):
            flags.append("low_breadth")
        if pd.notna(row["participation"]) and not bool(row["participation_gate_pass"]):
            flags.append("low_participation")
        if pd.notna(row["concentration_max"]) and not bool(row["concentration_gate_pass"]):
            flags.append("concentrated_rally")
        if pd.notna(row["rank_improvement_4v13"]) and not bool(row["rank_improvement_gate_pass"]):
            flags.append("rank_not_improving")
        warning_lists.append(flags)
    output = _append_status_warnings(
        output,
        output["status"].astype(str),
        warning_lists,
        recommendation=NOT_A_BUY_RECOMMENDATION,
    )
    output["is_watchlist"] = output["status"].eq("watch")
    return output


def hold_candidate_12m(
    data: pd.DataFrame,
    config: HoldCandidate12MConfig | None = None,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return 12-month hold candidates gated by confirmed RS/durability/risk.

    A candidate requires positive 13/26/52-week SPY-relative strength,
    26-week M0, breadth and participation, a two-of-three-week confirmation,
    and explicit volatility/drawdown/concentration risk gates.  Rows without
    52 valid weekly returns are retained with ``status=insufficient_history``.
    """

    cfg = config or HoldCandidate12MConfig()
    if not isinstance(cfg, HoldCandidate12MConfig):
        raise TypeError("config must be HoldCandidate12MConfig")
    prepared, metrics = _empty_or_prepared(data, cfg)
    if not metrics:
        return _empty_output()
    dates: pd.DatetimeIndex = metrics["dates"]
    themes: tuple[str, ...] = metrics["themes"]
    if as_of is not None:
        as_of_date = _coerce_dates(pd.Series([as_of])).iloc[0]
        if pd.notna(as_of_date):
            keep = dates <= pd.Timestamp(as_of_date).normalize()
            metrics = {**metrics, "dates": dates[keep]}
            for key in (
                "theme_returns",
                "participation",
                "m0",
                "m0_rank",
                "history",
                "theme_index",
            ):
                metrics[key] = metrics[key].loc[keep]
            for key in ("breadth", "rs", "relative_ratio", "concentration", "top_share"):
                metrics[key] = {window: value.loc[keep] for window, value in metrics[key].items()}
            metrics["next_sessions"] = metrics["next_sessions"].loc[keep]
            dates = metrics["dates"]
    rs: Mapping[int, pd.DataFrame] = metrics["rs"]
    breadth: Mapping[int, pd.DataFrame] = metrics["breadth"]
    m0: pd.DataFrame = metrics["m0"]
    m0_rank: pd.DataFrame = metrics["m0_rank"]
    participation: pd.DataFrame = metrics["participation"]
    concentration: Mapping[int, pd.DataFrame] = metrics["concentration"]
    theme_returns: pd.DataFrame = metrics["theme_returns"]
    theme_index: pd.DataFrame = metrics["theme_index"]
    history: pd.DataFrame = metrics["history"]

    rs_rank = {window: _percentile_rank(rs[window]) for window in (13, 26, 52)}
    breadth_average = (breadth[13] + breadth[26] + breadth[52]) / 3.0
    concentration_frame = _nanmax_frames(
        [concentration[13], concentration[26], concentration[52]], dates, themes
    )
    annual_volatility = theme_returns.rolling(26, min_periods=26).std(ddof=1) * sqrt(52.0)

    def rolling_drawdown(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        running_peak = np.maximum.accumulate(values)
        return float(np.min(values / running_peak - 1.0))

    max_drawdown = theme_index.rolling(52, min_periods=52).apply(rolling_drawdown, raw=True)
    m0_pass = m0.ge(cfg.min_relative_strength)
    rs_pass = (
        rs[13].ge(cfg.min_relative_strength)
        & rs[26].ge(cfg.min_relative_strength)
        & rs[52].ge(cfg.min_relative_strength)
    )
    durability_score = (
        0.30 * breadth[13]
        + 0.35 * breadth[26]
        + 0.20 * breadth[52]
        + 0.15 * participation
    )
    durability_pass = (
        breadth[13].ge(cfg.min_breadth)
        & breadth[26].ge(cfg.min_breadth)
        & breadth[52].ge(cfg.min_breadth)
        & participation.ge(cfg.min_participation)
        & durability_score.ge(cfg.min_durability_score)
    )
    concentration_pass = concentration_frame.le(cfg.max_concentration)
    volatility_pass = annual_volatility.le(cfg.max_annualized_volatility)
    drawdown_pass = max_drawdown.ge(-cfg.max_drawdown)
    risk_pass = concentration_pass & volatility_pass & drawdown_pass
    confirmed_raw = rs_pass & m0_pass & durability_pass
    confirmation_count = confirmed_raw.astype(float).rolling(
        cfg.confirmation_window, min_periods=cfg.confirmation_window
    ).sum()
    confirmed_pass = confirmation_count.ge(float(cfg.min_confirmed_weeks))
    history_ok = history.ge(float(cfg.min_history_weeks))
    finite = (
        rs[13].notna()
        & rs[26].notna()
        & rs[52].notna()
        & breadth_average.notna()
        & annual_volatility.notna()
        & max_drawdown.notna()
    )
    candidate_score = (
        0.35 * (rs_rank[13] + rs_rank[26] + rs_rank[52]) / 3.0
        + 0.35 * durability_score
        + 0.15 * (1.0 - annual_volatility.div(cfg.max_annualized_volatility).clip(upper=1.0))
        + 0.15 * (1.0 + max_drawdown.div(cfg.max_drawdown).clip(lower=-1.0, upper=0.0))
    )
    candidate = (
        history_ok
        & finite
        & rs_pass
        & m0_pass
        & durability_pass
        & risk_pass
        & confirmed_pass
        & candidate_score.ge(cfg.min_score)
    )
    status = pd.DataFrame("not_candidate", index=dates, columns=themes, dtype=object)
    status = status.mask(~history_ok | ~finite, "insufficient_history")
    status = status.mask(history_ok & finite & (~rs_pass | ~m0_pass), "warning")
    status = status.mask(history_ok & finite & rs_pass & m0_pass & ~durability_pass, "warning")
    status = status.mask(
        history_ok & finite & rs_pass & m0_pass & durability_pass & ~risk_pass,
        "risk_rejected",
    )
    status = status.mask(
        history_ok & finite & rs_pass & m0_pass & durability_pass & risk_pass & ~confirmed_pass,
        "awaiting_confirmation",
    )
    status = status.mask(candidate, "hold_candidate")

    columns: dict[str, pd.DataFrame | pd.Series | object] = {
        "m0_26w_spy_relative": m0,
        "m0_rank": m0_rank,
        "m0_available": metrics["m0_available"],
        "benchmark_26w_available": metrics["benchmark_available"],
        "rs_13w": rs[13],
        "rs_26w": rs[26],
        "rs_52w": rs[52],
        "rs_rank_13w": rs_rank[13],
        "rs_rank_26w": rs_rank[26],
        "rs_rank_52w": rs_rank[52],
        "breadth_13w": breadth[13],
        "breadth_26w": breadth[26],
        "breadth_52w": breadth[52],
        "breadth_average": breadth_average,
        "participation": participation,
        "concentration_13w": concentration[13],
        "concentration_26w": concentration[26],
        "concentration_52w": concentration[52],
        "concentration_max": concentration_frame,
        "annualized_volatility_26w": annual_volatility,
        "max_drawdown_52w": max_drawdown,
        "durability_score": durability_score,
        "confirmation_count": confirmation_count,
        "candidate_score": candidate_score,
        "history_weeks": history,
        "m0_baseline_pass": m0_pass,
        "rs_gate_pass": rs_pass,
        "durability_gate_pass": durability_pass,
        "risk_gate_pass": risk_pass,
        "confirmation_gate_pass": confirmed_pass,
        "is_hold_candidate": candidate,
        "status": status,
    }
    output = _frame_from_columns(metrics, columns)
    output = _add_common_aliases(output)
    warning_lists: list[list[str]] = []
    for _, row in output.iterrows():
        flags: list[str] = []
        if row["status"] == "insufficient_history":
            flags.append("insufficient_history")
        if not bool(row["benchmark_26w_available"]):
            flags.append("missing_spy_26w_baseline")
        if pd.notna(row["m0_26w_spy_relative"]) and not bool(row["m0_baseline_pass"]):
            flags.append("negative_m0_26w_spy_relative")
        if pd.notna(row["rs_13w"]) and not bool(row["rs_gate_pass"]):
            flags.append("relative_strength_gate_failed")
        if pd.notna(row["durability_score"]) and not bool(row["durability_gate_pass"]):
            flags.append("durability_gate_failed")
        if pd.notna(row["concentration_max"]) and not bool(row["risk_gate_pass"]):
            if row["concentration_max"] > cfg.max_concentration:
                flags.append("concentrated_rally")
            if (
                pd.notna(row["annualized_volatility_26w"])
                and row["annualized_volatility_26w"] > cfg.max_annualized_volatility
            ):
                flags.append("high_volatility")
            if pd.notna(row["max_drawdown_52w"]) and row["max_drawdown_52w"] < -cfg.max_drawdown:
                flags.append("deep_drawdown")
        if row["status"] == "awaiting_confirmation":
            flags.append("awaiting_confirmation")
        warning_lists.append(flags)
    output = _append_status_warnings(output, output["status"].astype(str), warning_lists)
    output["is_hold_candidate"] = output["status"].eq("hold_candidate")
    return output


# PascalCase aliases match the product's three named outputs while retaining
# snake_case functions for Python callers.
CurrentLeader = current_leader
EmergingRadar = emerging_radar
HoldCandidate12M = hold_candidate_12m


__all__ = [
    "SignalConfig",
    "CurrentLeaderConfig",
    "EmergingRadarConfig",
    "HoldCandidate12MConfig",
    "current_leader",
    "emerging_radar",
    "hold_candidate_12m",
    "percentile_rank",
    "cross_sectional_percentile_rank",
    "CurrentLeader",
    "EmergingRadar",
    "HoldCandidate12M",
]
