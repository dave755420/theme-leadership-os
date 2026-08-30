"""연도별 주도 테마와 초입 상승 신호.

이 모듈은 ``signals.core``의 동일한 주간·PIT(시점 기준) 가격 준비 로직을
재사용한다. 따라서 연도별 순위표와 초입 레이더가 서로 다른 데이터 처리
규칙을 사용하지 않는다. 반환값은 연구·관찰용이며 매수 주문을 생성하지 않는다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..signals.core import (
    SignalConfig,
    _percentile_rank,
    _prepare,
)

ANNUAL_DISCLAIMER_KO = "무료 공개 데이터 기반 연구·관찰용 결과이며 투자 권고나 매수 신호가 아닙니다."
EARLY_RECOMMENDATION_KO = "관찰 전용 · 매수 추천 아님"


@dataclass(frozen=True)
class AnnualLeadershipConfig:
    """연도 순위표와 초입 레이더의 고정 규칙.

    ``min_weeks_per_year``와 ``min_coverage``는 결측이 많은 테마가 연간
    1~2개 관측치만으로 1위를 차지하지 못하게 하는 데이터 품질 게이트다.
    초입 규칙은 순위·가속·참여율을 함께 요구하지만, 미래 52주 수익률을
    계산해 현재 신호에 섞지 않는다.
    """

    benchmark_security: str = "SPY"
    min_participation: float = 0.50
    min_weeks_per_year: int = 26
    min_coverage: float = 0.70
    top_n: int = 3
    signal_history_weeks: int = 26
    early_min_rank: float = 0.70
    early_min_acceleration_rank: float = 0.70
    early_min_rank_improvement: float = 0.10
    early_min_breadth: float = 0.50
    early_min_participation: float = 0.50
    early_min_fast_rs: float = 0.0
    # 초입에서 26주 상대강도가 아직 약할 수 있으므로 완전한 양수만
    # 요구하지 않는다. 다만 -30%보다 깊은 하락은 관찰 신호에서 제외한다.
    early_min_m0: float = -0.30
    confirmation_window_weeks: int = 3
    confirmation_count: int = 2

    def __post_init__(self) -> None:
        if not self.benchmark_security or not str(self.benchmark_security).strip():
            raise ValueError("benchmark_security must be non-empty")
        if not 0.0 < self.min_participation <= 1.0:
            raise ValueError("min_participation must be in (0, 1]")
        if self.min_weeks_per_year <= 0 or self.top_n <= 0:
            raise ValueError("min_weeks_per_year and top_n must be positive")
        if not 0.0 < self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be in (0, 1]")
        for name, value in (
            ("early_min_rank", self.early_min_rank),
            ("early_min_acceleration_rank", self.early_min_acceleration_rank),
            ("early_min_breadth", self.early_min_breadth),
            ("early_min_participation", self.early_min_participation),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.signal_history_weeks <= 0:
            raise ValueError("signal_history_weeks must be positive")
        if self.confirmation_window_weeks <= 0 or self.confirmation_count <= 0:
            raise ValueError("confirmation settings must be positive")
        if self.confirmation_count > self.confirmation_window_weeks:
            raise ValueError("confirmation_count cannot exceed confirmation_window_weeks")


def _empty_annual() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "year",
            "period_label",
            "theme",
            "theme_return",
            "benchmark_return",
            "excess_return",
            "rank",
            "coverage",
            "benchmark_coverage",
            "observations",
            "start_date",
            "end_date",
            "is_partial_year",
            "status",
            "disclaimer",
        ]
    )


def _empty_early() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "decision_date",
            "theme",
            "early_watch",
            "early_confirmed",
            "signal_type",
            "rs_4w",
            "rs_13w",
            "m0_26w_spy_relative",
            "rs_rank_4w",
            "acceleration_rank",
            "rank_improvement_4v13",
            "breadth_4w",
            "participation",
            "history_weeks",
            "confirmation_count",
            "status",
            "recommendation",
            "disclaimer",
        ]
    )


def _as_of_frame(data: pd.DataFrame, as_of: str | pd.Timestamp | None) -> pd.DataFrame:
    """Filter raw rows before any rolling calculation."""

    if as_of is None:
        return data
    if "date" not in data.columns:
        raise ValueError("data must contain a date column")
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert(None)
    dates = pd.to_datetime(data["date"], errors="coerce")
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_convert(None)
    return data.loc[dates <= cutoff.normalize()].copy()


def _prepared_metrics(
    data: pd.DataFrame,
    config: AnnualLeadershipConfig,
    as_of: str | pd.Timestamp | None,
) -> tuple[Any, dict[str, object]]:
    frame = _as_of_frame(data, as_of)
    if frame.empty:
        return None, {}
    signal_config = SignalConfig(
        benchmark_security=str(config.benchmark_security).strip().upper(),
        min_participation=config.min_participation,
        m0_window=26,
        min_history_weeks=config.signal_history_weeks,
    )
    prepared = _prepare(frame, signal_config)
    if len(prepared.weeks) == 0 or not prepared.themes:
        return prepared, {}
    # ``_base_metrics`` also builds concentration and 52-week durability
    # panels for the three full signal lanes.  The annual/early view needs
    # only a small subset; calculating it here avoids an unnecessary nested
    # per-date/per-theme loop when a broad universe is supplied.
    dates = pd.DatetimeIndex(prepared.weeks)
    theme_returns = prepared.theme_returns.reindex(index=dates, columns=prepared.themes)
    benchmark_returns = prepared.benchmark_returns.reindex(dates)

    def compound(values: pd.DataFrame | pd.Series, window: int) -> pd.DataFrame | pd.Series:
        return (1.0 + values).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0

    benchmark_window = {window: compound(benchmark_returns, window) for window in (4, 13, 26)}
    theme_window = {window: compound(theme_returns, window) for window in (4, 13, 26)}
    rs = {
        window: theme_window[window].sub(benchmark_window[window], axis=0)
        for window in (4, 13, 26)
    }

    # Participation is point-in-time: a member first becomes part of the
    # denominator only after its first observed price.
    participation = pd.DataFrame(index=dates, columns=prepared.themes, dtype=float)
    for theme in prepared.themes:
        columns = [column for column in prepared.security_returns.columns if column[0] == theme]
        if not columns:
            continue
        known = prepared.security_prices.loc[:, columns].notna().expanding(min_periods=1).max()
        known_count = known.sum(axis=1).replace(0, np.nan)
        participation[theme] = prepared.security_returns.loc[:, columns].notna().sum(axis=1).div(
            known_count
        )

    breadth: dict[int, pd.DataFrame] = {}
    for window in (4,):
        security_window = compound(prepared.security_returns, window)
        valid_benchmark = benchmark_window[window].notna()
        breadth_frame = pd.DataFrame(index=dates, columns=prepared.themes, dtype=float)
        for theme in prepared.themes:
            columns = [column for column in security_window.columns if column[0] == theme]
            if not columns:
                continue
            sec = security_window.loc[:, columns]
            valid = sec.notna() & valid_benchmark.to_numpy()[:, None]
            denominator = valid.sum(axis=1)
            winner = sec.gt(benchmark_window[window], axis=0) & valid
            breadth_frame[theme] = winner.sum(axis=1).div(denominator.replace(0, np.nan)).where(
                denominator > 0
            )
        breadth[window] = breadth_frame

    history = theme_returns.notna().cumsum().astype(float)
    return prepared, {
        "dates": dates,
        "themes": prepared.themes,
        "theme_returns": theme_returns,
        "theme_index": prepared.theme_index.reindex(index=dates, columns=prepared.themes),
        "benchmark_window": benchmark_window,
        "theme_window": theme_window,
        "rs": rs,
        "m0": rs[26],
        "m0_rank": _percentile_rank(rs[26]),
        "breadth": breadth,
        "participation": participation,
        "history": history,
    }


def _last_valid(series: pd.Series, mask: pd.Series) -> tuple[pd.Timestamp | None, float | None]:
    values = pd.to_numeric(series.loc[mask], errors="coerce").dropna()
    if values.empty:
        return None, None
    return pd.Timestamp(values.index[-1]), float(values.iloc[-1])


def _start_value(
    series: pd.Series,
    dates: pd.DatetimeIndex,
    year: int,
) -> tuple[pd.Timestamp | None, float | None, bool]:
    year_start = pd.Timestamp(year=year, month=1, day=1)
    prior = pd.to_numeric(series.loc[dates < year_start], errors="coerce").dropna()
    if not prior.empty:
        return pd.Timestamp(prior.index[-1]), float(prior.iloc[-1]), False
    current = pd.to_numeric(series.loc[dates.year == year], errors="coerce").dropna()
    if current.empty:
        return None, None, True
    return pd.Timestamp(current.index[0]), float(current.iloc[0]), True


def _year_label(year: int, last_date: pd.Timestamp) -> tuple[str, bool]:
    # A last observation before 15 December is shown as YTD, not as a complete
    # calendar-year result. This keeps the current year visibly provisional.
    partial = last_date < pd.Timestamp(year=year, month=12, day=15)
    return (f"{year}년 누적" if partial else f"{year}년"), partial


def annual_theme_leaders(
    data: pd.DataFrame,
    config: AnnualLeadershipConfig | None = None,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """연도별 상위 테마와 시장 초과수익률을 반환한다.

    입력은 ``date,theme,security,value`` 형식이다. 테마 수익률은 알려진
    구성종목의 동일가중 주간 수익률이며, 전년도 마지막 관측치부터 올해
    마지막 관측치까지 계산한다. 첫 관측연도나 결측률이 높은 행은 상태를
    표시하고 순위에서 제외한다.
    """

    cfg = config or AnnualLeadershipConfig()
    if not isinstance(cfg, AnnualLeadershipConfig):
        raise TypeError("config must be AnnualLeadershipConfig")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    prepared, metrics = _prepared_metrics(data, cfg, as_of)
    if not metrics:
        return _empty_annual()

    dates = pd.DatetimeIndex(metrics["dates"])
    theme_returns: pd.DataFrame = metrics["theme_returns"]
    theme_index: pd.DataFrame = metrics["theme_index"]
    benchmark: pd.Series = prepared.benchmark.reindex(dates)
    years = sorted({int(value) for value in dates.year})
    rows: list[dict[str, object]] = []
    for year in years:
        year_mask = pd.Series(dates.year == year, index=dates)
        year_dates = dates[year_mask.to_numpy()]
        if len(year_dates) == 0:
            continue
        year_label, is_ytd = _year_label(year, pd.Timestamp(year_dates[-1]))
        benchmark_start_date, benchmark_start, partial_benchmark = _start_value(
            benchmark, dates, year
        )
        benchmark_end_date, benchmark_end = _last_valid(
            benchmark, year_mask
        )
        benchmark_return = (
            benchmark_end / benchmark_start - 1.0
            if benchmark_start is not None and benchmark_end is not None and benchmark_start > 0
            else np.nan
        )
        benchmark_obs = int(pd.to_numeric(benchmark.loc[year_mask], errors="coerce").notna().sum())
        expected_weeks = max(len(year_dates), 1)
        benchmark_coverage = benchmark_obs / expected_weeks

        for theme in metrics["themes"]:
            index_series = theme_index[theme].reindex(dates)
            start_date, start_value, partial_theme = _start_value(index_series, dates, year)
            end_date, end_value = _last_valid(index_series, year_mask)
            values = pd.to_numeric(theme_returns[theme].loc[year_mask], errors="coerce")
            observations = int(values.notna().sum())
            coverage = observations / expected_weeks
            theme_return = (
                end_value / start_value - 1.0
                if start_value is not None
                and end_value is not None
                and start_value > 0
                else np.nan
            )
            excess = (
                theme_return - benchmark_return
                if np.isfinite(theme_return) and np.isfinite(benchmark_return)
                else np.nan
            )
            status = "ok"
            if observations < cfg.min_weeks_per_year or coverage < cfg.min_coverage:
                status = "insufficient_coverage"
            if start_date is None or end_date is None or not np.isfinite(theme_return):
                status = "insufficient_history"
            if not np.isfinite(benchmark_return) or benchmark_coverage < cfg.min_coverage:
                status = "benchmark_unavailable"
            rows.append(
                {
                    "year": year,
                    "period_label": year_label,
                    "theme": str(theme),
                    "theme_return": float(theme_return) if np.isfinite(theme_return) else np.nan,
                    "benchmark_return": float(benchmark_return)
                    if np.isfinite(benchmark_return)
                    else np.nan,
                    "excess_return": float(excess) if np.isfinite(excess) else np.nan,
                    "rank": np.nan,
                    "coverage": round(float(coverage), 4),
                    "benchmark_coverage": round(float(benchmark_coverage), 4),
                    "observations": observations,
                    "start_date": start_date.date().isoformat() if start_date is not None else None,
                    "end_date": end_date.date().isoformat() if end_date is not None else None,
                    "is_partial_year": bool(is_ytd or partial_theme or partial_benchmark),
                    "status": status,
                    "disclaimer": ANNUAL_DISCLAIMER_KO,
                }
            )

    if not rows:
        return _empty_annual()
    output = pd.DataFrame(rows)
    eligible = output["status"].eq("ok") & output["theme_return"].notna()
    output.loc[eligible, "rank"] = (
        output.loc[eligible]
        .groupby("year")["theme_return"]
        .rank(method="min", ascending=False)
        .astype(float)
    )
    output = output.loc[eligible & output["rank"].le(float(cfg.top_n))].copy()
    output["rank"] = output["rank"].astype(int)
    return output.sort_values(["year", "rank", "theme"], kind="mergesort").reset_index(drop=True)


def early_rise_signals(
    data: pd.DataFrame,
    config: AnnualLeadershipConfig | None = None,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """초기 상승 국면을 포착하는 주간 관찰·확인 신호를 반환한다.

    신호 시점에는 미래 수익률이나 연간 1위 결과를 사용하지 않는다. 4주
    상대강도가 13주보다 빠르게 개선되고, 단면 순위·breadth·참여율·26주
    기준선이 동시에 통과한 뒤 3주 중 2주가 유지될 때 ``초입 확인``으로
    표시한다. 모든 행은 관찰용이며 자동 주문을 만들지 않는다.
    """

    cfg = config or AnnualLeadershipConfig()
    if not isinstance(cfg, AnnualLeadershipConfig):
        raise TypeError("config must be AnnualLeadershipConfig")
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    _prepared, metrics = _prepared_metrics(data, cfg, as_of)
    if not metrics:
        return _empty_early()

    dates = pd.DatetimeIndex(metrics["dates"])
    themes = tuple(metrics["themes"])
    rs: Mapping[int, pd.DataFrame] = metrics["rs"]
    breadth: Mapping[int, pd.DataFrame] = metrics["breadth"]
    participation: pd.DataFrame = metrics["participation"]
    history: pd.DataFrame = metrics["history"]
    m0: pd.DataFrame = metrics["m0"]

    rs_rank_4 = _percentile_rank(rs[4])
    rs_rank_13 = _percentile_rank(rs[13])
    acceleration = rs[4].div(4.0) - rs[13].div(13.0)
    acceleration_rank = _percentile_rank(acceleration)
    rank_improvement = rs_rank_4 - rs_rank_13
    fast_rs = rs[4]
    watch = (
        history.ge(float(cfg.signal_history_weeks))
        & fast_rs.ge(float(cfg.early_min_fast_rs))
        & m0.ge(float(cfg.early_min_m0))
        & rs_rank_4.ge(float(cfg.early_min_rank))
        & acceleration_rank.ge(float(cfg.early_min_acceleration_rank))
        & rank_improvement.ge(float(cfg.early_min_rank_improvement))
        & breadth[4].ge(float(cfg.early_min_breadth))
        & participation.ge(float(cfg.early_min_participation))
    )
    persistent_count = watch.astype(float).rolling(
        cfg.confirmation_window_weeks, min_periods=cfg.confirmation_window_weeks
    ).sum()
    confirmed = persistent_count.ge(float(cfg.confirmation_count))
    first_confirmed = confirmed & ~confirmed.shift(1, fill_value=False)

    rows: list[dict[str, object]] = []
    for date in dates:
        for theme in themes:
            history_value = history.loc[date, theme]
            if pd.isna(history_value) or float(history_value) < cfg.signal_history_weeks:
                status = "자료 부족"
            elif bool(first_confirmed.loc[date, theme]):
                status = "초입 확인"
            elif bool(watch.loc[date, theme]):
                status = "초입 관찰"
            else:
                status = "조건 미충족"
            rows.append(
                {
                    "decision_date": pd.Timestamp(date),
                    "theme": str(theme),
                    "early_watch": bool(watch.loc[date, theme]),
                    "early_confirmed": bool(first_confirmed.loc[date, theme]),
                    "signal_type": status,
                    "rs_4w": _number_or_nan(rs[4].loc[date, theme]),
                    "rs_13w": _number_or_nan(rs[13].loc[date, theme]),
                    "m0_26w_spy_relative": _number_or_nan(m0.loc[date, theme]),
                    "rs_rank_4w": _number_or_nan(rs_rank_4.loc[date, theme]),
                    "acceleration_rank": _number_or_nan(acceleration_rank.loc[date, theme]),
                    "rank_improvement_4v13": _number_or_nan(rank_improvement.loc[date, theme]),
                    "breadth_4w": _number_or_nan(breadth[4].loc[date, theme]),
                    "participation": _number_or_nan(participation.loc[date, theme]),
                    "history_weeks": _number_or_nan(history_value),
                    "confirmation_count": _number_or_nan(persistent_count.loc[date, theme]),
                    "status": status,
                    "recommendation": EARLY_RECOMMENDATION_KO,
                    "disclaimer": ANNUAL_DISCLAIMER_KO,
                }
            )
    return pd.DataFrame(rows).sort_values(["decision_date", "theme"]).reset_index(drop=True)


def _number_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def annual_leadership_report(
    data: pd.DataFrame,
    config: AnnualLeadershipConfig | None = None,
    *,
    as_of: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """연도별 리더와 최신 초입 신호를 한 번에 반환한다."""

    annual = annual_theme_leaders(data, config, as_of=as_of)
    early = early_rise_signals(data, config, as_of=as_of)
    if early.empty:
        latest = _empty_early()
    else:
        latest = (
            early.sort_values("decision_date")
            .groupby("theme", sort=True, as_index=False)
            .tail(1)
        )
        latest = latest.loc[latest["early_watch"] | latest["early_confirmed"]].reset_index(drop=True)
    return {
        "annual_leaders": annual,
        "early_signals": early,
        "latest_early_signals": latest,
        "disclaimer": ANNUAL_DISCLAIMER_KO,
        "as_of": as_of,
    }


__all__ = [
    "ANNUAL_DISCLAIMER_KO",
    "EARLY_RECOMMENDATION_KO",
    "AnnualLeadershipConfig",
    "annual_theme_leaders",
    "early_rise_signals",
    "annual_leadership_report",
]
