from __future__ import annotations

import numpy as np
import pandas as pd

from theme_leadership_os.validation import (
    AnnualLeadershipConfig,
    annual_leadership_report,
    annual_theme_leaders,
    early_rise_signals,
)


def _panel(periods: int = 110) -> pd.DataFrame:
    dates = pd.date_range("2022-01-07", periods=periods, freq="W-FRI")
    rows: list[dict[str, object]] = []
    for i, date in enumerate(dates):
        rows.append(
            {
                "date": date,
                "theme": "__market__",
                "security": "SPY",
                "value": 100.0 * 1.002**i,
            }
        )
        # A is deliberately weak for most of the sample and then accelerates;
        # B is a stable control.  This leaves room for a genuine 4v13-week
        # rank improvement instead of a theme that was already the leader.
        a_values = [1.0 + (-0.01 if j < 70 else 0.025) for j in range(i)]
        rows.append(
            {
                "date": date,
                "theme": "A",
                "security": "A1",
                "value": 100.0 * np.prod(a_values),
            }
        )
        rows.append(
            {"date": date, "theme": "B", "security": "B1", "value": 100.0 * 1.001**i}
        )
    return pd.DataFrame(rows)


def test_annual_leaders_rank_and_excess_return() -> None:
    output = annual_theme_leaders(
        _panel(),
        AnnualLeadershipConfig(min_weeks_per_year=20, min_coverage=0.50, top_n=1),
    )
    assert set(output["year"]) == {2022, 2023}
    assert output.loc[output["year"] == 2023, "theme"].iloc[0] == "A"
    row = output.loc[output["year"] == 2023].iloc[0]
    assert row["theme_return"] > row["benchmark_return"]
    assert row["excess_return"] > 0.0
    assert row["status"] == "ok"


def test_early_signal_requires_persistence_and_is_no_lookahead() -> None:
    frame = _panel()
    signals = early_rise_signals(frame)
    confirmed = signals.loc[signals["early_confirmed"]]
    assert not confirmed.empty
    assert set(confirmed["theme"]) == {"A"}
    first = confirmed.iloc[0]
    assert first["signal_type"] == "초입 확인"
    assert first["recommendation"] == "관찰 전용 · 매수 추천 아님"

    cutoff = pd.Timestamp("2023-05-05")
    future = frame.loc[frame["date"] > cutoff].copy()
    future.loc[future["theme"] == "A", "value"] *= 100.0
    full = early_rise_signals(pd.concat([frame.loc[frame["date"] <= cutoff], future]))
    before = signals.loc[signals["decision_date"] <= cutoff].reset_index(drop=True)
    after = full.loc[full["decision_date"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)


def test_as_of_filters_future_years_and_report_has_latest_signals() -> None:
    frame = _panel()
    report = annual_leadership_report(frame, as_of="2023-12-29")
    assert report["annual_leaders"]["year"].max() <= 2023
    assert report["early_signals"]["decision_date"].max() <= pd.Timestamp("2023-12-29")
    assert "latest_early_signals" in report


def test_missing_benchmark_does_not_produce_annual_rank() -> None:
    frame = _panel().loc[lambda value: value["security"] != "SPY"]
    output = annual_theme_leaders(
        frame,
        AnnualLeadershipConfig(min_weeks_per_year=20, min_coverage=0.50),
    )
    assert output.empty
