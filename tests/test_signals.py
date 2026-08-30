from __future__ import annotations

import numpy as np
import pandas as pd

from theme_leadership_os.signals import (
    CurrentLeaderConfig,
    EmergingRadarConfig,
    current_leader,
    emerging_radar,
    hold_candidate_12m,
)


def _panel(
    *,
    periods: int = 80,
    rates: dict[str, list[float]] | None = None,
    benchmark_rate: float = 0.005,
    add_next_session: bool = False,
) -> pd.DataFrame:
    """Create Friday observations with optional Monday labels."""

    dates = pd.date_range("2023-01-06", periods=periods, freq="W-FRI")
    rates = rates or {
        "Leader": [0.015, 0.014, 0.016],
        "Laggard": [0.002, 0.003, 0.001],
    }
    rows: list[dict[str, object]] = []
    for i, date in enumerate(dates):
        rows.append(
            {
                "date": date,
                "theme": "__market__",
                "security": "SPY",
                "value": 100.0 * (1.0 + benchmark_rate) ** i,
            }
        )
        for theme, member_rates in rates.items():
            for member, rate in enumerate(member_rates):
                rows.append(
                    {
                        "date": date,
                        "theme": theme,
                        "security": f"{theme[:2]}{member}",
                        "value": 100.0 * (1.0 + rate) ** i,
                    }
                )
        if add_next_session and i < periods - 1:
            # The Monday label is intentionally after the decision Friday.
            # Its value is not needed by the signal calculation.
            rows.append(
                {
                    "date": date + np.timedelta64(3, "D"),
                    "theme": "__market__",
                    "security": "SPY",
                    "value": 100.0 * (1.0 + benchmark_rate) ** i,
                }
            )
    return pd.DataFrame(rows)


def _last_row(output: pd.DataFrame, date: pd.Timestamp, theme: str) -> pd.Series:
    row = output.loc[(output["decision_date"] == date) & (output["theme"] == theme)]
    assert len(row) == 1
    return row.iloc[0]


def test_current_leader_exposes_m0_rs_breadth_and_next_session() -> None:
    frame = _panel(add_next_session=True)
    decision = pd.Timestamp("2024-06-14")
    output = current_leader(frame)
    leader = _last_row(output, decision, "Leader")

    assert leader["status"] == "current_leader"
    assert bool(leader["is_current_leader"])
    assert leader["m0_26w_spy_relative"] > 0.0
    assert leader["rs_13w"] > 0.0
    assert leader["rs_26w"] > 0.0
    assert leader["rs_52w"] > 0.0
    assert leader["breadth_13w"] == 1.0
    assert leader["breadth_26w"] == 1.0
    assert leader["breadth_52w"] == 1.0
    assert leader["decision_date"].weekday() == 4
    assert leader["effective_date"] == decision + np.timedelta64(3, "D")
    assert bool(leader["effective_is_next_session"])


def test_no_lookahead_m0_is_unchanged_by_future_rally() -> None:
    frame = _panel(periods=75, rates={"Theme": [0.001, 0.001, 0.001]})
    cutoff = pd.Timestamp("2024-03-29")
    # Add a sharp rally only after the cutoff.  It must not change the
    # cutoff-Friday baseline when the full frame is supplied.
    future_dates = pd.date_range(cutoff + np.timedelta64(7, "D"), periods=10, freq="W-FRI")
    future_rows = []
    for i, date in enumerate(future_dates):
        future_rows.append(
            {"date": date, "theme": "Theme", "security": "Th0", "value": 500.0 * 1.2**i}
        )
        future_rows.extend(
            {
                "date": date,
                "theme": "Theme",
                "security": f"Th{member}",
                "value": 100.0,
            }
            for member in (1, 2)
        )
        future_rows.append(
            {"date": date, "theme": "__market__", "security": "SPY", "value": 100.0}
        )
    full = pd.concat([frame, pd.DataFrame(future_rows)], ignore_index=True)
    full_out = current_leader(full)
    old_out = current_leader(frame.loc[frame["date"] <= cutoff])
    full_row = _last_row(full_out, cutoff, "Theme")
    old_row = _last_row(old_out, cutoff, "Theme")

    assert np.isclose(full_row["m0_26w_spy_relative"], old_row["m0_26w_spy_relative"])
    assert np.isclose(full_row["rs_13w"], old_row["rs_13w"])
    assert full_row["decision_date"] == cutoff


def test_future_member_does_not_reduce_prior_participation() -> None:
    frame = _panel(periods=70)
    cutoff = pd.Timestamp("2024-02-23")
    later = pd.date_range(cutoff + np.timedelta64(7, "D"), periods=4, freq="W-FRI")
    new_member = pd.DataFrame(
        {
            "date": later,
            "theme": "Leader",
            "security": "Le3",
            "value": 100.0,
        }
    )
    full = pd.concat([frame, new_member], ignore_index=True)
    config = CurrentLeaderConfig(min_participation=1.0)
    before = current_leader(frame.loc[frame["date"] <= cutoff], config)
    after = current_leader(full, config)
    before_row = _last_row(before, cutoff, "Leader")
    after_row = _last_row(after, cutoff, "Leader")

    assert np.isclose(before_row["participation"], 1.0)
    assert np.isclose(after_row["participation"], 1.0)
    assert np.isclose(before_row["rs_26w"], after_row["rs_26w"])


def test_emerging_radar_warns_on_concentrated_false_positive() -> None:
    dates = pd.date_range("2023-01-06", periods=70, freq="W-FRI")
    rows: list[dict[str, object]] = []
    for i, date in enumerate(dates):
        rows.append({"date": date, "theme": "__market__", "security": "SPY", "value": 100.0})
        # One member rallies sharply in the fast window; the other three do
        # not participate, a deliberate false-positive/concentration case.
        for member in range(4):
            value = 100.0 * (1.25 ** max(i - 64, 0)) if member == 0 else 100.0
            rows.append(
                {"date": date, "theme": "Narrow", "security": f"N{member}", "value": value}
            )
    output = emerging_radar(
        pd.DataFrame(rows),
        EmergingRadarConfig(
            min_acceleration_rank=0.5,
            min_rank_improvement=-1.0,
            score_threshold=0.0,
        ),
    )
    row = _last_row(output, dates[-1], "Narrow")

    assert row["concentration_4w"] > 0.65
    assert row["breadth_4w"] < 0.35
    # High sensitivity keeps the narrow rally visible as a watch item; the
    # quality problems are explicit warnings rather than a hidden exclusion.
    assert row["status"] == "watch"
    assert "concentrated_rally" in row["warnings"]
    assert row["recommendation"] == "WATCH_ONLY_NOT_A_BUY_RECOMMENDATION"
    assert bool(row["not_a_buy_recommendation"])
    assert bool(row["is_watchlist"])


def test_hold_candidate_retains_insufficient_history_status() -> None:
    frame = _panel(periods=30)
    output = hold_candidate_12m(frame)
    assert not output.empty
    assert set(output["status"]) == {"insufficient_history"}
    assert output["m0_26w_spy_relative"].notna().any()
    assert output["rs_52w"].isna().all()
    assert output["warnings"].str.contains("insufficient_history").all()


def test_cross_sectional_ranks_are_deterministic_under_input_shuffle() -> None:
    frame = _panel(periods=65)
    first = current_leader(frame)
    shuffled = current_leader(frame.sample(frac=1.0, random_state=7).reset_index(drop=True))
    columns = ["decision_date", "theme", "rs_rank_13w", "rs_rank_26w", "rs_rank_52w", "leader_rank"]
    left = first.loc[:, columns].sort_values(["decision_date", "theme"]).reset_index(drop=True)
    right = shuffled.loc[:, columns].sort_values(["decision_date", "theme"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_missing_benchmark_is_explicit_and_non_actionable() -> None:
    frame = _panel(periods=60).loc[lambda value: value["security"] != "SPY"]
    output = hold_candidate_12m(frame)
    assert not output.empty
    assert output["m0_26w_spy_relative"].isna().all()
    assert output["status"].eq("insufficient_history").all()


def test_benchmark_matching_is_case_insensitive_and_custom_m0_gate_is_preserved() -> None:
    frame = _panel(periods=80)
    frame.loc[frame["security"] == "SPY", "security"] = "spy"
    config = CurrentLeaderConfig(min_relative_strength=0.5, benchmark_security="spy")
    output = current_leader(frame, config)
    row = _last_row(output, pd.Timestamp("2024-06-14"), "Leader")
    assert row["m0_26w_spy_relative"] < 0.5
    assert bool(row["m0_baseline_pass"]) is False
