from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from theme_leadership_os.validation.early_validation import validate_early_radar
from theme_leadership_os.validation.protocol import load_protocol, protocol_alignment_mismatches


def _panel(periods: int = 190, themes: int = 18) -> pd.DataFrame:
    dates = pd.date_range("2019-01-04", periods=periods, freq="W-FRI")
    rows: list[dict[str, object]] = []
    levels = {f"T{i:02d}": 100.0 for i in range(themes)}
    spy = 100.0
    for i, date in enumerate(dates):
        spy *= 1.002
        rows.append(
            {
                "date": date,
                "available_at": date,
                "theme": "__market__",
                "security": "SPY",
                "value": spy,
            }
        )
        for j in range(themes):
            theme = f"T{j:02d}"
            if j == 0:
                # Keep the 13-week rank weak long enough for 4-week acceleration
                # to persist for the frozen 2-of-3 confirmation rule.
                weekly = -0.01 if i < 75 else (0.025 if i < 115 else 0.004)
            else:
                weekly = 0.0005 + j * 0.00003
            levels[theme] *= 1.0 + weekly
            rows.append(
                {
                    "date": date,
                    "available_at": date,
                    "theme": theme,
                    "security": f"{theme}A",
                    "value": levels[theme],
                }
            )
    return pd.DataFrame(rows)


def _manifest(**extra: object) -> dict[str, object]:
    base: dict[str, object] = {
        "total_return_adjusted": True,
        "point_in_time_membership_complete": True,
        "delisting_complete": True,
        "source_registry_frozen": True,
        "availability_semantics_frozen": True,
        "portfolio_backtest_complete": False,
    }
    base.update(extra)
    return base


def test_frozen_protocol_matches_python_signal_defaults() -> None:
    assert protocol_alignment_mismatches(load_protocol()) == []


def test_validator_builds_m0_and_continuous_early_score_comparison() -> None:
    result = validate_early_radar(_panel(), data_manifest=_manifest())
    assert result.status == "NOT_READY"
    assert "early_score" in result.signal_panel.columns
    assert "m0_rank" in result.signal_panel.columns
    assert result.summary["m0_26w"]["weeks"] > 0
    assert result.summary["early_score_26w"]["weeks"] > 0
    assert "rank_ic_delta" in result.summary["comparison_vs_m0"]
    assert result.summary["data_integrity"]["availability_violations"] == 0


def test_missing_data_manifest_is_never_promoted_to_pass() -> None:
    result = validate_early_radar(_panel())
    assert result.status == "NOT_READY"
    assert not result.summary["data_integrity"]["passed"]
    assert "total_return_adjusted" in result.summary["data_integrity"]["manifest_missing"]


def test_temporal_violation_aborts_metrics() -> None:
    frame = _panel()
    frame.loc[0, "available_at"] = pd.Timestamp(frame.loc[0, "date"]) + pd.Timedelta("1D")
    try:
        validate_early_radar(frame, data_manifest=_manifest())
    except ValueError as exc:
        assert "timing violation" in str(exc)
    else:
        raise AssertionError("temporal violations must stop validation")


def test_formal_episode_file_enables_early_discovery_measurement() -> None:
    frame = _panel()
    first = validate_early_radar(frame, data_manifest=_manifest())
    confirmed = first.signal_panel.loc[
        first.signal_panel["theme"].eq("T00") & first.signal_panel["early_confirmed"]
    ]
    assert not confirmed.empty
    onset = pd.Timestamp(confirmed.iloc[0]["decision_date"])
    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    onset_pos = int(np.where(dates == onset)[0][0])
    peak = dates[min(onset_pos + 12, len(dates) - 1)]
    episodes = pd.DataFrame(
        [{"theme": "T00", "onset": onset, "peak": peak, "qualifies": True}]
    )
    protocol = deepcopy(load_protocol())
    protocol["early_discovery_acceptance"]["min_independent_episodes"] = 1
    protocol["early_discovery_acceptance"]["min_episode_themes"] = 1
    result = validate_early_radar(
        frame,
        episodes=episodes,
        data_manifest=_manifest(),
        protocol=protocol,
    )
    gate = result.summary["early_discovery_gate"]
    assert gate["ready"]
    assert gate["episode_count"] == 1
    assert gate["recall"] == 1.0
    assert not result.episode_metrics.empty
