from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from theme_leadership_os.validation import (
    EpisodeLabelConfig,
    classify_episode,
    evaluate_early_hits,
    future_labels,
    future_panel_labels,
    label_episodes,
)


def test_major_and_explosive_thresholds_are_deterministic():
    major = classify_episode(0.80, 0.40, breadth=0.60, active_names=4)
    assert major.major_surge is True
    assert major.explosive is False
    assert major.label == "major_surge"

    explosive = classify_episode(1.00, 0.10, breadth=0.60, active_names=4)
    assert explosive.explosive is True
    assert explosive.label == "explosive"

    no_excess = classify_episode(1.00, 0.70, breadth=0.60, active_names=4)
    assert no_excess.explosive is False

    breadth_fail = classify_episode(1.20, 0.0, breadth=0.59, active_names=4)
    assert breadth_fail.qualifies is False

    missing_breadth = classify_episode(1.20, 0.0, active_names=4)
    missing_names = classify_episode(1.20, 0.0, breadth=0.60)
    assert missing_breadth.qualifies is False
    assert missing_names.qualifies is False
    with pytest.raises(ValueError):
        classify_episode(1.20, 0.0, breadth=0.60, active_names=4, min_breadth=60)
    assert not classify_episode(
        0.80,
        0.20,
        breadth=0.60,
        active_names=4,
        theme_rank_percentile=0.89,
    ).major_surge


def test_nested_protocol_mapping_preserves_frozen_defaults():
    config = EpisodeLabelConfig.from_mapping(
        {
            "major_surge": {
                "horizon_weeks": 52,
                "min_theme_gain": 0.80,
                "min_spy_excess": 0.40,
                "min_breadth_up": 0.60,
                "min_breadth_beat_spy": 0.50,
                "min_valid_names": 4,
            },
            "explosive": {"min_theme_gain": 1.00},
        }
    )
    assert config.horizon_weeks == 52
    assert config.min_names == 4
    assert config.explosive_return == 1.0


def test_future_labels_use_next_session_not_decision_close():
    index = pd.date_range("2024-01-05", periods=70, freq="W-FRI")
    theme = pd.Series(np.arange(100.0, 170.0), index=index)
    spy = pd.Series(100.0, index=index)
    labels = future_labels(theme, spy, horizons=(13, 26, 52))
    first = labels.iloc[0]
    assert first["entry_at_13w"] == index[1]
    assert first["exit_at_13w"] == index[14]
    assert first["future_13w_return"] == (theme.iloc[14] / theme.iloc[1] - 1)
    assert pd.isna(labels.iloc[-1]["future_13w_label"])


def test_panel_future_labels_add_ranked_leader_labels_and_drawdown():
    index = pd.date_range("2024-01-05", periods=60, freq="W-FRI")
    panel = pd.DataFrame(
        {
            "leader": np.linspace(100, 220, len(index)),
            "lagger": np.linspace(100, 105, len(index)),
            "negative": np.linspace(100, 80, len(index)),
        },
        index=index,
    )
    labels = future_panel_labels(panel, pd.Series(100.0, index=index), horizons=(13,))
    assert labels.index.names == ["decision_at", "theme"]
    assert bool(labels.loc[(index[0], "leader"), "Leader_13w"])
    assert not bool(labels.loc[(index[0], "negative"), "Leader_13w"])
    assert "future_13w_max_relative_drawdown" in labels.columns


def test_episode_labels_require_breadth_and_min_names():
    index = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    theme = pd.Series(np.linspace(100, 190, len(index)), index=index)
    spy = pd.Series(np.linspace(100, 110, len(index)), index=index)
    breadth = pd.Series(0.60, index=index)
    names = pd.Series(3, index=index)
    labels = label_episodes(
        theme, spy, breadth=breadth, active_names=names, config=EpisodeLabelConfig(horizon_weeks=13)
    )
    assert labels["matured"].any()
    assert set(labels.loc[labels["matured"], "label"]) <= {"major_surge", "explosive", "none"}
    failed = label_episodes(
        theme,
        spy,
        breadth=pd.Series(0.2, index=index),
        active_names=names,
        config=EpisodeLabelConfig(horizon_weeks=13),
    )
    assert not failed.loc[failed["matured"], "qualifies"].any()


def test_early_hit_requires_persistence_and_excludes_immature_labels():
    index = pd.date_range("2020-01-03", periods=40, freq="W-FRI")
    theme = pd.Series(np.linspace(100, 260, len(index)), index=index)
    spy = pd.Series(100.0, index=index)
    signal = pd.Series(False, index=index)
    signal.iloc[0:2] = True
    report = evaluate_early_hits(
        signal,
        theme,
        spy,
        horizon_weeks=13,
        persistence=2,
        absolute_threshold=0.3,
        excess_threshold=0.2,
    )
    assert report.persistent_signals == 1
    assert bool(report.observations.iloc[0]["persistent"]) is False
    assert bool(report.observations.iloc[1]["persistent"]) is True
    assert report.matured == 2
    assert report.observations.iloc[1]["within_onset_window"] is None
    assert pd.isna(report.observations.iloc[1]["realized_log_fraction"])


def test_early_hit_uses_two_of_three_window_and_log_move_fraction():
    index = pd.date_range("2020-01-03", periods=120, freq="W-FRI")
    onset = index[30]
    peak = index[70]
    prices = pd.Series(100.0, index=index)
    prices.iloc[31:71] = np.linspace(100.0, 220.0, 40)
    spy = pd.Series(100.0, index=index)
    signal = pd.Series(False, index=index)
    signal.iloc[29:31] = True  # two true observations in the 3-week window
    episodes = pd.DataFrame({"onset": [onset], "peak": [peak], "qualifies": [True]}, index=[onset])
    report = evaluate_early_hits(
        signal,
        prices,
        spy,
        episodes=episodes,
        horizon_weeks=13,
        absolute_threshold=0.3,
        excess_threshold=0.2,
    )
    row = report.observations.loc[index[30]]
    assert bool(row["persistent"])
    assert bool(row["within_onset_window"])
    assert row["realized_log_fraction"] < 0.25
    assert row["remaining_move_fraction"] > 0.75
