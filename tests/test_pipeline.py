from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from theme_leadership_os.pipeline import PipelineIntegrityError, run_signal_pipeline


def _prices() -> pd.DataFrame:
    dates = pd.date_range("2023-01-06", periods=70, freq="W-FRI")
    rows: list[dict[str, object]] = []
    for i, date in enumerate(dates):
        rows.append({"date": date, "theme": "benchmark", "security": "SPY", "value": 100 + i})
        for theme, multiplier in (("solar", 1.8), ("defensive", 0.8)):
            for member in ("A", "B", "C"):
                rows.append(
                    {
                        "date": date,
                        "theme": theme,
                        "security": f"{theme}_{member}",
                        "value": 100 + multiplier * i + (ord(member) - 65) * 0.01,
                    }
                )
    return pd.DataFrame(rows)


def test_pipeline_keeps_three_outputs_and_is_deterministic() -> None:
    prices = _prices()
    first = run_signal_pipeline(prices)
    second = run_signal_pipeline(prices.sample(frac=1.0, random_state=7))

    assert first.input_fingerprint == second.input_fingerprint
    assert set(first.latest()) == {
        "current_leaders",
        "emerging_radar",
        "hold_candidates_12m",
    }
    assert "not an investment recommendation" in first.disclaimer
    assert first.manifest()["schema"] == "theme_leadership_signal_run.v1"


def test_pipeline_as_of_ignores_later_prices() -> None:
    prices = _prices()
    cutoff = pd.Timestamp("2024-01-05")
    base = run_signal_pipeline(prices, as_of=cutoff)
    future = prices.copy()
    future.loc[future["date"] > cutoff, "value"] *= np.where(
        future.loc[future["date"] > cutoff, "theme"].eq("solar"),
        20.0,
        1.0,
    )
    changed = run_signal_pipeline(future, as_of=cutoff)
    assert base.input_fingerprint == changed.input_fingerprint
    pd.testing.assert_frame_equal(base.current_leaders, changed.current_leaders)


def test_pipeline_rejects_late_availability() -> None:
    prices = _prices().iloc[:10].copy()
    prices["feature_available_at"] = prices["date"]
    prices.loc[prices.index[0], "feature_available_at"] = (
        pd.Timestamp(prices.loc[prices.index[0], "date"]) + np.timedelta64(2, "D")
    )
    with pytest.raises(PipelineIntegrityError, match="later than"):
        run_signal_pipeline(prices)
