from __future__ import annotations

import pandas as pd
import pytest

from theme_leadership_os.validation import (
    IntegrityError,
    assert_integrity,
    audit_integrity,
    challenge_case_metadata,
    evaluate_rank_metrics,
)


def test_integrity_audit_gates_availability_membership_inception_and_coverage():
    frame = pd.DataFrame(
        {
            "decision_at": pd.to_datetime(["2024-01-05", "2024-01-05"]),
            "feature_available_at": pd.to_datetime(["2024-01-04", "2024-01-06"]),
            "symbol": ["A", "B"],
            "coverage": [1.0, 0.5],
        }
    )
    report = audit_integrity(
        frame,
        membership_dates={"A": ("2023-01-01", "2025-01-01"), "B": ("2024-02-01", None)},
        etf_inception={"A": "2023-01-01", "B": "2024-01-01"},
        min_coverage=0.8,
    )
    assert not report.passed
    codes = {issue.code for issue in report.issues}
    assert {"feature_after_decision", "membership_out_of_range", "insufficient_coverage"} <= codes
    with pytest.raises(IntegrityError):
        assert_integrity(frame, min_coverage=0.8)


def test_etf_inception_requires_twenty_six_weeks_history():
    frame = pd.DataFrame(
        {
            "decision_at": pd.to_datetime(["2024-02-01"]),
            "feature_available_at": pd.to_datetime(["2024-01-31"]),
            "symbol": ["ETF"],
            "asset_type": ["etf"],
            "inception_date": pd.to_datetime(["2024-01-01"]),
            "history_weeks": [4],
            "coverage": [1.0],
        }
    )
    report = audit_integrity(frame)
    assert "insufficient_etf_history" in {issue.code for issue in report.issues}


def test_missing_coverage_is_a_blocking_gate():
    frame = pd.DataFrame(
        {
            "decision_at": pd.to_datetime(["2024-01-05"]),
            "feature_available_at": pd.to_datetime(["2024-01-04"]),
        }
    )
    report = audit_integrity(frame)
    assert "missing_coverage" in {issue.code for issue in report.issues}


def test_rank_metrics_report_rank_ic_spread_and_hits():
    dates = pd.date_range("2024-01-05", periods=3, freq="W-FRI")
    scores = pd.DataFrame([[1, 2, 3], [3, 2, 1], [1, 3, 2]], index=dates, columns=["a", "b", "c"])
    returns = scores.astype(float) * 0.1
    report = evaluate_rank_metrics(scores, returns, min_assets=3)
    assert report.rank_ic_count == 3
    assert report.rank_ic_mean == pytest.approx(1.0)
    assert report.rank_ic_ir is None
    assert report.top_bottom_spread > 0
    assert report["top_minus_universe_excess"] > 0
    assert report.hit_rate == pytest.approx(1.0)


def test_challenge_manifest_is_holdout_only_and_has_negative_controls():
    metadata = challenge_case_metadata()
    names = {row["name"] for row in metadata}
    assert {"solar_2013", "solar_2020", "quantum_2024", "evtol_2023", "evtol_2024"} <= names
    assert {
        "3d_printing",
        "cannabis",
        "metaverse",
        "hydrogen",
        "space",
        "ev_battery",
        "clean_energy",
        "china_internet",
        "spac",
    } <= names
    assert any(row["is_negative_control"] for row in metadata)
    assert all(row["used_for_tuning"] is False for row in metadata)
