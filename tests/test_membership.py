from datetime import date

import pytest

from theme_leadership_os.catalog import load_catalog
from theme_leadership_os.domain import (
    ExposureTier,
    LookaheadError,
    Observation,
    ThemeMembership,
    assert_no_lookahead,
    filter_memberships_as_of,
)


def _membership(**overrides: object) -> ThemeMembership:
    values: dict[str, object] = {
        "theme_id": "solar",
        "permanent_id": "ticker:aaa",
        "symbol": "AAA",
        "valid_from": date(2020, 1, 1),
        "valid_to": date(2021, 1, 1),
        "evidence_available_at": date(2020, 2, 1),
        "exposure_tier": ExposureTier.CORE,
        "confidence": 0.9,
    }
    values.update(overrides)
    return ThemeMembership(**values)


def test_as_of_requires_validity_and_evidence_timing() -> None:
    row = _membership()
    assert filter_memberships_as_of([row], date(2020, 1, 31)) == []
    assert filter_memberships_as_of([row], date(2020, 2, 1)) == [row]
    assert filter_memberships_as_of([row], date(2020, 12, 31)) == [row]
    # valid_to is a half-open boundary, so no stale membership leaks forward.
    assert filter_memberships_as_of([row], date(2021, 1, 1)) == []


def test_strict_no_lookahead_guard_fails_loudly() -> None:
    row = _membership(evidence_available_at=date(2020, 6, 1))
    with pytest.raises(LookaheadError):
        assert_no_lookahead([row], date(2020, 5, 31))
    with pytest.raises(LookaheadError):
        filter_memberships_as_of([row], date(2020, 5, 31), strict=True)
    # Normal selection is safe by exclusion, not by silently changing dates.
    assert filter_memberships_as_of([row], date(2020, 5, 31)) == []


def test_domain_observation_rejects_impossible_evidence_time() -> None:
    with pytest.raises(ValueError, match="cannot precede observed_at"):
        Observation(
            permanent_id="ticker:aaa",
            observed_at=date(2020, 1, 2),
            evidence_available_at=date(2020, 1, 1),
            value=10.0,
        )


def test_seed_catalog_keeps_lilm_and_negative_controls() -> None:
    catalog = load_catalog()
    assert {theme.theme_id for theme in catalog.themes} >= {
        "solar",
        "quantum_computing",
        "evtol",
        "broad_market",
        "traditional_oil",
        "consumer_staples",
    }
    lilm = [row for row in catalog.memberships if row.symbol == "LILM"]
    assert len(lilm) == 1
    assert lilm[0].permanent_id == "ticker:lilm"
    assert lilm[0].historical_data_status == "partial_delisted"
    assert lilm[0].valid_to == date(2024, 11, 6)
    assert lilm[0].suspension_date == date(2024, 11, 6)
    assert lilm[0].delisting_date == date(2025, 3, 31)
    assert lilm[0].data_coverage_to == date(2024, 11, 5)
    assert lilm[0].notes and "no fabricated" in lilm[0].notes.lower()
    assert "sec.gov" in (lilm[0].evidence_reference or "")
    controls = catalog.negative_controls()
    assert {theme.theme_id for theme in controls} >= {
        "broad_market",
        "traditional_oil",
        "consumer_staples",
    }
