"""Pre-registered challenge-case metadata.

These are labels for evaluation reports only.  The manifest intentionally does
not contain per-case thresholds, and no code path reads it while fitting or
choosing thresholds.  Keeping the cases here as metadata makes it harder to
accidentally turn a famous historical move into a tuning set.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ChallengeCase:
    name: str
    theme: str
    start: str
    end: str
    expected_positive: bool
    case_type: str
    description: str
    used_for_tuning: bool = False
    source: str = "research_challenge_manifest"

    @property
    def is_negative_control(self) -> bool:
        return not self.expected_positive

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["is_negative_control"] = self.is_negative_control
        return result


# Do not change this list in response to a model result.  Any additions should
# be reviewed as a pre-registered holdout and should not add tuning parameters.
CHALLENGE_CASES: tuple[ChallengeCase, ...] = (
    ChallengeCase(
        name="solar_2013",
        theme="solar",
        start="2013-01-01",
        end="2013-12-31",
        expected_positive=True,
        case_type="historical_surge",
        description="solar-theme historical surge challenge window",
    ),
    ChallengeCase(
        name="solar_2020",
        theme="solar",
        start="2020-01-01",
        end="2020-12-31",
        expected_positive=True,
        case_type="historical_surge",
        description="solar-theme pandemic-era surge challenge window",
    ),
    ChallengeCase(
        name="quantum_2024",
        theme="quantum",
        start="2024-01-01",
        end="2024-12-31",
        expected_positive=True,
        case_type="historical_surge",
        description="quantum-theme 2024 challenge window",
    ),
    ChallengeCase(
        name="evtol_2023",
        theme="evtol",
        start="2023-01-01",
        end="2023-12-31",
        expected_positive=True,
        case_type="historical_surge",
        description="eVTOL-theme 2023 challenge window",
    ),
    ChallengeCase(
        name="evtol_2024",
        theme="evtol",
        start="2024-01-01",
        end="2024-12-31",
        expected_positive=True,
        case_type="concentration_stress",
        description="eVTOL-theme 2024 concentration stress window",
    ),
)


NEGATIVE_CONTROLS: tuple[ChallengeCase, ...] = (
    ChallengeCase(
        name="3d_printing",
        theme="3d_printing",
        start="2018-01-01",
        end="2025-12-31",
        expected_positive=False,
        case_type="negative_control",
        description="3D-printing boom/bust control",
    ),
    ChallengeCase(
        name="cannabis",
        theme="cannabis",
        start="2018-01-01",
        end="2025-12-31",
        expected_positive=False,
        case_type="negative_control",
        description="cannabis boom/bust control",
    ),
    ChallengeCase(
        name="metaverse",
        theme="metaverse",
        start="2018-01-01",
        end="2025-12-31",
        expected_positive=False,
        case_type="negative_control",
        description="metaverse failed-leadership control",
    ),
    ChallengeCase(
        name="hydrogen",
        theme="hydrogen",
        start="2018-01-01",
        end="2025-12-31",
        expected_positive=False,
        case_type="negative_control",
        description="hydrogen boom/bust control",
    ),
    ChallengeCase(
        name="space",
        theme="space",
        start="2018-01-01",
        end="2025-12-31",
        expected_positive=False,
        case_type="negative_control",
        description="space failed-leadership control",
    ),
    ChallengeCase(
        name="ev_battery",
        theme="ev_battery",
        start="2018-01-01",
        end="2025-12-31",
        expected_positive=False,
        case_type="negative_control",
        description="EV and battery failed-leadership control",
    ),
    ChallengeCase(
        name="clean_energy",
        theme="clean_energy",
        start="2018-01-01",
        end="2025-12-31",
        expected_positive=False,
        case_type="negative_control",
        description="clean-energy failed-leadership control",
    ),
    ChallengeCase(
        name="china_internet",
        theme="china_internet",
        start="2018-01-01",
        end="2025-12-31",
        expected_positive=False,
        case_type="negative_control",
        description="China internet failed-leadership control",
    ),
    ChallengeCase(
        name="spac",
        theme="spac",
        start="2018-01-01",
        end="2025-12-31",
        expected_positive=False,
        case_type="negative_control",
        description="SPAC pre-business-combination eligibility control",
    ),
)


def challenge_cases(*, include_negative_controls: bool = True) -> tuple[ChallengeCase, ...]:
    return CHALLENGE_CASES + (NEGATIVE_CONTROLS if include_negative_controls else ())


def get_challenge_case(name: str) -> ChallengeCase:
    for case in challenge_cases():
        if case.name == name:
            return case
    raise KeyError(f"unknown challenge case: {name}")


def challenge_case_metadata(*, include_negative_controls: bool = True) -> list[dict[str, Any]]:
    """Return serialisable metadata for reports and audit manifests."""

    return [
        case.as_dict()
        for case in challenge_cases(include_negative_controls=include_negative_controls)
    ]


def validate_challenge_manifest(cases: Iterable[ChallengeCase] | None = None) -> None:
    """Raise if a manifest violates the no-tuning/unique-name invariants."""

    values = tuple(cases or challenge_cases())
    names = [case.name for case in values]
    if len(names) != len(set(names)):
        raise ValueError("challenge case names must be unique")
    if any(case.used_for_tuning for case in values):
        raise ValueError("challenge cases must remain holdout-only (used_for_tuning=False)")
    for case in values:
        if case.start > case.end:
            raise ValueError(f"challenge case has reversed dates: {case.name}")


validate_challenge_manifest()


__all__ = [
    "ChallengeCase",
    "CHALLENGE_CASES",
    "NEGATIVE_CONTROLS",
    "challenge_cases",
    "get_challenge_case",
    "challenge_case_metadata",
    "validate_challenge_manifest",
]
