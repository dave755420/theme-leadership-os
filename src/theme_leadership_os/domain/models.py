"""Typed domain objects used by the point-in-time (PIT) data layer.

The domain layer deliberately uses the standard library only.  In particular,
membership and observation objects carry the date at which a fact became
available.  Consumers must use :meth:`is_available_as_of` (or the helpers in
``membership.py``) instead of looking at a row's observation date alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum, StrEnum
from math import isfinite
from typing import Any, TypeAlias

Temporal: TypeAlias = date | datetime


class ExposureTier(StrEnum):
    """How directly an instrument expresses a theme."""

    CORE = "core"
    ADJACENT = "adjacent"
    INDIRECT = "indirect"
    NEGATIVE_CONTROL = "negative_control"


class Confidence(StrEnum):
    """Human-readable confidence bands for catalog assertions."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class HistoricalDataStatus(StrEnum):
    """Explicit coverage labels; absence of data is not treated as zero."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    PARTIAL_DELISTED = "partial_delisted"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"
    PERSONAL_RESEARCH_ONLY = "personal_research_only"


def parse_temporal(value: Temporal | str) -> Temporal:
    """Parse an ISO date/datetime while preserving its granularity.

    A date is intentionally kept as a date.  This makes a catalog's
    ``evidence_available_at: 2024-01-15`` mean the complete UTC calendar day,
    which is the useful convention for daily PIT data.  Naive datetimes are
    accepted as UTC for deterministic local fixtures.
    """

    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"expected date, datetime, or ISO string; got {value!r}")
    text = value.strip()
    try:
        if "T" in text or " " in text:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return parsed
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid ISO temporal value: {value!r}") from exc


def _utc_datetime(value: Temporal) -> datetime:
    """Convert a temporal value to a comparable UTC datetime."""

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    return datetime.combine(value, datetime.min.time(), tzinfo=UTC)


def temporal_leq(left: Temporal, right: Temporal) -> bool:
    """Compare temporals, treating a date ``right`` as the whole day."""

    if isinstance(right, date) and not isinstance(right, datetime):
        if isinstance(left, datetime):
            return left.date() <= right
        return left <= right
    return _utc_datetime(left) <= _utc_datetime(right)


def temporal_geq(left: Temporal, right: Temporal) -> bool:
    """Compare temporals, treating a date ``right`` as the whole day."""

    if isinstance(right, date) and not isinstance(right, datetime):
        if isinstance(left, datetime):
            return left.date() >= right
        return left >= right
    return _utc_datetime(left) >= _utc_datetime(right)


def temporal_lt(left: Temporal, right: Temporal) -> bool:
    """Strict temporal comparison with date granularity preserved."""

    if isinstance(right, date) and not isinstance(right, datetime):
        if isinstance(left, datetime):
            return left.date() < right
        return left < right
    return _utc_datetime(left) < _utc_datetime(right)


def temporal_to_string(value: Temporal | None) -> str | None:
    return value.isoformat() if value is not None else None


def _normalise_tier(value: ExposureTier | str) -> ExposureTier | str:
    if isinstance(value, ExposureTier):
        return value
    text = str(value).strip().lower().replace("-", "_")
    try:
        return ExposureTier(text)
    except ValueError:
        # Keeping an unknown future tier as a string permits forward-compatible
        # catalog files while known values remain type-safe enums.
        return text


def _normalise_status(value: HistoricalDataStatus | str) -> HistoricalDataStatus | str:
    if isinstance(value, HistoricalDataStatus):
        return value
    text = str(value).strip().lower().replace("-", "_")
    try:
        return HistoricalDataStatus(text)
    except ValueError:
        return text


def _normalise_confidence(value: float | int | Decimal | Confidence | str) -> float:
    if isinstance(value, Confidence):
        return {
            Confidence.HIGH: 0.9,
            Confidence.MEDIUM: 0.65,
            Confidence.LOW: 0.35,
        }[value]
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"high", "medium", "low"}:
            return {"high": 0.9, "medium": 0.65, "low": 0.35}[text]
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"confidence must be in [0, 1], got {value!r}") from exc
    if isinstance(value, bool):
        raise TypeError("confidence must be numeric, not bool")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"confidence must be numeric, got {value!r}") from exc
    if not isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"confidence must be finite and in [0, 1], got {value!r}")
    return numeric


@dataclass(frozen=True, slots=True)
class Observation:
    """One normalized, non-raw observation.

    ``evidence_available_at`` is the first date/time at which a backtest is
    allowed to know this value.  It may be later than ``observed_at`` for
    delayed or revised data, but never earlier.  The object intentionally has
    no raw provider payload field so adapters cannot accidentally redistribute
    provider-specific response data.
    """

    permanent_id: str
    observed_at: Temporal | str
    value: float | int | Decimal
    evidence_available_at: Temporal | str
    source: str = "unspecified"
    symbol: str = ""
    metric: str = "close"
    currency: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        permanent_id = str(self.permanent_id).strip()
        if not permanent_id:
            raise ValueError("permanent_id must not be empty")
        source = str(self.source).strip()
        if not source:
            raise ValueError("source must not be empty")
        observed_at = parse_temporal(self.observed_at)
        evidence_available_at = parse_temporal(self.evidence_available_at)
        if not temporal_geq(evidence_available_at, observed_at):
            raise ValueError(
                "evidence_available_at cannot precede observed_at "
                f"({evidence_available_at!r} < {observed_at!r})"
            )
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float, Decimal)):
            raise TypeError("observation value must be numeric")
        numeric = float(self.value)
        if not isfinite(numeric):
            raise ValueError("observation value must be finite")
        symbol = str(self.symbol).strip().upper()
        metric = str(self.metric).strip().lower()
        if not metric:
            raise ValueError("metric must not be empty")
        metadata = dict(self.metadata)
        object.__setattr__(self, "permanent_id", permanent_id)
        object.__setattr__(self, "observed_at", observed_at)
        object.__setattr__(self, "evidence_available_at", evidence_available_at)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "metadata", metadata)

    @property
    def available_at(self) -> Temporal:
        """Alias used by providers and generic data contracts."""

        return self.evidence_available_at

    @property
    def available_time(self) -> Temporal:
        return self.evidence_available_at

    @property
    def event_time(self) -> Temporal:
        return self.observed_at

    @property
    def instrument_id(self) -> str:
        return self.permanent_id

    def is_available_as_of(self, as_of: Temporal | str) -> bool:
        return temporal_leq(self.evidence_available_at, parse_temporal(as_of))

    def to_dict(self) -> dict[str, Any]:
        return {
            "permanent_id": self.permanent_id,
            "symbol": self.symbol,
            "observed_at": temporal_to_string(self.observed_at),
            "evidence_available_at": temporal_to_string(self.evidence_available_at),
            "value": self.value,
            "metric": self.metric,
            "currency": self.currency,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ThemeMembership:
    """A point-in-time theme membership assertion.

    Validity uses a half-open interval: ``valid_from <= as_of < valid_to``.
    ``valid_to=None`` means the assertion has no recorded end.  This avoids
    overlap when a member is reclassified on the same boundary date.
    """

    theme_id: str
    permanent_id: str
    valid_from: Temporal | str
    evidence_available_at: Temporal | str
    valid_to: Temporal | str | None = None
    exposure_tier: ExposureTier | str = ExposureTier.CORE
    confidence: float | int | Decimal | Confidence | str = 1.0
    symbol: str = ""
    listing_date: Temporal | str | None = None
    delisting_date: Temporal | str | None = None
    evidence_source: str = "unspecified"
    evidence_reference: str | None = None
    historical_data_status: HistoricalDataStatus | str = HistoricalDataStatus.COMPLETE
    data_coverage_from: Temporal | str | None = None
    data_coverage_to: Temporal | str | None = None
    notes: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        theme_id = str(self.theme_id).strip().lower()
        permanent_id = str(self.permanent_id).strip()
        if not theme_id:
            raise ValueError("theme_id must not be empty")
        if not permanent_id:
            raise ValueError("permanent_id must not be empty")
        valid_from = parse_temporal(self.valid_from)
        valid_to = parse_temporal(self.valid_to) if self.valid_to is not None else None
        evidence_available_at = parse_temporal(self.evidence_available_at)
        if valid_to is not None and not temporal_lt(valid_from, valid_to):
            raise ValueError("valid_to must be later than valid_from")
        listing_date = parse_temporal(self.listing_date) if self.listing_date is not None else None
        delisting_date = (
            parse_temporal(self.delisting_date) if self.delisting_date is not None else None
        )
        if listing_date is not None and delisting_date is not None:
            if not temporal_lt(listing_date, delisting_date):
                raise ValueError("delisting_date must be later than listing_date")
        coverage_from = (
            parse_temporal(self.data_coverage_from)
            if self.data_coverage_from is not None
            else None
        )
        coverage_to = (
            parse_temporal(self.data_coverage_to) if self.data_coverage_to is not None else None
        )
        if coverage_from is not None and coverage_to is not None:
            if not temporal_leq(coverage_from, coverage_to):
                raise ValueError("data_coverage_to must not precede data_coverage_from")
        evidence_source = str(self.evidence_source).strip()
        if not evidence_source:
            raise ValueError("evidence_source must not be empty")
        object.__setattr__(self, "theme_id", theme_id)
        object.__setattr__(self, "permanent_id", permanent_id)
        object.__setattr__(self, "valid_from", valid_from)
        object.__setattr__(self, "valid_to", valid_to)
        object.__setattr__(self, "evidence_available_at", evidence_available_at)
        object.__setattr__(self, "exposure_tier", _normalise_tier(self.exposure_tier))
        object.__setattr__(self, "confidence", _normalise_confidence(self.confidence))
        object.__setattr__(self, "symbol", str(self.symbol).strip().upper())
        object.__setattr__(self, "listing_date", listing_date)
        object.__setattr__(self, "delisting_date", delisting_date)
        object.__setattr__(self, "evidence_source", evidence_source)
        object.__setattr__(
            self,
            "historical_data_status",
            _normalise_status(self.historical_data_status),
        )
        object.__setattr__(self, "data_coverage_from", coverage_from)
        object.__setattr__(self, "data_coverage_to", coverage_to)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def id(self) -> str:
        """Permanent instrument identifier (never a display ticker)."""

        return self.permanent_id

    @property
    def instrument_id(self) -> str:
        return self.permanent_id

    @property
    def security_id(self) -> str:
        return self.permanent_id

    @property
    def asset_id(self) -> str:
        return self.permanent_id

    @property
    def evidence_at(self) -> Temporal:
        return self.evidence_available_at

    @property
    def effective_from(self) -> Temporal:
        return self.valid_from

    @property
    def effective_to(self) -> Temporal | None:
        return self.valid_to

    @property
    def suspension_date(self) -> Temporal | None:
        """First non-tradable boundary when a row has a finite validity end."""

        return self.valid_to

    def is_valid_at(self, as_of: Temporal | str) -> bool:
        point = parse_temporal(as_of)
        if not temporal_geq(point, self.valid_from):
            return False
        return self.valid_to is None or temporal_lt(point, self.valid_to)

    def is_available_as_of(self, as_of: Temporal | str) -> bool:
        """Whether this assertion is both valid and knowable at ``as_of``."""

        point = parse_temporal(as_of)
        return self.is_valid_at(point) and temporal_leq(self.evidence_available_at, point)

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme_id": self.theme_id,
            "permanent_id": self.permanent_id,
            "symbol": self.symbol,
            "valid_from": temporal_to_string(self.valid_from),
            "valid_to": temporal_to_string(self.valid_to),
            "evidence_available_at": temporal_to_string(self.evidence_available_at),
            "exposure_tier": self.exposure_tier.value
            if isinstance(self.exposure_tier, Enum)
            else self.exposure_tier,
            "confidence": self.confidence,
            "listing_date": temporal_to_string(self.listing_date),
            "delisting_date": temporal_to_string(self.delisting_date),
            "evidence_source": self.evidence_source,
            "evidence_reference": self.evidence_reference,
            "historical_data_status": self.historical_data_status.value
            if isinstance(self.historical_data_status, Enum)
            else self.historical_data_status,
            "data_coverage_from": temporal_to_string(self.data_coverage_from),
            "data_coverage_to": temporal_to_string(self.data_coverage_to),
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ThemeDefinition:
    """Catalog metadata independent of any one membership interval."""

    theme_id: str
    name: str
    description: str = ""
    category: str = ""
    is_negative_control: bool = False
    historical_data_status: HistoricalDataStatus | str = HistoricalDataStatus.COMPLETE
    metadata: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        theme_id = str(self.theme_id).strip().lower()
        name = str(self.name).strip()
        if not theme_id or not name:
            raise ValueError("theme_id and name must not be empty")
        object.__setattr__(self, "theme_id", theme_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", str(self.description).strip())
        object.__setattr__(self, "category", str(self.category).strip())
        object.__setattr__(
            self,
            "historical_data_status",
            _normalise_status(self.historical_data_status),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme_id": self.theme_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "is_negative_control": self.is_negative_control,
            "historical_data_status": self.historical_data_status.value
            if isinstance(self.historical_data_status, Enum)
            else self.historical_data_status,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "Confidence",
    "ExposureTier",
    "HistoricalDataStatus",
    "Observation",
    "Temporal",
    "ThemeDefinition",
    "ThemeMembership",
    "parse_temporal",
    "temporal_geq",
    "temporal_leq",
    "temporal_lt",
    "temporal_to_string",
]
