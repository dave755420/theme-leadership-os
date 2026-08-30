"""Opt-in personal-research adapter for yfinance.

This module is intentionally isolated from the default fixture path.  It does
not import yfinance until a caller explicitly enables the adapter and it only
returns normalized :class:`Observation` values; the provider's raw DataFrame is
never returned, cached, serialized, or included in an exception.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

from ..domain.models import Observation, Temporal, parse_temporal
from .contracts import ProviderError, SourceManifest

ENABLE_ENV = "THEME_LEADERSHIP_ENABLE_YFINANCE"


class PersonalResearchDisabled(ProviderError):
    """Raised unless a user explicitly opts into personal-research downloads."""


class YFinanceUnavailable(ProviderError):
    """Raised when the optional yfinance dependency cannot be imported."""


class YFinancePriceProvider:
    """Explicitly gated, non-redistributable personal-research price provider.

    The adapter is not a catalog source and is not PIT-complete: yfinance's
    current response generally does not carry the original publication time.
    Returned observations are therefore marked with retrieval-time evidence and
    historical calls using an earlier ``as_of`` correctly return no rows.
    """

    redistributable = False
    pit_complete = False

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        retrieval_clock: Any | None = None,
    ) -> None:
        requested = os.environ.get(ENABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
        if not (requested if enabled is None else enabled):
            raise PersonalResearchDisabled(
                f"personal research adapter disabled; set {ENABLE_ENV}=1 "
                "or pass enabled=True for local use"
            )
        self._retrieval_clock = retrieval_clock or (lambda: datetime.now(UTC))
        self.manifest: SourceManifest | None = None

    @staticmethod
    def _ticker_from_identifier(permanent_id: str) -> str:
        text = str(permanent_id).strip()
        if text.casefold().startswith("ticker:"):
            text = text.split(":", 1)[1]
        return text.upper()

    def get_prices(
        self,
        permanent_id: str,
        start: Temporal | str,
        end: Temporal | str,
        *,
        as_of: Temporal | str | None = None,
    ) -> Sequence[Observation]:
        start_point = parse_temporal(start)
        end_point = parse_temporal(end)
        if as_of is not None:
            # Personal-research evidence is retrieval-time evidence.  A caller
            # may ask for a current as-of date, but never for a prior date.
            evidence_point = parse_temporal(as_of)
        else:
            evidence_point = None
        try:
            import yfinance as yf  # type: ignore[import-not-found]
        except ImportError as exc:
            raise YFinanceUnavailable(
                "yfinance is optional; install it in a personal environment "
                "to use this explicitly enabled adapter"
            ) from exc

        ticker_symbol = self._ticker_from_identifier(permanent_id)
        retrieval_time = parse_temporal(self._retrieval_clock())
        if evidence_point is not None and not _temporal_on_or_after(evidence_point, retrieval_time):
            return tuple()
        # yfinance's end bound is exclusive.  A one-day extension preserves
        # the provider-neutral contract's inclusive end bound for daily bars.
        end_date = _as_date(end_point) + timedelta(days=1)
        try:
            frame = yf.Ticker(ticker_symbol).history(
                start=_as_date(start_point).isoformat(),
                end=end_date.isoformat(),
                auto_adjust=False,
            )
            # Extract only scalar close/date values below.  Do not return or
            # retain the raw frame after this method returns.
            rows: list[Observation] = []
            for observed_index, row in frame.iterrows():
                close = row.get("Close")
                if close is None:
                    continue
                try:
                    value = float(close)
                except (TypeError, ValueError):
                    continue
                observed_at = (
                    observed_index.to_pydatetime()
                    if hasattr(observed_index, "to_pydatetime")
                    else observed_index
                )
                rows.append(
                    Observation(
                        permanent_id=permanent_id,
                        symbol=ticker_symbol,
                        observed_at=observed_at,
                        evidence_available_at=retrieval_time,
                        value=value,
                        source="yfinance:personal-research",
                        metadata={
                            "redistributable": "false",
                            "historical_data_status": "personal_research_only",
                        },
                    )
                )
            # Deliberately drop the provider frame reference before returning.
            del frame
            self.manifest = SourceManifest(
                source_id="yfinance:personal-research",
                uri=f"yfinance://{ticker_symbol}",
                retrieved_at=retrieval_time,
                evidence_available_at=retrieval_time,
                content_type="normalized-observation",
                row_count=len(rows),
                complete=False,
                redistributable=False,
                notes=(
                    "Personal research only; original provider payload is not "
                    "stored or redistributed and publication timing is incomplete."
                ),
            )
            if evidence_point is not None:
                rows = [row for row in rows if row.is_available_as_of(evidence_point)]
            return tuple(sorted(rows, key=lambda row: row.observed_at))
        except ProviderError:
            raise
        except Exception as exc:
            # Avoid embedding provider response text, which could contain raw
            # data or account-specific details, in the raised exception.
            raise ProviderError(f"personal research download failed for {ticker_symbol}") from exc

    def get_observations(
        self,
        permanent_id: str,
        start: Temporal | str,
        end: Temporal | str,
        *,
        as_of: Temporal | str | None = None,
    ) -> Sequence[Observation]:
        return self.get_prices(permanent_id, start, end, as_of=as_of)

    def history(
        self,
        permanent_id: str,
        start: Temporal | str,
        end: Temporal | str,
        *,
        as_of: Temporal | str | None = None,
    ) -> Sequence[Observation]:
        return self.get_prices(permanent_id, start, end, as_of=as_of)


def _as_date(value: Temporal) -> date:
    return value.date() if isinstance(value, datetime) else value


def _temporal_on_or_after(left: Temporal, right: Temporal) -> bool:
    left_date = _as_date(left)
    right_date = _as_date(right)
    return left_date >= right_date


PersonalResearchPriceProvider = YFinancePriceProvider


__all__ = [
    "ENABLE_ENV",
    "PersonalResearchDisabled",
    "PersonalResearchPriceProvider",
    "YFinancePriceProvider",
    "YFinanceUnavailable",
]
