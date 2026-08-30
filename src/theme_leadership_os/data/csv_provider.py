"""Deterministic CSV fixture price provider."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..domain.models import Observation, Temporal, parse_temporal, temporal_geq, temporal_leq
from .contracts import Cache, SourceManifest


def _first(row: Mapping[str, str], *names: str) -> str | None:
    lowered = {str(key).strip().lower(): value for key, value in row.items() if key is not None}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def _date_for_range(value: Temporal | str) -> Temporal:
    return parse_temporal(value)


class CSVPriceProvider:
    """Read normalized daily prices from a local CSV fixture.

    Accepted columns are intentionally small and provider-neutral:
    ``permanent_id`` (or ``instrument_id``), ``symbol``, ``date``/``observed_at``,
    ``close``/``price``/``value``, and optional
    ``evidence_available_at``/``available_at`` and ``source``.  If the
    evidence column is omitted, the observation date is used, which is suitable
    only for a fixture that explicitly models same-day availability.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        source_id: str | None = None,
        default_evidence_available_at: Temporal | str | None = None,
        cache: Cache | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.source_id = source_id or self.path.name
        self.default_evidence_available_at = (
            parse_temporal(default_evidence_available_at)
            if default_evidence_available_at is not None
            else None
        )
        self.cache = cache
        # Read once so a fixture cannot change half-way through a deterministic
        # run.  The manifest hashes exactly the bytes used to parse rows.
        self._payload = self.path.read_bytes()
        self._rows, missing_evidence = self._parse_rows(
            self._payload, default_evidence_available_at=self.default_evidence_available_at
        )
        self.manifest = SourceManifest.from_bytes(
            self._payload,
            source_id=self.source_id,
            uri=self.path.resolve().as_uri(),
            content_type="text/csv",
            row_count=len(self._rows),
            complete=not missing_evidence,
            notes=(
                "Fixture evidence dates default to observed_at when the CSV "
                "omits evidence_available_at."
            ),
        )

    @staticmethod
    def _parse_rows(
        payload: bytes,
        *,
        default_evidence_available_at: Temporal | None = None,
    ) -> tuple[tuple[Observation, ...], bool]:
        try:
            text = payload.decode("utf-8-sig")
            reader = csv.DictReader(text.splitlines())
        except UnicodeDecodeError as exc:
            raise ValueError("CSV fixture must be UTF-8") from exc
        if reader.fieldnames is None:
            raise ValueError("CSV fixture must include a header row")
        observations: list[Observation] = []
        missing_evidence = False
        for line_number, row in enumerate(reader, start=2):
            try:
                symbol = _first(row, "symbol", "ticker") or ""
                permanent_id = _first(
                    row,
                    "permanent_id",
                    "instrument_id",
                    "security_id",
                    "id",
                )
                if not permanent_id:
                    if not symbol:
                        raise ValueError("missing permanent_id and symbol")
                    permanent_id = f"ticker:{symbol.lower()}"
                observed_text = _first(row, "observed_at", "date", "timestamp")
                if not observed_text:
                    raise ValueError("missing observed_at/date")
                evidence_text = _first(
                    row,
                    "evidence_available_at",
                    "available_at",
                    "evidence_at",
                )
                observed_at = parse_temporal(observed_text)
                if evidence_text:
                    evidence_at = parse_temporal(evidence_text)
                elif default_evidence_available_at is not None:
                    evidence_at = default_evidence_available_at
                else:
                    # Same-day fallback is useful for a compact fixture, but
                    # manifest completeness remains false so the assumption is
                    # visible to callers.
                    evidence_at = observed_at
                    missing_evidence = True
                value_text = _first(row, "close", "adj_close", "price", "value")
                if value_text is None:
                    raise ValueError("missing close/price/value")
                value = float(value_text.replace(",", "").replace("$", ""))
                metric = _first(row, "metric") or "close"
                currency = _first(row, "currency")
                source = _first(row, "source", "source_id") or "csv_fixture"
                metadata = {
                    key: str(value)
                    for key, value in row.items()
                    if key
                    and str(key).strip().lower()
                    in {"adjusted", "data_status", "historical_data_status"}
                    and value not in (None, "")
                }
                observations.append(
                    Observation(
                        permanent_id=permanent_id,
                        symbol=symbol,
                        observed_at=observed_at,
                        evidence_available_at=evidence_at,
                        value=value,
                        source=source,
                        metric=metric,
                        currency=currency,
                        metadata=metadata,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid CSV fixture row {line_number}: {exc}") from exc
        return tuple(observations), missing_evidence

    @property
    def observations_data(self) -> tuple[Observation, ...]:
        """Read-only normalized snapshot; no raw CSV rows are exposed."""

        return self._rows

    def _matches(self, observation: Observation, identifier: str) -> bool:
        requested = str(identifier).strip()
        if not requested:
            return False
        lowered = requested.casefold()
        permanent = observation.permanent_id.casefold()
        symbol = observation.symbol.casefold()
        return lowered in {permanent, symbol}

    def get_prices(
        self,
        permanent_id: str,
        start: Temporal | str,
        end: Temporal | str,
        *,
        as_of: Temporal | str | None = None,
    ) -> Sequence[Observation]:
        start_point = _date_for_range(start)
        end_point = _date_for_range(end)
        if not temporal_leq(start_point, end_point):
            raise ValueError("start must not be after end")
        evidence_point = parse_temporal(as_of) if as_of is not None else None
        selected = [
            observation
            for observation in self._rows
            if self._matches(observation, permanent_id)
            and temporal_geq(observation.observed_at, start_point)
            and temporal_leq(observation.observed_at, end_point)
            and (
                evidence_point is None
                or observation.is_available_as_of(evidence_point)
            )
        ]
        return tuple(sorted(selected, key=lambda row: row.observed_at))

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


CSVFixtureProvider = CSVPriceProvider


__all__ = ["CSVFixtureProvider", "CSVPriceProvider"]
