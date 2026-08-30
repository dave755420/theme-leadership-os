"""Load the small versioned PIT catalog shipped in ``data/catalog``."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data.contracts import SourceManifest
from ..domain.membership import filter_memberships_as_of
from ..domain.models import (
    HistoricalDataStatus,
    Temporal,
    ThemeDefinition,
    ThemeMembership,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG_DIR = PACKAGE_ROOT / "data" / "catalog"


@dataclass(frozen=True, slots=True)
class Catalog:
    """Immutable catalog snapshot with PIT membership accessors."""

    themes: tuple[ThemeDefinition, ...]
    memberships: tuple[ThemeMembership, ...]
    sources: tuple[SourceManifest, ...] = ()
    version: str = "1"

    def __post_init__(self) -> None:
        themes = tuple(self.themes)
        memberships = tuple(self.memberships)
        sources = tuple(self.sources)
        theme_ids = {theme.theme_id for theme in themes}
        unknown = sorted({membership.theme_id for membership in memberships} - theme_ids)
        if unknown:
            raise ValueError(f"memberships reference unknown themes: {', '.join(unknown)}")
        object.__setattr__(self, "themes", themes)
        object.__setattr__(self, "memberships", memberships)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "version", str(self.version))

    def theme(self, theme_id: str) -> ThemeDefinition:
        requested = str(theme_id).strip().lower()
        for theme in self.themes:
            if theme.theme_id == requested:
                return theme
        raise KeyError(f"unknown theme: {theme_id}")

    def memberships_as_of(
        self,
        as_of: Temporal | str,
        *,
        theme_id: str | None = None,
        exposure_tier: str | None = None,
        strict: bool = False,
    ) -> list[ThemeMembership]:
        return filter_memberships_as_of(
            self.memberships,
            as_of,
            theme_id=theme_id,
            exposure_tier=exposure_tier,
            strict=strict,
        )

    def as_of(self, as_of: Temporal | str, **kwargs: object) -> list[ThemeMembership]:
        return self.memberships_as_of(as_of, **kwargs)

    def negative_controls(self) -> tuple[ThemeDefinition, ...]:
        return tuple(theme for theme in self.themes if theme.is_negative_control)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return _minimal_yaml(path.read_text(encoding="utf-8"))
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"catalog YAML root must be a mapping: {path}")
    return payload


def _minimal_yaml(text: str) -> Mapping[str, Any]:
    """Parse the deliberately simple catalog YAML without a hard dependency."""

    # This fallback supports the shipped ``key: value`` and list-of-mappings
    # shape.  It is not intended to be a general YAML parser.
    root: dict[str, Any] = {}
    active_key: str | None = None
    active_item: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":") and not line.startswith("-"):
            active_key = line[:-1].strip()
            root[active_key] = []
            active_item = None
            continue
        if line.startswith("-"):
            if active_key is None:
                raise ValueError("invalid minimal YAML list")
            item: dict[str, Any] = {}
            root[active_key].append(item)
            active_item = item
            remainder = line[1:].strip()
            if remainder:
                key, value = remainder.split(":", 1)
                item[key.strip()] = _yaml_scalar(value.strip())
            continue
        key, value = line.split(":", 1)
        target = active_item if active_item is not None else root
        target[key.strip()] = _yaml_scalar(value.strip())
    return root


def _yaml_scalar(value: str) -> Any:
    if not value:
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def _optional(row: Mapping[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _parse_csv_memberships(path: Path) -> tuple[ThemeMembership, ...]:
    rows: list[ThemeMembership] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"membership CSV has no header: {path}")
        required = {
            "theme_id",
            "permanent_id",
            "valid_from",
            "evidence_available_at",
            "exposure_tier",
            "confidence",
        }
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"membership CSV missing columns: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                rows.append(
                    ThemeMembership(
                        theme_id=str(row["theme_id"]),
                        permanent_id=str(row["permanent_id"]),
                        symbol=str(row.get("symbol") or ""),
                        valid_from=str(row["valid_from"]),
                        valid_to=_optional(row, "valid_to"),
                        evidence_available_at=str(row["evidence_available_at"]),
                        exposure_tier=str(row["exposure_tier"]),
                        confidence=str(row["confidence"]),
                        listing_date=_optional(row, "listing_date"),
                        delisting_date=_optional(row, "delisting_date"),
                        evidence_source=str(row.get("evidence_source") or "unspecified"),
                        evidence_reference=_optional(row, "evidence_reference"),
                        historical_data_status=str(
                            row.get("historical_data_status") or HistoricalDataStatus.COMPLETE.value
                        ),
                        data_coverage_from=_optional(row, "data_coverage_from"),
                        data_coverage_to=_optional(row, "data_coverage_to"),
                        notes=_optional(row, "notes"),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid membership row {line_number}: {exc}") from exc
    return tuple(rows)


def _parse_themes(payload: Mapping[str, Any]) -> tuple[ThemeDefinition, ...]:
    raw_themes = payload.get("themes", ())
    if not isinstance(raw_themes, Sequence) or isinstance(raw_themes, (str, bytes)):
        raise ValueError("themes.yaml themes must be a list")
    themes: list[ThemeDefinition] = []
    for raw in raw_themes:
        if not isinstance(raw, Mapping):
            raise ValueError("each theme entry must be a mapping")
        themes.append(
            ThemeDefinition(
                theme_id=str(raw.get("theme_id", "")),
                name=str(raw.get("name", "")),
                description=str(raw.get("description", "")),
                category=str(raw.get("category", "")),
                is_negative_control=bool(raw.get("is_negative_control", False)),
                historical_data_status=str(raw.get("historical_data_status", "complete")),
            )
        )
    return tuple(themes)


def _parse_sources(path: Path) -> tuple[SourceManifest, ...]:
    if not path.is_file():
        return tuple()
    payload = _load_yaml(path)
    raw_sources = payload.get("sources", ())
    if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
        raise ValueError("sources.yaml sources must be a list")
    sources: list[SourceManifest] = []
    for raw in raw_sources:
        if not isinstance(raw, Mapping):
            raise ValueError("each source entry must be a mapping")
        sources.append(
            SourceManifest(
                source_id=str(raw.get("source_id", "")),
                uri=str(raw.get("uri", "")),
                sha256=str(raw.get("sha256", "")),
                retrieved_at=raw.get("retrieved_at"),
                evidence_available_at=raw.get("evidence_available_at"),
                content_type=str(raw.get("content_type", "text/csv")),
                row_count=(int(raw["row_count"]) if raw.get("row_count") is not None else None),
                complete=bool(raw.get("complete", True)),
                redistributable=bool(raw.get("redistributable", True)),
                notes=str(raw.get("notes", "")),
            )
        )
    return tuple(sources)


def load_catalog(catalog_dir: str | Path | None = None) -> Catalog:
    """Load and validate the bundled catalog or an explicit fixture directory."""

    root = Path(catalog_dir) if catalog_dir is not None else DEFAULT_CATALOG_DIR
    themes_path = root / "themes.yaml"
    memberships_path = root / "memberships.csv"
    if not themes_path.is_file():
        raise FileNotFoundError(themes_path)
    if not memberships_path.is_file():
        raise FileNotFoundError(memberships_path)
    theme_payload = _load_yaml(themes_path)
    return Catalog(
        themes=_parse_themes(theme_payload),
        memberships=_parse_csv_memberships(memberships_path),
        sources=_parse_sources(root / "sources.yaml"),
        version=str(theme_payload.get("version", "1")),
    )


__all__ = ["Catalog", "DEFAULT_CATALOG_DIR", "load_catalog"]
