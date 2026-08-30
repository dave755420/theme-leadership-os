"""Command line entry points for Theme Leadership OS.

The CLI is intentionally local-first.  ``score`` and ``validate`` accept a
CSV/JSON file and never make a network request unless a future caller opts in
explicitly.  The scoring adapter in this module is deliberately small so the
CLI remains usable while the project's research engines are optional; when
those engines are available, callers can still pass their exported rows to
the UI without changing the presentation contract.

The project uses Typer when it is installed.  A tiny argparse fallback keeps
``python -m theme_leadership_os.cli`` useful in a fresh checkout before
optional UI dependencies have been installed (and makes ``doctor`` useful in
that situation too).
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import importlib
import json
import math
import statistics
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # Typer is a runtime dependency in the packaged application, but optional here.
    import typer  # type: ignore
except Exception:  # pragma: no cover - exercised only in minimal environments
    typer = None  # type: ignore


DISCLAIMER = "Research candidate, not investment recommendation."
TAB_LABELS = ("Current Leader", "Emerging Radar", "12M Hold Candidate")
COMPONENT_WEIGHTS = {
    "momentum": 0.40,
    "breadth": 0.25,
    "quality": 0.20,
    "risk_control": 0.15,
}
_MISSING = {"", "-", "--", "n/a", "na", "nan", "null", "none", "—", "–"}


def _repo_root() -> Path:
    """Best-effort repository root for both editable and installed layouts."""

    # ``src/theme_leadership_os/cli.py`` -> repository root is parents[2].
    try:
        return Path(__file__).resolve().parents[2]
    except IndexError:  # pragma: no cover - defensive for unusual loaders
        return Path.cwd()


def _candidate_catalog_dirs() -> list[Path]:
    roots = [Path.cwd(), _repo_root(), Path(__file__).resolve().parent]
    seen: set[Path] = set()
    result: list[Path] = []
    for root in roots:
        for candidate in (root / "data" / "catalog", root / "catalog"):
            candidate = candidate.resolve()
            if candidate not in seen:
                seen.add(candidate)
                result.append(candidate)
    return result


def _first_catalog_file(path: Path | None = None) -> Path | None:
    if path is not None:
        if path.is_file():
            return path
        if path.is_dir():
            files = sorted(
                p
                for p in path.iterdir()
                if p.suffix.lower() in {".csv", ".json", ".jsonl"}
            )
            return files[0] if files else None
        return None
    for directory in _candidate_catalog_dirs():
        if directory.is_dir():
            files = sorted(
                p
                for p in directory.iterdir()
                if p.suffix.lower() in {".csv", ".json", ".jsonl"}
            )
            if files:
                return files[0]
    return None


def _canonical(value: Any) -> str:
    text = str(value or "").strip().lower()
    return "".join(ch for ch in text if ch.isalnum())


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if text.lower() in _MISSING:
        return None
    text = text.replace(",", "").replace("$", "")
    is_percent = text.endswith("%")
    if is_percent:
        text = text[:-1]
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number / 100.0 if is_percent else number


def _parse_date(value: Any) -> _dt.date | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in _MISSING:
        return None
    # ISO timestamps, including a trailing Z, are common in exports.
    try:
        return _dt.datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in ("%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%d-%b-%Y", "%b %d %Y"):
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read local CSV/JSON/JSONL rows without requiring pandas."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, Mapping)]
        if isinstance(payload, Mapping):
            for key in ("rows", "data", "records", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [dict(row) for row in value if isinstance(row, Mapping)]
            return [dict(payload)]
        return []
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, Mapping):
                    rows.append(dict(value))
        return rows
    # CSV is the supported interchange format for long and wide extracts.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        return [dict(row) for row in reader if any(str(v or "").strip() for v in row.values())]


def _pick_key(keys: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_canonical = {_canonical(key): key for key in keys}
    for candidate in candidates:
        if _canonical(candidate) in by_canonical:
            return by_canonical[_canonical(candidate)]
    return None


def _normalise_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalise both long and wide extracts into entity/date/value rows.

    Long examples usually look like ``date,symbol,value``.  Wide examples
    commonly look like ``date,IONQ,BOTZ,CIBR``; each numeric column becomes an
    entity while preserving the source column in ``source_field``.
    """

    normalised: list[dict[str, Any]] = []
    entity_candidates = ("entity", "theme", "symbol", "ticker", "asset", "name", "series", "id")
    date_candidates = ("date", "as_of", "asof", "timestamp", "datetime", "period")
    value_candidates = (
        "value",
        "close",
        "price",
        "index",
        "score",
        "signal",
        "return",
        "return_12m",
        "momentum",
    )

    for original in rows:
        row = {str(key).strip(): value for key, value in original.items() if key is not None}
        keys = list(row)
        date_key = _pick_key(keys, date_candidates)
        entity_key = _pick_key(keys, entity_candidates)
        value_key = _pick_key(keys, value_candidates)
        date_value = row.get(date_key) if date_key else None
        parsed_date = _parse_date(date_value)

        if entity_key:
            entity = str(row.get(entity_key) or "").strip()
            if not entity:
                entity = "Unknown"
            value = _safe_float(row.get(value_key)) if value_key else None
            # If no conventional value name exists, choose the first numeric
            # field other than metadata.  This supports small analyst exports.
            if value is None:
                for key, raw in row.items():
                    if key in {entity_key, date_key}:
                        continue
                    candidate = _safe_float(raw)
                    if candidate is not None:
                        value_key, value = key, candidate
                        break
            if value is not None:
                normalised.append(
                    {
                        "entity": entity,
                        "date": parsed_date,
                        "value": value,
                        "source_field": value_key or "value",
                        "raw": row,
                    }
                )
            continue

        # Wide row: any numeric non-date column is one entity observation.
        numeric_columns: list[tuple[str, float]] = []
        for key, raw in row.items():
            if key == date_key:
                continue
            candidate = _safe_float(raw)
            if candidate is not None:
                numeric_columns.append((key, candidate))
        for key, value in numeric_columns:
            normalised.append(
                {
                    "entity": key.strip() or "Unknown",
                    "date": parsed_date,
                    "value": value,
                    "source_field": key,
                    "raw": row,
                }
            )
    return normalised


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _freshness_days(as_of: _dt.date | None) -> int | None:
    if as_of is None:
        return None
    return max(0, (_dt.date.today() - as_of).days)


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.80:
        return "High"
    if confidence >= 0.55:
        return "Medium"
    return "Low"


def _score_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compute a transparent, deterministic local scorecard.

    This is a presentation-safe fallback, not a replacement for the research
    engine.  It intentionally exposes each component and its weight so a user
    can see how a candidate arrived on a dashboard tab.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        entity = str(record.get("entity") or record.get("symbol") or "Unknown").strip()
        grouped.setdefault(entity, []).append(record)

    output: list[dict[str, Any]] = []
    for entity, items in grouped.items():
        ordered = sorted(
            items,
            key=lambda row: row.get("date") or _dt.date.min,
        )
        values = [
            numeric
            for row in ordered
            if (numeric := _safe_float(row.get("value"))) is not None
        ]
        dates = [row.get("date") for row in ordered if isinstance(row.get("date"), _dt.date)]
        returns: list[float] = []
        for previous, current in zip(values, values[1:]):
            if previous != 0:
                returns.append(current / previous - 1.0)
        if len(values) >= 2 and values[0] != 0:
            growth = values[-1] / values[0] - 1.0
        else:
            growth = 0.0
        volatility = statistics.pstdev(returns) if len(returns) >= 2 else 0.0
        momentum = _clamp(50.0 + growth * 150.0)
        # Observations are a useful proxy for breadth in a single-series CSV;
        # multiple source fields/rows naturally increase this component.
        breadth = _clamp(25.0 + len(values) / 12.0 * 75.0)
        quality = _clamp(35.0 + min(1.0, len(values) / 20.0) * 65.0)
        risk_control = _clamp(100.0 - volatility * 500.0)
        score = sum(
            COMPONENT_WEIGHTS[key] * component
            for key, component in {
                "momentum": momentum,
                "breadth": breadth,
                "quality": quality,
                "risk_control": risk_control,
            }.items()
        )
        as_of = max(dates) if dates else None
        freshness = _freshness_days(as_of)
        confidence = _clamp(
            0.55 * min(1.0, len(values) / 20.0)
            + 0.25 * (1.0 if dates else 0.0)
            + 0.20 * (1.0 if len(values) >= 2 else 0.0),
            0.0,
            1.0,
        )
        risk_flags: list[str] = []
        if len(values) < 5:
            risk_flags.append("sparse history")
        if not dates:
            risk_flags.append("missing as-of date")
        if volatility > 0.08:
            risk_flags.append("high volatility")
        if freshness is not None and freshness > 45:
            risk_flags.append("stale data")
        if not risk_flags:
            risk_flags.append("none flagged")
        if score >= 70:
            bucket = TAB_LABELS[0]
        elif score >= 50:
            bucket = TAB_LABELS[1]
        else:
            bucket = TAB_LABELS[2]
        components = {
            "momentum": round(momentum, 2),
            "breadth": round(breadth, 2),
            "quality": round(quality, 2),
            "risk_control": round(risk_control, 2),
        }
        output.append(
            {
                "entity": entity,
                "score": round(score, 2),
                "signal_score": round(score, 2),
                "bucket": bucket,
                "components": components,
                "component_weights": COMPONENT_WEIGHTS.copy(),
                "data_confidence": round(confidence * 100.0, 1),
                "confidence": round(confidence, 3),
                "confidence_label": _confidence_label(confidence),
                "risk_flags": risk_flags,
                "as_of": as_of.isoformat() if as_of else None,
                "freshness_days": freshness,
                "observations": len(values),
                "disclaimer": DISCLAIMER,
            }
        )
    return sorted(output, key=lambda row: (-float(row["score"]), str(row["entity"])))


def _tidy_signal_frame(raw_rows: Sequence[Mapping[str, Any]]) -> Any:
    """Build the core signal engine's tidy frame when the schema is present.

    The import is intentionally optional.  This adapter lets the CLI expose
    the project's PIT-aware signal functions when a user supplies their
    ``date,theme,security,value`` export, while the generic local scorer still
    handles simple long/wide CSVs and minimal installations.
    """

    try:
        import pandas as pd  # type: ignore
    except Exception:
        return None
    if not raw_rows:
        return None
    first = {str(key).strip().lower() for key in raw_rows[0]}
    aliases = {
        "date": ("date", "observed_at", "timestamp"),
        "theme": ("theme", "theme_id"),
        "security": ("security", "symbol", "ticker"),
        "value": ("value", "close", "price", "adj_close"),
    }
    keys: dict[str, str] = {}
    for name, choices in aliases.items():
        key = next((candidate for candidate in choices if candidate in first), None)
        if key is None:
            return None
        keys[name] = key
    columns: list[dict[str, Any]] = []
    for row in raw_rows:
        lowered = {str(key).strip().lower(): value for key, value in row.items()}
        columns.append(
            {
                "date": lowered.get(keys["date"]),
                "theme": lowered.get(keys["theme"]),
                "security": lowered.get(keys["security"]),
                "value": _safe_float(lowered.get(keys["value"])),
            }
        )
    frame = pd.DataFrame(columns)
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date", "theme", "security", "value"])
    return frame if not frame.empty else None


def _scalar(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _core_component(
    frame_row: Mapping[str, Any], names: Sequence[str], default: float = 0.0
) -> float:
    for name in names:
        value = frame_row.get(name)
        if value is None:
            continue
        number = _scalar(value, float("nan"))
        if math.isfinite(number):
            return _clamp(number * 100.0 if abs(number) <= 1.5 else number)
    return _clamp(default)


def _core_scorecard(raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]] | None:
    """Adapt PIT-aware signal outputs to the dashboard scorecard contract."""

    frame = _tidy_signal_frame(raw_rows)
    if frame is None:
        return None
    try:
        from .signals import current_leader, emerging_radar, hold_candidate_12m
    except ImportError:
        # Only a build missing the core package may use the generic adapter.
        return None
    outputs = (
        (TAB_LABELS[0], current_leader(frame), "leader_score"),
        (TAB_LABELS[1], emerging_radar(frame), "radar_score"),
        (TAB_LABELS[2], hold_candidate_12m(frame), "candidate_score"),
    )

    # Keep one latest row per engine/theme.  A theme can legitimately appear
    # in more than one lane while it is moving through gates; collapsing these
    # rows would hide the distinction between leader, radar, and hold logic.
    by_theme: dict[str, dict[str, Any]] = {}
    for bucket, output, score_column in outputs:
        if output is None or getattr(output, "empty", True):
            continue
        try:
            frame_output = output.copy()
            if "decision_date" in frame_output:
                frame_output["decision_date"] = frame_output["decision_date"].astype(str)
                frame_output = frame_output.sort_values(["decision_date", "theme"])
            latest = frame_output.groupby("theme", sort=True, as_index=False).tail(1)
        except Exception:
            continue
        for _, row in latest.iterrows():
            theme = str(row.get("theme") or "Unknown")
            status = str(row.get("status") or "")
            score = _core_component(row, (score_column, "score", "signal_score"), 0.0)
            momentum = _core_component(
                row,
                ("rs_rank_13w", "rs_rank_4w", "acceleration_rank", "candidate_score"),
                score,
            )
            breadth = _core_component(
                row, ("breadth_average", "breadth_4w", "breadth_13w"), 50.0
            )
            quality = _clamp(_scalar(row.get("history_weeks"), 0.0) / 52.0 * 100.0)
            volatility = _scalar(row.get("annualized_volatility_26w"), float("nan"))
            risk_control = (
                _clamp(100.0 - volatility * 100.0) if math.isfinite(volatility) else 75.0
            )
            flags: list[str] = []
            warning_value = row.get("warnings") or row.get("warning")
            if warning_value is not None and str(warning_value).strip().lower() not in {
                "",
                "nan",
                "none",
            }:
                flags.extend(flag.strip() for flag in str(warning_value).split(";") if flag.strip())
            if (
                status
                in {"insufficient_history", "warning", "risk_rejected", "awaiting_confirmation"}
                and not flags
            ):
                flags.append(status)
            if not flags:
                flags.append("none flagged")
            as_of = row.get("decision_date")
            if hasattr(as_of, "date"):
                as_of = as_of.date().isoformat()
            elif as_of is not None:
                as_of = str(as_of)[:10]
            freshness = _freshness_days(_parse_date(as_of))
            if freshness is not None and freshness > 45:
                flags.append("stale data")
            history = _scalar(row.get("history_weeks"), 0.0)
            confidence = _clamp(
                min(1.0, history / 52.0)
                * (0.85 if flags != ["none flagged"] else 1.0),
                0.0,
                1.0,
            )
            by_theme[f"{bucket}:{theme}"] = {
                "entity": theme,
                "score": round(score, 2),
                "signal_score": round(score, 2),
                "bucket": bucket,
                "engine": {
                    TAB_LABELS[0]: "current_leader",
                    TAB_LABELS[1]: "emerging_radar",
                    TAB_LABELS[2]: "hold_candidate_12m",
                }[bucket],
                "status": status,
                "components": {
                    "momentum": round(momentum, 2),
                    "breadth": round(breadth, 2),
                    "quality": round(quality, 2),
                    "risk_control": round(risk_control, 2),
                },
                "component_weights": COMPONENT_WEIGHTS.copy(),
                "data_confidence": round(confidence * 100.0, 1),
                "confidence": round(confidence, 3),
                "confidence_label": _confidence_label(confidence),
                "risk_flags": list(dict.fromkeys(flags)),
                "as_of": as_of,
                "freshness_days": freshness,
                "observations": int(history) if history else None,
                "disclaimer": DISCLAIMER,
            }
    if not by_theme:
        return None
    return sorted(by_theme.values(), key=lambda row: (-float(row["score"]), str(row["entity"])))


def synthetic_demo_rows() -> list[dict[str, Any]]:
    """Return a small in-memory scorecard for the dashboard and smoke tests."""

    as_of = (_dt.date.today() - _dt.timedelta(days=2)).isoformat()
    templates = [
        ("AI Infrastructure", 86.4, TAB_LABELS[0], 88.0, 84.0, 91.0, 79.0, 0.88, ["none flagged"]),
        (
            "Cybersecurity",
            66.8,
            TAB_LABELS[1],
            73.0,
            65.0,
            70.0,
            53.0,
            0.71,
            ["valuation sensitivity"],
        ),
        (
            "Grid Modernization",
            44.7,
            TAB_LABELS[2],
            43.0,
            56.0,
            63.0,
            18.0,
            0.58,
            ["high volatility", "sparse history"],
        ),
    ]
    rows: list[dict[str, Any]] = []
    for entity, score, bucket, momentum, breadth, quality, risk, confidence, flags in templates:
        rows.append(
            {
                "entity": entity,
                "score": score,
                "signal_score": score,
                "bucket": bucket,
                "engine": {
                    TAB_LABELS[0]: "current_leader",
                    TAB_LABELS[1]: "emerging_radar",
                    TAB_LABELS[2]: "hold_candidate_12m",
                }[bucket],
                "components": {
                    "momentum": momentum,
                    "breadth": breadth,
                    "quality": quality,
                    "risk_control": risk,
                },
                "component_weights": COMPONENT_WEIGHTS.copy(),
                "data_confidence": round(confidence * 100.0, 1),
                "confidence": confidence,
                "confidence_label": _confidence_label(confidence),
                "risk_flags": flags,
                "as_of": as_of,
                "freshness_days": 2,
                "observations": 24,
                "disclaimer": DISCLAIMER,
            }
        )
    return rows


def _catalog_items(path: Path | None = None) -> tuple[Path | None, list[dict[str, Any]]]:
    # Prefer the versioned catalog loader when the canonical themes.yaml +
    # memberships.csv pair is present.  This keeps catalog output aligned with
    # the domain objects (including their point-in-time metadata) while the
    # generic reader below still accepts ad-hoc CSV/JSON files.
    catalog_dirs: list[Path] = []
    if path is not None:
        # An explicit file means "list this file"; only a directory selects
        # the canonical themes/memberships pair.
        if path.is_dir():
            catalog_dirs.append(path)
    else:
        catalog_dirs.extend(_candidate_catalog_dirs())
    for catalog_dir in catalog_dirs:
        if not (catalog_dir / "themes.yaml").is_file() or not (
            catalog_dir / "memberships.csv"
        ).is_file():
            continue
        try:
            from .catalog import load_catalog

            loaded = load_catalog(catalog_dir)
            membership_counts: dict[str, int] = {}
            for membership in loaded.memberships:
                membership_counts[membership.theme_id] = (
                    membership_counts.get(membership.theme_id, 0) + 1
                )
            items = []
            for theme in loaded.themes:
                item = theme.to_dict()
                item["theme"] = item.get("name") or item.get("theme_id")
                item["memberships"] = membership_counts.get(theme.theme_id, 0)
                item["source"] = str(catalog_dir)
                items.append(item)
            if items:
                return catalog_dir / "themes.yaml", items
        except Exception:
            # Keep catalog listing useful if optional YAML support or a draft
            # catalog has a validation issue; the raw CSV fallback follows.
            pass
    selected = _first_catalog_file(path)
    if selected is not None:
        try:
            rows = _read_rows(selected)
            items = []
            for row in rows:
                item = {str(k): v for k, v in row.items()}
                item.setdefault("source", str(selected))
                items.append(item)
            if items:
                return selected, items
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    # The built-in fallback describes the UI's three lanes and is not a live
    # recommendation universe.
    fallback = [
        {"theme": "AI Infrastructure", "lane": TAB_LABELS[0], "source": "bundled demo"},
        {"theme": "Cybersecurity", "lane": TAB_LABELS[1], "source": "bundled demo"},
        {"theme": "Grid Modernization", "lane": TAB_LABELS[2], "source": "bundled demo"},
    ]
    return selected, fallback


def _doctor_report() -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    checks.append({"name": "python", "status": "ok", "detail": sys.version.split()[0]})
    checks.append(
        {
            "name": "typer",
            "status": "ok" if typer is not None else "optional",
            "detail": (
                "installed"
                if typer is not None
                else "not installed; argparse fallback enabled"
            ),
        }
    )
    try:
        importlib.import_module("pandas")
        pandas_status, pandas_detail = "ok", "installed"
    except Exception:
        pandas_status, pandas_detail = "optional", "not installed; CSV parser remains available"
    checks.append({"name": "pandas", "status": pandas_status, "detail": pandas_detail})
    try:
        importlib.import_module("streamlit")
        streamlit_status, streamlit_detail = "ok", "installed"
    except Exception:
        streamlit_status, streamlit_detail = (
            "optional",
            "not installed; dashboard command will explain how to install it",
        )
    checks.append({"name": "streamlit", "status": streamlit_status, "detail": streamlit_detail})
    catalog = _first_catalog_file()
    checks.append(
        {
            "name": "catalog",
            "status": "ok" if catalog else "optional",
            "detail": str(catalog) if catalog else "no local catalog; bundled demo available",
        }
    )
    checks.append(
        {
            "name": "network",
            "status": "disabled",
            "detail": "local-first; live fetch is never automatic",
        }
    )
    return {"ok": True, "checks": checks, "disclaimer": DISCLAIMER}


def _validate_file(path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if not path.exists():
        return {
            "path": str(path),
            "valid": False,
            "errors": [f"file not found: {path}"],
            "warnings": [],
        }
    try:
        raw_rows = _read_rows(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "path": str(path),
            "valid": False,
            "errors": [f"could not read file: {exc}"],
            "warnings": [],
        }
    if not raw_rows:
        errors.append("file has no data rows")
    rows = _normalise_rows(raw_rows)
    if raw_rows and not rows:
        errors.append("no numeric value column found; expected long or wide CSV")
    if rows:
        source_entities = {str(row.get("entity") or "") for row in rows}
        if not source_entities:
            errors.append("missing entity/ticker columns")
        if any(row.get("date") is None for row in rows):
            warnings.append("some observations have no parseable date")
        if len(rows) < 5:
            warnings.append("fewer than five observations; confidence will be low")
        seen: set[tuple[str, Any]] = set()
        duplicate_count = 0
        for row in rows:
            identity = (str(row.get("entity")), row.get("date"))
            if row.get("date") is not None and identity in seen:
                duplicate_count += 1
            seen.add(identity)
        if duplicate_count:
            warnings.append(f"{duplicate_count} duplicate entity/date observations")
    return {
        "path": str(path),
        "valid": not errors,
        "rows": len(raw_rows),
        "normalised_rows": len(rows),
        "entities": sorted({row["entity"] for row in rows}) if rows else [],
        "errors": errors,
        "warnings": warnings,
    }


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    values = [[str(value if value is not None else "") for value in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in values:
        for index, value in enumerate(row):
            if index < len(widths):
                widths[index] = max(widths[index], len(value))
    line = "  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers))
    divider = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(row[index].ljust(widths[index]) for index in range(len(headers)))
        for row in values
    ]
    return "\n".join([line, divider, *body]) if body else line


def _emit(payload: Any, *, json_output: bool, table: str | None = None) -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    elif table is not None:
        print(table)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _as_json(json_output: bool = False, output_format: str = "table") -> bool:
    """Resolve the human/machine output switches used by all commands."""

    return json_output or str(output_format).strip().lower() == "json"


def _run_doctor(*, json_output: bool = False, output_format: str = "table") -> int:
    report = _doctor_report()
    rows = [(item["name"], item["status"], item["detail"]) for item in report["checks"]]
    _emit(
        report,
        json_output=_as_json(json_output, output_format),
        table=_format_table(("check", "status", "detail"), rows),
    )
    return 0


def _run_catalog(
    path: Path | None = None,
    *,
    json_output: bool = False,
    output_format: str = "table",
) -> int:
    source, items = _catalog_items(path)
    payload = {
        "source": str(source) if source else "bundled demo",
        "items": items,
        "disclaimer": DISCLAIMER,
    }
    table_rows = []
    for item in items:
        theme = item.get("theme") or item.get("name") or item.get("entity") or "(unnamed)"
        lane = item.get("lane") or item.get("bucket") or ""
        table_rows.append((theme, lane, item.get("source", "")))
    _emit(
        payload,
        json_output=_as_json(json_output, output_format),
        table=_format_table(("theme", "lane", "source"), table_rows),
    )
    return 0


def _run_score(
    path: Path | None = None,
    *,
    input_path: Path | None = None,
    json_output: bool = False,
    output_format: str = "table",
    demo: bool = False,
    live: bool = False,
) -> int:
    selected = input_path or path
    if live:
        # Deliberately fail closed: this command's contract is local-only.
        print(
            "Live fetching is not enabled by the local-first CLI; "
            "provide --input/CSV data.",
            file=sys.stderr,
        )
        return 2
    if demo or selected is None:
        scored = synthetic_demo_rows()
        source = "bundled in-memory demo"
    else:
        try:
            raw_rows = _read_rows(selected)
            # Prefer the project's PIT-aware signal engine for its tidy
            # ``date,theme,security,value`` contract.  Generic long/wide
            # extracts use the local deterministic adapter below.
            scored = _core_scorecard(raw_rows)
            if not scored:
                scored = _score_records(_normalise_rows(raw_rows))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"Could not score {selected}: {exc}", file=sys.stderr)
            return 2
        source = str(selected)
        if not scored:
            print(f"Could not score {selected}: no numeric observations found", file=sys.stderr)
            return 2
    payload = {"source": source, "rows": scored, "disclaimer": DISCLAIMER}
    table_rows = [
        (
            row["entity"],
            row["score"],
            row["bucket"],
            f'{row["data_confidence"]:.1f}%',
            ", ".join(row["risk_flags"]),
            row.get("as_of") or "unknown",
        )
        for row in scored
    ]
    _emit(
        payload,
        json_output=_as_json(json_output, output_format),
        table=_format_table(
            ("entity", "score", "lane", "confidence", "risk flags", "as-of"),
            table_rows,
        ),
    )
    return 0


def _run_validate(
    path: Path | None = None,
    *,
    input_path: Path | None = None,
    json_output: bool = False,
    output_format: str = "table",
) -> int:
    selected = input_path or path
    if selected is None:
        selected = _first_catalog_file()
    if selected is None:
        report = {
            "valid": True,
            "errors": [],
            "warnings": ["no input supplied; no local catalog found"],
        }
        _emit(
            report,
            json_output=_as_json(json_output, output_format),
            table="VALID  no input supplied; no local catalog found",
        )
        return 0
    report = _validate_file(selected)
    table_rows = [("ERROR", message) for message in report.get("errors", [])]
    table_rows.extend(("WARN", message) for message in report.get("warnings", []))
    if not table_rows:
        table_rows = [("OK", f"{report.get('normalised_rows', 0)} normalised rows")]
    _emit(
        report,
        json_output=_as_json(json_output, output_format),
        table=_format_table(("status", "detail"), table_rows),
    )
    return 0 if report.get("valid") else 1


def _run_dashboard(
    data_path: Path | None = None,
    *,
    port: int = 8501,
    headless: bool = False,
    no_browser: bool = False,
) -> int:
    try:
        importlib.import_module("streamlit")
    except Exception:
        print(
            "Streamlit is not installed. Install the optional dashboard dependency and retry.",
            file=sys.stderr,
        )
        return 2
    dashboard_path = Path(__file__).resolve().parent / "app" / "dashboard.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port",
        str(port),
    ]
    if headless or no_browser:
        command.extend(["--server.headless", "true"])
    if data_path is not None:
        command.extend(["--", "--data", str(data_path)])
    return subprocess.call(command)


def _raise_typer_exit(code: int) -> None:
    if code and typer is not None:
        raise typer.Exit(code=code)


if typer is not None:
    app = typer.Typer(
        name="theme-leadership-os",
        help="Local-first theme leadership research console.",
        no_args_is_help=True,
        add_completion=False,
    )

    @app.command("doctor")
    def doctor(
        json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
        output_format: str = typer.Option(
            "table", "--format", help="Output format: table or json."
        ),
    ) -> None:
        """Check local dependencies and data paths without network access."""

        _raise_typer_exit(_run_doctor(json_output=json_output, output_format=output_format))

    @app.command("catalog")
    def catalog(
        path: Path | None = typer.Option(
            None,
            "--path",
            "--input",
            "-i",
            help="Local catalog CSV/JSON.",
        ),
        json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
        output_format: str = typer.Option(
            "table", "--format", help="Output format: table or json."
        ),
    ) -> None:
        """List local theme catalog entries (or the bundled demo catalog)."""

        _raise_typer_exit(_run_catalog(path, json_output=json_output, output_format=output_format))

    @app.command("score")
    def score(
        path: Path | None = typer.Argument(None, help="Local long/wide CSV or JSON file."),
        input_path: Path | None = typer.Option(
            None,
            "--input",
            "--csv",
            "-i",
            help="Local data file.",
        ),
        json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
        output_format: str = typer.Option(
            "table", "--format", help="Output format: table or json."
        ),
        demo: bool = typer.Option(
            False, "--demo", help="Use bundled in-memory synthetic data."
        ),
        live: bool = typer.Option(
            False, "--live", help="Reserved; live fetching remains disabled."
        ),
    ) -> None:
        """Score a local CSV/JSON; no live fetch occurs by default."""

        _raise_typer_exit(
            _run_score(
                path,
                input_path=input_path,
                json_output=json_output,
                output_format=output_format,
                demo=demo,
                live=live,
            )
        )

    @app.command("validate")
    def validate(
        path: Path | None = typer.Argument(None, help="Local long/wide CSV or JSON file."),
        input_path: Path | None = typer.Option(
            None,
            "--input",
            "--csv",
            "-i",
            help="Local data file.",
        ),
        json_output: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
        output_format: str = typer.Option(
            "table", "--format", help="Output format: table or json."
        ),
    ) -> None:
        """Validate local extract shape, dates, values, and duplicate keys."""

        _raise_typer_exit(
            _run_validate(
                path,
                input_path=input_path,
                json_output=json_output,
                output_format=output_format,
            )
        )

    @app.command("dashboard")
    def dashboard(
        data_path: Path | None = typer.Option(
            None,
            "--data",
            "--input",
            "-i",
            help="Optional local score CSV/JSON.",
        ),
        port: int = typer.Option(
            8501, "--port", min=1, max=65535, help="Streamlit server port."
        ),
        headless: bool = typer.Option(
            False, "--headless", help="Run Streamlit without opening a browser."
        ),
        no_browser: bool = typer.Option(False, "--no-browser", help="Alias for headless mode."),
    ) -> None:
        """Launch the Streamlit research dashboard using local/demo data."""

        _raise_typer_exit(
            _run_dashboard(data_path, port=port, headless=headless, no_browser=no_browser)
        )

else:

    class _FallbackApp:
        """Small callable facade used when Typer is not installed."""

        def __call__(self, args: Sequence[str] | None = None) -> int:
            return main(args)

        def __repr__(self) -> str:  # pragma: no cover - cosmetic
            return "<fallback theme-leadership-os app>"

    app = _FallbackApp()
    doctor = _run_doctor
    catalog = _run_catalog
    score = _run_score
    validate = _run_validate
    dashboard = _run_dashboard


def _fallback_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-first theme leadership research console.")
    subparsers = parser.add_subparsers(dest="command")
    doctor_parser = subparsers.add_parser("doctor", help="Check local dependencies and data paths.")
    doctor_parser.add_argument("--json", action="store_true", dest="json_output")
    doctor_parser.add_argument("--format", default="table", dest="output_format")
    catalog_parser = subparsers.add_parser("catalog", help="List local catalog entries.")
    catalog_parser.add_argument("--path", "--input", "-i", type=Path)
    catalog_parser.add_argument("--json", action="store_true", dest="json_output")
    catalog_parser.add_argument("--format", default="table", dest="output_format")
    for name, help_text in (
        ("score", "Score local long/wide CSV or JSON."),
        ("validate", "Validate a local extract."),
    ):
        command_parser = subparsers.add_parser(name, help=help_text)
        command_parser.add_argument("path", nargs="?", type=Path)
        command_parser.add_argument("--input", "--csv", "-i", type=Path, dest="input_path")
        command_parser.add_argument("--json", action="store_true", dest="json_output")
        command_parser.add_argument("--format", default="table", dest="output_format")
        if name == "score":
            command_parser.add_argument("--demo", action="store_true")
            command_parser.add_argument("--live", action="store_true")
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch the Streamlit dashboard.")
    dashboard_parser.add_argument("--data", "--input", "-i", type=Path, dest="data_path")
    dashboard_parser.add_argument("--port", type=int, default=8501)
    dashboard_parser.add_argument("--headless", action="store_true")
    dashboard_parser.add_argument("--no-browser", action="store_true", dest="no_browser")
    return parser


def main(args: Sequence[str] | None = None) -> int:
    """Fallback executable entry point; Typer handles normal packaged use."""

    if typer is not None:
        # ``typer`` handles argv and raises SystemExit as expected for a CLI.
        app()  # type: ignore[operator]
        return 0
    parser = _fallback_parser()
    namespace = parser.parse_args(args)
    if not namespace.command:
        parser.print_help()
        return 0
    values = vars(namespace)
    command = values.pop("command")
    if command == "doctor":
        return _run_doctor(**values)
    if command == "catalog":
        return _run_catalog(**values)
    if command == "score":
        return _run_score(**values)
    if command == "validate":
        return _run_validate(**values)
    if command == "dashboard":
        return _run_dashboard(**values)
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess smoke tests
    if typer is not None:
        app()
    else:
        raise SystemExit(main())
