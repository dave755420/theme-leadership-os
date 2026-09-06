"""Frozen validation-protocol loader and drift checks.

The repository keeps human-reviewable thresholds in ``config/episodes.yml``.
This module makes that file the validation source of truth while retaining a
small stdlib-only parser fallback so core validation does not require PyYAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

PROTOCOL_PATH = Path(__file__).resolve().parents[3] / "config" / "episodes.yml"


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return {}
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [] if not inner else [_parse_scalar(item) for item in inner.split(",")]
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    try:
        if any(token in text for token in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text.strip('"\'')


def _fallback_load(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        stripped = raw_line.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        line = stripped.strip()
        if ":" not in line:
            raise ValueError(f"unsupported protocol line: {raw_line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        if indent == 0:
            parsed = _parse_scalar(value)
            if isinstance(parsed, dict):
                current = {}
                root[key] = current
            else:
                root[key] = parsed
                current = None
        elif indent == 2 and current is not None:
            current[key] = _parse_scalar(value)
        else:
            raise ValueError("fallback protocol parser supports one nested mapping level")
    return root


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    """Load the frozen protocol mapping from YAML with a stdlib fallback."""

    source = PROTOCOL_PATH if path is None else Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        data = _fallback_load(text)
    else:
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError("protocol root must be a mapping")
        data = loaded
    return dict(data)


def early_signal_config_from_protocol(
    protocol: Mapping[str, Any] | None = None,
):
    """Build ``AnnualLeadershipConfig`` from the frozen early-rise section."""

    from .annual import AnnualLeadershipConfig

    mapping = dict((protocol or load_protocol()).get("early_rise_signal", {}))
    return AnnualLeadershipConfig(
        signal_history_weeks=int(mapping.get("history_weeks", 26)),
        early_min_rank=float(mapping.get("min_4w_rank", 0.70)),
        early_min_acceleration_rank=float(mapping.get("min_acceleration_rank", 0.70)),
        early_min_rank_improvement=float(mapping.get("min_rank_improvement_4v13", 0.10)),
        early_min_breadth=float(mapping.get("min_breadth_4w", 0.50)),
        early_min_participation=float(mapping.get("min_participation", 0.50)),
        early_min_fast_rs=float(mapping.get("min_4w_spy_excess", 0.00)),
        early_min_m0=float(mapping.get("min_26w_spy_excess", -0.30)),
        confirmation_count=int(mapping.get("confirmation_count", 2)),
        confirmation_window_weeks=int(mapping.get("confirmation_window_weeks", 3)),
    )


def protocol_alignment_mismatches(
    protocol: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return any drift between Python defaults and frozen YAML thresholds."""

    from .annual import AnnualLeadershipConfig

    cfg = AnnualLeadershipConfig()
    frozen = early_signal_config_from_protocol(protocol)
    fields = (
        "signal_history_weeks",
        "early_min_rank",
        "early_min_acceleration_rank",
        "early_min_rank_improvement",
        "early_min_breadth",
        "early_min_participation",
        "early_min_fast_rs",
        "early_min_m0",
        "confirmation_count",
        "confirmation_window_weeks",
    )
    return [
        f"{name}: python={getattr(cfg, name)!r}, protocol={getattr(frozen, name)!r}"
        for name in fields
        if getattr(cfg, name) != getattr(frozen, name)
    ]


__all__ = [
    "PROTOCOL_PATH",
    "early_signal_config_from_protocol",
    "load_protocol",
    "protocol_alignment_mismatches",
]
