"""A small, transparent Streamlit dashboard for research candidates.

The dashboard is deliberately presentation-only.  It consumes score rows from
the local CLI adapter (or a compatible research engine) and keeps a bundled
synthetic scorecard in memory for demos and smoke tests.  No network call is
made by this module.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from ..cli import (
        COMPONENT_WEIGHTS,
        DISCLAIMER,
        TAB_LABELS,
        _core_scorecard,
        _normalise_rows,
        _read_rows,
        _score_records,
        synthetic_demo_rows,
    )
except Exception:  # pragma: no cover - supports direct execution from a source checkout
    from theme_leadership_os.cli import (  # type: ignore
        COMPONENT_WEIGHTS,
        DISCLAIMER,
        TAB_LABELS,
        _core_scorecard,
        _normalise_rows,
        _read_rows,
        _score_records,
        synthetic_demo_rows,
    )


def build_demo_scorecard() -> list[dict[str, Any]]:
    """Return deterministic in-memory rows for the first-run dashboard."""

    return synthetic_demo_rows()


def load_scorecard(path: Path | None = None) -> tuple[list[dict[str, Any]], str]:
    """Load a local scorecard or return the bundled demo.

    A scorecard may be a pre-scored JSON list (rows containing ``score`` and
    ``bucket``) or a long/wide CSV/JSON observation extract.  The latter is
    normalised and scored by the local deterministic adapter.
    """

    if path is None:
        return build_demo_scorecard(), "bundled in-memory demo"
    try:
        raw = _read_rows(path)
        if raw and any("score" in {str(key).strip().lower() for key in row} for row in raw):
            # Preserve an engine-produced scorecard while filling presentation
            # fields that older exports may not have.
            rows: list[dict[str, Any]] = []
            for row in raw:
                item = dict(row)
                item.setdefault(
                    "entity",
                    item.get("theme") or item.get("symbol") or item.get("ticker") or "Unknown",
                )
                item["score"] = _number(item.get("score"), 0.0)
                item.setdefault("signal_score", item["score"])
                item.setdefault("bucket", _bucket_for_score(item["score"]))
                item.setdefault(
                    "engine",
                    {
                        TAB_LABELS[0]: "current_leader",
                        TAB_LABELS[1]: "emerging_radar",
                        TAB_LABELS[2]: "hold_candidate_12m",
                    }.get(item["bucket"], "local_scorecard"),
                )
                item.setdefault("components", _default_components(item["score"]))
                item.setdefault("component_weights", COMPONENT_WEIGHTS.copy())
                item.setdefault("data_confidence", _number(item.get("confidence"), 0.5) * 100.0)
                item.setdefault("confidence", _number(item.get("data_confidence"), 50.0) / 100.0)
                item.setdefault(
                    "confidence_label", _confidence_label(_number(item["confidence"], 0.5))
                )
                item.setdefault("risk_flags", ["none flagged"])
                item.setdefault("as_of", item.get("date") or item.get("asof"))
                item.setdefault("freshness_days", None)
                item.setdefault("disclaimer", DISCLAIMER)
                rows.append(item)
            return rows, str(path)
        # Tidy signal exports go through the three PIT-aware engines first;
        # generic long/wide extracts retain the local deterministic fallback.
        scored = _core_scorecard(raw)
        if not scored:
            scored = _score_records(_normalise_rows(raw))
        return (scored or build_demo_scorecard()), str(path)
    except (OSError, ValueError, TypeError):
        # The visual demo should still open when a user points at an incomplete
        # file.  The source caption makes the fallback explicit.
        return build_demo_scorecard(), f"bundled demo (could not read {path})"


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bucket_for_score(score: float) -> str:
    if score >= 70:
        return TAB_LABELS[0]
    if score >= 50:
        return TAB_LABELS[1]
    return TAB_LABELS[2]


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.80:
        return "High"
    if confidence >= 0.55:
        return "Medium"
    return "Low"


def _default_components(score: float) -> dict[str, float]:
    return {key: round(score, 2) for key in COMPONENT_WEIGHTS}


def _as_dataframe(rows: Sequence[Mapping[str, Any]]) -> Any:
    """Return a pandas DataFrame when available, otherwise the original rows."""

    try:
        import pandas as pd  # type: ignore

        return pd.DataFrame(list(rows))
    except Exception:
        return list(rows)


def _flat_row(row: Mapping[str, Any]) -> dict[str, Any]:
    flags = row.get("risk_flags") or []
    if isinstance(flags, str):
        flags = [flags]
    return {
        "Theme": row.get("entity") or row.get("theme") or "Unknown",
        "Score": round(_number(row.get("score"), 0.0), 1),
        "Data confidence": f'{_number(row.get("data_confidence"), 0.0):.1f}%',
        "As-of": row.get("as_of") or "unknown",
        "Freshness": (
            f'{row["freshness_days"]}d old' if row.get("freshness_days") is not None else "unknown"
        ),
        "Risk flags": ", ".join(str(flag) for flag in flags),
    }


def _render_candidate(st: Any, row: Mapping[str, Any]) -> None:
    """Render one candidate with its evidence and score components."""

    entity = row.get("entity") or row.get("theme") or "Unknown"
    score = _number(row.get("score"), 0.0)
    confidence = _number(row.get("data_confidence"), 0.0)
    flags = row.get("risk_flags") or []
    if isinstance(flags, str):
        flags = [flags]
    st.subheader(str(entity))
    columns = st.columns(4)
    columns[0].metric("Signal score", f"{score:.1f}/100")
    columns[1].metric("Data confidence", f"{confidence:.1f}%")
    columns[2].metric("As-of", str(row.get("as_of") or "unknown"))
    columns[3].metric(
        "Freshness",
        f'{row.get("freshness_days")}d' if row.get("freshness_days") is not None else "unknown",
    )

    if flags and flags != ["none flagged"]:
        st.warning("Risk flags: " + ", ".join(str(flag) for flag in flags))
    else:
        st.success("Risk flags: none flagged")

    components = row.get("components") or {}
    weights = row.get("component_weights") or COMPONENT_WEIGHTS
    st.markdown("**Component transparency**")
    component_rows = []
    for name, weight in weights.items():
        value = _number(components.get(name), 0.0) if isinstance(components, Mapping) else 0.0
        component_rows.append(
            {
                "Component": str(name).replace("_", " ").title(),
                "Weight": f"{_number(weight, 0.0) * 100:.0f}%",
                "Value": f"{value:.1f}/100",
                "Contribution": f"{value * _number(weight, 0.0):.1f}",
            }
        )
    st.dataframe(_as_dataframe(component_rows), use_container_width=True, hide_index=True)
    if row.get("observations") is not None:
        st.caption(
            f"Evidence: {row['observations']} observations. "
            "Confidence is data quality, not expected return."
        )


def render_dashboard(
    data_path: Path | None = None,
    *,
    streamlit_module: Any = None,
) -> int:
    """Render the dashboard; return a non-zero status when Streamlit is absent.

    ``streamlit_module`` is injectable for tests and keeps importing this file
    safe in environments where Streamlit is an optional dependency.
    """

    st = streamlit_module
    if st is None:
        try:
            import streamlit as st  # type: ignore
        except Exception:
            return 2
    rows, source = load_scorecard(data_path)
    st.set_page_config(page_title="Theme Leadership OS", page_icon="🧭", layout="wide")
    st.title("Theme Leadership OS")
    st.caption("Local-first research console · transparent scorecard · no automatic live fetch")
    # Keep the safety language visually prominent on every render.
    st.error("RESEARCH CANDIDATE, NOT AN INVESTMENT RECOMMENDATION")
    st.info(f"Data source: {source}. Scores are hypotheses for research, not personalized advice.")

    if rows:
        as_of_values = [str(row.get("as_of")) for row in rows if row.get("as_of")]
        freshest = max(as_of_values) if as_of_values else "unknown"
        st.markdown(
            f"**Freshness / as-of:** latest row {freshest} · "
            "each candidate shows its own age and flags."
        )
    st.subheader("How to read the score")
    st.markdown(
        "The composite is intentionally inspectable: **Momentum 40% · Breadth 25% · "
        "Quality 20% · Risk control 15%**. Data confidence describes coverage and recency; "
        "it is not a probability of performance."
    )

    tab_objects = st.tabs(list(TAB_LABELS))
    for tab, label in zip(tab_objects, TAB_LABELS):
        with tab:
            lane_rows = [row for row in rows if row.get("bucket") == label]
            st.header(label)
            if not lane_rows:
                st.caption("No candidates in this lane for the current local extract.")
                continue
            st.dataframe(
                _as_dataframe([_flat_row(row) for row in lane_rows]),
                use_container_width=True,
                hide_index=True,
            )
            for row in lane_rows:
                with st.expander(
                    f"{row.get('entity', 'Unknown')} · inspect evidence", expanded=True
                ):
                    _render_candidate(st, row)
    st.caption(DISCLAIMER)
    return 0


def _data_path_from_argv(argv: Sequence[str] | None = None) -> Path | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", "--input", dest="data_path")
    parsed, _ = parser.parse_known_args(list(argv) if argv is not None else sys.argv[1:])
    return Path(parsed.data_path) if parsed.data_path else None


def main() -> int:
    return render_dashboard(_data_path_from_argv())


if __name__ == "__main__":  # pragma: no cover - Streamlit executes this file
    raise SystemExit(main())
