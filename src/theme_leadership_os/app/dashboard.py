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
        return "높음"
    if confidence >= 0.55:
        return "중간"
    return "낮음"


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
        "테마": row.get("entity") or row.get("theme") or "알 수 없음",
        "점수": round(_number(row.get("score"), 0.0), 1),
        "자료 신뢰도": f'{_number(row.get("data_confidence"), 0.0):.1f}%',
        "기준일": row.get("as_of") or "알 수 없음",
        "자료 경과": (
            f'{row["freshness_days"]}일 전'
            if row.get("freshness_days") is not None
            else "알 수 없음"
        ),
        "위험 경고": ", ".join(str(flag) for flag in flags),
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
    columns[0].metric("신호 점수", f"{score:.1f}/100")
    columns[1].metric("자료 신뢰도", f"{confidence:.1f}%")
    columns[2].metric("기준일", str(row.get("as_of") or "알 수 없음"))
    freshness = (
        f'{row.get("freshness_days")}일'
        if row.get("freshness_days") is not None
        else "알 수 없음"
    )
    columns[3].metric(
        "자료 경과",
        freshness,
    )

    if flags and flags != ["none flagged"]:
        st.warning("위험 경고: " + ", ".join(str(flag) for flag in flags))
    else:
        st.success("위험 경고 없음")

    components = row.get("components") or {}
    weights = row.get("component_weights") or COMPONENT_WEIGHTS
    st.markdown("**구성요소 투명성**")
    component_rows = []
    for name, weight in weights.items():
        value = _number(components.get(name), 0.0) if isinstance(components, Mapping) else 0.0
        component_rows.append(
            {
                "구성요소": {
                    "momentum": "모멘텀",
                    "breadth": "breadth",
                    "quality": "품질",
                    "risk_control": "위험 통제",
                }.get(str(name), str(name)),
                "가중치": f"{_number(weight, 0.0) * 100:.0f}%",
                "값": f"{value:.1f}/100",
                "기여도": f"{value * _number(weight, 0.0):.1f}",
            }
        )
    st.dataframe(_as_dataframe(component_rows), use_container_width=True, hide_index=True)
    if row.get("observations") is not None:
        st.caption(
            f"근거: 관측치 {row['observations']}개. "
            "신뢰도는 자료 품질이며 기대수익률 확률이 아닙니다."
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
    st.set_page_config(
        page_title="Theme Leadership OS · 테마 주도권 리서치",
        page_icon="🧭",
        layout="wide",
    )
    st.title("Theme Leadership OS")
    st.caption(
        "로컬 우선 리서치 콘솔 · 투명한 점수표 · 실시간 자동 수집 없음"
    )
    # Keep the safety language visually prominent on every render.
    st.error("연구·관찰용 결과이며 투자 권고가 아닙니다")
    st.info(
        f"자료 출처: {source}. "
        "점수는 연구 가설이며 개인화된 조언이 아닙니다."
    )

    if rows:
        as_of_values = [str(row.get("as_of")) for row in rows if row.get("as_of")]
        freshest = max(as_of_values) if as_of_values else "unknown"
        st.markdown(
            f"**자료 경과 / 기준일:** 최신 행 {freshest} · "
            "각 후보의 자료 경과와 경고를 표시합니다."
        )
    st.subheader("점수 읽는 법")
    st.markdown(
        "합성 점수는 구성요소를 공개합니다: **모멘텀 40% · breadth 25% · "
        "품질 20% · 위험 통제 15%**. 자료 신뢰도는 커버리지와 최신성을 "
        "뜻하며 "
        "성과 확률이 아닙니다."
    )

    display_labels = ("현재 주도", "초입 레이더", "12개월 검토")
    tab_objects = st.tabs(list(display_labels))
    for tab, label, display_label in zip(tab_objects, TAB_LABELS, display_labels):
        with tab:
            lane_rows = [row for row in rows if row.get("bucket") == label]
            st.header({
                "현재 주도": "현재 주도 테마",
                "초입 레이더": "초입 상승 레이더",
                "12개월 검토": "12개월 검토 후보",
            }[display_label])
            if not lane_rows:
                st.caption("현재 로컬 추출본에는 이 구간의 후보가 없습니다.")
                continue
            st.dataframe(
                _as_dataframe([_flat_row(row) for row in lane_rows]),
                use_container_width=True,
                hide_index=True,
            )
            for row in lane_rows:
                with st.expander(
                    f"{row.get('entity', '알 수 없음')} · 근거 펼쳐보기", expanded=True
                ):
                    _render_candidate(st, row)
    st.caption("연구·관찰용 결과이며 투자 권고가 아닙니다.")
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
