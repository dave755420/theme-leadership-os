"""Command-line entry point for the frozen early-radar validation pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .validation.early_validation import validate_early_radar
from .validation.protocol import load_protocol


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "records" in payload:
            payload = payload["records"]
        if not isinstance(payload, list):
            raise ValueError("JSON input must be a list or {'records': [...]} mapping")
        return pd.DataFrame(payload)
    raise ValueError(f"unsupported input format: {path.suffix}")


def _read_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("data manifest must be a JSON object")
    return dict(payload)


def _default_output(as_of: str | None) -> Path:
    stamp = (as_of or pd.Timestamp.utcnow().date().isoformat()).replace(":", "-")
    return Path("artifacts") / "validation" / f"early_radar_{stamp}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="theme-leadership-validate-early",
        description=(
            "Compare the frozen early-rise radar with M0 and write an auditable validation pack."
        ),
    )
    parser.add_argument("--input", required=True, help="Tidy price CSV/JSON")
    parser.add_argument("--episodes", help="Optional formal episode CSV/JSON: theme,onset,peak")
    parser.add_argument(
        "--data-manifest",
        help="JSON evidence manifest for PIT/adjustment/delisting/economic gates",
    )
    parser.add_argument("--protocol", help="Override config/episodes.yml path")
    parser.add_argument("--as-of", help="Optional cutoff date")
    parser.add_argument("--output", help="Validation-pack JSON path")
    parser.add_argument(
        "--include-rows",
        action="store_true",
        help="Include row-level signal/weekly/episode tables in the JSON artifact",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prices = _read_frame(Path(args.input))
    episodes = None if not args.episodes else _read_frame(Path(args.episodes))
    manifest = _read_manifest(None if not args.data_manifest else Path(args.data_manifest))
    protocol = load_protocol(args.protocol)
    result = validate_early_radar(
        prices,
        episodes=episodes,
        data_manifest=manifest,
        protocol=protocol,
        as_of=args.as_of,
    )
    output = Path(args.output) if args.output else _default_output(args.as_of)
    result.write_json(output, include_rows=args.include_rows)
    summary = result.summary
    print(f"status: {result.status}")
    print(f"artifact: {output}")
    print(f"M0 26w IC: {summary['m0_26w']['mean_rank_ic']}")
    print(f"Early 26w IC: {summary['early_score_26w']['mean_rank_ic']}")
    print(f"IC delta vs M0: {summary['comparison_vs_m0']['rank_ic_delta']}")
    print(f"Early discovery ready: {summary['early_discovery_gate']['ready']}")
    return 0 if result.status in {"PASS", "NOT_READY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
