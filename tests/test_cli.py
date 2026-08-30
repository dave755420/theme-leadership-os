"""Offline smoke tests for the local-first CLI surface."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_doctor_is_local_first(capsys: pytest.CaptureFixture[str]) -> None:
    from theme_leadership_os.cli import _doctor_report, _run_doctor

    report = _doctor_report()
    assert report["ok"] is True
    assert any(
        check["name"] == "network" and check["status"] == "disabled"
        for check in report["checks"]
    )
    assert _run_doctor() == 0
    assert "network" in capsys.readouterr().out


def test_catalog_has_offline_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    from theme_leadership_os.cli import _run_catalog

    assert _run_catalog() == 0
    output = capsys.readouterr().out
    assert "Solar energy" in output or "AI Infrastructure" in output
    assert "source" in output


def test_help_lists_commands() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "theme_leadership_os.cli", "--help"],
        cwd=ROOT,
        env=_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0
    help_text = completed.stdout + completed.stderr
    for command in ("doctor", "catalog", "score", "validate", "dashboard"):
        assert command in help_text


def test_score_accepts_wide_csv_without_network() -> None:
    from theme_leadership_os.cli import _normalise_rows, _read_rows, _score_records

    rows = _read_rows(ROOT / "examples" / "sample_prices_wide.csv")
    scored = _score_records(_normalise_rows(rows))
    assert {row["entity"] for row in scored} == {
        "AI Infrastructure",
        "Cybersecurity",
        "Grid Modernization",
    }
    assert all("component_weights" in row and "risk_flags" in row for row in scored)
