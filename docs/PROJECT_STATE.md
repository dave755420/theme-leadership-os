# Project state

As of 2026-08-30, the repository is at the v0.1 implementation stage. The
local core is usable for research fixtures, but its release gate is not yet
passing.

## Complete

- Python package metadata targeting Python 3.11+.
- Importable `theme_leadership_os` package exposing the three signal engines and
  their typed configurations.
- Apache-2.0 software license and third-party notice.
- CI checks for linting, compilation, importability, and the test suite.
- Domain models for normalized observations, PIT memberships, theme definitions,
  temporal comparisons, coverage status, and explicit look-ahead errors.
- Provider-neutral data contracts, SHA-256 source manifests, memory/file caches,
  deterministic CSV fixture providers, and an explicitly gated yfinance
  personal-research adapter.
- Versioned catalog loader and seed catalog with solar, quantum-computing, eVTOL,
  and broad-market/traditional-oil/consumer-staples negative controls.
- Three deterministic signal engines for `Current Leader`, `Emerging Radar`, and
  `12M Hold Candidate`, including M0, weekly windows, quality warnings, and
  next-session labels.
- 연도별 테마 수익률·SPY 초과수익률 순위표와 미래 데이터를 사용하지 않는
  `초입 관찰`/`초입 확인` 레이더(`annual_theme_leaders`,
  `early_rise_signals`).
- Validation primitives for integrity audits, rank/spread/hit metrics, purged
  walk-forward splits, episode/early-hit labels, and holdout challenge metadata.
- Local-first CLI (`doctor`, `catalog`, `score`, `validate`, `annual`, `dashboard`) and an
  optional Streamlit presentation dashboard.
- Architecture, data policy, validation protocol, roadmap, and release-gate
  documentation.

## Current limitations and release blockers

- The exploratory baseline is explicitly `exploratory_rejected`: it used
  unadjusted prices and incomplete point-in-time membership/delisting coverage.
  M0 exceeded the rejected composite in that exploratory window, but neither
  result is a validated performance claim.
- Seed catalog coverage is partial for the theme series, and missing/delisted
  history is surfaced rather than fabricated.
- The yfinance adapter is disabled by default, personal-research-only,
  non-redistributable, and not PIT-complete.
- The CLI never fetches live data automatically; Streamlit and Typer are
  optional presentation dependencies.
- A full frozen PIT-complete validation run, challenge/control review, and core
  sign-off remain outstanding.
- The analyst module and **Wall Street Insight Daily** remain blocked until the
  core validation passes.

## Non-negotiable invariants

1. `Current Leader`, `Emerging Radar`, and `12M Hold Candidate` remain separate
   product outputs; theme definition, signal ledger, and validation pack remain
   separate supporting artifacts.
2. PIT eligibility is based on availability time, not event time alone.
3. M0 is frozen and transparent before comparisons are interpreted.
4. Target challenge cases and negative controls are release gates.
5. Analyst tooling stays disabled until the core passes.
6. Raw market data is not redistributed.

## Next decisions for the release stage

- Approve a free/public source registry with license evidence and a defensible
  per-source availability method.
- Replace exploratory unadjusted/partial coverage with adjusted total-return,
  point-in-time membership, and delisting-complete inputs where permitted.
- Freeze the M0 evaluation windows, metrics, and run-manifest convention before
  reviewing results.
- Execute and independently review the pre-registered challenge cases and
  negative controls without tuning on their outcomes.
- Define the reviewer sign-off that changes core status from `not_ready` to
  `passing`.

These decisions should be recorded in source registries and validation manifests,
not inferred from a dashboard narrative.
