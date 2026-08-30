# Changelog

All notable changes to this project will be documented here.

## [0.1.0] - 2026-08-30

### Added

- Importable Python 3.11+ package with Apache-2.0 licensing and permissive
  NumPy/pandas runtime dependencies.
- Typed PIT domain objects: normalized observations, theme definitions,
  membership intervals, temporal comparisons, coverage statuses, and explicit
  look-ahead guards.
- Provider-neutral data contracts with SHA-256 manifests, memory/file caches,
  deterministic CSV fixture providers, and an opt-in personal-research yfinance
  adapter that does not expose raw provider payloads.
- Versioned seed catalog for solar, quantum-computing, and eVTOL themes plus
  broad-market, traditional-oil, and consumer-staples negative controls.
- Transparent weekly signal engines for `Current Leader`, `Emerging Radar`, and
  `12M Hold Candidate`, including M0, component/gate columns, warnings, and
  next-session labels.
- Validation primitives for integrity audits, rank-IC/spread/hit metrics,
  purged walk-forward splits, episode/early-hit labels, and holdout challenge /
  negative-control metadata.
- Local-first CLI commands (`doctor`, `catalog`, `score`, `validate`) and an
  optional Streamlit dashboard with research-only disclaimers.
- Documentation for architecture, data policy, validation protocol, roadmap,
  project state, and release-gated next actions.

### Validation status

- The checked-in exploratory baseline is deliberately `exploratory_rejected`:
  M0 exceeded the rejected composite in the observed window, but prices were
  unadjusted and historical membership/delisting coverage was incomplete.
- The analyst module and **Wall Street Insight Daily** remain blocked until a
  PIT-complete core validation run passes all challenge, negative-control, and
  review gates.
