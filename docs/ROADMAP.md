# Roadmap

The roadmap is gated by evidence quality. v0.1 has an implemented local core,
but the exploratory baseline is rejected and the core release gate is still
open. Dates are intentionally omitted until the PIT-complete source plan is
approved.

## M0 — foundation (complete)

- Establish Apache-2.0 software licensing and permissive development tooling.
- Document the three-output boundary (`Current Leader`, `Emerging Radar`, and
  `12M Hold Candidate`), PIT discipline, free-data policy, and validation
  protocol.
- Add an importable package, CI, offline fixtures, and an explicit research
  disclaimer.

Exit: repository foundation is reviewable and the current test suite is green.

## M1 — core contracts and source registry (implemented; hardening pending)

- `Observation`, `ThemeMembership`, `ThemeDefinition`, and temporal helpers
  enforce explicit availability and validity timing.
- `SourceManifest` records source IDs, URIs, checksums, coverage, completeness,
  and redistribution flags; memory/file caches preserve the manifest boundary.
- `load_catalog()` reads the versioned themes, memberships, and source manifests
  under `data/catalog/`, including negative-control themes and partial/delisted
  coverage states.

Exit: implementation and boundary tests exist. Release still requires a reviewed
source registry with PIT-complete coverage and permitted derived use.

## M2 — evidence and signal core (implemented in v0.1)

- The provider-neutral interface and deterministic CSV provider accept normalized
  local observations and PIT-safe `as_of` filtering.
- `current_leader()` applies 13/26/52-week relative-strength, breadth,
  participation, M0, and transparent score/rank gates.
- `emerging_radar()` applies 4/8/13-week acceleration/rank-improvement logic
  with visible quality warnings and watchlist-only language.
- `hold_candidate_12m()` applies 13/26/52-week durability, confirmation,
  concentration, volatility, and drawdown gates.
- `annual_theme_leaders()` ranks each calendar year against SPY, while
  `early_rise_signals()` emits a no-lookahead `초입 관찰`/`초입 확인` status from
  4·8·13-week acceleration, breadth, participation, M0, and 2/3-week persistence.
- All engines use Friday-ending weekly decisions, retain rejected rows and gate
  reasons, and label the next effective session without using it in the signal.

Exit: deterministic signal panels and product lanes are available. The optional
yfinance adapter remains personal-research-only and is not PIT-complete.

## M3 — M0 and validation primitives (implemented; release pending)

- M0 is the explicit 26-week SPY-relative-strength baseline in the signal core.
- Integrity audits cover feature availability, membership dates, ETF inception,
  coverage, and duplicate observations.
- Metrics cover cross-sectional rank IC, top/bottom spreads, and hit rates;
  purged walk-forward splitters protect label maturity and embargo boundaries.
- Episode/early-hit labels and pre-registered challenge/negative-control metadata
  are available without tuning thresholds from the cases.
- The checked-in exploratory baseline is marked `exploratory_rejected`: M0 mean
  rank IC is 0.1141 versus 0.0879, and mean top-minus-bottom spread is 2.00%
  versus 0.70%, but the data is not PIT-complete and prices are unadjusted.

Exit: a clean, frozen, PIT-complete run must produce a validation pack with all
challenge/control outcomes before the core can be marked passing.

## M4 — core release gate (next)

- Replace the exploratory input with approved free/public data that has adjusted
  total-return handling, point-in-time membership, and delisting coverage.
- Run provenance, PIT/leakage, deterministic rerun, challenge-case, and negative
  control gates on identical frozen windows.
- Publish limitations and `pass`/`not_ready` status with independent review.
- Treat unexplained leakage, a failed negative control, or missing source terms as
  release-blocking.

Exit: the core is explicitly marked `passing` by the agreed reviewer(s).

## M5 — analyst module / Wall Street Insight Daily (blocked)

- Design analyst-facing exploration, ranking, alerts, and the planned Wall Street
  Insight Daily only as consumers of the three validated product lanes.
- Preserve provenance, PIT eligibility, validation status, and research-only
  disclaimers in every view.

The analyst module and Wall Street Insight Daily must not start or make claims
before M4 passes. The current CLI/dashboard presentation layer does not satisfy
this gate and must not be treated as analyst approval.

## Out of scope until separately approved

- Paid or restricted market-data feeds.
- Redistribution of raw market data.
- Unversioned backfills or retrospective timestamp repair.
- Claims of investment performance or recommendations.
- Analyst automation before the core release gate.
