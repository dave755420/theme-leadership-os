# Next actions

Priorities below close the v0.1 core release gate. No analyst module or Wall
Street Insight Daily work should begin until the final core item is complete.

## P0 — make the core validation-ready

1. Freeze the source registry, license evidence, checksums, availability
   semantics, attribution requirements, and run-manifest convention.
2. Replace the exploratory input with approved free/public data that supports
   adjusted total-return handling, point-in-time membership, delisting coverage,
   and reproducible availability timestamps.
3. Keep the existing M0 contract fixed as the 26-week SPY-relative-strength
   baseline; pre-register evaluation windows, metrics, and missingness rules
   before inspecting the replacement run.
4. Exercise the existing PIT guards and integrity audits on delayed publication,
   restatements, conflicts, stale/missing data, duplicates, coverage gaps, ETF
   inception, and timezone boundaries.

## P1 — run and review the core

5. Run the three signal engines on identical frozen windows and retain the full
   output panels, including rejected statuses, warnings, M0 fields, and
   next-session labels.
6. Execute the pre-registered positive challenge cases (`solar_2013`,
   `solar_2020`, `quantum_2024`, `evtol_2023`, `evtol_2024`) and negative controls
   without tuning thresholds on their outcomes.
7. Use purged walk-forward splits and rank-IC/top-bottom/hit metrics to produce
   a validation pack with explicit `pass`, `failed`, or `not_ready` states.
8. Independently review PIT behavior, source terms, raw-data handling, M0
   comparison, challenge/control outcomes, and limitations.
9. Record why the exploratory baseline remains `exploratory_rejected`; do not
   promote its unadjusted or incomplete-coverage results into a product claim.

## P2 — unlock analyst tooling only after the gate

10. Mark the core `passing` only after every release gate is green and signed
    off.
11. Draft analyst-module requirements as consumers of the three product-lane
    contracts and supporting artifacts; preserve provenance, PIT eligibility,
    validation status, and research-only disclaimers in every view.
12. Revisit the analyst module and **Wall Street Insight Daily** only after that
    sign-off. The current CLI/dashboard presentation layer does not constitute
    approval.

## Handoff rule

If any P0/P1 action is blocked by missing source terms, unavailable timestamps,
partial/delisted coverage, leakage, or an unexplained negative-control result,
record `not_ready` and resolve the blocker before expanding scope.
