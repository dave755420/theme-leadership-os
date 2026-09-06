# Early-Radar Upgrade: Repair Plan and Evolution Path

Status: implementation upgrade in progress; **not investment-use certified**.

This document separates two kinds of work that should not be mixed:

1. **Repair / verification work**: make the current early-rise radar falsifiable, reproducible, and directly comparable with M0.
2. **Evolution work**: only after the repaired baseline is frozen, add richer data and models that must beat the same OOS benchmark without tuning on showcase episodes.

## 1. Why this upgrade exists

The 2026-08-30 early-rise radar was implemented after the saved exploratory baseline artifact. Therefore the old M0/composite metrics and the stored solar/quantum/eVTOL sentinel dates do not prove that the new `early_rise_signals` function works in real historical markets.

The upgraded validation path treats that as a hard distinction:

- old exploratory numbers remain exploratory;
- software tests prove determinism/no-lookahead behavior, not market skill;
- the new radar must be evaluated against M0 on the same dates, same universe, same future labels;
- a result cannot become `PASS` unless the data itself satisfies PIT, adjusted-return, delisting, coverage, and economic-test requirements.

## 2. Immediate repair direction

### 2.1 Freeze one source of truth for thresholds

`config/episodes.yml` now contains the frozen early-rise rule, continuous-score weights, OOS window, cross-sectional minimum, statistical acceptance gates, and early-discovery gates.

`validation.protocol` loads those settings and fails when Python signal defaults drift from the frozen rule. This prevents a notebook, CLI, and production code path from silently evaluating different thresholds.

### 2.2 Add an auditable continuous early score

The binary watch/confirm rule is useful operationally but is a poor research comparison surface. The validator therefore derives a continuous score from only contemporaneously observable components:

- acceleration percentile rank: 30%
- 4-week SPY-relative-strength percentile rank: 25%
- 4w-vs-13w rank improvement: 15%
- 4-week breadth: 15%
- participation: 15%

This score is **not a new buy rule**. It exists so the early-radar information content can be compared fairly with M0 using rank IC and top-bucket future excess return.

### 2.3 Compare M0 and the new radar apples-to-apples

For each eligible weekly cross-section the validator measures:

- M0 26-week rank IC vs future 26-week excess return
- Early Score 26-week rank IC vs future 26-week excess return
- Early Score 52-week rank IC
- top-20% future excess return
- Early Score minus M0 IC delta
- Early Score minus M0 top-bucket delta
- 52-week block-bootstrap lower bound for the 26-week IC sequence

A week is excluded from certification metrics when fewer than the frozen minimum number of themes have a valid score/outcome pair.

### 2.4 Make data-integrity failures impossible to hide

Certification requires an explicit data manifest. The following flags must all be true:

- `total_return_adjusted`
- `point_in_time_membership_complete`
- `delisting_complete`
- `source_registry_frozen`
- `availability_semantics_frozen`

The price panel must also expose one of:

- `feature_available_at`
- `evidence_available_at`
- `available_at`

A timestamp after the observation session aborts validation instead of producing a tainted metric. Research can still be run without the manifest, but the release result remains `NOT_READY`.

### 2.5 Separate research coverage from certification coverage

The display-oriented annual leader table may continue using a lower research coverage threshold. Certification uses a separate 95% minimum so a visually interesting but sparse historical panel cannot pass the release gate.

### 2.6 Measure true early-discovery quality

A formal episode file must provide at least `theme`, `onset`, and `peak`. The validator then measures confirmed early signals against those episodes using the frozen onset window and maximum realized-move rule.

Required early-discovery gate:

- at least 30 independent matured episodes
- at least 10 themes
- episode recall >= 70%
- false-discovery rate <= 35%
- median first-signal lag <= onset + 2 weeks
- median remaining relative move >= 60%

Solar, quantum, eVTOL and negative controls remain evaluation cases, not tuning targets.

### 2.7 Add a machine-readable validation pack

`theme-leadership-validate-early` writes a JSON artifact containing:

- protocol hash
- input fingerprint
- OOS window
- data-integrity result
- cross-sectional eligibility
- M0 metrics
- Early Score metrics
- delta vs M0
- episode metrics
- economic-gate state
- final `PASS` / `FAILED` / `NOT_READY`

The artifact can optionally include row-level signal, weekly metric, and episode tables for audit.

## 3. Release-state semantics

### NOT_READY

Used when market skill cannot yet be judged because one or more required evidence blocks are missing. Typical causes:

- no PIT-complete membership history
- no adjusted total-return history
- missing delisted securities
- missing availability timestamps
- <95% certification coverage
- insufficient matured OOS observations
- insufficient formal episodes
- economic/cost backtest not supplied

### FAILED

Used only when all required evidence blocks are present but one or more frozen gates fail.

### PASS

Reserved for a fully measurable run where all data, statistical, early-discovery, and economic gates pass. `PASS` is not a prediction that the next trade will make money; it means the frozen research claim survived the predefined validation protocol.

## 4. Evolution direction after the repaired baseline is frozen

The system should evolve as a **model ladder**, never by replacing the benchmark after seeing outcomes.

### Stage E0 — Data foundation

Goal: eliminate the largest research-risk sources before adding signal complexity.

- adjusted total-return weekly series
- explicit `valid_from` / `valid_to` theme membership
- public `evidence_available_at` for membership changes
- retained delisted/bankrupt names
- stable entity identifiers across ticker changes
- source checksum and provenance
- deterministic snapshot manifests

Exit condition: the integrity gate is measurable and clean on 2019-2025 OOS.

### Stage E1 — M1 acceleration model

Keep M0 frozen, then evaluate a transparent acceleration model using only price/volume-derived theme features available at decision time.

Candidate features:

- 4/8/13-week relative-strength slope
- percentile-rank velocity
- acceleration persistence
- drawdown recovery speed
- distance from 26-week relative high
- volatility-adjusted relative momentum

Rule: feature additions are accepted only when OOS improvement survives year/theme removal, 1-week execution delay, and doubled costs.

### Stage E2 — M2 breadth/participation/concentration model

Improve distinction between a true theme-wide move and one-stock concentration.

Candidate features:

- fraction of members beating SPY at 4/8/13 weeks
- fraction above 13/26-week trend
- median member relative-strength acceleration
- top-1/top-3 contribution concentration
- dispersion of member returns
- breadth impulse and breadth persistence
- number of independently tradable active names

This stage is especially important for quantum/eVTOL-like episodes where a basket can be dominated by a small number of names.

### Stage E3 — Regime-conditioned scoring

Do not create arbitrary bull/bear rules. First classify observable regime state, then test whether the same early score behaves differently across regimes.

Candidate state variables:

- SPY 13/26/52-week trend
- cross-asset volatility regime
- market breadth regime
- rates/liquidity proxy regime
- theme-correlation regime

The score may be calibrated by regime only if each regime has enough independent OOS episodes; otherwise it remains one global model.

### Stage E4 — PIT fundamental/filing diffusion

Only after price/breadth models are stable, add information whose historical availability can be proven.

Examples:

- revenue/capex/order/backlog diffusion by theme
- filing keyword diffusion
- estimate-revision breadth from legally available sources
- ETF flow proxies when historical availability is reproducible

No current constituent list or current company description may be projected backward.

### Stage E5 — Probabilistic early-event model

Replace the operational binary label only at the research layer with a calibrated probability such as:

`P(major_theme_episode within 26w | information available at decision time)`

Candidate models should begin with interpretable regularized logistic/ordinal models before tree ensembles or neural nets. Evaluation must add:

- AUPRC
- Brier score
- calibration slope/intercept
- reliability bins
- probability-decile lift

The binary watch/confirm UI can then be derived from frozen probability thresholds, but thresholds must be selected on development data only.

### Stage E6 — Purged walk-forward model selection

Any learned model uses annual or slower retraining, only matured labels, and purge/embargo rules that cover the label horizon. Hyperparameter selection must be nested inside the development/training window.

The 2019-2025 frozen OOS window is never recycled as a general hyperparameter laboratory after failures are observed.

### Stage E7 — Shadow/live evidence

2026 remains shadow/live-only until its forward labels mature. Store every decision-time cross-section and never rewrite past live outputs after new data arrives.

Track:

- signal count per week
- new confirmations
- stale confirmations
- ex-post hit/miss only after maturity
- calibration drift
- universe/coverage drift
- model/version/protocol hash

Only after a sufficiently long shadow period should any operational use be considered.

## 5. Anti-overfitting rules

The following are explicitly prohibited:

- tuning thresholds until solar 2020, quantum 2024, or eVTOL 2024 looks correct
- replacing a failed negative control with a different negative control
- changing the OOS window after seeing results
- using current ETF constituents for historical breadth
- deleting delisted members because the data source is inconvenient
- promoting an exploratory artifact to a validation claim
- selecting a model because one best year/theme dominates aggregate performance

Any research change must record:

- git SHA
- protocol hash
- data manifest/checksum
- development/OOS split
- exact feature set
- exact thresholds/hyperparameters
- result state (`PASS`, `FAILED`, `NOT_READY`)

## 6. Priority order from here

1. Build/freeze the PIT-complete adjusted-return dataset and membership history.
2. Generate formal independent episode labels and negative-control labels without tuning the radar.
3. Run `theme-leadership-validate-early` for the full 2019-2025 matured OOS period.
4. Add the separate cost/turnover portfolio backtest and robustness pack.
5. Only then decide whether the current early radar is worth keeping, simplifying, or replacing with M1/M2.
6. If it passes, begin E1/E2 model-ladder experiments while keeping M0 and the passing radar frozen as benchmarks.

The most important evolution principle is simple: **better models are allowed; easier tests are not.**
