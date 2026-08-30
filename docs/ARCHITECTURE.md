# Architecture

Status: v0.1 implementation, core validation pending

## Purpose and boundaries

Theme Leadership OS is an evidence-to-validation workflow. It is not a trading
system, a market-data vendor, or an investment recommendation engine. The v0.1
implementation enforces the following boundaries:

1. ingest only permitted public/free sources;
2. record source and availability metadata before deriving signals;
3. produce three independently reviewable outputs; and
4. validate core behavior before exposing analyst-facing tooling.

The core implementation is available through `theme_leadership_os.data`,
`.domain`, `.catalog`, `.signals`, and `.validation`. The CLI and optional
Streamlit dashboard are local presentation surfaces. The analyst module / Wall
Street Insight Daily remains blocked until the core validation gate passes.

## Implemented components

| Layer | v0.1 implementation | Boundary or limitation |
| --- | --- | --- |
| Domain | `Observation`, `ThemeMembership`, `ThemeDefinition`, temporal comparison, exposure/confidence/coverage statuses, and explicit look-ahead guards | A membership is usable only when both its validity interval and evidence availability pass |
| Catalog | `load_catalog()` reads the versioned `themes.yaml`, `memberships.csv`, and optional `sources.yaml` bundle | The seed catalog is curated and coverage is explicitly partial for several themes; missing history is not fabricated |
| Data | `SourceManifest` checksums, `MemoryCache`/`FileCache`, provider-neutral `PriceProvider`, and deterministic `CSVPriceProvider`/`CSVFixtureProvider` | The default path is local; raw provider response objects do not cross the provider boundary |
| Personal adapter | Opt-in `YFinancePriceProvider` / `PersonalResearchPriceProvider` gated by `THEME_LEADERSHIP_ENABLE_YFINANCE` or `enabled=True` | Personal-research only, non-redistributable, and not PIT-complete; it is never an automatic catalog source |
| Signals | `current_leader`, `emerging_radar`, and `hold_candidate_12m` with typed configs and transparent component/gate columns | Tidy DataFrame input is required; weekly decisions use Friday-ending weeks and no later observations |
| Validation | Integrity audits, cross-sectional rank/spread/hit metrics, purged walk-forward splits, episode/early-hit labels, and holdout challenge metadata | The exploratory baseline is rejected; passing the core gate still requires a clean PIT-complete run and review |
| Presentation | Local CLI (`doctor`, `catalog`, `score`, `validate`, `dashboard`) and optional Streamlit dashboard | Presentation does not change signal values or turn research candidates into recommendations |

## The three product outputs

The fixed product contract is three separate, versioned lanes. Each lane has its
own status and evidence references. A renderer may display them together, but
they must remain separable at rest and in APIs.

| Output lane | Answers | Minimum contents | Must not imply |
| --- | --- | --- | --- |
| **Current Leader** | Which themes are already demonstrating current leadership? | Theme ID, as-of time, eligible signal summary, coverage/quality flags, validation status, and provenance links | A durable or investable forecast |
| **Emerging Radar** | Which themes show early evidence worth watching? | Theme ID, first-observed and available times, triggering evidence, uncertainty/quality flags, and validation status | A confirmed leader or recommendation; radar is research/watchlist only until gates pass |
| **12M Hold Candidate** | Which themes may merit a one-year research horizon? | Theme ID, horizon/as-of time, PIT-eligible evidence, M0 comparison, challenge/control status, and limitations | A promise of 12-month performance or investment advice |

The lane boundary is a correctness boundary. A theme may appear in one lane
without being copied into another; moving a theme requires an explicit,
versioned rule and a new as-of record. No lane may conceal an unrun or failed
validation gate.

The v0.1 signal engines implement these lanes as full decision-date/theme
panels:

- `current_leader()` uses 13/26/52-week relative-strength and breadth gates,
  participation, a positive M0 baseline, and a transparent weighted score/rank.
  Statuses include `current_leader`, `eligible_not_top`, `warning`, and
  `insufficient_history`.
- `emerging_radar()` uses 4/8/13-week acceleration and rank improvement for a
  high-sensitivity watchlist. Breadth, participation, and concentration remain
  visible warnings rather than silently suppressing a fast but narrow move. Its
  output is explicitly marked
  `WATCH_ONLY_NOT_A_BUY_RECOMMENDATION`.
- `hold_candidate_12m()` uses 13/26/52-week relative strength, breadth,
  participation, durability, two-of-three-week confirmation, concentration,
  annualized volatility, and drawdown gates. Statuses distinguish
  `hold_candidate`, `awaiting_confirmation`, `risk_rejected`, `warning`, and
  `insufficient_history`.

All three engines expose `m0_26w_spy_relative`, M0 availability/pass fields,
warnings, and next-session labels. An `as_of` argument limits decision dates
without allowing later rows into an earlier calculation.

## Supporting artifact contracts

The three product lanes are backed by three separately versioned internal
artifacts. These are evidence and audit contracts, not additional product lanes:

| Artifact | Answers | Minimum contents | Must not contain |
| --- | --- | --- | --- |
| **Theme definition** | What is the theme and why is it in scope? | Stable theme ID, thesis, inclusion/exclusion rules, universe, version, author, and source references | Unvalidated performance claims or silently inferred signals |
| **Signal ledger** | What dated evidence supports or contradicts the theme? | Signal ID, source ID, event time, availability time, retrieval time, transformation/version, value, quality flags, and citations | Any observation that was unavailable at the decision timestamp |
| **Validation pack** | Does the core survive baseline, challenge, and control tests? | M0 comparison, test window, metrics, challenge-case outcomes, negative-control outcomes, failures, and limitations | A recommendation presented without the evidence and gate status |

The supporting artifacts preserve auditability: a theme definition can be updated
without rewriting the signal ledger, and a validation pack can fail without
being hidden by a polished lane narrative.

## Data and time model

The implemented domain uses these temporal fields:

- `Observation.observed_at`: when the underlying event or measurement occurred;
- `Observation.evidence_available_at`: the earliest time the value could have
  been known by the permitted research process;
- `SourceManifest.retrieved_at`: when the source artifact was fetched or stored;
- signal `decision_date`: the Friday-ending evaluation timestamp; and
- signal `effective_date`: the next observed session label, which is not used in
  the decision calculation.

For any decision, an observation or membership assertion is eligible only when
`evidence_available_at <= decision_date` and its validity interval contains the
decision date. `observed_at` alone is never sufficient for point-in-time
eligibility. Corrections, restatements, delayed publication, and timezone
conversions must preserve the original availability record. The domain helpers
`is_available_as_of`, `filter_memberships_as_of`, and `assert_no_lookahead`
provide the explicit safe path.

`SourceManifest` also carries a stable source ID, URI, content type, SHA-256
checksum, coverage bounds, completeness and redistribution flags, and notes. A
missing availability timestamp is a data-quality failure, not permission to
substitute retrieval time. CSV fixtures may use a same-day fallback, but their
manifest is marked incomplete so the assumption remains visible.

## Processing boundaries

The implemented dependency direction is:

`catalog/source manifest → normalized observations → weekly signal panels → validation primitives → product lane`

`ThemeDefinition` and `ThemeMembership` are separately versioned catalog inputs
to the signal workflows. `Observation` values are normalized before they cross a
provider boundary; raw provider response objects are intentionally not part of
the interface. The signal engines return complete tidy panels, including
rejected rows, warnings, M0 fields, and gate columns so downstream consumers can
inspect why a theme did not qualify. Validation consumes caller-provided frozen
frames and never mutates the evidence it evaluates. Lane assignment is
downstream of the signal output and cannot change its values or gate status.

### Core boundary

The core consists of source/provenance contracts, PIT eligibility checks, theme
and signal artifact contracts, the three lane contracts, and the
M0/challenge/control validation primitives. v0.1 provides the building blocks:
`audit_integrity`/`assert_integrity` for availability, membership, inception,
coverage, and duplicate checks; rank-IC/top-bottom/hit metrics; purged
walk-forward splitters; and deterministic episode/early-hit labels. The
pre-registered challenge manifest contains historical positive windows for
solar, quantum, and eVTOL plus negative-control windows. The core is considered
passing only when the release gates in
[`VALIDATION_PROTOCOL.md`](VALIDATION_PROTOCOL.md) are met.

### Analyst boundary

The current dashboard is presentation-only: it renders local scorecards, keeps
the three lanes separate, shows component weights/data confidence/as-of/freshness
and risk flags, and displays a prominent research disclaimer. An analyst module
(interactive exploration, ranking, alerting, or narrative assistance) and the
planned **Wall Street Insight Daily** remain intentionally blocked until the core
passes. Any future analyst feature must consume the three product lanes and
their supporting artifacts; it may not bypass PIT checks, invent source
provenance, or turn a failed validation pack into a positive claim.

## v0.1 invariants and forward guardrails

1. **No silent time travel.** PIT eligibility is explicit and testable for every
   signal.
2. **No source laundering.** A derived value retains the source IDs and
   transformation versions that produced it.
3. **No output collapse.** `Current Leader`, `Emerging Radar`, and
   `12M Hold Candidate` have independent IDs, schemas, and statuses. Their
   supporting artifacts also remain independently addressable.
4. **No hidden baseline.** M0 is versioned and run on the same frozen window as
   any proposed method.
5. **No analyst bypass.** The analyst module cannot be enabled while a core gate
   is failing or unrun.
6. **No raw-data redistribution.** Repository artifacts contain metadata,
   hashes, and permitted derived summaries—not restricted vendor payloads or
   raw market-data dumps.

## Failure handling

Failures are first-class records in the validation pack. A missing source,
timestamp conflict, license ambiguity, parser error, or failed negative control
must be visible with severity, scope, and remediation status. The system should
prefer an explicit `unknown`/`blocked` state to imputation that could introduce
look-ahead bias.

The checked-in exploratory run demonstrates this posture: the simple M0
26-week relative-strength rank exceeded the rejected composite on the observed
rank-IC and top-minus-bottom spread, but the run used unadjusted prices and
incomplete point-in-time membership/delisting coverage. Its status is
`exploratory_rejected`; it is not a passing validation pack or an investment
claim. The next core run must resolve those data limitations before any analyst
or Wall Street Insight Daily surface is unlocked.
