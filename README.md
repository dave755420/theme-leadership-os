# Theme Leadership OS

Theme Leadership OS v0.1 is a local-first Python package for turning public,
point-in-time evidence into auditable theme and signal research. The first
implementation now includes the data/domain contracts, a versioned catalog,
three deterministic signal engines, validation primitives, and an offline CLI /
dashboard surface.

## Design commitments

- **Three separate outputs.** The product exposes three distinct lanes:
  `Current Leader`, `Emerging Radar`, and `12M Hold Candidate`. Each lane keeps
  its own evidence, status, and review context; a combined narrative is not a
  substitute for the three outputs.
- **Free data only.** The project may use public sources whose terms permit the
  intended use. It does not require a paid feed and does not redistribute raw
  market data.
- **Point-in-time (PIT) discipline.** Every observation must distinguish when an
  event happened from when it became available to the researcher. Future
  information cannot enter an earlier decision window.
- **M0 first.** The fixed baseline is a transparent, reproducible 26-week
  SPY-relative-strength rank. Claims are evaluated against its frozen definition
  and against pre-registered challenge cases and negative controls.
- **Core before analyst tooling.** The presentation dashboard is available for
  local research, but the analyst module / Wall Street Insight Daily remains
  blocked until the core contracts, PIT checks, and validation gates pass.

The three lanes are backed by separate theme-definition, signal-ledger, and
validation-pack artifacts; those supporting artifacts are not a replacement for
the product lanes. The governing details live in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
[`docs/VALIDATION_PROTOCOL.md`](docs/VALIDATION_PROTOCOL.md), and
[`docs/DATA_POLICY.md`](docs/DATA_POLICY.md).

## Current status

The v0.1 core implementation is present, but it is not release-passing yet.
The bundled catalog contains solar, quantum-computing, and eVTOL themes plus
broad-market, traditional-oil, and consumer-staples negative controls. The
catalog and CSV provider are local and PIT-aware; the yfinance adapter is
explicitly opt-in personal research only. The exploratory baseline is marked
`exploratory_rejected` because its prices were unadjusted and its historical
membership/coverage was not fully point-in-time complete. It is a research
record, not an investment claim. See [`docs/PROJECT_STATE.md`](docs/PROJECT_STATE.md) and
[`docs/NEXT_ACTIONS.md`](docs/NEXT_ACTIONS.md) for the current gate and handoff.

## Implemented surfaces

- **Data and domain:** normalized `Observation` and `SourceManifest` records,
  SHA-256 provenance, memory/file caches, CSV fixtures, PIT-safe as-of filtering,
  temporal membership guards, and explicit coverage/delisting statuses.
- **Catalog:** `data/catalog/themes.yaml`, `memberships.csv`, and `sources.yaml`
  load through `theme_leadership_os.catalog.load_catalog`; PyYAML is optional and
  a constrained parser covers the shipped catalog shape.
- **Signals:** `current_leader`, `emerging_radar`, and `hold_candidate_12m`
  consume tidy `date, theme, security, value` data, aggregate to Friday weeks,
  expose component/gate columns, and preserve M0 and next-session labels.
- **Validation:** integrity audits, rank-IC/spread/hit metrics, purged
  walk-forward splits, episode/early-hit labels, and holdout challenge/negative
  control metadata.
- **Local tooling:** `python -m theme_leadership_os.cli` provides `doctor`,
  `catalog`, `score`, `validate`, and optional `dashboard` commands. The
  Streamlit view is presentation-only and does not enable analyst claims.

The exploratory result is summarized in
[`research/EXPLORATORY_BASELINE.md`](research/EXPLORATORY_BASELINE.md) and
must not be used as a validated performance statement.

## Development

Python 3.11 or newer is required. Install the development tools with:

```bash
python -m pip install -e '.[dev]'
```

The checks used by CI are:

```bash
ruff check src tests
python -m compileall -q src
python -m pytest -q
```

Useful offline commands after installation:

```bash
theme-leadership doctor
theme-leadership catalog
theme-leadership score --demo
theme-leadership score --input examples/sample_prices_wide.csv
theme-leadership validate --input examples/sample_prices_wide.csv
```

Typer and Streamlit are optional app dependencies; the CLI falls back to
argparse when Typer is absent, and the dashboard reports a clear optional
dependency status when Streamlit is absent. Live fetching is disabled by the
local-first CLI. The yfinance adapter requires an explicit personal-research
opt-in and is not PIT-complete.

To run the local dashboard, install the app extra and launch it through the
same command surface:

```bash
python -m pip install -e '.[app]'
theme-leadership dashboard
```

## License and data reminder

The software in this repository is licensed under the Apache License, Version
2.0. See [`LICENSE`](LICENSE). Third-party software is listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Data terms are separate:
consult [`docs/DATA_POLICY.md`](docs/DATA_POLICY.md) before adding a source or
committing an artifact.
