# Exploratory baseline — 2026-08-30

This note freezes the first exploratory result before the production validation
pipeline is built. It is deliberately a failure-aware research log, not a
backtest claim and not an investment recommendation.

## Data posture

- Weekly closes from the public Nasdaq historical endpoint, 2018-01-01 through
  2026-08-30.
- 50 downloaded symbols and 37 sector/theme series, including three synthetic
  baskets.
- Prices were **not** adjusted for dividends or splits.
- Synthetic baskets used currently observable survivors and did not completely
  resolve delistings. `LILM` and `NOVA` were unavailable from the endpoint.
- Raw prices are intentionally not committed. Only aggregate results are kept.

These limitations make the run exploratory. It cannot pass the project's
point-in-time, coverage, or delisting integrity gates.

## Cross-sectional result

Both models were evaluated on the same 400 weekly cross-sections and future
26-week excess return versus SPY.

| Model | Mean Spearman rank IC | Median rank IC | Mean top-minus-bottom quintile excess return |
|---|---:|---:|---:|
| `M0`: 26-week relative-strength rank | 0.1141 | 0.1357 | 2.00% |
| Rejected composite | 0.0879 | 0.1216 | 0.70% |

The composite mixed 13/26-week excess return, relative slope, acceleration, and
distance from a relative moving average. It underperformed the simple M0
baseline on both IC and portfolio spread, so it is rejected rather than promoted
into the product.

The composite's strict major-rally alert produced 473 weekly confirmation events
with weighted precision of only 12.9% under the exploratory label definition.
This is far below the planned precision gate.

## Sentinel observations

| Series | Mechanically selected 26-week window | Window return | Signal in the routine's audit window |
|---|---|---:|---|
| TAN | 2020-06-26 to 2020-12-25 | +194.6% | Confirmed 2020-07-03 |
| Solar pure basket | 2020-04-03 to 2020-10-02 | +249.3% | Confirmed 2020-02-21 |
| Quantum pure basket | 2024-06-28 to 2024-12-27 | +1,704.0% | No qualifying confirmation |
| eVTOL pure basket | 2024-06-28 to 2024-12-27 | +125.6% | No qualifying confirmation |

The extreme quantum figure is especially sensitive to small-cap corporate
actions and the unadjusted-price limitation. It must not be quoted as an
investable return. The important result is that this specific detector did not
qualify either quantum or eVTOL within its predeclared audit window.

## Frozen design decision

1. Keep M0 as the benchmark every later model must beat out of sample.
2. Separate `Emerging Radar`, `Current Leader`, and `12M Hold Candidate`.
3. Treat radar alerts as research/watchlist items until all acceptance gates pass.
4. Add point-in-time membership, adjusted total-return data, delisting stress,
   negative controls, purged walk-forward splits, and execution delay before any
   investment-use claim.
5. Do not tune weights on solar, quantum, or eVTOL after viewing these results.
