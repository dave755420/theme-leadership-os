# Validation protocol

Status: **implemented primitives, not yet investment-use certified**

This protocol is frozen before tuning the signal engines. Solar, quantum
computing, and eVTOL are sentinel cases, not optimization targets. Passing a
famous case cannot compensate for weak out-of-sample results.

## Three questions, three evaluations

| Product lane | Question | Primary horizon |
|---|---|---:|
| Current Leader | Is the theme already leading the market? | trailing 13/26/52 weeks |
| Emerging Radar | Is leadership starting to form? | forward 13/26 weeks |
| 12M Hold Candidate | Has leadership been durable enough to research for a fixed one-year hold? | forward 52 weeks |

Radar output is a high-sensitivity research watchlist. It is never a buy
recommendation. Theme discovery and fixed-52-week investability are scored
separately: a theme can be discovered early and still be a poor one-year hold.

## Point-in-time hard rules

- Calculate a weekly signal after Friday's close and execute no earlier than
  the next trading session.
- Require `feature_available_at <= decision_at` for every feature.
- Use an ETF only after inception and after at least 26 weeks of history.
- Never apply current ETF holdings or current company knowledge to an earlier
  date.
- Activate a synthetic-basket member on the next trading day after public
  evidence supports the exposure.
- Identify securities with a permanent identifier/CIK; tickers alone are not a
  corporate-action history.
- Retain acquired, bankrupt, and delisted members. Unresolved delistings are
  tested with last-price, -50%, and -100% recovery assumptions.
- A normal synthetic basket needs at least five names. Three or four names are
  labelled `thin_low_confidence`; fewer than three are insufficient coverage.
- Store source ID, accession/evidence URL, retrieval time, availability time,
  transform version, and checksum. Raw data with unclear redistribution rights
  stays outside Git.

Any violation blocks performance publication, regardless of the apparent
return.

## Frozen outcome labels

### Forward labels

For each Friday decision date, enter on the next session and record absolute
and SPY-relative total return for 13, 26, and 52 weeks, parent-sector excess
return, maximum relative drawdown, and exact 52-week hold return. A weekly
`Leader_h` label requires positive excess return and a top-20% cross-sectional
future-return rank for horizon `h`.

### Major Leadership Episode

A candidate episode must satisfy all of the following within the next 52 weeks:

- maximum theme gain of at least 80%;
- SPY excess return of at least 40 percentage points;
- top 10% among eligible themes;
- at least 60% of members rise and at least 50% beat SPY; and
- at least four valid names.

A qualifying episode above 100% is additionally labelled `Explosive`. The
onset is the lowest weekly close in the 26 weeks before the first 80% crossing;
the peak is the subsequent 52-week maximum. Candidates less than 13 weeks apart
are merged.

An early hit is the first signal that appears in at least two of three weeks,
lands from four weeks before to eight weeks after onset, uses only then-public
data, and occurs before 25% of the onset-to-peak log move has been realised.
Reports show both remaining upside to the peak and exact 52-week return from the
signal date.

## Model ladder

Every layer uses the same universe, dates, execution delay, and costs.

1. `M0`: 26-week SPY-relative-strength rank.
2. `M1`: relative-strength levels and acceleration.
3. `M2`: M1 plus breadth, participation, and concentration.
4. `M3`: M2 plus point-in-time SEC/DART fundamentals and filing-term diffusion.
5. Logistic regression only if M3 beats the prior layer.
6. A tree model only if logistic regression beats the prior layer.

A layer that does not improve out-of-sample performance is removed. Historical
analyst revisions, ETF flows, and historical holdings are excluded from v1
unless a durable free point-in-time source is demonstrated.

## Walk-forward design

- Development observations end 2018-12-31.
- 2019-01-01 through 2025-12-31 is fully out of sample.
- 2026 observations are live/shadow only until their outcome horizon matures.
- Retraining occurs at most annually and uses only fully matured labels.
- Purge at least the label horizon (13/26/52 weeks), followed by a four-week
  embargo.
- Report a 52-week moving-block bootstrap and a monthly non-overlapping sample;
  overlapping weekly labels do not justify an ordinary independent t-test.
- Do not retune after opening a sentinel case. Any such revision starts a new
  protocol version and requires unseen time/theme holdouts.

## Required metrics

- weekly Spearman rank IC and ICIR;
- top-minus-bottom quintile and top-minus-universe future excess return;
- precision, recall, false-discovery rate, AUPRC, Brier score, and calibration;
- early-hit lead/lag, fraction of the move remaining, and annual false alerts;
- cost-adjusted CAGR, excess CAGR, information ratio, Sharpe, drawdown, Calmar,
  and turnover;
- exact 52-week hold hit rate and excess-return distribution; and
- results after removing the best year and best theme, delaying entry one week,
  and doubling costs.

Default round-trip cost is 20 bp for ETFs and 50 bp for synthetic baskets;
stress cost is 50 bp and 100 bp respectively.

## Acceptance gates

### Data integrity

- zero availability-time violations;
- zero pre-inception ETF observations;
- zero retroactive current-constituent uses;
- zero membership-interval violations;
- at least 95% required price/membership coverage; and
- unresolved delisting impact below 1%, or the result still passes under -100%
  recovery stress.

### Statistical and economic value

- OOS 26-week mean rank IC at least 0.05;
- 52-week rank IC above zero and the 26-week block-bootstrap 95% lower bound
  above zero;
- improvement over M0 of at least 0.02 IC and 2 percentage points in top-bucket
  26-week excess return;
- positive after-cost excess return in at least 70% of OOS years;
- positive result after removing the best year and best theme;
- at least 70% of base performance after a one-week delay and doubled costs;
- at least 30 independent 12M episodes across at least 10 themes; and
- 12M candidate hit rate at least 60% with median 52-week excess return at least
  5%.

### Early discovery

- episode recall at least 70%;
- false-discovery rate at most 35%;
- median first signal no later than two weeks after onset; and
- median remaining relative move at signal at least 60%.

Until every applicable gate passes, the UI must say `experimental/watchlist`
and must not describe any output as an investment recommendation.

## Sentinel and negative-control library

Formal sentinel cases are solar 2013 and 2020, broad pure-play quantum 2024,
and eVTOL 2023. eVTOL 2024 is a concentration stress case. Controls include
boom/bust or failed leadership in 3D printing, cannabis, metaverse, hydrogen,
space, EV/battery, clean energy, China internet, and SPAC themes. SPAC prices
before a business combination exposed the target theme are ineligible.

The first exploratory run is preserved in
[`research/EXPLORATORY_BASELINE.md`](../research/EXPLORATORY_BASELINE.md). Its
legacy composite lost to M0 and is rejected; it is not a passing validation.
