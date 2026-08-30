# Data policy

Status: foundation policy, M0

This project is designed around free/public data and auditable point-in-time
research. Software licensing does not grant rights to data. Before a source is
used, its terms, access path, and retention/redistribution conditions must be
recorded in a source registry.

## Allowed sources

A source may enter the workflow only when all of the following are documented:

- it is publicly accessible or otherwise explicitly approved for the intended
  research use;
- its license or terms permit the planned retrieval, processing, and derived
  output use;
- attribution and rate-limit requirements are understood;
- the source provides enough publication/availability information for PIT
  reconstruction, or the record is explicitly excluded from PIT-sensitive use;
  and
- a stable locator, retrieval timestamp, content hash, and parser/transform
  version can be recorded.

Paid-only feeds, credentials copied from a third party, bypassed access
controls, and sources with unclear redistribution terms are out of scope.

## Point-in-time requirements

Store and distinguish `event_time`, `available_time`, `retrieved_time`, and
`decision_time`. Use the earliest defensible public availability time—not the
event date and not a later backfill timestamp—to decide eligibility. Preserve
revisions and corrections as lineage; never overwrite a historical observation
without recording what changed and when it became available.

If availability cannot be established, mark the observation `pit_unknown` and
exclude it from PIT-sensitive calculations. Do not infer availability from a
web page's current state.

## Storage classes

| Class | Examples | Repository treatment |
| --- | --- | --- |
| Public metadata | Source ID, URL, license note, retrieval time, hash | May be committed when accurate and non-sensitive |
| Permitted derived data | Aggregates or feature values allowed by source terms | May be committed with provenance and transformation version |
| Raw public payload | Downloaded pages, API responses, market-data files | Keep local or in an approved external store; commit only when terms explicitly permit redistribution |
| Restricted/vendor payload | Paid feed snapshots, credentials, or non-redistributable data | Never commit or redistribute |
| Secrets/personal data | Tokens, cookies, private identifiers | Never commit; use approved secret storage and minimize collection |

Raw market data is not redistributed by this repository. `.gitignore` contains
conservative local-data paths, but an ignore rule is not a license decision.
Review staged files before every commit.

## Derived outputs and attribution

Every signal ledger record and validation result must retain source IDs, source
locators, availability/retrieval timestamps, content hashes where possible, and
the transform/version that produced it. Derived does not mean unlicensed: a
source's terms may restrict commercial use, redistribution, or publication of
derived values. Include required attribution in the appropriate output or notice
file.

## Access, retention, and deletion

Use the minimum data needed for the declared research question. Keep raw
payloads only as long as the source terms and an audit need justify; remove or
quarantine data when terms expire, access is revoked, or a deletion request is
valid. Retain non-sensitive provenance and hashes when they do not reproduce the
payload. Do not put credentials or private user information in manifests,
fixtures, logs, or issue attachments.

## Review checklist

Before adding a source or artifact, a reviewer should be able to answer:

1. Who published it, under what terms, and where is that evidence recorded?
2. Can its availability time be reconstructed for each PIT-sensitive record?
3. Is the planned output permitted, and is attribution included?
4. Does the artifact contain raw market data, secrets, or personal data?
5. Can an auditor reproduce the derivation using the source ID/hash without this
   repository redistributing the raw payload?

An unresolved answer blocks the source from the core validation path.
