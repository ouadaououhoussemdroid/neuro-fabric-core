# ADR 0002: Preserve `FLOAT8[]` for EEG analysis embeddings

- Status: accepted
- Date: 2026-07-10

## Context

Two historical migrations defined `public.eeg_analyses` incompatibly. The
earlier migration (`20260607000000_eeg_analyses.sql`) creates
`embedding FLOAT8[]`; the later migration attempted to create the table again
with `embedding JSONB`. This duplicate creation makes a fresh migration run
fail before the experiments schema can be installed.

The hosted Supabase schema was inspected during the G-01 investigation and
uses `embedding FLOAT8[]` (`_float8`). Application upload paths write numeric
vectors (`number[]`), which match PostgreSQL arrays directly. The checked-in
Supabase types incorrectly described the column as JSON because they were
generated from a non-authoritative schema state.

## Decision

`public.eeg_analyses.embedding` remains `FLOAT8[]`; its TypeScript contract is
`number[]`.

The first migration remains the sole creator of `eeg_analyses`. The duplicate
EEG-analysis DDL, policies, grants, and index are removed from the later
migration, which now owns only experiments and experiment runs.

No forward migration is added in G-01. The live database already has the
authoritative `FLOAT8[]` contract, and a later migration cannot repair a fresh
bootstrap that fails earlier in the sequence. No deployed data is converted.

## Consequences

- Fresh environments create `eeg_analyses` once and then create experiments.
- Existing environments are not changed merely by this repository correction;
  migration tools do not replay applied versions.
- A future pgvector migration must introduce a separately versioned vector
  storage contract rather than changing this persisted analysis field in
  place.
- The generated Supabase types must be regenerated against the hosted schema
  when privileged Supabase tooling is available; this G-01 edit applies the
  confirmed embedding type only.
