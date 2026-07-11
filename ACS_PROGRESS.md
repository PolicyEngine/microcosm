# ACS multispine v1 progress

- 2026-07-10 — Source verification blocked, not waived: the live Census PUMS
  index ends at vintage 2024, and the 2024 1-Year directory lists
  `csv_hus.zip` (about 240 MB) and `csv_pus.zip` (about 575 MB), both dated
  2025-12-04. The mission's “2023” label conflicts with its 2024 URL and
  `acs_2024_1yr` spine name, so the official 2024 files are authoritative.
  Census publishes no archive checksums, terminal/browser downloads are
  unavailable in this sandbox, and no local copies exist; exact SHA-256 pins
  remain blocked rather than guessed.
- 2026-07-10 — Loader milestone (this milestone commit): added chunked national
  household/person ZIP-member loading, `SERIALNO` joins and lineage, ST/PUMA
  geography, deterministic relationship-derived US unit structure, occupied
  housing filtering, and explicit WGTP/PWGTP group-quarters weight handling.
  Checks: Ruff clean; 13 ACS-loader + ASEC-pool tests pass.
- 2026-07-10 — Optional base-pool milestone (this milestone commit): added
  entity-level `asec_puf` / `acs_2024_1yr` spine provenance, safe ID-remapping
  concatenation, source-specific missing-column alignment, support provenance,
  and mass-conserving configurable ACS allocation. The no-ACS path returns the
  identical frame object and has a serialized-digest regression. Checks: Ruff
  clean; 18 tests pass, including floating-point and bounded-memory cases.
- 2026-07-10 — Native-input mapping milestone (this milestone commit): mapped
  ACS demographics, adjusted WAGP/SEMP/SSIP, tenure, and observed property tax;
  retained combined SSP/RETP/INTP and RNTP/GRNTP as native predictors rather
  than inventing component splits or pre-subsidy rent. Blank universes remain
  missing, and finite adjustment factors and output collisions are validated.
  Checks: Ruff clean; 10 mapping-group and no-synthesis tests pass.
- 2026-07-10 — Loader mixed-GQ hardening (this milestone commit): scoped the
  PWGTP lookup to WGTP=0 group-quarters records so ordinary multi-person
  household duplicate SERIALNO values cannot make pandas reject the lookup.
  Checks: Ruff clean; 8 loader tests pass, including a mixed HU/GQ regression.
- 2026-07-10 — Loader structural-integrity milestone (this milestone commit):
  recoded official ACS MAR values into CPS marital codes using spouse presence;
  passed adjusted WAGP/SEMP/INTP/RETP/SSP to microunit only for dependency
  construction; required the 2024 native mapping columns; and used NP for
  vacancy, GQ, and person-count validation. Smoke caps now select occupied
  housing units deterministically and discard unselected person chunks during
  streaming. Checks: Ruff clean; 13 loader tests pass.
