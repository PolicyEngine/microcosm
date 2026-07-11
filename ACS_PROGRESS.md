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
