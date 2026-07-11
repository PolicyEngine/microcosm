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
- 2026-07-10 — Fit-transfer milestone (this milestone commit): used the
  existing populace-fit QRF transfer API to cover donor-observed required
  model inputs without overwriting native ACS leaves. Fits are conditioned on
  native state and observed predictor patterns; boolean, integer, and joint
  immigration-status support is preserved; donor geography is explicitly
  deferred; and every imputed column has immutable pattern/seed/channel/weight
  provenance. Checks: Ruff clean; 19 transfer tests pass.
- 2026-07-10 — Bounded-memory pool milestone (this milestone commit): replaced
  copy-heavy intermediate Frame rebuilds with direct final entity assembly,
  chunked ID-overlap checks, and guarded in-place block consolidation. Added a
  conservative 30 GB peak-memory preflight while preserving exact no-ACS
  identity and source immutability. A 100k-row, 64-column two-spine benchmark
  reduced measured peak growth from 1,260.7 MiB to 607.8 MiB. Checks: Ruff
  clean; 27 base-pool tests pass.
- 2026-07-10 — Contract-hardening milestone (this milestone commit): corrected
  the uncalibrated mixed spine to `IMPORTANCE` weights, made the memory limit a
  literal decimal 30 GB, recorded native observed/missing row counts, exposed
  resolved donor-channel provenance, and aligned preflight channel selection
  with transfer. Checks: Ruff clean; 57 pool, native-mapping, and transfer
  tests pass.
- 2026-07-10 — Hermetic acquisition milestone (this milestone commit): pinned
  the official 2024 1-Year national `csv_hus.zip` (251,500,587 bytes;
  SHA-256 `8281008e…e515d0`) and `csv_pus.zip` (602,847,146 bytes; SHA-256
  `afdc6d90…469894`). Added bounded streaming, timeout/size enforcement,
  unique verified temporaries, atomic cache replacement, strict manifest
  validation, and a repository-ignored content-addressed cache. Hosted hashing
  verified the full archives; the local sandbox DNS blocked cache
  materialization and left no partial file. Checks: Ruff clean; 14 acquisition
  tests pass.
- 2026-07-10 — Multispine staging milestone (this milestone commit): wired
  acquire/load/map/transfer/pool orchestration and a standalone nullable H5
  staging builder. The builder gates the selected donor channel before
  download, validates every planned ACS transfer, audits native/deferred input
  nulls, bounds retained QRF forests to eight targets per fit, records fit
  configuration and weights, releases the donor before a separate 30 GB
  export preflight, and round-trips fixed-format nullable tables one at a time.
  Group-quarters tenure/rent gaps and sub-PUMA geography remain explicit and
  the summary marks the artifact non-simulation-ready until downstream
  geography allocation/calibration. Checks: Ruff clean; 19 integration and
  staging-writer tests pass.
