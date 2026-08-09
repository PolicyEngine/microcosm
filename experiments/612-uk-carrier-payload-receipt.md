# #612 carrier-swap payload receipt (old writer vs Frame writer)

Committed receipt for the #618 acceptance claim (review ask on that PR): the
retired shadow-carrier writer and the Frame writer produce **payload-identical**
staging artifacts from the same input, verifiable offline against the digests
below. Companion JSON: `612-uk-carrier-payload-receipt.json`, produced by
`tools/compare_uk_h5_payload.py` (dtype-object comparison, digest-bound —
the post-review hardened version on the #618 branch).

## Provenance

- Input to both sides: the certified Microcosm UK candidate
  `populace_uk_2023.h5`, revision
  `populace-uk-2023-dd68c73-4aa4b14-20260619T023711Z`, sha256 `f17306cc…`
  (verified against the pinned `CERTIFIED_UK_CANDIDATE_SHA256` before use).
- **Left** (`roundtrip_old.h5`, sha256 in the JSON): produced on `main` at
  `e6be79a` via `load_uk_national_dataset` → `write_uk_national_dataset`
  (the shadow-carrier pair this PR retires).
- **Right** (`roundtrip_new.h5`, sha256 in the JSON): produced on
  `uk-frame-inc1-carrier-swap` via `load_uk_national_frame` →
  `write_uk_national_frame`.
- Run 2026-08-06 on the credentialed build machine; artifacts retained
  locally (`retention: local_untracked`), not committed — the digests bind
  this receipt to those exact bytes.

## Verdict

`payload_identical: true` — same store keys in write order; person
(1,157,100), benunit (618,980), household (535,080), and time_period tables
equal in column order, dtypes (object-level), index, row order, and values;
root attributes equal by raw value. The two files' own sha256 digests differ,
as expected: HDF5 stamps write times, which is why acceptance is defined at
payload level, never byte level.

The receipt is SDC-safe: it contains schema names, row counts, booleans, and
file digests only — no unit-record values.

Full acceptance context (gate-report parity, timings, the preflight):
https://github.com/PolicyEngine/microcosm/issues/612 (comments of 2026-08-06).
