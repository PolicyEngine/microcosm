# Final report: battery blocker lane — rare signed tails

## Outcome

The lane is complete locally. All 48 frozen red QED checks were classified
from frozen donor support, the one proven generating-code defect was fixed,
and exact realized-regime provenance is now durable. The branch was not pushed
and no pool build or logbook-chain operation was started.

The full adjudicated report is
`experiments/battery_rare_signed_tails/REPORT.md`; the machine-readable
48-check matrix is `experiments/battery_rare_signed_tails/realized_regimes.json`.

## Findings and changes

- All 48 QED checks recompute to gated fits: 35
  `zero_inflated_positive` and 13 `three_sign` at check level (35/7 across 42
  targets). None is degenerate or single-sign, so regime substitution or
  sign-conditional draw logic is not an honest fix. The QRF regime machinery
  is at `packages/microcosm-fit/src/microcosm/fit/qrf.py:92-150,950-1003,
  1333-1380` (authority/main numbering).
- Transfer ownership is 17 early gap-fill checks and 31 late
  producer-complement checks. Five intact sparse donors remain evidence
  blockers: ordinals 16, 28, 33, 46, and 75. Their verified sign-carrier
  counts are +18, +61, +27, -89, and -48; all exceed the five-carrier QED
  support floor (`stacked_spine.py:3027-3031`).
- Ordinals 78/80/82 and the Keogh leg share upstream retirement-support
  deletion. The old uniform 5,000-row cap retained 102/2,057 401k carriers,
  4/161 403b, 2/61 SEP, and 0/2 Keogh. The fixed helper retains the union of
  all four targets' nonzero rows, samples only common zeros, and calibrates
  only sampled-zero weights
  (`retirement_distributions.py:337-450,486-584`). The cap and seed values did
  not move.
- Realized QRF regimes now survive ordinary and banked transfer
  (`acs_transfer.py:1348-1418,1813-1823`), target-bank persistence
  (`acs_transfer_bank.py:25,47-53,249,357`), and exact early/late/aggregate
  receipts (`stacked_spine.py:3754-3772,8215-8360,9260-9268,9315-9331`).
  Authentic canonical JSON with sorted regime-map keys resumes correctly;
  missing target keys still fail closed.

## Keogh disposition

Keogh is not structurally absent. Native ASEC contains positive values 2,040
and 30,000; the old cap dropped both before transfer fitting, and all 1,736,840
finite frozen bank draws are zero. ACS absence was therefore manufactured on
the transfer donor path and is fixed by preserving source support.

The declared-absence route is rejected. If the signal had been structurally
impossible, it would require an exact recipient-scope absence equation
(`stacked_spine.py:8168-8195`) or a canonical tolerated-absence receipt
(`stacked_spine.py:6491-6528`), plus clone-exact battery structural receipts
(`stacked_spine.py:7186-7205`). Approval belongs to the US pool owner through
the owner-only reviewed route; this lane added no exclusion.

## Verification

- calibrate: 201 passed.
- data: 275 passed, 1 skipped.
- fit: 93 passed.
- frame: 294 passed, 36 skipped.
- build: 5,973 passed, 39 skipped.
- repository Ruff: all checks passed.
- US bundle generator check: passed, bundle SHA `5b0014c3…9554`.
- frozen regime-evidence generator check: passed.

The existing 1% before artifact projects to 127 failure lines over 93 legs;
47/48 frozen QED reds are visible. The committed baseline projection reproduced
byte-for-byte, and its self-diff is empty. No actual after-build exists for
this branch because the binding headless instruction forbade this lane from
starting one. The host-queue owner must run
`experiments/battery_rare_signed_tails/run_1pct_offchain_build.sh` and feed its
gates file to `diff_1pct_failures.py`. Keogh itself requires a full-scale build
because a 1% ASEC sample almost surely contains neither native carrier.

## Remaining owner work

Forty intact-support shape failures remain blocked on held-out calibration or
mapping evidence, and five sparse-donor failures remain blocked on denser
evidence. REPORT §2 gives the smallest target-scoped generating-mechanism
change for each of the nine classes. None calls for a gate, band, ceiling,
floor, fold, seed, cap-value, or exclusion change.
