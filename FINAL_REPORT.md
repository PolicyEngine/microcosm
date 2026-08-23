# Final report: QBI ownership and coupled amount surface

## Outcome

Workstream 5 ownership is resolved. All eight red terminal QBI amount cells
were produced through the late `qrf_transfer` route. That terminal provenance
does not mean every mismatch begins there: all four incidence checks and UBIA
QED first turn red in transfer, while BDC, REIT/PTP, and W2 QED are already red
on the clone-1 PUF producer. The SHA-verified decomposition and exact values
are in `experiments/qbi_ownership/evidence.json:4-713,121397-121862`.

The separate two-check terminal-role fix is implemented because the evidence
supports it: exactly
`sstb_self_employment_income_would_be_qualified` and `business_is_sstb` now
compare clone 1, where both signals are live and in band. Every other physical
battery target remains clone 0. No exclusion was added.

The safe amount-surface groundwork for this lane is complete: realized regimes
and origin ownership are persisted and fail-closed through production
receipts, ready-H5 loading authenticates both early and late stacked receipts,
and a regression makes missing ownership channels unreleasable. No amount
model was changed without the host diagnostic build. The remaining refit and
25% certification are explicitly handed off in
`experiments/qbi_ownership/REFIT_PLAN.md`.
All eight amount checks remain blockers until that model work is demonstrated;
this lane resolves ownership, not the red margins themselves.

## Ownership evidence

The evidence extractor authenticates the adjudication's failed publication,
stage checkpoints, and 13 bank targets before computing any result
(`experiments/qbi_ownership/extract_qbi_ownership_evidence.py:809-995,999-1233`).
It treats clone 1 in the transferred checkpoint as the producer, clone 0 in
that checkpoint as post-transfer/pre-reconciliation, and clone 0 in the
simulated checkpoint as terminal. Those stage meanings follow the checkpoint
phase contract and placement of QBI reconciliation
(`tools/build_us_multispine_pool.py:2139-2141`;
`packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:2851-2908`).

| amount | producer ratio / QED | transfer ratio / QED | terminal ratio / QED | first-red incidence / QED |
|---|---:|---:|---:|---|
| BDC | 1.068456 / 0.584598 | 0.329192 / 0.827384 | 0.328928 / 0.751361 | transfer / producer |
| REIT/PTP | 1.047306 / 0.597071 | 0.385159 / 1.162639 | 0.446585 / 1.157625 | transfer / producer |
| UBIA | 1.107017 / 0.187807 | 0.774548 / 0.590155 | 0.774548 / 0.590155 | transfer / transfer |
| W2 | 1.172296 / 0.451897 | 0.760975 / 1.337128 | 0.760975 / 1.337128 | transfer / producer |

The production comparator groups each target by its registered clone role,
scopes positive-weight ASEC/ACS rows, computes the incidence ratio, and then
the conditional-quantile envelope
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:13891-13893,14006-14391,14441-14555,14598-14643`).
The extractor repeats that math and explicitly separates terminal provenance
from first-failing criterion
(`experiments/qbi_ownership/extract_qbi_ownership_evidence.py:546-635,1507-1525,1611-1767`).

For the historical artifact, `qrf_transfer` is a code-plus-receipt inference
because the old receipt predates `origin.channel`. Its bank bytes bind each
model target, and its late receipt binds `puf_clone` producer rows, 964,699
imputed complement rows, and zero residual/unmodeled rows
(`experiments/qbi_ownership/extract_qbi_ownership_evidence.py:809-995,1087-1195,1668-1717`).
The runtime selects ASEC clone 1 as donor, declares clone>0 rows producer-owned,
fills the complement, and verifies producer byte identity and accounting
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:10920-10972,11025-11041,11076-11191,11194-11313`).

Future receipts state that route directly. Group and aggregate ownership
fields, producer roles/counts, target origins, model-target bindings, pattern
catalogs, and sibling regimes all fail closed, with live producer counts
recomputed at finalization
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:5072-5198,5201-5754,11635-11759`).

## Realized regime before remedy

The frozen 108,073-row ASEC clone-1 donor support was reconstructed for each
of four availability patterns. The replay recomputes all 52 cells (13 chained
targets × four patterns); all 16 cells for the four red amounts are
`zero_inflated_positive`
(`experiments/qbi_ownership/extract_qbi_ownership_evidence.py:1857-1972`).
The failed attempt did not persist those regimes; the replay closes the
historical evidence question and the new code closes the receipt gap going
forward.

This result is established by donor replay. A self-consistent receipt alone
does not prove regime truth. Current monolithic and banked fitting recompute
the regime from the exact donor model frame and reject QRF disagreement
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1631-1651,1705-1885,1908-2206`).
Current receipts persist the complete origin and pattern-regime map
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:561-607,634-723`),
while stacked validation proves canonical predictor/catalog/model-target
structure and agreement among receipt copies
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:4344-4423,4426-4687,5364-5416,5676-5742`).

## Coupled whole-pool result

Before reconciliation, the nine coupled invariant counts are 35,121;
34,184; 2,996; 22,350; 34,817; 810; 13,433; 26,636; and 4,917. All nine are
zero at the terminal checkpoint (`experiments/qbi_ownership/evidence.json:121373-121396`).
The kernel owns the SSTB/non-SSTB splits plus the BDC and REIT/PTP exposure
caps, and its summary recomputes the nine equations
(`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1266-1359,1377-1487`).

Reconciliation does not cure the transfer defect. UBIA and W2 are unchanged
between transferred and terminal. BDC and REIT/PTP are level-adjusted by their
exposure caps, but their incidence ratios remain red. The producer-side QED
failures and transfer-side incidence failures therefore require a coupled
producer/transfer refit with whole-pool reconciliation afterward.

The frozen host 1% baseline at
`/Users/maxghenis/PolicyEngine/_buildo-runtime/out/battery-verify/baseline1pct/pool.gates.json`
(SHA-256
`1d6059868680f872fe04d452a536bcc3c215bafabb4c50d7740a469fe6a8b56a`)
has all eight amount checks red. The lane was forbidden to start a pool build,
so no unsupported before/after claim is made. The host 1% run must add
target × channel × availability-pattern gate diagnostics and exact exposure
bases, demonstrate the coupled refit, rerun reconciliation, and clear the same
eight checks plus all nine invariants. Only then should the fixed candidate be
certified at 25%.

## Exact-two SSTB terminal-role fix

The terminal clone-0 SSTB values are intentionally cleared by reconciliation
(`packages/microcosm-build/src/microcosm/build/us_runtime/qbi_inputs.py:1242-1264`),
while the SHA-bound clone-1 ratios are live and in band at 1.06307 and 1.06573.
The immutable authority consequently names exactly those two clone-1 targets
and defaults all others to clone 0
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:2133-2160`).
Validation rejects any altered role set and materializes role-aware registry
keys (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3030-3131`).
The canonical YAML agrees
(`packages/microcosm-build/src/microcosm/build/us/spec/battery.yaml:459-503,824-826`).

This is a comparator ownership correction, not an exclusion: removing the
checks would discard the only terminal signal for those PUF-detail outputs.
It is supported by the SHA-bound historical clone-1 evidence and contract
regressions; no post-fix pool artifact was built in this lane.

## Reproducible ownership regression

For each of the four amount targets, the test finds the group, aggregate, and
signed execution receipt copies, requires the same nonempty `qrf_transfer`
origin and exact model target, deletes `origin.channel` from each copy in turn,
rehashes every outer receipt, and proves validation still rejects the forgery
(`packages/microcosm-build/tests/test_us_stacked_spine.py:6581-6720`).
Additional rehashed tests reject donor-route, producer-role/count, predictor,
and single-target pattern-catalog forgeries
(`packages/microcosm-build/tests/test_us_stacked_spine.py:6723-6849`).
Ready stacked H5 loading also requires and validates both early gap-fill and
late-DAG proofs
(`packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py:594-640`;
`packages/microcosm-build/tests/test_us_multispine_pool_h5_io.py:1337-1376`).

## Verification

- `uv sync --all-packages --extra us`: completed successfully using the
  sandbox-writable uv cache.
- Canonical evidence replay: zero adjudication mismatches; output byte-identical
  to the committed JSON at SHA-256
  `38e60c1ec5e39b86df957148c877b3062ca97028f33ea0d1411013c2911c4b55`.
- Focused merged ownership/receipt suite: 20 passed, 278 deselected; the two
  corrected live-count cases then passed exactly.
- Exhaustive package gate on the resolved tree: frame 294 passed/36 skipped;
  fit 93 passed; calibrate 203 passed; data 275 passed/1 skipped; build 6,323
  passed/39 skipped across 6,362 unique tests. The build package ran in
  deterministic file partitions, then in one exact package-wide process. That
  82-minute process passed 6,321 tests and skipped 39; exactly two trade-entry
  subprocesses hit their unchanged 300-second limits under accumulated host
  load. The unchanged complete trade-entry file then passed 13/13 fresh, and
  the complete IMDB bulk file also passed in its fresh foreground run.
- `uv run ruff check .`: passed repository-wide. The two final compatibility
  files are Ruff-format-clean; both cached and unstaged `git diff --check`
  passed.
- Spec-engine coverage: 41,471/41,471 configuration fields and 40/40 inventory
  checks.

## Commits and constraints

- `9835fb4b` — start the QBI ownership journal.
- `f042bfa8` — persist realized QRF transfer regimes.
- `7e0c5081` — establish QBI amount ownership evidence.
- `742ec164` — assign the two SSTB checks to the PUF-detail role.
- The merge commit containing this report — integrate the frozen
  `origin/main` snapshot `d69131a3` and harden end-to-end ownership receipts.

No push, pool build, publication, QBI exclusion/register entry, gate/band/
ceiling/fold/seed tuning, or logbook-chain operation was performed. Unrelated
mainline changes were merged verbatim. `ps` access was denied by the managed
sandbox; the lane nevertheless started no build process and used only the
pre-existing host artifacts named above.
