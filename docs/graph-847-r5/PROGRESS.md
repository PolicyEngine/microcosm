# PR #847 gate round 4 progress

## State

Eight of ten findings are fixed on the standalone `fix-847-r5` lane. The frozen
interface remains unchanged. Findings 4 and 6 are next, followed by the
complete required verification block and lane report.

## Done

- Read the repository operating instructions and the GitNexus debugging
  workflow. GitNexus query tools are not exposed in this lane, so source and
  focused-test traces will provide the debugging evidence.
- Confirmed `HEAD` matches `HEAD_SHA.txt`; the standalone clone's local branch
  is named `fix-847-r5`.
- Preserved the historical repository-root journals. This lane journal lives
  under `docs/graph-847-r5/` to comply with the explicit root-journal boundary.
- Read `docs/graph-acceptance.md`, including amendments 11--17 and the
  interface freeze; `docs/graph-interface.lock`; all seven named graph modules;
  the acceptance burndown tool; both named fit modules; and the H-parity test.
- Confirmed amendment 17's `NumericScope` interface already exists in frozen
  `kernel.py`, while executor context construction still supplies only the old
  tolerance projection.
- Reproduced finding 3 with five red parametrized cases: annotated
  `pytestmark`, an assigned skip alias, and all three `unittest.skip*`
  decorators returned no suppression problem. The scanner now rejects every
  decorator on a collected test that it cannot prove is an allowed direct
  pytest mark, and recognizes annotated/augmented module `pytestmark`
  assignments. The complete burndown-tool unit file and focused Ruff pass.
- Reproduced finding 5 through the graph store: `POPULACE_FIT_N_JOBS=1` and
  `=2` produced the same node/artifact identities but pickle bytes differed at
  byte 798. `_Forest` now serializes a shallow model copy with canonical
  `n_jobs=1`, leaves the live fitted model untouched, and restores the current
  runtime setting on trusted unpickle. The cache-collision regression and all
  fit-kernel tests pass.
- Reproduced finding 2 by changing only an authenticated body seed from integer
  `1` to float `1.0` and recomputing the serialized key; schema-v2 loading did
  not raise. Current-body validation now compares canonical bytes per receipt
  and tier, and also requires the accepted serialized key to equal the rebuilt
  manifest key. The full manifest unit file passes.
- Reproduced finding 9 with a 400-digit schema-v2 tolerance integer;
  `RunManifest.load` leaked `OverflowError: int too large to convert to float`.
  The parser now passes validated numeric values into `Tolerance` unchanged so
  its existing normalization raises `ValueError`, which the loader wraps as
  `StoreCorruptError`. The full manifest unit file remains green.
- Reproduced finding 1 from a real signed certified graph manifest: deleting
  only top-level decisions preserved its key/body and `load_certified` still
  returned certified. Release receipts now authenticate the normative required
  decision names, and certified loading rederives the outcome from those names
  and the carried signed records. Missing requirements fail closed; missing
  records report `unreached`. Focused executor, full manifest, and acceptance
  F-gate tests pass.
- Reproduced finding 10 by removing portable mass evidence from a calibration
  receipt while retaining an attached partitioned `MassRecord`; totals rendered
  but the partition heading and row were absent. The fallback now uses the
  canonical `mass_record_receipt` projection, preserving every partition field.
  All graph-explanation tests pass.
- Reproduced finding 8 with three cached EXPAND corruptions: an incumbent cell
  changed, an undeclared column added, and a declared new column removed. All
  three replayed without error. Cached restoration now requires the exact base
  plus declared column set for every entity, validates declared dtypes, and
  compares every incumbent column's complete prefix with storage semantics
  before checking copied additions. The focused regressions and full population
  unit file pass.
- Reproduced finding 7 with a dotted `expand_cells` column: the cold execution
  did not raise and authored provenance the warm parser cannot represent.
  Entity and column names are now required to be dot-free by the shared EXPAND
  parser, executor receipt-coordinate discovery uses that parser, and an
  all-node preflight refuses malformed declarations before source hashing,
  keys, cache I/O, or kernels. Both cold/auto regression variants pass without
  creating store objects. The combined executor/population run passes apart
  from the known QRF fixture pin made stale by finding 5 and due for finding 6.

## Next

- Implement and verify amendment 17 numeric-scope aggregation (finding 4) and
  the H1 node-key parity pin plus fixture regeneration (finding 6), preserving
  the frozen files.
- Run the complete required verification block and write the lane report to
  the requested output file.
