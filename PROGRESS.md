# Round 13 progress

## State

The Round 13 failure and serializer inventory are complete. The supplied 1%
smoke reached both terminal gates and wrote their receipt, then the terminal US
H5 writer passed `person.is_female` (the first of 31 complete nullable-boolean
columns) directly to PyTables. All eight physical production
Frame/table-collection HDF serializers and all seven non-Frame writable HDF
sites now live in an executable registry guarded by a repository-wide AST
completeness test. The red dtype-family round-trip matrix is next. Battery
metrics and tolerances remain out of scope.

## Done

- Confirmed the worktree was clean, on `tail-stratum-support-652`, at
  `c079688fb82e41c85d4c67bbf35c59064bd89dca`.
- Preserved the requested branch despite its stale configured `origin/main`
  comparison; the no-network order forbids fetching a newer base.
- Read `CLAUDE.md`, the PolicyEngine repository standards, and the GitNexus
  debugging workflow.
- Confirmed GitNexus graph tools are unavailable in this session, so the
  serializer audit will use direct source searches and call-site tracing.
- Located the supplied smoke receipts/checkpoints and began enumerating all
  direct `HDFStore`, `to_hdf`, and PyTables use sites.
- Traced the exact exception through `_write_stacked_outputs` ->
  `write_nullable_us_h5` -> `_write_nullable_us_h5_file` ->
  `store.put(entity, table, format="fixed")`. The simulated checkpoint proves
  the first rejected block is `person.is_female`; 27 person and four SPM-unit
  nullable booleans are complete and therefore belong on the NumPy-bool path.
- Confirmed the 1% phase chain reached `terminal_gates` and
  `terminal_receipt_written` but not `publication_completed`. Completeness
  passed 131/131 targets. The battery evaluated all 132 comparisons with zero
  untestable and failed 127 (75 incidence, 49 quantile, three dead-both-zero),
  so its 124 metric misses are a later data question, not this code fix.
- Exhaustively classified eight physical HDF serializers: generic Frame
  checkpoints; shared US terminal publication; shared UK national/rowwise;
  Axiom entity tables; PolicyEngine-US adapter export; the preserved legacy
  two-spine writer; ACS local lean checkpoints; and fiscal target-frame
  checkpoints.
- Classified terminal-gate, diagnostics, and error receipts as JSON rather
  than Frame-table serializers; classified QRF/raw-draw HDF writers and
  attrs-only mutations as explicit non-Frame exclusions. No production
  `to_hdf` sink or ninth Frame-table serializer exists.
- Established version doctrine: retain the frozen US artifact kinds, HDF keys,
  and `entity_hdf_format="fixed_nullable"`; advance stacked publication schema
  7 -> 8 and bind a stacked-only H5 materializer version; preserve legacy
  schema-4 bytes. Any changed fiscal checkpoint codec owns its independent
  schema/materializer bump. Existing Frame-checkpoint schema v3 stays put.
- Added `FRAME_TABLE_SERIALIZERS`, with exactly eight logical sinks and their
  routes/version owners, plus seven explicit raw-array/attrs-only HDF
  exclusions. Its source scanner fails on any new writable production
  `HDFStore`/`h5py.File` site or any production `DataFrame.to_hdf` bypass.
- Proved the four registry/completeness tests pass in the dependency-complete
  local environment, without syncing or downloading packages.

## Next

1. Add failing registry-driven dtype-family round-trip coverage for all eight
   physical sinks.
2. Implement the
   lossless nullable-boolean representation, and bump changed serializer
   contracts without changing frozen published-artifact format identifiers.
3. Run focused tests, the exact 495-test #583 proof, full-workspace chunked
   exact-count proof, UK byte goldens, ruff/format/diff checks, and changelog
   validation. No builds will run.
4. Obtain an independent audit, close actionable findings, commit the final
   ledger state, and report the gradeable 10% dev-r7 prediction.
