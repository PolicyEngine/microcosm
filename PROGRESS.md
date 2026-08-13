# Round 13 progress

## State

The Round 13 failure and serializer inventory are complete. The supplied 1%
smoke reached both terminal gates and wrote their receipt, then the terminal US
H5 writer passed `person.is_female` (the first of 31 complete nullable-boolean
columns) directly to PyTables. All eight physical production
Frame/table-collection HDF serializers and all seven non-Frame writable HDF
sites now live in an executable registry guarded by a repository-wide AST
completeness test. A shared PyTables boundary codec now implements the doctrine,
and all six PyTables-facing serializers now consume it. The fiscal h5py
checkpoint has the same explicit values/mask doctrine, so all eight registry
rows are green. Stacked terminal publication now binds schema 8 to H5
materializer 2 in both the manifest and frozen metadata key; legacy schema 4
remains isolated. The changelog records the complete serializer closure.
Battery metrics and tolerances remain out of scope.

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
- Added one registry-driven round-trip contract over native bool, complete
  `BooleanDtype`, and missing `BooleanDtype` for all eight sinks. It pins
  source immutability, native-bool bytes, canonical false bits under nulls,
  exact NA masks, and semantic reloads. The red run produced seven intended
  failures: five PyTables BooleanArray failures, one PyTables BooleanCol
  failure shared by the two table-format routes, and the fiscal codec's
  missing-bool conversion failure. The generic Frame checkpoint is green.
- Added the shared nullable-boolean materializer in `microcosm-frame`:
  complete extension columns become native NumPy bool with identical logical
  bytes; missing columns become explicit object-backed Python bool + `pd.NA`
  and force fixed HDF format; inputs remain untouched. The common canonical
  values/mask primitive normalizes every masked value bit to false.
- Refactored Frame checkpoint schema v3 to use that primitive. In the locked
  local HDF environment, both the pre-change and post-change code produced
  identical bytes for the legacy fixture (`e55095...`) and nullable fixture
  (`7a6502...`); all 31 non-golden checkpoint/materializer tests passed. The
  committed legacy fixture hash (`7671ab...`) already disagrees with this
  environment on the unmodified parent and remains to be resolved during the
  exact golden proof rather than papered over here.
- Routed shared US terminal H5, UK national/rowwise, Axiom, PolicyEngine-US,
  preserved legacy two-spine, and ACS lean-checkpoint writers through the
  shared boundary. PolicyEngine-US now owns the compatible HDF layout locally
  and still reloads it with `USSingleYearDataset`, closing the external
  `.save()` bypass. The registry matrix passes seven rows, and 152 focused
  writer/reader tests pass (with expected optional-engine skips).
- Advanced fiscal target-frame checkpoints to schema 2/materializer 11 and
  stored nullable booleans as canonical bool values plus an optional uint8
  mask. The reader fails closed on missing, unexpected, nonbinary, empty, or
  misaligned masks, hidden true bits, malformed metadata, and schema-1 files.
  The full eight-sink registry matrix plus focused fiscal identity/corruption
  tests now passes (24 selected tests).
- Advanced only the stacked terminal envelope to manifest schema 8 and bound
  H5 materializer 2 in the terminal H5 metadata and `pool_h5` receipt. The
  reader requires exact, non-boolean integer agreement at both locations and
  rejects the version on legacy envelopes. The frozen artifact kind, HDF keys,
  and `entity_hdf_format="fixed_nullable"` are unchanged. Focused version,
  stacked-entrypoint, reader, and legacy-publication golden tests pass (27
  selected tests); the schema-4 legacy path carries no new field.
- Updated the #652 changelog entry from stacked schema 7 to schema 8/H5
  materializer 2 and recorded the eight-sink nullable-boolean doctrine,
  fiscal schema 2/materializer 11, frozen identifiers, and preserved legacy
  and unaffected UK artifacts.

## Next

1. Run focused tests, the exact 495-test #583 proof, full-workspace chunked
   exact-count proof, UK byte goldens, ruff/format/diff checks, and changelog
   validation. No builds will run.
2. Obtain an independent audit, close actionable findings, commit the final
   ledger state, and report the gradeable 10% dev-r7 prediction.
