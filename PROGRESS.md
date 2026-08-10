# Progress: microcosm #653

## State

The failure mechanism and late-producer/source-input inventory are implemented
on `tail-stratum-support-652`, based on the three preserved #652 commits. The
checkout was clean at the start and was three commits ahead of the locally
available `origin/main` (`e9a352ca`). No fetch was performed because this task
forbids network access. A shared-ref update outside this worktree has since made
Git report the branch behind by one; the task remains on its required checkout
without rebasing, resetting, or shelving. Every producer execution row is now
content-bound to its declared inputs, outputs, callback receipt, and predecessor;
the top receipt is bound to entry/output frame content and independently carried
transition authority. That authority is propagated through cold/resumed pool
checkpoints, H5 schema 6, manifest construction, simulation, and publication.
The independent audit found additional hidden inputs in the source callbacks
and primary-PUF wrapper. Source execution controls and the ACS earnings-
universe materializer are now declared DAG nodes/resources. Every
physical input and virtual runtime resource is content-bound: donor bytes,
resolved PUF/QRF/tail controls,
the routed primary-QRF bank and stale-bank sidecar, source receipts, transfer
controls, and target-bank identities. The legacy envelope is restored to its
pre-#653 identity and cannot be selected by stripping stacked markers. Focused
suites and all final proof gates will be rerun from the final tree after the
documentation audit closes.

## Done

- Read the repository agent guide and the applicable debugging, data-pipeline,
  and development-standard instructions.
- Confirmed the active branch and preserved #652 commit chain:
  `c2bc06fe`, `9f184a07`, and `54d2dee6`.
- Located the late-transfer, post-clone source-completion, adult-care, SSTB,
  operator-boundary, checkpoint, and ordering-document surfaces to audit.
- Confirmed the executable order is PUF pass, then all post-clone source
  operators, then the 70-target late transfer. Adult care strictly consumes
  `sstb_self_employment_income_before_lsr` before that transfer can fill it.
- Reconstructed the failing checkpoint population: 342,732 ACS-origin rows and
  43,260 ASEC-origin rows per clone role. All 43,260 failing cells are on
  ASEC-origin clone-0 recipients; the issue's ACS-origin parenthetical is not
  supported by the saved checkpoint.
- Audited all 16 post-clone source operators, the primary PUF/tail producer,
  and all 19 canonical late-transfer groups. The direct scheduling edges are
  PUF/SSTB transfer to adult care, PUF/tuition transfer to education,
  pregnancy to WIC, and childcare to adult care, plus the declared PUF-role
  predictor and producer-to-transfer edges.
- Confirmed education currently hides its tuition dependency with
  `fillna(0.0)` and incorrectly claims the tuition passthrough as a source
  output. The DAG must make tuition PUF-only, transfer it before education,
  and make nonfinite tuition fail closed.
- Added red regressions for unfilled-input refusal before callback invocation,
  a deterministic named synthetic cycle, and byte-stable topology under
  reversed registry iteration. The focused test initially failed at collection
  because the deliberately specified DAG module did not yet exist.
- Implemented the pure producer-DAG core. It canonicalizes contracts and
  edges, derives lexically stable Kahn waves, reports a deterministic DFS cycle
  path, hashes canonical JSON bytes, and fences callbacks on exact filled-input
  or declared-absence evidence. Its three doctrine regressions now pass.
- Made qualified tuition a strict PUF-owned education input: nonnumeric,
  nonfinite, or negative tuition/assistance now fails; tuition is preserved
  byte-for-byte; education owns only assistance plus five AOTC facts. The
  source-producer surface is now 29 targets with two PUF overlaps, and 30
  education/partition regressions pass.
- Declared and import-validated the production late graph: one primary-PUF
  producer, all 16 post-clone source producers with full structured kernel
  input inventories, and the exact 19 bounded late-transfer groups covering
  70 targets. Its derived edges include pregnancy to WIC, childcare and
  SSTB batch 5 to adult care, and tuition batch 2 to education. Seven graph
  and registry doctrine regressions pass, including reconstruction under
  reversed registry iteration.
- Tightened every descriptive inventory into an executable contract gate:
  primary PUF declares 15 effective requirements and all 65 outputs; all 16
  source and 19 transfer nodes declare required alternatives and named
  tolerated-absence receipts for optional availability predictors. The full
  graph now derives 48 real edges, including primary-PUF dependencies into ten
  source operators and all 19 transfer groups. Nine DAG regressions pass.
- Split the post-clone source chain into a guarded single-producer entrypoint
  and an exact 16-receipt finalizer. The compatibility entrypoint now uses the
  same narrow API; deferred source inputs materialize only once after complete
  execution. All 54 multispine-pool tests pass.
- Integrated primary PUF/tail, all 16 source operators, and all 19 bounded
  transfers into one executable schedule. Clone attachment is now an explicit
  primary-PUF output and prerequisite of every post-clone source, so the first
  wave contains only `primary_puf_qrf`; the graph has 36 nodes, 54 edges, and
  wave sizes `(1, 17, 14, 3, 1)`.
- Put the primary PUF callback behind the same readiness fence as every other
  producer. Its donor and checkpoint are carried as explicit available-input
  receipts; optional sidecars remain counted declared absences, never zero
  fills. Finite-numeric input kind is now contract data, so object-backed
  `inf` or nonnumeric late inputs fail at the DAG boundary.
- Bound the complete DAG receipt to stacked authority v8, pool and stacked
  checkpoint materializers v4/v8, late-registry schema v2, and companion pool
  manifest schema v5. Cold execution, checkpoint emission/resume, manifest
  construction, publication, and schema-5 consumer loading all validate the
  exact 36-row execution, 16-source finalization, 19 transfer groups, and
  source/transfer aliases.
- Added executor-level and publication regressions for the derived order,
  single source finalization, batch-5-before-adult-care, nonfinite object
  numerics, forged execution rows, forged derived order, and missing schema-5
  DAG proof. The combined DAG, stacked, tool, and H5 suites pass after an
  independent review exposed and the implementation closed the hidden
  clone-attachment and unauthenticated-receipt gaps.
- Published the full primary-PUF, 16-source, and common/per-group transfer input
  inventories in the operator-ordering doctrine, together with all 54 edges,
  the five derived waves, schedule/payload hashes, readiness rule, cycle rule,
  new schema versions, and corrected 43-PUF/29-source/two-overlap accounting.
  Extended the #652 changelog fragment so the stacked tail and late-DAG fixes
  ship as one local PR train.
- A documentation-to-registry audit found that executable alternative columns
  dropped their declared `finite_numeric` kind during contract construction.
  Preserved the kind in the production registry and added a registry-level
  regression covering primary PUF, adult-care SSTB, education tuition, and an
  optional transfer predictor; the focused DAG file now passes all 10 tests.
- Replaced a Python/Pandas-version-specific pickle golden in the #652 tail
  preservation regression with an exact same-runtime comparison against the
  pre-#652 allocation path. The assignment SHA remains pinned, and the live
  pre-fix path and new all-adequate path have identical tables, dtypes, weights,
  strata, mass log, and frame digest.
- Ran the focused late-DAG, stacked, tool, H5, tail, adult-care, education, and
  transfer suites in the foreground: exactly 518 tests passed. The only golden
  changed by the finite-kind fix was the expected authority-bound legacy
  manifest digest; pool H5 and agreement bytes remained unchanged. Targeted
  Ruff check, format check, and diff check pass.
- Ran the #583 source-spine-blindness shard in the foreground. Its first pass
  fail-closed on the two new modules, so classified the pure scheduler as a
  reviewed non-operator module, classified the data-only registry as a narrow
  provenance owner, required both in the pool import graph, and moved the
  pinned graph size from 61 to 63. The complete shard then passed exactly 495
  tests.
- Ran the full workspace in eight non-overlapping foreground chunks. Exact
  results were: 795 passed/36 skipped; 1,446/26; 1,161/1; 460/0; 653/2;
  480/0; 495/0; and 324/1. Total: 5,814 passed, 66 skipped, 5,880 collected,
  with zero failures or errors. JUnit receipts independently carry those
  counts and prove the partition covers all 190 build test files plus every
  frame, fit, calibrate, and data test.
- Ran repository-wide `ruff check .`: pass. Repository-wide
  `ruff format --check .` reports 30 pre-existing files outside this branch's
  diff; none was rewritten. The format check over all 19 Python files changed
  since the task base passes, as do both the branch and worktree
  `git diff --check` gates. The worktree is clean.
- Final independent review found that adult-care support role should require
  clone index plus channel; several callback-numeric inputs were declared only
  nonnull; optional absence receipts currently conflate missing cells with
  invalid nonfinite cells; transfer wrappers consume undeclared all-entity
  cross-grain provenance; and schema-5 execution rows trust persisted counts
  rather than a hash-linked live transition proof. Reopened the implementation
  rather than issuing a premature ready verdict.
- Added production-registry regressions that require adult care's clone index
  plus channel as one all-of support-role input and require every value passed
  to a strict numeric callback path to carry `finite_numeric` contract
  semantics. The focused DAG suite now fails only on those deliberately red
  assertions (the first failure masks two additional source-kind assertions).
- Corrected those registry contracts and bumped the late-registry schema to
  v3: adult care now requires clone index and channel together; `PEDISDRS`,
  full-time-college status, and raw `ED_VAL` are finite-numeric; and every
  component or ACS aggregate in the transfer social-security, retirement, and
  investment alternatives is finite-numeric. All 12 focused DAG tests pass;
  the contract-only change preserves 54 edges and wave sizes `(1, 17, 14, 3,
  1)` while changing the schedule/payload identity as intended.
- Added red readiness regressions proving that a declared-absence receipt may
  authorize missing optional cells but never nonnumeric or nonfinite values.
  One exercises the generic fence and one poisons a canonical adult-care
  transfer predictor; both fail because the implementation does not yet expose
  separate missing and invalid counts.
- Split readiness into independent missing-row and invalid-value maps. Missing
  optional cells alone can mint the named absence receipt; present nonnumeric,
  infinite, or nonfinite values always refuse the callback and name both the
  logical input and declared producing stage. The generic and canonical
  transfer regressions pass, as does the real 36-node executor regression.
- Made both readiness maps exact contract surfaces: omitting a declared input
  can no longer default to a false zero, and extra inputs also fail with a
  canonical diagnostic. The focused DAG file now passes all 14 regressions.
- The receipt-integrity audit identified the source finalizer as an undeclared
  mutating producer: it consumes all 16 source receipts and creates three typed
  null SCF deferral columns. Added a red registry regression requiring an
  explicit 37th finalizer node, 16 incoming edges, and its exact three-output
  surface; collection fails because that node is not implemented yet.
- Implemented the source finalizer as a first-class producer. Each source now
  emits a declared receipt output; the finalizer consumes all 16 exact receipt
  resources before it may materialize `bank_account_assets`, `bond_assets`,
  and `stock_assets` with their explicit deferral receipts. Removed the hidden
  after-source callback. Registry schema v4 now has 37 producers, 70 edges,
  and wave sizes `(1, 17, 14, 3, 2)`; all 14 DAG tests plus the real executor
  regression pass.
- Added a red 19-group registry audit for the exact common transfer-wrapper
  surface: 28 physical provenance columns across six grains, household weight,
  and the assembly/stacked/PUF-attachment metadata receipts. It fails on the
  currently undeclared peer-grain inputs as expected.
- Bound that full validation surface into registry schema v5. Primary PUF now
  declares the 28 remapped structural columns, six resolved-weight resources,
  and attachment metadata as outputs; each transfer declares all peer-grain
  columns, household weight, and the three exact frame manifests. A poisoned
  family clone index and each missing manifest refuse a person-target transfer
  before its callback, naming the input and producing stage.
- Completed the strict numeric audit across all 16 source inventories and the
  70 late targets. Raw CPS code/value fields, wrapper IDs, weeks/role fields,
  adult/education inputs, and optional transfer predictors now fail on present
  nonfinite values. Direct late-target dependencies are import-partitioned into
  51 finite numerics, 17 domain-checked booleans, and two strings. The registry
  remains 37 producers/70 edges with waves `(1, 17, 14, 3, 2)`; all 17 DAG
  regressions and the targeted runtime refusals pass.
- Added executor-level red regressions requiring an immutable live-frame
  transition authority, an independently carried authority digest, a signed
  top-level DAG receipt, and rejection of both a fully rehashed forged receipt
  and a changed late output cell. All three fail on the deliberately absent
  content-binding API.
- Signed the full late transition: each execution row now hashes every
  declared alternative's scoped content, every declared output, the exact
  callback receipt, and its predecessor; the top receipt binds entry/output
  frame digests, the chain terminus, source finalization, and all nineteen
  transfer groups. The output Frame carries an immutable authority object and
  the executor returns its independently transportable SHA-256. The three red
  authority/content-drift regressions now pass.
- Bound that receipt doctrine into registry schema v6 and stacked authority
  v9. The canonical schedule payload now names the row, top-level, and
  immutable transition-authority contracts, so old identities cannot silently
  accept the stronger receipt semantics.
- Updated the operator-ordering doctrine to publish the 46-requirement primary
  inventory, the 15-requirement wrapper plus full kernel inventory for every
  source, the finalizer's sixteen receipt inputs, the 32-item validation plus
  12-item model bundle shared by all transfers, every per-group target-owner
  delta, all 70 dependency edges, the five derived waves, content-binding
  rules, version ledger, and canonical schedule/payload hashes. Extended the
  existing #652 changelog fragment so both fixes ship together.
- Propagated the independently carried late transition authority through the
  outer pool checkpoint dataclass, cold and resumed execution, transferred H5
  metadata/sidecar identity, stacked results, manifest construction, simulated
  stages, and both publication paths. The exact transferred frame is validated
  against the top DAG output digest; later declared mutations retain and
  validate the immutable transition anchor.
- Bumped the outer stacked checkpoint materializer to v9, the shared pool stage
  checkpoint materializer to v5, and the companion H5 manifest to schema v6.
  Schema-6 loads restore the signed transition authority into Frame metadata
  and reject a missing, stale, mismatched, or forged independently carried
  digest.
- Rebuilt the tool's synthetic late-DAG fixture as a structurally signed
  37-row receipt with exact input/output evidence, callback receipts, source
  finalizer resources, transfer reconstruction, execution hash chain, top
  receipt SHA, and live-frame authority binding. The full tool suite passes
  147 tests; the multispine runtime suite passes 54; the H5 suite passes 21
  with one optional-dependency skip.
- Re-read the saved 10% evidence without running a build: the assembled
  checkpoint has exactly 385,992 clone-0 people split into 342,732 ACS and
  43,260 ASEC rows; the authenticated QRF recipient bank has 209,854 ACS and
  23,146 ASEC clone-1 tax units; target checkpoint 051 is the SSTB input and
  hashes to `2c11f221fb965fe75e1fbc4abf29715d6022fd3f296909d87ec9119ff679a820`.
  The failing adult-care projection is ASEC-scoped, so its 43,260 invalid cells
  are the ASEC clone-0 recipients, not ACS-origin rows.
- Completed the final resource-identity audit and moved the registry to schema
  v7/receipt v2. The primary producer now declares 47 requirements, including
  its execution config; each transfer declares 46, including exact model
  config and target-bank resources. Kind-specific validators reject a shallow
  or internally rehashed incomplete binding, forged missing mandatory virtual
  evidence, evidence/receipt digest disagreement, and an identityless bank.
- Replaced the pandas 64-bit hash intermediate with domain-separated SHA-256
  over canonical scalar bytes, dtype, null bitmap, index, columns, and order.
  Fixed vectors cover null payloads, object scalar domains, serialization
  normalization, dtype/order drift, and a 250,000-by-four benchmark completed
  in 0.049 seconds.
- Bound the live PUF donor content, resolved default predictor/output lists,
  clone/QRF/tail controls, doctrines, and audit sinks into the primary
  execution row. The primary-QRF cache now writes and validates an exact
  late-resource sidecar, so a same-row donor mutation cannot reuse a stale
  internally valid bank. Transfer rows bind seed/fit controls and either the
  bank identity SHA or explicit ephemeral mode; changing bank or primary
  config changes transition authority.
- Isolated the retiring lineage at manifest schema 5 and checkpoint
  materializer 4 with a dedicated identity that omits the live stacked DAG.
  Its agreement golden remains byte-exact; its manifest golden changed once to
  the stable separated identity. Stacked publication remains schema 6 and
  materializer 5.
- Updated the ordering doctrine with the 47-input primary bundle, 14-input
  transfer model bundle, exact resource semantics, registry/receipt versions,
  and canonical hashes: schedule
  `250ef9f0a4fed5ca69672db9e39c51fa3d987d3d4cc2a0850f4c446eb955c52a`,
  payload
  `3144e82a11a4455a77541f135b06587e4cfe62cac62890e3fa026684a2dc684b`.
  Extended the existing #652 changelog fragment with the resource binding.
- Closed the source-callback audit gap in registry schema v8: every one of the
  16 post-clone source producers now declares a hash-bound execution config
  covering the fixed seed, fixed/absent period, retirement force-imputation
  switch, and explicit `not_supplied` mode for the only two optional sidecar
  arguments. Removed unreachable sidecar alternatives from the executable
  inventories, so every declared alternative can actually reach its kernel.
- Split the ACS PUMS earnings-universe-zero materializer out of the primary
  callback as registry-schema-v9 producer `acs_pums_earnings_universe`. It
  declares age, WAGP, SEMP, both mapped earnings columns, channel scope, and
  the exact rule/config identity; explicitly tolerates its structural input
  absences; emits the live application receipt; and gates primary QRF on both
  ACS earnings outputs plus that receipt. The derived graph is now 38 nodes,
  71 edges, and six waves `(1, 1, 17, 14, 3, 2)` with schedule SHA
  `070fdaac27446c7b367d24a160cb75a2df666c07135bd5d98961328b004ad303`
  and payload SHA
  `5f62351fe0d2d85d9d4a09fa699298e75e1bb82609ce657a102746c8477864b4`.
  A real-entry-shape regression starts with ACS under-15 raw and mapped nulls,
  proves the universe producer runs first, and proves primary sees explicit
  receipted zeros; a missing universe receipt refuses primary before callback
  and names its producing stage.
- Closed both persisted-readiness integrity gaps: the receipt validator now
  recomputes each logical requirement's missing and invalid counts from its
  exact physical alternatives, rejects inconsistent duplicate evidence, and
  enforces kind-specific input/output status and scope schemas. Completed
  producers cannot emit absent declared outputs. Generic absence receipts now
  bind the consuming producer and canonical reason, so a receipt cannot cross
  producer boundaries. The stronger row doctrine is checkpoint-bound in the
  schema-v9 payload; schedule SHA remains
  `070fdaac27446c7b367d24a160cb75a2df666c07135bd5d98961328b004ad303`
  and the final payload SHA is
  `525c1f47698a6a6bd54db7a3a1eb39bd2647680455770cfaa6be3ec1ef9a2994`.
- Restored the retiring two-spine envelope to the exact pre-#653 manifest
  schema 4 and checkpoint materializer 3. Its generated H5, diagnostics, and
  normalized manifest hashes match preserved #652 commit `54d2dee6` exactly:
  `ced797ecdd44a638c2a3945f07ad612098a7095ca53a5f458699bca6d6e38b3e`,
  `f39f0d918bf7ee01dddb5517d8830b8adb541273c5be084307be91397caca3cb`,
  and `14e6b3a409dfe2108253668a65ed32c0365b246f379ad895d8441c939adde65e`.
  H5 loading now classifies by schema plus the complete envelope surface,
  rejects stacked-only top-level/nested markers on the legacy route, and has a
  regression for stripping stacked pipeline/terminal fields, lowering both
  document schemas, and recomputing the diagnostics digest.

## Next

- Publish the final 38-node/71-edge doctrine, inventories, version ledger, and
  hashes; extend the changelog and reconcile stale progress wording.
- Rerun the focused aggregate, exact #583 shard, eight non-overlapping
  foreground workspace chunks, and Ruff check/format-check/diff-check gates.
- Write the final gradeable mechanism/edge/fix/proof report to the output file,
  echo it to stdout, commit the final progress state, and leave the worktree
  clean.
