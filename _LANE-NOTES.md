# Retirement model and data audit lane notes

## 2026-08-22 — lane start

### Scope and frozen constraints

- Branch/worktree: `battery-retirement-model-data` at `2aa96795`, tracking
  `origin/main` with a clean starting tree.
- Assigned package: `retirement_model_and_data`, 16 already-red checks across
  11 physical retirement legs. These are MODEL/DATA work, not eligibility
  credits; this lane will not add or request exclusions.
- Frozen experimental controls include gates, bands, ceilings, folds, seeds,
  and comparator definitions. None may be tuned toward passing.
- Pool builds are prohibited in this lane. The serial host owns any future 25%
  run; this lane may use only code/static tests and existing 1%/25% receipts.
- No push, publication, promotion, Logbook chain operation, or pending-chain
  edit is authorized. Peak RSS must remain below 15 GiB.

### Evidence read before implementation

- Read the assigned adjudication and JSON in the `microcosm-arm-split`
  worktree. Its guardrail requires a source/label/carrier/amount audit and a
  frozen-donor realized-regime recomputation for each red retirement leg.
- Read `experiments/phase_p_arm_split/SYNTHESIS.md`. The retirement regression
  is diagnosed there as a bundle-only predictor-replacement interaction; loss
  of `__acs_transfer_social_security_income` and
  `__acs_transfer_retirement_income` is a hypothesis, not an eligibility-arm
  credit or a proven baseline-red fix.
- Read the retirement-factorial lane's full `_LANE-NOTES.md`, relevant commit
  history, exact-row ledger, static f001 proof, and owner charter inventory.
  Its removal-only and broad-addition-only surgeries will not be duplicated in
  this lane.

### Environment receipt

- `uv sync --all-packages --extra us` failed first because the managed sandbox
  denied the default uv cache. Retrying with
  `UV_CACHE_DIR=/private/tmp/microcosm-retirement-audit-uv-cache` created the
  environment but could not download locked `blosc2==4.4.2` because DNS/network
  access is disabled. Neither dependency metadata nor `uv.lock` changed.
- `microcosm-spec-engine` has a complete environment and byte-identical lock
  SHA-256
  `ea7af7806a0beefe7394adefd5516649f3eba4740ae95ccdaa9aaa252249bc3a`.
  Its `.venv` was cloned copy-on-write. Its editable `.pth` files name the
  sibling worktree, so validation commands explicitly force this worktree's
  `microcosm-build`, `microcosm-calibrate`, `microcosm-data`, `microcosm-fit`,
  and `microcosm-frame` `src` directories ahead of those sibling paths.
- GitNexus debugging instructions were read, but query/context tools are not
  exposed in this session. Repository-wide search, history, and direct source
  inspection are the evidence fallback.

### Next

- Establish the exact ledger and inspect the frozen gate artifacts without
  building a pool.
- Audit every source and transformation path with current-tree line citations.
- Separate code-correctable derivations from declared concept absences and
  dense-rung refits; test and commit coherent fixes independently.

## 2026-08-22 — mandatory baseline import repair

- The required `packages/microcosm-build/tests` run reached 5,853 passed / 37
  skipped before three adjacent hard-crash publication tests failed their
  unchanged 60-second fresh-interpreter timeout; the run was stopped after the
  repeated condition at 96% (`8967.16s`). No retirement implementation code
  existed in that tree.
- The trace was import-only: the publication tool imports `us_trade`
  (`tools/build_us_import_entry_margins.py:110-120`), which first executes the
  broad runtime initializer. That initializer eagerly imported
  `spine_agreement`; the agreement module imports the typed take-up bridge
  (`packages/microcosm-build/src/microcosm/build/us_runtime/spine_agreement.py:42-53`)
  and formerly built its canonical registry at module load. The child therefore
  compiled the PolicyEngine-US ABI before it could reach the intended
  `os._exit(9)` crash point. Direct, warmed, idle, and alternative exact-lock
  Python 3.14 runs all exceeded the unchanged 60-second child budget. An offline
  exact-lock Python 3.13 sync was impossible because the locked Torch cp313
  wheel was not cached. No timeout, skip, dependency, authority, or test
  predicate was changed.
- The repaired package boundary keeps all ten names in `__all__`, declares the
  lazy surface, and resolves/caches each name on access
  (`packages/microcosm-build/src/microcosm/build/us_runtime/__init__.py:1072-1085,1940-1949,1989-2001`).
  The agreement module separately caches the typed canonical registry on first
  use, and the default gate uses that same accessor
  (`packages/microcosm-build/src/microcosm/build/us_runtime/spine_agreement.py:312-328,953-980`).
  This lets `multispine_pool` define the functions recursively inspected by the
  ABI before its module-level pool registry is constructed. Direct and `from`
  package imports retain object identity; the regressions cover the fresh
  unrelated import, identity/cache semantics, and the default gate
  (`packages/microcosm-build/tests/test_us_runtime_lazy_exports.py:27-133`).
- Focused validation before the collection-order extension: two lazy-export
  regressions and all four existing crash cases passed with their original
  timeouts. The first complete rerun then stopped at collection with 16 copies
  of the newly exposed partial-`multispine_pool` cycle (exit 2 after 4,158.24s),
  under a 13 GiB fail-closed guard whose peak aggregate RSS was 7,310,639,104
  bytes, maximum sample gap 3.183 seconds, and process-group-empty postcondition
  was true. The shared registry accessor above repairs both the cycle and the
  default-gate lookup; its two in-process focused tests pass. A fresh-process
  rerun hit the unchanged 60-second bound while host load exceeded 90, so it is
  not a green receipt and will be repeated before commit.
- Once host load fell, the unchanged fresh-process 60-second regression passed.
  A second complete run then collected successfully and reached 9% with no test
  failure and a 2,301,132,800-byte observed peak before the inherited guard
  aborted. Its shutdown treated any still-live historical PID as an owned
  survivor; rapid subprocess PID reuse exposed an unrelated protected process,
  and the attempted group stop raised `EPERM` before a final receipt was
  written. This run is discarded, not green evidence.
- At this point the lane-local guard used a 13 GiB limit; the later fixed 14 GiB
  reconciliation is recorded below. Its current implementation preserves 5 ms
  sampling, process-group and descendant enumeration, host-default escape
  refusal, and fail-closed exit
  (`experiments/retirement_model_and_data/rss_guard.py:16-18,81-104,203-260,310-350`).
  It reports monitor errors before shutdown, ignores recycled historical PIDs
  outside the owned group at the final postcondition, waits for group teardown,
  and treats an already-reused/unowned PID as non-killable instead of crashing
  before its receipt
  (`experiments/retirement_model_and_data/rss_guard.py:107-125,255-294`).
- A third complete run crossed 9% and exposed the suite's deliberate
  `start_new_session` child. Host builds must still abort such an escape. For
  test validation only, the guard now has an explicit
  `--allow-descendant-session-escape` mode that keeps the descendant in the RSS
  aggregate and direct-kill set; launch/final receipts disclose the opt-in and
  every observed escaped PID
  (`experiments/retirement_model_and_data/rss_guard.py:184-252,281-291,315-341`).
  The host-default self-test exits 0 with disposition `rss_limit`, two processes
  observed, a 135,053,312-byte aggregate peak against the 96 MiB test limit,
  0.072-second maximum sample gap, and an empty process group. A separate
  opt-in regression observes one escaped session, peaks at 42,483,712 bytes,
  exits 0, and proves the process group empty. Ruff and diff hygiene pass.
- The next clean full run reached 78% with passes and expected skips only, then
  the conservative 13 GiB guard stopped the process tree. The final discarded
  receipt reports disposition `rss_limit`, a 13,969,653,760-byte aggregate
  peak, a 13,964,328,960-byte individual peak, a 2.290-second maximum sample
  gap, no escaped descendant, and a true process-group-empty postcondition.
  This is an abort receipt, not green suite or gate evidence.
- The progress position coincided with the mounted 3.73 GB SIPP vehicle donor
  regression, which loads only required columns in chunks and retains December
  rows (`packages/microcosm-build/src/microcosm/build/us_runtime/sipp_vehicles.py:368-424`),
  then reduces them to household donors
  (`packages/microcosm-build/src/microcosm/build/us_runtime/sipp_vehicles.py:425-505`).
  Running that exact regression alone under the unchanged 13 GiB guard passed:
  peak aggregate RSS was 13,064,765,440 bytes, peak individual RSS was
  13,043,515,392 bytes, maximum sample gap was 0.155 seconds, and the process
  group was empty. The full-run stop was therefore expected donor-loading peak
  plus state retained by earlier tests, not a pytest failure.
- The retirement-factorial owner charter already requires stopping the full
  process tree when either individual or aggregate RSS reaches 14 GiB, expressly
  to preserve margin below the 15 GiB lane ceiling
  (`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-retirement-factorial/experiments/retirement_factorial/OWNER_25PCT_CHARTER.md:111-131`).
  The lane guard now uses that one fixed 14 GiB
  stop (`experiments/retirement_model_and_data/rss_guard.py:16,229-253`), has no
  runtime limit override, and will not be raised again. This resource-supervisor
  reconciliation does not alter any frozen scientific gate, band, ceiling,
  fold, seed, or comparator.
- The first run under that fixed 14 GiB stop again reached the mounted donor
  regression at 77–78%, with no pytest failure, then stopped at a
  15,197,569,024-byte aggregate peak and a 15,191,785,472-byte individual peak.
  Its maximum sample gap was 0.616 seconds, two intentionally escaped test
  sessions were disclosed, and the process-group-empty postcondition passed.
  The stop is fixed and was not raised.
- Source inspection identified retained input lifetimes rather than a semantic
  transform defect: the loader kept every copied December slice, its final raw
  chunk, and the text reader live after concatenation while numeric columns and
  many grouped reductions allocated additional surfaces. It now closes the
  reader when chunk selection ends and releases the source list and loop
  temporaries immediately after concatenation, before derivation and grouped
  reduction (`packages/microcosm-build/src/microcosm/build/us_runtime/sipp_vehicles.py:393-431`).
  The input columns, December predicate, 100,000-row chunk size, ordering,
  transformation, sample cap, and exact full-donor assertions are unchanged.
  This is a memory-lifetime correction required to validate below the frozen
  resource stop, not a gate/test/seed adjustment.
- The lifetime-only exact-donor rerun did not complete: the fixed guard stopped
  it at 15,056,240,640 aggregate bytes and 15,035,039,744 individual bytes,
  with a 0.051-second maximum sample gap and an empty process group. That
  receipt proves the source-slice cleanup alone is insufficient.
- The remaining large allocation was the pandas parser batch over a 3.73 GB,
  very wide source. The final loader keeps the same pinned source-column
  contract, omits only the contract-only `PNUM` from parsed columns, uses
  `low_memory=True`, and bounds the default chunk at 25,000 rather than 100,000
  rows
  (`packages/microcosm-build/src/microcosm/build/us_runtime/sipp_vehicles.py:368-424`).
  Chunk size is an execution-memory control, not a sampling operation: every
  physical row is still read once, the same December predicate is applied, and
  all selected slices are concatenated in source order. A regression compares
  the complete frame—including dtype and row order—across 1-row, 10-row, and
  default chunks
  (`packages/microcosm-build/tests/test_us_sipp_vehicles.py:395-422`). No test,
  source value, scientific gate/band/ceiling/fold/seed, or RSS stop changed.
- The exact mounted-donor regression then passed under the unchanged 14 GiB
  guard with all pinned support assertions intact. Its final receipt records a
  6,220,431,360-byte aggregate peak, a 6,199,197,696-byte individual peak, a
  0.052-second maximum sample gap, no escaped descendant, and a true
  process-group-empty postcondition. Compared with the first passing isolated
  receipt (13,064,765,440 aggregate bytes), the bounded parser removes more than
  6.8 GB of peak RSS without changing the transformed donor.
- A subsequent full run emitted two failures and six errors in its early block,
  so it was deliberately interrupted for a fail-fast traceback. The interrupt
  exposed a guard bug: `SIGINT` arrived during the 5 ms polling sleep, which was
  outside the per-iteration exception boundary, and the supervisor propagated
  `GuardInterruptedError` without a final receipt. The traced process group and
  all three last-observed PIDs were independently checked and were gone; no
  orphan remained. The sleep now sits inside the monitored boundary
  (`experiments/retirement_model_and_data/rss_guard.py:203-261`), so the same
  signal takes the fail-closed monitor-error path, terminates the selected tree,
  and writes the final group postcondition. A synthetic signal regression is
  required before another full run.
- That synthetic regression now passes: `SIGINT` produces exit 97 and the
  disposition `monitor_error:GuardInterruptedError:received signal 2`; the
  child returns -9, peak aggregate RSS is 13,778,944 bytes, maximum sample gap
  is 0.022 seconds, and the final process-group-empty postcondition is true.
- A compact guarded fail-fast run localized the first suite assertion to
  `test_spec_engine_country_bundles.py`: changing `sipp_vehicles.py` necessarily
  changed the source-inventory portion of the shared spec envelope, moving the
  BE hash from `262091db...` to `2fe12821...` and the UK hash from
  `e12a2cb...` to `6e652647...`. The canonical US generator `--check` passed
  without writing anything and still reports `f66894e0...`, proving the typed
  bundle resources are not stale. A guarded `spec_envelope_digests.py be uk`
  run recomputed the exact replacements at 417,480,704 aggregate bytes with no
  escape and an empty group. At this stage only the two country test
  attestations were repinned; no
  resource, source value, seed, scientific gate, band, ceiling, fold, or
  comparator changed.
- The full BE/UK country-bundle test matrix and the byte-for-byte US generator
  reproduction test pass together (16 tests). The guard reports a
  3,112,632,320-byte aggregate peak, a 3,091,398,656-byte individual peak, a
  0.090-second maximum sample gap, no escaped descendant, and an empty process
  group.
- Before restarting the mandatory suite, direct inspection of the cloned
  environment showed that its five editable `.pth` entries name the sibling
  exact-lock worktree. Earlier focused tests already forced this lane's changed
  `microcosm-build/src`; subsequent commands now prepend all five current-lane
  shard roots so unchanged dependencies cannot be selected from sibling
  source. The first fully isolated suite run passed through the country hashes
  and then produced exactly six setup errors in
  `test_spec_engine_coverage_tool.py`; it was interrupted through the repaired
  guard. The receipt records a 4,810,489,856-byte aggregate peak, return code
  -9, no escape, the expected `GuardInterruptedError` disposition, and an empty
  process group.
- The fully isolated coverage traceback identified only the source-derived seed
  protocol and compiled owner-map pins. A guarded canonical compile measured
  `e1b2dc753231cf2d3cd1fb6a03dc0914e8664379cbc9c5a99cf433e826325f7e`
  and
  `f490677f4cd455b0e13590806313d2e97778e15553e52b8e247bf2248a218c13`,
  while the US spec remained
  `f66894e0742999b15de737a20914443d0f14df87bd7546174a1ee0cee8dcab68`.
  The two reviewed inventory constants were mechanically repinned, and the
  canonical `tools/spec_engine_coverage.py` generator updated the committed
  report's expected/observed digests and US binding. It still reports
  41,379/41,379 fields and 40/40 inventory checks. All seven coverage-tool tests
  pass under a 1,754,890,240-byte guarded aggregate peak.
- A broader spec sweep found one further source-inventory golden vector in the
  synthetic loader test. Its semantic hash moved from `9afebede...` to
  `c6d3b112...`; only that exact assertion was repinned. All eight loader tests
  then passed, followed by the complete 280-test `test_spec_engine_*.py` plus
  `test_us_spec_bundle.py` sweep. The final guard receipt records a
  5,013,291,008-byte aggregate peak, a 4,989,190,144-byte individual peak, a
  1.806-second maximum sample gap, no escaped descendant, and an empty process
  group. No normative bundle resource, source value, RNG seed, scientific gate,
  band, ceiling, fold, or comparator changed.
- The next fully isolated build-shard attempt crossed the formerly failing
  early block and reached 62% before fail-fast found one remaining literal copy
  of the old US spec binding in
  `test_constants_adapter_equals_live_constants_and_stays_out_of_identities`.
  This was not a runtime mismatch: the observed binding was the already
  attested `f66894e0...`, while the test still expected `586491f0...`. The guard
  completed normally with pytest exit 1, a 12,179,341,312-byte aggregate peak,
  a 12,144,885,760-byte individual peak, a 0.187-second maximum sample gap,
  three disclosed intentional test-session escapes, and an empty process
  group.
- Repository-wide search found exactly two obsolete `586491f0...` literals,
  both in `test_us_multispine_pool_tool.py`: the live constants-adapter binding
  and its fixed checkpoint fixture. Both were mechanically repinned to
  `f66894e0...`. The two direct consumers pass, followed by the complete
  165-test tool file. That final sweep peaked at 2,663,579,648 aggregate bytes
  and 2,642,444,288 individual bytes, had no escaped descendant, and left an
  empty process group. Search now finds no copy of the superseded US spec hash.
- The next complete build-shard retry reached the same mounted SIPP source at
  77% with no pytest failure, but the fixed 14 GiB guard stopped the process at
  15,108,767,744 aggregate bytes and 15,084,699,648 individual bytes. Five
  intentional test-session escapes were disclosed, the maximum sample gap was
  2.755 seconds, and the final process group was empty. The guard and all
  scientific controls remain unchanged.
- Before changing the parser again, the exact mounted donor was saved as a
  complete pandas frame. Its shape is 16,841 by 21 and its canonical
  column/dtype/value/index digest is
  `12388b83a9d8f5fbd59bb1f7bedf21a00faca389a6518f36dd398fc3e544cd6c`.
  That guarded capture peaked at 6,425,657,344 aggregate bytes and left an empty
  process group.
- Source inspection localized the remaining peak to wide parser tokenization.
  The header still checks the full 20-column contract, including `PNUM`, but
  the reader now omits unused `PNUM`; `MONTHCODE` is discarded at the December
  predicate; tokenizer inference is bounded; and all retained non-ID fields are
  normalized before each selected slice is held
  (`packages/microcosm-build/src/microcosm/build/us_runtime/sipp_vehicles.py:393-430`).
  A fixture mixes integer, decimal, and blank numeric tokens across parser
  boundaries and proves exact output/dtype/order equality for 1-row, 10-row,
  and default chunks; another assertion proves omitted `PNUM` still fails the
  source contract
  (`packages/microcosm-build/tests/test_us_sipp_vehicles.py:395-443`).
- The exact mounted donor after that parser change is frame-equal to the saved
  pre-change output with `check_exact=True`, retains digest `12388b83...`, and
  now peaks at 494,583,808 aggregate bytes and 473,382,912 individual bytes.
  Its maximum sample gap was 0.051 seconds, no descendant escaped, and the
  process-group-empty postcondition passed. The committed mounted-source test
  pins that complete output digest in addition to the existing support counts
  (`packages/microcosm-build/tests/test_us_sipp_vehicles.py:445-489`).
- The reviewed source-inventory ripple from the final parser form is: BE
  `f565b00f7d820992aaa3f748465fc00ebb5caf326c7c83d7f6c236aa7b35a776`, UK
  `33bd40d1886691502a5e0eda9e29065bb5b78ea6bfddfa15243dd98506f404d0`, US
  `6d305cbd857623360bdc549372b2a969de71468de93dd5fa3ae4ff037377b51b`, seed
  protocol `5e446a76ddd528efc013534ef3633a02462328a25377a07a11fb8331f6960ab0`,
  seed map `d5e006a729dd6a69e982d78e1cf301578deb1974db71fabc69127c371384d0fb`,
  and minimal loader vector
  `7c9e48536528b0111cb80bd3c71e3803cf873520ca5b89e00de608e0b7eaf717`.
  The canonical US generator `--check` remained byte-clean. The canonical
  coverage generator reports 41,379/41,379 configuration fields and 40/40
  inventory checks. All 33 direct country, coverage, loader, generator, and
  multispine consumers pass; peak aggregate RSS was 3,554,951,168 bytes, no
  descendant escaped, and the process group was empty.
- The next full build-shard run proved the final SIPP parser in process: it
  passed the mounted source near 78%, then aggregate RSS fell to
  11,531,567,104 bytes. The run continued through 5,956 passing/skipped tests
  (97%) before the fixed guard stopped a different test at 15,206,301,696
  aggregate bytes and 15,163,572,224 individual bytes. There was no pytest
  failure, no escaped descendant, and the process group was empty.
- Collection order identified the active node as the second real IMDB archive
  fidelity check. That test loaded the 204,721,869-byte ZIP into a bytes object,
  duplicated it in `BytesIO`, inflated a member from the 1,736,614,841-byte
  archive, decoded it, and split 2,193,719 lines, while the long-lived suite
  allocator was already resident. It now streams the SHA-256 from the file,
  opens the ZIP path directly, scans each member once, verifies the complete
  declared line count, and retains only the manifest-selected raw records
  (`packages/microcosm-build/tests/test_us_trade_imdb_goldens.py:88-105,209-238`).
  This preserves and strengthens archive fidelity: size, whole-archive hash,
  all member line counts, selected raw lines, and layout bytes are checked.
- Both cached IMDB archive parameters pass after the streaming change in 2.50
  seconds. The guard recorded a 446,431,232-byte aggregate peak, a
  425,213,952-byte individual peak, no escaped descendant, and an empty process
  group. Focused Ruff passes.
- The full build-shard rerun passed the final SIPP parser and both streamed IMDB
  archive cases, then reached 5,963 completed tests (97%). The fixed guard
  stopped the active mounted voluntary-filing donor at 15,190,982,656 aggregate
  bytes and 15,164,506,112 individual bytes. Three intentional test-session
  escapes were disclosed, there was no pytest failure, and the process group
  was empty.
- The voluntary-filing loader reads the same 3.73 GB SIPP source but still used
  100,000-row `low_memory=False` parser batches and retained the open reader and
  every selected slice. A fresh default run was independently stopped at
  15,053,864,960 aggregate bytes. Its existing `chunksize=25_000` execution
  control completed at 6,179,848,192 bytes and supplied a complete preserved
  output: 22,296 rows by eight columns, all source-audit attributes, and digest
  `464b1a76504481d3a6d5bc87834ea83f85b7a57602385e25634e3e598fd8f4b8`.
- The default now uses 25,000-row bounded tokenizer inference, closes the
  parser, removes `MONTHCODE` at the December predicate, releases parser
  temporaries before numeric validation, and otherwise retains the exact
  validation/derivation path
  (`packages/microcosm-build/src/microcosm/build/us_runtime/voluntary_filing.py:350-418`).
  A mixed integer/decimal/blank fixture proves exact frame and audit-attribute
  equality across 1-row, 100-row, and default chunks
  (`packages/microcosm-build/tests/test_us_voluntary_filing.py:346-377`).
- The optimized mounted default is exactly frame- and attrs-equal to the
  preserved output and retains digest `464b1a76...`; peak aggregate RSS is now
  521,519,104 bytes and peak individual RSS 500,285,440 bytes, with no escape
  and an empty process group. The mounted-source test pins that complete digest
  in addition to its prior response/weight facts
  (`packages/microcosm-build/tests/test_us_voluntary_filing.py:468-525`).
- The voluntary source is outside the seed/spec implementation inventory. A
  guarded recomputation left the BE, UK, US, seed-protocol, seed-map, and loader
  golden hashes unchanged. The canonical US generator `--check` and coverage
  `--check` both pass byte-for-byte; coverage remains 41,379/41,379 fields and
  40/40 inventory checks.

## 2026-08-23 — mandatory baseline green receipt

- The present SIPP parser supersedes the intermediate `low_memory=False`
  experiment recorded above: the final exact-donor form uses bounded
  `low_memory=True` inference and normalizes retained December slices before
  accumulation
  (`packages/microcosm-build/src/microcosm/build/us_runtime/sipp_vehicles.py:393-430`).
  The preserved 16,841-row frame and digest remain exact, as pinned by the lane
  regression
  (`packages/microcosm-build/tests/test_us_sipp_vehicles.py:395-489`).
- The first final full-directory build-shard command used the required current
  worktree `PYTHONPATH`, `UV_NO_SYNC=1`, the writable uv cache, the fixed guard,
  and `uv run pytest packages/microcosm-build/tests -q`. The execution service
  sent `SIGTERM` at 56% after approximately 89 minutes. The guard correctly
  returned 97 with disposition
  `monitor_error:GuardInterruptedError:received signal 15`, peak aggregate RSS
  12,140,625,920 bytes, peak individual RSS 12,115,410,944 bytes, two disclosed
  intentional session escapes, and a true process-group-empty postcondition.
  No pytest failure had appeared, but this is not green evidence.
- To stay inside that external session lifetime without changing tests, 259
  sorted `test_*.py` files were partitioned once into the exhaustive disjoint
  ranges 1–50, 51–100, 101–139, 140–180, 181–220, and 221–259. Each range ran
  through the same fixed guard and exact `uv run pytest <files> -q` command.
  The US ranges used `uv` as the guard's direct child, matching the original
  package-directory process topology. A discarded shell-wrapped range 140–180
  received status 143 inside a process-control test; the identical direct-child
  rerun passed. The six accepted receipts are:

  | Slice | Trace | Return | Peak aggregate RSS | Escape receipt | Empty group |
  |---|---|---:|---:|---|---|
  | 1–50 | `/private/tmp/retirement-build-baseline-batch1.trace.jsonl` | 0 | 5,064,916,992 | none | yes |
  | 51–100 | `/private/tmp/retirement-build-baseline-batch2.trace.jsonl` | 0 | 2,195,652,608 | none | yes |
  | 101–139 | `/private/tmp/retirement-build-baseline-batch3.trace.jsonl` | 0 | 736,870,400 | none | yes |
  | 140–180 | `/private/tmp/retirement-build-baseline-batch4-direct.trace.jsonl` | 0 | 8,398,045,184 | none | yes |
  | 181–220 | `/private/tmp/retirement-build-baseline-batch5.trace.jsonl` | 0 | 7,078,264,832 | two intentional test sessions | yes |
  | 221–259 | `/private/tmp/retirement-build-baseline-batch6.trace.jsonl` | 0 | 9,061,023,744 | none | yes |

- The other required whole-directory package suites returned zero under the
  same fixed 14 GiB guard: frame peaked at 7,012,007,936 aggregate bytes, data
  at 11,890,933,760, calibrate at 498,663,424, and fit at 913,031,168. Every
  process group was empty. `uv run ruff check .` reports `All checks passed!`,
  and `git diff --check` is clean.
- Current `origin/main` is `057dd95f`, 212 commits beyond the lane's frozen
  base `2aa96795`. The supplied baseline/pkg3 artifacts and adjudication are
  bound to this lane base, so no merge or re-pin was inferred. No push, pool
  build, gate/band/ceiling/fold/seed change, exclusion, publication, or chain
  operation occurred.

### Next

- Commit this baseline-enabling step, then add the reproducible frozen-artifact
  audit and all 11 classifications as a separate coherent step.

## 2026-08-23 — retirement source/model/data audit

### Frozen proof and exact command

- Added `experiments/retirement_model_and_data/audit_frozen_artifacts.py` and
  its canonical `f001_audit.json`. Before decoding, the proof checks SHA-256 for
  the raw ASEC checkpoint, assembled/transferred checkpoints, baseline/pkg3
  pools and gates, adjudication JSON, and all 11 target-bank H5 files. It then
  resolves 4,311 unique raw `person_id` values through assembled
  `person_source_id` and proves all 16 retirement source columns equal, with
  missingness preserved, raw → assembled → transferred
  (`experiments/retirement_model_and_data/audit_frozen_artifacts.py:1764-1851`).
- The proof recomputes each target from the frozen source columns, requires an
  exact bit match to ASEC clone 0, selects the actual early clone-0 or late
  clone-1 donor role, and recomputes the QRF regime from that selected support
  (`experiments/retirement_model_and_data/audit_frozen_artifacts.py:1111-1335,1919-1975`).
  QRF regime selection is based on realized sign support, with weighted gate
  and conditional-amount fits for zero-inflated outcomes
  (`packages/microcosm-fit/src/microcosm/fit/qrf.py:83-150,950-1003,1333-1442`).
- This lane ran no pool build. The exact generation command was:

  ```bash
  PYTHONPATH="$PWD/packages/microcosm-build/src:$PWD/packages/microcosm-calibrate/src:$PWD/packages/microcosm-data/src:$PWD/packages/microcosm-fit/src:$PWD/packages/microcosm-frame/src" \
  UV_NO_SYNC=1 \
  UV_CACHE_DIR=/private/tmp/microcosm-retirement-audit-uv-cache \
  uv run python experiments/retirement_model_and_data/rss_guard.py \
    --trace /private/tmp/retirement-f001-final5-generate.trace.jsonl \
    --log /private/tmp/retirement-f001-final5-generate.log \
    -- uv run python \
      experiments/retirement_model_and_data/audit_frozen_artifacts.py
  ```

  It returned zero, peaked at 457,834,496 aggregate and 436,682,752 individual
  RSS bytes, observed no escaped descendants, and left its process group empty.
  The resulting JSON SHA-256 is
  `37e92f7358119c44670c104335d9452a8a4e9e22f28627a70c589691e4dc92bf`.
  The final freshness check uses the identical command with a fresh trace/log
  and `--check` appended to the audit script argv. It returned zero at
  453,787,648 aggregate / 432,586,752 individual bytes, with no escape and an
  empty group. The proof pins the battery's five-carrier QED minimum and now
  independently reproduces the f001 Social Security dependents/survivors QEDs
  `0.917093391855645` and `0.2780886784330607`
  (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3026-3030,11912-11930`).

### Eleven-leg source, carrier, amount, and regime result

All source, terminal, target-bank, and gate details are in
`experiments/retirement_model_and_data/AUDIT.md` and the canonical JSON. The
compact carrier ledger is:

| Leg | Frozen ASEC source equation | Exact source sign rows `(−,0,+)` | Actual QRF donor role and sign rows | Realized regime | Classification |
|---|---|---:|---|---|---|
| tax-exempt private pension | `0.410*(nz(PNSN_VAL)+nz(ANN_VAL))` | `(0,4050,261)` | ASEC clone 0, `(0,4050,261)` | zero-inflated positive | concept mismatch |
| taxable private pension | `0.590*(nz(PNSN_VAL)+nz(ANN_VAL))` | `(0,4050,261)` | ASEC clone 0, `(0,4050,261)` | zero-inflated positive | concept mismatch |
| taxable IRA | `Σs 1[DST_SCs=4]*DST_VALs` | `(0,4252,59)` | ASEC clone 0, `(0,4252,59)` | zero-inflated positive | dense-rung refit |
| SS retirement | current `SS_VAL` reason precedence | `(0,3661,650)` | ASEC clone 0, `(0,3661,650)` | zero-inflated positive | concept mismatch |
| SS disability | current `SS_VAL` reason precedence | `(0,4247,64)` | ASEC clone 0, `(0,4247,64)` | zero-inflated positive | concept mismatch |
| SS dependents | current `SS_VAL` reason precedence | `(0,4297,14)` | ASEC clone 0, `(0,4297,14)` | zero-inflated positive | concept mismatch |
| SS survivors | current `SS_VAL` reason precedence | `(0,4299,12)` | ASEC clone 0, `(0,4299,12)` | zero-inflated positive | concept mismatch |
| Keogh | `Σs 1[DST_SCs=5]*DST_VALs` | `(0,4311,0)` | ASEC-origin clone 1, `(0,4311,0)` | degenerate zero | dense-rung refit |
| taxable 401(k) | `Σs 1[DST_SCs=1]*DST_VALs` | `(0,4225,86)` | ASEC-origin clone 1, `(0,4249,62)` | zero-inflated positive | dense-rung refit |
| taxable 403(b) | `Σs 1[DST_SCs=2]*DST_VALs` | `(0,4305,6)` | ASEC-origin clone 1, `(0,4310,1)` | zero-inflated positive | dense-rung refit |
| taxable SEP | `Σs 1[DST_SCs=6]*DST_VALs` | `(0,4307,4)` | ASEC-origin clone 1, `(0,4309,2)` | zero-inflated positive | dense-rung refit |

The 59/41 pension equation is implemented before cloning
(`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:41-43,188-200,345-362`).
The account-code sums and strict source contract are implemented at
`packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:97-112,210-269`;
taxable IRA also has the pre-clone CPS-carried code-4 producer
(`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:490-500`).
The four Social Security equations use the current reason precedence at
`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:365-410`.
Early transfer preserves ASEC donor cells before clone expansion
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:7581-7698,7808-7820`),
whereas the late account transfer selects ASEC-origin clone 1 after the
internal CPS-trained PUF-role QRF
(`packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:336-473`;
`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8340-8483,8518-8538,8631-8729`).

### Classification and implementation boundary

- **Pensions — two concept mismatches.** Frozen ASEC has 83 government-only
  pension carriers totaling $3,361,227 that the current equation puts wholly
  in private leaves. Four rows mix private/government codes, seven have an
  unresolved source, and `ANN_VAL` has no ownership label; `PEN_VAL1/2` are
  absent from every frozen checkpoint. The owner equation in `AUDIT.md` routes
  unambiguous pure-source pensions and declares annuity, mixed, railroad/other,
  and unclassified evidence absent. It marks 56 f001 rows/$1,804,558 ambiguous.
  It is not implemented and no exclusion was added. A narrow public-leaf patch
  is unsafe because clone expansion copies pre-clone leaves and the PUF role
  maps its aggregate pension amount to the private leaf before selective
  overwrite
  (`packages/microcosm-build/src/microcosm/build/us_runtime/puf_support.py:208-224,1993-2047,2083-2143,3990-4008`).
- **Social Security — four concept mismatches.** Current precedence conserves
  `SS_VAL` but cannot identify a unique component for 35 of 740 positive f001
  rows/$744,734: 24 code-7/8 rows/$495,214 and 11 distinct recognized
  multi-category rows/$249,520. `AUDIT.md` gives the exact owner-only
  declared-absence equation. ACS supplies only adjusted combined `SSP`, not the
  four leaf labels
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_inputs.py:133-144,177-195,307-331`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:148-164,186-212,1891-1966`).
- **Taxable IRA — dense-rung refit.** Run/bank the current early family first,
  then apply a single-target weighted two-part overlay trained on the exact
  ASEC clone-0 code-4 label, pattern by pattern, with ACS `RETP` only as a
  predictor. Keep the compatibility bank and every non-IRA early output
  byte-identical, and apply only the original IRA complement. The frozen family
  string is `puf_tax_itemization__clone0_taxable_ira_overlay`; neither its name
  nor derived seeds may be selected after observing f025. This avoids changing
  Social Security retirement, which follows IRA in the same current batch
  (`packages/microcosm-build/src/microcosm/build/us/spec/imputation.yaml:371-514`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2338-2379,2902-2916`;
  `packages/microcosm-fit/src/microcosm/fit/qrf.py:649-662,1151-1230`).
- **Keogh/401(k)/403(b)/SEP — four dense-rung refits.** Keep the internal
  PUF-role QRF for its intended PUF output. The candidate must run the current
  clone-1 family first, then use a source-ID-keyed hybrid donor that replaces
  only these four target labels with clone-0 evidence. Fit/draw Keogh over the
  union of all four recipient complements; then fit `[401(k), 403(b), SEP]`
  over that union with refit Keogh and the unchanged
  compatibility-pass tax-exempt IRA raw draw as fixed prefixes. Both the
  passing target's bank and terminal output must be byte-identical. Apply only
  the four declared recipient masks, preserve producer cells and all controls,
  and persist exact donor masks/sign counts/regimes. Recompute Keogh's regime
  from realized support, retain the existing QRF behavior, and abort only on
  existing validation errors; never add, synthesize, or reweight carriers. The
  frozen family strings are
  `source_operator_retirement_distributions__clone0_keogh_overlay` and
  `source_operator_retirement_distributions__clone0_accounts_overlay`; their
  names may not be selected after observing f025
  (`packages/microcosm-build/src/microcosm/build/us_runtime/puf_support.py:2121-2143`;
  `packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8405-8440,8477-8483,8518-8538,8631-8729`;
  `packages/microcosm-fit/src/microcosm/fit/qrf.py:118-150,649-662,1151-1230,1333-1407`).
- Both refits require the future `REFIT_SHA` to add a typed
  post-compatibility overlay executor. The current transfer derives draw scope
  from target nulls, writes only nulls, starts an empty-prefix chain, and
  rejects duplicate target declarations; it cannot express authenticated
  non-null overlay masks, a wider Keogh draw scope, or the two fixed late raw
  prefixes
  (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:967-1043,1461-1523,2283-2334`;
  `packages/microcosm-fit/src/microcosm/fit/qrf.py:1110-1230`).
  `HOST_25PCT_PLAN.md` freezes a separate ordered overlay registry, typed
  `{family_id, draw_scope, write_masks, fixed prefixes, separate bank}`
  contract, bank schema, exact prefix order, and required resume/mask/identity
  regressions. Compatibility declarations and banks remain untouched.
- **Zero derivation defects.** Every current source equation bit-matches frozen
  clone 0, so there is no 1%-verifiable derivation patch to implement. Fixing
  pension or Social Security without owner decisions would change concepts;
  changing the other five is model fitting that requires f025 evidence.

### Blocked serial-host f025 plan and exact entry points

`experiments/retirement_model_and_data/HOST_25PCT_PLAN.md` is the complete
fail-closed charter. Cold execution is currently blocked: historical f025 and
f001 stages reached 81,026,367,488 and 84,729,479,168 RSS bytes, respectively,
and neither a reviewed sub-14-GiB cold implementation nor an authenticated
candidate resume bundle exists. `REFIT_SHA` and the owner decisions for the
six concept legs also do not exist. Consequently there is no truthful,
authorized executable f025 build command yet.

Once those prerequisites exist, the exact outer command is one literal Darwin
lock held across the refit, removal, and broad runs:

```bash
/usr/bin/lockf -t 0 -k \
  /private/tmp/microcosm-retirement-f025.serial.lock \
  /ABSOLUTE/REVIEWED/retirement-f025-supervisor \
  /ABSOLUTE/REVIEWED/run-authority.json
```

The reviewed supervisor must execute the no-extra-argument f025 argv printed
verbatim in `HOST_25PCT_PLAN.md`, through the fixed 14 GiB guard, after exact
revision/input/authority/checkpoint authentication. It must run in this order:

1. reviewed five-leg `REFIT_SHA`;
2. frozen removal commit `1a8ad451c6eff17d405ef75cbdd014de72447153`;
3. frozen broad commit `539e415defb27bf103a40081239f123ce9d76c6d`.

The exact standalone Phase-P reference recovery and candidate invocation are:

```bash
git -C /Users/maxghenis/PolicyEngine/_worktrees/microcosm-arm-split \
  show 9d6eecb4:tools/predictor_set_oos_gate.py \
  > "$PHASE_P_REFERENCE_RUNNER"
test "$(/sbin/sha256sum "$PHASE_P_REFERENCE_RUNNER" | awk '{print $1}')" = \
  319cd4292577c5c88322fcfe6205e176a354094950b98ca45047f7f2bb320ca1

"$WORKTREE/.venv/bin/python" "$PHASE_P_CANDIDATE_RUNNER" \
  --baseline /Users/maxghenis/PolicyEngine/_buildo-runtime/out/stacked-f025-r1 \
  --candidate "$OUT_DIR" \
  --output "$OUT_DIR/phase_p/verdict.json" \
  --markdown-output "$OUT_DIR/phase_p/report.md"
```

These commands remain blocked with the build. The host charter requires an
atomic all-exit/signal status writer, full artifact/authority/target-bank
authentication, baseline-versus-refit and baseline/removal/broad 16-row
ledgers, unchanged Phase-P controls and exact 32/118/74/211 completeness. It
does not authorize an exclusion, control tuning, a cold build above the memory
limit, or acceptance based only on a process exit code.

### Final exact-tree validation

- Frame, data, calibrate, and fit package-directory suites returned zero. Their
  guarded aggregate/individual peaks were respectively
  `6,995,214,336/6,974,128,128`,
  `11,917,328,384/11,896,111,104`,
  `498,696,192/477,511,680`, and
  `863,649,792/821,182,464` bytes. No descendant escaped and every process
  group was empty.
- The 259 tracked build test files were sorted once and partitioned into the
  six accepted direct-pytest ranges below. Raw accepted command count, unique
  accepted command count, and tracked file count are all exactly 259; the set
  difference is empty.

  | Files | Trace | Return | Aggregate / individual peak bytes | Escaped test sessions | Empty group |
  |---|---|---:|---:|---|---|
  | 1–50 | `/private/tmp/retirement-final-build-batch1-allow.trace.jsonl` | 0 | 4,941,611,008 / 4,908,285,952 | 96949, 98572, 99070 | yes |
  | 51–100 | `/private/tmp/retirement-final-build-batch2.trace.jsonl` | 0 | 2,194,948,096 / 2,173,763,584 | none | yes |
  | 101–139 | `/private/tmp/retirement-final-build-batch3.trace.jsonl` | 0 | 726,106,112 / 695,779,328 | none | yes |
  | 140–180 | `/private/tmp/retirement-final-build-batch4.trace.jsonl` | 0 | 10,059,890,688 / 10,017,013,760 | 43150 | yes |
  | 181–220 | `/private/tmp/retirement-final-build-batch5.trace.jsonl` | 0 | 7,517,306,880 / 7,474,413,568 | 53856, 54306 | yes |
  | 221–259 | `/private/tmp/retirement-final-build-batch6.trace.jsonl` | 0 | 10,281,369,600 / 10,238,705,664 | none | yes |

- All disclosed escaped test-session PIDs were included in the guard's RSS and
  kill set; direct `kill -0` checks after completion report every one gone. The
  first range-1 host-default attempt correctly stopped the intentional escape
  PID 89150 (`monitor_error`, return `-9`, 2,293,039,104-byte aggregate peak,
  empty group). That discarded supervisor receipt is not pytest evidence. The
  identical range passed under the guard's explicit test-only escape tracking.
- The canonical audit generation and independent freshness check both returned
  zero, reproduced the two small-support QED failures, and left empty process
  groups. The JSON SHA-256 is
  `37e92f7358119c44670c104335d9452a8a4e9e22f28627a70c589691e4dc92bf`;
  the script SHA-256 is
  `a12a8f3192017a74738f5344b4c25dd6174363f834f25de9ef28ca12be781378`.
- Final repository-wide `uv run ruff check .` returned zero. Its fixed-guard
  trace records a 46,776,320-byte aggregate peak, a 25,559,040-byte individual
  peak, no escaped descendant, and an empty process group.

### Next

- Owner adjudication, a reviewed five-leg `REFIT_SHA`, and a demonstrated
  memory-safe cold path or authenticated exact resume are required before the
  serial host queue can run the blocked f025 charter.
- No push, pool build, exclusion, control tuning, publication, or chain
  operation is authorized from this lane.
