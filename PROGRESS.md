# Progress: retirement model and data audit

## State

The `battery-retirement-model-data` lane has completed the frozen
source/label/carrier/amount audit for the adjudicated 16 red checks / 11
retirement legs. No pool build has run. The machine proof classifies 6 concept
mismatches, 5 dense-rung refits, and 0 derivation defects; therefore no safe
production derivation fix or exclusion was made. Mandatory baseline validation
is green across all five package shards plus Ruff under the fixed 14 GiB guard.
It required four unrelated baseline-enabling repairs:
fresh-interpreter imports are lazy and identity preserving
(`packages/microcosm-build/src/microcosm/build/us_runtime/__init__.py:1072-1085,1989-2001`;
`packages/microcosm-build/src/microcosm/build/us_runtime/spine_agreement.py:325-328,953-980`),
the mounted SIPP vehicle and voluntary-filing parsers are bounded
(`packages/microcosm-build/src/microcosm/build/us_runtime/sipp_vehicles.py:368-430`;
`packages/microcosm-build/src/microcosm/build/us_runtime/voluntary_filing.py:350-418`),
and the IMDB archive fidelity test streams its source
(`packages/microcosm-build/tests/test_us_trade_imdb_goldens.py:209-238`).
Exact full-donor and archive regressions preserve the prior values and
identities. The reproducible f001 proof now verifies all 16 raw ASEC source
columns through assembly and transfer, all 11 clone-0 equations, actual donor
roles/sign support/regimes, final pool/gate equality, and ACS disagreement
entry points. The serial-host f025 charter is complete but explicitly blocked
pending a reviewed sub-14-GiB cold implementation or exact resume authority,
the five-leg refit commit, and owner decisions for pension/Social Security.
Final exact-tree validation is green. The audit/handoff is complete; no further
lane action is authorized beyond its required local commit.

## Done

- Read the repository rules, the retirement work-package adjudication, the arm
  attribution guardrails, and the retirement-factorial lane's notes and commit
  history before changing implementation code.
- Attempted the required `uv sync --all-packages --extra us`. The default cache
  was sandbox-inaccessible; a writable-cache retry reached dependency download
  and then failed because network/DNS is disabled.
- Located a provisioned sibling environment with the byte-identical `uv.lock`
  SHA-256 (`ea7af7806a0beefe7394adefd5516649f3eba4740ae95ccdaa9aaa252249bc3a`),
  cloned it copy-on-write, and reserved this worktree's five `src` roots for
  test-time `PYTHONPATH` precedence.
- Confirmed the lane starts at clean `origin/main` commit `2aa96795`; no push,
  pool build, gate tuning, exclusion, or chain operation has occurred.
- The first `packages/microcosm-build/tests` baseline attempt reached 5,853
  passes / 37 skips before three atomic-publication crash tests timed out while
  their fresh interpreters eagerly constructed the unrelated PolicyEngine-US
  ABI registry. Exact-lock environments, warming, direct Python, and an idle
  serialized reproduction all exceeded the unchanged 60-second child budget.
- Removed only the eager `spine_agreement` re-export from the broad US-runtime
  initializer. All ten public names remain lazy, discoverable, and identity
  preserving. The agreement module now also caches its typed canonical registry
  on first use, after the recursively inspected pool functions exist. Focused
  identity/default-gate tests pass; no timeout or publication mechanism changed.
- Copied the already-reviewed retirement guard into the assigned experiment
  directory and hardened its shutdown against recycled PIDs outside the owned
  process group. Its host-default synthetic child/grandchild allocation test
  reaches the test RSS limit, observes both processes, kills them, and proves
  the process group empty. A separate test-only opt-in tracks an intentionally
  escaped child session through clean exit; host commands will not pass it.
- Recorded the discarded full-suite 13 GiB stop at 78%: no test had failed,
  peak aggregate RSS was 13,969,653,760 bytes, peak individual RSS was
  13,964,328,960 bytes, and the process-group-empty postcondition passed.
- Isolated the mounted full-file SIPP vehicle donor regression under the same
  guard. It passed with a 13,064,765,440-byte aggregate peak, a
  13,043,515,392-byte individual peak, a 0.155-second maximum sample gap, and
  an empty process group. This identifies expected input loading, not a test
  failure, as the full-run spike.
- Reconciled the guard with the already-written retirement-factorial owner
  charter: it requires a fixed 14 GiB individual-or-aggregate stop as margin
  below 15 GiB. The guard now uses that exact stop, offers no runtime limit
  override, and will not be raised again. Scientific gates, statistical bands,
  ceilings, folds, and seeds remain byte-for-byte unchanged.
- Kept the fixed 14 GiB stop when the next full run reached 77–78% and stopped
  at 15,197,569,024 aggregate bytes (15,191,785,472 individual bytes), with no
  pytest failure and an empty process group. The loader retained its list of
  copied December slices, final raw chunk, and reader while allocating the
  concatenated frame's downstream groupby surfaces. It now closes the chunk
  reader before concatenation and releases those source temporaries before any
  numeric derivation or groupby. No source value, row predicate, chunk size,
  test assertion, scientific control, or resource limit changed.
- The lifetime-only exact-donor retry was itself stopped at 15,056,240,640
  aggregate bytes, proving that reader retention was not the entire peak. The
  remaining mechanism is the wide-file parser batch: only 20 columns are
  selected, but pandas tokenizes and infers them across each requested chunk of
  the 3.73 GB source. The default batch is now bounded at 25,000 rather than
  100,000 rows. A fixture regression proves exact output, dtype, and ordering
  invariance across 1-row, 10-row, and default chunks. This intermediate
  bounded-batch form still used `low_memory=False`; the later exact-donor
  audit superseded it with bounded tokenizer inference at
  `packages/microcosm-build/src/microcosm/build/us_runtime/sipp_vehicles.py:393-430`.
- Repeated the exact mounted 3.73 GB donor regression after bounding the parser.
  It passed with the same pinned support assertions, a 6,220,431,360-byte
  aggregate peak, a 6,199,197,696-byte individual peak, a 0.052-second maximum
  sample gap, no escaped descendant, and an empty process group. This is more
  than 6.8 GB below the first passing isolated aggregate peak.
- Stopped a non-green full run after it emitted two failures and six errors at
  13%. `SIGINT` arrived during the supervisor sleep, outside the iteration's
  error handler, so the guard exited without a final receipt. The independently
  resolved root process group and all three last-observed PIDs were already
  gone, but the missing fail-closed receipt was a real defect. The sampling
  sleep now executes inside the monitored exception boundary, so a handled
  signal records the error, kills the selected process tree, and reaches the
  final postcondition.
- Reproduced that interrupt path with a synthetic child: the guard exited 97,
  recorded `monitor_error:GuardInterruptedError:received signal 2`, killed the
  child (return code -9), observed a 13,778,944-byte peak and 0.022-second
  maximum sample gap, and proved the process group empty.
- Captured the early assertion with `pytest -x`: the SIPP source edit changed
  the shared implementation inventory, so the pinned BE and UK spec-envelope
  hashes changed even though their country resources did not. The canonical US
  generator `--check` passed unchanged (`f66894e0...`). A guarded digest run
  repinned BE to `2fe12821...` and UK to `6e652647...`; its aggregate peak was
  417,480,704 bytes and its process group was empty. The two-country compile
  matrix and the byte-for-byte US generator regression then passed (16 tests),
  under a 3,112,632,320-byte guarded aggregate peak with an empty process group.
- Audited the cloned venv's editable path files before the mandatory rerun.
  They point at the exact-lock sibling worktree, so every subsequent command
  explicitly prepends all five source roots from this lane (`build`,
  `calibrate`, `data`, `fit`, and `frame`). The earlier focused build tests
  already selected this lane's changed build code; the corrected declaration
  now also excludes sibling source from every unchanged shard.
- Under that complete source isolation, the next fail-fast error was the
  reviewed inventory's source-derived seed attestations. A guarded canonical
  recomputation moved the seed protocol to `e1b2dc75...`, its compiled owner map
  to `f490677f...`, and confirmed the US spec at `f66894e0...`. The repository
  coverage generator rewrote only those digest fields and the already-moved US
  spec binding; all 41,379 configuration fields and 40 inventory checks remain
  covered. All seven coverage-tool tests pass.
- The minimal semantic-envelope golden vector moved to `c6d3b112...` for the
  same source-inventory reason. After that mechanical repin, all eight loader
  tests pass, followed by the complete 280-test spec-engine and US-generator
  sweep. Its guarded peak was 5,013,291,008 aggregate bytes, with no escaped
  descendant and an empty process group.
- The next fully isolated full-shard attempt passed 62% before fail-fast found
  one old `586491f0...` US spec binding in the constants-adapter assertion. The
  guard completed normally with pytest exit 1, a 12,179,341,312-byte aggregate
  peak, an empty process group, and three disclosed intentional test-session
  escapes. Repository-wide search found exactly two copies in the same tool
  test file; both now use the already-proven `f66894e0...` binding.
- Both direct consumers pass, followed by all 165 multispine tool tests. The
  latter sweep completed at a 2,663,579,648-byte aggregate peak, with no escape
  and an empty process group. The superseded US spec hash no longer occurs in
  the repository.
- A later full-shard retry reached the mounted SIPP regression with passes and
  expected skips only, then the fixed guard stopped it at 15,108,767,744
  aggregate bytes (15,084,699,648 individual bytes). This was another resource
  receipt, not a pytest failure, and the process group was empty.
- Preserved the complete pre-parser-change donor frame and canonical digest
  `12388b83a9d8f5fbd59bb1f7bedf21a00faca389a6518f36dd398fc3e544cd6c`.
  The loader still validates every contract header, including unused `PNUM`,
  but no longer parses that column, drops `MONTHCODE` at December selection,
  uses pandas' bounded tokenizer inference, and normalizes each retained slice.
  Mixed-token fixture output is exactly invariant across 1-row, 10-row, and
  default chunks, and a dedicated regression proves `PNUM` remains required.
- Loaded the exact mounted source after the parser change and compared its
  complete output directly with the saved pre-change frame using exact pandas
  equality. It matched, retained digest `12388b83...`, peaked at only
  494,583,808 aggregate bytes (473,382,912 individual bytes), disclosed no
  escaped descendant, and left an empty process group.
- Mechanically refreshed the implementation-derived hashes to BE `f565b00f...`,
  UK `33bd40d1...`, US `6d305cbd...`, seed protocol `5e446a76...`, seed map
  `d5e006a7...`, and minimal loader vector `7c9e4853...`. The US generator
  remained byte-clean, coverage regenerated at 41,379/41,379 fields and 40/40
  inventory checks, and all 33 direct attestation consumers pass under a
  3,554,951,168-byte aggregate peak with an empty process group.
- The complete build shard passed 5,956 tests (97%) before the fixed guard
  stopped the active second IMDB archive-fidelity case at 15,206,301,696
  aggregate bytes. It was a resource stop, not a pytest failure, and the
  process group was empty. The preceding mounted SIPP regression had already
  passed in-process, after which aggregate RSS fell to 11,531,567,104 bytes.
- The archive test formerly copied each ZIP, inflated an entire member, decoded
  it, and split millions of strings merely to compare a handful of golden
  lines. It now hashes the archive as a stream, opens the path directly, scans
  each member line by line, checks the declared full line count, and retains
  only declared golden rows. Both exact cached archive cases pass in 2.50
  seconds at a 446,431,232-byte aggregate peak, with an empty process group.
- The next complete run passed both the SIPP vehicle and streamed IMDB cases,
  then stopped in the mounted voluntary-filing donor at 5,963 completed tests
  (97%). Peak aggregate RSS was 15,190,982,656 bytes; there was no pytest
  failure, and the process group was empty. The same unoptimized loader also
  stopped in a fresh process at 15,053,864,960 bytes.
- Preserved the voluntary-filing transform using its pre-existing 25,000-row
  execution control: 22,296 rows, eight columns, all audit attributes, and
  canonical digest
  `464b1a76504481d3a6d5bc87834ea83f85b7a57602385e25634e3e598fd8f4b8`.
  The default now uses that batch size, bounded tokenizer inference, a closed
  reader, and drops `MONTHCODE` after selection. Mixed numeric tokens are exact
  across 1-row, 100-row, and default chunks. The mounted default is frame- and
  attrs-equal to the preserved output and peaks at 521,519,104 aggregate bytes.
- Recomputed all six implementation-derived hashes after the voluntary edit;
  each remained unchanged. The canonical US generator and committed coverage
  report both pass byte-for-byte checks.
- The first immutable full build-shard validation session was externally sent
  `SIGTERM` at 56% by the 89-minute execution-session lifetime. The guard
  failed closed with exit 97, a 12,140,625,920-byte aggregate peak, disclosed
  two intentional test-session escapes, killed the tree, and proved the final
  process group empty. This is an infrastructure receipt, not green evidence.
- Re-ran all 259 sorted `packages/microcosm-build/tests/test_*.py` files in six
  disjoint direct `uv run pytest` batches so no test was omitted and no batch
  exceeded the execution-session lifetime. All six returned zero; their peak
  aggregate RSS values were 5,064,916,992, 2,195,652,608, 736,870,400,
  8,398,045,184, 7,078,264,832, and 9,061,023,744 bytes. Every final process
  group was empty. A discarded shell-wrapped attempt received status 143 in a
  process-control test; rerunning the identical fourth slice with `uv` as the
  guard's direct child matched the original full-shard topology and passed.
- The complete frame, data, calibrate, and fit package-directory suites had
  already returned zero on this exact code snapshot, with peak aggregate RSS
  of 7,012,007,936, 11,890,933,760, 498,663,424, and 913,031,168 bytes,
  respectively. Ruff and `git diff --check` pass on the commit candidate.
- `origin/main` advanced to `057dd95f` while the frozen lane remained based at
  `2aa96795`. No merge was performed because the supplied f001 artifacts and
  source audit are bound to the lane base; no push, pool build, gate tuning,
  exclusion, publication, or chain operation occurred.
- Added a deterministic, hash-bound frozen-artifact proof at
  `experiments/retirement_model_and_data/audit_frozen_artifacts.py`. It verifies
  the 16-check/11-leg adjudication; exact raw→assembled→transferred carriage of
  all 16 ASEC source columns; bitwise source→clone-0 equality; actual donor
  clone, sign counts, and recomputed regime; baseline/pkg3 gate and target-array
  equality; target-bank identities/raw draws; terminal amount/carrier/quantile
  facts; and native ACS `RETP`/`SSP` disagreement.
- The canonical `f001_audit.json` classifies tax-exempt/taxable private pension
  and four Social Security components as 6 concept mismatches; taxable IRA,
  Keogh, 401(k), 403(b), and SEP as 5 dense-rung refits; and exactly 0
  derivation defects / exclusions. All 11 current equations reproduce clone 0,
  so no 1%-verifiable production derivation patch exists.
- Wrote the exact owner-only declared-absence equations for pension and Social
  Security. The pension proposal marks 56 rows/$1,804,558 ambiguous; the
  Social Security proposal marks 35 positive rows/$744,734 ambiguous. Neither
  equation nor an exclusion was implemented.
- Wrote `HOST_25PCT_PLAN.md` with the exact five-leg refit, frozen removal/broad
  revisions, immutable input/authority bindings, literal Darwin lock entry,
  atomic all-exit status prerequisite, standard no-extra-argument build argv,
  refit/factorial 16-row ledgers, and exact Phase-P runner recovery/invocation
  and completeness checks. Execution is explicitly blocked because historical
  RSS is over 80 GB and no reviewed memory-safe cold path or exact candidate
  resume bundle exists; the refit commit and concept-owner decisions also do
  not yet exist.
- The corrected final audit generation returned zero under the guard at
  457,834,496 aggregate / 436,682,752 individual RSS bytes. Its independent
  freshness check also returned zero at 453,787,648 / 432,586,752 bytes. Both
  observed no escaped descendants and left empty process groups. The proof
  uses the battery's pinned five-carrier QED minimum, so it independently
  reproduces the f001 dependent/survivor distances `0.917093391855645` and
  `0.2780886784330607`. The canonical JSON SHA-256 is
  `37e92f7358119c44670c104335d9452a8a4e9e22f28627a70c589691e4dc92bf`.
- Final exact-tree frame, data, calibrate, and fit package suites returned zero
  under the fixed guard at peak aggregate RSS of 6,995,214,336,
  11,917,328,384, 498,696,192, and 863,649,792 bytes, respectively. Every
  process group was empty and no descendant escaped.
- All 259 tracked build test files were run exactly once across six accepted,
  disjoint direct-pytest ranges. Their return codes were all zero and aggregate
  peaks were 4,941,611,008, 2,194,948,096, 726,106,112, 10,059,890,688,
  7,517,306,880, and 10,281,369,600 bytes. Every group was empty. Intentional
  test-session escapes in ranges 1, 4, and 5 were included in RSS, disclosed,
  killed, and independently verified gone.
- The first range-1 attempt used the host-default escape refusal and therefore
  stopped at the suite's intentional detached-session regression (return `-9`,
  2,293,039,104-byte aggregate peak). It is a discarded process-control receipt,
  not a pytest failure. The unchanged range then passed with the explicit test
  validation opt-in. Exact accepted-command union, unique union, and tracked
  build file set are all 259 with an empty set difference.
- Final repository-wide `uv run ruff check .` returned zero under the fixed
  guard at 46,776,320 aggregate / 25,559,040 individual RSS bytes, with no
  escape and an empty process group.

## Next

- The owner must adjudicate the six declared concept absences, review a
  five-leg `REFIT_SHA`, and provide either a demonstrated sub-14-GiB cold path
  or an authenticated exact candidate resume before the serial host queue may
  execute `HOST_25PCT_PLAN.md`.
- Do not push, build a pool from this lane, add an exclusion, or change any
  frozen gate, band, ceiling, fold, seed, or comparator.
