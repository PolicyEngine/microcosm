# Final report: microcosm #462 register alignment

## Outcome

Completed the split-PR remediation on `loss-contract-alignment`, based on
`origin/main` at `7b6e10b`. The change is now register alignment only: one
shared critical-target register, one shared congressional-district classifier,
two consumers, builder contract-row gating, and behavioral containment of the
publish contract.

The critical-row loss multiplier was removed entirely per
[microcosm#492](https://github.com/PolicyEngine/microcosm/issues/492). There is no
constant, CLI option, validation, loss overlay, telemetry, diagnostics/scorer
provenance, or historical replay pin left. `_fiscal_target_loss_weights` is
source-identical to `origin/main`, and its output therefore preserves main's
bit-level behavior for the same registry and family multipliers.

## Sol round-1 findings

1. **Table 1.4 selector parity:** removed the builder-only
   `accepted_name_prefixes=("irs_soi.",)` constraint. The adapter now has
   exactly the shared requirement's substring and suffix selectors. The
   outside-prefix reproduction is builder-rejected.
2. **Congressional-district parity:** added exported, stdlib-only
   `is_congressional_district_target(name, metadata)` and made the publisher
   and builder classifiers thin wrappers. It ORs layout dimension, source-id
   token, geography level, geography scope, truthy CD GEOID, and name token.
   The builder's exact/semantic, Table 1.4, and zero-support paths now see the
   same registry metadata.
3. **Recorded relative-error shape:** a matched row with missing/`None`
   `relative_error` now fails with the publish-contract message instead of
   silently passing after recomputation. Existing non-numeric and stale-value
   checks remain.
4. **Behavioral anti-drift:** the load-bearing test now runs adversarial rows
   through both consumers for exact-name, family+role, Table pattern,
   missing/non-finite values, and a disallowed incumbent escape at the 0.25
   hard stop. A production Ledger compile supplies six separate CD evidence
   rows; builder and publisher exclude identical six-name sets and counts.
   Field comparisons remain as fast checks, and any added conjunctive prefix
   is proven to trip the guard.

The [#490](https://github.com/PolicyEngine/microcosm/issues/490) medical 0.25
adjudication tolerance and its adjacent comment in `us_critical_targets.py`
remain byte-for-byte unchanged, as required.

## Reproduction receipts

The Table 1.4 prefix reproduction now returns:

```text
SOI Table 1.4 national dollar fit failed: other.table_1_4.all.bad_amount@2024: relative_error=1 exceeds 0.25 for SOI Pub 1304 Table 1.4 national dollar rows (soi_table_1_4_national_dollar_rows); target=100.0, final_estimate=200.0.
```

The missing-relative-error reproduction now returns:

```text
SOI Table 1.4 national dollar fit failed: irs_soi.ty2023.table_1_4.all.adversarial_amount@2024: missing recorded relative_error; the publish contract requires a numeric value.
```

The CD reproduction has the owner-mandated exclusion result:

```text
builder_excluded=True
publisher_excluded=True
builder_failures=[]
```

Calling that row "rejected" would contradict the required OR-union exclusion
semantics. The two malformed critical rows are rejected; the CD row is
symmetrically excluded by both consumers.

## Verification

The requested suite ran with `UV_NO_SYNC=1` to use the already-synced workspace
environment in the network-restricted sandbox:

```text
uv run --package microcosm-build --extra us --group dev python -m pytest packages/microcosm-data/tests packages/microcosm-build/tests/test_us_fiscal_refresh_builder.py packages/microcosm-build/tests/test_us_state_files_scorer.py -q
264 passed, 3 skipped (267 collected)
```

Additional receipts:

- Complete `test_gates.py`: passed.
- Required multiplier grep: zero Python hits.
- Ruff check: clean on all ten touched Python files.
- Ruff format check: clean on the eight non-exempt touched Python files; the
  two historical experiment files were not reformatted, as instructed.
- `git diff --check`: clean.
- The medical adjudication block compares byte-for-byte equal to pre-fix
  commit `068854d`.
- Pytest emitted non-failing macOS temporary-directory cleanup warnings; no
  test failed.

## Remediation commits

- `5077f95` — start microcosm#462 Sol remediation progress.
- `c48ba37` — remove the microcosm#462 loss multiplier per microcosm#492.
- `afa910a` — fix Sol finding 1 selector parity.
- `89f74f4` — fix Sol finding 2 CD classifier parity.
- `77040fb` — fix Sol finding 3 relative-error shape.
- `bad7145` — fix Sol finding 4 behavioral containment.
- `3c96514` — apply the finding-2 classifier's required Ruff formatting.

Nothing was pushed at the time of this report; the branch was subsequently
pushed and merged as #491 (2026-07-22).

The sandbox rejected writing
`/Users/maxghenis/PolicyEngine/_reviews/sol-491-fix-out.md` with `Operation not
permitted`; the full completion report is therefore committed here and will be
printed to stdout as the requested fallback.

---

# Final report: F1 continuation r4 (2026-08-20)

## Outcome

**Stopped honestly before deliverables 5 and 6.** Cached `origin/main` and the
cached final #698 ref were merged, the 72-site identity correction is complete,
deliverable 7 is complete in its parallel lane, and the tracked code suite is
green. The live remote could not be fetched or independently verified because
sandbox DNS failed. Implementation and validation work is committed locally;
this report is part of the final handoff commit. Deliverable 4 is not complete:
bundle mode constructs typed authorities and provenance surfaces, but the
production pool driver still dispatches physical stages through constants and
does not invoke the sealed artifact comparator. A D5/D6 PASS would therefore
be false.

No fixture or sample build ran. The four 1% builds and kill/resume leg were not
started, no certification evidence files were created, nothing was pushed, and
`logbook-pending-chain.txt` was not touched.

## Merge and correction result

The required fetch was attempted first but failed on sandbox DNS. Cached
`origin/main` at `164027e2` was merged in `35fb3ed0`, then cached final #698 at
`c4e1eb7f` was merged in `da45dfcd`. Conflict resolution preserved main's
integer schema variants, narrowed exception guards, and scoped archive work
while retaining this lane's 72-site seed ledger over #698's interim 53-site
ledger. The required `uv sync --all-packages --extra us` then completed.

Commit `5875be22` corrected the remaining F0 defects: finalizer JOIN semantics,
late-transfer effects, exact cap-site draw/fill predicates, source seed
materials, and five deterministic-hash classifications. The final protocol is
72 sites, 57 owners, and 131 bindings; coverage is 41,911/41,911 fields and
40/40 inventory checks.

Final pins:

| Surface | Identity |
|---|---|
| compiler ABI v5 | `72659ec091a611e3ca63b0187d27249c817ed29b72f851e192f7f7c03bc1745a` |
| seed protocol | `fd22ba3ab69bc88eb5336261104e4b3d38f721521b4e2bbb04e8ddfa773c130e` |
| compiled seed map | `f79d1646f01ad73a991433ebd0b2d6e5625ccabca682f55476a7b9ebfc6e3b30` |
| normalized graph nodes | `f0d1341b2077da85698ce9993497a366bcbb3fdda0b3721509fafcce7f82b64a` |
| BE `spec_sha256` | `1c5a9449438e1bd30e17105d954f3c5375d20564456be1d005dcab6a798d4aaf` |
| UK `spec_sha256` | `1de6c5f462370cafd5acf0a861dc29603b0e2cb18d6a5c54318114fa08df5ad5` |
| US `spec_sha256` | `d0e4d3c1b3f055dde1056d75837384d4464478be8e2014370aab45ac4a7e8faa` |
| minimal-loader golden | `aee96b0afefbbc4776f5dcb0c3e268e2be4351e535314fa6d49025c416e1ab16` |

The complete pre-merge→final vector and the explicitly superseded #698
53-site interim vector are recorded in `_F1-LANE-NOTES.md` so neither owner
branch can be re-pinned to the smaller ledger accidentally.

## Validation

- Generated US bundle check: pass.
- F0 coverage check: 41,911/41,911 fields; 40/40 inventory checks.
- F0 correction suite: 101 passed.
- Merge/D7 integration suite: 229 passed.
- D7 exact-cell suite: 70 passed; the latest D7 verification receipt is
  `c0a75253`.
- Complete spine-blindness module: 495 passed after the reviewed inventory fix
  in `12df8c45`.
- Clean full-suite rerun: 7,262 tests collected, 100% reached, exit 0, expected
  skips only.
- Repository-wide Ruff: pass. The sole mechanical import-order fix is
  `030c0613`; the complete pool-tool test module exits 0 afterward.
- Whitespace check: pass.

## Why D4–D6 are not complete

- Production has no call to `spec_engine.executor.execute_node`; the typed
  `USPoolKernelAuthorities` object is not threaded into physical execution.
- The correct seam requires per-node dispatch for 38 compiled nodes/20 kernel
  references, including mixed NONE/JOIN/EXPAND structural contracts.
- Physical stochastic callsites still create private RNGs from integer seeds
  instead of consuming production broker stream tokens.
- Source consumers, QRF sinks/processes, and transfer target banks are not all
  enclosed by the required brokers.
- The production path lacks the closed kernel registry, narrow Frame
  projection/patch merge, row classifier, virtual receipt codec, per-node
  journal, and persisted node-reuse map.
- The production collector/comparator is not wired, exact H5/bank member
  inventories are not compiler-sealed, and calibration ownership remains
  outside the execution graph.

Consequently, no stage-by-stage fixture/1% digests, timings, or RSS rows exist.
The D6 within-mode, cross-mode, and resume predicates are **NOT RUN**. The
certification JSON/Markdown files were intentionally not created.

The host ceiling is also independently decisive: four recorded cold 1%
primary-QRF peaks were 78.91–96.95 GiB RSS, versus the 20 GiB lane limit.
Additional host RAM does not relax that per-process limit.

## Handoff

There is no valid owner 25% command yet. The current bundle command would
still execute physical stages through constants and omit the production
comparison gate. A future continuation must first close the physical
executor/broker routing, sealed artifact and calibration inventories, and cold
D4 fixture gate; then make the 1% path keep every process below 20 GiB RSS (or
obtain explicit owner authority to change that constraint) and use isolated
constants, bundle, and resume namespaces. Only after those receipts pass can
an exact f025 command and full expected-identity vector be issued.

For any future off-chain run, both the CLI predecessor option and the
`POPULACE_LOGBOOK_PREV_ROW_DIGEST` environment fallback must be absent.

The authoritative detailed handoff is `_F1-LANE-NOTES.md`; current state and
next actions are in `PROGRESS.md`.

### Resumed continuation receipt (2026-08-20)

The owner-requested r4 resumption does not change the outcome: F1 stops
honestly at deliverable 5, with deliverables 5 and 6 **NOT RUN**. The required
first command, `git fetch origin && git merge origin/main --no-edit`, was
attempted exactly; sandbox DNS prevented the fetch and therefore the chained
merge. Cached `origin/main` is still `164027e2`. Final #698 head `c4e1eb7f` is
already an ancestor through `da45dfcd`, including its scoped-archive,
integer-schema, and narrowed-guard content, while the later 72-site ledger
continues to supersede #698's 53-site version. The authorized dependency sync
completed for 100 packages.

All identities were recomputed from the resolved sources and already matched
their pins: compiler ABI
`72659ec091a611e3ca63b0187d27249c817ed29b72f851e192f7f7c03bc1745a`,
seed protocol
`fd22ba3ab69bc88eb5336261104e4b3d38f721521b4e2bbb04e8ddfa773c130e`,
compiled seed map
`f79d1646f01ad73a991433ebd0b2d6e5625ccabca682f55476a7b9ebfc6e3b30`,
and US spec
`d0e4d3c1b3f055dde1056d75837384d4464478be8e2014370aab45ac4a7e8faa`.
The generated-bundle and coverage checks pass at 72 sites, 57 owners, 131
bindings, 41,911/41,911 fields, and 40/40 inventory checks. No re-pin edit was
required.

Fresh production tracing reconfirmed the blocker. Bundle mode still does not
call the executor, typed kernel authorities, artifact collector, or comparator
for physical stages. Exact H5/bank/calibration inventories remain unsealed,
and no production dual-mode fixture, stage receipt runner, four-build runner,
or kill/resume certification gate exists. Four authenticated cold 1% QRF
profiles remain 78.91--96.95 GiB RSS, above the hard 20 GiB per-process bound.
Starting D5 would therefore have violated both its D4 prerequisite and the
host rule.

Current-head validation produced one transparent contention receipt. The
one-shot serial full suite finished with 7,189 passed, 74 skipped, and one
fixed-60-second child-process timeout after 8,044.09 seconds. Source tracing
found no blocking code path; the exact failed test passed unchanged, then all
78 tests in its module passed unchanged in 331.51 seconds. Ruff and whitespace
checks pass. This is recorded as decomposed green evidence, not as a clean
one-shot exit. No test threshold or production code was changed to obtain it.

No pool build, fixture build, 1% run, or larger sample ran; no D5 receipt rows
or D6 evidence files were created; deliverable-7 surfaces were not modified by
this continuation; nothing was pushed; and `logbook-pending-chain.txt` was not
touched. There is still no valid 25% owner command until D4 is completed and
the 1% path is made behavior-preservingly sub-20-GiB per process.

### Verification reprise at `6d91aea9` (2026-08-20)

The continuation was re-entered from committed honest-stop HEAD `cddb6f18` and
independently audited again. The mandated fetch-and-merge command remained the
first action; sandbox DNS blocked the fetch, while the follow-up local merge of
cached `origin/main` reported already up to date. Final #698 `c4e1eb7f` remains
an ancestor, and its integer schemas, narrowed guards, and scoped archive
coexist with the superseding 72-site ledger.

Fresh checks reproduced the pinned state without edits:

- dependency sync: 100 packages checked;
- US generated bundle: PASS at
  `d0e4d3c1b3f055dde1056d75837384d4464478be8e2014370aab45ac4a7e8faa`;
- coverage: PASS at 41,911/41,911 fields and 40/40 inventory checks;
- seed protocol: 72 sites at
  `fd22ba3ab69bc88eb5336261104e4b3d38f721521b4e2bbb04e8ddfa773c130e`;
- compiled seed map: 57 owners and 131 bindings at
  `f79d1646f01ad73a991433ebd0b2d6e5625ccabca682f55476a7b9ebfc6e3b30`;
- compiler ABI v5:
  `72659ec091a611e3ca63b0187d27249c817ed29b72f851e192f7f7c03bc1745a`.

Three separate read-only audits reached the same disposition. The production
driver explicitly leaves physical stage execution on constants; the executor,
aggregate kernel authority, collector, and comparator have no production
callers. RNG/effect brokerage, exact H5/bank/calibration member closure,
node-reuse persistence, the dual/four-build runner, resume-forbid audit, and a
real kill/resume harness remain absent. Source snapshots are now correctly
brokered, narrowing one older blocker, but that does not close D4.

Deliverables 5 and 6 therefore remain **NOT RUN**. Launching the existing 1%
path would independently breach the process limit: its authenticated cold QRF
peaks are 78.91--96.95 GiB against 20 GiB. No build, certification artifact,
threshold change, logbook-chain mutation, D7 edit, push, or publication was
performed. Commit `6d91aea9` records the refreshed state and detailed journal;
this final-report update is the output-file handoff.

### F1 continuation r5 final report (2026-08-20)

The certification runner and exact high-memory-host handoff are delivered,
but Deliverable A is **NOT COMPLETE**. Bundle mode now compiles
`USPoolKernelAuthorities` from the sealed runtime plan and threads typed
authorities through the bundle-only physical call graph while preserving the
constants oracle. Production physical stages still do not dispatch their
compiled nodes through `execute_node`, however, and no non-vacuous full-stage
dual-mode fixture proves exact executor/broker consumption and D4 raw-byte
artifact equality. The executor lacks sealed sink/checkpoint/process effects;
the present projections and RNG grants are insufficient for a byte-faithful
legacy adapter. An experimental stub seam was removed rather than presented
as evidence.

Deliverable B adds `tools/f1_certification_run.py`. Its `run` command claims a
fresh root and launches exactly one constants or bundle build with
`--resume-policy forbid`; its typed receipt validates and seals the full plan
lock, production selector evidence, raw-byte artifact digest vector, source
pins, and zero-resume audit. Streaming collection avoids retaining the whole
encoded vector. Its `compare` command consumes exactly constants A/B and
bundle A/B receipts, recompiles the current plan and selector contract, and
emits `us-f1-certification.json` plus `us-f1-certification.md`. Its
`resume-gate` command writes only the host predicate procedure and starts no
build, kill, or resume.

The implementation deliberately fails closed. Production cannot yet prove
runtime node reuse, an independent exact final-H5 inventory, or calibration
coverage. The four-build comparator is therefore expected to return status 1
and write a structurally valid **FAIL**, even if every observed normative
artifact digest matches. Status 0 would mean PASS and status 2 malformed input
or execution error. Current output must not be described as certification.

Deliverable C is recorded in `_F1-LANE-NOTES.md`: environment setup, the
documentation-only gate, four exact sequential 1% cold-build commands, the
four-receipt comparator, recovery constraints, and expected memory. The prior
authenticated peaks are 78.91, 84.15, 96.28, and 96.95 GiB RSS; they are not
mode/A-B mapped, so every host build needs more than 96.95 GiB plus margin and
the builds must not overlap. The host runs those commands; the owner
adjudicates the verdict.

Committed coherent steps are `1646ee69` (r5 journal), `bc069829` (exact
artifact-bank coverage), `66fad8fa` (bundle physical-authority plumbing), and
`aefa83b8` (fail-closed certification runner). Focused verification passed
78/78 runner/schema/collector/comparator/coverage tests and 19/19 pool-tool
authority/evidence/resume-policy tests. Ruff and whitespace checks pass, and a
real documentation-only `resume-gate` invocation exited 0. An independent
read-only audit found no commit-blocking correctness issue. Two low-risk
spy-test gaps remain documented in the lane journal.

All executed checks were unit or fixture scale. No pool build, 1% rung,
four-build comparison, or kill/resume exercise ran. The repository-wide build
shard was not rerun because its prior authenticated 28.82-GiB peak exceeds this
lane's 20-GiB process limit, so no repository-wide green claim is made. There
was no push, stash, or `PROGRESS.md` operation. This lane stops after the host
handoff as ordered.

### F1 continuation r6 final report (2026-08-20)

Deliverables B and C are complete. The already-committed
`tools/f1_certification_run.py` was revalidated rather than rewritten: it runs
one cold constants or bundle build into an absent root, enforces forbid-resume
semantics, authenticates all six source path/digest pairs, and writes a typed
receipt whose normative artifact vector hashes raw bytes while provenance is
sealed under the plan-lock vector. Its comparator accepts exactly constants
A/B and bundle A/B, checks within-mode determinism, cross-mode equality, and
vector coverage, then exclusively writes `us-f1-certification.json` and
`us-f1-certification.md`. The documentation-only resume gate starts no build.

The current production evidence remains intentionally incomplete: node-reuse
inventory is empty/incomplete, exact final-H5 selector inventory is unsupported,
and calibration inventory is incomplete. The high-memory comparator is
therefore expected to return status 1 and emit a structurally valid **FAIL**,
even if all collected raw-byte digests match. Status 0 means PASS and status 2
means malformed input or execution error. No verdict was fabricated here.

The exact host handoff is appended to `_F1-LANE-NOTES.md` and committed at
`424c4998`. It gives the six source variables and SHA-256 pins, documentation
gate, four cold 1% commands in constants A, constants B, bundle A, bundle B
order, comparator command, exit semantics, and recovery constraints. Historic
primary-QRF peaks are 84,729,479,168, 90,351,255,552, 103,374,684,160, and
104,102,936,576 bytes (78.91, 84.15, 96.28, and 96.95 GiB). They are not
role-mapped, so every build needs more than 96.95 GiB plus margin and the four
processes must never overlap.

The final B/C verification batch passes 72/72 tests in 30.82 seconds, including
all 35 synthetic runner/comparator tests. Repository-wide Ruff, byte
compilation, runner CLI help, whitespace, generated bundle, and coverage checks
pass. The regenerated US spec is
`05edd87390d841c5b444267cd674d8bb15ed518b12577268d2e2c2de82976079`,
with 41,911/41,911 fields and 40/40 inventory checks. The coherent r6 commits
before this report are `9047610b`, `3c76dace`, `7265a88a`, and `424c4998`.

One separate boundary prevents a repository-wide green claim. A 301-module,
fresh-process suite attempt passed its first 26 modules (609 tests, one expected
skip, 2.016 GiB maximum RSS), then the committed brokered primary-QRF fixture
failed. Joblib's parallel path calls `time.sleep(0.01)`, which the physical
broker refuses as ambient clock access. Limiting Loky to one worker avoids that
call but records 16 refused `os.stat` probes of `/sys/fs/cgroup/cpu.max` and
`/sys/fs/cgroup/cpu/cpu.cfs_quota_us`, outside the declared sink. That is
deliverable-A wiring, which this B/C-only order explicitly forbids changing;
the lane preserved and documented it rather than masking it.

All executed work was unit or fixture scale and stayed below 20 GiB. The
initial 22-file uncommitted wiring attempt was audited and reverted file by
file without stash because it was not a coherent green advance; the five
untracked owner charter copies were preserved byte-untouched. No pool build,
1% run, comparator host run, kill/resume exercise, push, or publication was
performed. The host now runs the exact four commands, and the owner adjudicates
the emitted verdict.
