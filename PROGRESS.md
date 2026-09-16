# Publisher compatibility range at source-enrichment certification

Lane: `max/certify-compatible-model-range-20260914`, off `origin/main` at
`18271b28d`. Started 2026-09-14. Everything below the `---` rule at the end of
this section is prior-lane history; see "Root journals are history, not state"
in `CLAUDE.md`.

## State

Pushed as draft PR
[#928](https://github.com/PolicyEngine/microcosm/pull/928). No publication of
any kind: this lane changes producer/validator source only, and builds,
certifies and publishes no artifact.

An adversarial review lane run against this worktree wrote into it while it
worked: a reviewer checked out `origin/main` copies of the changed files for a
byte-for-byte default-path comparison, and a `git add -A` in this session
committed and pushed that reverted tree as `4b0ae3624`, briefly deleting the
feature from the PR and committing a reviewer's scratch test module. The branch
was reset to `8f82d0c6a` and the two intended commits reapplied; `git diff
8f82d0c6a <head>` is now exactly the docs and journal changes, with the feature
and test files byte-identical to `8f82d0c6a`. Review lanes must run in their own
worktree, not this one.

## Problem

`certify_source_enrichment` writes
`compatible_{model,core}_packages = [{"name": pkg, "specifier": "==<tested version>"}]`
over whatever the candidate manifest held (`source_enrichment.py:940-942`), and
`_check_compatibility` then requires exactly that list at every later validation,
including publish preflight (`source_enrichment.py:610-615`). The contract layer
(`contract.py::_check_compatible_package_entries`) and both consumers (Microcosm
`loader.py::_package_certification`, policyengine.py
`provenance/certification.py::validate_release_manifest`) already accept any PEP
440 specifier set that contains the built-with version — so the exact pin is a
producer-tooling choice, not a schema limit. The consequence: each country patch
release moves the binding rather than widening it, and a data release whose H5
bytes are unchanged still needs re-certification.

## Plan

1. Add a validated publisher claim at certification time, recorded in both
   `release_manifest.json` and `source_enrichment.json`, with the default path
   byte-identical to today.
2. Relax the certification-time equality gate to "equals the default exact pin,
   or equals the claim the report records", keeping every other guard.
3. Tests for accept/reject/default/consumer-read.
4. Docs: when a range is legitimate and when it is not.

## Done

- Read both gates, the contract layer, both consumers and the 2026-09-12 dry-run
  report that motivated the change.
- `--compatible-model-specifier` / `--compatibility-claim-declared-by` on the
  source-enrichment CLI and `certify_source_enrichment`, validated by
  `parse_compatibility_claim_requirement` (PEP 508, names the built-with
  package, no URL/extras/marker) and `compatibility_claim_entry` (valid and
  non-empty PEP 440 set, contains the tested version under the consumers' own
  containment, bounded above, accountable declarer).
- The claim recorded in `source_enrichment.json`
  (`compatibility.publisher_claims.model`) and `release_manifest.json`
  (`compatible_model_packages[0]`, `basis: publisher_claim`, `declared_by`),
  cross-checked at every later validation so a manifest widened after
  certification has no declaration behind it.
- `contract.py` refuses a `publisher_claim` basis with no declarer, and a
  declarer with no basis, for every release type.
- 20 tests across `test_source_enrichment.py` and `test_contract.py`;
  `packages/microcosm-data/tests/` 550 passed, 2 skipped; `ruff check .` clean;
  `tools/ci_test_groups.py --verify` ok.
- Docs: a "Declaring a publisher compatibility range" section in
  `docs/us-native-spm-role-source-enrichment.md` with when to use a range and
  when not to, a note in the `microcosm-data` README, and a changelog fragment.

## Second session, 2026-09-14 evening

The lane was re-entered after the first session ended at `a4e7e131c`. Nothing
was rewritten: the feature and test files are untouched, and this session's job
was to confirm the branch rather than extend it. Done here:

- `packages/microcosm-data/tests/` re-run from scratch in the lane venv: 565
  passed, 2 skipped, matching what the PR body claims.
- A second, independent adversarial pass over the branch, run out of two
  detached review worktrees so no reviewer could write into this one (the
  collision recorded above must not repeat): PEP 440 containment parity against
  both consumers, guard-preservation and bypass, wrapper accept/refuse measured
  rather than described, mutation testing of every new guard, a claims audit of
  the PR body and docs, and a blast-radius sweep for anything that reads
  `compatible_model_packages`.

## Third session, 2026-09-14 — second-pass review, part two

The second-pass review (`review-928-r2.md`, written against `7cb8eae6e`)
confirmed the six earlier fixes and left five items. This session applies them,
each with a test that fails before and passes after.

- **M1** — the boundedness guard bounds a claim above and never below, so
  `policyengine-us<2.1` over a 2.0.1 build is accepted and certifies a consumer
  running 0.9.0. Add a lower-bound probe.
- **L1** — `compatibility.narrowed_claims` is written and never read; surface it
  in the validation/preflight output.
- **L2** — the "pass the flags" remediation suffix fires even when the flags
  were passed this run; gate it on the claim being absent.
- **L3** — the narrowing loop calls a Core pin change a narrowed "claim",
  although no producer can declare a Core range; reword.
- **I2** — doc only: a prerelease built-with version cannot carry a range, so
  the exact default pin is the only option there. (Filed as such; the premise
  did not survive measurement — see "Done".)

### Done

- **M1.** A third boundedness probe in `compatibility_claim_entry` asks whether
  the claim still admits `Version(f"{tested.epoch}!0")`. Measured first:
  `Version("0") in SpecifierSet("<2.1")` is `True`, and `False` for
  `>=2.0.1,<2.1`, `~=2.0.1`, `==2.0.*` and `>=2.0.1,<3`. `<2.1` and `<=2.0.5`
  moved from the accepted parameters to the refused ones, a
  certification-level `policyengine-us<2` case was added, and the next-major
  error text now cites `'>=2.0.1,<2.1'` instead of the `'<2.1'` the new probe
  refuses. Four tests failed before, pass after. The probe carries the tested
  version's epoch because a claim may mix epochs: over a `1!2.0.1` build,
  `>=2.0.1,<1!2.1` admits `1!0` while excluding a bare `Version("0")`. The
  first draft justified that backwards — claiming an epoch-0 zero sits outside
  an epoch-bearing claim, when `Version("0") in SpecifierSet("<1!2.1")` is
  `True` — and its test survived replacing the probe with a bare
  `Version("0")`. Corrected after the third pass; the test now uses the
  mixed-epoch claim and kills that mutant.
- **L1.** `recorded_narrowed_claims` reads the record back, and both validation
  (`python -m microcosm.data.source_enrichment` without `--certify`) and
  `microcosm-publish-release --preflight-only` print `narrowed_claims` beside
  their verdict when a bundle carries one; publication repeats it on stderr,
  since reaching publication does not require running the preflight first.
  It reports rather than gates: an absent or malformed record reads as no
  record. Three tests failed before.
- **L2.** The "pass the flags" suffix is gated on `claim_specifier is None`.
  Both branches tested through a re-certification that tightens a declared
  range (`>=1.998.0,<2` → `>=1.999.0,<2`): warns, no suffix.
- **L3.** Message construction moved to `_narrowing_notice`; Core reads "moves
  the policyengine-core compatibility pin". The Core branch turns out to be
  unreachable through `certify_source_enrichment` — the input gate re-runs the
  loader qualification and requires the recorded receipt to equal the runtime,
  so a moved Core version is refused first. Both the wording and that wall are
  now pinned by tests.
- **I2.** The review's premise was wrong and the docs say the accurate thing
  instead. Installed `packaging` 26.2 matches prereleases by default
  (`SpecifierSet.contains` documents it; `Version("2.1.0rc1") in
  SpecifierSet(">=2.0.1,<2.2")` is `True`). The real constraint is ordering: a
  prerelease sorts below its own release, so `>=2.0.1,<2.1` and `~=2.0.1`
  exclude a `2.0.1rc1` build while `>=2.0.1rc1,<2.1` and `==2.0.*` reach it and
  pass all three probes. A characterization test pins all five outcomes; it
  passes before and after.
- `packages/microcosm-data/tests/` 611 passed, 2 skipped; the two named files
  379 passed; `ruff check` and `ruff format --check` clean on
  `packages/microcosm-data` and on every changed `.py`; repo-wide `ruff check`
  clean; `tools/ci_test_groups.py --verify` ok.

### Third adversarial pass

Five read-only reviewers over the five changes (guard correctness, reporting
path, wording and reachability, docs/claims audit, test quality), each finding
put to an independent refuter: 33 raised, 2 survived. Both are fixed.

- **The epoch rationale was a fabricated mechanism, and its test was inert.**
  The comment and docstring said an epoch-0 zero sits outside an epoch-bearing
  claim; `Version("0") in SpecifierSet("<1!2.1")` is `True`. The verifier
  mutated the probe to a bare `Version("0")` and the whole file stayed green —
  the test named for the epoch carry survived dropping it. The carry is
  justified by mixed-epoch claims instead (`>=2.0.1,<1!2.1` over `1!2.0.1`
  admits `1!0` and not `0`), which is the case the test now uses; the mutant
  fails it.
- **The record reached the preflight and not the publish run.** Already closed
  mid-flight, before the pass reported it: publication repeats it on stderr,
  since `tools/publish_release.sh` passes its arguments straight through and
  the runbook's "remove `--preflight-only`" step is a habit, not a gate.

Two refuted findings were worth acting on anyway. The documented probe residue
is now executable on both sides (`<2.1,!=0` admits `0.9.0` exactly as
`>=2.0.1,!=3.0.0,!=99999.0.0` admits `5.0`), so the sentence describing the
guard's limit cannot drift from it. And `--certify` now reports the narrowing
it caused in its own verdict, with `_narrowed_claims` giving all four verdicts
one tolerance so none can report a bundle differently from the others.

### Next

- PR CI. Do not merge; do not mark ready.

## Next

- Whole-workspace run and PR CI to finish; hand to human review. Do not merge;
  do not publish.

# Grouped solver x calibration target snapshots integration - 2026-09-12

Historical note, 12 September 2026: this grouped-lane journal was subsequently
integrated with PR #914's corrected iteration identities and the actual fiscal
host observer. Its 443-test evidence and statements that host wiring is absent
describe that earlier lane. Current source and acceptance are recorded in
[the fiscal-host evidence](experiments/fiscal-target-snapshot-host-20260912.md).

Lane: `microcosm-grouped-target-snapshot-integration-20260912`, branch
`grouped-target-snapshot-integration-20260912`. Base: fresh `origin/main`
`116d46ee9dc2aafdc68259b7c06e4c3462522e8b` (unmoved; it is exactly the shared
base both reviewed heads were cut from). Integrates the exact reviewed heads
`536f1ceefcdafda3cc619c14b4da18e012a7be57` (G, US grouped/fixed-zero Adam) and
`b43369dc49e175803f62020cc1e72fa53926aed8` (S, calibration target snapshots,
PR #914) against the read-only checklist
`grouped-snapshot-integration-review.md`.

Everything below this section is prior-lane history and was accurate when
written; see "Root journals are history, not state" in `CLAUDE.md`.

## State

Complete and local. Four commits: the merge, the grouped instrumentation, the
identity recomputation, and the cross-product controls. Nothing pushed, no PR,
no release action. Root reviews and integrates.

Scope is the shared solver seam only. `graph_fiscal_dense_calibration.py` still
calls `calibrate` without an observer, so this branch emits no snapshot for the
real US fiscal path; that host wiring is a separate step, and no observer
registry, `Node.params` callback or replay-reruns-the-optimizer claim was
invented here. See
[the lane experiment](experiments/grouped-target-snapshot-integration-20260912.md)
for scope, evidence and residual risks.

## Done

- Fetched `origin/main`; confirmed it is still `116d46ee9`, so no main drift
  had to be preserved. Recorded as the integration base.
- Created this worktree on a new branch from that base; fast-forwarded to G,
  then merged S. Verified all four pinned source hashes
  (`solve.py` and `calibrate/__init__.py` on both heads) match
  `grouped-snapshot-integration-review-pins.json` byte for byte before merging.
- `calibrate/__init__.py` auto-merged as a true union: `GroupedUpperBounds`
  export retained alongside every snapshot export.
- `solve.py`: the single textual conflict was the `calibrate()` signature;
  resolved as a union of S's `target_snapshots` and G's
  `grouped_upper_bounds` / `grouped_preserve_zeros` /
  `_post_projection_observer`.
- `test_us_multispine_pool_tool.py`: resolved semantically. G's unrelated US
  integration delta (schema_version 2 PUMA-ladder fixture with joint
  PUMA/tract/CD overlap arrays and per-layer `source` labels) merged cleanly
  and is retained; the only textual conflict was the `spec_sha256` pin.
- The five source-derived identity conflicts
  (`inventory_coverage.py` EXPECTED_HASHES, `us-f0-coverage.json`,
  `test_spec_engine_loader.py` golden, the multispine `spec_sha256`, and the
  calibrate parity `pins.json`) carry a placeholder in this merge commit. None
  of them is an ours/theirs decision: both sides' values describe their own
  tree, and neither describes the merged one. They are recomputed against the
  final merged sources in a later commit on this branch.

## Done (continued)

- Instrumented `_optimize_grouped`: the grouped early return in `_optimize`
  bypassed every hook S added, so grouped runs emitted only a closing snapshot.
  The in-loop emission sits after the progress callback and before `backward()`
  and reads the exact float32 estimate tensor the epoch's loss was computed
  from. Returned weights, trajectory and RNG state are bit-identical with the
  observer on and off, and no evaluation is added.
- Separated retain-best detection from the receipt for grouped runs, so a later
  change that populated a grouped receipt cannot make the reused final emitter
  claim a retained best. The receipt stays empty.
- Recomputed six source-derived identity artifacts against this checkout (the
  five conflicted ones plus the country-bundle digests, which were not
  conflicted because only G had touched them but which move for the same
  reason). No value equals either branch's. The simulate and fit.qrf pins were
  deliberately left alone.
- Added the checklist's cross product to the three existing grouped test files
  rather than a new one, so every case reuses fixtures already there.

## Next

Root's independent review and integration. Open follow-ups, none owned here:
the host observer seam for the US fiscal dense calibration path, a country-scale
cadence choice backed by a real measurement, and the dashboard consumer.

---

# #893 reconciliation to main's amended graph interface (amendments 19 and 20)

Lane: `microcosm-us-launch-verified-lanes-20260910` (the live integration
worktree for PR #893, branch `microcosm-us-launch-integration-20260909`).
Started 2026-09-12 at `069d5ed9a`. Everything below the `---` rule at the end
of this section is prior-lane history; see "Root journals are history, not
state" in `CLAUDE.md`.

## State

Merges done, four graph pieces re-applied, graph package green (518 passed),
fit green (191), spec check / groups verify / ruff clean. Build-side consumer
suites running at the time of this entry; final results in `out.md` §5 and
`experiments/893-reconciliation-amendments-19-20-20260912.md`. Nothing
pushed, no PR, no new branch, `uv.lock` untouched.

## Done

- Read `CLAUDE.md`, the charter's "Interface freeze" at `23ba24770`, PR
  #893's body, the Amendment 19 lane's "Scope" list; diffed the graph package
  against `23ba24770` per file; AST-scanned every non-graph consumer.
- `uv sync --all-packages --locked --extra us --extra uk` exit 0. Baseline
  graph suite: 5 failed / 362 passed (the five the brief names).
- `3010b7788` merge `origin/main`: graph package, fit sources, lock and
  charter resolved to main's bytes; branch-only `attachments.py`,
  `availability.py`, `schema.py` and the two branch-only graph tests removed
  in the merge, to return only where consumed.
- `051357909` merge `23ba24770` (amendment 20, merged from the branch head
  because #912 was still on CI; the dispatcher re-runs `git merge origin/main`
  after it lands — expected no-op for graph/fit/lock/charter).
- `dc621c14c` seed digests re-pinned (the branch's `acs_transfer` and
  `housing_inputs` plus amendment 20's `fit.qrf` move them); coverage report
  regenerated; `--check` exit 0.
- `cff8fbf32` calibrate/simulate H1 pins re-recorded (the branch's solver and
  `Frame.__reduce__` changes move them); parity files 27 passed.
- Pieces re-applied on main's files, each with graph-level tests the branch
  never had: `752ab840f` raw-byte codec (13 US consumers; 8 codec tests),
  `db1b7821a` Frame-metadata store (3 named consumers; 274 passed across the
  store/population/executor/manifest files), `d1019762b` execution states
  (the post-clone geography gate's typed artifact; replaces the amendment-19
  gate refusal and its test; whole graph package 518 passed), `a16eeacf8`
  population observer (4 consumers; 94 passed on the executor + B/F files).
- Dropped for lack of a consumer: lazy retention (`attachments.py`, layer
  08), `schema.py`, `keys._stream_file`, the `_write_node` refactor.
- Layer map computed from `git log --name-only` attribution plus an AST
  import scan with hard / name-hard / soft link classes; written to `out.md`.

## Next

- Record the build-side consumer results in `out.md` §5, commit the
  `experiments/` copy of the report, leave `out.md` uncommitted (it is
  another lane's tracked report; see the memory note).
- For Max: piece C supersedes amendment 19's gate refusal (`out.md` §8.1);
  the dropped lazy retention / `_stream_file` / `_write_node` pieces (§8.2–3).

---

# Amendment 19 — typed opaque artifacts on the graph interface

Lane: `amend-typed-artifacts`, off `origin/main` at `3094bfe84`. Started
2026-09-11. Everything below the `---` rule at the end of this section is
prior-lane history; see "Root journals are history, not state" in
`CLAUDE.md`.

## State

Landed, reviewed, and re-verified from scratch on `amend-typed-artifacts`. The
whole-workspace run is the last command outstanding; every gate the brief names
has been re-run green in this session. Nothing pushed, no PR, no branches
created, `uv.lock` untouched.

> Historicized 2026-09-12: the branch was pushed as PR #911 on 2026-09-11 and
> peer-gated; the whole-workspace run above was superseded by the PR's CI. The
> interface lock is now enforced — `graph-interface-lock-test` merged as #910
> on 2026-09-11 and this branch passes it. The paragraphs below are the lane's
> record as written, not current state.

## Scope (what is in, and what is deliberately out)

In, from `git diff origin/main origin/microcosm-us-launch-integration-20260909
-- packages/microcosm-graph/src`:

- `decl.py`: `ArtifactType`, `ArtifactOutput`, `ArtifactInput`,
  `Node.artifact_inputs` / `Node.artifact_outputs`, their validation, their
  elision from the canonical projection when empty, and the artifact-edge
  arm of `compile_graph`.
- `kernel.py`: `ArtifactValue` and `KernelContext.artifacts`.
- `artifact_edges.py` (new): numeric scope payloads, scope compatibility,
  typed descriptors, `typed_contracts`, `value_from_descriptor`.
- `keys.py`: `opaque_artifact_key` and the `typed_artifacts` term in
  `node_key`.
- `serialize.py`, `view.py`, `manifest.py`, `executor.py`: the minimal
  support for the executor to honour declared artifact inputs/outputs.

Out, because it is not needed for artifacts (each is its own lane):

- `SeedSource.KEYED` and `randomness.py` (`keyed_uniform`).
- `availability.py` / execution state / `unreached` / `blocked_by` /
  `gate_exception` propagation, and manifest schema 4.
- `attachments.py`, `_PopulationRetention`, lazy populations,
  `_population_observer`.
- `store.py` Frame-metadata storage (`microcosm-graph-frame-v2`) and the
  non-finite JSON decode hooks.
- `keys.py` `_stream_file` chunked source hashing.
- `codecs.py` `SourceBytesCodec` / `load_source_bytes`; `schema.py`.
- The `_write_node` per-coordinate memory refactor.

## Done

- Read `CLAUDE.md`, `docs/graph-acceptance.md`, `DESIGN.md`, and the
  amendment-17 precedent (`cdbf71888`, `80b63ba14`, `ed36f6cb3`).
- Measured the branch diff per file and fixed the in/out boundary above.
- `uv sync --all-packages --locked --extra us --extra uk` → exit 0.
- Captured the baseline node keys of the three toy acceptance graphs before
  touching any source, so the node-key answer is measured, not asserted.
- `68a6ecc4b` (red, exit 2, 4 collection errors) → `e591c52d7`: the frozen
  declaration interface, `compile_graph`'s artifact edge, the elided
  canonical projection, `opaque_artifact_key`, serialization, and the view.
- `a2b6dfb0b`: the acceptance suite's B2 `KernelContext` field set, as its
  own commit, matching `80b63ba14`.
- `1cce8eceb` (red, **8 of 8 failing** — the commit message and an earlier
  version of this line both say 7, which is wrong; see the correction under
  "Re-verification") → `15f9d2c67`: `artifact_edges.py` and
  the executor, cache record, and manifest support.
- `38b9a9e4d`: amendment 19 in the charter, the relock, the changelog
  fragment. `8c2e7faab`: the graph explorer's receipt payload.
- Node keys re-measured after the change: byte-identical for all 20 nodes
  of the three toy graphs.
- `packages/microcosm-graph/tests` 359 passed, exit 0.
  `tools/ci_test_groups.py --verify` ok, `tools/spec_engine_coverage.py
  --check` 42156/42156 + 41/41, `tools/graph_acceptance_burndown.py
  --verify` ok, `ruff check` and `ruff format --check` clean.

- Ran a five-dimension adversarial review of the extraction against the
  integration branch (fidelity/minimality, executor paths, identity and
  store, manifest provenance, charter/lock/changelog), each finding put to
  two skeptics. Nine findings; seven real and fixed here:
  `3a0726f93` (a corrupt typed manifest surfaced as `NodeRejectedError`
  rather than `StoreCorruptError`), `9ebb60e4b` (`ArtifactValue.key`
  described as a content identity it is not; relock), `402d9a631` (a
  malformed `gate_ancestry` regressed to a bare `TypeError` on manifests
  with no artifacts at all; charter graph list corrected; the F2 sentence
  split into its two mechanisms), `5c4a8efde` (the artifact miss decision
  moved back inside the recompute fallback), `9018c4420` (the
  identity-preservation claim corrected, and payloads read only on the path
  that runs a kernel). Two were the documented decisions and stand.
- Added coverage the review motivated: manifest ancestry authentication,
  the F2-over-bytes path, cross-version edges under all three resume
  policies, every artifact declaration field being normative, A3 through a
  byte edge, and a cache hit that reads no payload.

## Re-verification, 2026-09-11 (independent of the landing session)

Everything below was re-run from a clean read of the tree, not carried over
from the landing session's notes.

- The node-key answer re-measured with a script that varies only the
  graph-kernel code: `microcosm.graph` resolved once from this branch and once
  from `origin/main`'s sources (shadowed through `PYTHONPATH`, confirmed by the
  loaded `decl.py` hash `635fef92...` on the main run), with `microcosm.build`
  identical in both. Six graphs, 5+5+6+9+41+8 = 74 nodes: **all 74 node keys and
  all 74 canonical projections byte-identical.** The amendment's per-graph counts
  are each correct.
- `docs/graph-interface.lock` re-checked against `shasum -a 256` of the two
  frozen files: both match.
- Re-run green: `packages/microcosm-graph/tests` 370 passed exit 0; the
  acceptance subset 113 passed exit 0; `test_graph_kernel_contract.py` 15 passed
  exit 0; the `KernelContext(` consumers (calibrate/fit/frame `test_kernels.py`
  plus `test_us_graph.py`, `test_uk_graph.py`) 36 passed exit 0.
  `tools/ci_test_groups.py --verify` ok, `tools/spec_engine_coverage.py --check`
  42156/42156 + 41/41, `tools/graph_acceptance_burndown.py --verify` ok,
  `ruff check .` clean — all exit 0.
- No consumer constructs `KernelContext` positionally: all five non-test sites
  use keyword arguments, so the new field's placement could not have broken one.
- `ruff format --check .` exits 1 on 81 pre-existing files, none of them touched
  by this lane (all 17 changed Python files pass `ruff format --check`
  individually). CI's lint lane runs only `ruff check .`, so this is repo drift,
  not a gate this lane moved.
- The red commit `1cce8eceb` records "Red: 7 of 8 fail against the executor as
  it stands", and this journal repeated it. **It was 8 of 8.** Measured by
  extracting the whole tree at `1cce8eceb` (`git archive | tar -x`), pointing
  `PYTHONPATH` at that tree's six shard `src` directories (confirmed:
  `microcosm.graph.executor` resolves into the extract, and
  `microcosm.graph.artifact_edges` has no spec there, so the executor support
  genuinely had not landed), and running the commit's own
  `test_graph_executor.py` against its own sources: **8 failed, 63 passed**, the
  8 being exactly the amendment-19 tests the commit added. Red-first discipline
  holds — the commit was redder than claimed — but the count in its message is
  wrong and stays wrong, because rewriting landed history to fix a tally would
  be worse than recording the correction here.
- An adversarial audit line-traced the new module: eleven non-docstring
  statements of `artifact_edges.py` never executed in the whole graph suite, all
  on the foreign-provenance parsing surface. Closed in `e3f69a4c4` with three
  tests through the public `NodeReceipt`/`RunManifest` surface; the trace now
  reports zero. The same trace showed `run_graph`'s consumer-side receipt
  comparison is unreachable as a refusal — both skeptics confirmed the charter's
  wording claims only the check's ordering, which does execute — so the branch
  is now commented the way this file already marks such guards, rather than
  chased with a test that cannot be written honestly.
- `packages/microcosm-build/tests/test_release_target_parity.py` fails two
  tests locally. **Not this lane, and not CI**: both are guarded by
  `_feed_or_skip` on a 131 MB pinned feed that lives *outside the repository*
  (`~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl`,
  dated 2026-07-23), so CI skips them; the local artifact predates #855's
  hierarchy-label requirement. The same two fail identically with `origin/main`'s
  graph sources swapped in, and `ledger_targets.py` imports no
  `microcosm.graph`. This is the US twin of the UK instance the #791 lane already
  recorded in `experiments/791-household-composition-receipts.md:111`.

## Next

- Whole-workspace `uv run pytest` is the only command still running. Every gate
  the brief names is green at `HEAD`, and the complete set of tests this change
  can reach — the graph package (373) plus the 8 test files outside it that
  import `microcosm.graph` directly or through the seven source modules that do
  (81) — is green at exit 0. The lane report is in `out.md`.
- For Max, in `out.md` §8: the gate-artifact-output refusal is the one interim
  ruling this lane made on his behalf; the interface lock had **no automated
  enforcement** when this was written (the charter's freeze was a human gate,
  which is how the integration branch changed both frozen files unnoticed) —
  the sibling branch `graph-interface-lock-test` (`8bd6e05ec`) added the test
  and merged as #910 on 2026-09-11; this branch passes it, exit 0; plus the pre-existing
  `ruff format` drift and the stale local `_buildh-runtime` feed.

---

# Issue #907 — population `_storage_parts` hashes object dtype by pointer

Lane: `fix-907-population-stamp-object-storage`, branched from
`origin/main` at `295130c9f901e08db11457f16dbdee4e2349c5ba` on 2026-09-11.
Report: `/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/lanes/out-907-build-r1.md`.

## State

Implementation complete on `fix-907-population-stamp-object-storage`; ten
commits, nothing pushed, no new branch, no stash. `decl.py`, `kernel.py`,
`docs/graph-interface.lock`, every `test_acceptance_*`, `uv.lock`, spec pins
and evidence JSON are untouched — `shasum -a 256` on `decl.py`/`kernel.py`
still matches the lock byte for byte.

## Done

- Read `CLAUDE.md` and `docs/shared-constants.md`; confirmed the lane rules.
- Reproduced #907 directly, and end to end: two independently constructed
  equal object-dtype Series give different `_storage_parts` value bytes
  (PyObject addresses), which on `origin/main` surfaces as a **spurious**
  `PopulationError: Structural node 'n1' changed carried storage in
  household.tenure_type` for content that did not change.
- Landed the red regression in `packages/microcosm-graph/tests/`
  (`test_graph_population.py`, an already-tracked file, so
  `tools/ci_test_groups.py` needed no change). Final tests against the pre-fix
  source: 34 failed, 24 passed, 46 deselected; direct exit 1.
- Fixed `_storage_parts` by routing any materialized object array through a
  length-prefixed encoding whose body is `store._encode_object_scalar` — the
  graph package's existing object-leaf codec, the one `ContentStore` writes
  and reads back. A second, parallel vocabulary would have left a column
  unequal to its own persisted-and-reloaded self for `pd.NA`, `pd.NaT` and
  NumPy scalars. Unsupported leaves raise `PopulationError` under the static
  code `storage-object-leaf` instead of being `repr()`-ed.
- Kept the masked, numeric and `StringDtype` branches byte-identical, verified
  by a pre-fix/post-fix byte diff over 20 dtypes and pinned by hex-literal
  characterization tests. Only object, Categorical, DatetimeTZ, Period and
  Interval move — every one of them a dtype whose old bytes were addresses.
- Documented at `storage_equal` what the parts do and do not seal, with
  regressions pinning the masked half, the numpy-normalization, the new
  refusals, and a real `ContentStore` frame round trip.
- Adversarial review (three lenses) found two real defects, both fixed: the
  refusal missed `UnicodeEncodeError` from a lone-surrogate `str`, and two
  deliberate normalizations were undocumented.
- Green: `pytest packages/microcosm-graph` 399 passed; the nine graph-adjacent
  build/frame/fit/calibrate files 578 passed; `test_graph_population.py` 119
  passed; `ruff check .` 0; `ruff format --check` on both changed files 0;
  `tools/ci_test_groups.py --verify` 0; `tools/spec_engine_coverage.py
  --check` 0 (42156/42156 fields, 41/41 inventory); `tools/graph_acceptance_
  burndown.py --verify` 0 (green 41, red 0, missing 0).

## Next

- Human review. One item this lane could not complete: the issue asked for a
  note at `_population_stamp` in `us_runtime/survey_atomic_geography.py`, a
  module that exists only at the tip of the unmerged
  `origin/microcosm-us-launch-integration-20260909`. The verified wording —
  corrected, because the issue's own phrasing was incomplete — is in the lane
  report for whoever owns that branch.
- No pinned digest moves. The three `*_population_sha256` stamps on that
  branch will take new values once this merges; their old values were process
  addresses, they are recomputed on both sides of every comparison, and
  `survey_origin_budget` already excludes them from the persisted identity.

---

# F1 portable worker identity — CI crawl fix

## State

Complete on 2026-09-05 on `f1-portable-worker-identity`, started on
2026-09-04 at `50c9232b7597bd1f47897a3132ed08894fc11d93`. Process memoization,
real session priming, explicit live opt-outs, and all requested verification
are complete. No GitHub network, push, new branch, or stash.
Final report, exact commands, direct exits, and per-file timing tables:
`/private/tmp/microcosm-pr871-ci-crawl/out.md`.

## Done

- Read the guide, prior journals, worker identity/bootstrap/launch, three live
  stacked binding sites, and fixture caches. Used direct source tracing and
  independent agent reviews; the GitNexus skill's tools were unavailable.
- Completed both unchanged baselines before production/test behavior edits:
  identity selection 77 passed, 245 deselected, exit 0, 734.406s wall;
  first 200 tests 200 passed, 122 deselected, exit 0, 2033.172s wall.
  Source-order collection used `-p no:randomly`, then 122 explicit deselections;
  the final selected node was
  `test_primary_refuses_missing_universe_receipt_before_callback`.
- Repaired only ignored `.venv/bin/pytest` after its stale interpreter shebang
  produced direct exit 127. Used the supplied synced Python 3.14.4 environment,
  `uv run --no-sync`, and an external writable UV cache; no dependency sync.
- Committed fail-before memo regressions: 4 failed, 7 passed, 322 deselected;
  exit 1; 30.073s wall. Implemented the memo keyed by raw lock argument and
  both bound fit-control environment values, with an exported/documented clear
  API and shared read-only semantic graph. Execution bindings clone the graph
  so artifact mutation cannot poison subsequent identities.
- Primed real session identities for stacked-spine, H5, pool-tool, and the
  indirectly reached spec-bundle suite. Removed redundant stacked/H5 caches.
  Identity/source/runtime/backend/environment mutation tests explicitly opt out
  and clear before/after; deliberate byte edits within a test clear between calls.
  Preflight/fiscal-refresh tests already stub their live-binding paths.
- Added 14 regression cases covering memo reuse/reset, both environment keys,
  invalid controls and locks, lock separation, artifact-copy isolation, real
  session consistency, opted-out namespace-byte changes, and generator ordering.
  Memo/session checks passed: 13 passed, 322 deselected; exit 0; 64.224s wall.
- Broader call-path audit found spec generation could precede function priming.
  Committed a real ordering regression before the fixture fix: 1 failed,
  25 deselected; exit 1; 129.447s wall. Ordered priming before cached generation;
  the same test then passed: 1 passed, 25 deselected; exit 0; 99.303s wall.
- Same post-fix identity selection: 90 passed, 245 deselected; exit 0;
  160.517s wall, a 4.58x speedup despite 13 added selection cases.
- Full stacked-spine file: 335 passed, 2,378 warnings, no failures/skips;
  exit 0; 347.905s wall (5m 47.905s), pytest 337.74s.
- Ran actual CI process expansions from `--list GROUP:PROCESS`; `--procs`
  returns process names, so the prompt's literal per-file loop was not valid.
  Every process ran serially with exact CI file arguments and no cacheprovider.
  An external observation-only plugin saved each file's counts and wall span.
- Initial us-qs:build before bundle priming: 1,368 passed, no skips; exit 0;
  626.070s wall. Reran the affected complete process after its final fix.
- Final CI process results (all exit 0):

  | Process | Files | Passed | Skipped | Wall seconds |
  |---|---:|---:|---:|---:|
  | us-qs:build | 29 | 1,369 | 0 | 654.966 |
  | us-qs:frame | 5 | 84 | 8 | 111.252 |
  | us-am:build | 59 | 1,899 | 1 | 855.001 |
  | us-am:other-shards | 48 | 1,000 | 2 | 139.672 |

- Final groups total 4,352 passed, 11 existing skips, no failures/errors across
  141 files. Skips cover eight optional Axiom cases, one opt-in 3.7 GB SIPP
  audit, and two existing live data-loader tests. No skip was added by this fix.
- Ruff packages/tools passed (exit 0, 0.066s); all six changed Python files
  passed format checking (exit 0, 0.021s). Diff whitespace passed (0.036s).
- Spec proof passed: 42,154/42,154 configuration fields, 41/41 inventories;
  exit 0, 81.683s wall. No spec pins moved. Protected-file and AST verification
  passed (exit 0, 0.080s): the uncached identity body, existing validators,
  authenticator, probe, and launch policy are unchanged. The full
  `worker_execution` subtree remains excluded from inventory digests.
- Independent reviews found no actionable defects. Engine-free compatibility
  was reviewed from source; no engine-free test run is claimed. Python 3.13,
  GitHub/Linux runners, wheels, unrelated CI groups, and certification were
  not run in this local fix lane. All command receipts are in the report.

## Next

No implementation or requested local verification remains. No push.

# Historical: Astra gate round 1 journal

> The section below records the prior lane at `50c9232b`; its results and
> next steps are historical, not the current CI-crawl fix state.

# F1 portable worker identity — Astra gate round 1

## State

Complete on 2026-09-04 on `f1-portable-worker-identity`, starting at
`32ce6f518e8847647f23b3f6f11e4a8dc060ed01`. Both peer findings have committed
fail-before regressions and fixes. All eight requested suites passed (964
tests); Ruff, formatting, and the spec-pin proof passed. No pins moved.
No GitHub network or push. Final report:
`/private/tmp/microcosm-pr871-astra-round1/out.md`.

## Done

- Read `CLAUDE.md`, the prior Sol journal, the worker identity/bootstrap,
  pinned QRF launcher, stacked launch/binding, and relevant test sections.
- Confirmed the assigned branch and clean starting tree.
- Read the GitNexus debugging workflow; its tools are unavailable, so trace
  execution directly from source and offline tests.
- Preserved the full `worker_execution` digest exclusion and all hard-boundary
  files, including the seed-attested `puf_qrf_chain.py` launcher.
- No concrete `-o` path was supplied. Asked asynchronously and wrote the
  report outside the repository at the stated default path,
  `/private/tmp/microcosm-pr871-astra-round1/out.md`.

- Reproduced finding 1 before any production edit: the fresh interpreter at
  the real stacked/chain launch observes Torch autoload `1` before the worker
  module loads (expected `0`); one test failed, direct exit 1.
- Reproduced finding 2 before any production edit: both source-tree and
  inherited-prefix valid-header stale caches execute altered code in the
  production identity probe; two cases failed, direct exit 1. Control children
  prove that disabling bytecode writes alone still executes those caches.
- The recovered pytest script uses an obsolete interpreter shebang. Use
  `uv run --no-sync python -m pytest` with a writable `UV_CACHE_DIR`; direct
  pytest script attempts exited 4 before collection. No sync was needed.
- Recorded both failing commands and observations in the external report.

- Fixed finding 1 by passing the semantic binding's forced overrides into
  the stacked launch environment before Python starts. Retained the worker
  bootstrap guard and corrected its comment; the pinned chain is untouched.
- The fresh-interpreter first-Torch-import regression now passes: 1 passed
  in 58.11s, direct exit 0. Its child sees `0` while the parent retains `1`.

- Fixed finding 2 using a shared launch context that forces an empty fresh
  `PYTHONPYCACHEPREFIX` and `PYTHONDONTWRITEBYTECODE=1` in both probe and stacked
  worker. The execution binding records the stable `{empty_pycache_dir}`
  placeholder, and traces refuse unexpected namespace/stdlib bytecode paths.
- The cache reproduction now passes in both identity-probe and worker `-m`
  modes across both cache locations: 4 passed in 22.62s, direct exit 0.
  Probe/refusal/exception-cleanup checks also passed (5 passed in 23.95s).
- Extended the real stacked launch regression to check child Python cache
  flags, replacement of an inherited prefix, and cleanup without parent
  environment mutation. Added semantic tamper coverage for both cache controls.

- Extended only the test validation fixtures to reuse pristine real worker
  identities by lock/fit controls and return independent deep copies. The
  stacked mutation helper compares against an independently obtained baseline;
  H5 tests mutate artifacts only. Production factories and source/cache
  mutation regressions remain uncached. Focused H5 checks: 2 passed, exit 0.
- Independent read-only review found no blocking issue in the startup/cache
  fixes, regressions, hard boundaries, or fixture mutation separation.
- Final repository Ruff passed; changed-file format check passed (5 files).
  Required release-preflight (42), fiscal-refresh (224), and source-blindness
  (497) suites passed with direct exit 0 each.

- Extended the same test-only reuse to the 11-case tail-control matrix: it
  mutates parent constants, while the fresh worker's installed source and
  startup environment stay identical. Independent review confirmed the real
  resource extraction and digest comparison remain active.
- Interrupted the earlier focused worker run after discovering that matrix's
  22 redundant identities: direct exit 130, 58 passes before interruption;
  this is not final evidence. Restarted the complete requested worker selector.
- Launch/binding integration passed (2 tests, exit 0). Required H5 (98),
  inventory (15), coverage-tool (7), and imputation (4) suites passed, exit 0
  each. Ruff and changed-file formatting passed after the last test edit.
- Verified every forbidden path, docs/tools, and uv.lock is unchanged from
  `32ce6f51`.

- Standalone spec proof passed, direct exit 0: 42,154/42,154 configuration
  fields and 41/41 inventory checks. No pins moved; the unchanged spec SHA is
  `9db29b4d33424fbb21a83c63927c7de55ba9a333d631f6323935f67a496eee46`.
- The focused worker rerun completed with direct exit 0: 77 passed, 244
  deselected, 141 warnings in 866.48s. The real transfer-bank integration also
  passed. All eight required suites total 964 passes, with no failures or skips.
- Finished the external report with both fail-before commands/observations,
  fix SHAs (`b131afb7`, `4a575d7d`), cache-isolation rationale, exact final
  verification commands/counts, no-pin proof, and deliberate scope exclusions.

## Next

- Local review of the committed fixes and external report. No implementation
  or verification work remains for these two findings; no push was performed.

# Historical: Sol gate round 1 journal

> The section below is the prior round's handoff at `32ce6f51`. Its pending
> state and verification claims are preserved as historical evidence.

# F1 portable worker identity — Sol gate round 1

## State

In progress on 2026-09-04 on `f1-portable-worker-identity`, starting from
`b26708a1`. All four Sol findings have fail-before reproductions. The schema-9
envelope, Torch backend-autoload, loaded-runtime/stdlib, and real-resource
fixes are implemented and focused green. Pin proof and full verification remain.

## Done

- Read `CLAUDE.md` and the F1 PR body/progress brief.
- Confirmed the requested branch and clean starting tree at `b26708a1`.
- Read the GitNexus debugging workflow. This workspace exposes no GitNexus
  query/resource tools, so call-path analysis is being performed directly from
  source and tests.
- Recorded the hard boundaries: keep the complete `worker_execution` subtree
  out of spec-engine digests; do not edit graph interface/acceptance lock files;
  refuse before side effects; keep tests offline; commit each coherent step.
- Recorded the required focused and final verification suites and the
  requirement to report only commands actually run in `out.md`.
- Synced the locked all-package US/UK environment after directing uv's cache to
  a sandbox-writable path; the unmodified command's two environment-specific
  refusals and the successful command are recorded in `out.md`.
- Added fail-before coverage proving that schema 9 accepts missing/wrong
  pipelines and routes other missing envelope sections around the common
  validator; the focused result was 1 passed and 1 failed, exit 1.
- Added fail-before identity coverage proving that loaded-runtime and stdlib
  mutations are unbound, a synthetic unapproved `torch.backends` provider is
  accepted, and the real SOI interest-components resource is absent. The four
  focused nodes failed as intended, exit 1.
- Added a fail-before launcher regression proving that an inherited/caller
  `TORCH_DEVICE_BACKEND_AUTOLOAD=1` reaches the child unchanged. The focused
  node failed as intended, exit 1.
- Replaced the prior unrelated `out.md` with the current round's reproduction
  report; fix/pin/final-verification sections remain explicitly pending.
- Fixed the schema-9 bypass: schema 9 now has an explicit complete stacked
  field set, traverses the same envelope classifier as schema 10, and does so
  before any compatibility attestation is read or authenticated.
- The valid schema-9 metadata-restoration case and the seven-case malformed
  envelope regression pass together (2 passed, exit 0).
- Forced `TORCH_DEVICE_BACKEND_AUTOLOAD=0` in both the authenticated semantic
  environment and the worker module bootstrap, before its QRF/Torch import.
- Enumerated and bound selected `torch.backends` entry-point metadata, refused
  provider distributions outside the installed-code closure before clean
  worker import, and refused duplicate canonical distribution identities so a
  colliding provider cannot evade RECORD hashing.
- Focused provider-refusal, duplicate-provider, launch-override, semantic
  tamper, and legacy relocated-worker acceptance checks pass (exit 0 each).
- Replaced the two-file resource list with a fresh worker-import audit trace
  using the same inherited startup search path as the real worker. The semantic
  transitive-import digest now includes every opened
  Microcosm namespace file, with bytecode canonicalized to source, portable
  locators only, and ambiguous duplicate locators refused.
- The trace freezes and revalidates namespace roots, captures transient import
  origins and successful pre-open file paths, refuses disappeared namespace
  files, and rejects an empty or displaced worker trace. Its opened stdlib
  paths supplement final `sys.modules` so transient stdlib imports stay bound.
- Resolved the loaded Python image through platform mapping with static and
  sysconfig fallbacks, and bound its kind and byte digest without serializing
  its path. The interpreter identity also binds the clean import's file-backed
  stdlib source and extension bytes while excluding site packages.
- The mocked runtime-byte and stdlib-source mutation tests pass together; the
  real SOI interest-components resource is observed and changes the resource
  closure digest; and a real full identity constructs and validates with the
  mapped `libpython3.14.dylib` (all exit 0).
- Kept worker identity schema v1 because it is the still-unreleased exact
  schema introduced by this branch. Bumping it would churn authored spec
  templates despite the requirement that spec-engine pins remain fixed;
  structural validation now requires the added v1 fields.
- Restored `puf_qrf_chain.py` byte-for-byte after the first pin run proved that
  editing its operational launcher also moves the QRF seed-kernel source
  attestation. The worker bootstrap now owns the override, keeping the existing
  seed protocol and compiled seed-map pins intact without re-pinning.
- Added bound-environment-keyed, deep-copied caches only to the stacked-spine
  and H5 canonical test fixtures so parameterized/tiny-pool cases do not rebuild
  one identical production identity apiece; production identity generation
  remains uncached. The interrupted pre-cache final run had 9 passes before
  exit 130 and is not treated as final evidence.
- Added the same narrowly scoped reuse to the inventory and coverage-tool test
  modules: each module constructs one real binding and deep-copies it for
  repeated report builds whose digests deliberately strip `worker_execution`.
  This does not cache the production resolver or replace its first real check.

## Next

- Prove spec-engine pins remain fixed, run the complete requested verification
  block, and finish `out.md` with exact commands, counts, and exit codes.

# Historical: ACS predictor release join

> **Historical note (2026-08-28).** This journal describes the
> `acs-predictor-release-join` lane as of 2026-08-27. The branch has since
> been merged into the `stacked-release-fix-train` integration branch
> together with the #794 gate-alignment and #798/#799 pregnancy/prior-year
> fixes, with spec envelope digests and coverage evidence regenerated over
> the union tree. Treat the "State"/"Next" sections below as history;
> check git/GitHub for current truth.

## State

Complete on 2026-08-27. The owner-approved release-time join from stacked-pool
ACS source lineage to the SHA-pinned 2024 one-year ACS person/household zips is
implemented, receipted, real-pool exercised, and fully verified. It populates
the six archived donor models' CPS-named predictors through reviewed
native-ACS crosswalks with strict hash, lineage, collision, totality, universe,
and clone-fan-out contracts. Model selection logic and every gate threshold
remain unchanged. The completed evidence and handoff are in `out.md`. No
network access, pool build, release build, publication, push, retraining,
threshold change, or launcher-contract edit occurred.

## Done

- Read `CLAUDE.md` and the prior weeksgate report's six owner-ruling items with
  their release-call and model-consumer evidence.
- Confirmed the requested branch `acs-predictor-release-join` is clean at
  `606cbd69`, based on `stacked-release-gate-alignment`.
- Read the GitNexus exploration and impact-analysis workflows. This workspace
  exposes neither GitNexus repository resources nor query tools, so the same
  call/dependency analysis will be performed directly from source and tests.
- Recorded the required source zips and SHA-256 pins, strict exact/total join
  contract, explicit crosswalk and receipt requirements, and verification
  boundary.
- Proved that `person_source_id` is not a reversible ACS key: ACS people are
  sorted by `(SERIALNO, SPORDER)`, receive a zero-based raw spine ID, and then
  receive a collision-dependent assembly offset. The pool retains the raw
  spine ID, `source_row_id`, `source_person_id`, household `SERIALNO`, and clone
  metadata, so the release join will use the retained semantic
  `(SERIALNO, integral SPORDER)` key and treat `person_source_id` only as the
  one-to-many clone fan-out identity.
- Audited the supplied candidate pool read-only: 856,626 distinct ACS source
  people expand to 1,736,840 rows (856,626 clone 0, 856,626 clone 1, and
  23,588 clone 2), with no duplicate `(person_source_id, clone_index)` pair.
  Every ACS row agrees with its raw spine/source lineage, and all selected
  people match the pinned raw person archive exactly.
- Verified both local archives against the charter pins. The person archive
  has 3,422,888 unique `(SERIALNO, SPORDER)` rows and no household orphans;
  the household archive has 1,631,969 unique serials, including 1,531,614
  occupied records. Both contain every requested native predictor.
- Established the disability universes from the pinned archive and the
  archived repository mapping: DEAR/DEYE are complete at every age;
  DREM/DPHY/DDRS are asked from age 5; DOUT from age 15; native code 1 is the
  consumer's difficulty bin and code 2 (plus an age-valid universe blank) is
  its non-difficulty bin.
- Established the consumed race/Hispanic bins: both SCF models distinguish
  White, Black, Asian, Hispanic, and Other; ORG distinguishes Hispanic,
  non-Hispanic White, non-Hispanic Black, and Other. `RAC1P`/`HISP` can map
  exactly to those bins without inventing detailed CPS combinations.
- Recovered the complete 2024 Census detailed-occupation-to-`POCCU2` consumed
  grouping from the native ASEC relationship and confirmed that ACS `OCCP`
  uses the same detailed codes. `PEIOOCC` is therefore a direct carry, while
  `POCCU2` will use an explicit reviewed 53-bin table; blank out-of-universe
  occupation maps to code 0, military to 52, and code 9920 to 53.
- Confirmed ACS `TEN` maps to the SPM vehicle model's three consumed tenure
  bins (mortgaged owner, outright owner, non-owner); no-cash-rent and verified
  group-quarters blanks belong to the non-owner bin. Confirmed the SSI model's
  `SSI_VAL` use is only the `> 0` reporter anchor and that native ACS `SSIP` is
  already carried as harmonized `ssi_reported`, observed exactly from age 15.
- Added the dedicated `acs_release_predictors` release boundary. It verifies
  the two canonical archive pins before opening either zip, streams only
  selected households, validates exact archive members and headers, rejects
  raw/person/clone collisions, binds retained pool lineage to
  `(SERIALNO, SPORDER)`, requires total one-to-one source-person matching, and
  fans mapped values to clones only through `person_source_id`.
- Added explicit disability, race/Hispanic, 530-code occupation, and tenure
  tables. A canonical crosswalk payload is pinned at SHA-256
  `1d4906242e9c73e31b3283659e5cad8242b8cbc42914ab6fa59547a10c8770e9`
  and rides the JSON-ready join receipt with per-model/per-predictor
  ASEC-native, ACS-joined, and still-null counts.
- Preserved CPS disability universe semantics (`-1` below the question age)
  and the ACS occupation universe. Blank `PEIOOCC` uses the CPS NIU sentinel
  `-1`; blank `POCCU2` remains 0 through age 15 and maps to the consumed
  no-occupation code 53 only from age 16. This explicitly preserves the
  one-year ACS/CPS source-universe gap instead of assigning every ACS
  15-year-old a never-worked status without source evidence. The explicit
  occupation table covers every one of the 530 codes in the pinned ACS person
  archive and every consumed POCCU2 bin.
- Changed the SSI-disability reporter read, without source routing, to
  row-wise coalesce measured ASEC `SSI_VAL` with harmonized native ACS
  `ssi_reported`. Adult blanks and conflicting dual reporters fail; genuine
  below-age-15 ACS blanks remain null in the frame and become false only for
  the consumer's `> 0` predicate.
- Hardened the join after independent crosswalk review: raw ACS `SSIP` and
  `ADJINC` now travel through the pinned join and must agree exactly with every
  native clone-0 `ssi_reported` value under the established adjusted-dollar
  formula. Raw `ESR`/`OCCP` must obey their exact age-16 universes, and all
  ASEC predictor receipt cells must be numeric and finite with complete,
  nonnegative `SSI_VAL`.
- Updated the SSI signal diagnostic to use the same row-wise reporter coalesce
  as the model consumer, while retaining the archived native-role anchor
  scope. A lost positive ACS-native reporter can therefore no longer evade the
  release gate merely because `SSI_VAL` is null on physical ACS rows.
- Added focused tests for crosswalk identity/all consumed bins, exact join and
  clone invariance, ASEC byte preservation, receipt contents, missing joins,
  raw and source-identity collisions, hash refusal, no-ACS identity, and SSI
  coalescing/universe refusal. Coverage now also fixes the age-15 occupation
  gap, malformed ESR refusal, malformed ASEC SSI refusal, raw SSI attestation,
  and gate-side ACS reporter preservation. The complete join, SSI, and
  source-blindness test files pass together, and focused Ruff is green.

- The release CLI now accepts the person/household zip and lowercase 64-hex
  SHA-256 options as an all-or-none set. It invokes the authenticated join
  after the last unrelated native-input gate and before SCF wealth, therefore
  before all six archived donor-model stages, then carries the complete join
  receipt into both `build_manifest.json` and `release_manifest.json`.
- Added parser refusal tests, a source-order contract over all six model calls,
  an end-to-end mocked main corridor that verifies the exact four join
  arguments and runtime ordering, an AST contract that binds the saved receipt
  to the sole manifest call, and JSON round-trip assertions for both manifests.
  Focused Ruff, five parser/order/manifest cases, and all six parametrized main
  corridor cases pass.
- Exercised the hardened join read-only on the complete supplied candidate.
  The 3,239,263,147-byte H5 matches its frozen manifest SHA-256
  `871b7e6467675a1e9475b54fd1baf64c53c0f75a3258b8357303a8df0d53642d`.
  The current official loader refuses that older candidate before H5 loading
  because its archived primary-QRF worker binding predates this branch's
  execution identity; this is an existing candidate/code-version mismatch.
  Loading those independently manifest-hash-verified bytes with their frozen
  assembly receipt allowed the join boundary itself to be tested without
  writing an artifact.
- The real join passed every source, raw-key, universe, SSI-attestation,
  totality, collision, and clone-fan-out check: 856,626 unique ACS source
  people matched 856,626 raw people in 382,903 households and populated
  1,736,840 support rows (856,626 each at clone indices 0 and 1, plus 23,588
  at clone index 2). Every CPS-named predictor consumed by the six models has
  234,133 valid ASEC-native and 1,736,840 ACS-joined cells with zero nulls.
  The logical SSI reporter anchor has 234,133 ASEC cells, 1,475,235 observed
  ACS cells, and exactly 261,605 preserved child-universe null support rows.
- The first real-data attempt exposed fixed-format HDF's expected object dtype
  for mixed-source columns. Tightened the ASEC validator to inspect each cell,
  accepting object-wrapped real numbers while still refusing strings,
  nonfinite values, nulls, and negative SSI. A focused H5-shape regression and
  all 15 join tests pass before the successful full-pool rerun.
- Added `changelog.d/acs-release-predictor-join.fixed.md`, describing the
  pinned release join, reviewed mappings, fail-closed lineage, dual-manifest
  receipts, and unchanged model/gate behavior.
- Repository Ruff passes, and the CI inventory verifier reports 310 tracked
  tests with `verification=ok`. Four complete pytest shards pass in separate
  processes: frame 295 passed/36 skipped, fit 93 passed, calibrate 203 passed,
  and data 318 passed/2 skipped.
- The first complete build-shard process reached 100% with 6,575 passed and 45
  skipped, plus five failures and six fixture errors. All eleven were the same
  expected source-attestation drift: `ssi_disability_criteria.py` belongs to
  both the direct and QRF seed-kernel inventories, so this task's runtime edit
  moved the seed protocol, compiled US seed map, every country spec identity,
  the minimal loader golden, and the generated coverage evidence. No ACS join,
  release CLI, manifest, archived-model behavior, or gate test failed.
- Applied the repository's established five-file source-identity repin only:
  seed protocol `59a098f9...31d8b`, US seed map `ce3850d8...e42ab`, US spec
  `16b7d5e6...dca38`, UK spec `2f921e4c...33a62`, BE spec
  `c87a0012...34ba`, and minimal-loader golden `b4946105...f2af`; regenerated
  `docs/evidence/spec-engine/us-f0-coverage.json`. All 25 affected cases and
  focused Ruff pass. The US bundle generator `--check` passes at the new spec
  identity, and coverage `--check` passes at 42,122/42,122 fields and 41/41
  inventory checks.
- Re-ran the complete build shard after the reviewed repin: 6,586 passed and
  45 skipped, with exit code 0. Re-ran final repository Ruff, the 310-file CI
  inventory verifier, both retained spec `--check` commands, and
  `git diff --check`; all pass. Wrote the required final report to `out.md`.

## Next

- No work remains in this lane. The dispatcher owns rebasing and the launcher
  contract update. A future authorized build must produce a pool whose current
  source-attested worker identity passes the official release loader; the
  supplied older candidate is useful join evidence but cannot be promoted.

# Weeksgate: stacked release gates and integer-week provenance

## State

Complete on 2026-08-27. Real-pool provenance has refuted the proposed
post-transfer amount-mapping mechanism: every fractional week is an ACS-origin
non-native clone prediction outside the calibration's clone-0 recipient scope.
The source codec, weeks-gate architecture, source-scope, clone-layout, and
stable-identity repairs are implemented and focused-tested. The complete
release-call roster is classified; six archived-model input assumptions require
owner rulings and are deliberately reported instead of guessed. Repository-wide
Ruff, the CI inventory verifier, and all five full pytest shards pass in their
required independent processes. The completed provenance, audit, verification,
judgment calls, and host-owned checkpoint-rerun consequence are in `out.md`.
No network access, artifact build, publication, push, pool build, or release
build is in scope.

## Done

- Read `CLAUDE.md` and the GitNexus debugging workflow.
- Confirmed branch `stacked-release-gate-alignment` is clean at `4f453746`.
- Confirmed the local GitNexus CLI is installed but the repository is not yet
  indexed. Its offline analyzer parsed the repository but could not register
  the index because the sandbox forbids writes to `~/.gitnexus`; the generated
  local index was moved out of the worktree to `/private/tmp`.
- Recorded the four requested workstreams: fractional-week provenance and PUF
  misclassification; integer-support calibration repair; stacked/legacy weeks
  gate alignment; and the full release-side gate archaeology sweep.
- Recorded the required verification boundary: repository Ruff plus one pytest
  process per shard, with no pool/release builds.
- Read the fixed-format HDF5 blocks directly and classified all 369 noninteger
  `weeks_unemployed` rows: 360 are ACS clone 1 (355 UC=0, 5 UC>0) and 9 are ACS
  clone 2 (all UC=0); all are positive, all 369 values are distinct, and the
  exact range is 1.0003521955067698--37.796501228614694.
- Confirmed zero nonintegers on ASEC rows and ACS clone 0. The receipted
  calibration covers exactly the 856,626 ACS clone-0 rows, maps 8,419 carrier
  amounts onto observed ASEC support with zero donor-support violations, and
  records QED 0.5882352941176471 to 0.0.
- Reproduced the 5,218-row false "PUF" classification: the legacy role helper
  calls every clone index above zero `puf_tax_detail`, regardless of raw source
  channel. The rows are all ACS-origin clones: 4,733 integer clone-1 rows, 355
  fractional clone-1 rows, 121 integer clone-2 rows, and 9 fractional clone-2
  rows with nonzero weeks while UC is nonpositive.
- Traced the actual fractional mechanism to the ACS transfer target codec:
  PolicyEngine-US declares `weeks_unemployed` as physical `float`, so the
  generic QRF path treats it as continuous even though its reviewed source
  contract is integer-supported. The later calibration repairs clone 0 only.
- Bound every final fractional value bit-for-bit to the late-transfer target
  bank. Its raw QRF draw contains 711 nonintegers: 342 ACS clone 0, 360 clone 1,
  and 9 clone 2. Post-transfer calibration changes 13,417 clone-0 rows and
  eliminates all 342 clone-0 fractions; the 369 out-of-scope clone fractions
  pass through unchanged.
- Added `weeks_unemployed` to the ACS transfer's authority-bound discrete
  numeric target registry. The existing codec now snaps every prediction to
  actual observed ASEC donor support, and the execution-contract identity
  receipts the policy alongside the two mortgage-year targets.
- Added a focused ACS-transfer regression proving weeks predictions retain
  observed integer support and the execution contract declares the target.
- Regenerated the checked-in US imputation authority so
  `transfer_execution.discrete_numeric_targets` includes `weeks_unemployed`.
  The generator's compile and byte-staleness checks pass at bundle spec SHA
  `821d5838da3ac368170e61e017f1a72648f93e8a011aa40e33b8c2b4b14511f3`
  at that source-fix step; the later runtime/seed identity repin below
  supersedes this intermediate digest. The spec-bundle and imputation-
  semantics suites pass.
- Passed the complete ACS-transfer test file (65 tests), the complete
  post-transfer calibration receipt-contract file (47 tests), focused Ruff,
  and `git diff --check` using the prebuilt `.venv` directly. A task-local
  `UV_CACHE_DIR` later allowed the required `uv run --no-sync` commands to run
  against that same prebuilt environment without accessing `~/.cache/uv`.
- Modernized the weeks summary/gate to derive its roster from actual assembled
  source channels, while retaining the legacy ASEC/PUF role path. The ASEC
  source-validity scope, direct native-clone reconciliation scope, and reviewed
  UC-constraint scope are now distinct and explicitly receipted in details.
- Kept all four plausibility-band tuples and every numeric threshold unchanged;
  non-ASEC assembled channels use the unchanged legacy recipient band.
- Added stacked ASEC+ACS and legacy ASEC+PUF fixtures covering roster detection,
  raw-source scoping, native reconciliation, and UC constraint ownership. The
  complete weeks file passes (26 passed, 1 skipped) with focused Ruff.
- Replayed the updated gate over the supplied pool's exact live arrays and
  weights. It now reports 234,133 valid ASEC source rows, 108,073 exact native
  reconciliation rows, 982,686 UC-constrained rows, zero source/UC failures,
  both unchanged channel bands passing, and only the genuine 369 nonintegers.
- Added a centralized provenance-owner API that distinguishes validated
  physical source channels from legacy clone-operator roles. The weeks gate
  now consumes that API rather than reading provenance columns directly, and
  both repository source-blindness tripwires pass alongside the stacked and
  legacy provenance/weekly-signal suites.
- Made WIC's deterministic draw key prefer the assembly-unique
  `person_source_id` on multispine frames, before the source-local raw identity
  triple. Distinct ASEC/ACS records can no longer collide when their vintages
  align, while all support clones of one assembled person retain the same draw
  and the legacy key order remains unchanged.
- Completed the release-tool call-roster audit. It found unambiguous physical
  ASEC scoping repairs in SSI take-up, workers' compensation, alimony,
  retirement contributions/distributions, and Medicare; clone-2 layout fixes
  in Head Start, voluntary filing, and prior-year income; and a stable-key fix
  in WIC. Archived ASEC-only predictor assumptions in SSI disability, SCF
  wealth, SCF auto loans, and SIPP vehicles require explicit model-owner
  rulings and will be reported rather than guessed.
- Reworked the assembled Head Start and voluntary-filing receiver layouts to
  key by assembly-unique source ID plus explicit clone index, accept clone 2
  and later, reject duplicate source/clone rows, choose clone 0 (or the lowest
  surviving clone) deterministically, and fan one source-level decision to all
  clones. Their legacy role-only duplicate contract remains unchanged.
- Replaced occurrence-pair clone diagnostics in voluntary filing and
  prior-year income with all-clone grouping on assembled source IDs, so a
  clone-2-only divergence is now detected. Focused tests and Ruff passed for
  all three module/test pairs.
- Scoped Medicare and retirement release diagnostics to physical ASEC source
  rows, separating raw-source validity from native direct-carry reconciliation
  where transferred clones intentionally differ. Kept the producer kernels
  origin-blind: the authenticated-pool release path skips those producers, and
  indirect physical-source routing would violate the repository's population-
  operator boundary. Their 60 focused module/source-blindness tests and Ruff
  pass after that review correction.
- Scoped alimony and workers' compensation raw validity to every physical ASEC
  clone and exact source-carry checks to physical ASEC native rows, leaving all
  clone-operator plausibility bands unchanged. Stacked ASEC+ACS and legacy
  ASEC+PUF fixtures pass (28 alimony and 21 workers' compensation tests).
- Changed SSI reporter-lineage capture to validate `SSI_VAL` only on physical
  ASEC rows and accept null ACS raw-source cells. Assignment remains source-
  blind: it consumes the source-ID set captured before L0, or obtains that set
  through the reporter helper when no explicit set is supplied. The complete
  SSI take-up file passes (71 tests).
- Fixed SSI-disability's non-fatal clone-divergence diagnostic to group every
  assembled clone by source person, so clone-2-only divergence is reported.
  The existing decision not to make divergence gate-fatal remains unchanged
  for an owner ruling; the complete focused file and source-blindness checks
  pass.
- Narrowed physical-channel resolution to a gate/reporter-only provenance API,
  removed its general runtime/PUF-support re-exports, and added a static exact-
  caller contract (including internal mask-helper callers). A future derive,
  impute, or wrapper use now fails the source-blindness suite instead of
  passing through indirection.
- Replayed the repaired release gates read-only against the supplied pool.
  Alimony, Medicare, retirement contributions/distributions, workers'
  compensation, and SSI reporter capture pass; the weeks gate now fails only
  on the genuine 369 fractional values. Prior-year income remains outside its
  unchanged availability band and WIC finds pregnant nonfemale rows, both
  genuine data/spec outcomes rather than stacked-layout archaeology.
- Completed an adversarial review of the repaired code and focused tests with
  no additional implementation defect found. It confirmed six owner-ruling
  items: SSI disability criteria, SCF wealth, SCF auto loans, SIPP vehicles,
  SIPP tips, and ORG wages/FLSA all consume ASEC-only archived predictors on a
  frame whose 1,736,840 physical ACS rows carry null source cells. ORG is
  guaranteed to fail its unchanged race/occupation bands; SIPP tips' unchanged
  tipped-occupation band passes while the ACS channel is dead.
- Passed repository-wide Ruff and the CI test-group inventory verifier. The
  calibrate, data, fit, and frame shards pass in four independent pytest
  processes.
- Corrected the reviewed WIC seed protocol to match the implemented assembled-
  multispine key precedence: assembly-unique `person_source_id` first, then the
  unchanged legacy raw/support/person fallbacks. An exact seed-grammar test now
  binds that order.
- Re-pinned the fail-closed spec-engine proof after adding one authored
  transfer-execution field and one resolved seed-protocol field: 42,122 total
  fields (32,352 authored and 9,770 resolved), complete exact-pointer claims,
  and all 41 inventory checks. Regenerated the committed coverage report and
  validated the final US spec SHA
  `5f44d96d45e9aabcea2d565ef063d68bfc0652df1b38b08aa31ce6896d15f371`.
- Verified in a detached `origin/main` worktree, using the same prebuilt venv,
  that the old BE, UK, and minimal-spec golden vectors still pass there. Their
  current repins therefore reflect this branch's attested runtime and seed-
  protocol changes rather than environment drift. All 102 tests in the eight
  directly affected spec-engine files pass; generated-bundle and coverage-
  report byte checks, focused Ruff, and `git diff --check` also pass.
- Ran the full build shard after that coherent spec repin. It reached 100% with
  exactly one failure and no errors: the multispine constants-adapter fixture
  still expected the former live US spec SHA. Updated only that live-binding
  expectation to the regenerated final SHA; the separate arbitrary checkpoint
  identity fixture remains deliberately unchanged.
- Passed the complete multispine-pool-tool file after that correction and
  committed the coherent fixture repin as `12a918ed`.
- Reran the entire build shard from zero in one process: 6,608 tests collected,
  100% reached, and pytest exited 0 with expected skips only. All five full
  package shards, repository-wide Ruff, generated-artifact checks, the spec
  coverage proof, the CI test inventory, and `git diff --check` are green.
- Wrote the final provenance tables, mechanism verdict, per-file rationale,
  exhaustive release-gate audit, owner-ruling list, verification evidence, and
  judgment calls to `out.md`.

## Next

- Host session: rerun `late_transfer -> simulated -> terminal-gates` from the
  candidate checkpoints because the discrete weeks codec changes pool content.
- Review the six archived-model owner rulings in `out.md`; do not reinterpret
  their missing ACS predictors through a gate-only threshold/scope change.

# Historical: gate-failed base-pool release lane

## State

Complete on 2026-08-26. Containment, opt-in carriage, and preflight surfacing
are implemented and fully verified. The implementation rejects both redundant
green-pool waivers and any release-manifest receipt that does not exactly match
the pool authenticated by preflight; the preflight's historically required
base/selection inputs and exit semantics are unchanged. No pool or release was
built, no artifact was published, and nothing was pushed.

## Done

- Confirmed the assigned branch and worktree.
- Recorded the v2 charter: close the legacy bare-H5 multispine bypass, add an
  explicit release-build opt-in, carry the authenticated red verdict, and
  surface it in publication preflight without making it an automatic
  publication failure.
- Confirmed that no network, artifact builds, publishing, or pushes are in
  scope.
- Traced the strict manifest loader, current stacked-only terminal-failure
  exception, H5 identity stamp, pool sidecar naming, legacy release arm,
  release manifests, and both preflight output modes.
- Reviewed the salvage branch's final source and test diff line by line. Its
  shared classifier/path-binding/receipt approach closes the bypass without
  changing either loader's contract and was retained with the subsequent
  coherence corrections recorded below.
- Completed the `simulation_ready` / `gate_failed` / loader consumer audit.
  Exact-k remains deliberately strict and head-to-head scoring remains the
  existing authenticated evidence exception.
- Identified report-only downstream caveats: stacked producer metadata still
  names only the k-ladder readiness consumer; red pool publication returns
  status 1 and stops shell chains; ACS-local derivatives keep a donor revision
  but do not project the nested red verdict; generic release consumers tolerate
  and ignore the additive receipt.
- Added a release/preflight-specific authenticated pool loader over the shared
  `require_simulation_ready` seam. The strict simulation-ready and existing
  scoring-only loader contracts remain unchanged.
- Closed the bare-H5 path by detecting either the canonical sibling manifest
  or the H5's stamped pool identity, requiring the sidecar, authenticating the
  publication triple, and binding it to the exact requested H5 path.
- Added `--allow-gate-failed-base-pool` only to the legacy `--base-h5` arm.
  It admits only a current authenticated stacked `gate_failed` pool, rejects a
  green or non-pool use, and never affects the exact-k arm.
- Added the self-contained `base_pool` receipt to both manifests, including
  status/readiness, immutable pool identities, flag use, gates JSON SHA-256,
  failure count/list, and the complete terminal verdict.
- Kept the static preflight inputs mandatory, authenticated its base identically,
  displayed red evidence prominently without changing its exit calculation,
  and required optional release-manifest carriage to match the authenticated
  receipt exactly.
- Hardened carried verdict normalization so nested pass/failure pairs and the
  aggregate verdict must be coherent and a red battery cannot report zero
  failures.
- Passed focused Ruff and the complete builder/H5/preflight test files after
  the containment changes.
- Passed the broader exact-k, launcher, release-contract, and publish-guard
  regression suites.
- Passed repository-wide Ruff and the CI test-group inventory verifier.
- Passed every pytest shard in its own process: build 6,545 passed / 45
  skipped; calibrate 203 passed; data 318 passed / 2 skipped; fit 93 passed;
  frame 295 passed / 36 skipped. Aggregate: 7,454 passed, 83 skipped.
- Confirmed `git diff --check` is clean and that no battery bounds,
  tolerances, plans, or terminal gate logic changed.
- Wrote the complete handoff, consumer audit, manifest schema, verification
  receipts, and commit inventory to `out.md`.

## Next

- Human review and merge of `release-from-gate-failed-pool`.
- Any later artifact operation remains separate: an operator must deliberately
  choose the red-pool flag, then run publication preflight and make the human
  publication decision. This lane performed none of those operations.

## Historical prior lane

The stacked-pool-to-release CD-vintage provenance lane previously maintained
this journal and completed before this work began. It authenticated and
applied household geography after source assembly, carried that authority
through checkpoint and publication identities, published verified CD-vintage
H5 attributes, and reached the unchanged release guard through the shared
fixed/table-aware reader. Its final verification was 7,241 passed, 77 skipped,
with repository-wide Ruff and anti-rot checks green. Full details remain at
commit `2263df36` (the parent of this lane's first journal commit).

The still-earlier PolicyEngine-US 1.819.0 lock-bump lane merged into
`origin/main` at `7b90bb18` on 2026-08-24; its final state remains at commit
`05d254aa` and its detailed receipts remain in the historical section of
`_LANE-NOTES.md`.

## US launch integration staging — 2026-09-09

### State
Source-only staging in progress. Execution and source/data admission remain root-owned.

### Done
Verified requested clean branch, base HEAD, main ancestry and preservation pins.

### Next
Apply thirteen explicit source layers, commit each, then separately review isolated ordinary execution. Existing journal history above is retained.

Layer 1: SAFE-ADDITIVE.patch applied; all declared postimages and preservation hashes verified. No tests executed.

Layer 2: GRAPH-RESTORE.patch applied; all declared postimages and preservation hashes verified. No tests executed.

Layer 3: ACCEPTED-SHARED-RESTORE.patch applied; all declared postimages and preservation hashes verified. No tests executed.

Layer 4: PUF-SUPPORT-MERGE.patch applied; all declared postimages and preservation hashes verified. No tests executed.

Layer 5: SOLVE-MERGE-PROPOSAL.patch applied; all declared postimages and preservation hashes verified. No tests executed.

Layer 6: J-GRAPH-COMPATIBILITY.patch applied; all declared postimages and preservation hashes verified. No tests executed.

Layer 7: F-CATALOGUE-OPTIMIZATION.patch applied; all declared postimages and preservation hashes verified. No tests executed.

Layer 8: GRAPH-ATTACHMENT-METADATA.patch applied; all declared postimages and preservation hashes verified. No tests executed.

Layer 9: F-JOINT-GEOGRAPHY-GATE.patch applied; all declared postimages and preservation hashes verified. No tests executed.

Layer 10: SOURCE-CLOSURE.patch applied; all declared postimages and preservation hashes verified. No tests executed.

Layer 11: PLACEMENT-ADDITIONS.patch applied; all declared postimages and preservation hashes verified. No tests executed.

Layer 12: ORDINARY-CLOSURE.patch applied; all declared postimages and preservation hashes verified. No tests executed.

Layer 13: INTEGRATION-REGRESSIONS.patch applied; all declared postimages and preservation hashes verified. No tests executed.

### State
All thirteen source layers staged and committed; behavioral qualification pending.

### Done
Per-layer pins and actual commit messages checked; store/current-main preservation retained.

### Next
Root reviews exact ordinary guard/source/resource admissions before execution. Full65 findings and survey/SCF lanes remain separately owned. No remote action is authorized.

### Integration source-resource closure — 2026-09-09

Root preflight found seven JSON source definitions declared by the ordinary guard but omitted by the staged patch delivery. Added the exact previously reviewed resource bytes from the accepted full65 source projection; no new resource admission or genuine payload. Preserved source staging and earlier evidence.

## Layer14 full65 replay correction — 2026-09-09

State: exact accepted two-file correction staged; successor57 integration execution pending.
Done: verified clean84243 preimages, exact r2 postimages, fixed store/current-main sources and frozen36 evidence.
Next: root admits the separate exact-allowlist57-case guard and final source identities before execution; no new resources.

## Source review publication — 2026-09-09

The user explicitly authorized pushing the current source work and creating PRs for Anthony to review. This supersedes earlier source-only local restrictions for source publication; it does not authorize a data release, merge or deployment.

Integration controls passed 36 cases; the exact replay correction subsequently passed all 57 cases at dfa7f872cd3eba3c42adf5758cde8b3ca38f3d17. Source/control, model-declaration and resource hashes matched externally after both runs. A final formatting/import cleanup and explicit test-observer loop binding are included for CI; no release result is claimed. See docs/us-launch-review.md for current scope, evidence and related PRs.


# US completion host integration — 2026-09-13

State: source implementation and bounded tests authored for root review; all runtime checks and tests UNRUN.
The task-specific no-commit instruction overrides the standing commit order.
Base: 6aa745c8762c3e4a140b8168470de5827e12c5cb.

Done: inspected status, branch, HEAD, AGENTS.md, CLAUDE.md, frozen R2 plan and
independent source-plan review. Concurrent completion/tax seam files are excluded
from this work's ownership.

Done: added private completion custody, retained issued property parent, registry
copies, actual union artifact/state checks, role/child reconstruction and separate
tax gate wiring. Authored 10 test functions / 20 source-expanded cases and docs.
AST, direct Ruff lint/format, and tracked whitespace checks pass.

Next: root review and reconciliation, inventories/test-spine updates, bounded
resource/guard proposal, and actual fixture/cold/required/adversarial validation.
45/49 counts remain source-derived; combined 51 and fixture assumptions are UNRUN.
Final report: FINAL_REPORT.md. No acceptance or publication claim.

## Bounded completion-host source corrections — 2026-09-13

State: bounded source corrections complete; all runtime checks remain UNRUN.
This correction pass used the 20-minute source-only window beginning 22:09 UTC.
The later explicit no-commit/no-staging instruction governs this correction pass.
Ownership is limited to graph_survey_completion_host.py,
test_us_graph_atomic_completion_host.py, docs/us-survey-completion-host.md,
and this exact PROGRESS.md / FINAL_REPORT.md update.

Done: read the full independent source review and full approved R2 plan from
the supplied recovered absolute paths. The R2 SHA-256 matches
b402d6be9d374774e27da7bf42fb83a216305d4cddd2ad8347c8149536269a29.
Confirmed intentional dirty worktree and HEAD
6aa745c8762c3e4a140b8168470de5827e12c5cb. Earlier journal content is preserved.

Done: exact receiving1/roles0-or-2/child6/tax3 ordered rosters are bound before
compilation; child document count and knownness derive from retained nodes and
reconstructed/materialized-verified evidence. Authored separate 49/45 cold+required
acceptance, equal-values/replaced-Population refusal, real observed base-hit/union
and final-I/O identity controls, actual tax-rebased parent refusal, exact child
adversary reasons, full-spine writer ordering, both FILTER conservation ledgers,
dimensions and owner/version transitions. Source proves table copying; immediate
base preservation assertions remain alongside permanent revocation. Fixture-only
composition wrappers now restore before issuance to avoid cross-fixture chaining.

Done: AST parsing, direct Ruff lint/format and git diff --check pass. AST reports
16 test functions / 32 source-expanded cases. All tests, collection and project
imports remain UNRUN. Only the five explicitly owned files were edited; no
commits, staging, agents, dependency/network/browser access or payload execution.

Next: root reviews exact source/hashes and budgets separately selected execution.
T4 financial-only-parent coverage, remaining T5 initial/final/reconciliation
adversaries, T6 actual fixture validation and T7 GQ/zero-weight scientific fixtures
remain open with concrete reasons in FINAL_REPORT.md. The independently confirmed
old status/tax42 owner repair, inventories, CI/source closure, scientific/native
capacity and release/default/PR893 decisions remain separately owned.

---

# Lane: verify native sources once per run (`native-verify-once`)

Branch `native-verify-once`, worktree `~/PolicyEngine/_worktrees/microcosm-verify-once`,
base `f7bb88525a78786f91bc3ebe2083ef4b1c85de18` (PR #893 head). Started
2026-09-15.

**State (2026-09-16, second session):** all five mechanisms landed with
proofs and the epoch wired into both atomic capsules; draft PR #935 open. The
open work is the CI-shaped test sweep, the before/after probe measurement, the
report and the main-only split.

**Goal.** The 9/15 pilot v5 measurement
(`~/PolicyEngine/_recovered/pilot-runs/native45-v5/out.md` §2) attributes ~79 %
of a native 1/1000 run to repeated admission and verification of 3.48 GiB of
staged source: node execution is 1,143 s of a 5,362 s run. Make source
authentication and identity verification happen once per run instead of once
per accessor use and once per executed node, without weakening any refusal and
without moving any digest value.

**Done:** read the evidence base; located all five mechanisms at base HEAD.

**Done (2026-09-16):**

- `docs/us-native-verification-once.md` — the design note, written before any
  capsule changed.
- **Mechanism 5**: `_object_stream` builds each plain float/int/bool column's
  context-digest bytes with numpy. Byte-identical against a verbatim copy of the
  pre-change body over every column kind, the float specials, non-canonical NaN
  payloads, both int64 endpoints and a 200-frame random sweep. 0.47 s -> 0.09 s
  on a 6,928 x 240 frame.
- **Mechanism 4**: the ACS record fence now uses `bytes.find`/`bytes.count` with
  cached terminator cursors. Proven against the byte loop on 46,655 exhaustive
  short strings across five ceiling settings, 4,500 random strings, and the real
  staged `csv_pus.zip`: 3,422,890 records, identical digests, 226.9 s -> 4.6 s
  (49x). Re-pins `_ACCEPTED["acs_person_coverage_authentication.py"]`.
- **Mechanism 3**: `run_graph` carries a `_SourceIdentities` cache keyed on stat
  signatures, and re-derives every source in full before building the manifest.
  `RunManifest.source_identities` records it without moving any existing value.
  The refusal it protects had no test at all; `test_graph_executor_source_identity`
  now pins it. Graph suite 108 s -> 72 s.
- **Mechanisms 1 and 2**: `survey_population_preparation.verification_epoch()`,
  an opt-in scoped memo. Cheap tier every borrow (live authority, attached
  payloads, producer encoding, roster stat identities); expensive tier skipped
  only while a signature over every path it reads and every live buffer it
  digests is unchanged; unconditional full re-validation on leaving the epoch.
  The two moved inventory contracts are re-derived (below).

**Done (2026-09-16, second session):**

- Draft PR [#935](https://github.com/PolicyEngine/microcosm/pull/935), base
  `microcosm-us-launch-integration-20260909`.
- Main-only split branch `graph-verify-once-main`, worktree
  `~/PolicyEngine/_worktrees/microcosm-graph-verify-once-main`, branched from
  `origin/main` (51c314382). It carries the graph-shard change only: the three
  commits re-applied, plus a changelog fragment. Its one deliberate difference
  from this branch is that `_update_scalar` has no exact-float `struct.pack`
  shortcut on main -- that belongs to PR #893 -- so the parity test's reference
  copy was aligned to main's actual pre-change body, which makes it a literal
  verbatim copy there too.
- Before/after measurement staging in `.measure/` (gitignored): a parameterised
  copy of the v5 pilot probe that takes the source tree, the staged run inputs
  and the output directory from the environment, asserts every imported
  `microcosm` module resolves inside the measured tree, and enforces a 16 GiB
  resident ceiling alongside the CPU one. Baseline worktree
  `~/PolicyEngine/_worktrees/microcosm-verify-once-baseline` is detached at
  `f7bb88525`.

**Next:** finish both probe runs, the CI-shaped test sweep on this branch, the
19-node harness if memory allows, the report, push the split branch and open
its draft PR, and update #935's body with the measurement table.

**Pins re-derived so far:**

| pin | old | new |
|---|---|---|
| `acs_native_coverage_binding._ACCEPTED["acs_person_coverage_authentication.py"]` | `475aa795…fe85bcff` | `9ec68721…d88e8e49f` |
| `graph_implementation_inventory.json` `survey_population_preparation.py` `unbound_uses_sha256` | `29c09f6f…d296ef91` | `d114117c…dd4005910` |
| `graph_implementation_inventory.json` `asec_2024_native_population.py` `unbound_uses_sha256` | `71463df4…d7f287608` | `7bb20439…40d58cdf4` |

**Lane notes.** Root `out.md` is a tracked file holding the Amendment 19 lane's
committed report; this lane's report goes to
`experiments/native-verify-once/out.md`. The v5 cold run (`run_v5.py`, pid
81194) is live on this machine, so probe runs wait for the window the lane
brief allows (`pgrep -f run_v5.py` empty, or > 40 GB free).
