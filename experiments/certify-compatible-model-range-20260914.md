# Publisher compatibility range at source-enrichment certification

Lane receipt for PR [#928](https://github.com/PolicyEngine/microcosm/pull/928),
branch `max/certify-compatible-model-range-20260914` off `origin/main` at
`18271b28d`. 2026-09-14.

**Nothing was built, certified or published.** This lane changes producer and
validator source only.

## The problem, as measured

`certify_source_enrichment` wrote
`compatible_{model,core}_packages = [{"name": pkg, "specifier": "==<tested version>"}]`
over whatever the candidate manifest held, and `_check_compatibility` then
required that exact list at every later validation, publish preflight included.
There was no flag, and a hand-widened input manifest was silently normalised
back to the exact pin rather than rejected.

So each certification was a **swap**, not a widening: re-certifying the live US
release under `policyengine-us` 2.0.1 moves the binding off 2.0.0, and a country
patch release that changes nothing this lane measures still forces a new
certified data release even when the H5 bytes are identical.

The limit was producer tooling, not schema.
`contract._check_compatible_package_entries` already accepted any PEP 440
specifier set containing the built-with version, and both consumers honour one.

## The option

```bash
python -m microcosm.data.source_enrichment --certify \
  --release-dir ... --output-dir ... --parent-h5 ... --artifact-root ... \
  --compatible-model-specifier 'policyengine-us>=2.0.1,<2.1' \
  --compatibility-claim-declared-by 'PolicyEngine data release owner, microcosm#NNN' \
  --compatibility-wheel ...
```

Exact spellings: `--compatible-model-specifier`,
`--compatibility-claim-declared-by`. Python API: the same names as keyword-only
arguments to `certify_source_enrichment`.

### Validation rules

The specifier value must:

1. parse as a PEP 508 requirement (`packaging.requirements.Requirement`);
2. carry no URL, no extras and no environment marker;
3. name the built-with package (`policyengine-us`), compared with
   `canonicalize_name`, so `policyengine_us` is the same name and
   `policyengine-uk` is refused;
4. be a non-empty PEP 440 specifier set — `","` parses to an empty set and is
   refused;
5. contain the version certification actually tested, under the same
   `Version(v) in SpecifierSet(s)` semantics the consumers apply;
6. stop below the next major version after the tested one — for a 2.0.1 build,
   anything matching 3.0.0 is refused, so `>=2.0.1`, `!=2.0.5`, `>=2.0.1,<99998`
   and `>=2.0.1,<3.0.1` are all refused while `>=2.0.1,<2.1`, `~=2.0.1`,
   `==2.0.*` and `>=2.0.1,<3` are accepted;
7. state a lower bound as well — a range open below certifies every release the
   package ever made, so `<2.1` and `<=2.0.5` are refused although both are
   bounded above. Probed by asking whether the set still admits `0` at the
   tested version's epoch, which none of the four accepted forms above does.

The declarer must be trimmed printable text of at most 200 characters. Both
options are required together and only with `--certify`. Rules 1–4 and the
declarer are checked **before** the qualification run; 5, 6 and 7 need the
tested version and run after it.

### What is recorded

`release_manifest.json`:

```json
"compatible_model_packages": [
  {"name": "policyengine-us", "specifier": ">=2.0.1,<2.1",
   "basis": "publisher_claim", "declared_by": "..."}
]
```

and the same object in `source_enrichment.json` under
`compatibility.publisher_claims.model`. The specifier is stored exactly as
declared, save that PEP 508's optional parentheses are dropped.
`build.built_with_model_package` is untouched — it still names the exact version
certification tested.

**Model only.** Core keeps the exact pin it has always had; a `core` key in
`publisher_claims` is refused rather than honoured.

**Default unchanged.** Omit the options and certification emits what it always
did: no `basis` key, no `publisher_claims` key.

## Verification

| check | result |
|---|---|
| `uv run pytest packages/microcosm-data/tests/` | 565 passed, 2 skipped |
| new tests | 46 (`test_source_enrichment.py` 51 → 92, `test_contract.py` 236 → 241) |
| `uv run ruff check .` | clean |
| `python3 tools/ci_test_groups.py --verify` | `verification=ok` |
| default-path byte comparison vs `origin/main` | structurally identical |
| wrapper behaviour on a declared range | measured, see below |
| mutation check on the two guards the review found dead | both now killed |

### The default path is byte-identical

Ran the unchanged `test_certification_writes_new_bundle_and_preflight_replays`
on this branch and on a clean `origin/main` worktree with a controlled
`--basetemp`, then diffed the emitted `release_manifest.json` and
`source_enrichment.json` with SHA256 values normalised (each run builds a fresh
H5, so raw hashes differ): same files, same keys, same values, no `basis`, no
`publisher_claims`. Re-run after the review fixes, same result.

### What the consumers do

Measured against policyengine.py's installed provenance code. A manifest
declaring `>=2.0.1,<2.1` over `built_with` 2.0.1:

| runtime version | `validate_release_manifest` (bundle certify) | `certify_data_release_compatibility` (runtime bind) |
|---|---|---|
| 2.0.0 | refused | refused |
| 2.0.1 | accepted, basis `built_with_model_package`, silent | accepted, basis `exact_build_model_version`, silent |
| **2.0.2** | **accepted, basis `compatible_model_packages`, warns** | **accepted, basis `legacy_compatible_model_package`, silent** |
| 2.1.0 | refused | refused |

The 2.0.2 row is the point of the change. The extra `basis`/`declared_by` keys
are ignored by the wrapper's pydantic `CompatiblePackage`, not rejected.

Note the asymmetry: only bundle certification warns.
`src/policyengine/provenance/manifest.py` contains **zero** warning calls
(`grep -c warn` → 0), so a user running a claim-covered version gets no signal
and the recorded basis is the only trace. An earlier draft of the docs claimed
both paths warn; that was wrong and is corrected.

## Adversarial review

Five read-only reviewers (tamper surface, default identity, consumers,
tests/docs, correctness), 25 findings, each put to an independent refuter.
Two survived refutation, and I acted on three:

1. **Docs claimed the wrapper warns on the runtime-bind path.** It does not.
   Corrected, and the table above replaces the claim with measurements.
2. **Two validator branches were dead to the suite** — the report-entry shape
   guard and the URL/extras/marker rejection. Deleting either left the suite
   green. The shape guard needed a case holding the manifest canonical so the
   manifest-vs-report comparison cannot mask it. Both mutants now fail three
   tests each.
3. **Core was a real regression I had introduced** (refuted by its verifier; I
   disagree). `origin/main` compared `compatible_core_packages` against the
   exact pin unconditionally. My first version honoured a hand-forged `core`
   key in `publisher_claims` even though no producer can emit one — a guard
   relaxed for no feature. Claims are now model-only.

Also tightened, from findings whose verifiers called them non-defects but which
improve the change: boundedness was a single probe at version `99999`, which
`!=99999` and `<99998` walked past, so it is now the next-major rule; the
parenthesised PEP 508 spelling no longer lands parentheses in the recorded
specifier; and a malformed `publisher_claims` no longer also reports that no
claim was declared.

## Second pass, part two

A second adversarial review of head `7cb8eae6e` confirmed the six fixes above
and left five items. All five are in. Two of them did not survive contact with
measurement, and the code and runbook follow the measurement rather than the
review.

1. **The guard bounded a claim above and never below.** `<2.1` over a 2.0.1
   build contains the tested version, fires neither existing probe, and
   publishes — after which a consumer on 0.9.0 reads the manifest as certified,
   for a model predating the native-input loader path this lane measures.
   Measured before fixing: `Version("0") in SpecifierSet("<2.1")` is `True`, and
   `False` for `>=2.0.1,<2.1`, `~=2.0.1`, `==2.0.*` and `>=2.0.1,<3`. A third
   probe at `Version(f"{tested.epoch}!0")` now refuses it. The probe carries the
   tested version's epoch because a claim may mix epochs: over a `1!2.0.1`
   build, `>=2.0.1,<1!2.1` is open below within epoch 1 — it admits `1!0` —
   while excluding a bare `Version("0")`, which sorts under every epoch-1
   release. (An earlier draft of this note and of the code comment justified
   the carry the other way round, claiming an epoch-0 zero sits outside an
   epoch-bearing claim. It does not: `Version("0") in SpecifierSet("<1!2.1")`
   is `True`, and the test written on that reasoning survived replacing the
   probe with a bare `Version("0")`. The third adversarial pass caught it; the
   test now uses the mixed-epoch claim, which kills that mutant.) `<2.1` and
   `<=2.0.5` moved to the refused parameters and the next-major error text no
   longer offers `'<2.1'` as a good claim. The known residue is unchanged and
   still documented: `>=2.0.1,!=3.0.0,!=99999.0.0` names all three probe
   versions and passes.
2. **`compatibility.narrowed_claims` was written and read by nothing.** It is
   now read back by `recorded_narrowed_claims` and printed beside the verdict
   by validation and by `microcosm-publish-release --preflight-only`, and on
   stderr by publication as well — publication does not require the preflight
   first, since `tools/publish_release.sh` passes its arguments straight
   through. It reports rather than gates: an absent or malformed record reads
   as no record and never turns a valid bundle into an error.
3. **The "pass the flags" remediation fired even when the flags were passed.**
   Gated on the claim being absent this run. Re-certifying with a tighter range
   still warns — that is the point of the warning — without advising the
   operator to pass options they just passed.
4. **Core's narrowing branch is unreachable, not merely misworded.** The review
   asked for the wording, which is now "moves the policyengine-core
   compatibility pin". Trying to write an end-to-end test for it showed why
   none exists: `certify_source_enrichment` validates the input bundle first,
   and `_check_compatibility` re-runs the loader qualification and requires the
   recorded receipt to equal the current runtime, so a moved Core version is
   refused (`compatibility receipt differs from actual native loader
   tests/runtime`) before the emitted pin could differ from the carried one.
   The branch stays as defence in depth; the wall in front of it is now pinned
   by a test, and the wording by a unit test through `_narrowing_notice`.
5. **The prerelease item's premise was wrong.** The review held that PEP 440
   containment excludes prereleases by default, so a prerelease built-with
   version could take no range at all. Installed `packaging` 26.2 does the
   opposite: `SpecifierSet.contains` documents that with the default
   `prereleases=None` it follows PEP 440's recommendation and matches
   prereleases, and `Version("2.1.0rc1") in SpecifierSet(">=2.0.1,<2.2")` is
   `True`. The real constraint is ordering — a prerelease sorts below its own
   release — so over a `2.0.1rc1` build `>=2.0.1,<2.1` and `~=2.0.1` are refused
   for excluding the tested version, while `>=2.0.1rc1,<2.1` and `==2.0.*` are
   accepted and pass all three probes. The runbook says that, and a
   characterization test pins all five outcomes so a `packaging` release that
   moved either would fail rather than quietly rewrite the runbook.

Every item but the fifth has a test that failed before the change and passes
after. The fifth is doc-only; its test is a characterization test that passes on
both sides and is labelled as one.

## What I could not verify, and residual limits

- **This is a record of who claimed what, not a seal.** The report's only
  integrity anchor is its `artifacts` entry in the very manifest it
  authenticates, so a coordinated report+manifest edit with a restamped hash
  validates. That was equally true of the exact pin it replaces — whoever could
  edit one could edit the other. The controls are
  `_check_producer_source_identity`, the publish preflight, and the human
  publication decision. `test_producer_validation_is_not_the_tamper_control`
  pins this rather than leaving it implied.
- **Containment depends on the installed `packaging` major.** A producer and a
  consumer on different `packaging` versions could in principle disagree about
  a specifier. Pre-existing: `contract.py` and `loader.py` already use
  `SpecifierSet` for the exact-pin path.
- **No real certification was run.** Every test uses the synthetic fixture with
  `run_native_loader_compatibility` monkeypatched; the tested-version strings
  are fixture values (`1.999.0`, `3.99.0`), not a real runtime. Whether a given
  `policyengine-us` patch range is *substantively* safe is a judgment the
  publisher makes, and this change only records it.
- **`contract.py`'s basis check is a shape check, not authorisation.** A
  manifest of any release type may carry `basis: publisher_claim` with a
  declarer; only the source-enrichment validator ties it to a declaration.
- **Two workspace tests fail locally and are not mine.** Both are in
  `packages/microcosm-build/tests/test_release_target_parity.py`:
  `TestRegeneration::test_committed_artifacts_match_regeneration` and
  `TestRegeneration::test_gate_passes_on_real_compiled_registry`. Measured a
  second time on 2026-09-14 by running that file in a detached `origin/main`
  worktree at `18271b28d` and in this one: `2 failed, 35 passed` in both, same
  `LedgerHierarchyMetadataError` (Chronicle fact
  `arch.aggregate_fact.v2:01eb46de220addab5ce4827d`, dimension
  `bea_nipa.series_code`, no non-empty label), while main's own CI for that
  commit was green — a local feed/environment condition. The first session
  recorded one failing test here; the count was wrong, the diagnosis was not.
- **`source_enrichment.py` and `contract.py` are both in
  `PRODUCER_SOURCE_FILES`.** Once this merges, a candidate whose build receipt
  recorded the pre-merge hashes can no longer be published from a post-merge
  checkout; it needs a fresh producer run, as with any change to those files.

## Incident: a review lane wrote into this worktree

The first adversarial review ran against this live worktree. One reviewer
checked out `origin/main` copies of the changed files for a byte-for-byte
comparison at the moment a `git add -A` ran here, so commit `4b0ae3624`
committed the reverted tree — deleting the feature from the PR — and a
reviewer's scratch test module, and it was pushed.

Caught on the next inspection. `git diff 8f82d0c6a HEAD` was used to confirm the
restored tree is exactly the intended docs and journal changes with every
feature and test file byte-identical to the last good commit; the branch was
reset and force-pushed. The relaunched review was read-only, and the worktree
tree hash was unchanged across its whole run.

The mistake was mine: pointing five parallel agents at a live worktree without
isolation. Review lanes get their own worktree or explicit read-only rules.
