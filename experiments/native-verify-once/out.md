# Build lane: verify native sources once per run

Branch `native-verify-once`, worktree `~/PolicyEngine/_worktrees/microcosm-verify-once`,
base `microcosm-us-launch-integration-20260909` (PR #893's branch). The lane was
written against that branch at `f7bb88525a78786f91bc3ebe2083ef4b1c85de18` and
rebased onto its tip `363a9033b4e3c05c8758ef7411d69b7cd72a475e` on 2026-09-16,
so every measurement below was taken against `f7bb88525` while the diff at the
end of this report is against `363a9033b`. 2026-09-15/16.

**Where this report is.** The repository's root `out.md` is a tracked file
holding the Amendment 19 lane's committed report, and root `PROGRESS.md` is
cumulative. Writing this lane's report to either would delete another lane's
committed content, so the report is this file and the lane's journal is an
appended section of `PROGRESS.md`.

**What this is not.** Nothing here is a statement about dataset quality,
calibration or release eligibility. Every measurement is descriptive; none is a
build, a certification or a release artifact.

**Status, 2026-09-17.** **No test group failed**, in either round. Every pytest
group of the CI-shaped battery below exits 0, including the 3 h 15 min serial
run of the 62 `microcosm-build` test files that name or import a touched module;
and every group re-run after the 2026-09-16 verification findings exits 0 as
well, including a 3 h 07 min serial run of the 46 of those files that reach a
module the findings commits changed (see "After the verification findings"). One
non-test check does exit non-zero and is not softened here: `ruff format
--check .` exits 1 with `77 files would be reformatted, 1110 files already
formatted`. That condition is inherited, not earned by this branch — the
intersection of those 77 files with `git diff --name-only
origin/microcosm-us-launch-integration-20260909..HEAD` is empty (checked with
`comm -12`), the branch does not touch `pyproject.toml`, where the ruff config
lives, the 17 `.py` files it does touch report `17 files already formatted`,
and CI's `lint` job runs `uv run --no-sync ruff check .` only
(`.github/workflows/test.yml:183`), which passes. It is reported because the
lane brief asked for `ruff format --check`.

Every number below is traceable to a file path, and every number was measured
at branch head `284bc6e9624d211ba9c16dcdf04df63b9e87b10c`, which the report
commit sat directly on top of. The nine commits that resolve the 2026-09-16
verification findings, and the rebase that carries them, come after that head;
what they re-ran is the section "After the verification findings" at the end of
the test summaries, and nothing above it was re-measured. Draft PR:
[#935](https://github.com/PolicyEngine/microcosm/pull/935).

## The problem

`~/PolicyEngine/_recovered/pilot-runs/native45-v5/out.md` §2, measured by stack
sampling on real runs: on the 19-node financial graph at `Fraction(1, 1000)`
(3,168 expanded households, 5,362 s wall), node execution is 1,143 s and the
remaining ~79 % is admission and verification of 3.48 GiB of staged source,
repeated. QRF fits inside that run cost 0.06–0.07 s each.

The design note is [`docs/us-native-verification-once.md`](../../docs/us-native-verification-once.md):
per mechanism, the guarantee today, the change, why the guarantee survives and
the new cost class. It was written and committed before any capsule changed
(`d9136a20e`).

## Per mechanism

### 1. `AuthenticatedSurveyPopulationPreparation._checked()`

Every borrow re-ran the whole authentication — both source catalogues, the ACS
native coverage binding, the nested ASEC native population, the ten-file roster
re-hash and every pure seal — and the graph registers
`verify_survey_population_preparation` as a **per-node population observer**, so
a run paid one of these per executed node on top of every kernel borrow.

**Guarantee, in two sentences.** Inside a `verification_epoch()` every borrow
still pays, in full and unmemoised, the live authority, the attached owner
payloads and `_encode(_producer())` as refusals, and reads `_file_stats` over
the whole source roster — into the *signature*, not as a comparison — so every
producer change refuses in the cheap tier and every on-disk change that moves
any of five stat fields is a memo miss whose complete validation refuses at the
same borrow, with the same code, as an unmemoised borrow does today. What a
signature cannot see — an in-place write into a live buffer, or a value written
into a frozen plan row — is closed by an unconditional full re-validation when
the epoch closes, before the run returns anything; outside an epoch nothing is
memoised at all.

**Mutation tests.**
`test_a_source_appended_to_mid_epoch_refuses_at_the_next_borrow`
(3 parametrisations), `test_a_source_truncated_mid_epoch_refuses_at_the_next_borrow`
(2), `test_a_source_rewritten_in_place_mid_epoch_refuses_at_the_next_borrow`,
`test_a_file_added_to_a_source_directory_mid_epoch_refuses`,
`test_a_touched_source_refuses_on_its_stat_identity_alone`,
`test_a_change_a_signature_cannot_see_refuses_when_the_epoch_closes` and, added
for the 2026-09-17 re-verification,
`test_a_roster_stat_moved_inside_an_inner_close_refuses_at_the_next_borrow`.
Each asserts the refusal **twice** — at the borrow that follows the mutation,
and again when the epoch declines to close over it — except
`test_a_change_a_signature_cannot_see_refuses_when_the_epoch_closes`, which
asserts it once by design: that mutation is the one a signature cannot see, so
the borrow after it is a memo hit and the close is the only refusal.

**The memo-miss branch itself, and the code it raises.** Every mutation above
targets a roster file or a roster directory, so before the 2026-09-16
verification findings each of them refused in the cheap tier, which compared
`_file_stats` ahead of the memo lookup: the branch the design leans on hardest
— a signature miss running the complete validation — was never reached for this
capsule, and the code it raised inside an epoch (`SOURCE_STAT_CHANGED`)
differed from the code the same mutation raises with no memo at all. Two
changes close that. The roster stat identities moved out of the cheap tier into
the signature, so the miss runs `_validate` and `_validate` decides the code;
and `test_an_in_epoch_refusal_carries_the_code_it_carries_today` refuses four
mutations twice each, once inside an epoch and once on a fresh unmemoised
fixture, asserting the two codes are equal: `SOURCE_CHANGED` for an appended
`selection-request.json`, `PREPARATION_VERIFICATION_REFUSED` for an appended
`acs/csv_pus.zip` and for an appended `asec/pppub25.csv` (the ACS and ASEC
catalogues' own refusals, translated by `_checked`), and `SOURCE_STAT_CHANGED`
for a touched roster file. Separately,
`test_a_changed_snapshot_copy_refuses_through_the_memoised_tier` picks the one
kind of path the cheap tier cannot see at all — an ACS catalogue private
snapshot copy, inside the memo signature and outside `_file_stats` — and
rewrites one byte in place, restoring the mode and the modification time and
asserting `_file_stats(state.root)` is identical across the mutation, so only
the memo can be what refuses; it refuses with `PREPARATION_VERIFICATION_REFUSED`
at the borrow, again at the close, and identically to the unmemoised borrow.

**The window that fix opened, closed on 2026-09-17.** Moving the roster stats
out of the cheap tier removed the anchor that had made a touched roster file
refuse at the next borrow, and `_finalize_epoch` recorded its signature *after*
validating. At an inner nested close that memo survives into the outer epoch,
so a roster file whose stat moved after `_validate`'s own `SOURCE_STAT_CHANGED`
comparison and before the recording — a window containing the whole trailing
`_pure_final` — became the new normal, and every outer borrow up to the
outermost close was a hit. The refusal still fired, but at the end of the run,
after intervening nodes had written store records: the class this report
documents as mechanism 3's residual, reached here by a different route. Each
close now takes the signature before validating and again after and records
none when they differ, so the next borrow is a miss that pays the complete
validation and refuses at that borrow; the entry stays, so the outermost close
still re-validates it. Nothing is compared per borrow, so the `_file_stats`
walk the fix removed does not come back.
`asec_2024_native_population._epoch_exit` had the same ordering — present since
the original head, never anchored by a cheap-tier check — and is fixed the same
way. Both are pinned by
`test_a_roster_stat_moved_inside_an_inner_close_refuses_at_the_next_borrow` and
`test_a_native_source_moved_inside_an_inner_close_refuses_at_the_next_borrow`,
which move a source from a profile hook as that close's own validation returns
(rebinding a runtime callable refuses with `PRODUCER_CHANGED`, so observing it
is the only way in). Run against the previous ordering, in this worktree, both
fail with `Failed: DID NOT RAISE`; against this one the next borrow refuses
with `SOURCE_STAT_CHANGED` and `SOURCE_FILE_CHANGED` respectively.

**Before/after CPU.** Unit level, from
`test_an_epoch_validates_once_and_reuses_it`: six borrows of one preparation
cost six complete validations before (one `_source_files` pass each — the whole
3.48 GiB roster) and one after, plus one unconditional pass when the epoch
closes. Five of six borrows are memo hits; `_producer()` still runs on all six.

**The record.** `verification_epoch()` yields its own record — the protocol
label, the capsule count, the memo hits, the signature misses and the
unconditional final re-validations — and both runners now bind it and hand it
to `run_graph`, which attaches it to the manifest as
`RunManifest.verification_epoch`: outside the manifest key, outside its JSON,
outside every node receipt and cache record, exactly like `source_identities`.
On an actual nine-node atomic survey population run over invented sources
(`test_a_real_run_records_its_epoch_in_the_manifest`) the record reads
`{"capsules": 1, "hits": 20, "misses": 3, "final_validations": 1}`. So that run
borrowed the preparation capsule 23 times and paid four complete validations:
three signature misses inside the run and the one unconditional pass that
closed the epoch. **"Once per run" is the shape, not the literal count** — a
signature miss re-runs the full validation, and this run had three. The counts
come from a runner call over invented fixtures, not from the 19-node measured
run: the measurement harness predates the record and captures nothing of it.

Both runners are now pinned by a test, not only the nine-node one.
`test_nineteen_node_financial_cold_and_required_replay` reads the record off
both manifests the financial runner returns — its own epoch's on the outer
manifest, the nested nine-node population epoch's on the prefix's — and asserts
for the cold and the required-replay run that the protocol label is the epoch's,
that every capsule either epoch memoised was re-validated in full as it closed
(`final_validations == capsules >= 1`), and that `to_json` still cannot see any
of it. Delete `_verification_epoch=` from either runner and that test fails. It
asserts presence and closure, not counts: the counts are the nine-node test's,
because the financial fixture is module-scoped and shared, and pinning literal
hit and miss numbers there would bind an unrelated test's call pattern.

The measured runs still carry no counts, and no measurement in this report was
re-run to get them. What changed is the script:
`experiments/native-verify-once/probe_verify_once.py` now copies the returned
manifest's record into its output JSON and its console summary, as
`verification_epoch` (`null` until the runner returns, so every mid-run flush
and every ceiling exit carries `null`, and the before tree has no such record at
all). The three measurement files quoted below were written before that line
existed, so the next run of the committed probe is the first measured run that
will say how often it re-authenticated. The line is a read of a field that is
outside the manifest key, its JSON and every receipt, and the values are dropped
again immediately, so it changes no counted work and nothing the run is
identified by.

**Probe level.** There is no per-mechanism before/after ratio for this
mechanism, and the report does not manufacture one: the before probe stopped at
its CPU ceiling inside source admission (file A below,
`CEILING_REACHED_PARTIAL_MEASUREMENT`), so it never reached the steady state
this memo governs. What the after runs show is what remains. In the 19-node run
(file C, 2,010.07 process CPU s) the preparation's own identity work is
`asec_current_money_source._series_digest <- _frame_signature <-
asec_2024_native_population._frame_identity <-
survey_population_preparation._nested_seals` at **73.71 s (3.7 %)**, and the
seven ledger chains running `survey_population_preparation.update_cell <-
_frame_identity` under `survey_atomic_geography._population_stamp` sum to
**159.31 s (7.9 %)**, with
outer callers `run_atomic_survey_financial` (22.40 s),
`reconstruct_atomic_survey_geography` (26.18 s), a `<genexpr>` (60.80 s),
`survey_origin_budget._geography_binding` (16.06 s),
`_node_population_stamp` (15.97 s), `run_atomic_survey_population` (11.84 s) and
`observe` (6.07 s). How many full validations a run performs is pinned by the
unit test above, not inferred from a stack sampler.

### 2. `AuthenticatedAsec2024NativePopulation.frame`

Every `.frame`, `.context`, `.receipt` and `.to_bytes()` was a full
re-authentication including a SHA-256 of all seven ASEC source files.

**Guarantee, in two sentences.** `_encode(_implementation())` still runs on
every borrow, and the signature carries the stat identity of all seven source
files *and of their parent directories*, which `_file_identity` does not cover
because it opens `O_NOFOLLOW` on the final component only — so a roster change
reaches a memo that never lists the roster. The capsule's epoch closes **before**
its owner's: at the outermost close its memo is already cleared, so the owner's
final validation reaches the complete file check; at an inner nested close the
memo is live with a refreshed signature and the owner's validation is a memo
hit — unless a source moved while that close was validating, in which case the
close records no signature and the owner's validation is a miss that re-runs the
complete check. Nothing is skipped in net either way, because the capsule's own
exit has just re-validated it in full.

**Mutation tests.**
`test_the_native_capsule_still_refuses_a_changed_source_inside_an_epoch`
(`SOURCE_FILE_CHANGED` at the borrow, and again at the close);
`test_a_native_source_moved_inside_an_inner_close_refuses_at_the_next_borrow`,
added for the 2026-09-17 re-verification, which appends to a source from a
profile hook as the inner close's own `_validate_state` returns and asserts
`SOURCE_FILE_CHANGED` at the next borrow and again at the outer close; and,
added for the 2026-09-16 verification findings,
`test_the_native_capsule_refuses_a_rewritten_source_through_its_memo`: this
capsule's cheap tier compares no file stat at all, so a byte rewritten in place
with the length, the inode and the modification time preserved can only be
caught by the memo, and the test counts `_file_identity` calls to prove the
refusing borrow ran the complete `_validate_state` rather than a cheap check —
`SOURCE_FILE_CHANGED` at the borrow and again at the close. The existing
`test_changed_original_source_refuses_existing_borrow` is unchanged and still
passes: it runs outside an epoch.

**Before/after CPU.** `test_the_native_capsule_validates_once_per_epoch`: three
`validate()` calls cost three complete seven-file passes before and one after,
plus one at the close.

**Probe level.** In file C the ASEC capture cost appears under exactly two outer
callers: `asec_coverage_authentication._capture <- _reconstruct <-
authenticate_asec_coverage <- asec_population_catalogue.issue_asec_source_catalogue`
at **87.86 s (4.4 %)** and the same three inner frames under
`asec_2024_native_population.load_authenticated_asec_2024_native_population` at
**86.66 s (4.3 %)** — 174.52 s together. In the before probe (file A) the first
of those chains is 88.36 s and no
`load_authenticated_asec_2024_native_population` chain appears anywhere in its
60-chain ledger (whose cut is 2.11 s); because that run is truncated, the pair
is not a before/after comparison.

### 3. Per-node source content keys in the executor

`executor.py:2452-2457` re-derived `source_content_key` for every source a
cold-executed node declared; `keys._directory_identity` `read_bytes()`es every
file of the tree.

**Guarantee, in two sentences.** A cached key is reused only while the path's
stat signature — for a directory, its own identity plus the relative name, type
and identity of every entry `_directory_identity` would walk, and the resolved
identity of every entry that is a symlink, because `is_file()` and
`read_bytes()` both follow one — is identical to the signature taken both
immediately before and immediately after the read that produced the key, so
every mutation that moves any stat field still refuses at the same node, before
that node's `_write_node`. Every source is then re-derived
in full, cache bypassed, before the manifest is built, which also catches a
change made during a node that declares no source — something the per-node check
has never seen.

**Regression tests.** The refusal had **no test anywhere in the repository**
before this branch. `test_graph_executor_source_identity` now pins it:
`test_a_source_rewritten_by_its_own_node_refuses_that_node` (and asserts the
store gained no object), `test_a_file_added_to_a_directory_source_refuses`,
`test_a_file_removed_from_a_directory_source_refuses`,
`test_a_rewritten_source_still_refuses_when_only_its_bytes_moved`,
`test_a_source_changed_during_a_source_free_node_refuses_at_run_end`,
`test_a_symlinked_member_refuses_at_the_node_that_changed_its_target` (with
`test_a_directory_signature_follows_a_member_symlink` and
`test_a_broken_member_symlink_signs_as_absent_without_raising` on the signature
itself).

**Before/after CPU.** `test_each_source_is_read_twice_per_run_not_once_per_node`
and `test_an_unchanged_source_is_never_re_read_by_a_node`: two full derivations
per run regardless of how many nodes declare the source. The
`packages/microcosm-graph` suite fell from **108.28 s to 71.81 s** on the same
machine — a synthetic toy source, so the real saving scales with source bytes.
(The same suite is 46.66 s in the 2026-09-16 battery below on a differently
loaded machine: the node counts are the durable part of this row, the seconds
are not.)

**The record.** `RunManifest.source_identities`, attached exactly like
`populations` and `mass_ledgers`: `repr=False`, `compare=False`, outside
`content_addressed`, outside `to_json`, outside every node receipt and cache
record. `test_the_manifest_records_the_run_end_identities_without_moving_anything`
asserts the manifest's key, its JSON and its content-addressed body are
identical with and without it.

**Probe level.** No chain naming `executor.py` or `keys.py` appears in any of
the three ledgers. In file C the 60th and last listed chain is **5.00 s** of
2,010.07 CPU s, so per-node source-key derivation is below that cut on a run
that executed the 19-node graph and wrote **5,869** object files under
`.measure/after/harness19/graph-store/objects` (counted with `find … -type f`;
the v4 cold run's own store holds **5,866** — comparable volume, not identical
keys, because every node key moves with the stage `implementation_hash`). On
the before side absence proves nothing: neither `.measure/probe-before/graph-store`
nor `.measure/before/probe/graph-store` exists, although both probe scripts pass
`store_root=PROBE/"graph-store"`. That no node artifact was produced before the
cap ran out is an **inference** from the missing store plus the ledger's
contents; the receipts do not state it.

### 4. The ACS record fence

`acs_person_coverage_authentication._records` was a `for byte in block` loop
over 4,096-byte blocks, fencing 2.4 GB of ACS person CSV.

**Guarantee, in two sentences.** The state machine is unchanged — quote parity
still persists across records, CR/LF/CRLF are still preserved, and both ceilings
are still charged in the same order — only the boundaries are now located with
`bytes.find` and `bytes.count`. A record no longer than the smaller of the two
live ceilings cannot have violated either, because the token counter resets at
every record start; anything longer is replayed byte by byte, so the refusal
code and the byte it fires on stay the fence's own.

**Equality proof.** `test_us_acs_record_fence_scan` compares the shipped fence
against a verbatim copy of the byte loop on every input, comparing the yielded
record sequence and the refusal together: 20 boundary cases, 6 block-boundary
cases, 8 ceiling cases on both sides of every cap, 4,200 random strings, 300
row-shaped inputs, and **all 9,331 strings up to five bytes over the branching
alphabet under each of five ceiling settings — 46,655 exhaustive comparisons**.
On the real staged archive
(`experiments/native-verify-once/record-fence-parity.json`, produced by
`record_fence_real_archive.py`):

| member | bytes | records | digest | byte loop | chunked scan | speedup |
|---|---|---|---|---|---|---|
| `psam_pusa.csv` | 1,226,543,489 | 1,743,752 | `da955ca2…347fc827` identical | 116.72 s | 2.38 s | 49.1× |
| `psam_pusb.csv` | 1,178,924,624 | 1,679,138 | `9bbf6af8…90184bba` identical | 110.14 s | 2.25 s | 48.9× |

**226.86 s → 4.63 s per pass**, 3,422,890 records, digests identical.

**Probe level — the one chain-level before/after the partial run supports.**
`__init__.py:_read1 <- __init__.py:read <-
acs_person_coverage_authentication.py:_records <- _inventory` is **rank 1 at
760.23 s** (3,061 samples, 42.1 % of the before probe's partial 1,803.87 CPU s)
in file A, and it was still running when the ceiling hit, so that figure is a
floor. The identical chain is **rank 23 at 10.80 s** (42 samples, 0.7 %) in the
after probe (file B) and **rank 31 at 11.98 s** (46 samples, 0.6 %) in the
19-node run (file C).

### 5. The kernel context digest

`_update_series` boxed every value of a plain float, integer or boolean column
into a Python object and framed them one at a time; on pandas 3.0.3 such a
column is a `NumpyExtensionArray` with no `_data`/`_mask`, so the existing fast
path never applied to it.

**Guarantee, in two sentences.** `_object_stream` emits the identical framed
bytes — the same `object` dtype header, the same shape header, the same
per-value length prefixes and payloads — and returns `None` for every column
kind it cannot reproduce exactly. Both `_context_digest` calls per node remain,
because the second one is the "did the kernel mutate its context" guard and
reusing its result would delete the guard.

**Byte-equality proof.** `test_graph_executor_series_stream` compares the live
helper against a verbatim copy of the pre-change body **at the byte level**, not
only at the digest: 19 column kinds, the float specials (signed zero, both
infinities, the subnormal extremes), non-canonical NaN payloads with their sign
bit, both `int64` endpoints, empty/sliced/strided columns, labelled indexes, and
a 200-frame random sweep over ten kinds.

**The mutation guard, through the new path.** The second `_context_digest`
call per node is the "did the kernel mutate its context" guard, and it is
proven to still fire after the change by a test that predates it:
`test_executor_detects_mutation_even_when_pandas_replaces_a_buffer`
(`packages/microcosm-graph/tests/test_graph_executor.py:610-640`) adds one to
the int64 `age` column inside a kernel and asserts the run refuses with
`Node … mutated its input context`. That column now streams through
`_object_stream`'s integer branch, so the guard is exercised through the new
code rather than around it. The float and bool branches have **byte parity
only**: no test mutates a float or bool column inside a kernel. Byte-identity
makes the guard equivalent by construction on those branches, which is why this
is a coverage statement and not a contract gap — but it is stated rather than
implied.

The first version of the fast path mistook a `Categorical` for an integer column
because `Categorical._ndarray` is its codes array; that is now a named test,
`test_categorical_is_not_mistaken_for_its_integer_codes`, and the shipped guard
keys on `series.dtype` being a plain numpy dtype rather than on any private
attribute.

**Before/after CPU**, 6,928 × 240 frame: **0.466 s → 0.094 s**. Per value:
float64 182.6 → 17.9 ns, bool 225.6 → 13.9 ns, int64 383.3 → 119 ns.

**Probe level.** No `_update_series`, `_object_stream` or `_context_digest`
chain appears in any of the three ledgers; in file C that places the whole
mechanism under the 5.00 s cut of a 2,010.07 CPU-s run. The unit-level number
above is the measurement that carries this mechanism; the probe neither
confirms nor contradicts it.

## Digest equality, in one place

The rule the change obeys is that memoisation moves **when** a check runs, never
**what** it computes. Three proofs carry it, and they are of two different
kinds.

**Proven by rebuilding the bytes and comparing them.** Two mechanisms rebuild
bytes, and each is compared against a verbatim copy of the pre-change
implementation rather than against a description of it:

| what | comparison | scale |
|---|---|---|
| `_records`, the ACS record fence (mechanism 4) | shipped scan vs a verbatim copy of the byte loop — yielded record sequence **and** refusal compared together | 20 boundary + 6 block-boundary + 8 ceiling cases, 4,200 random strings, 300 row-shaped inputs, and all 9,331 strings up to five bytes over the branching alphabet under each of five ceiling settings (46,655 exhaustive comparisons); then the real staged `csv_pus.zip`: 3,422,890 records, `da955ca2…347fc827` and `9bbf6af8…90184bba` identical on both sides (`experiments/native-verify-once/record-fence-parity.json`, `"identical": true`) |
| `_object_stream` / `_update_series` (mechanism 5) | shipped helper vs a verbatim copy of the pre-change body, compared **at the byte level**, not at the digest | 19 column kinds, float specials (signed zero, both infinities, subnormal extremes), non-canonical NaN payloads with sign bit, both `int64` endpoints, empty/sliced/strided columns, labelled indexes, 200-frame random sweep over ten kinds |

**Proven by not rebuilding anything.** The other three mechanisms compute no
receipt field. `_validate`, `_validate_state` and `source_content_key` only
compare live values against values frozen at issuance, so deferring a comparison
cannot move a recorded value; `_validate` and `_validate_state` are themselves
untouched, and the memo is opt-in and scoped, so outside a `verification_epoch()`
the capsules behave byte for byte as they do today. Two new things are
recorded, and each is pinned not to move anything.
`RunManifest.source_identities` has
`test_the_manifest_records_the_run_end_identities_without_moving_anything`,
which asserts the manifest's key, its JSON and its content-addressed body are
identical with and without it. `RunManifest.verification_epoch` had no test in
the graph shard at all until 2026-09-17 — only a single `not in to_json()` line
in the build shard, which the graph-shard PR does not carry — and now has
`packages/microcosm-graph/tests/test_graph_verification_epoch.py`, which
asserts the same three identities both against a bare manifest and against a
separate cold run with no record, plus the live-view property the field exists
for and each of the three `__post_init__` refusals.

**Proven by recomputation over the whole inventory.** Every contract in
`graph_implementation_inventory.json` was recomputed through
`graph_implementation._dependency_contract` and all ten stage manifests were
built against the file as it stands on this branch — that is, after the two pins
below were re-derived: `moved: 0`, `all stage manifests build`. The `git diff`
of that file against the base branch is exactly two `unbound_uses_sha256`
lines.

## The five copies of one stat helper, and why four of them stay

`_path_stat` -- one path's `(st_dev, st_ino, st_size, st_mtime_ns,
st_ctime_ns)`, or `("absent", errno)`, never raising -- is written out verbatim
in four modules: `acs_native_coverage_binding.py:415`,
`acs_population_catalogue.py:178`, `asec_2024_native_population.py:456` and
`asec_population_catalogue.py:423`. A fifth copy, `_stat_or_absent` in
`survey_population_preparation.py`, had no caller at all and is deleted.
`docs/shared-constants.md` asks for one definition of shared static data, so
the remaining duplication is a deliberate choice and is stated here rather than
left to be discovered.

Consolidating them costs inventory pins. `graph_implementation._dependency_details`
records every relative import as an unbound use (`import:microcosm.build.us_runtime:<module>`)
and every *scope-qualified* use of the resulting alias
(`<scope>:microcosm.build.us_runtime:<alias>`), and `_dependency_contract`
hashes that set into `unbound_uses_sha256`, which
`graph_implementation_inventory.json` pins per module and
`implementation_manifest` refuses on mismatch. Importing a shared helper into
`asec_population_catalogue.py` was tried against the module's own generator to
check rather than assume: `imports` does not move (the package is already
there), and `unbound_uses_sha256` does. Four capsule modules would therefore
re-pin, on top of the two `unbound_uses_sha256` pins this lane already moves,
in the same file that binds what each authentication capsule is allowed to
reach. That is a larger and more security-relevant diff than the duplication it
removes, so the four copies stay and this lane does not widen its pin surface
for a style fix. A lane that consolidates them should do it as its own change,
with those four pins re-derived and stated.

The same argument applies to `_stat_identity`, which exists in both
`survey_population_preparation.py:268` and `executor.py:2105`, but it lands on
a different field. Those are different packages, so
`from microcosm.graph.executor import _stat_identity` moves that module's
pinned `imports` list rather than its `unbound_uses_sha256`: run through the
module's own generator, `imports` gains `microcosm.graph.executor` and
`unbound_uses_sha256` does **not** move, because every stage that carries this
file declares the graph package as a dependency and `_covered_imports` binds
everything beneath it. No new `import_classifications` entry would be needed —
`microcosm.graph.executor` is already classified there, since other modules
already import it — but the contract pin for `survey_population_preparation.py`
would move all the same, and it is the same file and the same review surface as
the four above.

All five measurements in the two paragraphs above were re-derived on
2026-09-16 through `graph_implementation._dependency_contract` against the
module's own `_covered_imports`, by adding the import and one use to a copy of
each module's bytes in memory and comparing the contract with the pinned one:
`unbound_uses_sha256` moves for all four `_path_stat` modules and `imports`
does not; for `survey_population_preparation.py` it is the other way round.

## Pins re-derived

| pin | old | new | command |
|---|---|---|---|
| `acs_native_coverage_binding._ACCEPTED["acs_person_coverage_authentication.py"]` | `475aa795c8a5b49a0dd3405a0877866dddd03012a1f2b9743447fab1fe85bcff` | `9ec68721a4cf480ef412c51ab354db9000eb7a7989e35e6e09b1574d88e8e49f` | `shasum -a 256 packages/microcosm-build/src/microcosm/build/us_runtime/acs_person_coverage_authentication.py` (re-derived after the final formatting pass; a stale pin refuses with `UNREVIEWED_PREPARATION`) |
| `graph_implementation_inventory.json` → `survey_population_preparation.py` → `unbound_uses_sha256` | `29c09f6fcb25ce8bc6111dd0dc76f3501d9a6b635929e94b80835889d296ef91` | `d114117ca19e9b866982147e5497d601ed91ca603d0568730d4051bdd4005910` | `graph_implementation._dependency_contract(payload, name, _covered_imports(name, inventory))` — the module's own generator |
| `graph_implementation_inventory.json` → `asec_2024_native_population.py` → `unbound_uses_sha256` | `71463df4f645cd2a98f537a212187b4d1e52bffcc4d2317defe6968d7f287608` | `7bb20439da26f75b48551e21e304c4bb7f3983ee07ef78ed804411640d58cdf4` | same |

No other pin moves. Verified by recomputing **every** contract in the inventory
through `graph_implementation._dependency_contract` and building all ten stage
manifests: `moved: 0`, `all stage manifests build`. In particular the three
foreign owners that gained a `_verified_source_stats()` helper
(`acs_population_catalogue`, `acs_native_coverage_binding`,
`asec_population_catalogue`) and both graph-shard modules keep their contracts
unchanged, because the additions introduce no `_RESOURCE_CALLS` name and no
non-stdlib import.

**Re-verified while writing this report**, in the clean worktree at `284bc6e96`:
`shasum -a 256` on `acs_person_coverage_authentication.py` returns
`9ec68721…d88e8e49f` and `acs_native_coverage_binding.py:38` holds that exact
string; `git show origin/microcosm-us-launch-integration-20260909:<same path> |
shasum -a 256` returns `475aa795…fe85bcff` and the base file's line 38 holds it,
so both halves of the first row are checked against the tree rather than quoted
from memory; and
`git diff origin/microcosm-us-launch-integration-20260909..HEAD --
…/graph_implementation_inventory.json` is exactly the two
`unbound_uses_sha256` lines of rows two and three, with no other line changed in
that file.

**The third fix pass moves no pin either**, and this was re-derived rather than
assumed: `graph_implementation._dependency_contract` was run over the edited
bytes of both changed modules against `_covered_imports(name, inventory)`, and
`imports`, `resource_accesses_sha256` and `unbound_uses_sha256` are all
unchanged for `survey_population_preparation.py` and for
`asec_2024_native_population.py`. The pass adds one module-level helper to each
file and touches no import and no resource call, which is why. Neither file
carries a byte digest anywhere else in the tree: `grep` for both filenames
across `packages/*/src` returns only the inventory's own contract and roster
entries.

**One thing that does move, by design and not by this lane's choice.** Each US
stage's `implementation_hash` is `_digest(module bytes)` over its whole module
roster, so editing any inventoried module moves it, and with it every node key.
That is true of any change to these files, including a comment. It is not a
digest this lane computes; the digests this lane computes — frame identities,
source file digests, context digests, source content keys, seals, receipts — all
keep their exact values, and the tests above prove it byte for byte where bytes
are rebuilt.

## Test summaries, verbatim

### The lane's own runs, as each mechanism landed

**These eight lines are historical, and they are not reproduced anywhere else
in this report.** Each was pasted from the run the lane made as that mechanism
landed, at that day's branch head and on whatever machine state existed then;
none of them is a verdict about the branch as it now stands, and the test
counts in several of them have since moved, because the verification findings
added tests. What stands for the branch now is the CI-shaped battery below and
the section "After the verification findings" that follows it. They are kept
because they are the record of what each mechanism cost as it landed.

```
$ .venv/bin/python -I -B -m pytest packages/microcosm-graph/tests -p no:randomly
766 passed, 1 skipped, 1 warning in 71.81s (0:01:11)
```

```
$ .venv/bin/python -I -B -m pytest packages/microcosm-graph/tests/test_graph_executor_series_stream.py
34 passed in 0.75s
```

```
$ .venv/bin/python -I -B -m pytest packages/microcosm-graph/tests/test_graph_executor_source_identity.py -p no:randomly
15 passed in 0.56s
```

```
$ .venv/bin/python -I -B -m pytest packages/microcosm-build/tests/test_us_acs_record_fence_scan.py -p no:randomly
52 passed in 1.08s
```

```
$ .venv/bin/python -I -B -m pytest \
    packages/microcosm-build/tests/test_us_acs_person_coverage_authentication.py \
    packages/microcosm-build/tests/test_us_acs_source_compile_cache.py \
    packages/microcosm-build/tests/test_us_acs_person_coverage_columns.py \
    packages/microcosm-build/tests/test_us_acs_housing_source.py \
    packages/microcosm-build/tests/test_us_acs_record_fence_scan.py -p no:randomly
544 passed, 5 warnings in 14.33s
```

```
$ .venv/bin/python -I -B -m pytest \
    packages/microcosm-build/tests/test_us_survey_population_preparation.py \
    packages/microcosm-build/tests/test_us_asec_2024_native_population.py \
    packages/microcosm-build/tests/test_us_acs_source_compile_cache.py -p no:randomly
112 passed, 2 warnings in 140.94s (0:02:20)
```

```
$ .venv/bin/python -I -B -m pytest packages/microcosm-build/tests/test_us_native_verify_once_epoch.py -p no:randomly
17 passed in 161.69s (0:02:41)
```

```
$ .venv/bin/python -I -B -m ruff check .
All checks passed!
```

```
$ python3 -I -B -S tools/ci_test_groups.py --verify
verification=ok
```

```
$ uv lock --check
Resolved 125 packages in 6ms
```

**CI does not run on this pull request, by design.** `.github/workflows/test.yml`
triggers on `pull_request: branches: [main]`, and this PR is stacked on
`microcosm-us-launch-integration-20260909`. Only that branch's own PR (#893)
reaches CI. Everything below was therefore run locally, against the same
`uv sync --all-packages --locked --extra us` environment CI uses
(Python 3.14.4, pandas 3.0.3, numpy 2.4.6).

All four new test files land in the groups they should and none under
`[defaulted]`: `test_us_acs_record_fence_scan.py` and both graph files in
`rest` / `us-am` / `wheels`, `test_us_native_verify_once_epoch.py` in `rest` /
`us-not` / `wheels`.

### The CI-shaped battery, 2026-09-16

Run on this branch at `284bc6e96` with the worktree clean and no measurement
running, and `PYTHONDONTWRITEBYTECODE=1` on every pytest invocation so nothing
was written into the package trees. Logs were kept outside the worktree in an
ephemeral session scratchpad (`…/scratchpad/ci/g1.log` … `g5b.log`), so the
lines below are the durable record: each is pasted exactly as the run printed
it. Note that the repo's
`addopts = "-q --import-mode=importlib"` makes an explicit `-q` into `-qq`,
which prints **no** summary line at all; these runs therefore use the repo's own
verbosity, which is also CI's invocation.

```
$ uv lock --check                                                    # rc 0
Resolved 125 packages in 5ms
```

```
$ uv run --no-sync ruff check .                                      # rc 0  (the lint lane's actual command)
All checks passed!
```

```
$ uv run --no-sync ruff format --check .                             # rc 1  (NOT run by CI; pre-existing, see Status)
77 files would be reformatted, 1110 files already formatted
```

```
$ uv run --no-sync ruff format --check $(git diff --name-only origin/microcosm-us-launch-integration-20260909..HEAD | grep -E '\.py$')   # rc 0
17 files already formatted
```

```
$ PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest packages/microcosm-graph/tests -p no:cacheprovider    # rc 0
766 passed, 1 skipped in 46.66s
```

```
$ PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest <62 files under packages/microcosm-build/tests> -p no:cacheprovider --durations=25   # rc 0
2350 passed, 3 skipped, 438 warnings in 11678.22s (3:14:38)
```

```
$ PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest \
    packages/microcosm-graph/tests/test_graph_executor_series_stream.py \
    packages/microcosm-graph/tests/test_graph_executor_source_identity.py -p no:cacheprovider   # rc 0
49 passed in 0.41s
```

```
$ UV_PROJECT_ENVIRONMENT=.venv-noengine uv sync --all-packages --locked         # rc 0
Installed 73 packages in 405ms
```

```
$ UV_PROJECT_ENVIRONMENT=.venv-noengine PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest packages/microcosm-graph/tests -p no:cacheprovider   # rc 0
765 passed, 2 skipped in 48.63s
```

```
$ python3 tools/ci_test_groups.py --verify                           # rc 0
tracked_test_files=551
verification=ok
```

```
$ python3 -I -B -S packages/microcosm-build/tests/test_ci_test_groups.py   # rc 0
Ran 15 tests in 0.298s -- OK
```

```
$ uv run --no-sync python tools/graph_acceptance_burndown.py --verify        # rc 0
verification=ok
```

`uv lock --check`, `ruff check .`, both `ci_test_groups` checks and the
acceptance burndown were re-run while writing this report and gave the same
verdicts (the stdlib matrix check reported `Ran 15 tests in 0.317s` / `OK` on
the second run — wall times move, verdicts do not).

**The 62-file selection.** The union of (a) the 47 build tests naming any
touched `us_runtime` module or `verification_epoch` and (b) the 20 build tests
importing `microcosm.graph.{executor,manifest,codecs}` by dotted path; 2,353
tests, and it already contains both new build test files. Twenty-seven further
build tests that reach the graph shard through the package-level re-export
(`from microcosm.graph import …`) — `test_us_graph.py`,
`test_us_survey_calibration.py`, `test_uk_graph.py` among them — were **not**
run; that is an open question below. The two new graph test files live in
`packages/microcosm-graph/tests`, so they ran inside the graph suite and again
standalone.

**The three skips in that run**, all expected: `test_uk_uc_capital_coherence.py`
#13 (`requires_uk`; `policyengine-uk` is not installed in this `.venv`), and
`test_us_atomic_block_api_sources_native_de.py` #1 and
`test_us_atomic_block_api_sources_native_national.py` #1, which self-skip with
"requires exact source pins from the external native DE guard", documented in
their own headers. `policyengine-us` **is** installed, so the US engine tests
really ran.

**The engine-free skip delta is the marker contract working.** 765 passed /
2 skipped against 766 / 1: the extra skip is
`packages/microcosm-graph/tests/test_acceptance_h_parity.py:319`
"requires policyengine-us extra"; the skip common to both environments is the
same file's `:245` "requires policyengine-uk extra". Verified by running that
one file with `-rs` in each environment.

**Wall times in this battery are contended, not clean-machine numbers.** During
the 62-file run an unrelated lane in `~/ThesisInstitute/procurement` was using
~187 % and ~168 % CPU and the 1-minute load average peaked at 34.68. CI splits
those same files across `us-am` / `us-p` / `us-qs` / `us-not` shards in
parallel; this was one serial process. Its three slowest items, from
`--durations=25`, are real cold graph runs against staged sources:
`test_us_graph_atomic_completion_host.py::test_roles_disabled_status_enabled_49_cold_required_and_output_identity`
1,228.96 s, `::test_roles_disabled_status_disabled_45_cold_required` 923.75 s,
and 826.19 s of setup for
`test_us_graph_us_survey_enrichment.py::test_actual_enrichment_cold_required_and_complete_parent_preservation`.
The same caution applies to the graph-suite figure quoted in mechanism 3
(108.28 s → 71.81 s on one machine state; the same suite is 46.66 s here): the
node counts are the durable part, the seconds are not.

### After the verification findings, 2026-09-17

An adversarial static verification of this branch on 2026-09-16 returned four
medium and seven low findings and no high one. Nine commits resolve them, and
one more restates this report; what follows is what was re-run afterwards, at
the branch head those commits produce, on the base tip `363a9033b`. The
invocations are the battery's, with the repo's own verbosity: the `addopts`
`-q` makes an explicit `-q` into `-qq`, which prints no summary line at all.
Logs are outside the worktree in
`~/PolicyEngine/_recovered/scratch-backup/893/lanes/verify-once-tests-20260916/`.

```
$ PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest packages/microcosm-graph/tests -p no:cacheprovider   # rc 0
769 passed, 1 skipped in 44.56s
```

```
$ PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest \
    packages/microcosm-build/tests/test_us_native_verify_once_epoch.py \
    packages/microcosm-build/tests/test_us_acs_record_fence_scan.py \
    packages/microcosm-graph/tests/test_graph_executor_series_stream.py \
    packages/microcosm-graph/tests/test_graph_executor_source_identity.py -p no:cacheprovider   # rc 0
128 passed in 188.39s (0:03:08)
```

The same four files one at a time, so each new test lands against a named file:

```
$ … pytest packages/microcosm-build/tests/test_us_native_verify_once_epoch.py -p no:cacheprovider      # rc 0
24 passed in 168.66s (0:02:48)
```

```
$ … pytest packages/microcosm-build/tests/test_us_acs_record_fence_scan.py -p no:cacheprovider         # rc 0
52 passed in 0.48s
```

```
$ … pytest packages/microcosm-graph/tests/test_graph_executor_series_stream.py -p no:cacheprovider     # rc 0
34 passed in 0.27s
```

```
$ … pytest packages/microcosm-graph/tests/test_graph_executor_source_identity.py -p no:cacheprovider   # rc 0
18 passed in 0.32s
```

```
$ PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest <46 files under packages/microcosm-build/tests> -p no:cacheprovider   # rc 0
1811 passed, 1 skipped, 38 warnings in 11200.54s (3:06:40)
```

```
$ uv run --no-sync ruff check .                                      # rc 0
All checks passed!
```

```
$ uv run --no-sync ruff format --check $(git diff --name-only origin/microcosm-us-launch-integration-20260909...HEAD | grep -E '\.py$')   # rc 0
17 files already formatted
```

```
$ uv lock --check                                                    # rc 0
Resolved 125 packages in 4ms
```

```
$ python3 tools/ci_test_groups.py --verify                           # rc 0
verification=ok
```

```
$ python3 -I -B -S packages/microcosm-build/tests/test_ci_test_groups.py   # rc 0
Ran 15 tests in 0.347s

OK
```

**The 46-file selection** is every file under `packages/microcosm-build/tests`
that names `survey_population_preparation`, `graph_atomic_survey_financial`,
`graph_atomic_survey_population` or `verification_epoch`, or that imports
`microcosm.graph.executor` or `microcosm.graph.manifest` by dotted path — that
is, every build test that reaches a module the findings commits changed. It is a
subset of the 62-file battery above and it contains
`test_us_native_verify_once_epoch.py`. Files that reach the graph shard only
through the package re-export are open question 5, unchanged.

**The one skip** is `test_uk_uc_capital_coherence.py:345`, "requires
policyengine-uk extra" — the same marker skip as the battery, confirmed by
re-running that file alone with `-rs` (`12 passed, 1 skipped in 30.85s`). No
other file skipped, because the two native-DE self-skips are in files this
selection does not include.

**Counts that moved, and why.** The graph suite is 769 / 1 against the battery's
766 / 1: three new tests in `test_graph_executor_source_identity.py` for the
member-symlink follow. That file is 18 rather than 15 for the same reason, and
`test_us_native_verify_once_epoch.py` is 24 rather than 17: two tests that reach
the memo-miss refusal itself, four parametrisations pinning that an in-epoch
refusal carries the code an unmemoised borrow carries, and one that reads a real
run's epoch record off its manifest. `test_us_acs_record_fence_scan.py` (52) and
`test_graph_executor_series_stream.py` (34) are unchanged in count; the second
re-ran because the float branch now casts to `np.float64`, and its byte-identity
comparison passes with that cast as it did with `<f8`.

**Wall times here are contended too.** The 46-file run shared the machine with
this session's own work and with the 1-minute load average between 3.7 and 5.2
throughout; 3 h 06 m of one serial process is not a clean-machine number, and CI
splits the same files across parallel shards.

**What was not re-run.** Everything above this section: the 62-file battery, the
engine-free environment, the acceptance burndown, the before/after probe and the
19-node harness. The findings commits changed three source files
(`survey_population_preparation.py`, `executor.py`, `manifest.py`) and two
runner call sites, so the measurement they would repeat is the same measurement,
and no measured number in this report was taken again.

### After the second fix pass, 2026-09-17

A second pass added the epoch-record assertions to the nineteen-node financial
test and the epoch read to the committed probe. No `packages/*/src` file changed
in it, so the two test files that cover the epoch record are the whole of what
it could break:

```
$ uv run --no-sync pytest packages/microcosm-build/tests/test_us_graph_atomic_survey_financial.py -p no:cacheprovider   # rc 0
7 passed in 119.98s (0:01:59)
```

```
$ uv run --no-sync pytest packages/microcosm-build/tests/test_us_native_verify_once_epoch.py -p no:cacheprovider        # rc 0
24 passed in 157.61s (0:02:37)
```

Both counts are unchanged — the new assertions went into an existing test rather
than adding one — and the epoch file was re-run because it is the twin that owns
the record's counts. `ruff check` and `ruff format --check` are clean on the two
changed Python files. The probe itself has no test and was not run: running it
is a measurement, and nothing here re-measures.

### After the third fix pass, 2026-09-17

The third pass changed two `packages/*/src` files —
`survey_population_preparation.py` and `asec_2024_native_population.py` — so the
selection is every build test file naming either of them, 34 files, plus the
epoch file itself, the graph shard in full, and the new graph test file inside
it. All rc 0:

```
$ uv run --no-sync pytest packages/microcosm-graph/tests -p no:cacheprovider                       # rc 0
778 passed, 1 skipped in 37.72s
```

```
$ uv run --no-sync pytest packages/microcosm-build/tests/test_us_native_verify_once_epoch.py -p no:cacheprovider   # rc 0
28 passed in 190.65s (0:03:10)
```

```
$ uv run --no-sync pytest $(the 34 build test files naming either changed module) -p no:cacheprovider   # rc 0
1607 passed, 38 warnings in 8311.33s (2:18:31)
```

```
$ uv run --no-sync pytest packages/microcosm-build/tests/test_us_asec_prepared_resources.py -p no:cacheprovider    # rc 0
8 passed in 0.68s
```

```
$ uv run --no-sync pytest packages/microcosm-graph/tests -p no:cacheprovider   # rc 0, in the graph-verify-once-main worktree
597 passed, 2 skipped in 17.75s
```

The graph shard rises from 769 to 778 because
`test_graph_verification_epoch.py` adds nine; the epoch file rises from 24 to 28
because the pass adds two close-window tests and two branch tests. The 34-file
selection is narrower than the second pass's 46 and the battery's 62 because
this pass changed no graph-shard source at all — the only graph-shard change is
the new test file, which runs inside the shard suite above and again in the
mirror worktree, where `ruff check .` is also clean. `test_us_asec_prepared_resources.py`
was added by hand: it is the file that builds the stage implementation
manifests, and it does not name either changed module, so the mechanical
selection misses it.

Also clean at the same head: `uv run --no-sync ruff check .`
(`All checks passed!`), `ruff format --check` on all five changed Python files
(`5 files already formatted`), `uv lock --check`,
`python3 tools/ci_test_groups.py --verify` (`verification=ok`, with the new
graph file in `fast`, `engine` and `wheels` and not under `[defaulted]`), and
`python3 -I -B -S packages/microcosm-build/tests/test_ci_test_groups.py`
(`Ran 15 tests in 0.284s` / `OK`).

One more line changed in the committed probe and was **not** run, for the same
reason as last pass: `probe_verify_once.py` read `manifest.verification_epoch`
unguarded, which would label an otherwise-complete before-tree run
`STOPPED_AttributeError`, since that tree has no such field. It now reads with a
`getattr(…, None)` default and keeps `None` as the "this tree carries no record"
value. No quoted figure is affected — both before probes stopped at the CPU
ceiling first — and nothing was re-measured.

## Measurement

All figures in this section were read from the files named in the "file"
column. Nothing was re-run to produce them and no source file was touched; the
worktree was clean at `284bc6e96` throughout.

### The files

| | file | what it is |
|---|---|---|
| A | `.measure/repeated-verification-before.json` | 9-node prefix, **before** (base `f7bb88525`, source tree `.measure/source-before`) |
| B | `.measure/after/probe/repeated-verification-measurement.json` | 9-node prefix, **after** (`284bc6e96`, source tree `.measure/source-after`) |
| C | `.measure/after/harness19/repeated-verification-measurement.json` | 19-node financial graph, **after** |
| D | `~/PolicyEngine/_recovered/pilot-runs/native19-v2/run/us-native-postclone-financial19-v4-cold.json` | 19-node financial graph, **before** = the accepted v4 cold receipt |
| E | `~/PolicyEngine/_recovered/pilot-runs/native19-v2/run/native-financial-manifest.json` | the v4 run's `timing.case_phases`, which isolates the runner call |

A, B and C are under `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-verify-once/`
and are gitignored measurement artifacts, not committed. Every one carries
`"release_eligible": false` and a scope line ending "not a build, not a
certification, not release eligible".

**Provenance, verified.** `.measure/source-before/head.txt` is
`f7bb88525a78786f91bc3ebe2083ef4b1c85de18` and its
`acs_person_coverage_authentication.py` hashes to `475aa795…fe85bcff`, the old
`_ACCEPTED` pin; `.measure/source-after/head.txt` is
`284bc6e9624d211ba9c16dcdf04df63b9e87b10c` and the same module hashes to
`9ec68721…d88e8e49f`, the new one. So the before run measured unchanged code and
both after runs measured this branch. B and C both record
`source_tree: .measure/source-after` and
`staged_run_inputs: ~/PolicyEngine/_recovered/pilot-runs/native45-v5/run`.

### Run level, as recorded

| run | status in file | wall s | CPU s | peak RSS (bytes / decimal GB) | file |
|---|---|---|---|---|---|
| 9-node prefix, before | `CEILING_REACHED_PARTIAL_MEASUREMENT` | 1,853.894 | **1,803.870** (cap 1,800) | 8,612,429,824 / 8.61 | A |
| 9-node prefix, after | `COMPLETED_PREFIX` | 1,451.762 | **1,444.776** | 12,285,526,016 / 12.29 | B |
| 19-node graph, after | `COMPLETED_NINETEEN_NODE` | **2,016.079** | **2,010.072** | 13,309,181,952 / 13.31 | C |
| 19-node graph, before (v4 cold receipt) | `result: 0` | 5,362.050 | 5,278.611 | 9,313,337,344 / 9.31 | D |
| — of which the v4 `run_atomic_survey_financial` call alone | — | 4,685.862 | 4,605.351 | — | E |

### The 9-node probe: what it can and cannot support

The before probe is **partial**, so there is no measured ratio for the prefix.

| comparable | before | after |
|---|---|---|
| Completed the prefix inside the 1,800 CPU-s cap? | **no** — ceiling reached at 1,803.87 CPU s | **yes** — 1,444.78 CPU s, 355.2 CPU s of headroom |
| Graph store written under the probe directory | none exists | 4,363 object files |
| Peak RSS | 8.61 GB — a **floor**, the run was cut short | 12.29 GB, whole prefix |

The defensible statements are: (a) the after run completes the whole prefix in
0.80× of the CPU the before burned **without finishing**, i.e. a **lower bound
of ≥ 1.25×** on the CPU reduction, with the true ratio unknown and certainly
larger; (b) per-chain absolute CPU at the moment each run ended. Share-of-process
percentages on the before side are shares of a truncated run and are **not**
comparable with the after's. Chains whose absolute CPU is higher in B than in A
— `_capture` under `load_authenticated_asec_2024_native_population` (absent from
A's ledger entirely, 80.19 s in B), `_require <- feed <- _capture <-
_reconstruct` (17.61 s → 29.56 s) and `_series_digest` under `_nested_seals`
(absent from A, 28.07 s in B) — must **not** be read as regressions: the before
run never reached that work.

### The 19-node harness: like-for-like at identical arguments

`.measure/harness19_verify_once.py` is `.measure/probe_verify_once.py` with only
the docstring, the ceilings (7,200 CPU / 9,000 wall / 32 GiB), one import, the
scope and status strings and the call changed. The call is
`run_atomic_survey_financial(RUN/"sources", snapshot_root=…, store_root=…,
fraction=Fraction(1, 1000), seed=20260908, geography_config=geography(seed
20260908), demographic_conditioning=True, n_estimators=2, resume="auto",
return_values=True)` against a cold store — the same arguments as the v4 test's
own call
(`~/PolicyEngine/_recovered/pilot-runs/native19-v2/run/test_native_nineteen_node_financial.py:126-137`).

| comparison | wall | CPU | peak RSS |
|---|---|---|---|
| v4 whole process (D) → after harness (C) | 5,362.05 s → 2,016.08 s (**2.66×**) | 5,278.61 s → 2,010.07 s (**2.63×**, −3,268.54 s, −61.9 %) | 9.31 GB → 13.31 GB (**1.43× higher**) |
| v4 runner call only (E) → after harness total (C) | 4,685.86 s → 2,016.08 s (**≥ 2.32×**) | 4,605.35 s → 2,010.07 s (**≥ 2.29×**) | — |

The second row is the conservative framing: C is a whole-process total including
imports, while E excludes everything outside the runner call. The v4 process
spent 676.2 s wall / 673.3 s CPU outside its runner — imports and collection,
artifact export and readback, the pending report and final checks, the guard
postcheck — none of which the harness runs at all.

**Differences between C and D that are not this change**, from the harness
docstring (`.measure/harness19_verify_once.py:12-17`, verbatim): "it ran on
source snapshot ``2ca11c85a``, and its ``person-income-attachment.h5`` was the
pre-restoration ``5996dcdd...`` rather than the staged ``9ebc0ef2...``. Both
files are exactly 666,333,922 B". The v4 receipt's `native_before` and
`native_after` both carry `5996dcdd…` at 666,333,922 B and the staged file is
666,333,922 B; the `9ebc0ef2…` value is the lane's claim and was not re-hashed
for this report. Load context also differs and runs against the after: D records
no load average; C started at `[7.34, 7.54, 8.09]` and ended at
`[7.83, 9.33, 11.38]`, i.e. it shared the machine.

**The RSS rise is real and is by design**: memoised authentication keeps
validated state resident. The like-for-like pair is D → C, 9.31 → 13.31 GB
(1.43×), well inside that run's 32 GiB ceiling. The probe pair's 8.61 → 12.29 GB
is an **upper bound** on the rise, not a measurement of it, because the before
figure is a floor.

### What is left, 19-node after run (file C)

Process CPU 2,010.07 s. These ten chains are 864.05 s = **43.0 %** of it; the 60
listed chains sum to 1,464.77 s of the 2,007.83 s sampled total, so ~543 s sits
below the ledger's cut of 5.00 s.

| # | CPU s | samples | share | chain (innermost → outermost) |
|---|---|---|---|---|
| 1 | 209.72 | 817 | 10.4 % | `zipfile:_read1 <- read <- peek <- readline` |
| 2 | 103.04 | 360 | 5.1 % | `string_arrow.py:isin <- algorithms.py:isin <- series.py:isin <- acs_pums.py:_read_archive` |
| 3 | 98.17 | 308 | 4.9 % | `zipfile:_read1 <- read1 <- acs_person_coverage_columns.py:lines <- _literal_csv_records` |
| 4 | 87.86 | 338 | 4.4 % | `asec_coverage_authentication.py:_capture <- _reconstruct <- authenticate_asec_coverage <- asec_population_catalogue.py:issue_asec_source_catalogue` |
| 5 | 86.66 | 332 | 4.3 % | same three frames `<- asec_2024_native_population.py:load_authenticated_asec_2024_native_population` |
| 6 | 73.71 | 285 | 3.7 % | `asec_current_money_source.py:_series_digest <- _frame_signature <- asec_2024_native_population.py:_frame_identity <- survey_population_preparation.py:_nested_seals` |
| 7 | 71.78 | 276 | 3.6 % | `zipfile:read <- _read2 <- _read1 <- read` |
| 8 | 60.80 | 235 | 3.0 % | `survey_population_preparation.py:update_cell <- _frame_identity <- survey_atomic_geography.py:_population_stamp <- <genexpr>` |
| 9 | 38.93 | 151 | 1.9 % | `acs_housing_universe_source.py:_persisted_sha <- acs_native_coverage_binding.py:_verify_sources <- _verify <- verify_acs_native_coverage` |
| 10 | 33.38 | 130 | 1.7 % | `c_parser_wrapper.py:read <- readers.py:read <- _read <- read_csv` |

Grouped: ACS archive streaming (rows 1 + 3 + 7) **379.67 s**; ASEC `_capture`
(rows 4 + 5) **174.52 s**; `_series_digest` under `_nested_seals` (row 6)
**73.71 s**; every `_population_stamp` chain **159.31 s**. Those are the next
lane's candidates, not this one's results.

### Caveats that bind every number above

- **Sampler, not counters.** All three JSONs have 16 rows in `checks`, every one
  `calls=0`, `cpu_seconds=0.0`, status
  `"not-wrapped: runtime refuses instrumented producers (PRODUCER_CHANGED); see sampled_cpu_ledger"`.
  Attribution is therefore statistical, from a 0.25 s stack sampler, truncated to
  four frames per chain and the top 60 chains per ledger (cuts: C 5.00 s, B
  2.83 s, A 2.11 s). A chain absent from a list only means it fell below that
  cut.
- **A second before-probe file exists**, and it is not the one quoted above: `.measure/before/probe/repeated-verification-measurement.json`,
  written 2026-09-16 01:34:11, `"measurement": "before f7bb88525"`, source tree
  `~/PolicyEngine/_worktrees/microcosm-verify-once-baseline`, 1,801.100 CPU s /
  1,821.775 wall s / 8,304,001,024 B, rank-1 chain the same ACS record fence at
  777.447 s, and `loadavg_at_start [54.72, 66.75, 68.97]`. So the before probe
  was run twice, on two copies of the same baseline code, by two scripts
  (`.measure/probe_repeated_verification.py` for file A;
  `.measure/probe_verify_once.py`, the script both after runs used, for this
  one). They agree on the finding — neither completed, the record fence
  dominates — and differ by 0.15 % in CPU, which is what a **ceiling** measures:
  the cap, not the work. File A is reported because it is the one the lane
  named; this one is disclosed because a reader comparing files should not have
  to discover it.
- **The committed probe is one line ahead of the measuring one.**
  `experiments/native-verify-once/probe_verify_once.py` was a byte-identical
  copy of `.measure/probe_verify_once.py` when every figure here was taken, and
  it now carries the verification-epoch read described above as well, which is
  why none of these JSONs has a `verification_epoch` field. Nothing else about
  it moved: same prefix, same sample, same seed, same sampler, same ceilings,
  same rows. The uncommitted `.measure/` scripts are left exactly as they ran.
- **Scope.** These are descriptive measurements. They say nothing about dataset
  quality, calibration or release eligibility, and nothing here is a build or a
  certification.

## The pull request

| | |
|---|---|
| PR | [PolicyEngine/microcosm#935](https://github.com/PolicyEngine/microcosm/pull/935) — **draft**, and it stays draft |
| Title | Verify native sources once per run, not once per access and per node |
| Base | `microcosm-us-launch-integration-20260909` (PR #893's branch) |
| Head sha at which every measurement and the CI-shaped battery were taken | `284bc6e9624d211ba9c16dcdf04df63b9e87b10c`, before the 2026-09-16 rebase rewrote it; the same tree, as a commit, is `41703d5d6`'s parent |
| Head sha now | the commits that resolve the 2026-09-16 verification findings, on top of the report commit `41703d5d6`, itself on the rebased base tip. The exact head is what `git rev-parse native-verify-once` returns; this table does not quote a sha it cannot have written. |
| Diff | **27 files** against the base branch at `db7a3a0ba`, in three parts: `packages/*/src` **12 files, +1,683 / −818**, which no edit to this report can move; `packages/*/tests` **5 files, +1,799**; docs, journal, changelog fragment, `.gitignore` and `experiments/` **10 files, +2,299 / −1**. The whole-branch total is +5,781 / −819 there. The third part and the total rise with every further commit to this file, which is why the parts are given as well and why each is dated to a sha. The 2026-09-17 06:40 figures — 26 files, tests **4 files, +1,780**, docs **10 files, +2,184 / −1**, total +5,647 / −819 at `baec266d2` — are what they were before the second fix pass added 19 lines to `test_us_graph_atomic_survey_financial.py`, an existing file this branch had not touched, which is the whole of the file-count change. Two-dot and three-dot agree, because the branch was rebased onto the base tip `363a9033b` on 2026-09-16 and no base drift is left to inflate either. The earlier figure in this table, 27 files / +4,399 / −849, was a two-dot diff taken while the branch was one commit behind, so its −30 in `docs/us-uk-release-path.md` was the base's own commit and not a lane change; it is superseded rather than corrected in place, because the tree has moved too. |
| CI | does not run on this PR by design: `.github/workflows/test.yml` triggers on `pull_request: branches: [main]`, so only #893 reaches CI. The battery above is the only gate this branch has. |

## Open questions for Max

**1. The before probe never finished. Buy a true ratio, or accept the bound?**
Both before attempts hit the 1,800 CPU-s cap, so the 9-node prefix has only a
`≥ 1.25×` lower bound; the like-for-like number is the 19-node pair (2.63× CPU).
(a) Accept the bound and quote 2.63× as the headline. (b) Re-run the before
probe with `ulimit -t 3600` (~35–60 min of one core) for a true 9-node ratio —
the lane brief allowed two probe runs at `ulimit -t 1800`, so raising the cap
needs your word.
(c) Run the **19-node** harness on the baseline worktree instead — same machine,
same staged inputs, same script — which costs about 1.5 h and would replace the
v4 receipt comparison with a same-conditions one, removing the snapshot and
`person-income-attachment.h5` differences from the story.

**2. Peak RSS rose 1.43× (9.31 → 13.31 GB at 1/1000).** The design's account is
that memoised authentication keeps validated state resident; note that no
receipt attributes the rise to particular objects, so that account is the
design's, not a measurement. It is well inside this run's 32 GiB ceiling.
(a) Accept it. (b) Narrow the epoch — one per stage, or one per node group,
instead of one spanning the runner call — trading peak residency for more full
validations, then re-measure. (c) Measure at 1/100 first, so how it scales is
known rather than assumed. (d) Attribute the rise before deciding anything,
since today nothing says which objects hold the extra 4 GB.

**3. Mechanism 3's residual: a run refused at run end can leave store objects
behind.** A byte rewrite that preserves all five stat fields is caught before any
caller receives a manifest, but after intervening nodes have written
self-consistent store records, so an operator must clear that run's store.
(a) Document it and leave it (today's behaviour). (b) Have the executor delete
the run's own written objects when the run-end re-derivation refuses. (c) Write
a refusal marker into the store so a later run cannot silently reuse those
objects.

**4. The main-only split — answered 2026-09-16, option (a); nothing is left
open.** The graph-shard hunks (`executor.py`, `manifest.py`, `codecs.py` and the
three new graph test files) depend on nothing in this stack, and they are
[#938](https://github.com/PolicyEngine/microcosm/pull/938): draft, base `main`,
head `33f3150bbe580d2742f8408eddc1ee93c1e247a1`, 7 files, +1,430 / −8,
`MERGEABLE` (`gh pr view 938`, 2026-09-17 07:43 UTC). It was opened on
2026-09-16 at head `8ea48447c` with 6 files, +988 / −8, and stood at
`1884d7f2a` with 6 files, +1,124 / −8 before the third fix pass; the earlier
figures are given because they are what this report carried at those heads.

**The mirror landed 2026-09-17.** The four graph-shard changes this branch made
after the verification findings are in #938, as four commits dated 06:45–06:46
UTC that day: `RunManifest.verification_epoch` with the `_verification_epoch`
parameter on `run_graph` (`6206b4a75`), the member-symlink follow in
`_source_stat_signature` (`d87196ce9`), the native-order float cast in
`_object_stream` — the endianness item #938's own author flagged —
(`284252e56`), and the docstring correction naming what the stat-preserving
rewrite test actually exercises (`1884d7f2a`). Diffed against this branch at
07:00 UTC on 2026-09-17: `manifest.py`, `codecs.py` and
`test_graph_executor_source_identity.py` are identical, and `executor.py` and
`test_graph_executor_series_stream.py` differ only by the documented `import
struct` and exact-`float` short-circuit hunks, which belong to #893's base and
not to this lane. So the split carries the whole graph shard as this branch has
it, and the open item this section previously handed to Max is closed. The
third fix pass added one more graph-shard file — the new
`test_graph_verification_epoch.py` — and it was mirrored the same way, as
`33f3150bb` on 2026-09-17, with `ruff check .` clean and
`packages/microcosm-graph/tests` green in that worktree before the push.

**5. Thirty-five build tests reach the graph shard but were never run here.**
They import it as `from microcosm.graph import …` rather than by dotted module
path, so neither the 62-file battery nor the post-findings selection picked them
up — `test_us_graph.py`, `test_us_survey_calibration.py`, `test_uk_graph.py`,
`test_us_full_puf_enrichment.py` among them. Counted on 2026-09-16 by grepping
`packages/microcosm-build/tests` for that import form and subtracting the files
already selected; `--collect-only` over the 35 reports **992 tests collected in
20.03s**, and collection is the only thing that was run against them. (An
earlier draft of this question said twenty-seven; that was the count against the
62-file battery's selection, not this one.) They were not run on this stacked
branch in the third fix pass either, and nothing here claims otherwise.

For the **graph-shard changes specifically** they are not uncovered: every
graph-shard hunk this lane makes is also on [#938](https://github.com/PolicyEngine/microcosm/pull/938),
whose base is `main`, and `.github/workflows/test.yml` triggers on
`pull_request: branches: [main]`, so #938 runs the full matrix — `lint`,
`fast`, `engine-shared`, `engine-us`, `engine-uk` and `wheels` — across
parallel shards that include those 35 files. CI there is real but not yet
complete: at 07:19 UTC on 2026-09-17, on the previous head `1884d7f2a`,
`gh pr checks 938` reported 10 of 23 jobs `pass` and 13 `pending`; pushing
`33f3150bb` restarted the matrix, and at 07:46 UTC it read 2 `pass` (`changes`,
`lint`) and 21 `pending`. No green verdict is claimed here — what is claimed is
that those files run there and do not run on this branch. What #938's CI does **not** cover is this
branch's build-shard work, because #935's base is #893's branch and CI does not
run there at all. (a) Run them now on this branch — serial hours on this
machine, since several are cold graph runs against staged sources. (b) Run only
the three named above. (c) Rely on #938's CI for the graph shard and on the
stack reaching `main` for the rest.

**6. `ruff format --check .` is red on 77 pre-existing files** in
`spec_engine/`, `uk_runtime/`, `tools/` and `experiments/`, and CI never runs
it. (a) Leave it. (b) One formatting-only PR against `main`. (c) (b), then add
`ruff format --check` to the lint lane so it cannot drift again.

**7. Where does the next lane aim?** What is left on the 19-node run is ACS
archive streaming 379.67 s, ASEC `_capture` 174.52 s, `_population_stamp`
159.31 s and `_series_digest` under `_nested_seals` 73.71 s. (a) Whole-roster
receipt transport, so the ACS archive is streamed once rather than per
authentication. (b) Per-node population retention, aimed at `_population_stamp`.
(c) Both, in one lane. (d) Stop here: 2.63× is enough for launch and the
remainder is not blocking.
