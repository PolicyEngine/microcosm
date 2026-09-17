# Verifying native sources once per run

A native US build spends most of its time re-proving things it already proved.
The 2026-09-15 pilot measurement on the nineteen-node financial graph at
`Fraction(1, 1000)` (3,168 expanded households, 5,362 s wall) attributes
**1,143 s to node execution and the remaining ~79 % to admission and
verification of the 3.48 GiB of staged source** — repeated. QRF fits inside
that run cost 0.06–0.07 s each. The graph should compile to roughly the
pre-graph runtime plus receipts; today it compiles to the pre-graph runtime
plus the source tree, once per accessor use and once per executed node.

This note records the five mechanisms that cause it, what each one guarantees
today, what changes, why the guarantee survives, and the new cost class.

## The rule the whole change obeys

> Memoisation changes **when** a check runs. It never changes **what** the
> check computes.

Every digest, key, seal and receipt value in this repository keeps its exact
value across this change. Nothing in `_validate`, `_validate_state` or
`source_content_key` produces a receipt field — all three only compare live
values against values frozen at issuance — so moving a comparison in time
cannot move a recorded value. The one place bytes are genuinely rebuilt
(`_update_series`, mechanism 5) is proven byte-for-byte against a verbatim copy
of the pre-change implementation.

Two guarantees are stated up front because every section below leans on them:

1. **Verified before first use.** Nothing is memoised until it has been
   verified in full, exactly as today.
2. **Proven unchanged at the end.** Every memo is re-verified in full,
   unconditionally, before the run hands anything back. A source or a live
   object that changed while a memo was warm still makes the run refuse.

## The verification epoch

Memoisation is **opt-in and scoped**. `microcosm.build.us_runtime.verification_epoch`
provides a context manager. Outside an epoch there is no memo at all and every
capsule behaves byte-for-byte as it does today — which is why every existing
capsule test, including the two that trace `_producer` and `_file_identity`
firing inside a validation, keeps passing unchanged.

Inside an epoch each capsule validation is split into two tiers:

* **Tier A — every access, never memoised.** The issuance binding proof, the
  live final authority and the attached owner payloads, the producer encoding
  (`_encode(_producer())` / `_encode(_implementation())`), and a *signature*:
  the stat identity `(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)` of
  every file the memoised tier reads, plus the structural witness of every live
  object the memoised tier digests (per table: the column axis, dtypes, row
  count, and for each column and index the buffer address, shape, strides and
  writeable flag). The signature is *input*, never a verdict: no Tier A check
  refuses because a stat identity or a witness moved.
* **Tier B — memoised on Tier A's signature.** Everything else: the source
  catalogues, ACS native coverage, the nested ASEC native validation, the
  whole-roster `_source_files` re-hash, the roster stat comparison itself, and
  the pure final seals other than the two Tier A repeats — the nested owner
  seals, the plan document digest and the three frame identities.

A signature mismatch is **not** a refusal. It is a memo miss: the complete
unmemoised validation runs, and whatever refusal it would have raised today it
raises now, with the same code, at the same access. This is why the roster's
stat identities are read into the signature rather than compared in Tier A: a
cheap comparison that refused first would answer an appended ACS archive with
`SOURCE_STAT_CHANGED` where an unmemoised borrow answers it with the ACS
catalogue's own refusal.
`test_an_in_epoch_refusal_carries_the_code_it_carries_today` pins that by
refusing the same mutation twice, once inside an epoch and once with no memo at
all: an appended `selection-request.json` raises `SOURCE_CHANGED`, an appended
`acs/csv_pus.zip` and an appended `asec/pppub25.csv` raise
`PREPARATION_VERIFICATION_REFUSED` (a foreign owner's own refusal, translated
by `_checked`), and a touched roster file raises `SOURCE_STAT_CHANGED`.

On leaving an epoch — and on leaving every nested epoch, so a run that ends
early through an inner scope is still covered — every capsule the epoch
validated is re-validated **in full, with the memo bypassed**, and the result
is recorded. A close records a new signature only when nothing moved while it
was validating: `_validate` compares the roster stats and then runs a whole
trailing `_pure_final`, and at an **inner** nested close the memo it records
survives into the outer epoch, so a signature taken after validating would
absorb a file that moved in that window and answer every outer borrow up to the
outermost close from the memo. Each close therefore takes the signature before
validating and again after, and when the two differ it records no memo answer
at all — the next borrow is a miss that pays the complete validation and
refuses with the code that validation raises. The entry itself is kept, because
the outermost close re-validates every capsule the memo still holds.
`test_a_roster_stat_moved_inside_an_inner_close_refuses_at_the_next_borrow` and
`test_a_native_source_moved_inside_an_inner_close_refuses_at_the_next_borrow`
pin both capsules by moving a source from a profile hook as that close's own
validation returns; against the previous ordering both fail with DID NOT RAISE.

`verification_epoch()` yields the close's record: the protocol label, the
capsule count, the number of memo hits, the number of signature misses and the
number of unconditional final re-validations. It is filled in as the epoch runs
and completed as the epoch closes; `epoch_record()` returns the innermost open
epoch's record, or `None` outside an epoch.

Both native runners bind the record they open and hand it to `run_graph`, which
attaches it to the manifest as `RunManifest.verification_epoch` — outside the
manifest key, outside its JSON, and outside every node receipt and cache
record, exactly like `source_identities`. The epoch closes before the runner
returns, so the record a caller reads there is the closed one:
`test_a_real_run_records_its_epoch_in_the_manifest` asserts that an actual
nine-node run over invented sources reports memo hits, at least one signature
miss, and one unconditional final re-validation for every capsule it memoised.
`test_nineteen_node_financial_cold_and_required_replay` asserts the other
runner, on both manifests it returns — its own epoch's record on the outer
manifest and the nested population epoch's on the prefix's — for the cold run
and the required replay alike: the epoch's protocol label, one final
re-validation for every capsule memoised, and nothing of it in `to_json`.

## The five mechanisms

### 1. `AuthenticatedSurveyPopulationPreparation._checked()`

`survey_population_preparation.py:1138-1162`. Every accessor — `.frame`,
`.context`, `.selection_plan`, `.receipt`, `.to_bytes()`, `.checked_view()`,
`validate()`, and the module-level `verify_survey_population_preparation` that
the graph registers as a per-node population observer — calls `_validate(state)`.

**Guarantee today.** At the moment of every borrow: the capsule is the one that
was issued and its payload is unchanged; both source catalogues, the ACS native
coverage binding and the nested ASEC native population re-verify their own
source bytes; the ten-file source roster re-hashes to `state.files`; the
producer modules re-encode to `state.producer`; the roster's stat identities
re-derive to `state.file_stats`; and the pure seals (live authority, attached
evidence, nested seals, plan document, prepared frame identity, both source
frame seals) match.

**Cost today.** One `_checked()` re-hashes the ACS archive pair fourteen times
(four in the ACS catalogue's source checks, eight across the native binding's
four snapshot dictionaries, two in `_source_files`) — about 11.1 GiB of
SHA-256 — plus three passes over the seven ASEC files, plus the whole roster.
It runs once per executed node through the population observer, plus once or
more inside every kernel that borrows the preparation.

**Change.** `_validate` becomes Tier A + Tier B as above. Tier A keeps the live
authority, the attached owner payloads and `_encode(_producer())` as refusals on
every access, and reads `_file_stats(state.root)` into the signature on every
access, adding the paths that `_file_stats` does not cover but the I/O arm
reads: the ACS catalogue's `source_dir` entries, its two private snapshot copies
and its projection path, the native binding's four snapshot dictionaries, and
the `source_files` rosters of both nested ASEC capsules. Tier B is the rest,
including the comparison of those roster stat identities against the ones frozen
at issuance.

**Why the guarantee survives.** Producer tampering and live-callable rebinding
still refuse at the same access (`_producer()` calls `_live()`, and it is never
memoised). Every on-disk change to any file the I/O arm reads — truncation,
append, in-place rewrite, replacement, rename, roster addition or removal,
symlink substitution — moves at least one of `st_size`, `st_mtime_ns` or
`st_ctime_ns`, or the directory's own stat identity, so it is a memo miss and
the existing refusal fires at the same access it fires today. Every structural
change to a live frame — a replaced column, a reindex, a dtype change, a
rebound buffer, a flipped writeable flag — moves the witness and is likewise a
memo miss. What is deferred to the epoch's unconditional final re-validation is
exactly two things: an in-place value mutation of a live buffer, and an on-disk
rewrite that leaves all five stat fields identical. On this machine the second
is not reachable without privilege: an unprivileged same-length in-place
rewrite followed by `os.utime` restores `st_size`, `st_ino` and `st_mtime_ns`
but not `st_ctime_ns`, and `setattrlist(ATTR_CMN_CHGTIME)` refuses with
`EPERM` (measured 2026-09-15, APFS). That is an observation about one
filesystem, not a guarantee, which is why the final re-validation is
unconditional rather than conditional.

**New cost class.** Per accessor use: O(modules) producer encoding + O(roster)
`lstat` + O(columns) witness — microseconds. Per run: one full validation at
first use plus one at each epoch exit, instead of one per accessor use, plus
two signature passes per memoised capsule at each exit, which bracket that
exit's own validation and cost the same `lstat` walk a borrow pays.

### 2. `AuthenticatedAsec2024NativePopulation.frame`

`asec_2024_native_population.py:449-499`. `.frame`, `.context`, `.receipt`,
`.to_bytes()` and `validate()` all route through `_state()` → `_checked()` →
`_validate_state(state)`: every frame access is a full re-authentication,
including a full SHA-256 of all seven ASEC source files.

**Guarantee today.** On every borrow: the implementation encoding matches, the
descendant and parent frame identities and the frame context match, the parent
capsule re-validates, the coverage/anchor/field owners re-verify, all seven
source files re-hash to their pinned digests with a before/after `fstat` guard,
and — because a file check or a foreign owner call may yield — the producer and
both frame identities are re-checked *after* the file loop as well as before.

**Change.** Same two tiers. Tier A keeps `_encode(_implementation())` and adds
the stat identity of the seven `state.source_files` paths and of their parent
directories (which `_file_identity` does not cover: it opens `O_NOFOLLOW` on
the final component only). `_State` here is a `NamedTuple` and a test pins its
immutability, so the memo lives in a side table keyed by `id(owner)`, holding a
weak reference to the owner beside the signature and the state. An entry
answers only when that reference still resolves to the same object, dead
entries are dropped as the epoch closes, and the whole table is cleared when
the outermost epoch exits.

The owner's own close runs after this capsule's. At the outermost close this
capsule's memo is already cleared, so the owner's final validation reaches the
complete file check; at an inner nested close the memo is still live with a
refreshed signature, so the owner's validation is a memo hit — unless a source
moved while this capsule's close was validating, in which case that close
records no signature and the owner's validation is a miss that re-runs the
complete check. Nothing is skipped in net either way, because this capsule's
exit has just re-validated it in full.

**Why the guarantee survives.** As in mechanism 1. The documented
yield-tolerance pattern is preserved intact: a memo hit runs neither the file
loop nor the checks that bracket it, and a memo miss runs the whole of
`_validate_state` unchanged, so the bracketing pair is never split.

**New cost class.** Per access: O(modules) + seven `lstat`s + O(columns)
witness. Per run: two full validations, each close's own bracketed by the two
signature passes described above.

### 3. Per-node source content keys in the executor

`executor.py:2452-2457` re-derives `source_content_key(name, source_paths[name])`
for every source a cold-executed node declares, and `keys._directory_identity`
(`keys.py:31-51`) `read_bytes()`es every file of a directory source and hashes
the whole tree. Cost is O(executed source-declaring nodes × source bytes): the
3.48 GiB tree is re-read per node.

**Guarantee today.** The check runs after the kernel returns and **before**
`_validate_result`, `_apply_result`, the population observer and `_write_node`.
Its real stake is the last one: no artifact computed from mutated source bytes
is ever persisted under a node key that claims the pre-mutation bytes, which
would poison the content-addressed store permanently.

**Change.** `run_graph` gains a per-run cache in its own locals — never inside
`source_content_key`, which stays pure because four existing assertions call it
twice on one path with the bytes changed in between and require different
answers. The cache is keyed by source name and a *stat signature*: for a
regular file, its `_stat_identity`; for a directory, the sorted recursive
roster of relative names with each member's `_stat_identity` and the stat
identity of every directory in the tree, plus the resolved identity of any
member that is a symlink — `_directory_identity` selects members with
`is_file()` and reads them with `read_bytes()`, both of which follow the link,
so the signature follows it too and the two sides walk the same members. The per-node check derives the
signature (an `lstat` walk, no byte reads); on a match it reuses the cached
key, on a mismatch it re-derives the content key in full and raises the same
`NodeRejected` if it moved. Immediately before the manifest is built, every
source is re-derived **in full, cache bypassed**, and a mismatch raises. The
run-end re-derivation is written inline rather than through
`_source_paths_and_keys`, because a build test counts calls to that exact code
object.

**Why the guarantee survives.** The refusal still precedes `_write_node` for
every mutation that moves any stat field — which is every ordinary mutation,
including one that adds, removes or renames a file inside a directory source,
since the roster is part of the signature, and including a change to the bytes
behind a member symlink, whose own five stat fields never move when its target
is rewritten. The narrowed case — a byte rewrite
that preserves all five stat fields — is caught at run end, before any caller
receives a manifest, but after intervening nodes have written store records.
That residual is stated rather than hidden: a run refused by the run-end check
may leave store entries behind, and those entries are self-consistent under
`_require_record_shape`, so the operator must clear the store for that run.
Against this, the executor gains a guarantee it does not have today: sources
are re-verified at run end even when the mutation happens during a node that
declares no source, which today is invisible.

**Record.** `RunManifest` gains `source_identities`, an attached mapping modelled
exactly on `populations` and `mass_ledgers` — `repr=False`, `compare=False`,
excluded from `content_addressed` and absent from `to_json`. It carries the
run-end re-derived key of every source. It moves no manifest key, no manifest
JSON byte, no node receipt and no cache record.

**New cost class.** Per node: an `lstat` walk of the source tree. Per run: two
full content derivations instead of one per executed source-declaring node.

### 4. The ACS record fence

`acs_person_coverage_authentication._records` fences raw logical CSV records
out of a zip member before any UTF-8 decoding or CSV allocation. It is a
pure-Python `for byte in block` loop over 4,096-byte blocks, and the member it
fences is the ACS person CSV inside `csv_pus.zip` — 2.4 GB expanded across
`psam_pusa.csv` and `psam_pusb.csv`. With `peek` and
`acs_person_coverage_columns.lines` it is 57–65 % of the admission phase in
every measured run.

**Guarantee today.** For every byte, in this order: the record length must be
below its cap (64 KiB for the first record, 400 KB afterwards) or
`CSV_RECORD_BYTES`; an unquoted comma resets the token counter, any other byte
advances it and must stay within 64 KiB or `CSV_TOKEN_BYTES`; a double quote
toggles quote parity, which **persists across records**; an unquoted LF ends a
record, an unquoted CR ends it and absorbs a following LF. Record boundaries
are located by quote parity only; CSV validity remains the unchanged strict
literal parser's job.

**Change.** The same state machine, driven by `bytes.find` and `bytes.count`
over 1 MiB blocks instead of a Python loop over individual bytes. Terminator
cursors are cached per block so an absent carriage return is searched for once
rather than once per record. Every completed record, and every in-flight record
at a block boundary, passes through a bound check: a record no longer than
64 KiB cannot have violated either cap, because the token counter resets at
each record start and so never exceeds the record's own length, and the record
cap is never below 64 KiB. Any record longer than that — which real ACS data
never produces — is replayed byte by byte through the original fence, so the
refusal code and the point at which it fires are exactly the original's.

**Why the guarantee survives.** The yielded record sequence and the refusal
raised are proven identical, not argued: `test_us_acs_record_fence_scan.py`
compares the shipped `_records` against a verbatim copy of the pre-change loop
over 4,200 randomised byte strings, every boundary case (bare CR, CRLF split
across a block boundary, quoted terminators, odd quote parity carried across
records, BOM, empty lines, trailing CR at EOF) and every bound case on both
sides of every cap. The same comparison runs against the real staged
`csv_pus.zip`.

**New cost class.** Linear in member bytes at C speed: measured 10.8 MB/s →
522.8 MB/s on the real archive, digests identical, a 48.5× reduction.

**Pin.** `acs_person_coverage_authentication.py` is one of two modules whose raw
file SHA-256 is pinned in `acs_native_coverage_binding._ACCEPTED`, so this
change re-pins it. The new value is derived by `shasum -a 256` on the changed
file and recorded in the lane report.

### 5. The kernel context digest

`_context_digest` (`executor.py`) digests every column of every table twice per
node — once before the kernel runs and once after, to prove the kernel did not
mutate its input context. `_update_series` took a fast path only for masked
extension arrays exposing `_data` and `_mask` ndarrays. On pandas 3.0.3 an
ordinary float, integer or boolean column is a `NumpyExtensionArray` with
neither, so it fell through to `to_numpy(dtype=object)` and a per-value Python
loop with two `hashlib` updates per value.

**Guarantee today.** The digest is a total function of the column: its dtype
string, the object projection's dtype and shape headers, and one framed payload
per value.

**Change.** `_object_stream` builds exactly those bytes with numpy for the three
kinds whose boxed value is always one exact Python type — `float` packed to
eight native IEEE-754 bytes, `bool` rendered as `b"b1"`/`b"b0"`, and `int`
rendered as a variable-width decimal. It returns `None` for everything else,
including a `Categorical`, whose values array is an integer-coded view and
would otherwise be mistaken for an integer column. Both `_context_digest` calls
remain: the second one is the mutation guard, and reusing its result would
delete the guard.

**Why the guarantee survives.**
`test_graph_executor_series_stream.py` compares the shipped helper against a
verbatim copy of the pre-change body **at the byte level**, not only at the
digest, across every column kind the projection can carry, the float specials
(signed zero, both infinities, the subnormal extremes), non-canonical NaN
payloads with their sign bit, both `int64` endpoints, empty, sliced and
strided columns, labelled indexes, and a 200-frame random sweep.

**New cost class.** Measured 0.47 s → 0.09 s for a 6,928 × 240 frame; 183 →
18 ns per value for floats, 226 → 14 for booleans, 383 → 119 for integers.
Still linear in rows, but with no Python-level work per value.

## What this note does not claim

Nothing here is a statement about dataset quality, calibration or release
eligibility. The PR-CI / certification boundary is unchanged: these are code
contracts, and green checks do not certify any data artifact.
