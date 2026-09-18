# Fix round result — UK-enabling shared graph contracts (amendments 25, 26)

> Historical source-only report. Later source corrections and the passing
> 101-test run are recorded in the
> [13 September acceptance record](experiments/uk-shared-graph-contract-acceptance-20260913.md).

Source-only. **Runtime is UNTESTED.** No pytest, no import of the
production package, no engine, no country model, no native source, no
install, no gated data, no network beyond `git fetch`, no publication. The
checks that were run are stdlib `ast` parses, `ruff check` / `ruff format
--check`, `tools/ci_test_groups.py --verify`, and reading source. Every
behavioural claim below is a claim about what the source says, not about
an observed run; the bounded runtime plan root needs is at the end.

## Identity

| Thing | Value |
| --- | --- |
| Worktree | `_worktrees/microcosm-uk-shared-graph-contracts-20260913` |
| Branch | `uk-shared-graph-contracts-20260913` |
| Base | `origin/main` `15ebde806cd1a262363f7217fe535c7234ff757f` (re-fetched on resume; unchanged, 0 behind) |
| HEAD | `03b5cb8e2` |
| Reviewed head the adjudication ran on | `6f4ba4ec989eba93786b5d88033631ad253fdc5c` |

### Commits, in order

| Commit | Kind | What |
| --- | --- | --- |
| `97428cd9f` | journal | Lane baseline: absence proof, the #901 consumer read |
| `ef2dc69c3` | source+tests+docs | Amendment 25: `WeightUpdate`, `weight_update_receipt` |
| `f33d47cda` | source+tests+docs | Amendment 26: the three `KernelContext` frame fields |
| `895aabf19` | acceptance suite | B2's field set, isolated (as amendment 19's was) |
| `6f4ba4ec9` | journal+receipts | Lane receipts and the first runtime plan |
| `6104459f6` | journal | File `FABLE-REVIEW.md`; open the fix round |
| `c0e275543` | source+tests+docs | **F1** — a node sees the mass log its own key binds |
| `6ec46d62c` | source+tests+docs | **F3/F4** — detach the frame view; digest all three fields |
| `09827aec8` | journal | Record the resume and what it re-verified |
| `f5aa65d40` | source+tests+docs | **F2/F4/F5/F6** — design ancestry stated and proven; cached-axis replay properties; the `calibrate.adam` claim withdrawn; `decl.py` re-locked |
| `c1a7920f7` | tests | The clone fixture copies its household's members, as a copied group requires |
| `ecc98107e` | receipts | Fix-round contract deltas; refreshed identity; full test-file disclosure |
| `03b5cb8e2` | journal | Fix-round outcome, the F2 source reading, the runtime plan |

The two commits the previous process had already landed (`c0e275543`,
`6ec46d62c`) were preserved and re-verified rather than trusted; what that
verification consisted of is in `PROGRESS-uk-shared-graph-contracts.md`
under "Verified on resume, not assumed from the commit messages".

### File hashes at `03b5cb8e2`

| File | SHA-256 |
| --- | --- |
| `graph/decl.py` | `11c2abd77f50389c6cd0b51e26cb3ebd5cbd0770ba96e9de1b53c71e9725eaa4` |
| `graph/kernel.py` | `2df6cc5b5b396adae578f885aedbd038d2c0e51f56b4a31b64c2f6b69e1b918d` |
| `graph/executor.py` | `5ab918e495fe4f8dd32c16155fe8c7a911e60e171cdbc8edb790626ce2d58c19` |
| `graph/population.py` | `33d1bb7bacea22870940288bf1907fb9eb24df7c245a216ff802e7fb41f5208f` |
| `graph/weight_update.py` | `0ccfe6fcd257ef62b1f771b12eecc8ac5d447f5aa7d0403102e0ae290de16720` |
| `graph/serialize.py` | `e5bf83c1082154f148626b6a36676614c6ff6c3fe0721aed94a1501da3021b1f` |
| `graph/explain.py` | `734a7b0e31692c31a99528cd83d9e74d3e508a317d913c1c169724a42d69d0de` |
| `graph/__init__.py` | `697c59a37989a36124e6d43c7b07dd3b0582d965f97303c1fb02c88b41db2d48` |
| `docs/graph-interface.lock` | `b857403be2206156b844958cbd6abcb25ef951d05c0cc11e22554169a9343d2e` |

The lock file records `decl.py` and `kernel.py` at exactly the hashes
above; `test_graph_interface_lock.py` is what enforces that. The lock moved
twice in this lane and both times for a numbered amendment:
`kernel.py` in amendment 26's commits (`f33d47cda`, then the F1 and F3/F4
docstring edits), `decl.py` in amendment 25's (`ef2dc69c3`, then
`f5aa65d40`'s semantics correction). It was never refreshed to make a test
green.

### Test files touched, with their roles

| File | Acceptance suite? | Change |
| --- | --- | --- |
| `test_graph_weight_update.py` | no | new (amendment 25); fix round adds three cached-axis replay properties and renames the weak one |
| `test_graph_frame_context.py` | no | new (amendment 26); fix round adds the boundary-log and isolation properties |
| `test_graph_population.py` | no | fix round adds three design-anchor properties beside the existing ones |
| `test_graph_kernel_contract.py` | no | one assertion of field *adjacency* relaxed to the ordering amendment 19 actually claims |
| `test_acceptance_b_ownership.py` | **yes** | B2's `KernelContext` field set, in its own commit `895aabf19` |

The lane's earlier "isolated to one test file" claim was true of the
acceptance suite only. That is now stated accurately in the receipts and
historicized in the journal.

---

## Point-by-point response to the adjudication

### F1 — `frame_mass_log` leaked a same-version sibling's record. **Accepted; fixed.** (`c0e275543`)

The finding was correct. `_project_context` passed
`population.frame.mass_log`, which for an ordinary node is the version's
cumulative log (`executor.py:2669` rewrites the version entry after every
ordinary member, and `_append_frame_mass_log` runs for ordinary nodes).
An ordinary node's key binds `frame_key(version_key)` and the owners of
the columns it declared (`keys.py:206-209`, `keys.py:181-201`) — never a
sibling — so a cache hit could replay output computed against a different
log.

The fix is the reviewer's own minimal option. `run_graph` records
`boundary_mass_logs[node.id] = updated.frame.mass_log` where a structural
version is admitted (`executor.py:2674`), beside
`populations[node.id] = updated`, which both cold execution and a restored
hit reach; an ordinary node is projected from
`boundary_mass_logs[compiled.versions[node_id]]` (`executor.py:2511`), a
structural node from `incumbent.frame.mass_log`, whose key binds the base
*and* every ordinary member through `members` (`keys.py:214-225`). The two
maps are written at the same point, so the boundary lookup cannot miss a
version whose incumbent lookup (`executor.py:2413`, which runs first)
succeeded.

`test_mass_log_is_the_incoming_log` was replaced, as the review said it
had to be. The properties now distinguish the two cases: an unread
same-version appender is invisible to an ordinary member and moves neither
its key nor its stored bytes; cold-with-sibling equals cold-without and
each replays into the other's store under `resume="require"`; the same
appender *is* visible to the structural boundary, whose key it moves. The
ancestor-tracking alternative was not taken: it would make a node's input a
function of graph topology the key does not bind either.

### F2 — a design-kind `WeightUpdate` should re-anchor. **Not accepted.** Source-backed response.

The premise that anchors become "stale and mixed" is not what the source
does, and the proposed fix would redefine a declared contract.

1. **Anchors are captured once, at CREATE.**
   `_create_population` calls `Population.from_frame(frame, node.id)`
   (`executor.py:1461`), whose `design_weights is None` default reads the
   frame's design weights (`population.py:333-338`). That default is
   reached exactly once per graph.
2. **Afterwards they are only carried.** Every later version gets
   `design_weights=_carry_design_weights(...)` passed **explicitly**
   (`population.py:1159`, `population.py:1180`), so the re-derive default
   never runs again.
3. **An update does not touch them.** `_apply_weight_update`
   (`population.py:2032-2040`) returns a frame with replaced weight values
   and nothing else, and it runs *before* the carry in `patch`.
4. **Clones are not "mixed".** The review's EXPAND case says entrants
   anchor from updated values "while retained rows keep original anchors".
   The code separates three cases, not two:
   `_carry_design_weights` maps an EXPAND's **copied** rows back to their
   source row's *original* anchor (`population.py:2145-2157`; the cached
   twin at `941-981` does the same), retained rows keep their own, and only
   a row with **no lineage at all** reads `frame.weights_for(entity)` —
   because it has no earlier weight to be anchored on. The assignment's
   framing is exactly the code's.
5. **The cap denominator is the original by declaration, not by accident.**
   `_assert_design_weight_cap` compares `current > design * cap` against
   the carried anchors (`population.py:2333-2350`) and its refusal has
   always read "above N * original design weight";
   `realized_max_weight_ratio` uses the same denominator
   (`population.py:2354-2375`), under a parameter the node must spell
   `weight_anchor='design'`.
6. **Re-anchoring would be the silent change.** It would let a
   normalization node inserted anywhere upstream widen every already-
   declared `max_weight_ratio` by its own factor — a non-local
   redefinition of a contract other nodes wrote against.

The review's T2 case is real behaviour, and it is now asserted as the
intended outcome rather than left implicit: with design weights doubled by
an update, calibrated weights equal to them are refused at
`max_weight_ratio=1.5` and admitted exactly at `2.0`, reporting a realized
ratio of `2.0`. A stage that wants a cap against normalized weights states
the ratio it means.

**No mixed-anchor defect beyond that definition was established**, so
nothing was stopped with a counterexample. One consequence is worth naming
and is now named in the amendment and the docstring: a row admitted
*after* an update is anchored on whatever design weight the `EXPAND`
installs for it. If a kernel derives an entrant's design weight from the
updated incumbent values, that derived number becomes the entrant's
anchor — because it is the weight the row entered carrying. That is the
anchor rule applied to a row with no ancestry, not a mixture of two rules.

The review's alternative ("refuse `kind='design'` in
`WeightUpdate.__post_init__`") was also declined: the lane's recorded read
of #901 shows `uk.full.normalize` declaring
`WeightUpdate("household", weight_kind, ...)` with `weight_kind` a
variable, so refusing the design arm would refuse a consumer this
amendment exists for, to avoid a defect that is not there.

What changed instead: `WeightUpdate`'s docstring no longer says "ancestry
is untouched" as a bare assertion but states the anchor rule and its three
cases; amendment 25 gains the same paragraph; and
`test_graph_population.py` gains three properties —
`test_a_same_kind_design_update_leaves_the_original_anchors_invariant`,
`test_a_design_update_moves_no_anchor_for_retained_clone_or_entrant_rows`
(the same EXPAND run over an updated and an un-updated population, with
the anchors asserted equal and the frames' design values asserted a factor
apart so it is not vacuous), and
`test_a_design_update_does_not_move_the_calibration_cap_denominator` (T2).

### F3 — the three fields shared live objects. **Accepted; fixed.** (`6ec46d62c`)

The review rated this non-blocking; root took it as blocking, because
"nothing writable through the public API" is not the property that matters
once a kernel holds the object: a frozen dataclass still yields to
`object.__setattr__`, and a shared `MassChangeRecord` is a live handle on
the version's log that, unlike a table, nothing would notice.

`_project_context` now hands out `deepcopy(frame.metadata)` and records
rebuilt through `_detached_record`, the rule `_observer_snapshot` already
followed and now shares. `_context_digest` binds all three fields: the
metadata through `store._encode_frame_metadata` (the frame format's own
codec, so an unchanged view digests as it persists), each mass record
field by field, and the projected column order. The codec call is guarded —
a value `Frame` would never have admitted is reported as the mutation it
is rather than raising while the comparison that would report it is being
computed.

Detachment and the digest are not redundant: the digest catches a kernel
whose output stops being a function of its declared inputs; detachment is
what stops a *retained* view from rewriting the live version after that
node's check has already passed.

### F4 — two tests were weaker than their names. **Accepted; fixed.** (`6ec46d62c`, `f5aa65d40`)

- The observer property now rewrites nested metadata and mass records as
  well as tables, over the graph whose snapshots actually carry a mass
  record, and asserts the rewrite landed on the snapshot before asserting
  the live version and the next node are unchanged.
- `test_cold_then_required_replay_revalidates_the_axis` is renamed
  `test_required_replay_reapplies_the_stored_update`, which is what it
  proves. Three new properties carry the claim it did not:
  `test_required_replay_refuses_a_cached_binding_against_another_axis`
  rewrites the stored record's binding to the same household ids in
  reverse order — the one case a count check cannot catch — and requires
  the replay to hit that record; it is refused with `different 'household'
  axis` and no kernel runs, so the refusal came from re-applying the
  cached result. A second does the same with a short axis. A third re-files
  the record *unchanged* and still replays, so the two refusals are about
  the binding rather than about a record having been re-filed. The tests
  also pin where the check fires: `resume="require"`'s preflight validates
  record shape, so a foreign axis surfaces at apply time, mid-run — which
  is the residual the review itself flagged.

### F5 — the `calibrate.adam` motivating claim. **Accepted; withdrawn.** (`f5aa65d40`)

Confirmed at source: `packages/microcosm-calibrate/src/` contains no
`weight_update` string at all, so `calibrate.adam@1` emits no
`receipt['weight_update']`, and `_apply_weight_update` would refuse the
declaration as unverifiable (`population.py:2025-2031`). Its own guard
also still asks for a `WeightTransition` by name. Amendment 25, the
`WeightUpdate` docstring and the test module docstring no longer offer
"a re-solve of an existing calibration" as a covered case; the amendment
now states plainly that re-solving through the shared kernel is a **future
consumer adaptation**.

### F6 — cosmetic and disclosure. **Accepted.** (`f5aa65d40`, `ecc98107e`)

The node mass-policy mismatch message said "weight transition's" on a path
both declarations reach; it now names neither. The receipts disclose every
test file touched and which one is the acceptance suite's, and the
journal's "isolated to one file" line is historicized in place rather than
edited away.

---

## Bounded runtime plan for root

Finite and invented-only: the toy country fixture under
`packages/microcosm-graph/tests/fixtures/toy_country/` and `tmp_path`. No
country model, engine, native source, gated microdata, credential or
network. Nothing below publishes, promotes or writes outside `tmp_path`
and the uv environment.

```
uv sync --all-packages --locked

# 1. The new and changed properties.
uv run pytest packages/microcosm-graph/tests/test_graph_weight_update.py \
              packages/microcosm-graph/tests/test_graph_frame_context.py \
              packages/microcosm-graph/tests/test_graph_population.py

# 2. Everything that reads the two frozen files, the executor, or replay.
uv run pytest packages/microcosm-graph/tests/test_graph_interface_lock.py \
              packages/microcosm-graph/tests/test_acceptance_b_ownership.py \
              packages/microcosm-graph/tests/test_graph_kernel_contract.py \
              packages/microcosm-graph/tests/test_graph_executor.py \
              packages/microcosm-graph/tests/test_graph_explain.py \
              packages/microcosm-graph/tests/test_graph_serialize.py \
              packages/microcosm-graph/tests/test_graph_decl.py \
              packages/microcosm-graph/tests/test_acceptance_replays.py \
              packages/microcosm-graph/tests/test_acceptance_d_weights.py

# 3. Both shards whole.
uv run pytest packages/microcosm-graph packages/microcosm-frame

# 4. Lint and the CI partition (both already pass here).
uv run ruff check .
uv run python tools/ci_test_groups.py --verify
```

### Helper dependencies each step needs

| Step | Needs | Why |
| --- | --- | --- |
| 1 | `packages/microcosm-graph/tests/_toy.py`, `fixtures/toy_country/*.csv`, `schema.json` | every graph-level property runs the toy country; `test_graph_frame_context.py` also asserts the **person column order** from `person.csv`'s header, so a fixture column reorder is a genuine (and intended) failure there |
| 1 | `pandas`, `numpy` from the locked env | the anchor properties call `patch()` directly and build pandas lineage objects; no store, no executor |
| 1 | a writable `tmp_path` | the frame-context and weight-update properties build real `ContentStore`s under it |
| 2 | `docs/graph-interface.lock` | `test_graph_interface_lock.py` reads it from the repo, not the wheel |
| 3 | nothing further | |
| 4 | `tools/ci_test_groups.py` | partition authority; `--verify` must stay `verification=ok` |

Expected: `test_graph_weight_update.py`'s 22 and
`test_graph_frame_context.py`'s 26 properties pass, `test_graph_population.py`
passes all 60 (57 of them pre-existing, 3 added this round), the
regression files stay green, `ruff check` and `--verify` stay clean, and no
node key moves anywhere (neither amendment adds a `Node` field or changes a canonical
projection, so no spec or seed digest should move; if one does, that is
main drift, not this lane).

### Where a source-only round could still be wrong, worth checking in this order

1. **`test_a_design_update_moves_no_anchor_for_retained_clone_or_entrant_rows`
   builds an EXPAND by hand.** `_patch_expand`'s guards are many; one
   already forced a correction in this round (`c1a7920f7` — a copied group
   requires the same number of copies of every incumbent member,
   `_remapped_expand_memberships`). A second guard firing would surface as
   a `PopulationError` naming that guard, not as a wrong anchor.
2. **The three cached-axis properties re-file a store record** through
   `ContentStore.put_json(..., verify_existing=False)`. That is the
   documented replace path (`store._put` →
   `_replace_write_only_collision`), and the record key is derived from the
   node key rather than from content, so a rewritten receipt is not a
   key/content mismatch. The third property (re-file unchanged, still
   replays) is the control that separates "the binding was refused" from
   "the re-filed record was unreadable".
3. **`_context_digest`'s metadata arm** calls `store._encode_frame_metadata`
   twice per node. Its guard (`TypeError`, `ValueError`, `RecursionError`)
   is reasoned from the codec's and `canonical_json`'s source, not
   observed. It also costs one encode per node per side; metadata is small,
   but that cost is real and unmeasured.
4. **`deepcopy(frame.metadata)` per node projection** is likewise
   unmeasured. `Frame` metadata is stage-level, not row-level, so this is
   expected to be negligible; it has not been timed.
5. **Pandas copy-on-write in the observer property.** The earlier plan
   flagged that a table-only rewrite could make that property vacuous; it
   now also rewrites nested metadata and mass records through
   `object.__setattr__`, which copy-on-write does not affect, so the
   property can no longer be vacuous in the way flagged — but the table arm
   of it can still be.
6. **The mass-record probe** in `test_graph_frame_context.py` states an
   unchanged household total from `context.weights["household"].values.sum()`;
   if `_append_frame_mass_log`'s bracketing disagrees, it surfaces as a
   `PopulationError` from the boundary-log properties.

## What this round does not claim

No UK build, calibration, native lane, release or publication was run,
prepared or authorized. No country model was imported. `#901` is
untouched: this lane adds no UK graph stage, kernel, target or gate, and
the two amendments remain separable in source — 25 touches `decl.py`,
`serialize.py`, `explain.py`, `population.py`, `graph/__init__.py` and the
new `weight_update.py`, leaving `kernel.py` byte-identical; 26 touches
`kernel.py` and `executor.py`, leaving `decl.py` byte-identical. Either can
be dropped without the other.
