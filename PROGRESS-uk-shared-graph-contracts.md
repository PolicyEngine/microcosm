# UK-enabling shared graph contracts (amendments 25 and 26)

Lane journal. Append-only within this lane; historicize rather than
overwrite once the branch merges (CLAUDE.md, "Root journals are history").

## State

Bounded, source-only slice extracting the shared graph contract that
María's UK full-build graph (#901) consumes, as numbered amendments on
current `main`. No UK graph stage, calibration science or country
kernel is added here — those stay in #901.

- Worktree: `_worktrees/microcosm-uk-shared-graph-contracts-20260913`
- Branch: `uk-shared-graph-contracts-20260913`
- Base: `origin/main` `15ebde806cd1a262363f7217fe535c7234ff757f`
- Reviewed UK head: #901 `051fb972b19d319d58277bd63306d0d0e0947ce2`
  (draft, base `microcosm-us-launch-integration-20260909`, unchanged
  since 2026-09-10T21:03:10Z; re-verified via `gh` on 2026-09-13)
- Source review followed: `uk-parallel-review.md` (2026-09-12), section
  "Best independent implementation slice"

**Runtime is UNTESTED in this lane.** Instructions forbid pytest,
production imports, engine, native sources and installation here. Only
stdlib `ast`/`ruff`/CI-inventory source checks were run. A finite
invented-only runtime plan is at the end of this file for root review
*before* execution.

## Absence verified on base 15ebde806

| Contract | Present on main? | Evidence |
| --- | --- | --- |
| `WeightUpdate` (same-kind weight replacement) | **absent** | `grep -rni weightupdate .` over the worktree returns nothing; `decl.py:320` carries only `WeightTransition`, whose `__post_init__` requires `to_kind` strictly later in `WEIGHT_KINDS`, and `population.py:1944` rejects a non-forward move. A same-kind update is therefore unrepresentable. |
| `weight_update_receipt` / ordered-axis evidence | **absent** | no `weight_update` module under `packages/microcosm-graph/src/microcosm/graph/`. |
| `KernelContext.frame_metadata` | **absent** | `kernel.py:361-370` lists the complete field set; no metadata field. |
| `KernelContext.frame_mass_log` | **absent** (read side) | same field list. The *write* side already exists: `population._append_frame_mass_log` (`population.py:2117`) already ingests `receipt['frame_mass_log_append']`, so only the kernel's view of the incoming log is missing. |
| `KernelContext.frame_column_order` | **absent** | same field list. The executor projects `tables[entity]` in declaration order (`executor.py:600-628`), not the population version's own column order, so a consumer cannot reconstruct the frame layout. |

Frame-side prerequisites already on main: `Frame.mass_log`,
`Frame.metadata`, `MassChangeRecord` and `_freeze_metadata`
(`packages/microcosm-frame/src/microcosm/frame/bundle.py`).

Interface lock on base matches the files exactly:
`decl.py ed0a859adcae12510d5ba74d51c694617201f7b448b108a3f602410f5da44876`,
`kernel.py dbf57c137330f0f12744c557ee594586a1b308b6d6adaba1938e2b6efded21ca`.

## Actual consumer read before specifying shape

`packages/microcosm-build/src/microcosm/build/uk_runtime/graph_population.py`
at `051fb972` (SHA-256 `fc6f5b33127020f6e0529b39304715fe2028d47a6627b152b1e52e0d69f2efdc`):

- `context_frame` (L58-80) reads `context.frame_column_order.get(entity, ...)`,
  `context.weights`, `context.strata`, `getattr(context, "frame_mass_log", ())`
  and `getattr(context, "frame_metadata", {})`.
- `UKSampleNormalizationKernel` (L386-412 region) declares
  `WeightUpdate("household", weight_kind, "Normalize sampled source-family mass.")`
  with `mass="declared"` and places `weight_update_receipt(ids)` under
  `receipt["weight_update"]`.

## Done

- (nothing yet; baseline recorded)

## Next

- Amendment 25: `WeightUpdate` + ordered-axis receipt.
- Amendment 26: `KernelContext` frame metadata / mass log / column order.

---

## Done (2026-09-13)

| Commit | What |
| --- | --- |
| `97428cd9f` | Lane baseline: absence proof + the exact #901 consumer read |
| `ef2dc69c3` | Amendment 25: `WeightUpdate` + `weight_update_receipt` |
| `f33d47cda` | Amendment 26: `KernelContext` frame metadata / mass log / column order |
| `895aabf19` | Acceptance suite B2 field set, isolated (as amendment 19's was, `a2b6dfb0b`) |

The two amendments are separable: 25 touches `decl.py` (re-locked) and
leaves `kernel.py` byte-identical; 26 touches `kernel.py` (re-locked) and
leaves `decl.py` byte-identical. Either can be dropped without the other.

## Contract decisions

1. **`reason` is normative.** It enters the node key, so two updates that
   state different purposes are different nodes. Follows #901's own
   declaration; the consequence is stated in the amendment.
2. **`to_kind` stays a property, not a field.** That is what keeps the two
   declarations' field sets disjoint (`{entity, to_kind, mass}` vs
   `{entity, kind, reason, mass}`), so neither canonical bytes nor
   declaration JSON can confuse them — and existing `to_kind` readers (the
   design-weight cap, the calibration view) keep working unchanged.
3. **No `free` mass on an update.** #901 declares this too. An update that
   neither moves kind nor bounds mass records nothing checkable.
4. **Replay is structural, not a parallel rule.** `_load_cached_result`
   already reconstructs the `KernelResult` and re-applies REWEIGHT to the
   current base, so a cache hit re-enters `_apply_weight_update`. No
   executor change was needed for the axis check.
5. **The frozen interface does not import another shard's private name.**
   #901's `kernel.py` imports `microcosm.frame.bundle._freeze_metadata`.
   This lane does not: `Frame` has already deeply frozen the metadata the
   executor passes, and `KernelContext` adds a read-only view over it. The
   docstring says exactly that and claims no deep freeze of its own.
6. **The three fields ride between `artifacts` and `tolerances`**, not at
   the end as in #901, so amendment 17's "numerics rides at the end of the
   context" stays literally true. Only amendment 19's unit assertion of
   *adjacency* relaxes, to the ordering it actually claimed.
7. **A column order may not name an unprojected column.** Set equality
   with the projected columns, not a subset: a column *name* is itself
   information about the version.

## Remaining risks

- **Runtime is UNTESTED here.** No pytest, import, engine or install was
  run, per the lane's instructions. Everything below the source level is
  unverified; see the runtime plan.
- The acceptance-suite commit `895aabf19` is the one change this lane made
  to a file the charter assigns to the suite lane. It is isolated to one
  file and follows the precedent the amendment-19 doc text states
  explicitly ("the acceptance suite's B2 field set gains it in its own
  commit"). If root's owner disagrees, dropping that commit leaves B2 red
  and the rest intact.
  *(2026-09-13, fix round: "isolated to one file" is true of the*
  *acceptance suite only. `test_graph_kernel_contract.py` — not a suite*
  *file — was also edited, and the fix round adds three properties to*
  *`test_graph_population.py`. The full list is in the receipts.)*
- Amendment 25 changes no node key; amendment 26 changes none either.
  Neither re-pins a spec digest. If a spec/seed digest moves in CI, that
  is main drift, not this lane (see `[[spec-engine-attested-modules]]`).
- `_context_digest` (B4's mutation check) was **not** extended to the
  three new fields. They are immutable views over an immutable `Frame`, so
  there is nothing for a kernel to mutate; stated here so the omission is
  a decision rather than an oversight.
  *(2026-09-13, fix round: this reasoning was wrong and the decision is*
  *reversed. A frozen dataclass still yields to `object.__setattr__`, so*
  *the fields are now handed out detached and all three are digested —*
  *commit `6ec46d62c`. Superseded; kept for the record.)*
- No claim is made that any UK build, native lane, calibration or release
  passes. This lane read source only.

## Next (for root, before execution)

Finite, invented-only runtime plan — nothing below touches a country
model, engine, native source, gated microdata or the network.

```
uv sync --all-packages --locked
uv run pytest packages/microcosm-graph/tests/test_graph_weight_update.py \
              packages/microcosm-graph/tests/test_graph_frame_context.py
uv run pytest packages/microcosm-graph/tests/test_graph_interface_lock.py \
              packages/microcosm-graph/tests/test_acceptance_b_ownership.py \
              packages/microcosm-graph/tests/test_graph_kernel_contract.py \
              packages/microcosm-graph/tests/test_graph_serialize.py \
              packages/microcosm-graph/tests/test_graph_decl.py \
              packages/microcosm-graph/tests/test_acceptance_d_weights.py \
              packages/microcosm-graph/tests/test_graph_population.py \
              packages/microcosm-graph/tests/test_graph_executor.py \
              packages/microcosm-graph/tests/test_graph_explain.py \
              packages/microcosm-graph/tests/test_acceptance_replays.py
uv run pytest packages/microcosm-graph packages/microcosm-frame
uv run ruff check .
uv run python tools/ci_test_groups.py --verify
```

Expected: 19 + 19 new tests pass; the ten regression files stay green;
`ruff check` and `--verify` already pass here. The four places a source-only
lane could be wrong, in the order worth checking:

1. `test_graph_frame_context.py::test_a_retained_mutating_observer_changes_nothing`
   mutates a detached snapshot with `table.loc[:, column] = table[column].iloc[0]`.
   If pandas copy-on-write makes that a no-op on the snapshot, the test
   passes vacuously rather than falsely; tighten it rather than trust it.
2. The toy person column order is asserted from `fixtures/toy_country/person.csv`
   (`person_id, person_household_id, person_release_id, age, income, ...`).
   If that fixture changes, `test_column_order_is_the_versions_own_order_not_the_projection`
   is the test that notices.
3. `_append_frame_mass_log` requires the record to bracket the real
   household totals; the probe states an unchanged total from
   `context.weights["household"].values.sum()`. A mismatch would surface
   as `PopulationError` from `test_mass_log_is_the_incoming_log`.
4. `test_round_trip_refuses_a_mixed_weights_payload` edits canonical JSON
   by string surgery and depends on lexicographic key order
   (`entity, kind, mass, reason`).

---

# Fix round (2026-09-13, after the independent Fable adjudication)

## State

The lane's four commits (`97428cd9f`..`6f4ba4ec9`) stand. An independent
read-only adjudication of exactly `6f4ba4ec9` over base `15ebde806`
returned **REQUEST_CHANGES**; its verbatim text is now filed in this
packet as `FABLE-REVIEW.md` (the reviewer ran without a write tool and
returned the review through its tool result instead).

Root owns adjudication. Findings F1, F3, F4, F5 and F6 are accepted and
implemented in this round. **F2 is not accepted as stated** — see the
point-by-point response below and in `FIX-RESULT.md`.

`origin/main` re-fetched before editing: still
`15ebde806cd1a262363f7217fe535c7234ff757f`, nothing new to merge, branch
is 5 commits ahead and 0 behind.

**Runtime remains UNTESTED in this round too.** No pytest, no production
import, no engine, no install, no network. Source, stdlib `ast`, `ruff`
and the CI group inventory only.

## Next

- F1: project ordinary nodes from their version boundary's mass log.
- F3/F4: detach the exposed frame objects; digest all three fields.
- F2: source-backed rejection plus the anchor-invariance tests.
- F5/F6: correct the docs' motivating claim and the disclosure.

## Fix round, resumed (2026-09-13)

The first fix-round process died on an external API DNS error after two
commits. Nothing was reset; the checkout resumed clean at `6ec46d62c`.
`origin/main` re-fetched on resume: still `15ebde806`, branch 8 ahead / 0
behind. Restrictions unchanged — no pytest, import, engine, native source,
install, publication or disallowed log; source, stdlib `ast`, `ruff` and
the CI group inventory only.

### Verified on resume, not assumed from the commit messages

- `executor.py` boundary selection: `boundary_mass_logs[node.id]` is
  written at exactly one place — beside `populations[node.id] = updated`
  in the structural arm of the admission step (`executor.py:2669-2674`),
  which both a cold run and a restored hit reach — and read at exactly one
  place, the `StructuralDelta.NONE` arm that projects a context
  (`executor.py:2508`). Its key set is therefore a subset of
  `populations`', so the ordinary lookup cannot miss a version whose
  incumbent lookup succeeded.
- Key binding re-read at source: `keys.py:206-209` binds an ordinary
  node's `population_input` to `frame_key(version_key)` only, and
  `keys.py:214-225` binds a structural node's `base` *and* `members`. The
  projection matches that split exactly.
- `store.py:1157-1174`/`1334-1368` round-trip `Frame.mass_log`, so a
  restored boundary carries the same records a computed one does.
- Detachment: `Frame.__init__` calls `_freeze_metadata`
  (`bundle.py:112`), and `_freeze_metadata_value` rebuilds every nested
  mapping, tuple and frozenset (`bundle.py:1382-1405`), so the
  `_observer_snapshot` metadata hand-off shares no mutable-by-`setattr`
  object with its parent. `_project_context` deep-copies the metadata and
  rebuilds every record through `_detached_record`.
- `_context_digest` additions cannot raise while computing the comparison
  that reports a mutation: `canonical_json` raises only `TypeError` /
  `ValueError` (`canonical.py:_json_value`), `_encode_frame_metadata`
  only `TypeError` (`store.py:1060-1082`), and both are caught alongside
  `RecursionError`.
- `_project_context`'s only other caller,
  `packages/microcosm-build/tests/test_uk_uc_capital_coherence.py:223`,
  passes no `mass_log` and so takes the documented empty default.
- `ruff check .` clean; `ruff format --check` clean on every touched file;
  stdlib `ast` parses clean.

### Next

- F4b: the required-replay axis property is still the weak one the review
  named. Strengthen it.
- F2: source-backed rejection plus the anchor-invariance properties.
- F5/F6: the unsupported `calibrate.adam` motivating claim, the
  transition-only wording, and an accurate receipts disclosure.

## Done (fix round)

| Commit | What |
| --- | --- |
| `6104459f6` | File `FABLE-REVIEW.md` and open the fix round |
| `c0e275543` | F1: a node sees the mass log its own key binds |
| `6ec46d62c` | F3/F4: detach the frame view, and digest it for mutation |
| `09827aec8` | Record the resume and what it re-verified |
| `f5aa65d40` | F2/F4/F5/F6: design ancestry stated and proven; replay axis properties; the `calibrate.adam` claim withdrawn; `decl.py` re-locked |
| `c1a7920f7` | The clone fixture copies its household's members, as a copied group requires |
| `ecc98107e` | Receipts: fix-round deltas and refreshed identity |

`decl.py` is now `11c2abd77f50389c6cd0b51e26cb3ebd5cbd0770ba96e9de1b53c71e9725eaa4`
and `kernel.py` `2df6cc5b5b396adae578f885aedbd038d2c0e51f56b4a31b64c2f6b69e1b918d`;
`docs/graph-interface.lock` matches both. `kernel.py` moved only in
amendment 26's commits, `decl.py` only in amendment 25's.

## F2: why the re-anchoring was refused

Read at source, not argued from the declaration text:

1. `Population.design_weights` is set from the frame **once**, at CREATE
   (`population.py:333-338` via `_create_population`, `executor.py:1461`).
2. Every later version gets its anchors from `_carry_design_weights`
   (`population.py:2125-2181`), which aligns the incumbent anchors to the
   new version by stable entity id.
3. `patch` passes those carried anchors to `Population.from_frame`
   **explicitly** (`population.py:1159-1180`), so the `design_weights is
   None` default that would re-derive them from the frame
   (`population.py:333-338`) is never reached after CREATE.
4. `_apply_weight_update` (`population.py:2032-2040`) replaces the frame's
   weight values. It touches `design_weights` not at all, and it runs
   *before* step 2 in `patch`.
5. So a design-kind update moves no existing row's anchor. The EXPAND arm
   of `_carry_design_weights` maps copied rows back to their **source
   row's original anchor** (`population.py:2145-2157`, and the cached
   twin at `941-981`), not to the source's current value, so a clone after
   an update inherits the pre-update anchor too. Only a row with no
   lineage reads `frame.weights_for(entity)` — and it has no earlier
   weight to be anchored on.
6. The cap therefore stays what its own refusal has always called it:
   `current > design * cap` against the **original** design weights
   (`population.py:2333-2350`), with `realized_max_weight_ratio` reported
   on the same denominator (`population.py:2354-2375`).

The review's T2 case is real behaviour and is the intended outcome, now
asserted as such: calibrated weights equal to design weights an update
doubled are refused at `max_weight_ratio=1.5` and admitted exactly at
`2.0`, realized ratio `2.0`. Re-anchoring would make the same numbers a
ratio of `1.0` and would widen every cap declared upstream of an unrelated
normalization by that normalization's factor — a non-local change to an
already-declared contract, which is what the assignment forbade.

No mixed-anchor defect beyond that definition was found, so nothing was
stopped. The one consequence worth naming, and now named in the
amendment: a row admitted *after* an update is anchored on whatever design
weight the EXPAND installs for it. If a kernel derives an entrant's design
weight from the updated incumbent values, that derived number is the
entrant's anchor — because it is the weight the row entered carrying.

## Next (for root, before execution) — fix round

Superseding the earlier plan's first block; the rest of that plan stands.

```
uv sync --all-packages --locked
uv run pytest packages/microcosm-graph/tests/test_graph_weight_update.py \
              packages/microcosm-graph/tests/test_graph_frame_context.py \
              packages/microcosm-graph/tests/test_graph_population.py
uv run pytest packages/microcosm-graph/tests/test_graph_interface_lock.py \
              packages/microcosm-graph/tests/test_acceptance_b_ownership.py \
              packages/microcosm-graph/tests/test_graph_kernel_contract.py \
              packages/microcosm-graph/tests/test_graph_executor.py \
              packages/microcosm-graph/tests/test_acceptance_replays.py \
              packages/microcosm-graph/tests/test_acceptance_d_weights.py
uv run pytest packages/microcosm-graph packages/microcosm-frame
uv run ruff check .
uv run python tools/ci_test_groups.py --verify
```

Where a source-only round could still be wrong, in the order worth
checking:

1. `test_graph_population.py::test_a_design_update_moves_no_anchor_for_retained_clone_or_entrant_rows`
   builds an EXPAND by hand. The clone-ordinal rule in
   `_remapped_expand_memberships` already forced one correction here
   (`c1a7920f7`); if another of `_patch_expand`'s guards fires, it will be
   a `PopulationError` naming the guard, not a wrong anchor.
2. The three cached-axis properties in `test_graph_weight_update.py`
   re-file a store record through `ContentStore.put_json(...,
   verify_existing=False)`. That is the documented replace path, but it is
   the one place these tests touch store internals; the third property
   (re-file unchanged, still replays) exists to separate "the binding was
   refused" from "the re-filed record was unreadable".
3. `_context_digest`'s metadata arm calls `store._encode_frame_metadata`
   twice per node. It is guarded against a kernel-planted value the codec
   cannot encode, but the guard's `except` list is reasoned from the
   codec's source, not observed.
4. The four risks the first plan listed are unchanged, except that the
   observer property named there now also rewrites nested metadata and
   mass records, so a pandas copy-on-write no-op could no longer make the
   whole property vacuous.
