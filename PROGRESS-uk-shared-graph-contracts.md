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
- Amendment 25 changes no node key; amendment 26 changes none either.
  Neither re-pins a spec digest. If a spec/seed digest moves in CI, that
  is main drift, not this lane (see `[[spec-engine-attested-modules]]`).
- `_context_digest` (B4's mutation check) was **not** extended to the
  three new fields. They are immutable views over an immutable `Frame`, so
  there is nothing for a kernel to mutate; stated here so the omission is
  a decision rather than an oversight.
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
