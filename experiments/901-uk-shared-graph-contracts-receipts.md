# Shared graph contracts extracted for the UK full-build graph (#901)

Source-only lane receipt, 2026-09-13. Runtime **UNTESTED** — see
`PROGRESS-uk-shared-graph-contracts.md` for the decisions, risks and the
finite invented-only runtime plan this lane leaves for root.

## Heads

| Thing | Exact identity |
| --- | --- |
| Worktree | `_worktrees/microcosm-uk-shared-graph-contracts-20260913` |
| Branch | `uk-shared-graph-contracts-20260913` |
| Base (reviewed main) | `15ebde806cd1a262363f7217fe535c7234ff757f` |
| #901 head (reviewed, re-verified) | `051fb972b19d319d58277bd63306d0d0e0947ce2` |
| #901 live state | OPEN, draft, `CONFLICTING`, updated 2026-09-10T21:03:10Z |
| Source review followed | `uk-parallel-review.md`, 2026-09-12 |

`origin/main` was re-fetched after the work and is still `15ebde806`.

## Absence, before anything was written

`grep -rni weightupdate` over the worktree at `15ebde806` returns nothing.
`decl.py:320` carries only `WeightTransition`, which requires `to_kind`
strictly later in `WEIGHT_KINDS`; `population.py:1944` rejects a
non-forward move. `kernel.py:361-370` is the complete `KernelContext`
field list and has no metadata, mass-log or column-order field. The
interface lock matched both files exactly. So both extensions were
genuinely absent, and no duplicate patch was produced.

The write half of the mass-log contract already existed on main
(`population._append_frame_mass_log`, `receipt['frame_mass_log_append']`);
only the kernel's view of the incoming log was missing. That is why this
lane adds a read side and no second ledger.

## Consumer that fixed the shape

`packages/microcosm-build/src/microcosm/build/uk_runtime/graph_population.py`
at `051fb972`, SHA-256
`fc6f5b33127020f6e0529b39304715fe2028d47a6627b152b1e52e0d69f2efdc`:

- `context_frame` (L58-80): `context.frame_column_order.get(...)`,
  `getattr(context, "frame_mass_log", ())`,
  `getattr(context, "frame_metadata", {})`.
- `uk.full.normalize`: `WeightUpdate("household", weight_kind, "Normalize
  sampled source-family mass.")`, `mass="declared"`, and
  `receipt["weight_update"] = weight_update_receipt(ids)` where `ids` is
  the household-id axis of the context frame.

Both signatures land exactly as #901 calls them, so no UK edit is needed
to consume this. Nothing was copied from #901's executor, and
`graph.attachments._PopulationRetention` is not imported anywhere.

## Amendments

**25 — a same-kind weight update is declarable.** `decl.py` gains
`WeightUpdate(entity, kind, reason, mass)` and
`WEIGHT_UPDATE_MASS_POLICIES`; a new non-frozen
`microcosm/graph/weight_update.py` gains `weight_update_receipt`. The
incumbent, declared and returned kinds must agree; mass is `conserve` or
`declared`; `reason` is required, non-empty and normative. The kernel
binds its ordered entity axis and the executor recomputes that binding
from the incumbent axis, on cold execution and on replay — replay for
free, because `_load_cached_result` already re-applies REWEIGHT to the
current base through the same function. `to_kind` is a property, so the
two declarations' field sets are disjoint and declaration JSON round-trips
each as itself; the transition payload is byte-for-byte unchanged.

**26 — the context carries the version's frame view.** `kernel.py` gains
`frame_metadata`, `frame_mass_log` and `frame_column_order`, riding after
`artifacts` and before `tolerances` so amendment 17's "numerics rides at
the end" stays literally true. A column order must be exactly an ordering
of the projected columns, so it can neither hide a column the node was
given nor name one it was not. The frozen interface does **not** import
`microcosm.frame.bundle._freeze_metadata` (as #901's does); the executor
passes `Frame.metadata`, already deeply frozen by `Frame`, and the
docstring claims only the read-only view it actually adds.

Neither amendment adds a `Node` field, changes a canonical projection, or
moves a node key. `decl.py` is re-locked by 25, `kernel.py` by 26.

## Identity

Files as of the fix round (`f5aa65d40`). The four earlier rows this table
carried for `decl.py`, `executor.py` and the two new test files were the
pre-fix bytes and are superseded here.

| File | SHA-256 |
| --- | --- |
| `decl.py` | `11c2abd77f50389c6cd0b51e26cb3ebd5cbd0770ba96e9de1b53c71e9725eaa4` |
| `kernel.py` | `2df6cc5b5b396adae578f885aedbd038d2c0e51f56b4a31b64c2f6b69e1b918d` |
| `weight_update.py` | `0ccfe6fcd257ef62b1f771b12eecc8ac5d447f5aa7d0403102e0ae290de16720` |
| `population.py` | `33d1bb7bacea22870940288bf1907fb9eb24df7c245a216ff802e7fb41f5208f` |
| `serialize.py` | `e5bf83c1082154f148626b6a36676614c6ff6c3fe0721aed94a1501da3021b1f` |
| `executor.py` | `5ab918e495fe4f8dd32c16155fe8c7a911e60e171cdbc8edb790626ce2d58c19` |
| `explain.py` | `734a7b0e31692c31a99528cd83d9e74d3e508a317d913c1c169724a42d69d0de` |
| `graph/__init__.py` | `697c59a37989a36124e6d43c7b07dd3b0582d965f97303c1fb02c88b41db2d48` |
| `tests/test_graph_weight_update.py` | `6ff1296b1d769454b9664ca95bf084b91cf8bd05ada99ed58d54c8d4768751c7` |
| `tests/test_graph_frame_context.py` | `bcee6caccb7ffce47f04785557d246d2e8d5211d760ce5a344f3023cb1d3bf0c` |
| `tests/test_graph_population.py` | `b1aeabac04dbe6af8aeaf3b1691c7306a1442a142279043868500fdfe6476eec` |
| `tests/test_graph_kernel_contract.py` | `b13105cd302702eadcb30278545e794c16f3d819f571693810fb8f838473b126` |
| `tests/test_acceptance_b_ownership.py` | `a78ad49a341d08fe2a76e98e94d6dfe2fb60140ce5589101603b93f933f5bb34` |
| `docs/graph-interface.lock` | `b857403be2206156b844958cbd6abcb25ef951d05c0cc11e22554169a9343d2e` |

### Every test file this lane touched

An earlier draft of this receipt said the lane's test change was "isolated
to one file". That is true only of the **acceptance suite**, which the
charter assigns to the suite lane. In full:

| File | Suite? | Change |
| --- | --- | --- |
| `test_graph_weight_update.py` | no | new (amendment 25) |
| `test_graph_frame_context.py` | no | new (amendment 26) |
| `test_graph_population.py` | no | three design-anchor properties added beside the existing ones (fix round, F2) |
| `test_graph_kernel_contract.py` | no | one assertion of *adjacency* relaxed to the ordering amendment 19 actually claims |
| `test_acceptance_b_ownership.py` | **yes** | B2's `KernelContext` field set, in its own commit (`895aabf19`), as amendment 19's was |

The lock was re-recorded as part of each numbered amendment, never
refreshed to make a test green: amendment 25 moved only the `decl.py`
line, amendment 26 only the `kernel.py` line.

## What stays with María

Every UK graph stage, kernel, calibration target, geography ladder,
scorecard and release gate in #901. This lane adds no country graph, no
UK node, and no second definition of anything #901 owns.

## Checks actually run

`ruff check .` (clean), `ruff format --check` on every file this lane
touched (clean), `python -I -B -S` stdlib `ast` parses of all 17 changed
files, and `tools/ci_test_groups.py --verify` (`verification=ok`; both new
test files land in `fast/rest` and the engine lane beside their 27 sibling
graph tests, neither `[defaulted]`). No pytest, no import of the
production package, no engine, no install, no network beyond `gh`
metadata and public blob reads.

---

## Fix round (2026-09-13), after the independent adjudication

An independent read-only review of `6f4ba4ec9` over `15ebde806` returned
REQUEST_CHANGES; its verbatim text is `FABLE-REVIEW.md` in this packet.
Root adjudicated. What changed in the contracts above:

**Amendment 26, mass log (F1).** The projection handed every node
`population.frame.mass_log`, which for an ordinary node is its version's
*cumulative* log. An ordinary node's key binds only its version's
structural boundary and the owners of the columns it declared, so a
sibling that ran earlier in the same version and appended a record could
change what the node was shown without moving its key. `run_graph` now
records each version's log as that version is admitted — one line, which
cold execution and a restored hit both reach — and projects ordinary nodes
from that boundary. A structural node still receives the cumulative log,
which its key binds through `base` and `members`.

**Amendment 26, isolation (F3/F4).** `Frame` deeply freezes its metadata
and its mass records, but a frozen dataclass still yields to
`object.__setattr__`, so passing those objects by reference made every
kernel a live handle on the population. The projection now hands out a
deep copy of the metadata and rebuilt records, through the rule
`_observer_snapshot` already followed (now the shared `_detached_record`),
and `_context_digest` binds all three fields — the metadata through the
frame format's own store codec, each mass record field by field, and the
projected column order.

**Amendment 25, design ancestry (F2 — not accepted as stated).** The
review asked a design-kind update to re-anchor `Population.design_weights`.
Root refused: an anchor is the design weight a row entered carrying, it is
captured once at CREATE and afterwards only carried by stable entity id,
and `_carry_design_weights` maps an EXPAND's copied rows back to their
source's *original* anchor rather than its current value. Re-anchoring
would silently redefine the denominator of every `max_weight_ratio`
declared upstream of an unrelated normalization. The amendment and the
`WeightUpdate` docstring now state the anchor rule instead of "ancestry is
untouched", and `test_graph_population.py` asserts it.

**Amendment 25, motivating claim (F5).** The amendment claimed a re-solve
of an existing calibration as a covered case. `calibrate.adam@1` emits no
`receipt['weight_update']`, so that declaration would be refused by the
axis check; the claim is now marked a future consumer adaptation.

### Checks actually run in the fix round

`ruff check .` (clean), `ruff format --check` on every touched file
(clean), stdlib `ast` parses of every touched Python file, and
`tools/ci_test_groups.py --verify` (`verification=ok`; the three touched
graph test files appear under `[fast]` and `[engine]`, none under
`[defaulted]`). Still no pytest, no import of the production package, no
engine, no install, no native source and no publication. `origin/main`
re-fetched on resume: `15ebde806`, nothing to merge.
