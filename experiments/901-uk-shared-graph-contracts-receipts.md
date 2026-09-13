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

| File | SHA-256 |
| --- | --- |
| `decl.py` | `e78c7359b48813c57eff6f697359682753b74f86f565f4e6974421e5a4048933` |
| `kernel.py` | `51f45e899ba578a6b2324636a9879253266b067812f68b648eacfcd5a184d245` |
| `weight_update.py` | `0ccfe6fcd257ef62b1f771b12eecc8ac5d447f5aa7d0403102e0ae290de16720` |
| `population.py` | `33d1bb7bacea22870940288bf1907fb9eb24df7c245a216ff802e7fb41f5208f` |
| `serialize.py` | `e5bf83c1082154f148626b6a36676614c6ff6c3fe0721aed94a1501da3021b1f` |
| `executor.py` | `ec4473bb033c4b1a36180c1518a42c755a46a2265d10461df1dfa00065862364` |
| `explain.py` | `734a7b0e31692c31a99528cd83d9e74d3e508a317d913c1c169724a42d69d0de` |
| `graph/__init__.py` | `697c59a37989a36124e6d43c7b07dd3b0582d965f97303c1fb02c88b41db2d48` |
| `tests/test_graph_weight_update.py` | `771ae0becbc57a4dd6198b9df229ec8c8262a5597647f335b7d5ed9be83471ff` |
| `tests/test_graph_frame_context.py` | `d58250d2194c81002be7282cd53597a7b9cff0906bb879b4b3926b4ac5b91ada` |
| `docs/graph-interface.lock` | `b42811ff0411dc179aaf9ddcf827aa270db866e08112cef8d1839df99c5a1f02` |

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
