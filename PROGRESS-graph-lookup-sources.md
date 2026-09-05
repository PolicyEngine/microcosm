# PROGRESS — raw-bytes source codecs for typed lookup imports

Branch `graph-lookup-sources-20260905`, worktree
`_worktrees/microcosm-graph-lookup-sources-20260905`, dependency base
`aad0defb` ("Reject invalid gate receipts and repair stale blocked cache
entries"). The base is intentional; it is never reset or rebased.

## Goal

Real US graph stages need authenticated NPZ/CSV *lookup* inputs.
`SourceRef` + the executor already authenticate source bytes (identity
pre-check, TOCTOU post-check), but `SourceCodecRegistry` decodes only
source-to-`Frame`. Registering a dummy `Frame` codec to satisfy the
executor's availability check would misrepresent a lookup table as a
population. This lane adds the smallest explicit raw-byte mode so a typed
import kernel can decode real source bytes honestly.

## State

- Implementation, tests, lint, CI-inventory and changelog complete on this
  branch (5 commits on top of `aad0defb`). Patch and receipt written to
  `/Users/maxghenis/architecture-reviews/microcosm-20260904/`. Awaiting root
  review; nothing pushed, no PR, no remote comment.

## Design (fixed before editing)

- `SourceBytesCodec = Callable[..., bytes]`, registered through a *separate*
  `SourceCodecRegistry.register_bytes` and read back through `load_bytes`.
- `load` keeps its exact contract and return type (`Frame`); it never widens
  into a `Frame | bytes` union. `load_bytes` returns `bytes` only.
- Names are unique across both modes; a cross-mode duplicate is a `ValueError`
  at registration, and a mode mismatch at load time is a `TypeError` naming
  the mode the codec actually has.
- `get` stays the availability resolver and now resolves either mode, because
  the executor's pre-run registration check calls it and a bytes codec must be
  *available* without becoming a Frame codec.
- `as_mapping()` keeps returning exactly the Frame loaders (unchanged type and
  contents); `as_bytes_mapping()` is its byte-mode sibling. A store configured
  with the Frame-only snapshot therefore fails closed on a bytes source.
- Built-in `raw-bytes-v1`: regular files only (checked on the descriptor it
  opened, opened non-blocking so a FIFO cannot hang a run), 64 MiB bounded
  read, explicit directory / non-regular / oversize refusal as `ValueError`;
  unknown codec and `ImportError` keep the fatal `StoreUnavailable`
  classification (acceptance property E2).
- `decl.py` and `kernel.py` are untouched. No dataclass schema, serialized
  shape, manifest field, or frozen API baseline moves.

## Done

- [x] Read the existing contract: `codecs.py`, `executor._source_paths_and_keys`,
      `store.ContentStore.__init__` codec validation, `keys.source_content_key`,
      `kernel.KernelContext.sources`, `decl.Node`/`SourceRef`.
- [x] Confirmed a `StructuralDelta.NONE` node may declare `sources` (decl.py),
      so an import kernel needs no synthetic population of its own.
- [x] Environment: this worktree has no synced venv; verification uses another
      worktree's interpreter with `PYTHONPATH` set to this worktree's
      `packages/*/src`, verified by printing `microcosm.<shard>.__file__` for
      every shard before trusting a run. No `uv sync`, no installs.
- [x] Baseline before any edit: `packages/microcosm-graph/tests` = 399 passed,
      1 skipped, 1 failed — `test_acceptance_h_parity.py::test_h3_us_post_transfer_parity`
      fails only because the borrowed venv lacks `jsonschema` (declared by
      microcosm-build). Recorded so it is never mistaken for lane damage.
- [x] `codecs.py`: byte mode (`SourceBytesCodec`, `register_bytes`, `load_bytes`,
      `as_bytes_mapping`, cross-mode name uniqueness, `raw-bytes-v1`,
      `load_source_bytes`), plus the docstring corrections a third,
      file-only codec forces in `codecs.py` and `keys.py`.
- [x] `microcosm/graph/__init__.py`: export `SourceBytesCodec`, `load_source_bytes`.
- [x] Tests: byte-mode unit tests in `test_graph_codecs.py`; the end-to-end
      source -> typed-artifact route in `test_graph_source_bytes.py`.
- [x] `tools/ci_test_groups.py --verify` green, new file in fast `rest` and
      engine `us-am`, never `[defaulted]`;
      `tools/graph_acceptance_burndown.py --verify` unchanged; `ruff check .` clean.
- [x] changelog fragment `changelog.d/graph-lookup-sources.added.md`.
- [x] Restored the root `PROGRESS.md` (another lane's ACS journal) after this
      journal first overwrote it; this lane's journal lives here instead.

## Next

- Root review. No push, no PR, no remote comment from this lane.
- Known residuals are listed in the receipt at
  `/Users/maxghenis/architecture-reviews/microcosm-20260904/graph-lookup-sources-receipt.md`.
