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

- Contract read and design fixed. Environment pinned and baseline captured.
  Implementation not yet written.

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
  contents); `as_bytes_mapping()` is its byte-mode sibling.
- Built-in `raw-bytes-v1`: regular files only, 64 MiB bound, explicit
  directory / non-regular / oversize refusal as `ValueError`; unknown codec
  and `ImportError` keep the fatal `StoreUnavailable` classification
  (acceptance property E2).
- `decl.py` and `kernel.py` are untouched. No dataclass schema or frozen API
  baseline moves.

## Done

- [x] Read the existing contract: `codecs.py`, `executor._source_paths_and_keys`,
      `store.ContentStore.__init__` codec validation, `keys.source_content_key`,
      `kernel.KernelContext.sources`, `decl.Node`/`SourceRef`.
- [x] Confirmed a `StructuralDelta.NONE` node may declare `sources` (decl.py),
      so an import kernel needs no synthetic population of its own.
- [x] Environment: this worktree has no venv; verification uses
      `/Users/maxghenis/PolicyEngine/microcosm/.venv/bin/python` with
      `PYTHONPATH` set to this worktree's `packages/*/src`, verified by printing
      `microcosm.<shard>.__file__` for every shard.
- [x] Baseline `packages/microcosm-graph/tests` before any edit: 43 passed,
      1 skipped, 1 pre-existing failure
      (`test_acceptance_h_parity.py::test_h3_us_post_transfer_parity`) — recorded
      so it is never mistaken for lane damage.

## Next

1. `codecs.py`: byte mode (`SourceBytesCodec`, `register_bytes`, `load_bytes`,
   `as_bytes_mapping`, cross-mode name uniqueness, `raw-bytes-v1`,
   `load_source_bytes`).
2. `microcosm/graph/__init__.py`: export `SourceBytesCodec`, `load_source_bytes`.
3. Tests: codec unit tests in `test_graph_codecs.py`; end-to-end source ->
   typed-artifact route in a new flat graph test file.
4. `tools/ci_test_groups.py --verify`; changelog fragment.
5. Patch + receipt under `/Users/maxghenis/architecture-reviews/microcosm-20260904/`.
   No push, no PR, no remote comment from this lane.
