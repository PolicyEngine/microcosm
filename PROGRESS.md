# Lane A progress

## State

Repository prepared on `node-graph/core`; the frozen contract and baseline are verified.

## Done

- Created an isolated clone at the exact `origin/node-graph` commit available to the source worktree.
- Confirmed the implementation branch starts at `517891f41d091e139b1e34f79c772e0f1265b8a3`.
- Read the acceptance charter, shard README, frozen declarations and kernel interface, interface tests, `Frame` primitives, and repository identity/serialization prior art in the prescribed order.
- Confirmed the locked SHA-256 digests for `decl.py` and `kernel.py` match `docs/graph-interface.lock`.
- Ran the untouched shard baseline: 16 interface tests pass.
- Reused the source worktree's synced environment read-only because this lane sandbox denies uv's global cache; all imports are forced to this clone.

## Next

- Implement canonicalization, key derivation, manifests, and the one-screen view with focused unit tests.
- Implement the store/codecs and population semantics.
- Integrate the executor and run the full charter-oriented unit suite.
