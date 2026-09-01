# Lane A progress

## State

Identity, provenance, the validated content store, and source codecs are implemented.

## Done

- Created an isolated clone at the exact `origin/node-graph` commit available to the source worktree.
- Confirmed the implementation branch starts at `517891f41d091e139b1e34f79c772e0f1265b8a3`.
- Read the acceptance charter, shard README, frozen declarations and kernel interface, interface tests, `Frame` primitives, and repository identity/serialization prior art in the prescribed order.
- Confirmed the locked SHA-256 digests for `decl.py` and `kernel.py` match `docs/graph-interface.lock`.
- Ran the untouched shard baseline: 16 interface tests pass.
- Reused the source worktree's synced environment read-only because this lane sandbox denies uv's global cache; all imports are forced to this clone.
- Implemented strict canonical JSON, domain-separated source/node/artifact/frame identities, and seed derivation.
- Implemented frozen decisions, complete per-node receipts, stable manifest content identity, JSON verification/round-trip, and attached final-population lookup.
- Implemented the one-screen node description with runtime receipt highlights; 19 focused tests pass and the touched layer is ruff-clean.
- Implemented atomic content-addressed columns, complete Frame versions, JSON receipts, and opaque bytes with load-time payload verification.
- Preserved dense/nullable dtypes, null bitmaps, UTF-8 strings without object arrays, and signed zero; equivalent frames serialize byte-for-byte across stores.
- Implemented `frame-store` and both supported `csv-tables` fixture layouts; 10 focused store/codec tests pass.

## Next

- Commit the completed population patch/weight/mass semantics.
- Integrate the executor and run the full charter-oriented unit suite.
