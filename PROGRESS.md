# Lane A progress

## State

Identity, storage, codecs, and immutable population transitions are implemented.

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
- Implemented storage-preserving masked patches, total ownership lineage, ABSENT enforcement, and exact dense/nullable dtype checks.
- Implemented structural id constraints, explicit immediate weight-kind transitions, per-stratum conservation, and the graph mass ledger; 14 focused population tests pass.
- Bound structural node keys to every completed member of their base version, preventing a cached structural Frame from carrying stale ordinary or weight-only patches.

## Next

- Integrate the executor and run the full charter-oriented unit suite.
- Replace the public API placeholders and run repository verification.
