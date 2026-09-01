# Lane A progress

## State

The runtime, store, immutable population transitions, manifest, and public API are implemented against the amended frozen interfaces; repository-level verification and publication remain.

## Done

- Created an isolated clone at the exact `origin/node-graph` commit available to the source worktree.
- Confirmed the implementation branch starts at `517891f41d091e139b1e34f79c772e0f1265b8a3`.
- Read the acceptance charter, shard README, frozen declarations and kernel interface, interface tests, `Frame` primitives, and repository identity/serialization prior art in the prescribed order.
- Confirmed the locked SHA-256 digests for `decl.py` and `kernel.py` match `docs/graph-interface.lock`.
- Rebased onto amended `origin/node-graph` commit `bb0d48d2`, adopted `KernelResult.keep`, `KernelRole`, and the shared Error-suffixed runtime exceptions, and reconfirmed the amended frozen hashes (`729e1487...` and `e650589c...`).
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
- Carried immutable original-design weight anchors by entity id across filters and reweights, enforced calibrated `max_weight_ratio` against that anchor, and authored the cap receipt from verified runtime state.
- Bound structural node keys to every completed member of their base version, preventing a cached structural Frame from carrying stale ordinary or weight-only patches.
- Bound every declared source consumer, not just CREATE nodes, to the verified source content key.
- Required FILTER, EXPAND, and REWEIGHT to carry incumbent rows' physical column storage unchanged after aligning by entity id.
- Gave typed-weight artifacts their own hash domain, protected retained explicit-weight topology and bytes across structural nodes, and rejected weight transitions that do not create a structural version.
- Reserved entity ids and person-membership columns to structural nodes so implicit context coordinates cannot change outside node-key dependencies.
- Integrated canonical-order execution, declared-slice read-only contexts, mutation hashing, strict result validation, cache reuse/preflight, deterministic RNGs, source verification, artifact writes, and abort-on-rejection behavior.
- Enforced exact CREATE data declarations and normalized malformed result containers into node rejections.
- Completed per-node manifest identities for columns, structural Frames, typed weights, and opaque artifacts; attached transient populations and mass ledgers without changing portable manifest identity.
- Replaced the package's runtime placeholders with the implemented public APIs and failures; 37 focused executor/manifest/population tests pass together.
- Made `resume="forbid"` serialize every result through atomic staging without reading or validating an incumbent object, including collision/interruption coverage.
- Made write-only collisions replace even a corrupt incumbent through a rollback-capable directory swap, so a successful `forbid` rerun leaves inspectable recomputed bytes.
- Allowed dense floating columns to be created under a row mask with exact-dtype NaNs outside ownership; dense bool/integer columns remain correctly unrepresentable there.
- Added id-only projections for owned output entities, executor-applied FILTER masks, per-store codec availability, authoritative capabilities in receipt highlights, and safe omission of ambiguous inherited group weights.
- Preserved the original acceptance decision-record mapping as a compatibility form of immutable `Decision`, without allowing it into node identities, and rejected embedded NULs in hash domains.
- The local shard suite now passes all 84 tests; compatible black-box acceptance probes for A3/A5/A7, B1/B2, D1/D2/D5, E2/E5, and G1 also pass un-xfailed.

## Next

- Run the full charter-oriented suite and repository verification, then publish the branch and draft PR.
