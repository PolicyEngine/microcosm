# Progress: round 11 checkpoint nullable booleans

## State

Investigation is in progress on `tail-stratum-support-652` at the requested
starting commit `cd4faa33`. The real 1% US build reached a frame-checkpoint
write before failing because `person.is_female` had pandas nullable `boolean`
dtype, which the current checkpoint schema rejects.

## Done

- Confirmed the checkout is clean, on `tail-stratum-support-652`, and exactly
  at `cd4faa33` before changes.
- Confirmed that commit already merges the locally available `origin/main` at
  `d1714a7c`; no network operation was performed.
- Loaded the repository, PolicyEngine data, development-standard, and
  debugging guidance.
- Identified the immediate investigation targets: the failing build receipt,
  every frame-checkpoint call site, the canonical dtype-family registry, and
  all checkpoint consumers including UK rowwise and legacy paths.

## Next

- Reconstruct the exact checkpoint stage and enumerate every extension-dtype
  column present at every checkpoint boundary.
- Add registry-driven red tests for canonical dtype-family round-trips and
  byte-identical legacy artifacts without extension dtypes.
- Implement lossless nullable serialization with a version bump, then run the
  requested focused, #583, full-workspace, lint, format, and golden proofs.
