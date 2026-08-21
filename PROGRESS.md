# Progress: one US target surface

## State

Migration in progress on `one-target-surface` from `origin/main` at
`2c7a7218`. The work will remove the congressional-district compilation
opt-in so every US artifact compiles one national + state + congressional-
district target registry; sparse and dense artifacts may differ in record
count, never target membership.

## Done

- Read `CLAUDE.md`, the target-parity declaration, the fiscal compiler and its
  never-controls doctrine, the current CD opt-in tests, and
  [microcosm#449](https://github.com/PolicyEngine/microcosm/issues/449) /
  [microcosm#569](https://github.com/PolicyEngine/microcosm/issues/569).
- Attempted the required `uv sync --all-packages --extra us`. The managed
  sandbox denied writes to the default uv cache, then its network restriction
  prevented a clean-cache download of `pyvis`. A byte-identical-lock sibling
  environment was cloned copy-on-write; tests use that complete environment
  with this worktree's package sources first on `PYTHONPATH` because an offline
  editable reinstall still requires unavailable build-isolation metadata.
- Attempted the GitNexus refactoring impact workflow. Local indexing completed,
  but GitNexus could not register the index because the sandbox forbids writing
  `~/.gitnexus/registry.json`; a direct source/call-site audit is the fallback.
- Confirmed the starting worktree was clean and no build or push was run.
- Ran the workspace suite for 1,137 seconds with no failure before interrupting
  it inside the unrelated PUF-QRF stale-checkpoint subprocess regression; the
  affected US target/compiler shard is the per-commit validation boundary, with
  a complete workspace run reserved for the final tree. Ruff is green.
- Established a green 10-file affected-suite baseline covering target
  compilation, parity, the release builder/scorers, CD vintage translation,
  Ledger profiles, and the generated calibration contract (100% in 349.59s).

## Next

- Remove every `include_congressional_district_targets` branch and caller while
  retaining the SOI row-level rule that CD aggregates never become national
  controls.
- Update parity doctrine/tests, prove target-registry identity across artifact
  scale, quantify the target-row delta and existing 25% timing evidence, then
  write `FINAL_REPORT.md`.
