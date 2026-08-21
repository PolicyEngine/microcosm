# Progress: one US target surface

## State

Runtime unification is complete on `one-target-surface`: every US entrypoint
now compiles one national + state + congressional-district target registry,
and the CLI/config switches that could delete CD or JCT target rows are gone.
Parity doctrine now protects that family as a red-line compile. Sparse and dense
artifacts may differ in record count, never target membership.

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
- Removed the congressional-district compilation option throughout the fiscal
  compiler, builder, fiscal scorer, state-file scorer, ACS local tool, aging
  diff, experiments, tests, docs, and generated contract. The canonical CD
  crosswalk is now the default at each production entrypoint.
- Removed the diagnostic JCT target-deletion option and its release-gate bypass;
  diagnostic tools now score the same registry as releases.
- Removed the parity generator's CD regime switch and regenerated the pinned
  manifest to 32 compiled / 52 reviewed families. The generated calibration
  contract declares all three geography layers and has no default-layer split.
- The 10-file affected suite reaches 100% with exit 0 after the runtime change;
  Ruff, byte compilation, and `git diff --check` pass.
- Promoted the CD family into the parity anti-rot red-line set, pinned the
  manifest's 32/52 header counts to parsed family counts, and asserted that its
  compiled entry carries no exclusion fields or fence.
- Strengthened the fiscal invariant: the CD aggregate is present in the
  compiled registry while the taxable-interest rebase still refuses it as a
  national control. The standard 10-file affected suite reaches 100% with exit
  0 after the parity-doctrine change.

## Next

- Remove the artifact-specific support-exclusion path, prove target-registry
  identity across artifact scale, quantify the target-row delta and existing
  25% timing evidence, then write `FINAL_REPORT.md`.
