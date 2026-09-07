# Enriched-origin age recount: exact indexed rescale

Session journal for the bounded performance/correctness change to
`verify_counts` in
`packages/microcosm-build/src/microcosm/build/us_runtime/enriched_origin_calibration.py`.
Historical once merged — see CLAUDE.md "Root journals are history, not state".

## Scope

Replace the avoidable `O(bands x households x persons)` independent age recount
(frozen source lines 490-514) with an exact indexed integer membership recount.
Implementation optimization only. No change to weights, the count kernel, the
band schema, population selection, PUF route/recipe, solver, targets, or
admission. All materialized checks, `same_carried`, ledger/anchor verification
and refusals are preserved.

## Provenance

- Frozen accepted integration (unchanged, root's small genuine run keeps using it):
  `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-origin-cap-puf-integration-20260907`
  at `669edf780d4b45e09eb4e38a9e458798b04352d2`, branch
  `us-origin-cap-puf-integration-20260907`, clean working tree.
- Upstream context: `origin/main` = `c1b8324183819cf91511c002bf724a2676524e87`;
  the frozen branch is 310 ahead / 0 behind, merge-base is `origin/main`.
- Successor worktree (this one):
  `/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-enriched-recount-scale-20260907`,
  branch `us-enriched-recount-scale-20260907`, base
  `669edf780d4b45e09eb4e38a9e458798b04352d2`.

## Runtime

Borrowed interpreter
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-launch-current-main-20260905/.venv/bin/python`,
Python 3.14.4, sha256
`bf4ea26231212e330b8e9b6f24969a30f894fec88665bf42754070d114ac7fc6`.
That venv carries `_editable_impl_microcosm_*.pth` pointing at a *different*
worktree, so every source-loading process runs
`-I -B -S -X pycache_prefix=<fresh absent absolute>` with an explicit bootstrap
that prepends this worktree's `packages/*/src` and appends the venv
site-packages. Loaded-project Git-byte proof is recorded per run.
numpy 2.4.6, pandas 3.0.3, already installed; nothing installed or updated.
Caps per process: 180s CPU (RLIMIT_CPU), 240s wall (watchdog), 2 GiB
(RSS watchdog; macOS refused RLIMIT_AS), one compute thread.

## State

Bootstrapping.

## Done

- Inspected frozen checkout status/branch/remotes; fetched and compared upstream.
- Created the successor worktree from the accepted integration.
- Built the isolated runner and proved loaded-project Git bytes
  (260 `microcosm.*` modules, all inside this worktree, all matching HEAD blobs).

## Next

- Implement the exact indexed recount.
- Invented-only direct tests against a brute-force independent oracle.
- Synthetic timing/resource witness at ~170k households / ~386k persons.
- Independent multi-lens review; address file/line findings.
- Changed-key inventory and report.

## Failed attempts

(none yet)
