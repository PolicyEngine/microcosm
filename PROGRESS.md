# Build E prereq 1: survivable target_compilation (issue #299)

Worktree: `~/PolicyEngine/_worktrees/populace-build-e-oomfix`
Branch: `build-e-compile-checkpoint` (from origin/main @ fd58d8b)
Driver: `tools/build_us_fiscal_refresh_release.py` (5,515 lines, 207 KB — NEVER read whole; grep/sed only)

## State: ROOT CAUSE FOUND; awaiting/assuming lead's call on scope (minute ~25)

## Task (from lead / Fable)
Make `target_compilation` survivable. Builds B & D died mid-compile on 224,026-hh pool,
2 of 5 JCT reform materializations complete. D on 220 GiB nonpreemptible box, likely OOM.
1. Study #227 checkpoint code, #236 streaming, #217 cache-key proposal. Determine WHY B's restart lost cache.
2. Implement per-reform durable checkpointing keyed on reform-vector + target-frame identity (#217).
   Bound peak memory: hold ≤1 reform's dense intermediate; chunk within-reform along household axis if needed.
3. TDD on tiny synthetic: kill-after-2-of-3, restart, assert 1&2 load / only 3 computes; cache keys invalidate on reform-vector/frame change.
4. pipefail pytest exit 0; ruff clean; PR via --body-file, no @-mentions, DO NOT merge. Note peak-mem bound.

## KEY FINDING (from #204 comment [11], 2026-07-03 — the failure analysis)
Prior agent claims the checkpoint machinery ALREADY WORKS:
- Per-reform JCT materialization writes atomic, content-addressed, sha256-verified cache entry
  to `<out>/artifacts/target_materialization_cache` AFTER each reform completes.
- Resume path re-reads verified entries at loop top; identity = base H5 + registry + PE version + commit + period + reform.
- Kill after 2/5 → 2 valid cached entries, restart resumes at reform 3 — PROVIDED out-dir is stable.
- **B's real failure: Modal auto-restarted to a FRESH out-dir, orphaning the cache** ("restart lost the materialization cache").
- D (nonpreemptible) hit hard OOM/timeout in a single unresumed pass.
- Prior proposed fix was ORCHESTRATION-ONLY (no code change): fixed volume `--out`, lower `--maximum-microsim-batch-size` 5000→~1000-2000, ≥300 GiB, commit() volume per reform.

## AMBIGUITY TO FLAG TO LEAD
Prior comment says checkpointing needs NO code patch (fix is orchestration). But lead directs me to
IMPLEMENT per-reform durable checkpointing + TDD. Need to verify against actual code whether:
(a) checkpointing genuinely already exists & works (then my job = harden/make out-dir independent of ephemeral path + TDD-prove + memory bound), or
(b) it does NOT actually persist per-reform (then my job = build it).
Resolve by reading the code before deciding.

## Done
- [x] Worktree created from origin/main
- [x] Read #299 spec + #204 comments [10] (A→B→D) and [11] (Build E halt / failure diagnosis)

## Next
- [ ] Grep checkpoint machinery in driver + packages (#227), streaming (#236), cache-key (#217)
- [ ] Read inspect_us_fiscal_refresh_checkpoints.py (reveals checkpoint format)
- [ ] Determine ground-truth root cause of B's lost cache
- [ ] Decide (a) vs (b) and flag to lead

## ROOT CAUSE OF B's LOST CACHE (code-confirmed) — the question #299 asks
Per-reform materialization checkpointing ALREADY EXISTS and is robust in isolation:
- `_read_reform_income_tax_cache` / `_write_reform_income_tax_cache` (lines 953-1029):
  atomic (os.replace), sha256-verified, shape-checked, identity-checked JSON+npy per reform.
- Reform loop (2998-3073): reads cache at loop top, computes on miss, writes IMMEDIATELY after
  each reform, `del reform_income_tax` + GC between reforms. Resume = "start at reform 3".
- Cache key (`_target_materialization_cache_identity`, 915-930) = context + reform_measure +
  neutralized_variable + n_households; context (5076-5086) = base_dataset_sha256, build_commit,
  policyengine_us_version, seed, period, target_registry_version, cd_crosswalk_sha256.

WHY B STILL LOST IT (two independent code-level fragilities):
1. **Ephemeral out-dir binding (primary).** `target_materialization_cache_dir` and
   `target_frame_checkpoint_path` DEFAULT to `args.out.resolve()/artifacts/...` (4825-4842).
   Modal preempted B → auto-restart to a FRESH out-dir → default cache path moved with it →
   the 2 completed entries orphaned on the dead ephemeral dir → all 5 recomputed. The `--target-
   materialization-cache-dir`/`--target-frame-checkpoint` overrides exist, but nothing forces a
   stable location; one missing flag (or Modal out-dir templating) reproduces B exactly.
2. **Over-coarse cache key (#217, compounding).** Reform-vector key includes `build_commit` +
   `target_registry_version`. A restart re-resolving HEAD to a newer commit, or any registry bump,
   invalidates all 5 reform entries even though the reforms/PE-US are unchanged. #217 wants the
   reform-vector key = ONLY (base_h5_sha256, policyengine_us_version, reform_id, period,
   cd_crosswalk_sha256).

## MEMORY-BOUND ANALYSIS
- Reform loop ALREADY household-batched (#236-style), default `--maximum-microsim-batch-size`=5,000
  (DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE=5_000, line 117). `_reform_household_income_tax` (2149-2194)
  loops batches, builds a fresh Microsimulation per batch, writes into preallocated output, GCs.
- Base materialization (2701-2977) runs ONE UNBATCHED Microsimulation over all 224,026 hh
  (line 2705) + full entity-table copy (2712) + ~a dozen variable caches held to line 2976. This is
  the un-chunked hotspot INSIDE target_compilation, before the reform loop.
- Driver's "~120 GB" dense worst-case = `~2·n_targets·n_records·8B` = 2×32,637×224,026×8 ≈ 117 GB
  is the CALIBRATION loss matrix — a LATER stage. D died in target_compilation, BEFORE calibration.
  So the >220 GiB kill on D is the base-sim + reform-loop peak, not the calibration matrix.

## AMBIGUITY FLAGGED TO LEAD (checkpoint semantics) — proceeding on best reading
Prior #204 comment [11] said "checkpointing needs NO patch; fix is orchestration-only." Code shows
per-reform checkpointing exists BUT has two real code-level fragilities (above) that make restarts
lose work in practice. Lead directs me to "implement per-reform durable checkpointing + TDD resume."
Reconciliation: the machinery exists; my job is to HARDEN it so a restart cannot silently orphan
completed reforms, and TDD-prove resume + key invalidation. Concretely:
  (A) Decouple the durable checkpoint location from the ephemeral out-dir: add a stable
      `--checkpoint-root` (or resolve cache/frame-checkpoint under a persistent volume path by
      default when provided) so a fresh out-dir on restart still finds completed reforms.
  (B) Implement #217's finer reform-vector key so a commit/registry bump on restart REUSES
      completed reforms, while still invalidating on base-H5 / reform-vector / PE-US / period /
      geography change (no stale poisoning).
  (C) Keep peak memory bounded: keep the reform-loop batch; OPTIONALLY batch the base-sim
      materialization too if worst-case says a single unbatched base pass exceeds the box.
  (D) TDD on tiny synthetic: kill-after-2-of-3 (raise), restart, assert 1&2 load / only 3 computes;
      assert key invalidates on reform-vector change and on frame identity change.
This is hardening, not a rewrite — matches both the lead's directive and the code reality.
Will proceed on this reading; will adjust if lead says otherwise.

## Decisions
- Treat task as HARDENING existing per-reform checkpoint machinery (not building from zero).
- Fix (A) ephemeral-out-dir binding + (B) #217 key are the load-bearing survivability fixes.
- Memory: reform loop already bounded; assess base-sim batching need via worst-case math before adding.
