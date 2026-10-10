# US head-to-head slice memory investigation

The measured retention is PolicyEngine Core's per-system variable modules in
`sys.modules`. This checkout mitigates it after each completed
`_score_household_slice`; it does not modify the installed engine.

## Retaining reference and calculation path

In the measured Core 3.32.5 installation,
`policyengine_core/taxbenefitsystems/tax_benefit_system.py:288–293`
constructs `<system id>_<absolute path hash>_<file stem>` names and assigns
`sys.modules[module_name] = module` in `add_variables_from_file`. There is no
corresponding removal when a system is discarded. A retained module owns its
variable classes and formula globals, independent of simulation teardown.

The scorer calls `release._materialize_target_frame` for every household slice.
In `tools/build_us_fiscal_refresh_release.py:6813` that constructs a fresh
`CountryTaxBenefitSystem` for metadata. Its reform measure path additionally
constructs `microsimulation_cls.default_tax_benefit_system(reform=reform)` at
line 5349 for every requested JCT reform. On the measured ledger, chunk 5
contains 73 specs and 11 reform families: **12 system constructions and 71,880
additional module registrations in the measured first slice**, even with only
two households. Subsequent growth can depend on system-id reuse, which
overwrites an old registration; the registry has no lifetime bound.

`score_targets` returns arrays and plain contracts, without storing an engine
or frame in a module registry. Existing simulation teardown already releases
holders, branch simulations, datasets, tracers and simulation backreferences.
The materialized reform arrays belong to the slice frame. The frame adapter's
source index is bounded to one entry and contains source metadata. None removes
Core's module registrations.

Shared parameter-tree date caches are a separate, intentional source of
first-calculation warming. Core's `ParameterNode._at_instant_cache`
(`parameters/parameter_node.py:226`) and the tax-benefit system's
`_parameters_at_instant_cache` retain derived date views. Microcosm's
`share_spm_policy` reuses the default system's parameter tree. The full first
slice increased derived `ParameterNodeAtInstant` objects from 12,621 to 397,140
while raw `ParameterNode`, `Parameter` and `ParameterAtInstant` counts remained
unchanged. The mitigation leaves these policy caches untouched.

## Mitigation and recycling

`temporary_engine_variable_modules` in
`packages/microcosm-build/src/microcosm/build/us_runtime/engine_lifecycle.py`
snapshots Core's generated registrations under the US variables directory.
The scorer's slice scope removes new registrations after all calculations and
source inspection complete, preserving existing registrations, ordinary
imports, unrelated paths, and the shared default system when imported lazily.
It restores pre-existing entries if Core reused a discarded system's id.
The returned slice contains no live engine objects.

`--worker-max-slices K` additionally passes `max_tasks_per_child=K` to the spawn
executor. The default remains no recycling. An explicit K also uses the pool
with `--workers 1`. Task submission and ordered reduction remain unchanged.

## Measurement method

The reproducible probe is [scripts/profile_slices.py](scripts/profile_slices.py).
It uses the scorer's own slice function, two consecutive incumbent households
per slice, actual ledger targets, `tracemalloc`, `gc.get_objects()` type counts,
and **current** `psutil.Process().memory_info().rss`. Both before and after
censuses collect unreachable cycles; RSS is not `ru_maxrss`. A watchdog exits
at 14 GiB, leaving headroom below the requested 16 GiB cap.

The interpreter is the existing, unchanged routea environment:
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-h2h-routea/.venv/bin/python`,
PolicyEngine US 2.2.1 / Core 3.32.5. Microcosm imports explicitly use this
checkout. The input is the snapshot filename (not its resolved blob):
`/Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/8ab57ffc2ca41d8af631ff62ebbce95e966299f3/populace_us_2024.h5`.
The ledger is
`/Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_c5e5bf8.jsonl`.
Only aggregate counts, allocation sites and score hashes are recorded.

The [full-chunk first-slice measurement](full-chunk-before.json) reproduced the
retention using all 73 chunk-5 specs: current RSS **4.289 → 7.460 GiB**,
US-variable module registrations **12,080 → 83,960**, and live traced allocation
**1,353,298 → 925,282,840 bytes**. Maximum watchdog-sampled RSS was 7.611 GiB.
This first slice took 1,529 seconds on the overloaded Mac with tracing enabled;
the full probe was stopped after attribution. Consecutive comparisons restrict
chunk 5 to its non-reform specs and its first requested reform family to reduce
system construction work while exercising the same leak.

The [three-slice baseline](before.json) uses 63 specs, including the SALT reform.
Each consecutive two-household slice constructs two systems. After each slice,
variable registrations number 24,060, 36,040, and 48,020. Derived parameter-node
counts are 397,140, 425,706, and 425,706; raw parameter-tree counts remain flat.
After this warm-up, traced live memory still increases by 112,984,364 bytes on
slice 3, alongside exactly 11,980 new modules. The full chunk repeats six times
as many system constructions, explaining why its tiny last chunk can consume
more memory than much larger target chunks. Scaling the warm two-system traced
increase to 12 systems gives about 0.63 GiB per slice, consistent with the
reported approximately 0.6 GiB slope (an extrapolation, not a full-chunk RSS
measurement after warm-up).

Reproduce the smaller baseline from this workspace (remove
`--without-cleanup` and change the output name for the paired cleanup run):

```sh
PYTHONDONTWRITEBYTECODE=1 /Users/maxghenis/PolicyEngine/_worktrees/microcosm-h2h-routea/.venv/bin/python \
  docs/evidence/us-h2h-memory/scripts/profile_slices.py \
  --incumbent /Users/maxghenis/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/8ab57ffc2ca41d8af631ff62ebbce95e966299f3/populace_us_2024.h5 \
  --ledger /Users/maxghenis/PolicyEngine/_buildh-runtime/inputs/consumer_facts_us_c5e5bf8.jsonl \
  --reform-measures 1 --without-cleanup \
  --output docs/evidence/us-h2h-memory/before.json
```

The [paired cleanup measurement](after.json) uses identical inputs and slice
positions. All three estimate/target/scale byte hashes match the baseline.
Current RSS before and after **each** slice is:

| Slice | Without cleanup, GiB | With cleanup, GiB | Without cleanup, post-slice traced MiB | With cleanup, post-slice traced MiB |
| --- | --- | --- | --- | --- |
| 1 | 4.423 → 6.405 | 4.416 → 6.373 | 380.707 | 283.236 |
| 2 | 6.526 → 7.724 | 6.462 → 7.490 | 498.339 | 406.348 |
| 3 | 7.724 → 7.740 | 7.563 → 7.537 | 606.091 | 406.460 |

Maximum sampled RSS was 7.862 GiB without cleanup and 7.565 GiB with cleanup.
RSS includes the full loaded incumbent, engine state and profiler overhead; it
does not return immediately to the amount of live Python data. Tracing begins
after loading, so traced bytes exclude allocations already live at startup.
For example, another collection before fixed slice 3 reduced its traced total
from 426,086,647 to 318,136,361 bytes without changing engine-object counts.
The post-slice totals are almost identical after fixed slices 2 and 3, rather
than growing by another 113 MB as in the baseline.

Object counts after each slice:

| Retained object | Baseline slices 1 / 2 / 3 | Cleanup slices 1 / 2 / 3 |
| --- | --- | --- |
| US variable modules registered | 24,060 / 36,040 / 48,020 | 12,080 / 12,080 / 12,080 |
| All GC-tracked modules | 28,644 / 40,624 / 52,604 | 16,664 / 16,664 / 16,664 |
| Classes | 32,916 / 45,124 / 57,332 | 20,708 / 20,708 / 20,708 |
| Functions | 128,633 / 137,477 / 146,321 | 119,789 / 119,789 / 119,789 |
| Derived parameter date views | 397,140 / 425,706 / 425,706 | 397,140 / 425,706 / 425,706 |

Raw parameter counts stay constant in both runs: 21,073 `ParameterNode`,
112,328 `Parameter`, and 1,204,745 `ParameterAtInstant`. Core's unbounded
`Enum._get_sorted_lookup_arrays` cache and vectorial per-node enum LUT caches
were inspected; this path's module/class/function measurements show no further
accumulation requiring those caches to be cleared. No parameter, enum,
simulation or formula cache was patched or cleared by this change.

These are small real-engine probes, not a rerun of the 10-hour full scoring job.
The full-chunk first slice establishes the multiplicative registration count;
the paired smaller run establishes release of that graph without changing
observed scores. Recycling bounds each worker's lifetime across slices even if
other engine or allocator state behaves differently on later households.

## Regression validation

The registered engine-free US scorer module and slice-memory module passed:
**42 tests** (36 scorer tests and six memory tests). The CLI fixture compares
sequential scoring with three workers without recycling, two workers with K=1
and K=2, and one worker with K=2. It compares scorecard JSON apart from explicit
run metadata, Markdown bytes and float64 estimate bits. Per-PID call logs also
verify the configured lifetime limit, across registry chunks.

The memory regressions use an **engine-free stand-in**, with Core's exact
generated module naming convention and a module/class/formula reference graph.
Six consecutive calls through the actual slice scorer leave all weakrefs dead
after collection and preserve score bits. Other cases cover existing modules,
id reuse, unrelated paths/imports, exceptions, nested scopes, and the first lazy
engine import's shared system. The real-engine probes supply independent
measurement of the production retaining reference.

The registered engine-workflow US scorer module also passed all **four** tiny
H5 loader/scoring fixtures in one fresh real-engine process. Total focused
validation is **46 passed**. Engine-free tests took 542.06 seconds (501.75 seconds
for the spawn/recycling fixture); H5 workflow tests took 91.38 seconds on this
shared Mac. The workflow log is in ignored
`out/h2h-memory/engine-workflow-pytest.log`.

`ruff check .` and formatting checks for all six changed Python files passed.
`tools/ci_test_plan.py verify` passed with the new regression module in the
existing engine-free US category. Repository-wide `ruff format --check .`
reports 54 pre-existing unformatted files; none is part of this change.

## Handoff

All edits and measurement artifacts are inside the assigned worktree. No
dependency environment was modified and nothing was pushed. Changes are
uncommitted: Git cannot create
`/Users/maxghenis/PolicyEngine/microcosm/.git/worktrees/microcosm-h2h-parallel/index.lock`
because the worktree's Git metadata lies outside the writable workspace.
The branch remains `h2h-scorer-workers`, based on
`022967d7d1f4d27aae84c2e45957cf9ad64ae0d9`.
