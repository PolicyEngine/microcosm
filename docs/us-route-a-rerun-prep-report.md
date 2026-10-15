# Route A preparation report — 15 October 2026

Prepared d1386 against cached main
`412f916536b66577e4e0b42c707285085df7c302`; no data build, Modal command,
credential lookup, upload, publication or push ran. The selected 18 October
commit is deliberately unset. See [the launch sequence](us-route-a-rerun-20261018.md)
for input findings, all uncertain settings, lineage requirements and commands.

## Files

- `docs/us-modal-stage-route-a-rerun-base-plan.json`: refuses validation until
  the commit placeholder is filled; CT ladder and ten historical base inputs.
- `docs/us-route-a-rerun-20261018.env.example`: reproducible settings, with new
  release feed, independent old base feed, new crosswalk, staging and d490.
- `docs/us-route-a-rerun-20261018.md`: exact launch-day commands and decisions.
- `docs/us-modal-stage-runbook.md`, `tools/route_a/README.md`: rerun links and
  registration/settings documentation.
- `tools/modal_us_stage_plan.py`: scoped rerun ToolSpec; preserves old replay.
- `tools/route_a/route_a.sh`, `route_a.env.example`: configurable explicit feed
  digest/size and crosswalk pin with historical defaults.
- `tools/route_a/make_local_reference.py`: derives evidence from real local
  000/001 checkpoints, context and verified inputs/source; launches nothing.
- `packages/microcosm-build/tests/engine_free/us/test_route_a_driver.py`,
  `test_us_modal_stage_puf_support_base.py`, `test_us_route_a_local_reference.py`:
  input-pin, registration, placeholder and provenance/refusal coverage.

## Validation

- 377 isolated engine-free tool tests passed, at most four pytest workers and
  one native numerical thread per worker. All added tests passed.
- `ruff check .` passed using locked Ruff 0.15.16.
- `ruff format --check` passed on all five changed Python files. The full
  repository check reports 45 formatting failures; every failing file's Git
  object hash was checked against cached main and is unchanged. They were
  preserved.
- `python3 tools/ci_test_plan.py verify` passed: 705 registered test files.
- Driver/settings and all seven documented Bash blocks pass `bash -n`.
- The documented pin-filling command was exercised using temporary plan and
  settings files and cached main. It produces a valid plan, refuses repinning,
  and the untouched template returns validation exit 2. The concrete plan's
  request-price timeout estimate is below $15 ($14.90).

The isolated pytest run used `--confcutdir=packages/microcosm-build/tests/engine_free/us`
to avoid the unrelated root telemetry fixture's installed-distribution needs.
The ordinary fixture-based invocation was attempted and failed because
`microcosm-fit` distribution metadata is absent. No project wheels/editable
packages were built. Four existing tests were excluded from the passing run:

- `test_the_tool_s_own_parser_reads_the_same_build_from_both_commands`
- `test_the_pinned_tool_locks_the_reference_s_run_config_from_the_modal_argv`
- `test_the_seed_stops_the_tool_s_census_download`
- `test_wrapper_secret_reaches_only_child_environment`

The first three require installed workspace distribution metadata; the last
requires `ps`, denied by the sandbox. Those checks must run in the launch-day
locked environment; the runbook's full test command includes them. This report
does not claim certification or a fully green ordinary environment.

## Sandbox and hand-off

Only the assigned workspace was written. Both `git fetch origin` and staging
files in the assigned checkout failed because Git metadata under the caller's
repository is read-only. The intended `_worktrees/` and machine-local settings
destinations are outside the writable sandbox too.

An independent cached-main checkout with its own Git metadata was therefore
prepared at `build/route-a-rerun-prep-main/` inside this workspace. Its branch is
`route-a-rerun-prep`; its commit is exported in the workspace-root
`route-a-rerun-prep.bundle` and `route-a-rerun-prep.patch`. These are portable
changes against cached main, avoiding the assigned checkout's older parent.
The assigned outer checkout remains **uncommitted**. It also holds copies of
the changed files for inspection. Use the bundle/patch to integrate with main;
do not treat the old outer checkout as a runnable current-main tree.

The actual uncommitted settings fallback is
`build/route-a-rerun-20261018/route_a.env`. The requested
`/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/route-a-rerun-20261018/route_a.env`
was not written. Install it with step 1 of the launch sequence; no actual
`route_a.env` is committed in the bundle.

Outside this sandbox, recover the committed prep worktree without pushing:

```bash
PE=/Users/maxghenis/PolicyEngine
MAIN=$PE/microcosm
ASSIGNED=/Users/maxghenis/.subfleet/worktrees/20261015-052227-route-a-rerun-prep
git -C "$MAIN" fetch "$ASSIGNED/route-a-rerun-prep.bundle" refs/heads/route-a-rerun-prep
git -C "$MAIN" worktree add -b route-a-rerun-prep "$PE/_worktrees/route-a-rerun-prep" FETCH_HEAD
```

That prep tree contains reviewed runner/driver changes. On 18 October, fetch
main again and select the build commit independently as the launch sequence
requires. If applying to a newer prep branch, use the patch or cherry-pick the
bundle's commit and review any conflicts. No historical run or worktree was
modified.
