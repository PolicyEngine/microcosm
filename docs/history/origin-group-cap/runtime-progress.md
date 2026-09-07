# Origin group cap runtime check history

## State

2026-09-06: Implemented the bounded invented-data audit launcher in the accepted
integration worktree. The root agent verified the saved agreement gate and exact
canonical plan. This supporting task read the complete current plan, repository
AGENTS.md and CLAUDE.md, and the executable CI inventory.

## Done

- Inspected branch and working-tree state without reading genuine source data,
  target values, checkpoints, or runtime stores.
- Verified the prescribed Python executable SHA-256 is
  `bf4ea26231212e330b8e9b6f24969a30f894fec88665bf42754070d114ac7fc6`.
- Identified required CI inventory and graph burndown checks, Ruff lint, the
  explicit test group matrix on Python 3.13/3.14, and the installed-wheel gate.
  The full suites, installations, and wheel builds are outside this local task.
- Added `tools/run_origin_group_cap_checks.py`: exact interpreter/version checks,
  explicit absent parent/child bytecode prefixes, isolated child, integrated
  shard sources, CPU/wall bounds, BLAS/Torch intra-op threads one, and full
  loaded-source byte hashes compared with Git before and after each check.
- The source-only CI inventory passed with 445 tracked test files; full output
  and source evidence are in the review directory's
  `us-origin-group-cap-runtime-inventory-003.json` and `.json.log`. Attempts 001
  and 002 preserve the observed macOS refusal to change its advisory RSS limit.
  The launcher now records the inherited RSS limits instead, and does not claim
  enforced memory isolation. CPU and wall bounds remain enforced.
- Ruff formatting and lint passed using the already-installed Mach-O native
  Ruff executable; it does not spawn a Python interpreter.
- The source-only graph burndown passed against the fetched `origin/main` with
  zero red or missing properties and 41 green properties. Its evidence is
  `us-origin-group-cap-runtime-burndown-001.json` and `.json.log`.
- Inspection found that Ruff's module entrypoint uses `execvp` on macOS, losing
  the check child's final audit. The initial `runtime-lint-final-001` evidence
  preserves that limitation. The launcher now invokes Ruff's native binary
  as a subprocess and refuses any absent/incomplete child audit even when a
  command returned zero. Final audit validation follows in a fresh evidence file.
- Final audited Ruff check passed at commit
  `1a5c4c4cd5071173640252f696eb0fa07581f148`. All 27 loaded repository Python
  sources matched Git both before and after, with no source mutation or foreign
  Microcosm import. Runtime was Python 3.14.4, NumPy 2.4.6, Torch 2.12.0,
  `arm64/darwin/py3.14`, CPU, Torch intra-op/inter-op 1/1. Parent and child
  pycache prefixes remained absent.
- The first complete solver test run exposed an inventory omission: package
  `tests/conftest.py` was loaded but excluded from the before/after snapshot.
  The harness correctly failed its source audit despite 42 passing tests.
  Source-only snapshot coverage now includes those configuration modules and
  Python helpers in this task's history directory; prior failed evidence remains.

Evidence SHA-256 values (files live in the architecture review directory):

- `us-origin-group-cap-runtime-inventory-003.json`:
  `51bfd5b654ebd78aa93157ae7107228f488b3a8c3324f1b2bcb49d08fb0fe560`
- `us-origin-group-cap-runtime-burndown-001.json`:
  `a35be3e30e121a3bae3f91c33547c2cc57fbcb0877559ffe5bb30eef2f13ada0`
- `us-origin-group-cap-runtime-lint-final-002.json`:
  `91ad374761703cfc08c84df0ca849732aea396f2c44054ac250b84a8141f337d`
- `tools/run_origin_group_cap_checks.py`:
  `22c6c74303ed18d9bbe3845119b67ada9327fcee9af806d0739210bfeadbb40e`

## Invocation

From this integration checkout, select fresh evidence and parent prefix names:

```sh
/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-launch-current-main-20260905/.venv/bin/python \
  -B -X pycache_prefix=/private/tmp/origin-cap-parent-UNIQUE \
  tools/run_origin_group_cap_checks.py \
  --output /Users/maxghenis/architecture-reviews/microcosm-20260904/CHECK-UNIQUE.json \
  -- pytest packages/SHARD/tests/test_INVENTED.py
```

Use `-- script tools/ci_test_groups.py --verify` for the source-only inventory,
`-- script tools/graph_acceptance_burndown.py --verify` for the source-only
ratchet with the already-fetched baseline, or `-- module ruff check PATHS` for
lint. Scripts must be checked-in or newly authored source in this checkout.
No package is installed. Default CPU/wall limits are 180/195 seconds; an explicit
`--seconds` may raise CPU time to at most 600 seconds. Logs and JSON are retained.

The legacy fixture recorded Torch intra-op 1 and inter-op 18 before the solver
was edited. Only that matching regression uses `--torch-interop-threads 18`;
the default is 1, BLAS and intra-op remain 1, and actual settings are recorded.
Dirty and new source bytes are reported honestly; changing any loaded source
during a check or importing Microcosm from a different checkout fails its audit.

## Next

- Independent admission probes were added in
  `test_us_origin_group_calibration_admission.py` (commit `6a0141e52`), covering
  real cold/warm solver-bypass refusal for both inequalities and actual typed
  group-map cache rekeying. The first audited run preserved in
  `us-origin-group-cap-admission-001.json` reached and passed both cold refusal
  assertions, then failed replay at complete budget identity checks. The root
  investigation identified cached population metadata loss; those tests remain
  red until the real storage path preserves the complete identity. The audit
  recorded no loaded-source mutation during this run.
- The implementing agents run the selected invented solver, budget, lineage,
  and adapter tests through this launcher. Each additional Python subprocess
  created by a test must also use explicit `-I -B -X` and its own fresh absent
  absolute pycache prefix; the launcher's direct child flags do not propagate
  implicitly to new Python processes. No full test suites, wheel installation,
  engine certification, genuine fit, or launch calibration was performed here.
