# US #958 increment 2: stopped on final-owner contract conflict

2026-09-19, branch `us-958-full-vector-agi-tail`, starting revision
`16c8e78d2`.

Implementation stopped before changing production code, following the task's
instruction: "If you hit a design conflict with the repo's contracts that this
brief did not anticipate, stop and report it instead of forcing it."

The requested full-vector AGI arm conflicts with the reviewed downstream
final-owner contract for three PUF outputs. Changing only selection, transfer,
provenance, manifests and their gates would either lose donor values later or
fail terminal validation. Resolving this requires a decision about downstream
ownership, beyond updating identities and pins.

## Evidence and required decision

All three columns belong to `PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS`:

- `qualified_tuition_expenses`
- `traditional_ira_contributions_desired`
- `self_employed_pension_contributions_desired`

The following paths are relative to
`packages/microcosm-build/src/microcosm/build/us_runtime/`.

| Contract or execution point | Existing behavior |
| --- | --- |
| `us_late_overlap_ownership.py:194` | Declares `clone_2_policy = inherit_or_mirror_clone_1_final_owner_bytes`; the closed matrix covers three targets, two origins and three clone roles. |
| `multispine_pool.py:2225` | Runs the overlap finalizer after the post-clone retirement source callback. |
| `multispine_pool.py:2651` | Actually assigns clone-1 values to both retirement-contribution fields on clone-2 rows. |
| `stacked_spine.py:12716` | Requires every non-capital-gains PUF output on clone 2 to equal its clone-1 parent. |
| `us_late_producer_registry.py:1473` | Requires the reviewed ownership matrix to exhaust the multi-producer overlap and rejects unadjudicated tail-owned overlap. |

A synthetic two-person-table-row probe called the existing retirement finalizer
without changing any runtime code. It demonstrated these overwrites:

| Field | Clone-2 donor value before | Clone-1 value | Clone-2 value after |
| --- | ---: | ---: | ---: |
| `traditional_ira_contributions_desired` | 999.0 | 101.25 | 101.25 |
| `self_employed_pension_contributions_desired` | 777.0 | -0.0 | -0.0 |

Thus, a successful full-vector transfer can subsequently lose donor values by
design. Expanding terminal preservation to require those donor values would
expose the contradiction. Leaving the current preservation contract unchanged
would reject donor values that differ from clone 1.

The needed decision is whether AGI-arm donors become the final owners of all
PUF outputs through the terminal pool, explicitly superseding clone-1
inheritance/mirroring on those rows. If so, implementation must introduce
arm-specific ownership and callback protections, along with terminal checks,
while preserving the existing capital-gains-only behavior. No exemption,
ownership reassignment, bypass or post-hoc value restoration was implemented.
An independent read-only agent audit reached the same conclusion.

## Changed files and decisions

- `docs/implementation/us-958/PROGRESS.md`: committed state/done/next journal,
  maintained from the start and updated when the conflict was confirmed.
- `docs/implementation/us-958/FINAL_REPORT.md`: this final output report.

The brief did not name an output path. The journal and report are under
`docs/implementation/us-958/` to satisfy the committed-journal requirement while
respecting the prohibition on root journals and committed `.lane958/` files.

Scratch-only additions are `.lane958/ownership_conflict_probe.py`, its JSON
output, and verification logs. They were not staged or committed. The supplied
scratch inputs were read without running any of the real-data prototypes.

Local commits preceding this final-report step:

- `cea0460fb`: start the implementation journal.
- `627910928`: record the final-owner contract conflict.

No production source, tests, generated specs, coverage evidence or pins changed.
No pin was moved. No added-feature changelog was written because the feature
was not implemented. No implementation design choices were finalized for AGI
components, thinning quotas, deciles, head allocation or manifest structure.

## Verification commands and exit codes

The focused test command was run exactly as follows, with output redirected to
a file and success determined from the process exit code:

```sh
.venv/bin/python -m pytest packages/microcosm-build/tests/test_us_puf_capital_gains_tail.py packages/microcosm-build/tests/test_us_multispine_pool.py::test_source_overlap_finalizer_mirrors_retirement_tail_bytes packages/microcosm-build/tests/test_us_late_producer_dag.py::test_late_overlap_ownership_exhausts_every_permitted_dual_write packages/microcosm-build/tests/test_us_stacked_spine.py::test_run_stacked_puf_pass_applies_clone_two_capital_gains_tail > .lane958/contract-tests.log 2>&1
```

Exit **0**: **23 passed**, no skipped tests, one warning, 86.92 seconds. The
warning was joblib's inability to detect physical core count; it fell back to
logical core count. These are existing contract tests, not evidence that the
requested feature works.

```sh
.venv/bin/python .lane958/ownership_conflict_probe.py > .lane958/ownership-conflict.json 2>&1
```

Exit **0**: the synthetic probe confirmed both overwrite examples above and
asserted that all three conflicting fields belong to the requested full vector.

```sh
.venv/bin/ruff check . > .lane958/ruff-check.log 2>&1
```

Exit **0**: all checks passed.

```sh
git diff --check
```

Exit **0** before writing this report; repeated on the final documentation diff
before committing it.

## Unfinished and unverified

The AGI arm, thinning, full-vector transfer, extended provenance/manifest/gates,
behavioral regression tests, changelog and pin regeneration were not implemented
because of the explicit stop condition. The full build shard suite was not run;
the focused tests above establish the blocker and existing behavior only.
No formatter or spec/coverage/digest generator was run because no corresponding
implementation file changed.

No base build was run. Behavior on real data is unverified. L0 budget compliance,
recipient capacity after thinning, weighted proxy-AGI mass changes, concentration
gates on real donors and reform revenue effects are unverified. No real-data
build, calibration, release or publication tool was run; no network command,
push or PR creation was attempted.
