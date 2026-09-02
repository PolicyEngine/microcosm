# H2 UK SPI fixture composition report

Date: 2026-09-02

Branch: `h2-fixture-spi-composition`

Verified handoff commit: `470ced30efbac6bcc9b985e8835ec50857e2d611`

Worktree: `/private/tmp/microcosm-h2-fixture-spi-composition`

No push was performed.

## Outcome

The hermetic H2 UK spine fixture now carries the composition-changing mass
property seen on licensed FRS data. Child presence follows
`household_id % 5`, independently of the `% 4` region cadence, so the
region-stratified SPI support allocation redistributes household weight across
different household sizes. The fixture remains small and deterministic: 135
root households and 162 root people.

The real fixture-backed SPI stage now has these totals:

- weighted person mass before: `272.7`
- weighted person mass after: `272.43`
- weighted person-mass delta: `-0.2699999999999818`
- household mass before: `226.8`
- household mass after: `226.8`

For comparison, the old fixture produced person mass `282.24 -> 282.24` and
therefore could not reject `mass="conserve"`.

The generator and committed `benunit`, `child`, and `person` fixture sources
were regenerated. A regression in
`packages/microcosm-build/tests/test_uk_graph.py` runs the real FRS root and
SPI support transforms from the committed fixture, asserts person mass changes
at the executor's `rtol=1e-9`, and asserts exact household-mass conservation in
both the Frame weights and mass-log receipt. The requested one-line towncrier
fragment was added.

`PRODUCED_BY.txt` and `NORMALIZATION.md` were not changed because their
statements remain accurate. The regenerated machine-local legacy oracle
identity was
`837a67b5d7fd6dd2a974ac41d772d6cb809ff2111d59b8b7c9c2a738ce2b42fa`;
as documented, it is not pinned.

## Deliberate `conserve` proof

I temporarily changed `_STRUCTURAL_MASS["spi_support_channel"]` from
`"declared"` to `"conserve"` and ran the fixture-backed driver path. The driver
path was used because H2 first compares the pinned graph JSON and would stop on
that declaration mismatch before executing the support node. The production
source was restored immediately afterward and has no residual diff.

Command:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-h2-uv-cache uv run pytest -o addopts='' packages/microcosm-build/tests/test_uk_graph.py::test_driver_projects_a_stage_record_for_every_graph_stage_on_the_fixture -q --tb=short
```

Failing output, verbatim:

```text
F                                                                        [100%]
=================================== FAILURES ===================================
___ test_driver_projects_a_stage_record_for_every_graph_stage_on_the_fixture ___
packages/microcosm-graph/src/microcosm/graph/executor.py:863: in _apply_result
    return patch(population, node, result)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
packages/microcosm-graph/src/microcosm/graph/population.py:605: in patch
    record = _mass_record(before, frame, node, result, policy)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
packages/microcosm-graph/src/microcosm/graph/population.py:1468: in _mass_record
    _assert_mass_mapping(
packages/microcosm-graph/src/microcosm/graph/population.py:1569: in _assert_mass_mapping
    raise PopulationError(
E   microcosm.graph.population.PopulationError: Node 'spi_support_channel' mass='conserve' changed stratum 'default': 272.7 -> 272.43.

The above exception was the direct cause of the following exception:
packages/microcosm-build/tests/test_uk_graph.py:475: in test_driver_projects_a_stage_record_for_every_graph_stage_on_the_fixture
    manifest = run_graph(
packages/microcosm-graph/src/microcosm/graph/executor.py:1397: in run_graph
    updated = _apply_result(node, result, incumbent, cache_hit=hit)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
packages/microcosm-graph/src/microcosm/graph/executor.py:867: in _apply_result
    raise NodeRejected(f"Node {node.id!r} patch rejected: {error}") from error
E   microcosm.graph.errors.NodeRejectedError: Node 'spi_support_channel' patch rejected: Node 'spi_support_channel' mass='conserve' changed stratum 'default': 272.7 -> 272.43.
----------------------------- Captured stdout call -----------------------------
microcosm-fit: near-discrete target (13 distinct values); bounding max_samples_leaf at 4096 for this forest.
microcosm-fit: near-discrete target (3 distinct values); bounding max_samples_leaf at 4096 for this forest.
microcosm-fit: near-discrete target (14 distinct values); bounding max_samples_leaf at 4096 for this forest.
microcosm-fit: near-discrete target (25 distinct values); bounding max_samples_leaf at 4096 for this forest.
microcosm-fit: near-discrete target (32 distinct values); bounding max_samples_leaf at 4096 for this forest.
microcosm-fit: near-discrete target (32 distinct values); bounding max_samples_leaf at 4096 for this forest.
microcosm-fit: near-discrete target (32 distinct values); bounding max_samples_leaf at 4096 for this forest.
microcosm-fit: near-discrete target (32 distinct values); bounding max_samples_leaf at 4096 for this forest.
----------------------------- Captured stderr call -----------------------------
  File "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-graph/.venv/lib/python3.14/site-packages/joblib/externals/loky/backend/context.py", line 249, in _count_physical_cores
    cpu_count_physical = _count_physical_cores_darwin()
  File "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-graph/.venv/lib/python3.14/site-packages/joblib/externals/loky/backend/context.py", line 318, in _count_physical_cores_darwin
    return int(cpu_info)
=============================== warnings summary ===============================
packages/microcosm-build/tests/test_uk_graph.py::test_driver_projects_a_stage_record_for_every_graph_stage_on_the_fixture
  /Users/maxghenis/PolicyEngine/_worktrees/microcosm-graph/.venv/lib/python3.14/site-packages/policyengine_uk/utils/scenario.py:8: PydanticDeprecatedSince20: Support for class-based `config` is deprecated, use ConfigDict instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.13/migration/
    class Scenario(BaseModel):

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED packages/microcosm-build/tests/test_uk_graph.py::test_driver_projects_a_stage_record_for_every_graph_stage_on_the_fixture
1 failed, 1 warning in 27.15s
```

## Verification

The locked UK environment sync completed:

```text
Resolved 125 packages in 2ms
Checked 98 packages in 2ms
```

The exact requested aggregate command
`uv run pytest packages/microcosm-graph packages/microcosm-build/tests/test_uk_graph.py -q`
exited 0. Because repository `addopts` already contains `-q`, that exact
invocation is double-quiet and suppresses the count line. The same test scope
was rerun with configured `addopts` cleared while retaining the requested
command-line `-q`; its exact summary was:

```text
208 passed, 1 skipped, 1 warning in 124.47s (0:02:04)
```

Other requested checks:

- `uv run ruff check .`

  ```text
  All checks passed!
  ```

- `uv run ruff format --check tools/graph_uk_spine_fixture.py packages/microcosm-build/tests/test_uk_graph.py`

  ```text
  2 files already formatted
  ```

- `uv run python tools/ci_test_groups.py --verify`

  ```text
  verification=ok
  ```

- `uv run python tools/graph_acceptance_burndown.py --verify --baseline origin/main`

  ```text
  green     37  A1 A2 A3 A4 A5 A6 A7 B1 B2 B3 B4 B5 C1 C2 C3 C4 D1 D2 D3 D4 D5 E1 E2 E3 E4 E5 F1 F2 F3 F4 F5 G1 G2 G3 H1 H2 H3
  red        0  none
  missing    0  none
  verification=ok
  ```

- `git diff --check origin/fix-spi-support-channel-mass-836...HEAD` exited 0
  with no output.

## Limitations

No requested work was omitted. The sandbox denied writes to the default warm
uv cache, so only the cached build requirements were copied into a writable
temporary cache; the locked sync and every requested command then completed
without network access. GitNexus repository resources were not exposed in this
session, so call/dependency and change-impact analysis was performed directly
from source, tests, and the branch diff.
