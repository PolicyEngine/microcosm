# Graph amendments 11–13 acceptance report

Date: 2026-09-02

Branch: `acceptance-11-13`

Acceptance implementation SHA: `721c5d4adf146d46e0f1107001872f3329571e53`

Base: `graph-amendments-11-13` at
`416bfab5747a6f6c29bdb3b16f1cb1d21f3412a8`

## Outcome

The acceptance lane now contains exactly three new strict-red charter tests:

- B6, `test_b6_entrants_are_declared`, specifies declared null-lineage
  entrants, complete carried-column materialization, copy/entrant lineage
  receipts, undeclared-entrant refusal, incomplete entrant refusal, and the
  declaration-time `entrants=True`/`mass="conserve"` conflict.
- C5, `test_c5_tolerance_is_declared`, specifies an exact tolerance in the
  producing receipt, coordinate-keyed propagation to a reading gate, `None`
  for bitwise outputs, and both declaration-time capability refusals.
- D6, `test_d6_mass_is_partitioned`, specifies partition-local conserve
  refusal, a partitioned free-policy mass ledger with independently calculated
  expectations, and compile-time refusal when CREATE omits the partition
  column.

The tests and toy kernels use public graph declarations, `run_graph` through
the toy harness, manifests, receipts, and public errors. No runtime module was
edited.

The marker lines use the flip tool's exact one-line form:

```text
@pytest.mark.xfail(strict=True, reason="charter B6: entrant execution pending")
@pytest.mark.xfail(strict=True, reason="charter C5: tolerance propagation pending")
@pytest.mark.xfail(strict=True, reason="charter D6: partitioned mass pending")
```

## Public shapes fixed for the implementation lane

B6 preserves the existing entity-keyed lineage receipt. Each entry remains a
`(new_id, source_id)` pair; an entrant is distinguished by a null source. The
positive fixture admits a household entrant and carries every household column
through the existing `materialized_expand_outputs` mechanism. Its new person
is a copy assigned to that household, avoiding an invented entrant-stratum
channel that is absent from the frozen `KernelResult` interface.

C5 expects this JSON-safe capability representation in the producer receipt
and in gate evidence:

```json
{"rtol": 0.000001, "atol": 0.0, "ulps": 0}
```

The gate obtains it from
`context.tolerances[("person", "bounded_value")]`; a bitwise owner's coordinate
maps to `None`.

D6 retains the existing flat mass fields and adds a backward-compatible
`receipt["mass"]["partition"]` block:

```text
entity: "person"
column: "period"
stratum_before: {"2024": {<stratum>: <mass>}, "2025": {...}}
stratum_after:  {"2024": {<stratum>: <mass>}, "2025": {...}}
```

Partition keys are the JSON-string form of the partition value; the next level
is stratum. The test independently computes each expected cell from the source
CSV, household design weights, and person-to-household relation.

## Honest RED evidence

Each test was rerun with `--runxfail`, which disables its marker without
editing the test. All three fail at the intended absent feature.

### B6

```text
FAILED packages/microcosm-graph/tests/test_acceptance_b_ownership.py::test_b6_entrants_are_declared
E microcosm.graph.errors.NodeRejectedError: EXPAND node 'scheduled_entries' returned malformed lineage: EXPAND lineage id <NA> is not a JSON scalar.
1 failed in 0.44s
```

This is the current executor's refusal of the amendment's null-lineage entrant
sentinel, before the positive path can run.

### C5

```text
FAILED packages/microcosm-graph/tests/test_acceptance_c_seeds.py::test_c5_tolerance_is_declared
E KeyError: 'tolerance'
1 failed in 0.41s
```

The tolerant producer runs, but its capabilities receipt does not yet record
the declared bound.

### D6

```text
FAILED packages/microcosm-graph/tests/test_acceptance_d_weights.py::test_d6_mass_is_partitioned
E assert '2024' in "Node 'period_conserve' patch rejected: Node 'period_conserve' mass='conserve' changed stratum 'rural': 206741.316 -> 164718.2845."
1 failed in 0.42s
```

The current conserve check sees the global stratum change but does not yet
identify or enforce the amended partition value.

## Verification

All commands ran offline. The locked environment was synced with the warm
local artifacts and the existing workspace virtual environment.

Requested full graph command:

```text
uv run pytest packages/microcosm-graph -q -p no:warnings
...x...x...x... [100%]
exit 0
```

Because the repository already supplies `-q` in `addopts`, the requested
second `-q` suppresses pytest's count line. A supplemental equivalent run with
only one quiet flag reported:

```text
203 passed, 3 xfailed in 86.14s (0:01:26)
```

Other requested checks:

```text
uv run ruff check .
All checks passed!

uv run ruff format --check <six touched Python files>
6 files already formatted

uv run python tools/ci_test_groups.py --verify
tracked_test_files=353
verification=ok

git diff --check
exit 0
```

## Burndown ratchet incompatibility

The requested command does not pass, for a direct conflict between the new
charter-marker requirement and the current baseline algorithm:

```text
uv run python tools/graph_acceptance_burndown.py --verify --baseline origin/main

group   markers  properties
B             1  B6
C             1  C5
D             1  D6
total         3

green     37  A1 A2 A3 A4 A5 A6 A7 B1 B2 B3 B4 B5 C1 C2 C3 C4 D1 D2 D3 D4 D5 E1 E2 E3 E4 E5 F1 F2 F3 F4 F5 G1 G2 G3 H1 H2 H3
red        3  B6 C5 D6
missing    0  none

baseline=origin/main
  rose   packages/microcosm-graph/tests/test_acceptance_b_ownership.py: 0 -> 1
  rose   packages/microcosm-graph/tests/test_acceptance_c_seeds.py: 0 -> 1
  rose   packages/microcosm-graph/tests/test_acceptance_d_weights.py: 0 -> 1

error: packages/microcosm-graph/tests/test_acceptance_b_ownership.py re-reds 1 property (0 -> 1)
error: packages/microcosm-graph/tests/test_acceptance_c_seeds.py re-reds 1 property (0 -> 1)
error: packages/microcosm-graph/tests/test_acceptance_d_weights.py re-reds 1 property (0 -> 1)
verification=failed
```

The inventory half is satisfied: all amended properties are found and
`missing 0`. The ratchet then unconditionally rejects any file whose marker
count exceeds `origin/main`; that baseline has zero markers in all three target
files. Thus the three required markers are exactly the three rejected `0 -> 1`
increases. Making this command green would require changing the tool/baseline
or omitting/obscuring a required marker. Both are outside this tests-only lane
and the latter would violate the charter and flip-tool contract. The dispatcher
or ratchet owner needs to add the explicit new-charter allowance or choose an
amended baseline before rerunning this command.

## Commits

```text
d581cc66 Start graph amendments acceptance journal
e3921cad Add red B6 entrant acceptance contract
5a63745a Add red C5 tolerance acceptance contract
64c7c0d1 Add red D6 partitioned mass acceptance contract
721c5d4a Track three amended charter properties as red
```

No network access, push, runtime edit, or destructive repository operation was
performed.
