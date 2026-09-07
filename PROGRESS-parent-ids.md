# PROGRESS — parent-ids-export (microcosm#884)

## State
Worktree `~/PolicyEngine/_worktrees/microcosm-parent-ids`, branch `parent-ids-export`
from `origin/main` @ 5ab1b056f.

Goal: export CPS `PEPAR1`/`PEPAR2` as person-level `parent_1_id` / `parent_2_id`
(person_id of the co-resident parent, 0 when absent/unresolved) from the
`eligibility_inputs` stage.

## Done
- Read [microcosm#884](https://github.com/PolicyEngine/microcosm/issues/884) and
  [policyengine-us#9404](https://github.com/PolicyEngine/policyengine-us/issues/9404).
- Created worktree + branch.

## Next
- Map every registration site for an `eligibility_inputs` output column.
- Implement resolution in `eligibility_inputs.py`.
- Register columns; regenerate `source_stages.json` via the repo mechanism.
- ACS-transferred path (`acs_pums.py`) check.
- Invariant in the stage gate + unit tests.
- ruff, pytest, commit, draft PR.
