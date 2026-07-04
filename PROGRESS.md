# PROGRESS — validation-inputs-produced-gate (issue #272)

## Goal
Add a build gate asserting every input variable referenced by validation configs
is actually produced by the dataset. Catches the "structurally missing input"
defect class (#252 auto-loan $0, #253 tuition, #254 AMT).

## Acceptance criteria (from #272)
- Walk every variable referenced by validation configs (obbba_reforms.json,
  tax_expenditure_reforms.json, soi_baseline_levels.json) + reform-validation
  measure/cap variables.
- Resolve against the PE-US variable graph; assert each pure-input leaf it
  depends on is declared as a source-stage output (source_stages.json outputs /
  produced-column inventory).
- TDD: failing test proves the gate CAN fail (config referencing never-produced
  variable) + a pass case.
- Match GateReport idiom (gates.py).

## Constraints
- CI's `uv sync --all-packages` does NOT install `[us]` extra → importorskip
  policyengine_us.
- No changelog system (no towncrier).
- `set -o pipefail` before piping pytest.
- PR #308 (weights-audit) concurrently edits gates.py → keep addition
  append-only + self-contained.

## Status
- [x] Worktrees created
- [ ] Explore gates.py GateReport idiom
- [ ] Explore validation configs + source_stages.json
- [ ] Write failing test
- [ ] Implement gate
- [ ] Wire into build
- [ ] Full pytest + ruff green
- [ ] PR
