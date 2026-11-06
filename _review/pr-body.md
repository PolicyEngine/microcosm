The US input-mass gate compares `partnership_income` alone with its reference, and a reference built before #1261 holds all of Schedule E line 32 in that column. A build that keeps the line can still fail the gate once the split leaves less than half of it there. `--allow-input-mass-drift` gets such a build through by turning the gate off for every column.

For example, this gate failed the release attempt at 9434c504b against `populace_us_2024.h5`, which holds $847.756B in `partnership_income` and $0 in `s_corp_income`. The export held $391.947B in `partnership_income`, 53.8% short of the reference, and $664.150B in `s_corp_income`, so the two columns together exceeded it by 24.6%. Calibration brought SOI's combined partnership and S corporation line to $1,147.2B against its $1,157.6B target.

## What changes

`us_line_32_on_reference_terms` counts both columns as the candidate's `partnership_income` when the reference has no `s_corp_income` mass above the gate's floor. The release's base and export gates, its dry-run and preflight checks and the L0 refit export all call it, and `worst_drifts.partnership_income` reports the combined drift on such a reference. Once a reference carries the split, the gate compares each column alone again, with no register entry to lift.

This changes which builds pass the gate and moves no target or weight. A build frozen at 9434c504b keeps the old comparison.

## Invariants

Hypothesis tests in `test_us_input_mass_line_32.py` check these across inputs:

- Only `partnership_income` moves, and only against a reference with no `s_corp_income` mass above the floor, and the helper leaves its input unchanged.
- Against a pre-split reference, moving income between the two columns never changes the gate's verdict.
- The dry-run preview and the gate agree, and the preview raises if they don't.

The gate still fails a build that loses line 32 mass, a candidate with no `partnership_income`, and, against a reference with the split, a column that falls short alone. Example tests cover each refusal. Removing the rule from any of its five callers fails a test, and so does each of five mutations of the rule.

## Status

WIP. An independent reviewer reads this next, and the PR leaves draft once they agree at the final head.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
