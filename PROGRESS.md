# Progress

## State

PR #583 round-19 guard remediation is implemented and validated on
`multispine-pool-build-578`. Bound and inline identity/partial forms now share
one classification path: identity dictionaries retain mapping semantics,
direct dict iteration yields keys, and other supported structural iteration
results survive one name binding.

## Done

- Confirmed the requested clean base `0bad771` and committed this live journal
  before changing guard behavior.
- Traced all four audit findings to the binder cutoff after
  `_static_string_list` and the context-free identity-`DictComp` row shortcut.
- Made identity dict comprehensions resolve as `_StaticDictEntries`, preserving
  `.items()`/`.values()` across binding and runtime-correct key iteration both
  inline and bound.
- Extended `_bind_name` through the shared iteration resolver without treating
  opaque results as successful bindings or granting mapping APIs to ordinary
  pair-row lists.
- Preserved known leaves and partiality through bound partial identity layers,
  yielding the same named catch plus fail-closed records as the inline form.
- Qualified the literal-coverage contract to non-owner executable dataflow and
  retained the pinned annotation/true-docstring exemptions.
- Added exact reviewer reproductions, bound/inline classification-multiset
  comparisons, direct bound-key coverage, `.items()`/`.values()` coverage,
  structural-row and partial-dict binding coverage, and fragment-free mirrors.
- Kept the fragile per-column partial-row regression and rounds 16-18 focused
  set green: 5 passed.
- Guard file: 132 passed. Benign battery: 21 passed. Focused
  registry/production/graph battery: 23 passed.
- Raw and governed scans are empty for `acs_transfer.py`,
  `capital_gain_details.py`, `housing_inputs.py`,
  `congressional_district_vintage.py`, and
  `congressional_district_vintage_crosswalk.py`.
- Full workspace: 3,927 passed, 132 skipped, with seven known warnings.
  Repository-wide ruff, guard-file format check, and `git diff --check` pass.

## Next

- Commit the coherent round-19 implementation locally.
- Restore `PROGRESS.md` exactly to `origin/main`, validate that final journal
  commit, and write `/private/tmp/583_fix7_handoff.md`.
- Do not push.
