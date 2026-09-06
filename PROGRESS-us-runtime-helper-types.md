# US runtime helper types — progress

**Lane:** explicit source-helper parameter types on the repo's existing typed-helper
boundary. Base: exact `cd971ff9b7140e9dfa54c82f4b59325b69eb6f8a` (classification-only).
Branch: `us-runtime-helper-types-20260905`.

**Not this lane:** the `442dc3e5` selector-shape scanner widening. The scanner
predicate stays byte-identical to `cd971ff9`.

## State
- [x] Read `us-runtime-classification-completion-REPORT.md` and
      `us-spine-scanner-precision-REPORT.md`
- [ ] Read the base scanner predicate and the documented typed boundary
- [ ] Inspect every target helper body, runtime checks and ALL local callers
- [ ] Annotate only demonstrated concrete roles
- [ ] AST identity after stripping annotations + new type imports
- [ ] Full 498-case scanner suite + focused per-module suites
- [ ] CI group verification, lint/format
- [ ] Report

## Done
- Fresh child worktree created at `cd971ff9`, own branch, clean.
- Root I (`_worktrees/microcosm-us-launch-current-main-20260905`, `9ece4cad`) verified
  clean and left untouched.
- Experimental child (`_worktrees/us-spine-scanner-precision-cd97`, `442dc3e5`) verified
  clean and left untouched.

## Next
Read `_source_spine_accesses` at `cd971ff9` and
`test_typed_parameter_subscript_is_closed_at_static_call_sites`.
