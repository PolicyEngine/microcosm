# US runtime helper types — progress

**Lane:** explicit source-helper parameter types on the repo's existing typed-helper
boundary. Base: exact `cd971ff9b7140e9dfa54c82f4b59325b69eb6f8a` (classification-only).
Branch: `us-runtime-helper-types-20260905`. HEAD `73345a26`.

**Not this lane:** the `442dc3e5` selector-shape scanner widening. The scanner
predicate stays byte-identical to `cd971ff9` (`sha256 42783a42…`, empty diff).

## State — complete
- [x] Read both prior reports and the documented typed boundary
      (`test_typed_parameter_subscript_is_closed_at_static_call_sites`)
- [x] Read the base scanner container propagation (`_visit_function`, `_column_container`,
      `_attribute_container`, `_visit_getattr`, `_visit_column_selector`)
- [x] Inspect every target helper body, its runtime checks and ALL local callers
- [x] Annotate 12 parameters across 7 files; only demonstrated concrete roles
- [x] AST identity after stripping annotations + new type imports: **7/7 identical**
- [x] Original 498-case scanner suite: **498 passed, 0 failed**
- [x] Focused invented suite (10 tests, new file): **10 passed**
- [x] Dependency contract recomputed with the repo API: **0 rows need updating**
- [x] Stage identity movement measured and reported (2 stages move)
- [x] CI group verification `verification=ok`, lint + format clean
- [x] Report written to `R/us-runtime-helper-types-REPORT.md` / `.json`

## Done
- 21 opacity findings across 7 modules → **0**, with the scanner unchanged.
- Adversarial checks: 0 named findings added/removed; contraband injected into every
  annotated helper is still caught by name; reverting an annotation reintroduces the
  findings (the change is load-bearing, not vacuous).
- Root I (`microcosm-us-launch-current-main-20260905`, `9ece4cad`) untouched and clean.
- Classification child (`us-runtime-classification-9ece`, `518e3df3`) untouched and clean.
- Experimental scanner child (`us-spine-scanner-precision-cd97`, `442dc3e5`) untouched
  and clean — preserved, not modified or erased.

## Honest caveats
- The `us_runtime` tree digest **moves** (`b5b1dadd` → `8cb54b7d`): unlike the two prior
  lanes, this one edits production source. All 7 files have new code hashes and two
  stage identities move. See the report's cache/producer identity section.
- No push, no merge, no PR. No genuine data, no network, no engine, no new packages.
