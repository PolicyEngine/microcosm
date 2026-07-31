# PR #583 Guard Completion Progress

## State

Round-4 hold remediation is in progress on `multispine-pool-build-578`, from
clean starting commit `19a7a1d`. Subscript resolution now catches all six
round-4 evasions; loop and comprehension targets now propagate every static
string choice or an opaque shadow; and the 54-module operator graph remains
clean. `str.format` now resolves all requested static field forms precisely.
Direct and aliased strict pandas methods now share the same checks, and dynamic
DataFrame `getattr` is fail-closed. Closure free variables now obey lexical
late-binding assignment counts. The five requested subsystems are implemented;
the cross-round invariant, final docstrings, and full validation remain.

No push or external mutation is authorized.

## Done

- Read `CLAUDE.md` and the round-4 review log.
- Confirmed the required clean starting HEAD and branch.
- Confirmed the local GitNexus index is absent; direct AST/source tracing will
  be used unless a safe local index becomes available.
- Completed parallel read-only audits of reviewer repros, resolver/loop
  behavior, and strict call-site/closure behavior.
- Extended the static resolver for walrus expressions, string multiplication,
  percent formatting, and all-static chained `str.replace`.
- Made inferred column-container subscripts resolve every static string member
  or record explicit opacity; subscript assignment targets are visited too.
- Added exact round-4 binding tests: walrus binds and reports both reads;
  multiplication, percent formatting, and replace report the guarded name;
  nested calls and dict indirection report fail-closed.
- Replayed the guard file (15 passed), focused Ruff, and the exact 54-module
  graph (no missing modules or offenders).
- Bound `for`/`async for` targets and list, set, dict, and generator
  comprehension targets before their bodies are visited.
- Represented statically resolvable iterables as tuples of every possible
  string, checking every member at a subscript; dynamic iterables explicitly
  shadow stale outer bindings with opacity.
- Added benign, mixed guarded, module-bound, dynamic, and comprehension-wrapped
  selector regressions. The guard file now passes 17 tests with the graph clean.
- Replaced bare `{}` substitution with `string.Formatter` field parsing for
  automatic, indexed, named, converted (`!s`/`!r`), and specified fields.
- Kept unresolved fields structurally opaque while preserving literal `*` in a
  fully static pandas expression as benign syntax.
- Bound exact-name checks for every unsafe format variant and zero-finding
  controls for named benign formatting, conversions/specs, escaped braces, and
  multiplication syntax. The guard file now passes 19 tests with the graph
  clean.
- Tracked scoped aliases of `query`, `eval`, `filter`, and `get`, including
  alias chains, explicit rebinding shadows, and static `getattr` aliases.
- Routed direct and aliased calls through shared strict handlers, including
  hidden/expanded argument and opaque-key failures.
- Made dynamic `getattr` fail closed for inferred DataFrame/Frame containers
  while distinguishing generic object and Series index access.
- Added direct-vs-alias exact-name tests for all four methods, benign and opaque
  batteries, rebinding/parameter-shadow controls, and static/dynamic `getattr`
  probes. The guard file now passes 22 tests with the 54-module graph clean.
- Added a lexical scope pre-pass that counts binding sites without descending
  nested function/class bodies and pre-shadows all Python locals.
- Deferred nested function-body analysis until the enclosing scope's bindings
  are complete; free names with exactly one defining-scope assignment resolve,
  while multi-assignment free names become explicit opacity.
- Preserved the same late-binding rule for lambdas and strict method aliases,
  so a rebound alias cannot degrade to an ignored ordinary Name call.
- Added the exact reviewer late-bound closure, module-level and lambda variants,
  stable guarded/benign controls, later-local shadowing, and stable/rebound
  alias tests. The guard file now passes 24 tests with the graph clean.

## Next

- Commit closure late-binding handling.
- Add the parametrized cross-round completeness invariant, expanded benign
  battery, exact 54-module count assertion, and completed-contract docstrings.
- Add the cross-round completeness invariant and benign battery.
- Run the guard file, the full `populace-build` suite, Ruff, and the 54-module
  graph scan.
- Write `/private/tmp/583_fix2_handoff.md` with per-subsystem rules, binding
  tests, any operator rewrite rationale, exact validation results, and commit
  inventory.
