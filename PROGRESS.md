# PR #583 Guard Completion Progress

## State

Round-4 hold remediation is in progress on `multispine-pool-build-578`, from
clean starting commit `19a7a1d`. Subscript resolution now catches all six
round-4 evasions; loop and comprehension targets now propagate every static
string choice or an opaque shadow; and the 54-module operator graph remains
clean. `str.format` now resolves all requested static field forms precisely.
Direct and aliased strict pandas methods now share the same checks, and dynamic
DataFrame `getattr` is fail-closed. Closure free variables now obey lexical
late-binding assignment counts. The five requested subsystems, consolidated
rounds 2-4 completeness invariant, benign battery, exact graph cardinality,
and completed-contract docstrings are implemented. Adversarial audit and full
validation remain. Adversarial alias composition probes are now closed.
Starred and nested format-field composition is now closed too, as are
late-bound writes through comprehensions, `nonlocal`, and `global`.

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
- Replaced the guard module and scan-test docstrings with the completed
  resolve-and-check-or-fail-closed contract and an explicit surface inventory.
- Added one parametrized invariant covering bound names, parameter f-strings,
  expanded kwargs, shadows, conditionals, mutation, all requested format
  variants, all six round-4 subscript evasions, all four method aliases,
  dynamic `getattr`, and late closure binding.
- Added a parametrized benign battery spanning every corresponding static
  surface and typed non-column controls.
- Pinned the multispine tool graph to exactly 54 runtime modules.
- Replayed the expanded guard file (70 passed), focused Ruff, and
  `git diff --check`; the exact 54-module graph remains clean.
- Preserved strict method identity through walrus expressions and structural
  tuple/list bindings; a previously strict alias rebound to an unresolved
  callable now becomes an explicit opaque alias rather than disappearing.
- Tracked aliases of builtin `getattr` and expanded literal starred arguments;
  unresolved starred calls fail closed.
- Added guarded-name, benign, rebound, unpacked, aliased-`getattr`, and starred
  `getattr` binding tests. The guard file now passes 71 tests with the graph
  clean.
- Expanded statically known `*args`/`**kwargs` before assigning format field
  positions, while any unresolved expansion that can supply a referenced field
  becomes opaque.
- Recursively resolved nested format specifications and static collection
  indexing, and resolved static mapping operands for percent formatting.
- Added exact-name and benign controls for nested specs, indexed mappings,
  static positional/keyword expansion, and percent mappings, plus the
  adversarial star-index-shift failure. The guard remains 71 tests green with
  the graph clean.
- Counted assignment-expression targets inside comprehensions in their actual
  containing Python scope and routed their runtime bindings past the synthetic
  comprehension scope.
- Counted nested `nonlocal`/`global` stores against the lexical scope they can
  rebind, while excluding those declarations from the nested function's local
  binding count.
- Added direct subscript and closure tests for comprehension walruses plus
  sibling `nonlocal` and module `global` writers. The guard remains 71 tests
  green with the graph clean.

## Next

- Commit closure write composition.
- Reconcile the universal column-container contract with the typed runtime
  graph findings from the independent audit.
- Run the guard file, the full `populace-build` suite, Ruff, and the 54-module
  graph scan.
- Write `/private/tmp/583_fix2_handoff.md` with per-subsystem rules, binding
  tests, any operator rewrite rationale, exact validation results, and commit
  inventory.
