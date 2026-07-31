# PR #583 Fix-3 Guard Completion Progress

## State

Fix-3 completion is in progress on `multispine-pool-build-578`, from clean
starting commit `b219b21`. The populace owner resolved the typed-DataFrame
decision: parameterized `df[column]` is a permitted boundary, while every
statically resolvable guarded column name is contraband in every expression
position in a non-owner operator module. The three-layer contract, composition
argument, sole runtime-data residual, and annotation/docstring exemptions are
now documented and executable. All requested binding tests pass, and both the
all-runtime scan and pinned 54-module graph are clean after three narrow
adjudications. Final guard/package/Ruff certification is green. The fix-3
handoff and required root-journal restoration remain.

No push or external mutation is authorized.

## Done

- Read `CLAUDE.md` and `/private/tmp/583_fix2_handoff.md`.
- Confirmed the required branch, clean worktree, and starting HEAD `b219b21`.
- Recorded the owner's explicit typed-parameter and contraband-literal decision.
- Added the all-expression contraband rule on top of the existing flow-sensitive
  string resolver, including static f-string alternatives from enumerated
  comprehension choices.
- Exempted only true module/class/function docstrings and annotation syntax;
  executable defaults, including lambda defaults, remain checked.
- Added binding tests for the typed parameter boundary, guarded and benign
  helper calls, list/tuple/set/dict positions, comparisons, returns,
  assignments, defaults, every requested composition form, static f-string
  enumeration, owners, annotations, and docstrings.
- Classified `operator_boundary.py` as a reviewed provenance owner because it
  enumerates provenance columns only to reject preassembled source frames.
- Moved the shared `person_support_channel` spelling into the existing
  `support_provenance.py` owner and imported it into `adult_care.py` and
  `ssi_take_up.py`, leaving both treatment modules fully guarded.
- Ran the guard file: 94 passed, with the all-runtime and exact 54-module graph
  scans clean. Focused Ruff and `git diff --check` pass.
- Closed post-implementation adversarial findings: function defaults and
  decorators now resolve at definition time while bodies retain late-binding
  analysis; class namespaces no longer corrupt enclosing constants; PEP 695
  class type parameters remain annotation-exempt; and attribute assignment,
  annotated-assignment, loop, and comprehension targets are visited.
- Documented the full non-strict receiver composition boundary for typed,
  constructed-DataFrame, and `Frame.table(...)` result subscripts, with no
  interprocedural proof and the same sole runtime-data residual.
- Added binding regressions for definition-time constant state, class namespace
  isolation, PEP 695 annotations, every attribute-store form, and the broader
  truthful subscript boundary.
- Preserved real nested lambda/comprehension closures while excluding class
  namespaces, and distinguished abstract static iteration choices so composed
  fragments resolve across those closures without treating ordinary tuple
  formatting as alternative values.
- The expanded guard file passes 99 tests; the all-runtime and exact 54-module
  scans remain clean.
- Completed independent adversarial review with a final clean verdict after
  binding every reported definition-time, class-scope, annotation, store,
  closure, and static-format precision repro.
- Ran the final guard file through `uv`: 99 passed.
- Ran the final full `populace-build` suite: 3,321 passed, 85 skipped, 5 known
  warnings in 107.66 seconds.
- Ran repository-wide `ruff check .`: passed.
- Ran focused `ruff format --check` on all four changed code/test files:
  passed. `git diff --check` and worktree status are clean.
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
- Generalized static string iteration from list/tuple syntax to sets, dict
  keys, literal strings, and statically concatenated collections while
  preserving literal containers for format-field indexing.
- Added benign and guarded loop/comprehension controls for every new iterable
  form. The guard file now passes 72 tests with the graph clean.
- Added branch-state joins for `if` statements/expressions and zero-or-more
  iteration joins for dynamic loops; divergent constants become opaque,
  possible DataFrame provenance is retained, and conditional method aliases
  cannot disappear.
- Preserved exact pre-loop bindings across provably empty static iterables.
- Added guarded, benign, conditional-alias, optional-loop, and empty-loop
  binding tests. The guard file now passes 73 tests with the graph clean.
- Ran the full `populace-build` suite: 3,295 passed, 85 skipped, 5 warnings.
- Ran repository-wide `ruff check .`: passed.
- Ran repository-wide `ruff format --check .`: found the existing 44-file
  formatting baseline; formatted only this guard file and replayed its 73 tests.
- Wrote `/private/tmp/583_fix2_handoff.md` with per-subsystem rules and binding
  tests, the no-rewrite rationale, exact validation receipts, commit inventory,
  and the remaining typed-container blocker.

## Next

- Write `/private/tmp/583_fix3_handoff.md`, then restore `PROGRESS.md` to
  `origin/main` and commit that restoration so root journals stay out of the PR.
