# Parallel executor lane: blocked attribution preflight

2026-09-19. Branch `graph-parallel-executor`, requested base
`16c8e78d2f60d629da5d70294f643bce5c4597e2`. This is an incomplete implementation
report with completed read-only findings and baseline checks. The requested
memoization and concurrent executor have **not** been implemented.

## Blocking finding

The requested main tree does not contain the 19-node harness's US runtime.
`graph_atomic_survey_financial`, `graph_atomic_survey_population`,
`survey_population_preparation`, `survey_atomic_geography`, and the ACS
coverage modules are absent from the base git tree, current checkout, and
local-venv import resolution. [preflight.json](preflight.json) records those
independent checks; [attribution-preflight.md](attribution-preflight.md)
contains the fuller inventory and source references.

The native modules exist in the supplied transport checkout, read without
modification, at `25d66f3c5`. Importing them would measure a mixed tree. The
user's required order is attribution on unchanged main, then measured design,
then implementation. That sequence cannot proceed until the measurement base
is resolved. No measurement was launched, so there is no measurement PID,
CPU-limit receipt, timed refusal, or recovered exception to report. The
observed memory availability (79.0 GB) was not the blocking condition.

A second constraint remains even with a native overlay: much of the reported
source authentication and reconstruction runs before `run_graph` or the
content store is called. Executor/store memoization alone cannot skip it.
The owner currently reconstructs and validates before comparing candidate
receipt bytes, and its issued preparation capsule requires live owner
identity and weakref registration. A reusable receipt needs a consumption
boundary in the owning runtime; replacing the pinned functions is outside
this lane's permitted scope.

## Attribution table

N/A means unavailable, never zero. These are mechanism-based classifications
from reading the native transport code; they are not fresh measured costs.
The detailed audit links every row to its source and distinguishes potential
reuse from reuse implementable inside this lane.

| Bucket | CPU-s cold | CPU-s replay | Classification |
| --- | ---: | ---: | --- |
| ACS archive inventory, decompression, CSV validation | N/A | N/A | Potentially memoizable across runs, bound to complete authenticated archive and parser contract; selected projection also binds roster. |
| Repeated ASEC capture/CSV fencing | N/A | N/A | Parsing potentially memoizable across nodes and runs; current path custody and byte authentication remain necessary. |
| ASEC HDF5 decoding/reconstruction | N/A | N/A | Potentially memoizable across runs after complete input authentication; current owner exposes no skip receipt. |
| Borrowing issued preparation/native capsules | N/A | N/A | Already memoized within the native verification epoch, with cheap live checks and final full validation; absent from main. |
| Live frame seals/mutation checks | N/A | N/A | Irreducible at current validation boundaries; changed node populations have different outputs. |
| Selection, normalization, origins/context | N/A | N/A | Potentially memoizable across runs for identical complete inputs/configuration; not a portable authority today. |
| Predictor/geography/population reconstruction | N/A | N/A | Distinct node outputs are irreducible; identical complete derivations could be reused only through an owner-supported boundary. |
| Store integrity validation/decoding | N/A | N/A | Derived decoding may be memoized after fresh byte verification; per-access corruption checks cannot be replaced with stat-only hits under today's refusal contract. |

The transport baseline's 333.29 seconds in both node loops and 1,712.62
seconds outside them are **wall durations**; its runner CPU total is
2,026.775 seconds. The after-run's 0.25-second sampler is cumulative over cold
and replay and truncated to 60 chains. It cannot provide separate phase CPU
buckets. Neither halving it nor subtracting a baseline at another head is
valid attribution. Historical values and this limitation are preserved in
[the audit](attribution-preflight.md#historical-measurements-what-they-do-and-do-not-establish).

**What #938 removed:** graph context digest scalar loops for supported
numeric/boolean columns and repeated per-node full reads of unchanged declared
sources within a run. It retains full source authentication at run start/end
and failure settlement. It also carries caller epoch statistics into the
manifest. It did not merge the native US epoch wrappers (#935), remove native
pre-executor reconstruction, or add persistent owner-accepted source receipts.

## Design answers established by reading main

The full provisional [design note](../../docs/graph-parallel-executor.md)
contains line references for every current guarantee. Its conclusions are:

- All node keys and typed contracts are computed before execution; required
  replay preflight still must precede any worker or callback.
- Readiness requires every dependency's admitted receipt. `_blocked_by`
  propagates unavailable products; a failed gate with an available verdict
  remains usable evidence.
- Populations live in an in-memory dictionary and ordinary nodes cumulatively
  patch their shared version. Typed artifact payloads come from the store.
- Concurrent whole-node loops would lose sibling columns. Results need
  canonical coordinator admission against the latest population. Column
  insertion and mass-log order affect stored bytes, so arbitrary topological
  application order is not byte-commutative today.
- Workers may compute in varied orders; coordinator application, callbacks,
  store writes, and receipt admission must retain `compiled.order`.
- Node products precede the node cache record; a consumer waits for the full
  producer receipt. Content keys, rather than directory insertion order,
  identify stored objects.
- Proposed failure policy: first observed hard failure stops scheduling;
  active workers finish and are joined inside the ledger scope; their
  unadmitted results are discarded. Workers never publish. Settlement then
  sees exactly all actual coordinator publications and no late writes.
  A failed concurrent run can have a shorter successful prefix than a failed
  sequential run; ledger completeness, not equal failed-prefix size, is the
  invariant. Stable sources preserve resume work; changed/unverified sources
  evict that run's writes.
- Both observer modes remain serialized before persistence. Detached mode
  preserves amendment 24 isolation; live mode retains amendment 25's caller
  promise not to retain or mutate the population.
- Threads versus processes is deliberately undecided until measurement.
  Standard process pickling does not directly transport mapping proxies and
  arbitrary registered kernels/closures. Native population/context sizes and
  serialization costs have not been measured.
- A worker count alone is insufficient. A memory reservation budget must
  include live populations, contexts, artifacts, queued results, worker
  transport, application temporaries and observer snapshots. Arbitrary kernel
  allocations and caller-retained observations limit any hard RSS guarantee.
  Numeric reservation coefficients remain unmeasured.

The proposed byte-identity argument is induction over canonical admissions:
unchanged keys and admitted dependencies imply identical projected inputs and
node RNG seeds; conforming kernels then yield identical results; unchanged
validation/application/serialization yield identical receipts and objects.
This is a design argument, **not an implemented or tested concurrency proof**.

## Moved pins and identity evidence

| Pin surface | Old | New | Generator |
| --- | --- | --- | --- |
| Producer/module pins | Unchanged base values | Unchanged | Not invoked; no producer edits |
| Interface lock (`decl.py`, `kernel.py`) | Unchanged base values | Unchanged | Not invoked; neither file edited |
| Dependency lock | Committed `uv.lock` | Identical | Locked sync; no relock |

No memoization buckets have been implemented, so no second-access store-hit
claim is made.

| Workers | Manifest key versus sequential | Every content receipt | Every store object |
| ---: | --- | --- | --- |
| 1, unchanged graph baseline | Existing suite passed; no new native manifest | Existing suite passed | Existing suite passed |
| 2 | Not implemented/tested | Not implemented/tested | Not implemented/tested |
| 4 | Not implemented/tested | Not implemented/tested | Not implemented/tested |

Receipt comparison will exclude existing operational `hit`/`wall_time` fields;
full manifest timestamps/host are likewise not content identity. Runtime
amendment 26, seeded scheduling sweeps and concurrent failure tests remain
outstanding and are not represented as adopted or passed.

## Measurements

| Fraction | Workers | Wall | CPU | Peak RSS | Loop/outside split | Required replay identity |
| --- | ---: | --- | --- | --- | --- | --- |
| 1/1000 | 1 | N/A | N/A | N/A | N/A | Missing native runner |
| 1/1000 | 2 | N/A | N/A | N/A | N/A | Runner absent; concurrency not implemented |
| 1/1000 | 4 | N/A | N/A | N/A | N/A | Runner absent; concurrency not implemented |
| 1/15 | 1 | N/A | N/A | N/A | N/A | Missing native runner |
| 1/15 | 2 | N/A | N/A | N/A | N/A | Runner absent; concurrency not implemented |

No dataset build, calibration, publication, gated-data transfer, other-worktree
edit, or background test job occurred. All started checks were waited for in
the foreground. No process was killed.

## Validation — verbatim summaries

Initial online sync failed on PyPI DNS. Cached wheel archives were copied
into this worktree, preserving the originals, and this succeeded:

```text
$ UV_CACHE_DIR="$PWD/.uv-cache" UV_LINK_MODE=copy uv sync --offline --all-packages --locked --extra us
Installed 102 packages in 8.53s
```

```text
$ UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync pytest packages/microcosm-graph/tests -p no:cacheprovider
632 passed, 1 skipped in 72.27s (0:01:12)

$ UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python tools/ci_test_groups.py --verify
verification=ok

$ UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python tools/spec_engine_coverage.py --check
spec-engine coverage: 42156/42156 configuration fields; 41/41 inventory checks

$ UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python tools/graph_acceptance_burndown.py --verify
verification=ok
```

Full outputs: [graph tests](graph-baseline-tests.log),
[CI grouping](ci-groups.log), [spec coverage](spec-coverage.log),
[acceptance burndown](acceptance-burndown.log). The spec log includes an
upstream IPython `SyntaxWarning`; its check exited successfully.

No Python files were touched, so there is no touched-file target for
`ruff check` or `ruff format --check`; neither was run against the package.
The graph suite is the unchanged baseline, not validation of a concurrency
patch. No full engine CI-group run is claimed.

## PR and committed state

PR URL: unavailable. A draft WIP PR body is prepared in
[pr-body.md](pr-body.md), ending with the requested attribution footer. The
branch push failed before a PR could be created:

```text
$ git push -u origin graph-parallel-executor
fatal: unable to access 'https://github.com/PolicyEngine/microcosm.git/': Could not resolve host: github.com
```

After network access returns, the prepared command is:

```sh
gh pr create --draft --base main --head graph-parallel-executor --title "WIP: audit parallel executor prerequisites on main" --body-file experiments/graph-parallel-executor/pr-body.md
```

That dependent PR command was not run after the push failed. No PR was marked
ready or merged. The implementation is pending the measurement-base decision;
these source-audit results are committed as work in progress, not presented
as a completed executor change. Initial journal: `f00743c9a`; source audit and
baseline evidence: `9cc1850fb`; report and completed-check outputs: `96d61789d`.
A subsequent WIP commit records this publication failure and clean handoff.

## Questions for Max

1. **Which measurement base should this lane use?**
   - **A (recommended):** Keep implementation/PR against main; authorize a
     separately recorded native-runtime overlay for measurements, with main's
     graph code taking precedence. Results would explicitly be mixed-tree
     evidence, with both revisions and module origins recorded.
   - **B:** Rebase the lane onto the native-build branch containing the runtime;
     revise the requested main-only PR boundary accordingly.
   - **C:** Keep main strict and proceed using synthetic executor evidence;
     defer native attribution and performance claims until the runtime lands.
2. **How should the pre-executor memoization boundary be handled?**
   - **A (recommended):** Have the native owner lane expose a versioned,
     authenticated reusable receipt boundary; this lane consumes it from
     executor/store without modifying pinned producer modules.
   - **B:** Keep that work deferred and complete only the main executor's
     concurrency work, explicitly without claiming to remove the reported 84%.

The first question was also sent through the text-input prompt. No answer
had arrived when this WIP report was written. No elapsed timeout has been
interpreted as permission to change the requested baseline or scope.
