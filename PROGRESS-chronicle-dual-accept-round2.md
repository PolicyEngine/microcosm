# Round-2 review fixes — PR #849 (`chronicle-dual-accept`)

Journal for the gate-findings pass on top of `1f78847c`. History, not state:
check git/GitHub for current truth.

## Verified ground truth (read this session, 2026-09-02)

Read from `PolicyEngine/chronicle` at `origin/main`:

| Identity | Where | Epoch |
|---|---|---|
| `policyengine_ledger.consumer_artifact.v2` | `policyengine_chronicle/consumer.py:30`, README line 299 | ledger |
| `ledger.consumer_fact.v1` | `chronicle/consumer_contract.py:28` | ledger |
| `ledger.fact.v1` | `chronicle/core.py:103` | ledger |
| `ledger.source_cell.v1` | `chronicle/sources/cells.py:24` | ledger |
| `ledger.source_row.v1`, `ledger.source_column.v1`, `ledger.source_row_value.v1` | `chronicle/sources/rows.py:27-29` | ledger |
| `ledger.{aggregate_fact,semantic_fact,concept_alignment,dimension_set,observed_measure,source_release,source_series,universe_constraint_set}.v2` | `chronicle/consumer_contract.py`, `policyengine_chronicle/consumer.py:195` | ledger |

`policyengine_ledger.consumer_artifact.v1` is what **Microcosm** mints
(`import_entry_facts.py:164`), never what Chronicle emits — it is ledger-era
history that must keep loading.

There is **no** `origin/epoch-dual-domain` branch on chronicle
(`git ls-remote --heads origin | grep -i epoch` → empty), so the chronicle-era
successors are declared here by the migration rule chronicle#143 states —
new namespace, same family, version bumped by one — and the brief pins the two
that matter: `policyengine_chronicle.consumer_artifact.v3` and
`chronicle.consumer_fact.v2`.

## State

Round 2 complete and pushed to `chronicle-dual-accept`; the PR body carries a
"Review fixes (round 2)" section with the same content as this journal's
verification block. **DO NOT MERGE** — the lane was authorized to fix and
push only.

## Verification (2026-09-02)

Both full `packages/microcosm-build` suites — clean shell and the
polluted-Logbook shell finding 2 specifies — returned **outcome-identical**
results: 7116 passed, 47 skipped, 2 failed, out of 7165.

The 2 failures are `test_release_target_parity.py::TestRegeneration` and are
**not this branch's**. They reproduce identically at `1f78847c` (this
branch's previous head) and at `origin/main` `47c74225`, both verified in a
throwaway worktree this session:

    ValueError: Ledger target reference
    'cbo.revenue_projection.ty2024.income_by_source.adjusted_gross_income.projected_amount'
    assertion_policy='observed_only' does not allow resolved fact assertion
    'source_projection'.

They run only where the pinned feed exists —
`~/PolicyEngine/_buildh-runtime/inputs/consumer_facts_buildn_v9_4.jsonl`,
outside the repo — and skip everywhere else, PR CI included. Worth a separate
issue: on a machine holding that feed, the committed
`cbo.revenue_projection.*` reference and the feed row it resolves to disagree
about assertion class.

`uv run ruff format --check .` is red repo-wide (116 files) at this branch
*and* at its merge base: ruff 0.15.16's formatter against a repo pinned to
`ruff>=0.8`, whose CI lint lane runs `ruff check` only. Every file this
branch touches is format-clean before and after.

## What changed structurally, for the next reader

The first pass treated epoch as something you could *read off* a key's
namespace. That is why it accepted a two-id set that excluded the id
Chronicle actually emits, and why `chronicle.anything.vN` was witnessed as
Chronicle-issued identity. Round 2 replaces it with a declared registry:
`DECLARED_IDENTITIES` in `chronicle_epoch.py`, keyed by
`(namespace, family, version)`. Anything not in that table is `undeclared`
and is reported as such. If Chronicle publishes its own successor
enumeration, that table is the one place that changes.

## Done

- [x] (1) Declared identity registry keyed by `(namespace, family, version)`
- [x] (2) Autouse Logbook env isolation + fail-closed network in tests
- [x] (3) UK `_ledger_provenance` delegates to `provenance()`
- [x] (4) Undeclared chronicle-namespace identities labelled `undeclared`
- [x] (5) `microcosm.build.logbook_env` module import no longer shadowed
- [x] (6) `CHRONICLE_US_SOURCE_COVERAGE_CONTRACT_COMMIT` in the us_runtime barrel
- [x] (7) Non-string `schema_version` raises the documented ValueError
- [x] (8) `tools/logbook.py --remote` help prefers `LOGBOOK_*`

## Next

Nothing in this lane. Reviewer decides on merge.
