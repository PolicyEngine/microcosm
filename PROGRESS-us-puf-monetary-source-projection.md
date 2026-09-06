# PUF monetary source projection — lane journal

Branch `microcosm-us-puf-monetary-source-projection-20260906`, worktree
`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-us-puf-monetary-source-projection-20260906`.
Fresh child of `origin/main` `c1b83241` with the reviewed PUF raw source
lineage `2fd9a5cb` merged (merge commit `2d327989`). Git hooks are disabled for
this repository (`core.hooksPath=/dev/null` in the shared config). Nothing is
pushed, merged or deployed from this lane.

## Scope fences (from the task brief and the admitted plan)

- Authenticate the **full 222 + 7 delivered layout**, not the 6 + 6 typed slice,
  and preserve full lexical source evidence for every delivered column.
- Implement a **small, explicitly enumerated monetary projection** for the first
  PUF fit: wages, interest, dividends, business/farm income, pensions and
  capital gains, only as far as the retained IRS booklets support.
- Every projected column declares exact units, sign policy, missingness policy,
  return grain, source-period status and source-pin identity.
- Source tax outcomes and unresolved fields stay **source-only**. Target-year use
  is refused until an independent growth decision exists.
- No E/P-prefix money inference. No `E22250`/`P22250` alias. No uncapped SE
  derivation from `E30400`/`E30500`. Unmatched demographics are absent, never
  zero, and unmatched rows are never dropped. The four disclosure aggregates are
  never people.
- Reuse the existing immutable graph artifact patterns and the existing
  `puf_growth` contract vocabulary. No new framework.
- **No real PUF bytes are read in this lane.** Invented full-width fixtures only.
- A selected field with no primary semantics is retained and marked unresolved,
  never invented and never a reason to halt the whole projection.

## State — complete and reviewable

- [x] Read the four required review documents and the source-document receipt.
- [x] Fetched the maintained base; frozen and sibling worktrees untouched.
- [x] Fresh worktree created from `origin/main` with `2fd9a5cb` merged.
- [x] Booklet evidence extraction: all three retained texts re-hashed against the
      receipt before use; 25 prose anchors located page-uniquely.
- [x] Full 222 + 7 layout authenticated in a new packaged document, with each
      column's verbatim record-layout row, exact character span and span digest,
      a February 2023 corroboration, and the 13 split-code positions listed.
- [x] The enumerated monetary projection: 12 columns, delivered order, per-column
      units / sign / missingness / grain / period / pins / edit status.
- [x] `puf_monetary_source.py`: closed document validation, pinned codec reuse,
      CSV decoder, three-class facts, self-proving envelope, producer and
      consumer kernels, node factories, registry helpers.
- [x] 96 focused tests, all invented fixtures, all passing.
- [x] `docs/us-puf-monetary-source-projection.md` plus a changelog entry.
- [x] `ruff check .` clean; `tools/ci_test_groups.py --verify` ok; the new test
      file lands in `rest` and `us-p`, never `[defaulted]`.

## Done

| Commit | What |
| --- | --- |
| `cf86618e` | Lane journal |
| `dd1efef3` | Authenticated layout + enumerated projection document, and its generator |
| `cf32465a` | The codec, decoder, envelope and graph nodes |
| `f4a1fd5d` | Layout / projection / refusal / grammar / structure / graph tests |
| `1826525a` | Moved the packaged document out of the country-package closure |
| `b86382b6` | Documentation and changelog |

## Inherited, not introduced

Two integration failures exist on the merge of `2fd9a5cb` into `origin/main`
**before** this slice and are untouched by it: the reviewed raw source
definition is undeclared in the US `country_package.json` (so
`spec_engine.load_bundle` refuses the whole US package), and the reviewed
lineage's test file names the retired US data package, which the launch
contract forbids in the live tree. Both are recorded in the documentation and
the report with their exact remedies.

## Next

Root review of this increment, then the next fit boundary: the contract in
`us-puf-monetary-source-projection-REPORT.md` §"Next fit boundary".

## Not done, by instruction

No push, no merge, no deploy, no nested agents, no real PUF read, no fit, no
calibration, no factor generation, no target-year use.
