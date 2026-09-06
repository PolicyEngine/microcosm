# PUF price baseline — root-review fixes

Bounded fixes to the root review's three actionable findings against the frozen
candidate `8a8f5d23d9d5d20fd85d43cac721dda97e5db5de`, plus the items in the
independent Fable 5.1 review that land inside the same scope. Nothing genuine
ran: no PUF delivery, demographic file, status payload, projection payload,
artifact store or processed dataset was read or statted, no country engine was
imported, nothing was installed and nothing was written to a remote. Every
execution that imports `microcosm` installs the lane's zero-tolerance audit
hook, and every run recorded `blocked_access_attempts: []` and
`country_engine_imported: false`.

## Provenance

| | |
| --- | --- |
| `origin/main` at fork | `c1b8324183819cf91511c002bf724a2676524e87` |
| Accepted source base | `688d00ca24060dbbe04e489ee99b7527750f9628` |
| Review target (frozen) | `8a8f5d23d9d5d20fd85d43cac721dda97e5db5de` |
| Head | `937509443f4584e51c7933b1e478e1509aa75c85` |
| Branch | `us-puf-price-baseline-root-fixes-20260906` |

Worktree cut fresh from `origin/main` and fast-forwarded to the review target;
its tree was byte-identical to the target's (`d0b58af7…`) before any edit. The
review target worktree `microcosm-us-puf-price-baseline-20260906` is untouched
and still reports `8a8f5d23` with an empty `git status --porcelain`.

## State

**Done.** All five required fixes landed, with negative tests. Full regression
green. Report written to
`us-puf-price-baseline-root-fixes-REPORT.md` / `.json`.

## Done

1. **Entry contract closed.** `require_price_baseline_contract` refuses a
   contract that is not the adopted baseline — authority, years, kind, sign
   branch, series, resolved factor bits, both published level texts, the
   `RECID` identity and the `S006` identity weight, and no rule in any other
   role. Called by the adapter, the invariants and `compile_price_baseline`.
   Receipts now read the source year, the weight series, the declared weight
   factor and the factor bits off the contract.
2. **Contract identity cross-checked.** `compiled.sha256`,
   `grown.contract_sha256`, the growth receipt's and the adaptation receipt's
   must be one string, and all four are recorded.
3. **Exact kept-row comparison.** Every kept row of every money column is
   compared bit for bit to the contract's own resolved float64 applied to that
   row's delivered amount, plus the two provenance columns against their
   sources.
4. **`design_weight_unchanged` is honest.** `None` unless a caller passes
   `design_weight_after`, in which case the arrays are compared against the
   adaptation's own `S006`. The declared identity is separately *measured*
   against this run's own weights.
5. **Generator reentrancy.** `extended_parent` restores the cached frozen
   module's declarations in a `finally`. Two same-process calls now produce
   identical documents and the parent's twelve-field check passes each time.

Plus, from the independent review: the sum tolerance is scaled by the gross
rather than the net; a subset contract names the columns it left behind and a
missing pairwise column reports as unobservable instead of counting `None`;
the invented run binds the decode to the status artifact digest.

## Evidence in `R = /Users/maxghenis/architecture-reviews/microcosm-20260904`

- `us-puf-price-baseline-root-fixes-tests-final-head/` — 484 tests, 0 failures
- `us-puf-price-baseline-root-fixes-invented-receipt.json` — graph + direct +
  warm second pass + the row counterexample at 48 rows
- `us-puf-price-baseline-root-fixes-generator-reentrancy-{before,after}.json`
- `us-puf-price-baseline-identity-root-fixes-head.json` — kernel hashes
- `us-puf-price-baseline-root-fixes-code-identity.json`

## Not in scope, still open

Root's fourth note and the independent review's second: the CPI resource
metadata is caller-asserted, the flat-file body is not retained, and a genuine
runner must verify the actual frozen BLS bytes and endpoint records. This lane
deliberately does not promote caller metadata to an authenticated public
resource, and adds no stat or read of microdata. Time-sensitive: BLS overwrites
that distribution monthly.
