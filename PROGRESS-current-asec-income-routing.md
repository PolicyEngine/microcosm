# Current ASEC income reporting/routing source qualifier — lane progress

Branch `current-asec-income-routing-20260912`, worktree
`_worktrees/microcosm-current-asec-income-routing-20260912`, base
`d5cbe60b2c6402648565f5139dbf7d94be209def` (reviewed US integration).

Journal, not state. Check git/GitHub for current truth.

## Scope

Deliver a thin **source qualifier** plus a **pure reporting/routing projection**
for the five original-channel ASEC income families the current graph does not
yet supply: pension/annuity, IRA distributions, net property income, farm, and
other income. Consumed later by the already-owned `graph_us_survey_enrichment`
host, which root's active amount owner owns. This lane creates **no** host, no
issuer, no engine execution, no native cell writes, no release claim.

## State

- [x] Read the actual-current-graph trace
      (`_recovered/scratch-backup/893/codex-takeover-20260912/original-tax-graph-gap-source-trace.md`)
      and confirmed the current host does not call `derive_cps_carried_current_leaves`
      or the preclone gap-fill.
- [x] Verified the official 2025 dictionary bytes locally:
      SHA256 `5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f`.
- [x] Read the accepted UC pattern
      (`current_asec_unemployment_source.py`, reviewed hash
      `624c76ae058e7b2a466a017eba348327c94751088d97011d3474cd8314187519`) read-only
      in the successor worktree; not mutated.
- [x] Read the in-base siblings `current_social_security_source.py`,
      `current_asec_demographics.py`, `current_survey_predictors.py`.
- [x] Extracted every target field's printed literal (length/position/range,
      verbatim Universe and Values) from the verified dictionary, and
      cross-checked each against `asec_current_money_domains_v1.json`: exact match
      on position, length, universe and values for all nine money fields.
- [x] Module
      `packages/microcosm-build/src/microcosm/build/us_runtime/current_asec_income_routing_source.py`
      (commit 12ac3a75b, corrected in aa7bde7d4 and 7e2c7e77b).
- [x] Tests `packages/microcosm-build/tests/test_us_current_asec_income_routing.py`
      — 64 passing, no engine, no PUF fixture.
- [x] Source contract note `docs/us-current-asec-income-routing-source.md`
      plus `changelog.d/us-current-asec-income-routing-source.added.md`.
- [x] `ruff check .` and `ruff format --check` clean;
      `tools/ci_test_groups.py --verify` reports `verification=ok` and the new
      file lands in `[fast] rest` and `[engine] us-am`, not `[defaulted]`.
- [x] Mutation-checked the regressions: default-zero completion of an ambiguous
      recipient zero, reading NIU from the raw number instead of the parent
      status axis, collapsing receipt-with-net-zero into known nonreceipt, and
      resolving an unknown slot account into a known non-IRA zero each turn the
      suite red; reverting each returns it green.

## Second review round: corrections after adversarial review

An independent five-lens adversarial review of the finished lane (source
fidelity, projection semantics, owner pattern, tests, repository fit) found a
further set of defects, every one of which is fixed on this branch:

1. **Fail-closed roster (CI-breaking).** `test_us_spine_blindness.py` globs
   every `us_runtime` module and fails closed on an unregistered one. The
   qualifier is now classified there and listed among the reviewed source-spine
   provenance owners. Three of that file's tests remain red for four modules
   already unclassified at the reviewed base `d5cbe60b2` —
   `graph_fiscal_dense_calibration.py`, `graph_fiscal_measurement.py`,
   `graph_puf55_route_attachment.py`, `graph_survey_puf55.py` — which this lane
   neither owns nor touched. **Root should route those to their owners.**
2. **Recipient zeros were completed on an unprinted rationale.** `RNT_VAL` and
   `FRSE_VAL` print the same `0 = none or niu` label the gross entries print, so
   a signed net measure's recipient zero is no longer a known amount. Only
   `ANN_VAL`, whose printed zero is `valid_zero_dollars`, resolves; that test is
   read from the pinned domains artifact rather than decided in code.
3. **Off-route distribution evidence yielded an affirmative zero.** Off-route
   dollars or an answered off-route recipiency now contradict the row, and an
   applicable slot declaring an account whose amount is a "none or niu" zero
   leaves both the total and the regular-IRA share unresolved.
4. **Other-income routing ignored the printed receipt universe**, so an
   out-of-universe row read as reported alimony.
5. **Allocation origins asserted publisher-confirmed non-allocation** from flags
   whose printed universes are conditional and unevaluated.
6. **The printed zero receipt label was collapsed** to one shared `niu_or_none`
   across nine entries that print three different labels.
7. **`asec_literals` escaped with no digest**, outside the final identity check.
8. **`I_FRMYN` prints an empty `Values:` block**; its codes were described as
   following `I_ANNVAL`. **`FARM_AMOUNT_SCOPE` dropped its composite clause.**
   The claim that every other family prints a 15+ floor was false — the farm
   family prints none.
9. **The doc called `TAXABLE_PENSION_FRACTION` archived**; it is live in
   `cps_carried_current.py` under the prepared-ASEC stage, so a second split
   attached here would double-count.
10. Printed entry tables are `NamedTuple`s now, the account code domain is
    compared against `retirement_distributions._VALID_ACCOUNT_CODES` rather than
    a re-typed literal, and `project_income_routing` validates its routing token
    arrays.

## Corrections made after independent review of the first draft

1. The money owner normalizes `ANN_VAL`'s printed `-1` to a stored zero and
   records `DECLARED_NIU` (`asec_current_money.py:974-977`). The first draft
   re-derived NIU from the stored number, which would have read that cell as a
   zero dollar annuity and also failed the literal-identity join. The amount
   reading now comes from the parent's status axis.
2. The nine printed money entries were retyped by hand; they are already
   attested per vintage in `asec_current_money_domains_v1.json`. They are now
   read from that packaged artifact under `money.RESOURCE_PINS[0]`.
3. The published-allocation roster wrongly listed `DST_VAL1`, `DST_VAL2`,
   `DST_YN`, `DST_SC1`, `DST_SC2` and `FRMOTR` as unflagged; all six are
   flagged. `OI_YN` is unflagged and was missing from the list.
4. Three printed universes were transcribed with ASCII `>=` where the
   dictionary prints U+2265; they now carry the printed character.

## Next

Root review and integration into `graph_us_survey_enrichment`. The open
decisions are listed under "Remaining work" in the contract note: canonical
attachment and clone policy for these PUF-overlapping leaves, the ACS clone0
conditional model and its reconciliation against the ACS aggregate anchors, the
unobserved pension and distribution tax composition, the net property
decomposition, and any other-income residual rule.

## Verified source literals (2025 dictionary, pages 43-49 + allocation pages)

All quoted verbatim from the pinned PDF.

| Field | Len | Pos | Range | Universe as printed |
| --- | --- | --- | --- | --- |
| PNSN_VAL | 7 | 571 | (0:9999999) | `PEN_YN = 1` |
| PEN_YN | 1 | 570 | (0:2) | `All Persons aged 15+` |
| ANN_VAL | 6 | 438 | (-1:999999) | `ANN_YN = 1` |
| ANN_YN | 1 | 444 | (0:2) | `All Persons aged 15+` |
| DST_VAL1 | 6 | 495 | (000000:999999) | `DST_SC1 = 1` |
| DST_VAL1_YNG | 6 | 501 | (000000:999999) | `DST_SC1_YNG = 1` |
| DST_VAL2 | 6 | 507 | (000000:999999) | `DST_SC2 = 1` |
| DST_VAL2_YNG | 6 | 513 | (000000:999999) | `DST_SC2_YNG = 1` |
| DST_SC1 | 1 | 491 | (0:7) | `DST_VAL1 > 0 and a_age >= 58` |
| DST_SC1_YNG | 1 | 492 | (0:7) | `DST_YN_YNG = 1 and a_age < 58` |
| DST_SC2 | 1 | 493 | (0:7) | `DST_VAL2 > 0 and a_age >= 58` |
| DST_SC2_YNG | 1 | 494 | (0:7) | `DST_VAL_YNG > 0 and a_age < 58` |
| DST_YN | 1 | 519 | (0:2) | `Persons aged 58 and over (a_age >= 58)` |
| DST_YN_YNG | 1 | 520 | (0:2) | `Persons under age 58 (a_age < 58)` |
| RNT_VAL | 6 | 621 | (-9999:999999) | `RNT_YN = 1` |
| RNT_YN | 1 | 627 | (0:2) | `All Persons aged 15+` |
| FRSE_VAL | 7 | 390 | (-9999999:9999999) | `ERN_YN=1 or FRMOTR=1` |
| FRSE_YN | 1 | 397 | (0:2) | `ERN_YN=1 or FRMOTR=1` |
| ERN_YN | 1 | 381 | (0:2) | `WORKYN=1 OR WTEMP=1` |
| FRMOTR | 1 | 389 | (0:2) | `ERN_OTR = 1` |
| OI_VAL | 6 | 549 | (0:999999) | `OI_YN = 1` |
| OI_OFF | 2 | 547 | (0:20) | `OI_YN = 1` |
| OI_YN | 1 | 555 | (0:2) | `All Persons aged 15+` |

Source-level questions that stay open (evidence, not modeling judgments):

- `PNSN_VAL` is printed as "total combined amount of pension income received
  from **all** pension sources". It is not an observed private/taxable amount.
  The legacy 0.590 split is a modeled assumption and is not applied here.
- `RNT_YN`'s printed question covers rent, royalties, roomers/boarders **and**
  estates or trusts; `RNT_VAL`'s printed question asks only about "income from
  rent after expenses". The receipt and amount questions do not have the same
  printed coverage, so the total is not independently labelled rental.
- `DST_SC*` code 4 is `Regular IRA`. Account identity does not observe a taxable
  fraction, so no taxable component is derived.
- `DST_VAL1`'s printed universe is `DST_SC1 = 1` (401k account) although its
  label is the source-1 distribution amount; preserved verbatim as an
  unresolved printed-universe ambiguity rather than silently corrected.
- `DST_SC2_YNG`'s printed universe names `DST_VAL_YNG`, a field with no
  dictionary entry; preserved verbatim.
- `OI_YN`'s printed `0` label is `none or niu`, unlike `PEN_YN`/`ANN_YN`/
  `RNT_YN`/`DST_YN` whose `0` is `niu`. Zero receipt literals are therefore not
  interchangeable across the five families.
- Published allocation flags exist for `ANN_VAL` (`I_ANNVAL`), `ANN_YN`
  (`I_ANNYN`), `PEN_YN` (`I_PENYN`), `RNT_VAL` (`I_RNTVAL`), `RNT_YN`
  (`I_RNTYN`), `OI_VAL` (`I_OIVAL`), `ERN_YN` (`I_ERNYN`), and the DST
  composites (`I_DSTSC`, `I_DSTSCCOMP`, `I_DSTVAL1COMP`, `I_DSTVAL2COMP`,
  `I_DSTYNCOMP`). The dictionary publishes **no** direct flag for `PNSN_VAL`,
  `FRSE_VAL`, `FRSE_YN`, `OI_OFF`, `DST_VAL*`, `DST_YN` or `FRMOTR`; their
  allocation provenance is therefore recorded as unresolved, not as "raw".
- `I_DSTVAL1COMP`'s printed `Universe:` line is empty in the dictionary.
