# Current ASEC income reporting and routing source

`microcosm.build.us_runtime.current_asec_income_routing_source` qualifies the
exact retained 2025 ASEC person member against the authenticated current-money
owner and projects five original-channel income families the current survey
graph does not yet supply:

| Family | Printed amounts | Printed receipt / routing |
| --- | --- | --- |
| `pension_annuity` | `PNSN_VAL`, `ANN_VAL` | `PEN_YN`, `ANN_YN` |
| `retirement_distribution` | `DST_VAL1`, `DST_VAL1_YNG`, `DST_VAL2`, `DST_VAL2_YNG` | `DST_YN`, `DST_YN_YNG`, `DST_SC1`, `DST_SC1_YNG`, `DST_SC2`, `DST_SC2_YNG` |
| `net_property` | `RNT_VAL` | `RNT_YN` |
| `farm` | `FRSE_VAL` | `FRSE_YN`, `ERN_YN`, `FRMOTR` |
| `other_income` | `OI_VAL` | `OI_YN`, `OI_OFF` |

Thirteen published allocation flags are read alongside them.

Both returned frames carry a digest: `evidence["projection_sha256"]` over
`person` and `evidence["literals_sha256"]` over `asec_literals`. Both are
re-checked at the end of the qualifier, so a host comparing the documented
digests detects a corrupted transport on either frame.

`qualify_current_asec_income_routing(preparation)` takes a live
`AuthenticatedSurveyPopulationPreparation`, requalifies it, captures the pinned
2024 member once, joins it to the money parent by exact native keys, and returns
`CurrentAsecIncomeRoutingValues(person, asec_literals, evidence)`. The returned
values are descriptive transport. Constructing or mutating them issues no source
authority: a consuming host re-runs this qualifier and compares the result
around its own I/O, exactly as the accepted unemployment qualifier requires.

## What the qualifier establishes

- **The exact source member.** The 2024 pin is matched against the retained
  native coverage receipt (member name, archive digest, member digest, row count
  and byte size) before the capture, and the captured file's stat identity and
  SHA-256 are re-checked after it.
- **Row-for-row parent correspondence.** `PH_SEQ`, `A_LINENO` and `A_AGE` from
  the member must equal the money parent's `source_household_id`, `A_LINENO` and
  `A_AGE` at the 2024 scope positions. The join is by native key, so member row
  order cannot change the projection.
- **Literal/amount identity.** Every projected amount is the money owner's own
  value, checked against the member literal, with the parent's `statuses`,
  `validity` and `zero_origin` carried through as `amount_status_*`,
  `amount_validity_*` and `zero_origin_*` columns.
- **A single bounded pass.** One capture, one read, one projection. There is no
  per-family re-capture, no nested run issuer, and no PUF borrow.

## What it deliberately does not establish

- `PNSN_VAL` is printed as the *total combined amount of pension income received
  from all pension sources* — `PEN_SC1`/`PEN_SC2` enumerate company, union,
  federal, state, local, military and railroad pensions. It is not an observed
  private or taxable amount. The archived `TAXABLE_PENSION_FRACTION = 0.590`
  split is a modelled assumption and is not applied or referenced here;
  `pension_annuity_private_share_applied` and
  `pension_annuity_taxable_amount_known` are constant `False`.
- Account code 4 is `Regular IRA`. That identity does not observe any taxable
  fraction, so `retirement_distribution_taxable_amount_known` is constant
  `False` and no taxable component is derived.
- `RNT_YN` asks about land, property rented to others, royalties, roomers or
  boarders, and estates or trusts, while `RNT_VAL` asks only about *income from
  rent after expenses*. Both scopes are carried on every row and
  `net_property_component_split_known` is constant `False`, so the total is not
  independently labelled rental and no subdivision is separately modelled.
- `FRSE_VAL` is farm self-employment, not nonfarm `SEMP_VAL`;
  `farm_is_nonfarm_self_employment` is constant `False`. Its printed label
  includes the composite clause naming `ERN_VAL` (when `ERN_SRCE=3`) and
  `FRM_VAL`, so the total already contains those components.
- `retirement_distribution_regular_ira_amount` is published only when every
  applicable slot is fully resolved on both axes. An unreadable account code
  could itself be a regular IRA, and a declared account whose amount is a
  "none or niu" zero does not observe a zero dollar distribution, so both leave
  the regular-IRA share unknown rather than at zero.
- `OI_OFF` code 20 is the reported alimony category. Nothing maps any other
  code — including 19, `anything else` — onto alimony or onto a miscellaneous
  residual; `other_income_residual_rule_applied` is constant `False`.
  `other_income_routing_status` separates `reported_category` from
  `receipt_without_category`, `category_without_receipt`,
  `unresolved_receipt_routing`, `missing_category_literal`,
  `unrecognized_category_literal` and `niu_category`, and
  `other_income_is_reported_alimony` is set only on a `reported_category` row,
  so an unreadable receipt literal never yields a reported alimony receipt.
- ACS-channel people are not projected at all. `acs_components_modeled` is
  `False` in the evidence and the projection covers only ASEC-channel rows.

## Reporting status vocabulary

`receipt_status` is the single classifier. Only `known_receipt`,
`known_recipient_zero` and `known_nonreceipt` establish a dollar reading
(`KNOWN_AMOUNT_STATUSES`); every other status leaves the canonical amount
unknown rather than completing it with zero.

| Status | Meaning |
| --- | --- |
| `known_receipt` | in universe, receipt yes, amount non-zero (a signed loss included) |
| `known_recipient_zero` | in universe, receipt yes, amount zero on an entry whose printed zero is valid dollars — `ANN_VAL` alone, whose NIU is the separate `-1` code |
| `receipt_with_net_zero` | in universe, receipt yes, amount zero on a signed net measure (`RNT_VAL`, `FRSE_VAL`). Distinct from a gross entry's recipient zero, but **not** a known amount: both print `0 = none or niu` |
| `ambiguous_recipient_zero` | in universe, receipt yes, amount zero on a gross entry whose printed zero reads "none or niu" |
| `known_nonreceipt` | in universe, receipt no, amount zero |
| `niu` | in universe, receipt 0, amount zero or a declared non-money code |
| `missing_amount` / `missing_receipt_literal` | one side of the pair is absent |
| `unrecognized_receipt_literal` | the receipt literal is malformed or outside the printed range |
| `contradictory_no_nonzero`, `contradictory_niu_nonzero`, `contradictory_declared_niu_amount`, `contradictory_outside_reporting_universe` | retained source contradictions, excluded from every canonical amount |
| `contradictory_offroute_evidence` | distributions only: the route that does not apply carries dollars or an answered recipiency |
| `unresolved_slot_composition` | distributions only: an applicable slot declares an account whose amount is a "none or niu" zero |
| `outside_reporting_universe` | the printed universe excludes the row and the literals agree |
| `unresolved_reporting_universe` | the printed universe cannot be resolved from the retained literals |

Which recipient zeros resolve is read from the pinned domains artifact, not
decided here: `_zero_is_dollars` compares each entry's `zero_semantics` against
`valid_zero_dollars`. Eight of the nine entries record
`none_or_niu_not_distinguishable_from_amount_alone`, including both signed net
measures, so a signed zero is separated by label but never completed.

The receipt code labels are per entry, not shared: `receipt_codes(field)` reads
the printed zero label from `RECEIPT_ENTRIES`, so `OI_YN` reports `none or niu`
where `PEN_YN` reports `niu` and `FRSE_YN` reports `Niu` exactly as printed.

Other-income routing (`other_income_routing_status`) follows the same universe:
`OI_OFF` is printed for `OI_YN = 1` and `OI_YN` for persons aged 15+, so a row
outside or unresolved on that universe reports
`outside_reporting_universe_routing` or `unresolved_reporting_universe_routing`
and never a reported category.

Universes are per family and never reduced to age alone. `PEN_YN`, `ANN_YN`,
`RNT_YN` and `OI_YN` print `All Persons aged 15+`. `FRSE_YN` prints
`ERN_YN=1 or FRMOTR=1`, so the farm universe is `True` when either literal is
`1`, `False` when both are known and neither is, and unresolved otherwise —
a nonfiler is never treated as a non-recipient.

## Source-level questions preserved as evidence

These are printed-dictionary facts, kept distinguishable from modelling
judgments. The first four are recorded under
`evidence["dictionary"]["printed_universe_questions"]`; the rest under
`evidence["dictionary"]["printed_scope_and_code_questions"]`:

- `DST_VAL1`'s printed universe is `DST_SC1 = 1` although its label names the
  source-1 distribution amount; taken literally that would restrict it to 401k
  accounts. Retained verbatim, not silently corrected.
- `DST_SC2_YNG`'s printed universe names `DST_VAL_YNG`, a field with no
  dictionary entry.
- `I_DSTVAL1COMP`'s printed `Universe:` line is empty.
- `DST_YN` and `DST_YN_YNG` print only the age-58 split. They print no 15+ floor
  as the four age-universe families (`PEN_YN`, `ANN_YN`, `RNT_YN`, `OI_YN`) do,
  and they are not gated on other literals as the farm family is — `FRSE_YN`
  prints `ERN_YN=1 or FRMOTR=1` and so carries no age floor either. Coverage
  below age 15 is therefore recorded as unresolved rather than resolved either
  way.
- `OI_YN`'s printed zero label is `none or niu` where `PEN_YN`, `ANN_YN`,
  `DST_YN` and `RNT_YN` print `niu`, so a zero receipt literal is not
  interchangeable across families.
- `DST_SC1` is gated on `DST_VAL1 > 0 and a_age ≥ 58` while `DST_SC1_YNG` is
  gated on `DST_YN_YNG = 1 and a_age < 58` — the two routes are not symmetric.

Both printed recipiency literals are retained per row
(`retirement_distribution_receipt_58_*` and `..._receipt_young_*`) alongside all
four slots. The age route selects which pair applies;
`retirement_distribution_offroute_receipt` and
`retirement_distribution_offroute_nonzero` flag an answered recipiency or a
non-zero amount on the route that does not apply, so the complete slot/receipt
pattern stays inspectable rather than being reduced to the applicable half.

## Amount semantics from the money owner

The authenticated money owner normalizes `ANN_VAL`'s printed `-1` to a stored
zero and records `CodebookStatus.DECLARED_NIU`. `amount_state` therefore reads
the dollar meaning of a cell from the parent's status axis, never from the
stored number: re-deriving NIU from the number would read that cell as a zero
dollar annuity. `evidence["declared_niu_normalized_to_zero"]` (a top-level key) lists the
affected entries.

The money owner restates non-2024 cohorts to the pinned price basis, so the
literal-identity join is taken only on `person_years == 2024`; widening it to
the pooled 2022/2023 cohorts would require handling that restatement first.
`evidence["restatement_note"]` records this.

## Allocation provenance

`*_allocation_origin` is derived from published flags only:

- `publisher_allocated` — some applicable flag is non-zero.
- `published_flags_all_zero` — every applicable flag is populated and reads
  zero, and every field in the family carries a published flag. This is **not**
  an assertion of non-allocation: each flag prints a conditional universe (for
  example `I_RNTVAL` is printed for `RNT_VAL > 0`) and those universes are not
  evaluated here, so a non-recipient row sits outside its own flag's universe.
- `published_flags_all_zero_with_unflagged_fields` — the same, but the family
  also holds a field the dictionary does not flag.
- `allocation_flag_not_populated` — a flag literal is absent. Every flag here
  prints a conditional universe, so this is not a defect and not "no
  allocation".
- `unresolved_allocation_provenance` — a flag literal is malformed or outside
  its printed range.

`PUBLISHED_ALLOCATION_FLAG_BY_FIELD` names which flag covers which field.
`UNFLAGGED_FIELDS` records the ten fields for which the 2025 dictionary
publishes no flag: `PNSN_VAL`, `FRSE_VAL`, `FRSE_YN`, `OI_OFF`, `OI_YN`,
`DST_VAL1_YNG`, `DST_VAL2_YNG`, `DST_YN_YNG`, `DST_SC1_YNG`, `DST_SC2_YNG`.
`AMBIGUOUS_FLAG_COVERAGE` records three printed ambiguities: `I_DSTSCCOMP`'s
label names `DST_SC(2)` while its universe names both routes, so its coverage of
the under-58 source codes is unresolved; `I_DSTSC`'s label uses the same
`DST_SC(2)` notation, so mapping it to both `DST_SC1` and `DST_SC2` reads the
parenthesis as a slot count that the dictionary does not state; and `I_FRMYN`
prints an empty `Values:` block, so only its `(0:9)` range header is published
and no meaning is claimed for its codes. A nonzero `I_FRMYN` alone yields
`allocation_code_meaning_unpublished`; a documented nonzero allocation flag
such as `I_ERNYN` can still establish `publisher_allocated`. All-zero readings
remain descriptions of the published codes, not assertions of nonallocation.
The original flag values and parse statuses remain unchanged. No row is ever labelled a raw
respondent value.

## Constants

The nine printed money entries are **not** retyped here: they are read on demand
from the packaged `asec_current_money_domains_v1.json` under its pinned digest
(`money.RESOURCE_PINS[0]`), which already attests printed position, length,
page, range, universe and values per vintage. The routing code systems
(`RECEIPT_CODES`, `ACCOUNT_CODES`, `OTHER_INCOME_CATEGORIES`,
`ALLOCATION_ENTRIES`) are not in that artifact and are defined here from the
pinned dictionary.

The printed entry tables (`RECEIPT_ENTRIES`, `ACCOUNT_ENTRIES`,
`ALLOCATION_ENTRIES`, `OTHER_INCOME_CATEGORY_ENTRY`) are `NamedTuple`s, so every
use site and the emitted receipt reach their fields by name rather than by
position.

Two adjacent definitions exist in tree and are deliberately **not** imported, so
this qualifier keeps the accepted source-only import neighbourhood
(`asec_coverage_authentication`, `asec_current_money`, `source_csv_builtin`,
`survey_population_preparation`, `support_provenance`) and stays free of the
modelled stage machinery:

- `retirement_distributions.US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS`
  — the same eight `DST_SC*`/`DST_VAL*` column names.
- `alimony._ASEC_ALIMONY_OTHER_INCOME_CODE` and
  `alimony._ASEC_STRIKE_BENEFITS_OTHER_INCOME_CODE` — the same category codes 20
  and 12.
- `retirement_distributions._VALID_ACCOUNT_CODES` — the same 0-7 account code
  domain that `ACCOUNT_CODES` gives printed labels for.

`test_routing_code_systems_agree_with_the_existing_domain_constants` asserts
both agreements, per `docs/shared-constants.md` rule 5. If a later change makes
the modelled modules importable from a source qualifier, import those
definitions directly and drop the local copies.

`RINT_SC1`/`RINT_SC2` reuse the identical 0:7 retirement-account enumeration for
retirement *interest*. They are out of this slice; a future consumer mapping
account codes to outputs must decide whether they share this code system.

## Remaining work

This lane delivers source qualification only. Still open, and explicitly not
decided here:

1. **Canonical attachment.** Which projected columns become graph leaves, on
   which clone, and under what name. These families overlap PUF-owned values, so
   the unemployment/health both-clone policy cannot simply be reused. Root's
   active amount owner owns `graph_us_survey_enrichment`.
2. **ACS clone0 components.** ACS publishes aggregate anchors
   (`acs_retirement_income`, `acs_interest_dividend_rental_income`) rather than
   these components. A conditional model plus a documented reconciliation rule
   is needed, and the existing interest/dividend draws and ACS `INTP` overlap
   mean rent cannot be added independently without checking that aggregate.
3. **Tax composition.** The pension private/taxable split and the distribution
   taxable fraction remain unobserved. Any split must be approved explicitly
   against the source, not inherited from the archived assumptions.
4. **Net property decomposition.** The receipt/amount scope mismatch must be
   resolved before rent, royalties, roomers/boarders and estate/trust income are
   modelled as separate components.
5. **Other-income residual.** Categories other than 20 are preserved but
   unrouted; a residual-to-miscellaneous rule is a separate, reviewed decision.
