# SOI modeled return universes

IRS return-based targets measure modeled filers. The fiscal materializer applies
the installed PolicyEngine-US `tax_unit_is_filer` result before it collapses
tax-unit contributions to households. The restriction applies to return counts,
positive-component indicators, and amounts at national, state, and congressional
district levels. Existing AGI, filing-status, taxable, credit, itemization,
geography, and component restrictions remain intersections with this mask.

For example, a household may contain two separate tax units with $5,000 of AGI
each. If only one files according to the model, the household contributes one
return and $5,000 to an applicable SOI target. Its other income remains in the
support frame and in calculations outside that SOI measurement.

Every IRS target must declare `soi_return_universe` in its metadata:

| Declaration | Modeled restriction |
| --- | --- |
| `all_returns` | Actual `tax_unit_is_filer` plus the target's other restrictions |
| `itemized_returns` | Actual filer and actual `tax_unit_itemizes`, plus other restrictions |
| `returns_excluding_dependents` | Explicitly unsupported and dropped pending a verified return-level dependency input contract |
| Missing, unknown, or contradictory | Refused before materialization |

Unsupported targets retain their names, source universes, periods, measures, and
reason in `unsupported_target_entries` and `dropped_target_names`. They never
produce modeled rows or enter the materialized registry, even when a same-named
column already exists. Table 4.3 facts pass through the real Ledger compiler and
remain visible in cold and warm diagnostics. Existing coverage, parity, and
release gates remain active: a dropped unsupported target is not calibrated.
This change adds no reviewed exclusion and does not authorize launch with the
missing universe.

The fiscal release builder checks the compiled Ledger registry before loading
or downloading a base population, and refuses unsupported SOI universes with
each target name, source universe, and reason. The check also applies to
`--evidence-release`, whose existing exception covers terminal verdicts only.
The diagnostic scorer and low-level materializers retain their supported and
unsupported target reporting. Target substitution, parity, coverage, and final
release gates remain active for builds that pass this preflight.

The itemized declaration requires itemization even without the redundant
`itemized_only=true` flag. An explicit false flag or contradictory source layout
is an error. An `all_returns` row may still declare `itemized_only=true` for its
specific measure. Missing or incompatible engine predicates also cause refusal.
The materializer does not substitute legal filing requirements, positive income,
positive liability, or a default all-filer mask for the actual model decision.

The [IRS table index](https://www.irs.gov/statistics/soi-tax-stats-individual-statistical-tables-by-size-of-adjusted-gross-income)
distinguishes all returns and itemized returns. Its
[tax-rate and income-percentile tables](https://www.irs.gov/statistics/soi-tax-stats-individual-statistical-tables-by-tax-rate-and-income-percentile)
separately identify returns excluding dependents. A dependent return means a
separate return filed by a person claimed on another return; it does not mean
the adult return claiming a child. The IRS explains this distinction in its
[article on dependent returns, page 1](https://www.irs.gov/pub/irs-soi/soi-a-inde-id1502.pdf).

## Filing decisions and provenance

The tested PolicyEngine-US 1.819.0 decision combines required filing, refundable
credit eligibility with willingness to file for credits, and voluntary filing.
The materializer executes the installed formula. It preserves existing inputs
and does not change filing propensities or their model defaults.

The compilation's `soi_return_universe` receipt records:

- Target names, measures, source and target periods, declared source universes,
  itemization requirements, and any existing target-value universe bridge.
- Ordered tax-unit identity, and hashes of the actual filer and required itemizer
  masks at the modeled year.
- For `would_file_if_eligible_for_refundable_credit` and
  `would_file_taxes_voluntarily`, either the supplied boolean column's hash or an
  explicit declared model default. A default is labeled a model assumption,
  not an observed survey response. Supplied nulls or numeric truth values fail.
- PolicyEngine-US and Core versions and hashes of the direct filing declarations.
  This is a direct declaration inventory, not a transitive model-source closure.

Target-frame materializer version 13 invalidates earlier cache identities. Cold
writes preserve the receipt and `soi_live_input_frame_sha256` in compilation
metadata. The live-input digest uses the existing deterministic late-producer
Frame content hash over ordered entity and link tables, weights, strata, mass
log, and metadata, with its existing string normalization and exclusion of the
self-referential late-producer authority key. It is computed before design
columns are added. This includes imputed filing propensities, credit take-up,
and other inputs that may change after the source checkpoint hash is fixed.

Warm reads first validate the stored filer receipt against the stored Frame.
They then require the actual live base Frame and matching input digest, and
validate its tax-unit order and supplied filing-input content against the
receipt. Missing live inputs or a changed digest cause a cache miss and normal
recomputation. Malformed stored receipts or input digests refuse. No filing
calculation is run to decide cache reuse. Runtime versions and direct declaration
bytes remain checked, and both receipt and digest remain available at their
same compilation keys on either path.

The checkpoint stores ordinary local compilation provenance. Its readback does
not independently authenticate the original raw source, recalculate the filer
mask, or establish a cryptographic seal over the cached design rows. Declared
default origins and mask hashes remain recorded claims from the cold calculation.

## Source-population limits

The mask establishes the modeled target-year filer measurement. It does not
establish complete coverage of separate dependent returns in an `all_returns`
support frame. The installed model exposes head/spouse dependency-elsewhere
predicates, but this correction does not establish authenticated coverage of
their person input or the exact IRS head/spouse exclusion criterion.

Source and modeled periods remain separate. National
[Publication 1304](https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-returns-complete-report-publication-1304)
uses sampled returns. The
[2022 state workbook](https://www.irs.gov/pub/irs-soi/22in54us.xlsx) describes
returns initially processed in calendar 2023, including earlier tax years, and
excludes subsequent corrections. State totals also distinguish Other Areas.
The [2022 CD guide](https://www.irs.gov/pub/irs-soi/22incddocguide.docx) describes
its processing period, geographic match exclusions, and suppression rules.
Applying a 2024 filer predicate does not reproduce those processing or geographic
populations. Existing source-period and district-crosswalk metadata remains
necessary.

The existing taxable-interest target-value bridge can rebase an all-return
distribution to an excluding-dependent control. Its
`soi_return_universe_bridge` metadata remains a declared approximation to a target
value; it does not redefine the modeled row as excluding-dependent or establish
population equivalence. This correction does not change target values, aging,
holdouts, acceptance criteria, or unsupported source filters.

The regression in
`packages/microcosm-build/tests/test_us_soi_filer_universe.py` uses four invented
tax units in three households and the actual installed model. It covers required,
credit-related and voluntary filing, positive nonfiler income, itemizers,
geographic slices, explicit input/default origins, actual compiled Table 4.3 targets, changed live inputs,
and checkpoint readback.
It does not certify a population dataset.
