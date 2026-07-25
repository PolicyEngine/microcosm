# QBI v3 evidence final report

## Outcome

The `qbi-v3-evidence` branch now contains deterministic estimation code,
tests, and two committed provisional evidence resources for populace #530,
task-board item 8. It is based on local `repeal-validation-298` at `e45f797`
and was developed only in
`.claude/worktrees/populace-wt-530-v3`.

Delivered:

- `packages/populace-build/src/populace/build/us_runtime/qbi_v3_evidence.py`
  implements SCF record construction, weighting, cell collapse, empirical
  margin quantiles, SOI parsing, ratio derivation, and strict validators.
- `tools/build_us_qbi_v3_evidence.py` is the configurable, path-agnostic,
  deterministic builder.
- `packages/populace-build/src/populace/build/us/qbi_employer_structure_v1.json`
  contains the SCF employer-presence, conditional employer-size, empirical
  profit-margin, and JCT-comparison evidence.
- `packages/populace-build/src/populace/build/us/qbi_wage_capital_priors_v1.json`
  contains the form/industry SOI wage and capital priors.
- Both JSON files are declared in the US spec-only country package, carry
  `"provisional": true`, contain no executable entrypoint or absolute local
  path, and are not wired into simulation code.
- Synthetic fixture coverage, packaged-resource contracts, a changelog
  fragment, and the two progress ledgers are committed.

## SCF employer evidence

### Implicate convention and sample

The estimator pools all five SCF implicates and assigns each record
`X42001 / 5`, following the SCF simple-statistic convention. Effective
unweighted counts are pooled record counts divided by five. This produces
point estimates only; it does not claim replicate-weight or
multiple-imputation variance estimates.

The extraction requires `X3103 == 1`, `X3104 == 1`, and `X3105 >= slot`, then
stacks detailed business slots 1 and 2. It does not further restrict on
whether the respondent or spouse is the family member working in the
business. Income bands use whole-business net income multiplied by the
household ownership percentage.

### SCF versus JCT

| Comparison | Pooled records | Effective n | Weighted business interests | Headcount <= 1 | Gap from JCT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main: owned net income > 0, forms 1/2/3/11/40 | 5,341 | 1,068.2 | 13,383,755.94 | 35.4919% | -48.7081 pp |
| Strict: owned net income > 0, forms 1/2/3/11 | 5,325 | 1,065.0 | 13,180,529.39 | 34.5661% | -49.6339 pp |
| JCT zero-W-2-employee QBID-generating firms | — | — | — | 84.2000% | — |

JCT also reports that zero-employee firms receive 35.7% of QBID dollars,
that more than 95% of sole proprietorships have zero employees, and that more
than 80% of partnerships have zero employees.

The SCF and JCT values should not be treated as competing estimates of the
same parameter:

- JCT counts QBID-generating tax firms with no W-2 employees; SCF observes
  household business interests.
- SCF headcount includes owners, paid workers, family members, and unpaid
  workers. Therefore `headcount > 1` is only an upper-bound employer-presence
  proxy, and `headcount <= 1` is a lower-bound zero-W-2-employer proxy.
- JCT covers tax year 2022 and actual deduction-generating firms. Survey-year
  2022 SCF business flows refer to 2021 and do not identify SSTB status,
  aggregation, carryforwards, taxable-income limits, or actual Section 199A
  eligibility.
- SCF provides detailed records for only the first two actively managed family
  businesses and cannot deduplicate a firm represented by separate owner
  households.

The gap is therefore expected to be large. The resource records it as an
external comparison, not a calibration target.

### Cell support and margins

The resource has all 192 requested
`6 income bands x 4 legal-form groups x 8 SCF industry bins` cells.

- Employer-presence donors: 13 exact, 99 income/form, and 80 form-level.
- Conditional employer-size donors: 8 exact, 88 income/form, and 96
  form-level.
- Profit-margin donors: 10 exact form/industry and 22 form-level.

The minimum is effective unweighted `n = 30`. Presence collapses
`income/form/industry -> income/form -> form -> all`. Employer size applies
the same hierarchy independently among records with headcount above one.
Validators reconstruct all candidate counts from requested cells and require
the earliest eligible donor.

`profit_margin_quantiles` replaces the invented Beta margins with weighted
empirical q05/q25/q50/q75/q95 values of whole-business pretax net income
divided by whole-business receipts. The sample requires positive owned net
income and positive receipts. Values are not capped; the SCF can report net
income above receipts.

## SOI wage and capital evidence

Ranges below use the finest classified industry level available for each
form. They are ratios of published aggregate industry dollars, not averages
of firm-level ratios.

| Form | Tax year | Published / finest classified | Valid wage cells | Wage-share range | Valid capital cells | Capital range | Capital interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Sole proprietorship | 2023 | 153 / 126 | 103 | 0.4568%–34.1482% | 123 | 0.2545%–46.7070% | Depreciation-deduction/receipts flow proxy; `proxy: true` |
| Partnership | 2023 | 20 / 18 | 18 | 2.3615%–42.2970% | 18 | 12.0216%–2,322.2143% | Gross depreciable assets/receipts |
| S corporation | 2022 | 90 / 75 | 73 | 4.1531%–78.4294% | 73 | 6.6613%–549.5035% | Gross depreciable assets/receipts |

Capital stock divided by annual receipts can legitimately exceed 100%,
especially for capital-intensive industries. No winsorization was applied.

Form-specific wage numerators are:

- Sole proprietorship: Table 1 Payroll, which exactly cross-checks to Table 2
  cost of labor plus salaries and wages.
- Partnership: cost of labor plus salaries and wages. Guaranteed payments to
  partners are retained as a diagnostic but excluded from wages.
- S corporation: compensation of officers plus salaries and wages. The table
  does not publish a separate cost-of-goods-sold labor line.

Partnership and S-corporation capital use gross depreciable book assets before
accumulated depreciation, excluding land. This is the closest public stock
analog in these tables, not statutory tax UBIA. Sole-proprietor public tables
do not provide an asset stock, so depreciation including Form 8829 is retained
as an explicitly marked flow proxy.

## SOI parsing landmines

- The two sole-proprietor inputs are genuine legacy BIFF `.xls` files.
  Disclosure markers are stored in custom number formats, so the parser must
  use `xlrd` with formatting information. Reading values alone leaks
  disclosure-combined numbers.
- Sole-proprietor Table 1 is industry-by-row while Table 2 is
  industry-by-column. The parser aligns all 153 numbered entries, checks exact
  receipts equality, and verifies Payroll against cost of labor plus salaries.
- Spreadsheet used ranges include titles, blank regions, notes, and caution
  footers. Parsing stops from numbered headers or audited labels rather than
  `max_row`/`max_column`.
- Partnership and corporation workbooks contain merged multirow headers.
  Each published industry path must resolve through the merged-cell anchor.
- Partnership Table 1 and Table 3 labels contain footnote suffixes and must be
  normalized before column alignment. “Nature of business not allocable” is
  excluded from finest summaries.
- “Unclassified establishments” is likewise excluded from the sole-proprietor
  finest universe. The 18 partnership sector totals remain finest available
  because the public partnership table has no more detailed rows.
- `**`/combined and `d`/deleted values become null. Caution-marked numeric
  estimates remain usable and retain their flag. Validators prohibit a
  disclosure-suppressed raw value or derived ratio from remaining numeric.
- Sole-proprietor Table 1 depreciation is canonical because it includes Form
  8829. Partnership/S-corporation depreciation is diagnostic only; its exact
  cells are still recorded.
- Partnership gross-asset aggregates omit some small returns exempt from
  Schedule L while receipts cover all partnerships, likely biasing partnership
  capital intensity downward.
- The all-corporation minor-industry balance-sheet table was inspected but not
  used: it mixes C and S corporations, so its finer columns cannot identify an
  S-corporation prior.

Every emitted industry entry records its source table, tax year, sheet,
industry header cell(s), receipts cell, wage component cell(s), capital cell,
publication flags, and calculation. Partnership and S-corporation
depreciation diagnostics also have dedicated cell references.

## Supervisor re-adjudication list

These are the judgment calls made in this lane. None is hidden as a factual
identity.

### SCF

1. Pool five implicates with `X42001 / 5` rather than first averaging each
   household. These are equivalent for linear point estimates but neither
   supplies uncertainty estimates.
2. Treat a household-business-implicate interest as the estimation unit, not a
   unique tax firm.
3. Include the first two detailed actively managed family businesses and omit
   additional businesses for which the public SCF has no equivalent detail.
4. Use the family-level active-management screen without requiring the
   respondent or spouse personally to be the working family member.
5. Use `headcount > 1` as employer presence despite owners, unpaid family, and
   other non-W-2 workers being included.
6. Use headcount bands 2–4, 5–9, 10–24, 25–99, and 100+, accepting SCF rounding
   above 10 and the 5,000 top code.
7. Define the main QBI-positive comparison as owned net income above zero and
   forms 1/2/3/11/40. This is a proxy, not an eligibility calculation.
8. Group code 40 (“not a formal business type”) with sole proprietorships in
   the main estimate and show a strict sensitivity that excludes it.
9. Group partnerships/cooperatives with LP/LLP/LLC records. Keep other
   corporations and unknown responses in a separate evidence group.
10. Band on household-owned net income rather than whole-firm net income,
    although whole-firm scale may predict headcount better.
11. Use the selected six income bands and the published seven-bin SCF industry
    collapse plus code 99.
12. Set effective minimum n to 30 and use the stated nested hierarchy. The
    final global donor is used even if global n were below 30.
13. Collapse conditional employer size independently of employer presence.
14. Use uncapped empirical weighted inverse-CDF margin quantiles at five
    probabilities and fall back form/industry -> form -> all.
15. Do not force the SCF employer estimate toward the JCT anchor because the
    units, worker definitions, years, and eligibility universes differ.

### SOI and crosswalk

16. Use ratios of aggregate published dollars rather than unobserved
    firm-level conditional distributions.
17. Make sole-proprietor Table 1 Payroll canonical; retain Table 2 components
    as an identity cross-check.
18. Include partnership cost of labor but exclude guaranteed partner payments
    from the wage numerator.
19. Include S-corporation officer compensation and salaries, while accepting
    that separately unpublished COGS labor may be omitted.
20. Treat gross depreciable book assets excluding land as a UBIA-intensity
    prior for partnerships and S corporations without an age/basis adjustment.
21. Use sole-proprietor depreciation including Form 8829 as a flow proxy rather
    than inventing an asset stock.
22. Apply no correction for partnership Schedule-L undercoverage. No analogous
    S-corporation small-return correction was applied because this lane did not
    verify an authoritative public coverage adjustment; the supervisor should
    confirm that choice.
23. Preserve published rollups but calculate ranges from finest classified
    rows: 126 sole details, 18 partnership sectors, and 75 S-corporation
    details.
24. Retain caution estimates, null combined/deleted values, and null ratios
    with nonpositive receipts.
25. Use 2023 sole-proprietor/partnership tables alongside the latest supplied
    2022 S-corporation table without vintage harmonization.
26. Exclude the finer all-corporation table because it cannot separate S
    corporations.
27. Map `census_bin_hint` conservatively from published labels to the SCF
    seven-bin composition. Mixed sectors remain null rather than receiving a
    forced bin; this is a seam for later crosswalk adjudication, not an
    official NAICS crosswalk.
28. Leave extreme published stock/receipt ratios uncapped.
29. Use no random draws. Both resources are deterministic and marked
    provisional until simulation wiring and crosswalk decisions occur.

## Reproduction and digests

The path-agnostic command recorded in both resources is:

```sh
SCF_2022_SOURCE=/path/to/scf2022s.zip \
SOI_SOURCE_DIR=/path/to/soi-industry-tables \
uv run python tools/build_us_qbi_v3_evidence.py \
  --scf "$SCF_2022_SOURCE" \
  --sole-prop-business-table "$SOI_SOURCE_DIR/23sp01br.xls" \
  --sole-prop-income-table "$SOI_SOURCE_DIR/23sp02is.xls" \
  --partnership-income-table "$SOI_SOURCE_DIR/23pa01.xlsx" \
  --partnership-balance-table "$SOI_SOURCE_DIR/23pa03.xlsx" \
  --s-corporation-table "$SOI_SOURCE_DIR/22co61ccr.xlsx" \
  --all-corporation-table "$SOI_SOURCE_DIR/22co51ccr.xlsx"
```

Input SHA-256 digests:

| Input | Bytes | SHA-256 |
| --- | ---: | --- |
| `scf2022s.zip` | 8,856,125 | `409e6811df895766d50b2f597c10b1b3c5813e7d3e0e45d910ad26c0cb07f4eb` |
| `p22i6.dta` inside the archive | 236,952,250 | `61e2fceb1594e4009eb996d6e25d38a5d8e4874930fc2bfce3c87ffa6946ad0a` |
| `23sp01br.xls` | 113,152 | `bac218b2b56737b01a623700b458c872aa6459767a66d2c5d9fc8d03cd8c75ea` |
| `23sp02is.xls` | 296,448 | `fcc458d3e759a89ef90d3b6e4c8e512dccfe16f03d44f8aa2246bd2d412f5978` |
| `23pa01.xlsx` | 150,076 | `6c8367841c0bb358be2d6509e204ff6e85b6c9827d5ac86166b5149953ca780e` |
| `23pa03.xlsx` | 165,528 | `f7267533b915161b3165e748cfa7a1f374a1b4f22699d4f1b450a7fec6800af2` |
| `22co61ccr.xlsx` | 137,574 | `d0eff09037b41fc25694171043b59b821c0dc8f0e3203eab992740ea30829b98` |
| `22co51ccr.xlsx` | 151,251 | `79bf78d8be1178f7b87e5aa28b31629cce4a5d233a75cfe1d7e0271942d3b1bf` |

Derived resource SHA-256 digests:

| Resource | SHA-256 |
| --- | --- |
| `qbi_employer_structure_v1.json` | `76d849a8da425208a2c615cde9667b493e8ab8beb984b2b685243d7073bc3b76` |
| `qbi_wage_capital_priors_v1.json` | `94431bba2b4fab287857786d87e5a86ef509cd06bd6639dfbbf6dbe0b45950b8` |

Repeated restricted-input builds after both review rounds reproduced the
committed resources byte-for-byte.

## Verification

- Task-specific QBI evidence tests: 17 passed.
- Focused QBI, spec-only, country-package, and retired-reference guards:
  48 passed.
- Full workspace: 3,450 passed, 59 skipped, 6 known warnings in 1,096.65
  seconds.
- `ruff check .`: passed.
- `ruff format --check` on the module, builder, and test file: passed.
- `uv lock --check --offline`: resolved 118 packages.
- JSON parsing, strict resource validators, absolute-path scan, spec-only
  executable-key scan, package declaration checks, and `git diff --check`:
  passed.

Repository-wide `ruff format --check .` identifies 34 pre-existing formatting
differences outside this lane; they were deliberately left untouched.

The six full-suite warnings are existing numerical/runtime warnings in target
support, unemployment-insurance allocation, legacy PUF support parity, and
PyTorch sparse CSR construction. None originates in the QBI evidence code.

## Commit sequence

- `d6de7fc` — start the progress ledger.
- `d061618` — add the legacy SOI workbook reader dependency.
- `27bd8e4` — add SCF employer-structure estimation.
- `dd6efb8` — add SOI wage/capital estimation.
- `254262c` — add the reproducible builder.
- `376ccfe` — emit the real-data resources.
- `1cab5b8` — declare and contract-test the resources.
- `7be0b8c` — resolve provenance and industry-classification review findings.
- `cd3c023` — harden nested schemas.
- `cdbda0d` — close the final adversarial schema-audit gaps.

No network access, push, or pull request was used. The supervisor can push the
local branch and make the listed adjudications before a later lane wires these
provisional priors into the simulation.
