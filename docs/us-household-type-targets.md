# US household type and filing status targets

The default US fiscal compile binds eight national ACS 2024 household cells
and sixteen IRS SOI tax-year 2023 filing-status cells (microcosm#880). The
next certified build consumes them from scratch. These targets address the
excess of married couples with children and the shortage of people living
alone; a change in fit still needs a calibration receipt.

## Sources and bound cells

Chronicle supplies ACS 2024 1-year tables B11001 and B11003 from
`packages/census/acs1_{b11001,b11003}_{national,state}_2024`, with period
`calendar_year: 2024`. The record sets are
`census_acs.acs1_2024.{b11001,b11003}.{national,state}`. Only national estimates
bind. All targets sum household indicators and use family `census_households`,
role `household_type` and materializer `household_type`.

| ACS cell | Indicator | National estimate |
| --- | --- | ---: |
| B11001 001 | Households | 132,737,146 |
| B11001 003 | Married-couple family | 61,980,084 |
| B11001 005 | Other family, male householder, no spouse present | 6,913,098 |
| B11001 006 | Other family, female householder, no spouse present | 16,073,357 |
| B11001 008 | Householder living alone | 38,315,363 |
| B11003 003 | Married-couple family with own children under 18 | 22,978,636 |
| B11003 010 | Male householder, no spouse present, with own children under 18 | 2,960,087 |
| B11003 016 | Female householder, no spouse present, with own children under 18 | 7,732,969 |

The compiler reviews and excludes state estimates because this change adds
national constraints only; Puerto Rico lies outside the US national universe.
Margins of error measure uncertainty rather than levels. The remaining
national estimates either follow from the bound partition or split own
children by age, which remains deferred. B11001 009, nonfamily households not
living alone, follows as 9,455,244 when the five B11001 constraints fit exactly.

Chronicle's `packages/irs_soi/table_1_3` supplies four record sets,
`irs_soi.ty2023.table_1_3.filing_status_{returns,agi,ctc_returns,ctc_amount}`,
with period `tax_year: 2023`. Each binds single, married filing jointly
(including surviving spouses), married filing separately and head of household
for return count, adjusted gross income, nonrefundable child and other
dependent credit return count, and that credit's amount. Existing SOI dollar
aging carries amounts to the build year; counts retain the published values.
These use the existing SOI compiler and tax-unit materializer.

The Table 1.3 all-status returns and AGI cells repeat Table 1.2's bound totals,
so only Table 1.2 supplies them. Table 1.2 in the pinned feed has no CTC count
or amount: Table 1.3's two CTC totals instead remain implied by the four bound
filing-status cells. Published rounding can leave a one-unit difference.

The [feed re-pin procedure](us-chronicle-feed-repin.md) records the source
commit, digest, row delta and reconstruction command. The generated parity
manifest gives each new row either a compiled entry or a reviewed exclusion
with its reason. The 2023 ACS reviewed exclusion remains separate from these
2024 household families.

## Existing constraints and overlap

The baseline Route A `national_state` surface binds 6,102 targets after CMS
Medicaid substitutions, with dollar aging enabled and the optional FSA family
disabled. Its full surface has 33,250 targets; district selection removes
27,148. No existing target counts all households, household types or a
specific filing status.

| Existing record sets | Cells and geography | Source period | Relation to new cells |
| --- | --- | --- | --- |
| `usda_snap.national_average_monthly_households`; `usda_snap.state_average_monthly_households.{maro,mpro,mwro,nero,sero,swro,wro}` | `average_monthly_households`, national and 51 states/DC | FY2024 | SNAP participants only, rather than all households |
| `hhs_acf_liheap.fy2024.national_profile` | `households_served`, `state_programs`, national | FY2024 | Assisted households only |
| `irs_soi.ty2022.historic_table_2.us` and `.state_broad.<postal>` | All-status return counts, national and 51 states/DC | TY2022 | All tax units rather than specific statuses |
| `irs_soi.ty2023.historic_table_2.state_agi.<postal>` | All-status return counts, four high-AGI bands per state/DC, rebased to state controls | TY2023 | Income distribution, rather than filing-status distribution |
| `irs_soi.ty2023.state_2022.us.return_count` and `.eitc_three_or_more_children_returns` | National all returns and EITC returns with three or more qualifying children | TY2023 | All-status total or receipt subset |
| `irs_soi.ty2023.table_1_1`, `.table_1_2.all_returns`, `.table_4_3.all_returns_excluding_dependents` | National all-status return counts, including the Table 4.3 dependent exclusion | TY2023 | No status split; Table 1.2 supplies the duplicated all-status returns and AGI |
| `irs_soi.ty2022.historic_table_2.us` | National all-status CTC claims and amount | TY2022 | Different source vintage from Table 1.3's TY2023 cells; dollars use existing aging |

The new eight ACS cells add household composition constraints; the sixteen
SOI cells split the existing family's universe by status. The two Table 1.3
all-status duplicates stay out of the registry. The older HT2 CTC constraints
remain because their vintage differs from the new cells. This change does
not settle the pre-existing choice to carry those older all-status controls
beside later national SOI tables.

## Materializer and household universe

`us_runtime/household_type_targets.py` reads the person's `person_household_id`,
`A_EXPRRP`, `A_MARITL`, `age` and `is_female` columns and aligns to the household
table's `household_id`. It raises for missing columns, missing parent
households, invalid codes or anything other than one householder per housing
household. It uses observed inputs without a microsimulation.

CPS relationship codes 1 and 2 identify the householder; code 1 declares a
family household. Codes 3 and 4 identify a spouse present. A family with a
spouse is married-couple; other families split by the householder's sex. One
member identifies living alone. An own child has relationship code 5, age
under 18 and marital-status code 7 (never married). The module also provides
nonfamily and child-presence complements for checking partitions. A state
filter, when supplied, reads household `state_fips`.

ACS's native household `TYPEHUGQ` distinguishes housing units (1) from
institutional and noninstitutional group quarters (2 and 3). The ACS loader
retains a one-person household-table placeholder for each group-quarters
resident, recodes their relationship to 14 and replaces zero household weight
with person weight. The indicators exclude both kinds before checking for a
householder. An ACS provenance row missing its type refuses. The ASEC frame
has no native household record-type field; it uses the carried CPS roster and
refuses a missing householder rather than guessing that a malformed household
is group quarters. This does not recover an ASEC group-quarters distinction
that the frame does not carry.

For every valid roster, each result is zero or one. Married-couple family,
other family with a male householder, other family with a female householder
and nonfamily partition households. For each family type, with and without
own children partition its total. State masks partition each national
indicator. Tests compare all eight bound indicators with a second plain
pandas implementation on Hypothesis-generated rosters, including group
quarters and shuffled rows, and require zero or two householders to raise.

## Household total and solver mass

Route A calls `tools/build_us_fiscal_refresh_release.py`. Previously that tool
rescaled base household weights to the Census person-population benchmark in
`_with_base_population_mass_repair`, then passed `mass="conserve"` to all
three solve arms: dense `calibrate`, sparse `calibrate_l0_refit` and
`calibrate_exact_k_ladder` (which passes it through selection and refit).
Their conserved total came from the materialized frame's household weights,
not from a household-count fact. The validated multispine arm preserves its
input importance weights instead of applying that person rescale.

A doc-linked Route A release's recorded mass repair starts at
134,690,323.34024483 households and multiplies by 1.0253031783870294 to reach
138,098,416.6187297, with 334,200,000 people. This differs from the 122.54M
starting total studied in microcosm#1212. Under that Route A conserved total,
the four bound type cells would imply 14,816,514.6187297 households in
B11001 009, rather than ACS's 9,455,244.

This change sets the shared `US_FISCAL_CALIBRATION_MASS` to `FREE_MASS` and
passes it at all three solve calls. B11001 001 binds the housing-unit total.
The initial person-population rescale and national-scale gate still check the
starting frame; their recorded `calibration_mass_policy` now says `free`.
The solver's usual weight caps, fit requirements and release gates still
apply. Changing three calls and binding the total is the smaller change that
also handles group quarters correctly: a single conserved total over all
household-table rows cannot imply a housing-unit count while group-quarters
placeholders carry positive weights. No new conserved total enters the solve.

Route A restores the publisher's `A_EXPRRP` from all three pinned Census
ASEC person archives before relationship recoding (`asec_pool._prepare_year_input`).
The source loader checks complete person-id joins and conflicting carried
values. A nonstandard source that uses the loader's relationship fallback
would inherit its less precise classification of multiperson households;
the certified Route A path does not use that fallback.

The doc-linked staging checkpoint expands to 19,435,668,409 dataset bytes,
above this change's optional cross-check limit of about 8 GB. No full-frame
cross-check, calibration or release ran for this change.

## SOI universe and remaining work

The existing SOI materializer in `_base_simulation_household_columns` filters
by AGI, taxable-only status, filing status, EITC receipt and child count,
itemization and geography. It applies no `tax_unit_is_filer` mask. A national
all-AGI return count therefore counts every tax unit. The [existing state
AGI-band documentation](us-fact-to-target.md#state-agi-bands-bind-as-shares-of-the-state-total-microcosm940)
explicitly describes this behavior, including nonfilers.

Table 1.3 uses that same universe. Adding a filer mask to its statuses alone,
while keeping Table 1.2's all-unit total, would force nonfilers' weight to
zero if the totals fit exactly. Whether the entire SOI family should instead
count filed returns needs a separate change and receipt. The new constraints
do not establish that all tax units are filed returns.

B11007 (households with people 65 and over) and B12002 (marital status by age
and sex) remain follow-ups from microcosm#1212. The calibration receipt and
next certified build remain separate operator work under the
[US release build rule](us-release-build-rule.md).
