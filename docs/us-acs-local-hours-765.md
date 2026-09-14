# Usual hours in the retained ACS local build

Issue [#765](https://github.com/PolicyEngine/microcosm/issues/765) concerns
`weekly_hours_worked_before_lsr`, the usual-hours input derived from ASEC
`HRSWK`. The separately transferred `hours_worked_last_week` measures the
reference week and is not a substitute.

The ACS loader now retains `WKHP`, `WKL`, and `FWKHP`. Native `WKHP` takes
priority. The [2024 FTP dictionary, page 45](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.pdf#page=45)
defines this as past-year usual weekly hours, codes 1–98 and topcode 99.
Its NIU value is blank; the [Census API](https://api.census.gov/data/2024/acs/acs1/pums/variables/WKHP.json)
instead uses zero for NIU. The CSV mapper rejects that API zero code.
It maps a blank to zero only where `WKL=2/3` confirms no work in the past
year, with contradictory universe evidence rejected. `AGEP<16` establishes
survey-universe absence, not measured nonwork: those blanks remain unresolved.
Eligible or unknown-universe blanks also remain unresolved; an absent `WKHP`
column remains absent. Current employment status does not zero annual hours.

`FWKHP` remains alongside the raw value. The native receipt separately counts
source values, structural zeros, source-universe-unavailable rows,
Census-allocated values and values with an unknown allocation flag.
A Census-allocated source value remains source data;
it is not described as an original respondent answer. `hours_worked_last_week`
is not produced by this mapping and remains a separate transfer input.

The local staging builder loads a certified donor directly, without the
fiscal refresh builder's source operator chain. Only if native mapping leaves
unresolved ACS hours does a lazy donor factory qualify complete raw ASEC
hours fields and run the existing `with_us_hours_worked_inputs` producer on
the ASEC observation role at source age 15 or above. The raw `A_AGE` is used
when available; otherwise the role's mapped `age` establishes its universe.
Younger or unknown-age ASEC rows are excluded, counted, and never filled from
their `HRSWK=0` NIU code. A fully native ACS spine does not open or require
those raw donor fields. Only missing ASEC usual-hours cells are attached to
the continuing base, by person ID. Existing ASEC values must agree with the
raw derivation when qualification is needed; they are preserved. The
temporary `weeks_worked` output is not added to the donor or transfer plan.

Tax-detail transfer retains its default PUF role. A separate hours-only pass
uses the qualified ASEC donor, preserving legitimate PUF hours that differ
from original ASEC hours. Source-incomplete ASEC rows or disagreements between
ASEC observations and their raw fields fail qualification. Unclassified or
new multispine donors require their own source-lineage contract; this helper
targets the retained legacy ASEC/PUF roles. Native qualification of the actual
published donor remains pending.

The [Census 2024 ASEC dictionary](https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asec2024_ddl_pub_full.pdf)
declares integer `HRSWK` codes 0–99, `A_HRS1` codes −1–99 and `WKSWORK`
codes 0–52. The raw donor qualification enforces these domains. `HRSWK=99`
retains its topcoded 99+ interpretation. `A_HRS1=-1` is outside its universe;
the existing producer maps that sentinel to zero in its temporary output.
This usual-hours repair does not overwrite the separate last-week input.
At known source age 15+, the dictionary routes nonworkers outside `WKSWORK`
and `HRSWK`; coherent zero codes represent that routing. Positive weeks must
have positive usual hours, and vice versa. `WORKYN`, when retained, must agree
with those codes; it is not a new mandatory source column.

An age-15 ACS recipient can use the qualified age-15+ ASEC fallback. Younger
unresolved recipients fail before fitting by default. A caller can explicitly
select `us_hours_under15_zero_completion_v1` through `hours_under15_policy`
or the staging CLI's `--hours-under15-policy`. This separately reviewed
tax-benefit input assumption sets only unresolved ages 0–14 to zero. It does
not estimate that every child actually works zero hours. It refuses unknown
ages or contrary positive work or wages, or nonzero self-employment evidence, and never
overwrites observed hours, including 0, 40, and 99. Missing earnings remain
unknown and are counted, not treated as evidence of no earnings.
Self-employment losses also refuse this optional assumption. A net loss does
not necessarily prove current work, but the zero-hours assumption cannot
settle that ambiguity. Unknown-earnings counts prefer raw `WAGP`/`SEMP` when
present, even if a mapped input is zero; the receipt identifies raw, mapped,
or unavailable earnings evidence explicitly.

The completion receipt has `provenance=modeled_assumption`, a policy version,
affected age counts, unknown-earnings counts, and before/after missing counts.
The original native receipt retains its missing and source-universe counts.
Coverage reconciles native, modeled, and fallback inputs without relabeling
modeled zeros as source observations. This policy is disabled by default.

The existing conditional transfer fills only unresolved usual-hours cells,
preserving native values and structural zeros. A fully native receipt satisfies
the local hours coverage contract without a fabricated fit receipt. The general
shared transfer plan remains unchanged.
Source identities, target coverage, fit weights and imputation provenance use
the existing staging checks. A value of 40 or 0 is not a missingness flag.

Before writing the pooled staging file, before materialization and export,
and again during finalization and packaging,
the local hours gate requires complete finite inputs in [0, 99] and signal in
each spine. It also inspects the source-null audit: consumer default filling
cannot turn an unresolved hours cell into a passed transfer. Per-spine counts
at 0 and 40 are diagnostics, not rules that individual values are invalid.
This prevents national signal from hiding an ACS-wide omission.

These checks establish coverage, not conditional model quality or a release.
A dense successor still needs native source/join validation, held-out validation
of any fallback, work-requirement
counterfactuals, and comparisons against the live data on the same protected
national/state/CD evidence. Existing default-filled H5 files require their
original missingness/source provenance and a separately reviewed repair
contract. The sparse SPM enrichment parent and certificate are unchanged.
