# ACS SPM field regrouping

`microcosm.build.acs_spm_regroup.regroup_acs_spm_units` is a pure table
transformation for the twelve SPM fields in the reviewed dense Build P parent.
It implements the explicit development policy
`acs_spm_conservation_regroup_v1`. It does not construct membership, run a
country model, recalibrate weights, or establish release quality.

The release caller must authenticate the parent and original source-null
register before creating `AcsSpmLegacyDefaults`. The helper checks the supplied
hash formats and field values; it does not open those artifacts or establish
their authenticity. The caller also supplies complete household rosters,
the accepted partition and a persisted, globally collision-checked numeric
identity map. No identifiers are allocated inside a chunk.

## Inputs and identity

The input person table supplies `person_id`, `person_household_id`,
`person_spm_unit_id`, raw `AGEP` and raw `RELSHIPP`. The household table supplies
`household_id`, `TYPEHUGQ`, `NP` and raw `TEN`. The proposed membership table
covers exactly the input people and supplies `new_spm_unit_id` and
`new_spm_unit_source_id`; optional `canonical_spm_unit_id` labels must be
complete and coherent within every new unit, with one numeric unit per
canonical label. The crosswalk retains these
canonical labels and both old and new source identities.

An unchanged member set retains its old numeric and source IDs. A split
component's new source ID is a generated identity, not a Census-provided
resource-unit identifier. Generated source IDs cannot collide with old origins
or other components in the supplied scope. Global validation across chunks
belongs to the caller. Whole-household chunking with one preallocated global
map gives the same results as processing those households together.

The helper refuses incomplete ACS household rosters, old-unit merges, changed
non-ACS units and changes to group-quarters membership. Group-quarters tenure
remains source-unavailable, with the existing SPM placeholder preserved. This
does not make group quarters part of the poverty measurement universe.

## Field treatment

| Fields | Treatment |
| --- | --- |
| `spm_unit_id`, `spm_unit_source_id` | Use the supplied identity map; keep old origin in the crosswalk. |
| `spm_unit_support_channel`, `spm_unit_support_clone_index`, `spm_unit_spine` | Retain old support provenance. A generated source ID is labeled separately. |
| `spm_unit_pre_subsidy_childcare_expenses` | Preserve unchanged units; allocate a split unit's positive imputation exactly once only under the rule below. |
| `takes_up_housing_assistance_if_eligible` | Inherit the old boolean under `acs_spm_housing_common_imputation_v1`. |
| `receives_housing_assistance`, `spm_unit_energy_subsidy`, `takes_up_tanf_if_eligible`, `takes_up_snap_if_eligible` | Preserve authenticated legacy defaults on ACS successors. Require zero for the default energy amount. |
| `spm_unit_tenure_type` | Explicitly select preservation or the native rent-free recode below. |

Any additional SPM field is rejected until it has a treatment. The function
returns only a new SPM table and audit tables. It does not write to persons,
households, tax membership, rent carriers, geography, income or weights.

The four default fields were registered source-null across ACS SPM units in
the Build P manifest and filled by the legacy finalizer. They are not four
observed nonreceipt measures or authenticated latent participation draws.
Unexpected nondefault or missing cells refuse this bounded treatment.

The ACS housing take-up boolean was transferred to the old SPM unit. Repeating
it across successors is an explicit inherited-imputation/common-shock
assumption, not observed participation of the new units. Housing dollars can
change when a country model uses the new composition even though no monetary
housing field is copied here. A release comparison must inspect this effect
and preserved person-level rent and utility carriers.

## Childcare conservation and exceptions

Under `acs_spm_childcare_unique_successor_v1`, a positive old imputation is
carried in full to the household reference person's successor only when that
is the unique successor containing potential childcare recipients. Other
successors receive an explicitly labeled allocation zero. The emitted cells
are checked to sum exactly to the original amount, with exactly one positive
carrier. This conserves the existing imputation for an unchanged recipient
subpopulation; it does not turn that imputation into an observed expense.

For this conservative allocation screen, ages 0–15 are potential recipients.
Ages 16–17 require caller-provided nullable boolean `older_care_evidence`,
indexed by person ID. True adds a potential recipient; false rules out that
older-minor care channel for this screen; absent evidence remains unknown.
This is not a benefit eligibility rule. The caller must record how any care
evidence was established, rather than treating a filled model default as an
observed negative.

The conservative age choice reflects the 2024 ASEC questionnaire universe
(`HCHCARE_YN`: age 15 and under), while published SPM descriptions sometimes
say under 15. The model's existing expense-to-person carrier uses `is_child`
(under 18). Explicit older-minor screening avoids silently assuming those
cases are out of scope. [2024 ASEC documentation, printed 6A-9 and questionnaire
Q95/CCAMT](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar24.pdf).

Multiple eligible successors, no potential recipient, a unique recipient
outside the reference unit, or unknown age/care evidence leaves every
successor childcare cell unavailable. The original amount appears in the
typed `childcare_ledger.unresolved_amount` and the reason in `exceptions`.
Missing original amounts remain missing. Zero old imputations can be inherited
as labeled imputed zeros. `require_complete()` refuses an unresolved ledger;
a complete ledger still does not establish release quality.

Conservation is a bounded alternative to refitting all childcare imputations.
Official ACS SPM research estimates expenses at the SPM-unit level. The
exception cases therefore need targeted reconditioning or another declared
allocation policy before a complete candidate is possible.
[ACS SPM methodology, childcare discussion](https://www.census.gov/content/dam/Census/library/working-papers/2015/demo/SEHSD-WP2015-09.pdf).

## Rent-free tenure sensitivity

The caller must select `preserve_parent_tenure_v1` or
`acs_ten4_no_mortgage_v1`. Both validate the authenticated parent's old mapping
against native `TEN`. The latter maps raw `TEN=4` (occupied without payment of
rent) to the SPM no-mortgage threshold category. The existing engine enum is
named `OWNER_WITHOUT_MORTGAGE`; this recode does not claim that a rent-free
occupant owns the home. Raw household tenure and rent inputs remain unchanged.

This legacy mapping is an artifact comparison contract, not a replacement for
the current source mapper. The output records the policy and recoded units so
partition-only and partition-plus-tenure outcome comparisons remain separable.
[2024 ACS dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.pdf),
[ACS SPM dictionary, SPM_TenMortStatus](https://www2.census.gov/programs-surveys/supplemental-poverty-measure/datasets/spm/spm-asc-data-dictionary.pdf).

## Verification boundary

Invented fixtures cover single allocation, absent/ambiguous care, typed
missingness, unchanged and group-quarters preservation, identity collisions,
native tenure sensitivity, row-order invariance, complete-household chunking
and unchanged non-SPM inputs. No native payload or country calculation is
needed by these tests. Native field qualification, full-map integrity,
regrouping sensitivity and matched national/local outcome comparisons are
separate release requirements.
