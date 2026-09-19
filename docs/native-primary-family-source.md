# Native source household classification

`current_survey_primary_family.classify_primary_household` classifies one complete
original household. `current_survey_primary_family_source.qualify_current_survey_primary_family`
borrows the existing authenticated survey preparation and householder projection,
reads the pinned original members, and returns a detached household table and
descriptive receipt. It grants no source admission, population admission, graph
attachment, calibration, or release eligibility. Future graph integration must
requalify against the live source after relevant I/O and use the maintained exact
householder/source-to-clone transport.

## Definitions and official source mappings

The [2024 ACS subject definitions](https://www2.census.gov/programs-surveys/acs/tech_docs/subject_definitions/2024_ACSSubjectDefinitions.pdf)
define own children as never-married biological, adopted, or stepchildren of the
householder under 18 (printed page 89). A household has at most one primary
family, and its married-couple classification requires the householder and spouse
to reside together (page 90). A related subfamily does not add another family to
that count (page 91). This classifier follows that household concept, rather than
the broader employment-tabulation child universe. The spouse relationship includes
both opposite-sex and same-sex couples.

The [2024 ACS PUMS dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.txt)
maps `RELSHIPP` 20 to the householder, 21/23 to the spouse, and 25/26/27 to
biological/adopted/stepchildren. `MAR=5` establishes never-married status;
`AGEP` supplies age. `TYPEHUGQ=1` identifies housing units and 2/3 group quarters.
`NP` must match the complete retained roster. Spouse and unmarried-partner
relationships appearing together are unresolved. ACS does not publish the general
resident spouse-line pointer needed here to count secondary couples; that result
is explicitly unavailable, even when the primary couple is known.

The [2025 ASEC dictionary](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf)
maps `A_EXPRRP` 1/2 to the reference person, 3/4 to the spouse, and 5 to own
children (printed page 6C-3). Code 6 is unnamed and stays unresolved. `A_MARITL=7`
supplies never-married status; `A_AGE` supplies age. `A_SPOUSE` is a line number
(page 6C-1), resolved reciprocally within `PH_SEQ` using `A_LINENO`. Missing,
out-of-household, self, asymmetric, and marital-inconsistent pointers stay unknown.
All valid resident pairs are counted separately from the primary family.

ASEC has no original `TYPEHUGQ` field. `H_LIVQRT` 1–7 identifies housing units;
8–12 identifies other units (page 6A-2). `HRHTYPE` 9/10 denotes group quarters
(page 6A-3); disagreement between those classifications is unavailable. The
classifier also requires an interview (`H_HHTYPE=1`), `H_NUMPER` equal to roster
size, and the independently qualified reference person. Household-type marriage
codes are a consistency check, never a substitute for spouse evidence.

The [CPS subject definitions](https://www.census.gov/programs-surveys/cps/technical-documentation/subject-definitions.html)
also require never-married status for own children under 18. CPS published counts
may include college-dorm residents associated with a family; this projection
describes retained resident household rosters and does not claim equality with
every CPS published family universe.

## Outputs and unknowns

The household table carries nullable integer columns:

- `census_household_count` and `census_household_population`;
- `census_married_primary_family`;
- `census_married_primary_family_own_child_u18` and
  `census_married_primary_family_own_child_u6` (at least one own child under 6);
- `census_primary_married_own_children_u18_count`;
- `census_household_size_1` through `_6`, and `_7_plus`;
- `census_all_resident_spouse_pairs`, a separately qualified ASEC measure.

Separate household, primary-family, own-child, and spouse-pair status columns
distinguish qualified values, exclusions, and missing or inconsistent evidence.
Known nonhousing units contribute zero to household-family/size indicators;
their pair count is unavailable. Unresolved universe evidence produces unknown
metrics. A known housing unit with an incomplete roster retains a household count
of one, but has unknown size/family metrics and an explicit incomplete-roster
status. Size bins partition the complete qualified housing-unit cohort only.

For a known nonmarried primary household, married-family child measures are zero.
For a married primary family, missing age or marital status of a potential own
child remains unknown. Grandchildren, in-laws, foster children, and unrelated
children do not become own children through age, tax dependency, SPM membership,
or co-residence. Native identifiers are handled as literal strings or exact Python
integers without float conversion.

The receipt records exact readsets, source and householder receipt digests,
source-year origins, output digest, and status counts. Published values may include
Census allocation; no unallocated-observation claim is made. Original source
mutation invalidates the existing preparation, and the qualifier reconstructs
all metrics/statuses after its final owner/source I/O. Returned tables are
detached values; their presence is not retained live authority.

This change adds no targets, tolerances, weighted estimates, status draws, or graph
nodes. It does not establish calibrated household composition or statistical
acceptance against a published table.
