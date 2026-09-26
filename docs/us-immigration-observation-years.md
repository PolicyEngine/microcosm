# Immigration rules and source observation years

The immigration rules and controls are adopted from Microcosm #779 at
`7ee36ac1bea125218912996b1030b77a208db963`. The port replaces the earlier
wage proxy with the source `A_LFSR` labor-force predicate and carries the paired
status codec, evidence-constrained humanitarian reconciliation, and existing
composition bands together. Only the immigration stage is adopted from that
revision; other source stages and transfer families keep their current contracts.

`observation_year_column` is an optional keyword on assignment, evidence,
candidate-mask, reconciliation, composition-gate, and ACS-transfer APIs. It names
a numeric person column containing a finite integral calendar year on every row.
Missing columns, missing rows after a join, strings, booleans, fractional years,
and non-calendar values refuse. It does not authenticate a source or establish
which year belongs to it. A native caller must bind the values to retained source
evidence: 2025 for current ASEC demographics and 2024 for current ACS. ASEC income
year 2024 remains distinct from its demographic observation year.

Both ASEC arrival helpers and mixed-source profile arithmetic use the explicit
row year. This matters at cohort boundaries: age 33 and entry midpoint 2007 imply
age at entry 15 in 2025 and 16 in 2024. Assignment, immutable-row reconciliation,
candidate selection/replay, and final composition validation must use the same
convention. The composition gate accepts scalar `time_period` (default 2024),
matching assignment. Original ages and the PEINUSYR 27/28 approximation are
unchanged. Omitting the keyword retains the adopted rules' scalar convention;
finding a column with a suggestive name does not enable it.

For paired ACS transfer, the column must exist on both frames and must survive
the donor projection. The three evidence predictors remain citizenship,
raw origin code, and approximate entry year, alongside the existing age/sex/state requirements. The
observation year is not added as a QRF predictor and does not change pattern
seeds. Rows without arrival evidence use a constant zero in the feature frame,
so the ASEC/ACS observation-year difference cannot leak through NIU entry values.
The rule profile's cohort arithmetic is unchanged. The execution contract version
4 binds the feature version, NIU value and complete ASEC approximation table;
paired checkpoint patterns independently bind the same contract so even a caller
without an outer bank identity cannot resume stale feature semantics.
The year column name and ordered typed person-ID/year pairs on both
frames enter the immigration checkpoint pattern identity. Existing bank
validation refuses stale patterns and rebuilds them. A caller-written outer bank
identity cannot bypass that check. Non-immigration patterns are unchanged.

Reconciliation receipts also record ordered resolved-weight and mutable-mask
digests. `us_immigration_humanitarian_transfer_selection_masks` optionally accepts
`reconciliation_receipt` and refuses changed weights, mask or seed. This checks
the numerical selection inputs; the receipt does not authenticate source evidence
or controls. Explicit-row-year receipts use `time_period: null` and retain their
separate observation-year declaration rather than claiming scalar 2024.

The published arrival documentation has a material inconsistency. The standalone
[2025 ASEC dictionary, printed 6C-6](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf#page=27)
lists PEINUSYR 28 as 2022–2024; the
[2025 technical documentation, printed 6C-6](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar25.pdf)
lists 2022–2025. Both enumerate codes 1–28, with no 29, despite a stale 0:26
header. The retained value 2023 is an explicit midpoint/floor-midpoint
approximation compatible with either interval; it does not resolve their endpoint
disagreement or prove an exact entry year or statutory cohort. Raw code 28 and
this unresolved interval must remain visible at any later source-admission step.

The technical documentation's Appendix J identifies 459 as Zaire (DR Congo),
and 461 as Zimbabwe. The
[2024 ACS Place of Birth list](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/code_lists/ACSPUMS2024CodeLists.xls)
identifies 451 as Sudan only; South Sudan is inseparable in public code 464 with
Tunisia and Western Sahara. The rules do not add that entire residual group.
The complete ACS and ASEC code domains are not interchangeable; matching active
named-country selectors does not create a general lossless crosswalk.

ACS YOEP is also not universally an exact year: the
[2024 PUMS dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.txt)
uses 1938 for 1938 or earlier and 1939 for 1939–1944. The reported year concerns
the most recent move to live in the US, not verified first entry or continuous
residence. Complete YOEP/CIT domain and source qualification remain separate from
this legacy rule helper; its permissive numeric input checks do not admit native
source literals.

These are rule interfaces, not a native status owner or a release certificate.
The separate full-original-ASEC source view must remain available without
assigning statuses. Later native assignment must reject unauthenticated existing
status pairs, qualify owning tokens before legacy coercion helpers, use original
ASEC household design weights once, and preserve original identities. Later ACS
reconciliation requires an authentic complete-source selection and the receiving
population's allocated pre-clone weights. Candidate exhaustion remains an error.

The imported control values are unchanged mixed-date imputation assumptions,
including broad unauthorized estimates, cumulative admissions, trailing flows,
TPS registrations, and an explicit-zero withholding assumption. They are not
measured legal-status stocks for a common calendar year. This port does not admit
those assumptions for a native release, update targets or tolerances, enable
native assignment or graph nodes, or qualify an actual export. Source-owner
admission, full-source reconciliation, no-worse comparisons, and final engine
and export checks remain separate work.
