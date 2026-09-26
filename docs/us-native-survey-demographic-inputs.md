# Native survey demographic inputs

The enrichment host offers two independent options, both disabled by default:
`demographic_inputs=True` binds `is_female`; `race_hispanic_inputs=True` binds
`cps_race` and `is_hispanic`. Each fragment borrows the original survey
preparation, binds values once per original person, and copies them through the
existing clone mapping. They do not add people, change weights, redraw values,
or assign geography.

The race/Hispanic fragment captures source literals from the same pinned ASEC
person member and ACS person archive authenticated by the retained preparation.
It checks the complete captured members, source identities, original key roster
and final parent identity. Its projection is descriptive evidence, not a new
source credential. The host retains it and requalifies it after relevant I/O.

## Source definitions

`cps_race` represents the published CPS `PRDTRACE` categories. ASEC supplies
those codes directly. Hispanic origin uses the direct `PEHSPNON` response with
the `PRDTHSP` detail-universe crosscheck. The fragment preserves `PXRACE1` and
`PXHSPNON` allocation histories. These observations come from survey 2025;
their associated income cohort is 2024. See the [2025 ASEC dictionary](https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asec2025_ddl_pub_full.pdf), PDF pages 27–28 and 31–32.

ACS 2024 Hispanic origin maps directly from `HISP`. Race uses `RAC1P` together
with `RACWHT`, `RACBLK`, `RACAIAN`, `RACASN`, `RACNH`, `RACPI`, `RACSOR`, and
`RACNUM`. The fragment checks agreement between the broad category, detailed
indicators and number of major groups. Native Hawaiian and other Pacific
Islander indicators form one major group. All combinations of the five groups
represented in CPS map to its published categories, including its residual
three-race and four/five-race codes. `FRACP` and `FHISP` preserve allocation
status. See the [2024 ACS dictionary](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.pdf), PDF pages 63, 106, 111, 127 and 129.

“Some Other Race,” alone or in combination, has no exact CPS counterpart here.
It remains unresolved. A missing, malformed, unlabelled, contradictory or
allocation-unresolved source item also remains unknown. Allocation histories
that end without a published value do not establish a known response. Published
allocated values are retained with their allocation status; they are not
described as unallocated responses. No unknown is replaced by false or zero.

## Graph inspection

The visible `survey_race_hispanic` fragment has source, raw-column, canonical
binding and clone-attachment nodes. Source literals, observation year,
allocation and mapping-status columns accompany the canonical nullable outputs.
The graph's private source artifact retains source evidence without exposing
raw native person keys in public node parameters. A missingness mask belongs to
the canonical column; mapping status explains why a value is unresolved.

The older `acs_release_predictors` mapping has a different contract: it uses
representative CPS codes for particular coarse donor-model categories. That
equivalence is not an exact source observation and is not reused here.

The maintained SCF wealth model also uses an internal feature named `cps_race`
with its own coding. Its recipient helper maps known Hispanic origin to 3 and
non-Hispanic White/Black/Asian/Other to 1/2/4/7. Those are model-feature categories,
not canonical PRDTRACE. A future SCF graph must call or reproduce that explicitly
declared consumer transformation from canonical race plus Hispanic origin; it
must not feed the canonical 1–26 codes directly into that feature. This fragment
does not implement or change SCF consumption.

## Possible future conditional harmonization

An additional explicitly modelled operation could draw a CPS-compatible category
for an unresolved ACS person, conditional on reviewed common predictors and
the observed ACS category. Such an operation would need a scientific mapping
convention, authenticated donor population, model/seed declaration, consistency
constraints, validation and distinct imputation provenance. Original ACS
literals and unresolved-source status would remain available. Model output
must not overwrite the historical source claim or be relabelled an exact
crosswalk. This fragment implements no such convention or imputation.

Tiny invented-source tests exercise capture, all supported category combinations,
unknowns, exact clone transport and cache replay. They do not certify national
data completeness, source comparability, a full enrichment host or release
eligibility. Native ACS required/default loader APIs remain unchanged.
