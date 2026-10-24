# US county calibration pilot (2026-10-24)

Tracking issue: microcosm#922, the LWI County Files. That deliverable is US
microdata calibrated to every county-equivalent in its 2023 geography: 3,144,
with Connecticut as its nine planning regions.

This pilot asks whether the published ACS-local release can carry county
distributions. It runs one recalibration with Microcosm's own calibrator on the
real release, adding county target rows from public sources, then measures fit,
weight movement, and per-county effective sample size. It is an experiment, not
a release. The production rung (county rows inside
`tools/build_us_acs_local_release.py`, 2026/2027 periods, ACS county tables as
Chronicle source packages) is separate work.

## Inputs

| Input | Identity |
| --- | --- |
| Release microdata | `policyengine/populace-us` @ `fb8ba90e7f576ad0bb3f29b5aff2d1ec7007e912`, `populace_us_2024_acs_local.h5`, sha256 `769756c3…a7ec` (build `buildo-acs-local-767312d60`, 2026-09-23) |
| ACS 2020–2024 5-year county tables (B01003, B11001, B19001, B19013, B19025, B19051, B19061) | Census API `data/2024/acs/acs5`, `group(<table>)`, `for=county:*`, fetched 2026-10-24; sha256 in `_build_artifacts/lwi-county-20261024/census_api_acs5_2024_county/SHA256SUMS` |
| PEP Vintage 2024 county population | Chronicle `db/data/census/pep_county_2024/co-est2024-alldata.csv` @ `200bf6d5`, sha256 `abcc8720…22b9` |
| IRS SOI county TY2022 | Chronicle `db/data/irs_soi/county_2022/22incyallnoagi.csv` @ `200bf6d5`, sha256 `2c04ff23…3343` |
| CT county-to-town crosswalk | Census `ct_cou_to_cousub_crosswalk.txt`, sha256 `13ad0025…cff8` |
| CT 2020 block assignment, town layer | Census BAF `BlockAssign_ST09_CT_MCD.txt`, sha256 `fe56bf38…b653` |
| 2020 tract to 2020 PUMA | Census `rel2020/2020_Census_Tract_to_2020_PUMA.txt`, sha256 `5262f460…0531` |
| 2020 CT tract population | Census API `data/2020/dec/pl` `P1_001N`, sha256 `2b31267f…aa2` |

## Geography

The release assigns each household a 2020 county inside its PUMA, drawn from
population-weighted PUMA overlaps. Only Connecticut differs between the 2020
and 2023 county sets. Each CT household gets its 2022 planning region from its
2020 tract where it carries one; tracts nest in towns, and towns nest in
regions. Otherwise the region comes from its PUMA and 2020 county.

Every CT (PUMA, 2020 county) cell lies inside one planning region except one:
Tolland County's part of PUMA 20301. That cell holds 12,257 people in 2020;
785 of them (the town of Union) are in the Northeastern region and 11,472 in
the Capitol region. Households in that cell are drawn by those populations with
a fixed seed, the way the PUMA ladder draws counties. All three public county
sources already publish CT as planning regions.

## Target design

Every county row is a within-state share target:

```
target(c) = M_s(w0) * public(c) / sum over counties c' in state s of public(c')
```

`M_s(w0)` is the state total of the measure under the release weights. State
totals stay where the release calibration put them, and the county rows only
redistribute each total across its counties.

| Family | Role | Measure (household weight) | Public source |
| --- | --- | --- | --- |
| `county_households` | train | 1 | ACS B11001_001E |
| `county_population` | train | persons in household | PEP V2024, July 2024 |
| `county_earnings` | train | wages + self-employment (stored inputs) | ACS B19061_001E |
| `cd_households`, `cd_population` | train (hold) | 1, persons | the district's own total under `w0` |
| `county_earning_households` | holdout | earnings ≠ 0 | ACS B19051_002E |
| `county_soi_wages` | holdout | wages (stored input) | SOI A00200 (no Kalawao row) |
| `county_income` | holdout | earnings + stored income inputs (approximate) | ACS B19025_001E |

Solver settings match the ACS-local release: Adam, learning rate 0.02,
conserved mass, `max_weight_ratio` 5 relative to the release weights,
`target_loss_cap` 1. The release weights are the warm start.

The pilot drops the release's other target families (SOI, national and state
admin totals) from this solve. The weight-ratio quantiles in the summary bound
how far the county rows moved records. The production rung trains the county
rows together with every release family, so the release fit is kept.

## Reliability grades (provisional)

Each county is graded by its Kish effective sample size under the calibrated
weights: A ≥ 300, B ≥ 100, C ≥ 30, D below 30. The thresholds are a starting
point for the grading scheme in microcosm#922, not a decision.

## Running

```bash
modal run experiments/us-county-pilot-20261024/pilot.py --step extract
modal run experiments/us-county-pilot-20261024/pilot.py --step calibrate --label run1
```

Tests (sources, CT mapping, share-target invariants):

```bash
cd experiments/us-county-pilot-20261024
uv run --no-project --with numpy --with pandas --with scipy --with pytest \
  --with hypothesis pytest --noconftest -c /dev/null --rootdir . test_county_targets.py
```

## Results

Pending the Modal run.
