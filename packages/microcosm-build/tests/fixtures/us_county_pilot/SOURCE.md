# County pilot fixtures

Subsets of the pinned public inputs of `experiments/us-county-pilot-20261024`
(sha256 of each full file in that folder's `pilot.py`, `INPUT_SHA256`), cut to
Connecticut (09), Hawaii (15), Rhode Island (44) and, in the ACS tables,
Puerto Rico (72):

- `acs5_2024_*_county.json`: Census API ACS 2020-2024 5-year county tables,
  rows of those states only, header kept.
- `pep_county_2024_subset.csv`: Census PEP Vintage 2024 `co-est2024-alldata.csv`
  (Chronicle `200bf6d5`), rows of those states.
- `soi_county_2022_subset.csv`: IRS SOI county data TY2022 `22incyallnoagi.csv`
  (Chronicle `200bf6d5`), rows of those states. SOI publishes no Kalawao
  County (15005) row.
- `ct_cou_to_cousub_crosswalk.txt`: Census CT county-to-town crosswalk, whole.
- `ct_block_mcd_first_block_per_tract_town.txt`: the 2020 CT block assignment
  file's town (MCD) layer, keeping the first block of each distinct
  (tract, county, town) triple. The tract-to-region lookup reads only the
  tract and town of each block, so it is unchanged by the cut.
- `ct_tract_puma_2020.txt`: the 2020 tract to 2020 PUMA relationship file,
  CT rows only.
- `ct_tract_population_2020.json`: 2020 PL P1_001N for CT tracts, whole.
