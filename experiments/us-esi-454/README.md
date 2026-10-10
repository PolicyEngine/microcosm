# ESI premium stage on the pinned ASEC pools (#454)

`run_stage_on_pinned_pool.py` pools real pinned ASEC inputs the way the base
build's source construction does (`build_pooled_asec_unit_frame`, with the
Census person-column restoration), runs the `meps_esi_premiums` stage, its
signal gate and the release anchor gate, clones the frame for PUF support and
grades the clone, and writes what it measured to `receipts/`.

```
uv run python tools/fetch_us_asec_sources.py --all-pinned   # prints the pinned paths
uv run python experiments/us-esi-454/run_stage_on_pinned_pool.py \
    --asec-h5 2023=<census_cps_2023.h5> --asec-h5 2024=<census_cps_2024.h5> \
    --asec-h5 2025=<census_cps_2025.h5> \
    --census-person-dir ~/.cache/microcosm/cps/asec_education
```

The script refuses an input whose SHA-256 is not its pin
(`asec_sources.ASEC_SOURCE_ARTIFACTS`, `spm_role_source.ASEC_SPM_ROLE_SOURCES`).
About 80 seconds per pool.

| Receipt | Pool |
|---|---|
| `receipts/stage_on_pool_2023_2025.json` | the default pool (`docs/us-asec-source-pins.md`) |
| `receipts/stage_on_pool_2022_2024.json` | the historical Build J/N/P pool |

Each receipt records the input digests, the columns the restoration added per
vintage with their observed code counts, the stage summary (scale factor, raw
and scaled totals, totals by tier and employer sector, weighted policyholder
counts), totals by vintage, both gate verdicts before and after the clone, and
`nhe_concept_check`: the same MEPS-IC cells applied to every current
policyholder, which is how NHE Table 24's broader concept was measured.

These are pre-calibration figures at pooled ASEC weights. They are evidence
about the stage, not about a release:
`packages/microcosm-build/tests/engine_free/us/test_us_esi_premiums.py` pins
the figures the docs and registers quote to these files.
