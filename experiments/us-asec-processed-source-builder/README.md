# Processed ASEC builder: differential receipts

Receipts for `microcosm.build.us_runtime.asec_processed_source` and
`tools/build_us_asec_processed_source.py`, Microcosm's builder for the
processed CPS ASEC inputs (`census_cps_YYYY.h5`). The builder replaces the
archived US data package's `CensusCPS` loader, at the loader revision that
produced the pinned 2024 file (`40314a75`, #824) plus the income-year 2025
patch (`9f177ddd`). `docs/us-asec-source-pins.md` describes the builder. This
folder holds the evidence that it reproduces the pinned files.

Everything here needs the gated local inputs listed below, so none of it runs
in CI. CI runs `test_us_asec_processed_source.py` on synthetic archives,
including the golden cases this folder generates.

## Inputs

All inputs were hashed before use, and each equals its pin.

| Input | SHA-256 | Pin |
|---|---|---|
| `asecpub23csv.zip` (income 2022) | `d2e00025…` | `ASEC_EDUCATION_ASSISTANCE_ARCHIVES[2022]` |
| `asecpub24csv.zip` (income 2023) | `cdb39cda…` | `[2023]` |
| `asecpub25csv.zip` (income 2024) | `318845a2…` | `[2024]` |
| `asecpub26csv.zip` (income 2025) | `fe819d0d…` | `[2025]`, added by #1044 |
| `census_cps_2022.h5` | `7ccca976…` | `ASEC_SOURCE_ARTIFACTS[2022]` |
| `census_cps_2023.h5` | `cb578173…` | `[2023]` |
| `census_cps_2024.h5` | `ec36604c…` | `[2024]` |
| `census_cps_2025.h5` | `4c5a3218…` | `[2025]`, added by #1044 |

RECEIPTS_PLACEHOLDER
