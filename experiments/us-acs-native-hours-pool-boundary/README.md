# Native ACS usual hours at the pool's operator boundary

`run_acs_input_path_on_pinned_archives.py` makes the three calls
`tools/build_us_multispine_pool.py` makes on its ACS arm before spine assembly
(`build_acs_pums_unit_frame`, `map_acs_native_inputs`,
`assert_operator_free_source_frame` with the native-input receipt), on the real
ACS 2024 1-year PUMS archives. It refuses an archive whose SHA-256 is not the
one in the checked-in ACS source manifest. It builds no pool and writes only
its receipt. About 25 minutes on a busy machine; nearly all of it is the
archive read.

```
uv run python experiments/us-acs-native-hours-pool-boundary/run_acs_input_path_on_pinned_archives.py \
    --archive-dir ~/.cache/microcosm/acs-pums/2024-1yr \
    --asec-h5 ~/.cache/microcosm/asec/census_cps_2025.h5 \
    --out experiments/us-acs-native-hours-pool-boundary/receipts/<name>.json
```

`--repo-root` selects the checkout whose tool is loaded and the interpreter
selects the `microcosm` sources, so the same script measures two checkouts.
Each receipt records both paths and the checkout's commit.

| Receipt | Code | Boundary |
|---|---|---|
| `receipts/origin_main_7639ef8b0.json` | origin/main at 7639ef8b0 | refuses the frame |
| `receipts/fix_56bc5e0b8.json` | this change (56bc5e0b8) | admits the frame |

## What the receipts show

On origin/main the boundary stops a pool build from the pinned archive before
assembly:

```
ACS native-mapped stacked input native_inputs['weekly_hours_worked_before_lsr']
is not a declared ACS native mapping output.
```

The usual-hours receipt is the same in both. Of 3,422,888 ACS person rows:

| Rows | Count |
|---|---|
| `WKHP` observed (1 to 99) | 1,767,132 |
| Blank `WKHP`, `WKL` 2 or 3: source-confirmed zero | 1,105,858 |
| Blank after mapping | 549,898 |
| of which under 15 | 510,098 |
| of which aged 15 | 39,800 |
| of which 16 and over | 0 |

Every cell the mapping leaves blank is outside the ACS usual-hours universe
(16 and over). Those are the cells the pool's ASEC-to-ACS gap fill covers; the
other 2,872,990 reach the pool as measured.

## Battery preview

`battery_preview` applies the by-origin battery's own functions to the
positive leg of usual hours: ACS native against the public Census ASEC 2025
file. It is a preview, with the caveats the receipt lists: no pool was built,
ACS blanks count as zero hours, and the ASEC file is not the pool's pooled
raw-stage artifact.

| | ASEC | ACS | Battery bound |
|---|---|---|---|
| Weighted share with positive hours | 51.9% | 53.9% | ratio 1.04, allowed 0.8 to 1.25 |
| q10, q25, q50, q75, q90 of positive hours | 20, 40, 40, 40, 50 | 20, 35, 40, 40, 50 | envelope distance 0.13, allowed 0.25 |
