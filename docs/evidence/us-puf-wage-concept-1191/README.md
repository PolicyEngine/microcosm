# PUF-channel wage concept audit (microcosm#1191)

Issue #1191 proposes binding the SOI and CBO wage targets to
`irs_employment_income`. Its recommendation waited on one audit: whether builds
from current main store PUF-channel wages net of elective deferrals. This folder
holds the scripts and receipts of that audit. The report is the audit comment
on the issue. Nothing here changes a build.

## Result

- The pipeline stores the finalized PUF QRF wage with no deferral added; the
  archived generator assigns that donor wage from E00200. 90.8% of earning
  PUF-clone tax units store a wage total on the generator's 2015-to-2021
  rounding grid.
- The engine subtracts the separately imputed 401(k) deferral from that wage:
  $172.2B on the PUF channel of the 30 September Route A release, 2.78% of its
  wages.
- Four stored PUF outputs fit their archived 2015-to-2021 factors and neither
  the 2022 nor the 2023 factor. The archived generator contains a later-year
  uprating loop whose body never runs.

## Files measured

| File | sha256 | Built |
|---|---|---|
| `base_populace_us_2024_puf_support.h5` | `0fb05b674899114e66dc01c2df0ff5349de88bbc8fed440755f4b19f287cedd2` | at `4b57d15a287c` (contains #1033) |
| `populace_us_2024.h5`, release `populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z` | `417d23aed4d044de2e657a7b5d856136975d25739dbdb4ddd72866f841de1559` | from that base |
| `populace_us_2024.h5`, certified default `populace-us-2024-spm-20260915` | `6496cc4393d4d3c6574f76eca231de5898c803b9067645591fd5c4d3e65aee84` | before #1033 |

The PUF donor file itself (`puf_2024.h5`, irs-soi-puf 1.8.0, sha256
`7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df`) was not
opened.

## Scripts and receipts

| Script | What it measures | Receipts |
|---|---|---|
| `scripts/raw_h5_checks.py` | The issue's stored-column checks, unchanged | `raw_h5_checks.*.txt` |
| `scripts/puf_wage_audit.py` | Wages and deferrals by channel and clone; the PUF wage against the same person's ASEC wage at person and tax-unit grain; wage distributions; the engine's pre-tax subtraction rebuilt from stored columns; which columns differ between base frame and release | `puf_wage_audit.*.json` |
| `scripts/puf_grid_check.py` | Whether PUF-clone tax-unit totals are whole source dollars times the archived generator's uprating factor | `puf_grid_check.*.json` |
| `scripts/puf_grid_by_deferral.py` | The same test split by whether the unit holds a deferral | `puf_grid_by_deferral.*.json` |
| `scripts/engine_wage_by_channel.py` | Engine wage concepts by channel (policyengine-us 2.2.1), and a per-person comparison of the engine's pre-tax amount and taxed wage with the same amounts rebuilt from stored columns | `engine_wage_by_channel.*.json` |
| `scripts/run_archived_uprating_loop.py` | The archived generator's 2021-to-2024 uprating loop, run verbatim on the archived factor table | `run_archived_uprating_loop.txt` |
| `scripts/gross_up_reference.py`, `scripts/test_gross_up_properties.py` | Standalone reference semantics of the proposed PUF-half gross-up and Hypothesis property tests of it; they do not exercise the build | `property_tests.txt` |

`puf_grid_check.py` reads `soi_targets.csv` from the archived
policyengine-us-data commit `42ed5d45c56df80d754fbe24cce21cfeb8d05cbe`
(`policyengine_us_data/storage/calibration_targets/soi_targets.csv`, sha256
`0d60ddbe0b2c0a00d8c0521ab1693e51487595fb941314e114ec52901bba04bb`) and applies
that commit's `datasets/puf/uprate_puf.py::get_growth`. `run_archived_uprating_loop.py` executes that
commit's `policyengine_us_data/utils/uprating.py` (sha256
`688a46f81e81788ce0c92a8a476addf66afe7e6a259af9f895e688f9e36fc77a`).

## Reproduce

```bash
uv sync --all-packages --extra us
S=docs/evidence/us-puf-wage-concept-1191/scripts
uv run python $S/raw_h5_checks.py BASE.h5
uv run python $S/puf_wage_audit.py BASE.h5 out.json --same-rows-as RELEASE.h5
uv run python $S/puf_grid_check.py BASE.h5 soi_targets.csv out.json
uv run python $S/puf_grid_by_deferral.py BASE.h5 1.187079295246555 out.json
uv run python $S/engine_wage_by_channel.py RELEASE.h5 out.json  # also on the certified file
uv run python $S/run_archived_uprating_loop.py uprating.py /tmp/uprating-out
uv run pytest $S/test_gross_up_properties.py
```

## Limits

- The grid test establishes lattice membership. Confirming that stored totals
  equal donor values, and comparing quantiles with the donor, require the donor
  file.
- The property tests sit outside the repository's `testpaths` and do not run in
  CI.
- The base build's stage checkpoints were not available, so the finished frame
  was tested and the stages were read in code.
- No build and no recalibration were run. Sums that combine columns are
  fixed-weight arithmetic on the named file.
