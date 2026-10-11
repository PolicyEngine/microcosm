# PUF donor uprating audit

Evidence for the finding that the processed PUF donor Microcosm US reads carries each amount at the level the archived generator's 2015-to-2021 step left it, with no 2021-to-2024 step. The findings, the cost on a release and the proposal are in the issue this folder is linked from. No build behavior changes here.

## Files measured

| | File | sha256 |
|---|---|---|
| Build donor | `puf_2024.h5`, read from the Modal volume `microcosm-us-stage-inputs` | `7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df` |
| TY2015 source | `puf_2015.csv`, same volume | `0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df` |
| Published release | `puf_2021.h5`, PolicyEngine/irs-soi-puf 1.8.0 | `1db51c815240ee5d41c62ae37f3bb9b815422ccc07c1790362249682f3902e45` |
| Published release | `puf_2024.h5`, PolicyEngine/irs-soi-puf 1.8.0 | `8182579ddfecaf5e5b872e2307b88f03e8e8def993171b648f701a19a847f37b` |
| Base frame | `base_populace_us_2024_puf_support.h5`, Route A, built at `4b57d15a2` | `0fb05b674899114e66dc01c2df0ff5349de88bbc8fed440755f4b19f287cedd2` |
| Base summary | `base_populace_us_2024_puf_support.summary.json` | `97a3db656a65c095aa985ed0b07969c2eb06790c1340e4d168d4d40137bcfb10` |
| Release | `populace_us_2024.h5`, `populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z` | `417d23aed4d044de2e657a7b5d856136975d25739dbdb4ddd72866f841de1559` |
| Target-frame checkpoint | `target_frame_checkpoint.h5`, same release | `8758635b4bf60aae62c990d36b2604fbcc1b9e2b12a33f47ba33a68898cdd28c` |
| Calibration weights | `populace_us_2024_calibration.npz`, same release | `d1403f75c549f030dcc13c2df3c2e4f7365b67105caa1bb0440e9ab8683f2d55` |
| Calibration diagnostics | `calibration_diagnostics.json`, same release | `64b55a02aaf573964cffee44d4a86bd63f33dc30c7f5012e4345348eecc06f80` |

The four PUF files are restricted IRS microdata. They were read from a local folder outside every repository, which was removed when the audit closed. Every receipt here is an aggregate: counts, shares, factors and weighted totals.

## What the folder holds

- `scripts/puf_donor_factor_audit.py`: runs the archived `uprate_puf` unchanged on the TY2015 source for 2021, 2022 and 2023, evaluates each donor column's archived source expression under each, and reports the share of records that match. Also compares the published `puf_2021.h5` with `puf_2024.h5` array by array, classifies the 82 columns of the donor table Microcosm's own loader builds, sets the donor's weighted totals beside the SOI actuals, and checks that TY2015 ordinary and qualified dividends can be recovered from the donor's two dividend columns.
- `scripts/release_puf_channel_cost.py`: splits every calibration target that binds a PUF-sourced variable into its ASEC, PUF-clone and capital-gains-tail parts at design and at final weights, from the release's target-frame checkpoint and calibration weights, and checks the parts against the estimates the build recorded. Also totals each donor column as the frame stores it, by arm, for comparison with the donor's own totals.
- `scripts/issue_tables.py`: renders the issue's tables from the two receipts.
- `scripts/run_archived_uprating_loop.py`: the archived generator's 2024 uprating loop, run verbatim (the same script as in `docs/evidence/us-puf-wage-concept-1191/`, PR #1195), with its output in `receipts/run_archived_uprating_loop.txt`.
- `archived/`: three files of policyengine-us-data at `42ed5d45c56df80d754fbe24cce21cfeb8d05cbe`, byte for byte (`git show 42ed5d45:policyengine_us_data/<path> | shasum -a 256` gives the hashes below). The two Python files carry a `.txt` suffix so that formatters leave them alone.

  | Archived path | Copy | sha256 |
  |---|---|---|
  | `datasets/puf/uprate_puf.py` | `uprate_puf_42ed5d45.py.txt` | `ab806fdd9f4c05366e1859150b082b71ece9d1666e2d292c2266a335df105e9b` |
  | `utils/uprating.py` | `utils_uprating_42ed5d45.py.txt` | `688a46f81e81788ce0c92a8a476addf66afe7e6a259af9f895e688f9e36fc77a` |
  | `storage/calibration_targets/soi_targets.csv` | `soi_targets_42ed5d45.csv.gz` (hash of the uncompressed file) | `0d60ddbe0b2c0a00d8c0521ab1693e51487595fb941314e114ec52901bba04bb` |
- `receipts/puf_donor_factor_audit.json`; `receipts/release_puf_channel_cost.json` (national rows and summaries) with `receipts/release_puf_channel_cost_targets.csv.gz` (all 2,955 rows); `receipts/issue_tables.md`; `receipts/property_tests.txt`; `receipts/mutation_check.txt`.
- `proposal/rebase_reference.py`, `proposal/test_rebase_properties.py`: a standalone statement of the proposed rebase and Hypothesis tests of its invariants. They sit outside `testpaths` and do not run in CI. `proposal/mutation_check.py` seeds nine faults in the reference and checks that a test catches each (`receipts/mutation_check.txt`).

## Reproducing

With the restricted files in `$R`, the Route A run folder in `$S` and the release folder `$S/release-out/<release id>` in `$REL`, from a checkout of the build commit with its environment (policyengine-us 2.2.1):

```bash
python -I scripts/puf_donor_factor_audit.py \
  --donor $R/modal/puf_2024.h5 --source-csv $R/modal/puf_2015.csv \
  --release-2021 $R/gh-1.8.0/puf_2021.h5 --release-2024 $R/gh-1.8.0/puf_2024.h5 \
  --archived-dir archived --microcosm-root <checkout at 4b57d15a2> \
  --summary-json $S/base-out/base_populace_us_2024_puf_support.summary.json \
  --out receipts/puf_donor_factor_audit.json
```

```bash
python -I scripts/release_puf_channel_cost.py \
  --checkpoint $S/release-checkpoints/target_frame_checkpoint.h5 \
  --calibration-npz $REL/artifacts/populace_us_2024_calibration.npz \
  --diagnostics $REL/releases/<release id>/calibration_diagnostics.json \
  --release-h5 $REL/artifacts/populace_us_2024.h5 \
  --base-h5 $S/base-out/base_populace_us_2024_puf_support.h5 \
  --out receipts/release_puf_channel_cost.json
```

```bash
uv run pytest docs/evidence/us-puf-donor-uprating/proposal/test_rebase_properties.py
```

```bash
uv run python docs/evidence/us-puf-donor-uprating/proposal/mutation_check.py
```

The donor audit takes about a minute and 6 GB; the release decomposition about 20 seconds.
