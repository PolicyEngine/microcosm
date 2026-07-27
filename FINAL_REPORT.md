# Final report: attempt 9 Build-P register maintenance

## Outcome

Completed the two measurement-driven runtime register updates. No stop
condition fired: all four proposed zero-support cells are exactly zero on the
base-P/frozen-selection target frame, and the old QRF register's three unused
entries reconcile exactly with the gate surface as two thin columns and one
dense column.

Deliverables:

- `/Users/maxghenis/PolicyEngine/_buildo-runtime/inputs/sparse_zero_support_exclusions_buildp.json`
  - 31 entries: all 27 Build-M entries preserved value-for-value, plus the
    four measured-zero KY/LA cells.
  - SHA-256:
    `182adea03000b8053ceb4a5a57d6fb3288cbfa3ea6d1f3030a93af8f4ed0938f`
- `/Users/maxghenis/PolicyEngine/_buildo-runtime/inputs/qrf_tail_exclusions_buildp.json`
  - 11 entries: exactly the columns whose gate-measured top-100 weighted
    absolute-mass share is strictly above 75%.
  - SHA-256:
    `07be70885ee56c995755e494835835420a626572db2235dfd28699ce7177f053`

No release was run and no threshold was changed.

## Frame and selection binding

The measurements bind to:

- Attempt-8c release:
  `populace-us-2024-buildo-sparse-rmloss100-3537155-20260727T024951Z`
- Code commit:
  `35371553b6c22bc0499d3b23110088e7cf582379`
- Base-P:
  `/Users/maxghenis/PolicyEngine/_buildo-runtime/out/base-p/base_populace_us_2024_puf_support.h5`
- Base-P SHA-256:
  `ad13e5ead6366060f704d4a53dc1f4974d610dd4927f54a89c69c72fb55e63cb`
- Frozen selection:
  `/Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/buildm_keogh_swap_selection_source.json`
- Selection count: 57,240 identities.
- Selection digest declared by the manifest and independently recomputed by
  the release selection loader:
  `1812b819b0a2e6b7f7f72b68125881dd2a171301c179befef6ff3df47589657e`
- Attempt-8c target-frame checkpoint:
  `/Users/maxghenis/PolicyEngine/_buildo-runtime/checkpoints/buildp-sparse2/target_frame_checkpoint.h5`
- Checkpoint identity SHA-256:
  `3cd8b82b8efc46bbcc79b5ad2dbd7f178936f5b4a61aec327858625f97b725a6`

The checkpoint identity carries the same base-P SHA and independently verified
selection digest. It contains 57,240 household rows and 5,675 compiled target
rows.

Attempt 8c failed before writing its candidate H5. The older
`out/buildo-run/sparse/artifacts/populace_us_2024.h5` has a July 22 timestamp
and was deliberately not used. The QRF calculation instead reconstructed the
gate's actual in-memory export frame by replaying the attempt-8c calibration
from the exact target-frame checkpoint.

The reconstruction matched attempt 8c exactly:

- Registry version: `99b84c3a6677`.
- Target matrix: 5,675 x 57,240 with 3,682,466 nonzeros.
- Final loss: `0.01852712099663438` in both the replay and 8c diagnostics.
- Maximum absolute difference across all 5,675 final target estimates: `0.0`.
- Target-surface SHA-256:
  `3cdeba355137c74976d1da1e997aae33a7a31936ff21227139e0910364788c36`
  in both.
- Target-name SHA-256:
  `72a58937d2f664f874bda95c6509b8626344f3fad5219e92002f8c13b351dfe9`
  in both.
- Target-value SHA-256:
  `27baaae5033c7fdee3b77b20e248718d6dd687874e934605144536554c14ac02`
  in both.

## Zero-support measurements

The direct measurement uses each checkpoint target vector and its aligned
design weights. Every proposed cell has zero nonzero records, zero absolute
signal, and a zero design-weighted aggregate. Attempt-8c diagnostics
independently report both initial and final estimates of zero.

| Source record ID | Target | Nonzero records | Absolute signal | Design-weighted aggregate | 8c initial | 8c final |
|---|---:|---:|---:|---:|---:|---:|
| `irs_soi.ty2022.historic_table_2.state_agi.ky.under_1.taxable_interest_returns` | 8,115.102326 returns | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `irs_soi.ty2022.historic_table_2.state_agi.ky.under_1.taxable_interest_amount` | $50,298,965.727026 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `irs_soi.ty2022.historic_table_2.state_agi.la.under_1.taxable_interest_returns` | 9,045.984509 returns | 0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `irs_soi.ty2022.historic_table_2.state_agi.la.under_1.taxable_interest_amount` | $70,061,556.969003 | 0 | 0.0 | 0.0 | 0.0 | 0.0 |

The register keys correctly omit the diagnostic `@2024` suffix because the
loader matches the underlying `source_record_id`.

## QRF gate reconstruction

The exact implementation is
`_qrf_tail_concentration_gate` in
`tools/build_us_fiscal_refresh_release.py:4869` and
`tail_concentration_gate` in
`packages/populace-build/src/populace/build/gates.py:2329`.

Its operative rules are:

1. Candidate names are all outputs of any source stage containing a
   `fit_weighted_qrf` operation: 105 outputs on this commit.
2. A numeric column enters the sparse surface when its raw nonzero share is at
   most 5%.
3. The gate resolves the column entity's final calibrated weights, computes
   `mass = abs(value) * weight` on finite rows, and retains positive mass.
4. Columns with fewer than 500 weighted carriers are thin and do not receive a
   top-100 share.
5. For the remaining columns, the statistic is the sum of the 100 largest
   masses divided by total mass.
6. The code comparison is strict `share > 0.75`; exactly 75% passes. All 11
   registered columns are well above the boundary, so there is no equality
   ambiguity.

Surface accounting is complete: 36 sparse numeric columns reached the core
gate, of which 22 had at least 500 carriers and 14 were thin; 31 candidates
were dense, 19 absent, and 19 nonnumeric. These groups sum to all 105
candidates.

### Evaluated top-100 shares

| Column | Carriers | Exact share | Percent | Disposition |
|---|---:|---:|---:|---|
| `bond_assets` | 2,129 | 0.9492729001109002 | 94.927290% | registered |
| `educator_expense` | 1,949 | 0.5058771408376097 | 50.587714% | below threshold |
| `estate_income` | 561 | 0.9980720742062316 | 99.807207% | registered |
| `farm_operations_income` | 1,739 | 0.8752292373826215 | 87.522924% | registered |
| `health_savings_account_ald` | 718 | 0.9126968012673277 | 91.269680% | registered |
| `miscellaneous_income` | 2,087 | 0.8206673970845759 | 82.066740% | registered |
| `non_sch_d_capital_gains` | 2,440 | 0.8901083682600798 | 89.010837% | registered |
| `partnership_income` | 4,768 | 0.5587544460442003 | 55.875445% | below threshold |
| `partnership_self_employment_net_earnings` | 1,445 | 0.7928916687038458 | 79.289167% | registered |
| `qualified_bdc_income` | 1,289 | 0.9892725791815580 | 98.927258% | registered |
| `qualified_tuition_expenses` | 523 | 0.9529395821051290 | 95.293958% | registered |
| `salt_refund_income` | 7,302 | 0.5453857611313601 | 54.538576% | below threshold |
| `self_employed_pension_contributions_desired` | 1,662 | 0.5816605480178914 | 58.166055% | below threshold |
| `social_security_dependents` | 6,117 | 0.2835374840946470 | 28.353748% | below threshold |
| `social_security_disability` | 7,398 | 0.2874156497596738 | 28.741565% | below threshold |
| `social_security_survivors` | 6,277 | 0.3170179173759859 | 31.701792% | below threshold |
| `sstb_self_employment_income_before_lsr` | 2,541 | 0.7172303821873613 | 71.723038% | below threshold |
| `sstb_unadjusted_basis_qualified_property` | 1,319 | 0.9219677344571345 | 92.196773% | registered |
| `student_loan_interest` | 2,957 | 0.4612055684372231 | 46.120557% | below threshold |
| `tax_exempt_interest_income` | 3,061 | 0.7292504699507804 | 72.925047% | below threshold |
| `unadjusted_basis_qualified_property` | 6,627 | 0.6625881231151178 | 66.258812% | below threshold |
| `unrecaptured_section_1250_gain` | 671 | 0.9981199302138782 | 99.811993% | registered |

### Thin sparse columns

The gate deliberately does not calculate a top-100 share below 500 carriers;
the carrier count is the measured gate value.

| Column | Carriers | Gate disposition |
|---|---:|---|
| `alimony_expense` | 180 | thin |
| `alimony_income` | 205 | thin |
| `casualty_loss` | 37 | thin |
| `domestic_production_ald` | 376 | thin |
| `farm_income` | 74 | thin |
| `farm_rent_income` | 370 | thin |
| `investment_income_elected_form_4952` | 116 | thin |
| `long_term_capital_gains_on_collectibles` | 63 | thin |
| `s_corp_income` | 0 | thin |
| `second_home_mortgage_balance` | 0 | thin |
| `second_home_mortgage_interest` | 0 | thin |
| `second_home_mortgage_origination_year` | 0 | thin |
| `sstb_w2_wages_from_qualified_business` | 135 | thin |
| `w2_wages_from_qualified_business` | 466 | thin |

### Old-register reconciliation

The complete 8c log list was:

```text
['farm_rent_income', 'qualified_reit_and_ptp_income', 'w2_wages_from_qualified_business']
```

All three classifications agree with the frame and the gate:

| Old entry | Measured Build-P classification | Action |
|---|---|---|
| `farm_rent_income` | sparse but thin: 370 carriers, below the 500-carrier floor | dropped |
| `qualified_reit_and_ptp_income` | dense: 8,383 / 166,321 nonzero person rows = 5.040253%, above the 5% sparse ceiling | dropped |
| `w2_wages_from_qualified_business` | sparse but thin: 466 carriers, below the 500-carrier floor | dropped |

No old unused/dropped entry was still in the checked-above-threshold surface;
all 11 other old entries remained above threshold and were retained. No
previously unregistered column newly crossed the threshold. The Build-P
register is therefore the 14-column Build-N set minus exactly these three
entries.

## Loader and gate validation

The release loaders at
`tools/build_us_fiscal_refresh_release.py:6971` and
`tools/build_us_fiscal_refresh_release.py:7000` require a top-level JSON object
whose values are nonempty reason strings. Measurement evidence is therefore
embedded in each reason string rather than placed in nested fields.

Validation evidence:

- Both files parse with `python -m json.tool`.
- Both satisfy an independent `jq` object/nonempty-string check.
- `_load_zero_support_exclusions` loaded 31 entries.
- A mapping comparison proved all 27 source entries are present with exactly
  their original string values and only the four requested keys were added.
- `_load_qrf_tail_concentration_exclusions` loaded 11 entries.
- Every QRF reason's recorded exact share and carrier count equals the
  reconstructed gate measurement.
- Re-running `_qrf_tail_concentration_gate` with the new register returned:
  `passed=True`, `failures=[]`, `stale_exclusions=[]`, and
  `dormant_exclusions=[]`.
- The gate's used exclusion keys equal all 11 register keys exactly.

## Diagnostic cross-checks; no action

Attempt-8c `calibration_diagnostics.json` confirms:

| Diagnostic | Target | Final estimate | Relative error |
|---|---:|---:|---:|
| Mortgage Table 2.1 home-mortgage-interest amount | $186,310,104,604.25217 | $224,295,761,736.252 | +20.388404% |
| Medical HT2 medical/dental-expense amount | $89,891,939,590.30547 | $105,466,551,398.37177 | +17.325927% |

Rounded as requested, mortgage is `$224,295,761,736` versus
`$186,310,104,604` (`+20.39%`), and medical HT2 is `+17.33%` (versus the
noted `+9.2%` on base-O). No action was taken on either diagnostic.
