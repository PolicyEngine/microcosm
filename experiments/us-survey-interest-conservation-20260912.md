# Conserving survey interest before PUF enrichment

The financial graph now retains the tax-exempt remainder of the same interest
total used for its taxable component. Previously it attached only the taxable
part, leaving the original survey support's exempt component unresolved after
the clone1-only PUF attachment.

The new eighth financial output is `INT_VAL - taxable_interest_income`, computed
before source-key alignment to both clones. ASEC uses its qualified current
amount; ACS uses the existing modeled interest total. The original taxable
calculation and legacy seven-leaf helper are unchanged. The graph's judgment
metadata identifies the split as an assumption, not separately observed
taxable and exempt amounts. It does not establish that the modeled ACS interest
total reconciles to the broader observed ACS property-income aggregate.

Two new numerical controls first reproduced the missing column. After repair,
all 25 predictor and atomic-financial tests passed in 294.225 seconds, including
nonconstant and zero-regime invented source runs, source-key permutation,
conservation and cold/required graph replay.

Three full PUF controls then passed in 617.283 seconds: the actual two-route
cold/required extension, independent whole-cohort finalizer comparison and
non-owned storage preservation, and retained checked-parent validation without
re-execution. The original survey complement remains identical through the
PUF attachment; clone1 retains its existing PUF ownership. All tests use
invented sources. No failure, error or skip occurred in either green run.

Independent source review found no actionable defect. Ruff and CI test inventory
verification pass. The executable source SHA256 is
`0df64e5f7768b326f43d7b0ac927ccb24c89de8e3d21556a9139628a369b5358`.
These checks do not certify a native candidate, fiscal calibration or release.
The running native PUF pilot remains on its earlier frozen source.
