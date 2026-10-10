# SIPP asset-gradient estimation

The SIPP evidence does **not** support the proposed decreasing propensity over
the entire $0–$2,000 resource-eligible range. The reviewed primary adult and
elderly receipt associations are positive. These signs are preserved; neither
the reform counts nor an elderly-gap target enter estimation. They cannot be
presented as an evidence-supported correction to the elderly overprediction.

| Sample | Rows (SSI receipt) | Slope on `log1p(assets)` | Household-clustered SE | 95% interval |
|---|---:|---:|---:|---:|
| Disabled/blind adults, 18–64; $0–$2,000 | 817 (260) | +0.033302 | 0.033420 | [−0.032200, +0.098804] |
| Aged 65+; $0–$2,000 | 342 (152) | +0.113963 | 0.044874 | [+0.026009, +0.201916] |
| Disabled/blind children, 15–17; $0–$2,000 | 115 (8) | −0.347979 | 0.217540 | [−0.774358, +0.078399] |

The operational under-18 coefficient explicitly transfers the adult estimate.
The 15–17 estimate above is reported separately, and no assets are measured for
children under 15 in these asset recodes. This transfer is an identification
limitation; it does not remove the existing child qualification fence.

`sipp_estimation.json` contains every estimate, its sample size, weighted mass,
receipt rate, coefficient standard errors, and maximum likelihood score. The
packaged `ssi_asset_gradient.json` preserves primary slopes and sampling
provenance. Data-regression intercepts in this evidence are not the runtime
intercepts: runtime intercepts are separately solved against SSA counts on the
stage's candidate basis, with direct reporters pinned true.

The estimator reads and verifies the repository's immutable 3,726,010,471-byte
SIPP 2023 donor SHA-256. The pin, three asset mappings, and allocation-status
mappings come from the existing `source_stages.json` financial-asset recipe,
without importing the US tax-benefit system. Only aggregate evidence is saved.
The assigned snapshot has no `full_sipp_donor.py`; the maintained financial-asset
and SSI disability donor producers were reviewed instead.

The sample consists of December 2022 person records from SIPP 2023, unmarried
(EMS 3–6), age 15+, with positive finite person weights, observed receipt and
source-allocation statuses, and own bank + stock/mutual-fund + bond assets. An
independent broad disability proxy uses reported work-limiting disability,
inability to work, disability income, disability recode, or seeing difficulty;
all aged people satisfy the age screen. It never infers disability from receipt.
The apparent-income screen removes SSI income itself, applies the $20 general
and $65 earned-income exclusions and half of remaining earned income, and
requires countable monthly income at most the 2022 individual federal rate,
$841. The sample has no full spouse/parent deeming or non-liquid-resource screen.
These are receipt associations among apparent candidates, not verified SSA
medical/resource eligibility or causal claiming effects. Definitions are
anchored in the [Census 2023 SIPP dictionary](https://www2.census.gov/programs-surveys/sipp/tech-documentation/data-dictionaries/2023/2023_SIPP_Data_Dictionary.pdf)
(bank assets p.190, bond assets p.192, personal income p.2871, monthly SSI p.3149)
and [SSA's 2022 Table IV.A2](https://www.ssa.gov/OACT/ssir/SSI22/SingleYearTables/IV_A2.html).

The principal fit excludes own assets above $2,000 so an eligibility cliff does
not masquerade as declining claiming. It includes zero assets because the
proposed smooth `log1p` function covers zero. Extrapolation above $2,000 is an
untested functional assumption, not identification of high-asset claiming.
Reverse causation is possible: current cash holdings can include received
benefits. The broad disability proxy and resource/deeming omissions also allow
eligibility composition to affect receipt. Household sandwich SEs account for
shared households and unequal weights, but do not replace SIPP replicate-weight
variance.

Sensitivity fits show why the truncation and zero-asset choice matter:

| Sensitivity | Adult slope (SE), rows | Elderly slope (SE), rows | Interpretation |
|---|---:|---:|---|
| Positive assets, $1–$2,000 | +0.003355 (0.069299), 401 | −0.227663 (0.092123), 196 | Zero-asset receipt composition changes the sign; diagnostic only |
| Annual receipt, $0–$2,000 | +0.028556 (0.033364), 817 | +0.113963 (0.044874), 342 | Same primary conclusion |
| Entire observed asset range | −0.055277 (0.026066), 911 | −0.058059 (0.027877), 417 | Receipt conflates statutory eligibility and claiming; never operational |

The child positive-assets sensitivity has only one recipient in 36 observations
and extreme fitted coefficients (slope +164.3). Its software SE is not a usable
precision claim; possible separation makes this a diagnostic warning. Choosing
the negative elderly positive-assets fit after seeing the primary sign would
change the design. It is retained as sensitivity evidence, not substituted.
The updated tool rejects complete or quasi-complete separation when the
positive and negative outcome ranges are ordered along the single asset
predictor. It has both complete- and quasi-separation behavior tests. The
historical child diagnostic preserves its original numerical output with an
explicit unstable/not-usable status; the microdata pass was not repeated.

CBPP's adult equation has a −0.04 probit coefficient on natural-log countable
assets (Romig, Nuñez and Sherman, 2023, methodological appendix p.14; see
[validation.md](validation.md)). It uses SSI and/or SSDI receipt and a different
sample, controls, vintage and transform. The primary adult SIPP sign disagrees;
the whole-range SIPP sign agrees but is confounded by resource eligibility.
For a transparent link-scale diagnostic at probability one-half and positive
assets large relative to $1, CBPP's local probability derivative is
`−0.04 / sqrt(2π) = −0.015958`, while the primary SIPP logit's is
`+0.033302 × 0.25 = +0.008325`. These derivatives do not rescale either fitted
coefficient and are not an equivalence test. CBPP publishes no coefficient SE.

The completed streaming pass used 50,000-row chunks, took 153.78 seconds after
imports, and reported a 12.50 GiB peak RSS. The initially proposed 4 GiB budget
was incorrect: the wide CSV parser retained more memory than its selected
columns suggested. The tool now uses 5,000-row chunks for future runs; that
lower-memory configuration was not rerun. `heavy --mem 4` was attempted first
and refused before launch because the sandbox disallowed `ps`; the completed
standalone pass therefore lacked admission. No heavy suites, release builds,
publication, certified-default engine measurement, or reform-output fitting
were performed.

Reproduction after a permitted memory admission:

```bash
heavy --mem 16 -- python tools/estimate_ssi_asset_gradient.py \
  --sipp /path/to/pu2023.csv \
  --output docs/evidence/ssi-take-up-asset-gradient/sipp_estimation.json \
  --packaged-output packages/microcosm-build/src/microcosm/build/us/ssi_asset_gradient.json
```

The conservative 16 GiB reservation reflects the measured historical peak;
lowering it requires measuring the revised chunking first.

The registered estimator tests passed all 18 cases through the ordinary
repository conftest in an engine-free temporary environment. Each of 18
deliberate source mutations then failed its selected registered test body with
pytest exit code 1. The working source stayed intact. The checked mutations
cover source mappings, own-asset summation, receipt/income independence, sample
exclusions, identity uniqueness, the log transform, household clustering,
non-identifiability/separation, the resource-range cut, and child transfer.
[estimator_mutations.json](estimator_mutations.json) records each changed source
fragment and its verbatim failure tail. The command environment and exit codes
are in [estimator-pytest-environment.json](estimator-pytest-environment.json).
The baseline and mutation run took 75.31 seconds; its verbatim log is:

```text
Estimator registered pytest baseline
..................                                                       [100%]

Estimator registered pytest source mutations
{"mutations": 18, "killed": 18}
```

Requested commits
could not be written: the assigned sandbox exposes `.git` read-only and refused
creation of `.git/index.lock`. No history was rewritten, pushed, or published.
