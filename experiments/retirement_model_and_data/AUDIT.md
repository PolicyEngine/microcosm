# Retirement model and data audit

## Disposition

The frozen f025 adjudication contains 16 failed criteria across 11 positive
monetary legs. This audit classifies the physical legs, not each separately
reported incidence and quantile check:

- **5 dense-rung refits required:** taxable IRA, Keogh, taxable 401(k), taxable
  403(b), and taxable SEP.
- **6 concept mismatches requiring owner adjudication:** tax-exempt and taxable
  private pension, plus Social Security retirement, disability, dependents, and
  survivors.
- **0 derivation defects and 0 production fixes.** All 11 current clone-0
  equations reproduce from the frozen raw columns. That arithmetic does not
  validate the pension or Social Security labels: both audits found unresolved
  source-to-engine taxonomy mismatches for which partial rowwise patches would
  be unsafe; the exact owner equations are below.

The exact frozen-artifact measurements, identities, raw-draw hashes, and
machine-readable classifications are in [`f001_audit.json`](f001_audit.json).
Numbers in this document are displays of that sidecar; code citations establish
mechanisms, not the measured values. Retirement remains MODEL/DATA work, not an
eligibility credit, and this audit proposes no exclusion.
That boundary is the source-of-truth adjudication
(`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-arm-split/experiments/battery_burndown/ADJUDICATION.md:46-63,81-86`).

The separate arm-split synthesis diagnosed a bundle-only predictor-replacement
effect or interaction and made removal of
`__acs_transfer_social_security_income` and
`__acs_transfer_retirement_income` the leading *hypothesis*, not a causal
finding. Its frozen removal/addition factorial remains a guardrail before any
future bundle replacement; it is not an explanation of these already-red
baseline legs
(`/Users/maxghenis/PolicyEngine/_worktrees/microcosm-arm-split/experiments/phase_p_arm_split/SYNTHESIS.md:346-352,376-379`).

## Frozen evidence identity

The audit binds release
`populace-us-2024-stacked-f001-s578-asec1688-acs15316-20260820T231241Z-7d93419e`,
sample fraction `0.01`, sample seed `578`, model seed `0`, 1,688 ASEC and
15,316 ACS households, 4,311 ASEC and 34,293 ACS people, and 80,395 final person
rows. The relevant immutable digests are:

| Artifact | SHA-256 |
|---|---|
| baseline f001 pool | `8ed64a03fdad77f7f1d3f9eea8e800a5f35b88bb7bb2a28c7d7b10b0632037f0` |
| baseline f001 gates | `1d6059868680f872fe04d452a536bcc3c215bafabb4c50d7740a469fe6a8b56a` |
| package-3 f001 pool | `2892596de1148711dc74da777b08e5fffec138ebc42253e34b32baf9d73886b9` |
| package-3 f001 gates | `3ace0af0fd9e2ed6cb37cb110280f0c5cade182118c62737635c7ad177050ac3` |
| assembled checkpoint file | `754ceb74fc737a41b588003912ce58cdfe29a7f3c114c9aba26cd2e6ae9956bd` |
| stacked checkpoint directory namespace | `2e45c4d60f66b4321bc00ffa22816470bf162c59fd91956514832f97e066ed3c` |
| transferred checkpoint | `5ff70151a9ea9c9707e794995bb739abcd76ddc1401aea37299da41314a91d68` |
| ACS-transfer namespace | `091dc2effbe638687de621f3ed4312f738489a58923bf1a7172ac5da8e3c6eb7` |
| canonical selected 11-comparison baseline/package-3 gate digest | `9f350b4efa9fb229c7e8bdc775f19e9e4d1a586199bb17361a6352cab1ead5ca` |

The raw ASEC `person_id` to assembled `person_source_id` join is unique and
fully resolved for all 4,311 sampled clone-0 people. All 16 retirement source
columns are value-identical, preserving missingness, from that raw join through
the assembled and transferred checkpoints
(`experiments/retirement_model_and_data/audit_frozen_artifacts.py:1764-1851`).

All 11 terminal target arrays and selected gate records are byte-identical
between baseline and package 3. The target-bank raw-draw digests are:

| Target | Raw-draw SHA-256 |
|---|---|
| tax-exempt private pension | `bee9b66005382ec505380bd52ba621ed38c89b8b25c84a2dc642394c79bfcdec` |
| taxable private pension | `07efb34b153a025fa375fd10956c4217959b005efd28b96ac65e035314d7d7d0` |
| taxable IRA | `edf2d13aaa6b8525af22cf03ce3e22776b4543047d55b2660096f72514af4df4` |
| SS retirement | `9a8565dd071ff94bfd0d8d856ee1de06722d50d64e28e8d091b5073481ece32b` |
| SS disability | `ec8dc95c6e6905b939880a9d894fe1bf140d9b4ed247df1bb01d8f742bfa955e` |
| SS dependents | `27570eb8d8dfc95b6dfa715dbc392cb807f024227dbf6568f0f2191d1bcb11a9` |
| SS survivors | `04850fd4563274d0d9842fbfc7b795eac2b0b4333ef84e910387ebd4718e56ca` |
| Keogh | `74a5ba958dad1499e160092240730191c5241594db19816db6c0a857987edd61` |
| taxable 401(k) | `f69ce5b357300d056f38348045a021959bec919e6a746f92bf2af9df2ca15758` |
| taxable 403(b) | `2712e4e36f42407bc67b511e3c73f119d7da5cf7fa81b71d22149b9089cc304f` |
| taxable SEP | `28f63d4cab4cf68c3ef323d70f149df23b47442744a097abba281f4c484a1e0c` |

## Source documentation and notation

The source meanings below are checked against the official
[2024 CPS ASEC technical documentation](https://www2.census.gov/programs-surveys/cps/techdocs/cpsmar24.pdf),
the [2024 ASEC variable index](https://api.census.gov/data/2024/cps/asec/mar/variables.html),
and the [2024 ACS PUMS variable index](https://api.census.gov/data/2024/acs/acs1/pums/variables.html).
Those Census resources, rather than repository code, support the semantic
claims that `PNSN_VAL` is combined income from all reported pension sources,
`ANN_VAL` is separate annuity income, `SS_VAL` is combined Social Security
income, `RESNSS1/2` are receipt reasons, the
`DST_SC*`/`DST_VAL*` pairs are retirement-account codes and amounts, ACS `RETP`
is combined retirement income, and ACS `SSP` is combined Social Security
income.

The exact ASEC `DST_SC*` legend is: `0` not in universe, `1` 401(k), `2`
403(b), `3` Roth IRA, `4` regular IRA, `5` Keogh, `6` SEP, and `7` other. The
implemented strict derivation accepts codes `0..7` and binds codes `1..6` to
the six named engine leaves
(`packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:97-112,210-269`).

Let `nz(x)` mean numeric coercion followed by missing-to-zero. That is the
implemented CPS-carried source rule
(`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:345-362`).
For retirement distribution slots, let
`s ∈ {1, 2, 1_YNG, 2_YNG}`. The strict account stage instead requires finite,
nonnegative amounts and integer codes in `0..7`
(`packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:97-112,210-269`).

## Frozen gates: all 16 checks and 11 legs

The terminal battery selects positive-weight rows at the declared clone index,
splits monetary comparisons by sign, tests weighted sign incidence, and then
tests weighted p10/p25/p50/p75/p90 conditional absolute-magnitude envelopes
when each origin has at least five carriers. Its frozen incidence band is
`[0.8, 1.25]` and its quantile-envelope ceiling is `0.25`
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:3026-3030,11306-11308,11523-11640,11860-11931,11990-12021`).

| Leg | Frozen f025 incidence / QED | f025 positive rows ASEC / ACS | Frozen f001 incidence / QED | f001 positive rows ASEC / ACS | Classification |
|---|---:|---:|---:|---:|---|
| `tax_exempt_private_pension_income` | 1.554057541 **FAIL** / 0.104283054 pass | 6,406 / 110,336 | 1.653314783 **FAIL** / 0.228438228 pass | 261 / 4,559 | concept mismatch |
| `taxable_private_pension_income` | 1.563313972 **FAIL** / 0.123893805 pass | 6,406 / 111,027 | 1.594681938 **FAIL** / 0.206278027 pass | 261 / 4,410 | concept mismatch |
| `taxable_ira_distributions` | 1.369226561 **FAIL** / 0.196796339 pass | 1,556 / 24,373 | 1.120522356 pass / 1.361715708 **FAIL** | 59 / 741 | dense-rung refit |
| `social_security_retirement` | 1.010336184 pass / 0.539906517 **FAIL** | 15,440 / 170,783 | 0.984545680 pass / 0.482190076 **FAIL** | 650 / 7,036 | concept mismatch |
| `social_security_disability` | 0.777259624 **FAIL** / 1.326558100 **FAIL** | 1,935 / 13,616 | 0.419575564 **FAIL** / 0.239893338 pass | 64 / 263 | concept mismatch |
| `social_security_dependents` | 1.627006704 **FAIL** / 1.255813953 **FAIL** | 263 / 3,180 | 0.145877895 **FAIL** / 0.917093392 **FAIL** | 14 / 22 | concept mismatch |
| `social_security_survivors` | 0.849837855 pass / 0.984261341 **FAIL** | 398 / 3,094 | 0.318087105 **FAIL** / 0.278088678 **FAIL** | 12 / 39 | concept mismatch |
| `keogh_distributions` | 0 **FAIL** / not evaluated | 2 / 0 | absent on both origins | 0 / 0 | dense-rung refit |
| `taxable_401k_distributions` | 0.792858078 **FAIL** / 0.461538462 **FAIL** | 2,057 / 16,979 | 0.323269080 **FAIL** / 1.333333333 **FAIL** | 86 / 235 | dense-rung refit |
| `taxable_403b_distributions` | 0.726005496 **FAIL** / 1.463917526 **FAIL** | 161 / 1,371 | 0 **FAIL** / insufficient carrier support | 6 / 0 | dense-rung refit |
| `taxable_sep_distributions` | 17.073910540 **FAIL** / 1.769633508 **FAIL** | 61 / 9,406 | 0.037702462 **FAIL** / insufficient carrier support | 4 / 2 | dense-rung refit |

The table has nine failed f025 incidence criteria and seven failed f025 QED
criteria. The f001 rung is diagnostic only: notably, taxable IRA reverses from
incidence-red/QED-green at f025 to incidence-green/QED-red at f001, and f001
contains no Keogh carrier. Neither result can be used to tune a treatment
toward a passing 1% gate.

## Actual frozen donor selection and recomputed regimes

### Early transfer

The seven pension, IRA, and Social Security targets are produced from CPS
evidence before cloning and belong to the declared early ASEC-to-ACS transfer
surface
(`packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:491-496,740-776,1744-1789`;
`packages/microcosm-build/src/microcosm/build/us/spec/imputation.yaml:371-392,449-514,559-579,625-635`).
The owner projects native ASEC-origin clone-0 rows, calls the spine-blind
transfer, and preserves donor-origin cells
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:7581-7698,7808-7820`).
Support cloning subsequently copies those prepared values into the PUF-detail
arm
(`packages/microcosm-build/src/microcosm/build/us_runtime/puf_support.py:540-573,629-680,2121-2143`).

Every frozen early availability pattern has the same 4,311-row ASEC clone-0
donor index (SHA-256
`58b9edf0e1779a0e47534e0b385f5332e45b1f607ab93147cf8916c9a06a6a00`):

| Pattern | Recipient rows | Predictors beyond `age`, `is_female`, state |
|---|---:|---|
| `pattern_00_677f6490` | 18 | household head |
| `pattern_01_5874881e` | 1,877 | employment, self-employment, combined Social Security, combined retirement, household head |
| `pattern_02_7c3bceda` | 5,276 | household head, tenure |
| `pattern_03_04f75638` | 27,122 | employment, self-employment, combined Social Security, combined retirement, household head, tenure |

### Late transfer

The retirement-account stage first derives exact ASEC account sums, preserves
only PUF taxable IRA, and then replaces 401(k), 403(b), Keogh, and SEP on the
PUF-detail half with the post-clone source-owner internal CPS-trained PUF-role
QRF
(`packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:272-333,336-473`).
The late contract then selects the ASEC-origin clone-1 projection as donor and
fills target-specific complements while requiring producer byte identity
(`packages/microcosm-build/src/microcosm/build/us/spec/imputation.yaml:2405-2463`;
`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8340-8483,8518-8538,8631-8729`).

Every frozen late availability pattern has the same 4,311-row ASEC-origin
clone-1 donor index:

| Pattern | Recipient rows | Predictors beyond `age`, `is_female`, state | Seed |
|---|---:|---|---:|
| `pattern_00_36785ebf` | 36 | employment, self-employment, household head | 2593627294 |
| `pattern_01_c6777728` | 4,073 | employment, self-employment, combined Social Security, combined retirement, combined investment, household head | 1021650046 |
| `pattern_02_5e7dd311` | 10,850 | employment, self-employment, household head, tenure | 1130471859 |
| `pattern_03_76a0101a` | 56,469 | all listed optional predictors, including tenure | 954292944 |

The transfer constructs recipient optional-predictor patterns, fits each on
donor rows complete for all family targets and exactly that predictor set, and
fills only missing recipient cells
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:880-918,1247-1380,1411-1625,1792-1811,3006-3035`).
QRF detects regimes from *unweighted* sign existence at `zero_atol=1e-6`;
zero-inflated regimes fit a directly weighted sign gate and sign-conditional
forest, while a degenerate-zero target returns zeros without a model
(`packages/microcosm-fit/src/microcosm/fit/qrf.py:83-150,950-1003,1333-1380,1389-1442`).
The frozen checkpoints did not persist the regime, so this audit recomputed it
from the exact donor mask for every availability pattern rather than inferring
it from terminal output. The QRF step result exposes `regime`, but the transfer
checkpoint stores only the raw draw and before/after chain states, and the
bank's closed metadata envelope has no regime field
(`packages/microcosm-fit/src/microcosm/fit/qrf.py:648-680,882-903,1225-1230`;
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:405-475,1629-1648`;
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer_bank.py:237-278,490-514`).

| Leg | Actual fit donor | Donor sign rows `(negative, zero, positive)` | Positive donor amount: sum `[min, max]` | Recomputed regime in every pattern |
|---|---|---:|---:|---|
| tax-exempt private pension | ASEC origin, clone 0 | `(0, 4050, 261)` | $2,801,038.82 `[$4.92, $73,800]` | `zero_inflated_positive` |
| taxable private pension | ASEC origin, clone 0 | `(0, 4050, 261)` | $4,030,763.18 `[$7.08, $106,200]` | `zero_inflated_positive` |
| taxable IRA | ASEC origin, clone 0 | `(0, 4252, 59)` | $1,253,493 `[$400, $150,000]` | `zero_inflated_positive` |
| SS retirement | ASEC origin, clone 0 | `(0, 3661, 650)` | $12,947,542 `[$1, $60,000]` | `zero_inflated_positive` |
| SS disability | ASEC origin, clone 0 | `(0, 4247, 64)` | $967,357 `[$2,622, $50,000]` | `zero_inflated_positive` |
| SS dependents | ASEC origin, clone 0 | `(0, 4297, 14)` | $214,420 `[$2,640, $30,779]` | `zero_inflated_positive` |
| SS survivors | ASEC origin, clone 0 | `(0, 4299, 12)` | $197,443 `[$4,948, $26,832]` | `zero_inflated_positive` |
| Keogh | ASEC origin, clone 1 after the post-clone source-owner internal CPS-trained PUF-role QRF | `(0, 4311, 0)` | $0 | `degenerate_zero` |
| taxable 401(k) | ASEC origin, clone 1 after the post-clone source-owner internal CPS-trained PUF-role QRF | `(0, 4249, 62)` | $880,632.313652 `[$300, $79,000]` | `zero_inflated_positive` |
| taxable 403(b) | ASEC origin, clone 1 after the post-clone source-owner internal CPS-trained PUF-role QRF | `(0, 4310, 1)` | $18,000 `[$18,000, $18,000]` | `zero_inflated_positive` |
| taxable SEP | ASEC origin, clone 1 after the post-clone source-owner internal CPS-trained PUF-role QRF | `(0, 4309, 2)` | $34,000 `[$17,000, $17,000]` | `zero_inflated_positive` |

No audited retirement target has a negative source, donor, or terminal carrier.

## Leg-by-leg source, amount, and ACS disagreement audit

### Private pension leaves

The implemented clone-0 equation is

```text
P = nz(PNSN_VAL) + nz(ANN_VAL)
taxable_private_pension_income = 0.590 * P
tax_exempt_private_pension_income = 0.410 * P
```

The fraction and both assignments are explicit code
(`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:41-44,188-200`).
It ignores `PEN_SC1/2`. The 59/41 split is therefore a current model
assumption, not a separately measured Census label; this audit does not claim
empirical validation for the fraction.

That equation does not match the target concepts. The official ASEC source
codes are `1` company, `2` union, `3` federal government, `4` state government,
`5` local government, `6` military, `7` railroad, and `8` other. In the locked
PolicyEngine-US 1.764.6 environment, the private leaves mean non-government
employee pensions, while separate public leaves mean government employee
pensions
(`packages/microcosm-build/src/microcosm/build/us/engine_abi.lock.json:1`;
`.venv/lib/python3.14/site-packages/policyengine_us/variables/household/income/person/retirement/taxable_private_pension_income.py:4-10`;
`.venv/lib/python3.14/site-packages/policyengine_us/variables/household/income/person/retirement/tax_exempt_private_pension_income.py:4-10`;
`.venv/lib/python3.14/site-packages/policyengine_us/variables/household/income/person/retirement/taxable_public_pension_income.py:4-10`;
`.venv/lib/python3.14/site-packages/policyengine_us/variables/household/income/person/retirement/tax_exempt_public_pension_income.py:4-10`).

Frozen source reconciliation finds 234 positive `PNSN_VAL` rows totaling
$6,100,418, 45 positive `ANN_VAL` rows totaling $731,384, and 18 overlapping
rows. Their 261-row union totals $6,831,802. Both clone-0 leaves have zero
*current-equation* mismatches. The pension source categories are:

| Frozen ASEC source category | Positive `PNSN_VAL` rows | `PNSN_VAL` amount |
|---|---:|---:|
| private-only codes 1/2 | 140 | $2,535,409 |
| government-only codes 3–6 | 83 | $3,361,227 |
| mixed private and government | 4 | $90,214 |
| code 8 / otherwise unresolved | 7 | $113,568 |
| separate positive `ANN_VAL` evidence | 45 | $731,384 |

The annuity row is not part of the mutually exclusive `PNSN_VAL` partition;
18 annuity carriers overlap a positive pension carrier. The current equation
deterministically labels all $3,361,227 on the 83 government-only rows as
private pension: $1,983,123.93 taxable and $1,378,103.07 tax-exempt. The frozen
raw, assembled, and transferred checkpoints do not persist source-specific
`PEN_VAL1/2`, so the four mixed rows cannot be split by source amount. These
measurements and the absence check are reproduced by
`experiments/retirement_model_and_data/audit_frozen_artifacts.py:1111-1205,1348-1401,1764-1851`.

| Leg | Clone-0 source and terminal ASEC | Terminal ACS | f001 gate |
|---|---|---|---|
| tax-exempt pension | 261 carriers; $2,801,038.82; range $4.92–$73,800 | 4,559 carriers; $45,925,810.690453; range $4.92–$73,800 | incidence 1.653314783; QED 0.228438228 |
| taxable pension | 261 carriers; $4,030,763.18; range $7.08–$106,200 | 4,410 carriers; $70,487,434.518606; range $7.08–$106,200 | incidence 1.594681938; QED 0.206278027 |

ACS has no taxable or exempt pension leaf. It maps only
`acs_retirement_income = RETP * ADJINC / 1,000,000`, preserving a missing source
as missing
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_inputs.py:1-9,133-144,177-195,307-331`;
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:72-99`).
For donors, the optional retirement predictor is the sum of taxable pension,
exempt pension, and taxable IRA; for recipients it is adjusted ACS `RETP`
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:148-164,186-212,1891-1966`).
The two deterministic ASEC pension leaves also sit in separate early transfer
families/chains
(`packages/microcosm-build/src/microcosm/build/us/spec/imputation.yaml:371-392,449-470,559-579,625-635`).
That is the exact point at which one ASEC total-and-share equation becomes two
independent ACS predictions.

Of 34,293 frozen ACS person rows, adjusted `RETP` is observed on 28,999,
missing on 5,294, and positive on 4,974. The terminal three-leaf retirement
union is positive on 4,966 rows. On rows with observed `RETP`, the three-leaf
sum differs from `RETP` on 4,976 rows; 10 rows are native-positive/output-zero
and two are output-positive/native-zero. Native `RETP` totals
$139,384,817.702 versus $127,178,539.5 across the three output leaves. The two
pension carrier masks differ on 765 ACS rows, and the terminal 59/41 identity
fails on 3,922 ACS rows.

#### Exact declared-absence equation for owner adjudication

Let `P = nz(PNSN_VAL)`, `N = nz(ANN_VAL)`, and let `R` be the set of nonzero
codes in `PEN_SC1/2`. Define `Rpriv = R ∩ {1,2}` and
`Rpub = R ∩ {3,4,5,6}`:

```text
A = 1[
  N > 0 or (
    P > 0 and (
      R ∩ {7,8} != ∅ or
      (Rpriv != ∅ and Rpub != ∅) or
      |Rpriv ∪ Rpub| == 0
    )
  )
]

if P == 0 and N == 0:
    private_taxable = private_exempt = public_taxable = public_exempt = 0
else if A == 0 and R ⊆ {1,2}:
    (private_taxable, private_exempt) = (0.590P, 0.410P)
    (public_taxable, public_exempt) = (0, 0)
else if A == 0 and R ⊆ {3,4,5,6}:
    (public_taxable, public_exempt) = (0.590P, 0.410P)
    (private_taxable, private_exempt) = (0, 0)
else:
    all four leaves = NA (declared absent)
```

At f001 this proposal marks 56 rows and $1,804,558 of combined pension/annuity
evidence as ambiguous. It is an owner proposal, not current behavior and not an
exclusion. The current structural-absence resolver accepts only the ACS
group-quarters rent rule, so adopting this equation requires owner-authorized
mask plumbing
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:1736-1827,1880-1895,7721-7805,11545-11576`).

A seemingly narrow fix that routes codes 3–6 to public leaves is unsafe. The
support clone copies the pre-clone pension leaves, the PUF donor maps its
aggregate taxable-pension field into `taxable_private_pension_income`, and PUF
finalization overwrites only requested outputs on clone 1
(`packages/microcosm-build/src/microcosm/build/us_runtime/puf_support.py:208-224,2121-2143,3990-4008,1993-2047`;
`packages/microcosm-build/src/microcosm/build/us_runtime/puf_aggregate_records.py:86-100`).
Adding copied public leaves without simultaneously adjudicating that PUF
aggregate alias would therefore count the clone-1 aggregate as private while
also retaining the copied public amount. Mixed-source allocation and annuity
placement remain unresolved as well.

**Classification:** both pension legs are concept mismatches. The current
equation reproduces its own source arithmetic but conflicts with the engine's
private/public concepts, and the frozen evidence is insufficient for a safe
partial repair. Pension is not classified as a dense-rung refit until the owner
adjudicates the exact source taxonomy, declared-absence rule, PUF aggregate
role, and early-transfer target surface. No pension production patch is made in
this lane.

### Taxable IRA distributions

The exact clone-0 equation is

```text
taxable_ira_distributions =
    Σs 1[DST_SCs == 4] * nz(DST_VALs),
    s ∈ {1, 2, 1_YNG, 2_YNG}.
```

The CPS-carried implementation is at
`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:490-500`;
the strict general slot derivation independently binds code 4 to taxable IRA
at
`packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:97-110,210-269`.
Frozen slot hit counts are 47, 6, 6, and 0. They produce 59 clone-0 carriers,
$1,253,493, range $400–$150,000, and zero equation mismatches.

Those are two lifecycle stages, not interchangeable provenance. The lenient
CPS-carried code-4 label is created before cloning and drives the early
ASEC-to-ACS transfer while every row is still clone 0. The post-clone source
operator later applies the strict finite/nonnegative code-and-amount contract,
re-derives the terminal ASEC clone-0 label, and preserves the PUF taxable-IRA
leaf
(`packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:491-496,582-586,1744-1789,1936-1942,1959-1988,2095-2164`;
`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:7581-7698,7808-7820`;
`packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:272-333`).
The two code-4 equations happen to agree exactly on the frozen f001 rows; the
early model's label remains the pre-clone one, while the reported terminal ASEC
clone-0 value is the later strict one.

Terminal f001 ASEC remains those 59 carriers and $1,253,493. ACS has 741
carriers totaling $10,765,294.290935, range $411–$150,000; incidence is
1.120522356 and QED is 1.361715708. ACS has no IRA-specific source column:
adjusted `RETP` enters only through the broader optional retirement predictor
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_inputs.py:133-144,177-195`;
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:186-212,1891-1966`).

**Classification:** dense-rung refit required. The exact ASEC code-4 equation
is intact. The f001/f025 incidence-versus-QED reversal is a sampling warning,
not permission to choose a 1% tail treatment.

**Proposed exact refit:** do not remove taxable IRA from its current early
chain. It is target 7 immediately before Social Security retirement in the
first bounded eight-target batch, so changing that chain would also change an
owner-blocked target's fitted prefix and RNG path
(`packages/microcosm-build/src/microcosm/build/us/spec/imputation.yaml:371-514`;
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2338-2379`;
`packages/microcosm-fit/src/microcosm/fit/qrf.py:649-662,1151-1230`).

Instead, run and bank the current early family unchanged, snapshot every early
terminal column, and then apply one post-compatibility single-target overlay.
Pin its exact family string to
`puf_tax_itemization__clone0_taxable_ira_overlay` and overlay registry ID to
`early/asec_survey_to_acs/person/puf_tax_itemization__clone0_taxable_ira_overlay`.
Train the unmodified weighted two-part QRF on the strict exact ASEC clone-0
code-4 label and frozen predictor/availability surface, with ACS `RETP` only as
a covariate. Use only the normally hash-derived family/pattern seeds, apply the
draw only to the original IRA recipient complement, and require the
compatibility bank plus every non-IRA early terminal column to remain
byte-identical
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:994-1043,1247-1408,2902-2916`;
`packages/microcosm-fit/src/microcosm/fit/qrf.py:83-150,1333-1442`).
This overlay requires the reviewed execution seam specified with the late
overlay below; the current public transfer only fills null target cells and
cannot overwrite the authenticated saved complement
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:967-1043`).
Persist every pattern's support and regime; validate held-out dense-rung
behavior and terminal f025 incidence/QED without changing any gate, band,
ceiling, fold, name, or numeric seed. This is proposed and unimplemented.

### Social Security component leaves

Let `S = nz(SS_VAL)`, `r1 = RESNSS1`, `r2 = RESNSS2`, and `a = A_AGE`.
The current implementation forms predicates for retirement code 1, disability
code 2, survivor codes 3/5, and dependent codes 4/6/7, then applies retirement
before disability before survivors before dependents. An otherwise
unclassified positive amount is retirement at age 62 or older and disability
below age 62
(`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:365-410`).
Equivalently:

```text
R = (r1 == 1 or r2 == 1)
D = (r1 == 2 or r2 == 2)
V = (r1 in {3,5} or r2 in {3,5})
P = (r1 in {4,6,7} or r2 in {4,6,7})
U = (S > 0 and not R and not D and not V and not P)

Y_ret = S * 1[R or (U and a >= 62)]
Y_dis = S * 1[(D and not R) or (U and a < 62)]
Y_sur = S * 1[V and not R and not D]
Y_dep = S * 1[P and not R and not D and not V]
```

Frozen positive reason pairs are:

```text
(1,0)=623, (1,2)=4, (1,3)=6, (1,4)=1, (1,8)=1,
(2,0)=63, (3,0)=9, (4,0)=5, (5,0)=3,
(6,0)=2, (7,0)=7, (8,0)=16.
```

The four outputs exactly sum to `SS_VAL` on all 4,311 clone-0 source rows, but
that arithmetic identity does not establish component-label validity. Eleven
positive rows report multiple recognized categories, seven use code 7, and 17
contain code 8. The current precedence assigns a full combined amount to one
leaf on every such row.

| Leg | Clone-0 source and terminal ASEC | Terminal ACS | f001 gate |
|---|---|---|---|
| SS retirement | 650 carriers; $12,947,542; range $1–$60,000 | 7,036 carriers; $135,155,624.588643; range $1–$60,000 | incidence 0.984545680; QED 0.482190076 |
| SS disability | 64 carriers; $967,357; range $2,622–$50,000 | 263 carriers; $3,731,806.097635; range $4,380–$50,000 | incidence 0.419575564; QED 0.239893338 |
| SS dependents | 14 carriers; $214,420; range $2,640–$30,779 | 22 carriers; $428,634.231206; range $4,441–$30,779 | incidence 0.145877895; QED 0.917093392 |
| SS survivors | 12 carriers; $197,443; range $4,948–$26,832 | 39 carriers; $786,993.187042; range $4,948–$26,832 | incidence 0.318087105; QED 0.278088678 |

ACS maps only `acs_social_security_income = SSP * ADJINC / 1,000,000`; it does
not observe the four leaves
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_inputs.py:1-9,133-144,177-195,307-331`;
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:72-99`).
The transfer uses the four-leaf ASEC sum for the donor combined predictor and
adjusted `SSP` for the recipient combined predictor
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:148-164,186-212,1891-1966`).
That is where a reason-coded ASEC partition meets an ACS total with no component
labels.

Adjusted ACS `SSP` is observed on 28,999 rows, missing on 5,294, and positive
on 7,546. At least one terminal component is positive on 7,231 rows. On
observed-`SSP` rows the component sum differs from `SSP` on 7,547 rows; 316 are
native-positive/output-zero and one is output-positive/native-zero. Native
`SSP` totals $141,172,197.815 versus $140,103,058.1045 across the four leaves.

#### Exact declared-absence equation for owner adjudication

Let categories be `R`, `D`, `V`, and `P` for retirement, disability,
survivors, and dependents:

```text
C(0) = ∅
C(1) = {R}
C(2) = {D}
C(3) = C(5) = {V}
C(4) = C(6) = {P}
C(7) = C(8) = ∅
U_i = C(r1_i) ∪ C(r2_i)

A_i = 1[
  S_i > 0 and (
    r1_i ∈ {7,8} or r2_i ∈ {7,8} or |U_i| != 1
  )
]

Y_ik = 0                         if S_i == 0
Y_ik = S_i * 1[U_i == {k}]       if S_i > 0 and A_i == 0
Y_ik = NA (declared absent)      if A_i == 1
```

At f001, `A_i` identifies 35 of 740 positive donors carrying $744,734: 24 rows
containing code 7 or 8 carry $495,214, and 11 distinct recognized
multi-category rows carry $249,520. The two groups are disjoint by construction
in the displayed partition. This equation is an owner proposal, not current
behavior and not an exclusion. The authority has a typed
structural-absence declaration, but the current resolver accepts only the ACS
group-quarters rent equation; an SS rule would require new owner-authorized
exact-mask plumbing
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:1736-1827,1880-1895,7721-7805,11545-11576`).

**Classification:** all four SS legs are concept mismatches. No component refit
is authorized until the owner decides the exact absence equation and the
meaning of codes 7, 8, and multiple reasons.

**Proposed exact post-adjudication refit:** for observed ACS
`T = SSP * ADJINC / 1,000,000`, preserve `T` and require `Y_k >= 0` and
`Σ_k Y_k = T`. Train component allocation only on owner-approved,
unambiguous ASEC rows. Where ACS `SSP` is missing, first fit one two-part total
model and then apply the same allocation model. Do not independently fit four
unconstrained amounts. Persist total and allocation support/regimes and verify
the sum identity plus all four f025 terminal legs. This is a proposed design,
not an implemented fix or a prediction that the gates will pass.

### Keogh, 401(k), 403(b), and SEP distributions

For account code `c`, the exact clone-0 source equation is

```text
Y_c = Σs 1[DST_SCs == c] * DST_VALs,
s ∈ {1, 2, 1_YNG, 2_YNG}.

c = 1: taxable_401k_distributions
c = 2: taxable_403b_distributions
c = 5: keogh_distributions
c = 6: taxable_sep_distributions
```

The code-to-leaf binding and rowwise summation are enforced at
`packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:97-110,210-269`
and declared in
`packages/microcosm-build/src/microcosm/build/us/source_stages.json:2112-2174`.
All four f001 clone-0 equations have zero source mismatches.

| Leg | Exact clone-0 source and terminal ASEC | ASEC clone-1 late donor | Terminal ACS | f001 gate |
|---|---|---|---|---|
| Keogh (`c=5`) | 0 carriers; $0 | 0 carriers; $0 | 0 carriers; $0 | both signs absent |
| taxable 401(k) (`c=1`) | 86 carriers; $1,654,192; range $36–$360,000 | 62 carriers; $880,632.313652; range $300–$79,000 | 235 carriers; $3,254,676.109307; range $300–$79,000 | incidence 0.323269080; QED 1.333333333 |
| taxable 403(b) (`c=2`) | 6 carriers; $58,512; range $3,000–$19,000 | 1 carrier; $18,000 | 0 carriers; $0 | incidence 0; QED insufficient support |
| taxable SEP (`c=6`) | 4 carriers; $22,630; range $200–$17,000 | 2 carriers; $34,000; both $17,000 | 2 carriers; $34,000; both $17,000 | incidence 0.037702462; QED insufficient support |

ACS has no native account-specific distribution leaf. Its only retirement
amount is combined adjusted `RETP`, which can enter as an optional predictor
but does not label any target
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_inputs.py:133-144,177-195`;
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:186-212,1891-1966`).
The exact disagreement entry is therefore twofold: the ACS targets are latent,
and the late fit does not use exact ASEC clone-0 labels. It uses the ASEC-origin
clone-1 values after the post-clone source-owner internal CPS-trained PUF-role
QRF. At f001 that intervening stage changes carrier counts from `0/86/6/4` to
`0/62/1/2` for Keogh/401(k)/403(b)/SEP before the late transfer sees its donor
(`packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:336-473`;
`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8340-8483`).

This establishes an upstream carrier-compression mechanism and a concrete
refit target. It does **not** establish that clone-1 compression alone causes
any f025 terminal failure: that causal claim requires the frozen 25% refit.
Keogh is especially non-generalizable from f001, where it is degenerate; the
f025 source has two ASEC carriers and zero ACS carriers.

**Classification:** all four account legs require a dense-rung refit. The
account-code derivations are exact; changing the late donor evidence is model
work, not a derivation correction.

**Proposed exact refit:** retain the post-clone source-owner internal
CPS-trained PUF-role QRF for its intended PUF outputs. Do **not** repoint all
five late targets to clone 0, because tax-exempt IRA is not a red leg. Instead,
the candidate must first run and snapshot the current clone-1 late output. It
must then construct a one-to-one hybrid donor projection keyed by the preserved
support source ID: begin with the current clone-1 projection; replace only
Keogh, taxable 401(k), taxable 403(b), and taxable SEP target evidence with the
corresponding exact clone-0 labels; and retain clone-1 tax-exempt IRA evidence,
predictors, and weights. The clone operation preserves source provenance, and
the strict derivation populates tax-exempt IRA on both clones while the internal
PUF-role QRF overwrites exactly the other four named outputs. The current late
transfer selects clone 1 as one complete donor frame
(`packages/microcosm-build/src/microcosm/build/us_runtime/puf_support.py:2121-2143`;
`packages/microcosm-build/src/microcosm/build/us_runtime/retirement_distributions.py:127-135,257-333,336-473`;
`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8405-8432,8477-8483`).

The refit is a four-output overlay, not a replacement five-target family. Form
the union of the four target-specific recipient complements and fit/draw Keogh
standalone over that entire union, including rows where terminal Keogh is
producer-owned but a downstream target is not. Then fit the ordered
`[401(k), 403(b), SEP]` chain over the union with two fixed prefix predictors:
exact clone-0 Keogh on the donor and its refit raw draw on the recipient; and
unchanged clone-1 tax-exempt IRA on the donor and its raw draw from the
compatibility pass on the recipient. Apply each terminal draw only on its own
complement. The current family likewise draws on the OR of target-missing masks
before its target-specific merge
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:994-1001,1030-1043,1285-1304,1461-1480,1561-1671`).
This preserves the original downstream conditioning edges without refitting
the passing tax-exempt IRA branch. QRF fits read observed donor prefixes and
require the exact raw recipient prefix, so a single hybrid five-target chain
could not keep that branch unchanged
(`packages/microcosm-fit/src/microcosm/fit/qrf.py:649-662,1151-1230`).

Those are required semantics, not an existing overlay API. The current public
path derives draw eligibility only from missing targets in that call, fills
only nulls, starts a chain with an empty completed-target prefix, and rejects a
target declared in two ordinary families
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:967-1043,1461-1523,2283-2334`).
`REFIT_SHA` must therefore add a typed post-compatibility overlay executor with
inputs `{family_id, draw_scope, per_target_write_masks, fixed_prefix_order,
donor_prefix_frame, recipient_raw_prefix_frame, separate_bank}`. IRA uses its
saved complement for both scope and write mask. The late overlay uses the
four-mask union as draw scope and each original complement as its write mask.
For `[401(k), 403(b), SEP]`, append the always-finite fixed prefix columns in
exact order `[keogh_distributions, tax_exempt_ira_distributions]` after the
required and realized optional predictors; donor prefixes are exact clone-0
Keogh plus unchanged clone-1 tax-exempt IRA, and recipient prefixes are their
refit/compatibility raw draws. Pass these as ordinary fixed predictors, not as
fabricated completed targets. Keep all overlays in a separate ordered registry
and invocation so the compatibility family declarations and banks remain
byte-identical. The executor must authenticate scopes/masks before permitting a
non-null overwrite and reject every out-of-mask diff. Its bank identity binds
the scope, masks, prefix order/bytes, compatibility bank, and overlay registry.
Required regressions cover compatibility bytes, a producer-owned-Keogh row in
the union, prefix dtype/index/order/hash, uninterrupted-versus-resumed equality,
and zero diffs outside write masks. The complete contract is frozen in
`HOST_25PCT_PLAN.md`.

Pin the exact family strings
`source_operator_retirement_distributions__clone0_keogh_overlay` and
`source_operator_retirement_distributions__clone0_accounts_overlay` (overlay
registry IDs `late/person/source_operator_retirement_distributions__clone0_keogh_overlay`
and
`late/person/source_operator_retirement_distributions__clone0_accounts_overlay`).
Use only the normal hash-derived family/pattern seeds—never selected names or
numeric seeds
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2902-2916`).
Apply overlay draws only to the four red targets' existing target-specific
recipient complements. Require both the compatibility raw bank and terminal
tax-exempt IRA column to remain byte-identical and preserve every
producer-owned cell. The current transfer snapshots and verifies those cells
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8408-8440,8518-8538,8631-8729`).

For both overlays, preserve the declared predictor blocks, availability
construction, importance weights, 100 trees, zero tolerance, build/model seed
input, `max_targets_per_fit=8`, producer masks, and gates. The unchanged
compatibility families must retain their exact batching, order, and derived
seeds. The overlays deliberately have pinned one-target and one-plus-three-target
topologies; their pinned IDs determine new numeric family/pattern seeds through
the unchanged hash derivation
(`packages/microcosm-build/src/microcosm/build/us/spec/imputation.yaml:2405-2463`;
`packages/microcosm-build/src/microcosm/build/us/spec/imputation.yaml:66-94`;
`packages/microcosm-fit/src/microcosm/fit/qrf.py:83-150`;
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:1247-1380,1792-1811`;
`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:8518-8538,8631-8729,11306-11308,11523-11640,11860-11931`).
Persist sign counts and regimes for every pattern. For Keogh, recompute the
regime from its realized support and run the unmodified QRF behavior; abort only
on existing validation errors, and never add, synthesize, or reweight a carrier
to satisfy a gate
(`packages/microcosm-fit/src/microcosm/fit/qrf.py:118-150,1333-1407`). This
treatment is proposed and must be tested on the frozen f025 host rung.

## Amount-shape detail at f001

For the seven adequately supported early legs, the audited weighted
p10/p25/p50/p75/p90 positive-amount quantiles are:

| Leg | ASEC quantiles | ACS quantiles |
|---|---|---|
| tax-exempt pension | 1,175.88 / 2,460 / 6,396 / 15,472.99 / 28,700 | 934.8 / 2,460 / 6,277.92 / 13,403.8379 / 24,600 |
| taxable pension | 1,692.12 / 3,540 / 9,204 / 22,266.01 / 41,300 | 1,416 / 4,354.2 / 10,266 / 21,240 / 37,524 |
| taxable IRA | 1,699 / 4,272 / 11,000 / 19,200 / 105,336 | 1,699 / 3,600 / 8,400 / 13,264.282 / 20,000 |
| SS retirement | 7,979 / 12,720 / 18,840 / 26,400 / 33,000 | 4,879 / 11,905 / 18,360 / 25,200 / 33,600 |
| SS disability | 7,796 / 11,760 / 13,979 / 17,696 / 21,179 | 7,160 / 9,241 / 12,041 / 17,696 / 21,179 |
| SS dependents | 4,441 / 10,560 / 14,041 / 18,164 / 25,764 | 11,963 / 14,041 / 22,548 / 22,548 / 30,352.2312 |
| SS survivors | 12,000 / 15,876 / 22,000 / 26,400 / 26,832 | 12,000 / 12,000 / 22,968 / 22,968 / 26,400 |

These use the battery's stable weighted inverse-ECDF implementation and
normalized envelope distance
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11918-11931,11990-12021`).

## Implementation boundary and 25% decision

No retirement production derivation is changed in this lane. The five
account-distribution model changes need a dense 25% fit. The two pension and
four Social Security concept mismatches need owner adjudication of their exact
declared-absence equations and target roles before either a derivation change
or a model refit is authorized. This lane added no exclusions, ran no pool
build, and changed no gate, band, ceiling, fold, seed, or target order. The host
must use the commands pinned in the root `_LANE-NOTES.md` with those controls
unchanged.

The acceptance sequence for any reviewed candidate is:

1. authenticate the frozen f025 inputs and candidate commit;
2. run exactly one serial, RSS-guarded 25% pool build;
3. verify persisted donor masks, sign counts, and realized regimes;
4. verify all source identities and producer byte identities;
5. evaluate the unchanged full terminal battery and frozen Phase-P controls;
6. treat any passing or failing result as evidence for that exact candidate,
   not permission to tune controls or infer causality from f001.
