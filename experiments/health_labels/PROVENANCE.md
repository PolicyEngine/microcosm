# Health-coverage label provenance and comparability

Date: 2026-08-21  
Lane: `battery-health-coverage-labels`

## Decision summary

The nine health labels do not form one uniformly comparable ACS/ASEC surface.
Two have exact published ACS counterparts and were missing native derivations;
seven require owner adjudication as concept mismatches, and the ASEC side of
one of those seven also had a leaf-selection defect. The lane fixes only the
generating defects. It adds no
battery exclusion, absence rule, register entry, threshold change, or waiver.

The blocker set remains the three checks adjudicated in
`../microcosm-arm-split/experiments/battery_burndown/ADJUDICATION.md:125-127`:
CHAMPVA, Indian Health Service (IHS), and other means-tested coverage. The
four-way experiment found no isolated landable arm
(`../microcosm-arm-split/experiments/phase_p_arm_split/SYNTHESIS.md:5-12` and
`:27-32`). CHAMPVA is therefore one leg of this label audit, not its goal.

The cold round-2 1% receipts are recorded in `ROUND2_DIAGNOSIS.md`. They show
that native `HINS7 == 1` is implemented exactly, while missing 2022/2023 ASEC
recode columns had silently become false. The repaired source contract makes
those incomplete inputs fail closed; the receipt-backed corrected sample
would move the IHS ratio from 2.468303 to 1.664408, without changing either
source operator. Because that faithful result remains outside the band and
the complete-source ratio is 1.942760, IHS is still an explicit owner decision:
retain the nominally matched questions under an approved common population
scope, or declare the detailed comparison structurally absent using the
equation below.

## Runtime path to clone 0

1. The ASEC operator defines all nine source-to-leaf mappings in one exact
   contract, requires every participating source vintage to carry the complete
   nonblank `{1, 2}` surface, then converts `code == 1` and fills only missing
   PolicyEngine leaves
   (`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:57-69`,
   `packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:431-438`,
   `packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:441-487`,
   `packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:555-560`).
   The raw pool validates each source before the pandas union can mask a
   missing-vintage column
   (`packages/microcosm-build/src/microcosm/build/us_runtime/asec_pool.py:240-247`).
   The multispine source-operator runner projects CPS-evidenced rows, executes
   that operator, and merges only its declared outputs by entity ID
   (`packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:2136-2174`,
   `packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:2231-2243`,
   `packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:2724-2832`).
2. The ACS reader now requires and retains only the exact health fields
   `HINS1` and `HINS7`
   (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:72-90`,
   `packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:217-225`,
   `packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:326-365`).
   Native mapping runs immediately after the ACS frame is built
   (`tools/build_us_multispine_pool.py:858-883`). It validates nonblank
   `{1, 2}` codes and emits boolean `HINS*=1` leaves with a source receipt
   (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_inputs.py:115-128`,
   `packages/microcosm-build/src/microcosm/build/us_runtime/acs_inputs.py:192-218`,
   `packages/microcosm-build/src/microcosm/build/us_runtime/acs_inputs.py:378-400`).
   The assembly boundary accepts those outputs only with the exact declared
   source/transformation receipt
   (`packages/microcosm-build/src/microcosm/build/us_runtime/operator_boundary.py:127-190`,
   `packages/microcosm-build/src/microcosm/build/us_runtime/operator_boundary.py:412-490`).
3. All nine labels remain declared in the boolean transfer family
   (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:356-365`).
   The transfer stage is called after source completion
   (`tools/build_us_multispine_pool.py:2454-2479`). If any bounded family member
   needs imputation, every declared member contributes its latent model draw to
   the ordered QRF chain; target-level null masks keep native HINS1/HINS7 values
   byte-for-byte and write only unavailable concepts
   (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:992-1043`,
   `packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:2383-2415`).
4. Physical support cloning happens after pre-clone source preparation
   (`packages/microcosm-build/src/microcosm/build/us_runtime/multispine_pool.py:3539-3563`).
   The origin battery then selects the registered clone index and positive
   weights before splitting ASEC and ACS channels
   (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11523-11580`)
   and calculates the ACS/ASEC weighted-incidence ratio
   (`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11819-11856`).

## Per-label source matrix

The survey definitions below use the [2024 ASEC data dictionary, pp. 76-77](https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asec2024_ddl_pub_full.pdf),
the [2024 ACS questionnaire, p. 13](https://www2.census.gov/programs-surveys/acs/methodology/questionnaires/2024/quest24.pdf),
the [2024 ACS PUMS dictionary, pp. 37-38](https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.pdf),
and the [2024 ACS Subject Definitions, pp. 80-83](https://www2.census.gov/programs-surveys/acs/tech_docs/subject_definitions/2024_ACSSubjectDefinitions.pdf).
Every ASEC mapping named in the table is implemented in
`packages/microcosm-build/src/microcosm/build/us_runtime/cps_carried.py:57-69`;
every currently unavailable ACS label enters transfer through
`packages/microcosm-build/src/microcosm/build/us_runtime/acs_transfer.py:356-365`.

| PolicyEngine person label | ASEC source and derivation | ACS source or absence | Clone-0 producer after this lane | Disposition |
|---|---|---|---|---|
| `has_esi` | `NOW_GRP == 1` | `HINS1 == 1` (employer or union) | Native ACS mapping; ASEC source operator | **Derivation defect fixed:** loader + native mapping + receipt |
| `has_indian_health_service_coverage_at_interview` | `NOW_IHSFLG == 1`; every pooled vintage must carry a valid recode | `HINS7 == 1` (Indian Health Service) | Native ACS mapping; validated ASEC source operator | **Two derivation defects fixed:** native ACS mapping and missing-vintage fail-closed contract; comparison still needs owner adjudication |
| `has_marketplace_health_coverage_at_interview` | `NOW_MRK == 1` | `HINS2` identifies direct purchase; `HIMRKS` does not separate unsubsidized Marketplace from non-Marketplace direct purchase | ASEC-derived label transferred to ACS | **Concept mismatch** |
| `has_non_marketplace_direct_purchase_health_coverage_at_interview` | `NOW_NONM == 1` | Same `HINS2`/`HIMRKS` surface as Marketplace | ASEC-derived label transferred to ACS | **Concept mismatch** |
| `has_medicaid_health_coverage_at_interview` | `NOW_CAID == 1` after this lane; it incorrectly used composite `NOW_MCAID` before | `HINS4` combines Medicaid, Medical Assistance, and other government-assistance plans for low-income or disabled people | ASEC-derived label transferred to ACS | **ASEC derivation defect fixed; cross-survey concept mismatch remains** |
| `has_other_means_tested_health_coverage_at_interview` | `NOW_OTHMT == 1` | No separate PUMS field; folded into the broader `HINS4` public-coverage concept | ASEC-derived label transferred to ACS | **Concept mismatch** |
| `has_tricare_health_coverage_at_interview` | `NOW_MIL == 1` (current TRICARE) | `HINS5` is TRICARE **or other military health care** and receives eligibility edits | ASEC-derived label transferred to ACS | **Concept mismatch** |
| `has_champva_health_coverage_at_interview` | `NOW_CHAMPVA == 1` | No separate ACS PUMS field; write-ins are edited into published types without a documented 2024 CHAMPVA destination | ASEC-derived label transferred to ACS | **Concept mismatch** |
| `has_va_health_coverage_at_interview` | `NOW_VACARE == 1` | `HINS6` asks about VA enrollment, but the public edit documentation does not establish whether CHAMPVA write-ins also enter that published bucket | ASEC-derived label transferred to ACS | **Concept mismatch under strict provenance** |

The seven unavailable detailed ACS concepts are intentionally not guessed from
`HINS2`, `HINS4`, `HINS5`, `HINS6`, or `HIMRKS`. The public ACS form has no
separate CHAMPVA checkbox, and the Subject Definitions say write-ins are edited
into the seven published types or rejected without publishing a CHAMPVA
destination. An older Census programming page associates CHAMPVA with a VA
family estimate, but that is evidence for an owner-reviewed aggregate, not a
proven row-level identity for either detailed label: [ACS programming guidance](https://www.census.gov/topics/health/health-insurance/guidance/programming-code/acs-estimates.html).

## Exact equations requiring owner adjudication

Let `1[P]` be the indicator of predicate `P`. For a person row `r`, define the
current battery base scope, directly matching the implemented selector at
`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11523-11533`,

```text
B(r) = 1[person_support_clone_index(r) = 0 and person_weight(r) > 0].
```

### Honest declared absence for a non-comparable detailed label

For each label `L` in

```text
{marketplace, non_marketplace_direct_purchase, medicaid,
 other_means_tested, tricare, champva, va},
```

the exact structural-absence equation an owner-approved comparator declaration
would need is

```text
A_L(r) = B(r) * 1[person_support_channel(r) = "acs"]
B(r) = 1  =>  isna(L(r)) = A_L(r)
T_L(r) = B(r) * (1 - A_L(r)).
```

Equivalently, in the clone-0 positive-weight scope, every ACS cell and no ASEC
cell is null. `T_L` then has no ACS support, so the detailed cross-origin ratio
is untestable and its honest disposition is documented non-comparability. The
implemented comparator already requires an exact live structural-null mask and
rejects unexpected nulls or filled structural rows
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:11545-11592`).
Its only current declared absence rule is ACS group-quarters rent
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:1880-1899`),
and unsupported rules fail closed
(`packages/microcosm-build/src/microcosm/build/us_runtime/stacked_spine.py:7729-7738`).
This lane does **not** add the health rules; that decision is owner-only.

For IHS, the source names are nominally matched, so it is not silently added
to the seven-label concept-mismatch set above. If the owner adjudicates its
survey-frame/reference construction as non-comparable after applying the
common-universe analysis below, the corresponding exact candidate is

```text
A_IHS(r) = B(r) * 1[person_support_channel(r) = "acs"]
B(r) = 1  =>  isna(IHS(r)) = A_IHS(r)
T_IHS(r) = B(r) * (1 - A_IHS(r)).
```

As with the seven detailed-label equations, implementation would require both
the exact live structural-null surface and an owner-approved comparator rule.
This lane implements neither.

### Published aggregate alternatives

These are the exact decompositions or candidate equations the owner can assess.
They are not implemented by this lane.

1. Marketplace/direct purchase:

   ```text
   HINS2 = marketplace_subsidized OR marketplace_unsubsidized
           OR non_marketplace_direct_purchase
   HIMRKS = 1  => marketplace_subsidized
   HIMRKS = 2  => marketplace_unsubsidized
                  OR non_marketplace_direct_purchase

   owner candidate:
   1[NOW_MRK = 1 OR NOW_NONM = 1]  <=>  1[HINS2 = 1]
   ```

   `HIMRKS=2` leaves two unknowns in one cell, so neither detailed ASEC label is
   recoverable. Only the combined direct-purchase equation is identified.

2. Medicaid/CHIP/other means-tested:

   ```text
   1[NOW_MCAID = 1]
     = 1[NOW_CAID = 1 OR NOW_PCHIP = 1 OR NOW_OTHMT = 1]

   owner candidate:
   1[NOW_MCAID = 1]  <=>  1[HINS4 = 1]
   ```

   The fixed PolicyEngine Medicaid leaf stays `NOW_CAID`; the composite is
   offered only as a possible comparator aggregate.

3. TRICARE/other military:

   ```text
   1[HINS5 = 1] = 1[TRICARE OR other_military]
   1[NOW_MIL = 1] = 1[TRICARE]
   ```

   An exact aggregate is unavailable unless the owner identifies an ASEC
   current `other_military` residual; treating `HINS5` as TRICARE alone is not
   an equality.

4. VA/CHAMPVA:

   ```text
   owner candidate only (not a proven identity):
   1[HINS6 = 1]  ~=  1[NOW_VACARE = 1 OR NOW_CHAMPVA = 1]
   ```

   The 2024 public edit documentation does not prove this candidate, so neither
   separate label nor the aggregate should be declared comparable without
   owner evidence.

### Common population scope needed by any approved aggregate

ASEC and ACS health estimates use different population frames; Census documents
that difference in its [ASEC/ACS health measurement note](https://www.census.gov/programs-surveys/cps/technical-documentation/user-notes/health-insurance-user-notes/health-ins-cov-meas-asec-acs.html)
and [survey-universe guidance](https://www.census.gov/data/what-is-data-census-gov/guidance-for-data-users/frequently-asked-questions/the-difference-between-the-universes-for-the-acs-decennial-and-cps.html).
An owner-approved aggregate comparison therefore also needs the explicit common
scope

```text
U_health(r) = B(r) * (
    1[channel(r) = "asec" and A_MJIND(r) != 14]
  + 1[channel(r) = "acs" and TYPEHUGQ(r) in {1, 3}
                          and ESR(r) not in {4, 5}]
)
```

before computing either origin's weighted incidence. `TYPEHUGQ` is retained by
the ACS loader (`packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:66-71`),
but `ESR` is not in the retained ACS person fields
(`packages/microcosm-build/src/microcosm/build/us_runtime/acs_pums.py:72-90`),
and the ASEC raw closure retains `A_MJOCC`, not `A_MJIND`
(`packages/microcosm-build/tests/test_us_plan.py:1081`,
`packages/microcosm-build/tests/test_us_plan.py:1112-1136`). The owner must
authorize the missing source fields and prove the intended active-duty coding;
this lane does not substitute `A_MJOCC` for `A_MJIND` or alter comparator scope.
