# A16 exclusion rows measured on the v20 national candidate

Evidence for the 2026-09-21 retirements on [PolicyEngine/microcosm#967](https://github.com/PolicyEngine/microcosm/pull/967) (commit `4173ab38`), filed on the review's ask that the retirement evidence live in the repository. `receipts.json` carries the digests of the candidate measured, the register and the two files here.

## What was measured

Every entry on the calibration measure-exclusion register (67 on 2026-09-21) is compiled in the contract register but pruned from the calibration surface, so the candidate never solved toward it. `scripts/measure_a16_rows_on_candidate.py` resolves and scores the full frozen contract register (705 specs) on the v20 national candidate's calibrated weights, pruning loudly only the rows whose measures cannot resolve at all, and reports each excluded row's relative error. A row inside the 25 % fence at weights that never saw it is evidence its exclusion can be retired; the retirement itself remains a signed register change.

Result (`outputs/a16_rows_on_v20.json`): 67 excluded entries, 62 measured, 5 unresolvable on the candidate, 10 inside the 25 % fence.

## The rows retired in `4173ab38`

- `hmrc/employment_income_count_income_band_1_000_000_to_inf` (microcosm#757): +16.9 % unbound
- `dwp/uc_payment_dist/LONE_PARENT_annual_payment_10_800_to_12_000` (uk-data#452): -8.1 % unbound
- `hmrc/self_employment_income_count_income_band_500_000_to_1_000_000` (microcosm#757): +0.0 % unbound
- `dwp.benefit_cap.capped_households_200_01_to_300` (microcosm#882): -10.1 % unbound
- `dwp.benefit_cap.capped_households_400_01_to_500` (microcosm#882): -3.8 % unbound

### What the retirements do and do not say

Three of the five rows were signed for *support*, not error size: `hmrc/employment_income_count_income_band_1_000_000_to_inf` (zero in-band carrier records on total income), `hmrc/self_employment_income_count_income_band_500_000_to_1_000_000` (four carriers under the SPI concept, sixteen under the proxy) and the two benefit-cap bands (a capped population resting on a handful of source households cloned many times; the two count as one concern). A +0.01 % or −3.8 % measured on a cell that thin does not speak to the concern those reasons named, that the solver distorts neighbouring cells to hit a published value with too few carriers. The retirement rests on the register's fence discipline (the row measures inside the fence at weights that never solved toward it), and the support concern stays live as a diagnostic: the cells are measured on every evaluation and the next build's target-fit diagnostics say whether binding them moved their neighbours (review note on PR #967, third pass). The other two rows (`dwp/uc_payment_dist/LONE_PARENT_annual_payment_10_800_to_12_000`, signed as a one-sided undershoot; the employment 1m+ row's sibling concern is above) were signed for error, which the measurement answers directly.

## The row measured inside the fence but kept

- `dwp.uc.households_housing_element` (microcosm#882): -2.8 % unbound. This is not noise inside the fence: DWP's any-tenure count includes an "other or unknown" tenure the engine's housing element cannot pay, 112,518 of 4,037,650 in calendar 2025 (2.79 %), so the model's any-tenure count is identically social plus private and the measured gap is the structural gap the entry records. María signed that reading in review of microcosm#921 on 2026-09-15; `4173ab38` had retired the entry on the measurement, and this PR restores it (the entry stays a measured-not-fit diagnostic to its 2026-10-15 expiry).

Four uk-data#452 payment bands measuring 18–25 % unbound stay excluded: binding-time competition can push a marginal cell over the fence, and the v20 self-employment 20–30k deferral on the target-fit register is that mechanism in action.

## Provenance

The evaluation stack is not a public repository, so the script is copied rather than linked (it is untracked there; the receipt digest is of the copy as taken, before this repository's formatter). Paths inside the output are machine-local. The candidate is the v20 national build (`microcosm_uk_2024.h5`, sha `8883e5925679…`), the same file the #968 poverty evidence measured.
