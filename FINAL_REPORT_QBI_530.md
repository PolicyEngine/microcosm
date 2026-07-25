# Final report: QBI v2 content (populace #530)

## Outcome

The adjudicated QBI v2 content is complete on local branch
`qbi-v2-content`, based on local `repeal-validation-298` commit `e45f797`.
The work stayed offline and is committed locally for the supervisor to push.

Version 1 remains unchanged. Version 2 now loads a live SSTB crosswalk,
applies the evidence-anchored content priors, and passes the fixed-seed PUF
replay diagnostic.

## What landed

### Live SSTB crosswalk and host wiring

- Added and declared `us/sstb_crosswalk_v1.json` with `status: "live"`.
- Preserved both adjudicated maps in one spec-only resource:
  - 27 Census 2017 industry entries;
  - 101 Census 2018 occupation entries.
- Preserved 10 industry and 11 occupation explicit non-SSTB documentation
  rows at probability zero, the legal basis, wiring notes, and the
  reputation-or-skill income-type-test note.
- Encoded the binding probabilities:
  - `clear_sstb`: `1.00`;
  - rationale containing `lean non-SSTB`: `0.10`;
  - hospital/facility industry codes `8191`, `8192`, `8270`, and `8290`:
    `0.25`;
  - the adjudicated ambiguous health technician/assistant occupation tier:
    `0.20`;
  - all other ambiguous entries: `0.30`;
  - absent codes and explicit non-SSTB documentation codes: `0.00`.
- Every ambiguous entry is provisional and cites the Section
  1.199A-5(c)(1) de-minimis rule as the low-prior basis.
- Updated the strict loader for the live resource schema and leading-zero
  Census codes. The host classifier consumes per-code probabilities, uses a
  configured industry field first, falls back to occupation, and resolves
  unmapped codes to zero.
- V2 currently wires `PEIOOCC` as the available host signal and keeps
  `industry_column: null`. The industry map ships as the required future seam.
- Retained `sstb_crosswalk_placeholder.json` for explicit fail-closed tests.

### Qualification content

- `self_employment_income`: derived; rationale records the active
  trade-or-business source and positive Schedule C host gate.
- `farm_operations_income`: derived as an active farm trade-or-business
  source.
- `partnership_s_corp_income`: provisional `0.90` prior, explicitly proxying
  guaranteed-payment and reasonable-compensation exclusions. The rationale
  cites JCT's 53% S-corporation and 17% partnership shares pending v3.
- `farm_rent_income`: `0.80` prior for unresolved trade-or-business facts.
- `rental_income`: `0.70` prior for unresolved Section 199A safe-harbor facts.
- `estate_income`: weakest prior at `0.60`, explicitly flagged for v3.
- Restored the partnership/S-corporation qualification flag to the v2 QRF
  target surface, which now has 52 person targets and 61 total targets.

## Passive SSTB prior derivation

OTA WP-118 supplies the complete arithmetic used in the resource:

```text
Population restriction effect:
  (221.00 - 162.76) / 221.00
  = 58.24 / 221.00
  = 0.263529...
  -> 0.264 below the coarse threshold

Residual after the income exception:
  (221.00 - 183.46) / 221.00
  = 37.54 / 221.00
  = 0.169864...
  -> 0.170 above the coarse threshold
```

JCT's above-threshold share is
`120.81 / 216.08 = 0.559098...`, or 55.91% of deduction dollars. The
assumptions rationale uses that statistic only to establish that the upper
band is material; it does not treat it as a conditional SSTB rate. The
`$200,000` AGI boundary is explicitly marked as a provisional coarse proxy
for filing-status-specific modified-taxable-income thresholds.

## REIT/PTP and BDC scale

The REIT/PTP metadata records the persisted published anchors:

- SOI Publication 4801 TY2023 Form 8995 line 6 income: `$21.07B`;
- SOI Publication 4801 TY2023 line 9 component: `$4.20B`;
- JCT 2022 comparison component: `$2.90B`.

The dividend exposure and both v1 Beta scales remain fixed. Raising only the
partnership/S-corporation receiving probability from `0.05` to provisional
`0.09` gives this fixed-seed restricted PUF replay:

```text
Weighted REIT/PTP income: $20,943,037,788.761116
Anchor factor:             0.993974266197
Provisional allowed band:  [0.3, 3.0]
```

The replay diagnostic reads the published anchor and factor band from the
assumptions resource and asserts that the weighted aggregate remains inside
the band.

No published BDC aggregate was present in the persisted evidence. The v1 BDC
probability and Beta scale therefore remain unchanged, with a provisional
small-scale regulatory-economics rationale. Its fixed-seed replay is
`$143,270,889.638501`.

## Tests and repository contracts

Added or updated coverage for:

- live crosswalk schema, exact probability tiers, ambiguous-entry bases,
  explicit zero-probability documentation, and occupation-code format;
- real-crosswalk v2 host routing on a synthetic frame;
- unmapped-code zero behavior and occupation-primary wiring;
- qualification modes, source-stage/QRF target selection, and evidence
  rationales;
- REIT/PTP published-anchor metadata and restricted PUF replay;
- spec-only package loading, incumbent guards, country planning, source
  runtime, base-builder dispatch, and the existing placeholder fail-closed
  behavior.

The v1 golden tests were not changed and remain green.

Final full-workspace command:

```text
UV_CACHE_DIR=/private/tmp/populace-wt-530-uv-cache \
UV_OFFLINE=1 UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=.venv \
POPULACE_PUF_2024_H5=/Users/maxghenis/ops/populace-qbi-port/assets/puf_2024.h5 \
POPULACE_RAW_PUF_2015_CSV=/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2015.csv \
uv run pytest
```

Result: **3,260 passed, 132 skipped, 0 failed** in 116.70 seconds.

Also green:

- full-workspace `uv run ruff check .`;
- `uv run ruff format --check` on all six changed Python files;
- `git diff --check`;
- spec-only, incumbent-reference, entrypoint-heuristic, and manifest contracts.

An extra whole-workspace `ruff format --check` reports 34 pre-existing,
untouched files outside this branch's diff. They were deliberately not
reformatted as part of this content change.

## Adjudication status

No binding adjudication was left unimplemented.

The unmapped reputation-or-skill field, null industry host column, provisional
priors, retained BDC scale, and in-tree placeholder are deliberate specified
outcomes rather than omissions.
