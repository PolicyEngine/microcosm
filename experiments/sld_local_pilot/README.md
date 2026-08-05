# SLD local layer: Utah pilot (populace#625)

First real run of the state-legislative-district per-district layer, on the
**published #512 rebuild** (`populace-us-2024-buildo-acs-local-77e2061-
20260724T110908Z`, artifact `populace_us_2024_acs_local.h5`), 2026-08-05.

## Inputs

- Artifact: the buildo ACS local release above (14,551 Utah households /
  39,679 persons in scope; 861 donor-spine rows with certified tract/block
  geography, the rest ACS-spine).
- Targets: the six ledger ACS 2020-2024 5-year SLD packages (Utah pilot,
  summary levels 610/620; PolicyEngine/ledger branch
  `census-acs-sld-2024`) — 3,744 facts serialized to one facts JSONL
  (sha256 recorded in `sld_layer_summary.json`).
- Membership: the national 2024 SLD membership ladder
  (`tools/build_us_sld_membership_ladder_artifact.py`; 331,449,281
  population conserved exactly; 1,950 upper + 4,833 lower districts;
  artifact sha in the summary).

## Command

```bash
uv run python tools/build_us_sld_local_layer.py \
  --artifact-h5 populace_us_2024_acs_local.h5 \
  --sld-facts ut_sld_facts_2024.jsonl \
  --ladder us_sld_membership_ladder_2024.npz \
  --out-dir ut_pilot_sidecar --epochs 512 --seed 0
```

## Results (receipts in this directory)

- **All 104 districts solved** (29 SLDU + 75 SLDL) under the doctrine
  constants (cap 10.0, ratio 100.0, uniform weights, target-defined
  scales). District pools: 352–671 households (SLDU), 118–279 (SLDL).
- **Fit**: median within-10% share = 1.00 in both chambers; worst district
  0.97 (SLDU) / 0.94 (SLDL).
- **Past-cap census**: 26 target rows past the cap at initialization, all
  26 escaped during the solve; **zero frozen, zero pushed out** — the #492
  dumping dynamic does not appear at this scale.
- **Membership method mix**: 12,972 `puma_cd_county_draw`, 718
  `puma_county_draw` (4.9% — rows whose independently seeded (PUMA, CD,
  county) combination has no joint block support, the measured
  independent-draw incoherence), 490 `tract_exact`, 371
  `tract_split_draw`; gate green, unassigned within bound.
- **Published-median sanity** (B19013, validation-only by doctrine): the
  bracket-calibrated weights reproduce published district medians with a
  median absolute gap of 3.3%, p90 8.0%, max 21.9%; 2 of 104 districts
  flagged past the 15% review threshold.
- **Statewide coherence**: summed district estimates vs summed district
  targets across all 70 (state, metric) sums lie in [0.948, 1.006];
  reported, not constrained.

## Adjudications recorded for the candidate review

1. **The 100x ratio bound binds** — both chambers realize max ratio ~100.0
   vs the artifact-weight anchor. The declared bound is load-bearing;
   whether 100x is the right shipped contract (vs a tighter bound trading
   worst-district fit for stability) is a review decision, not a knob to
   quietly tune.
2. **Thin-district concentration** — minimum district ESS fraction ~0.11
   (SLDU) / ~0.13 (SLDL). Declared in the boundaries statement; the
   register keeps per-district ESS.
3. The two median review flags and the 718 fallback-method rows are listed
   in `sld_local_diagnostics.json` for row-level inspection.

Doctrine held as declared: no per-target or per-district knobs anywhere in
the run.
