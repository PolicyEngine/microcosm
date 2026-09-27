# SLD local layer: Utah pilot (populace#625)

Pilot of the state-legislative-district per-district layer, on the
**published #512 rebuild** (`populace-us-2024-buildo-acs-local-77e2061-
20260724T110908Z`, artifact `populace_us_2024_acs_local.h5`). The receipts
in this directory are the **corrected run** (2026-08-07) after the
cross-family sol review of the first run's committed diff; the corrections
and what they changed are recorded below, not silently absorbed.

## Inputs

- Artifact: the buildo ACS local release above (14,551 Utah households /
  39,679 persons in scope; 861 donor-spine rows with certified tract/block
  geography, 12,073 ACS-spine housing units, 1,617 ACS-spine
  group-quarters person-rows).
- Targets: the six chronicle ACS 2020-2024 5-year SLD packages (Utah,
  summary levels 610/620; PolicyEngine/chronicle branch
  `census-acs-sld-2024`, 3,744 facts; sha in `sld_layer_summary.json`).
- Membership: the national 2024 SLD membership ladder
  (`tools/build_us_sld_membership_ladder_artifact.py`; 2020 apportionment
  population conserved exactly at 331,449,281; 1,950 SLDU + 4,833 SLDL
  districts; embedded boundary-vintage metadata validated at load).

## Command

```bash
uv run python tools/build_us_sld_local_layer.py \
  --artifact-h5 populace_us_2024_acs_local.h5 \
  --sld-facts ut_sld_facts_2024.jsonl \
  --ladder us_sld_membership_ladder_2024.npz \
  --out-dir ut_pilot_sidecar --epochs 512 --seed 0
```

## Results

- **All 104 districts solved** (29 SLDU + 75 SLDL) under the doctrine
  (cap 10.0, stretch 100.0 vs the artifact-weight anchor, uniform weights,
  fixed 1e-4 anchor floor — zero anchors actually floored).
- **Fit**: within-10% share median 1.00 / min 0.97 (SLDU); median 0.97 /
  min 0.91 (SLDL).
- **Past-cap census**: 25 target rows past the cap at initialization, all
  escaped; **zero frozen, zero pushed out** in both chambers.
- **Membership method mix** (identical across both runs — order-stable
  seeded reproducibility): 12,972 `puma_cd_county_draw`, 718
  `puma_county_draw` (4.9%, the measured independent-seeding incoherence),
  490 `tract_exact`, 371 `tract_split_draw`; gate green on both row share
  and weight share; zero certified-tract degradations.
- **Household universe**: 1,617 group-quarters rows support the population
  age bands only; household counts and income brackets bind on housing-unit
  households (TYPEHUGQ 1) — the B19001/B19013 universe.
- **Published-median sanity** (B19013, validation-only): median absolute
  gap 2.5%, p90 8.0%, max 16.0%; 2 of 104 districts flagged past the 15%
  review threshold.
- **Statewide coherence** (targets include every district; zero-support
  targets: none): solved/target across the 70 state-metric sums in
  **[0.865, 1.009]**. The low tail is concentrated in the low-income
  brackets — `sldl income_15000_to_19999` 0.865, `income_10000_to_14999`
  0.914 — where thin SLDL districts hit the ratio bound before attaining
  the bracket mass: a genuine artifact-support finding (fewer low-money-
  income households than the district ACS tabulates), not a solver defect.

## Review corrections (first run -> this run)

The sol review of the first committed diff returned **block**; every
confirmed finding was fixed and the pilot re-run:

1. **Group quarters**: the first run counted 1,617 GQ person-rows as
   households in the B19001/B19013 universes. Now excluded (age bands
   unchanged — S0101's universe is total population).
2. **Money-income recipe** (verified against policyengine-us variable
   definitions): dropped `tip_income` (employment income already includes
   tips — double count), added `sstb_self_employment_income_before_lsr`
   (disjoint from non-SSTB), replaced `farm_income` (Schedule J averaging)
   with `farm_operations_income` (Schedule F), and restricted person-level
   components to members aged 15+ (the ACS universe).
3. **Membership integrity**: cross-state geography components are ignored
   (district codes repeat across states); zero-population cells fall
   through to coarser conditioning; the ladder NPZ embeds and the loader
   enforces boundary-vintage metadata; the gate now also bounds the
   unassigned **weight** share, requires upper codes, and polices
   no-lower-chamber states in both directions.
4. **Reporting honesty**: coherence uses the true (unfloored) artifact
   anchor and includes zero-support districts; the achieved-vs-target
   table carries `district_status`; the boundaries statement states the
   fallback ladder and degradation counts instead of overclaiming full
   conditioning; medians validate on the household universe only; the
   sidecar records software identity.
5. **Doctrine surface**: `min_initial_weight` removed from the wrapper
   signatures (fixed doctrine constant, recorded in the doctrine record).

Net effect on results: median-income sanity **improved** (median gap 3.3%
-> 2.5%, max 21.9% -> 16.0%); the SLDU bound stopped binding entirely
(max realized ratio 89.5); SLDL fit tightened slightly (median 1.00 ->
0.97) because bracket targets must now be attained without GQ rows — and
the low-bracket support gap above became visible. That trade — slightly
worse-looking headline, honest universes — is the point.

## Standing adjudications for the candidate review

1. The 100x stretch bound binds in 15 of 75 SLDL districts (0 of 29 SLDU);
   its level remains a review decision, and the low-bracket support gap is
   the concrete case study.
2. Thin-district concentration: min district ESS fraction ~0.12 (SLDU) /
   ~0.08 (SLDL); per-district ESS in the register.
3. National-run posture: the district compile is one-pass; the fixed-format
   store read is full-table (documented memory note in the tool).

Doctrine held as declared: no per-target or per-district knobs anywhere.
