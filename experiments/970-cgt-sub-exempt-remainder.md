# UK CGT sub-exempt gainers: remainder amounts, incidence anchor and projection fence (microcosm#970)

*2026-09-22. Branch `uk-970-cgt-sub-exempt`. The licensed receipts live outside the tree, under
`data/ukds/acceptance/970-cgt-sub-exempt/` and `runs/uk-623-first-calibrated/spine-assessment-v22/`; aggregate-only
extracts of every number quoted below are committed under `docs/evidence/uk-cgt-970/`. The v20 numbers come from the
`uk-equalising-cgt` dashboard run of 2026-09-21 (PolicyEngine/uk-equalising-cgt PR #3) and the HMRC Capital Gains Tax
statistics 2026 release.*

## Question

The staged v20 candidate (`staged/uk-spine-assessment-v20-calibration`) carried 11.66 million weighted people with
capital gains at or below the £3,000 annual exempt amount, 10.82 million of them at exactly £3,000.00: the amounts
stage capped every gainer beyond HMRC's published taxpayer mass at the exempt amount, and the equal-mass incidence
clone left half of all household mass on gainer households. policyengine-uk freezes the exempt amount and uprates gains
by per-capita GDP growth, so every one of them became a CGT taxpayer in the first projected year: 11.54 million
taxpayers in 2026-27 against 597 thousand on the incumbent Enhanced FRS and 551 thousand in HMRC Table 1 for 2024-25.
Issue #970 fixes this in three parts, and this note records what each changed on the licensed rebuild.

## What changed

1. **Amounts** (`hmrc_cgt_gains_spine`): the remainder takes amounts from the Advani-Summers Table A1 within-band
   gains distribution restricted to the quantiles between each band's zero crossing and its crossing of the exempt
   amount, rank-preserving within the band and deterministic (no seed consumed). The table is used as published
   (2017-18 nominal), with no uprating; for the first income band the crossings sit at quantiles 0.1805 and 0.2354.
2. **Incidence** (`cgt_incidence_anchor`, new 31st spine stage, weights only): the sub-exempt and loss-making clone
   households are trimmed to the reporter composition implied by the redrawn liable mass and the Advani-Summers
   crossings (sub-exempt = liable × (q_AEA − q_0) / (1 − q_AEA), losses = liable × q_0 / (1 − q_AEA)), with a factor
   rising in the band's incidence and capped at one; every unit of mass a clone loses goes to its paired original, so
   pair mass, household mass and every non-CGT aggregate are conserved to rounding. Liable clones, zero-gain clone
   carriers (no reporters) and band donors are untouched. The groups, targets and receipt are clone-side quantities,
   not population counts: band donors and originals are outside the anchor. Gate:
   `uk_stage_cgt_incidence_anchor_composition`, which certifies the anchor's arithmetic against its declared
   composition. The first licensed build was blocked by that gate:
   a global exact-total correction borrowed from the clone stage had moved the summed rounding of 26,288 transfers
   onto one household (1.6e-13 relative on its pair); the fix conserves pairs arithmetically and records the total as
   realised.
3. **Fence** (`uk_cgt_projection_entrants`, calibration seam): the cumulative stock of build-period sub-exempt gainers
   whose uprated gains exceed each year's exempt amount is counted to 2030; the largest count must not exceed the
   vendored HMRC Table 2.1a taxpayers in the £3,000–£5,999 band (73,000 in 2024-25), people already above the exempt
   amount, so a plausibility ceiling rather than an entrant count. 2030 is the last year of the OBR per-capita growth
   path in the engine's parameter tree (later years repeat the 2030 rate); the path and the exempt amount are pinned in
   the gate's manifest entry and drift-checked, so an engine bump fails the gate visibly instead of moving its verdict.

## Measurements

Before: the fence evaluated on the calibrated v20 artifact, and the seam receipt of calibrating the v20 spine through
this branch (signed report, blocked at terminal, no H5).

- Entrants by uprating: 10,832,804 in 2025 rising to 10,978,624 in 2030, against the 73,000 bound. Sub-exempt weighted
  persons 11,664,957, of which 10,815,254 at exactly £3,000; the 10th, 50th and 90th percentiles of sub-exempt gains
  are all £3,000.

After: the 31-stage spine rebuilt on 24509d78 from the same licensed inputs as spine-s (all 21 spine gates pass), then
calibrated as v22 with the v20 recipe (frozen register pin, 1,500 epochs, family_equal). María's signed deferral of
the self-employment £20k–£30k band (90bdb809, not on main) was cherry-picked onto a measurement branch so the recipe
matched v20; without it main's `uk_target_fit` fails that band at +25.3%.

- Anchor receipt (spine, clone side only): clone sub-exempt 11,291,667 → 40,729; loss-making 2,707,829 → 150,250; liable clone mass
  486,720 untouched; liable mass 557,420; reporter composition q_0 = 0.2008, q_AEA = 0.2552 (implied reporters
  748,399); 25,436 of 26,288 clone households trimmed, none capped; largest pair error 2.2e-16.
- Remainder receipt (spine): 20,642 persons placed in (£0.24, £2,999.76], all 61 Advani-Summers bands represented.
- Calibration: 7/7 seam gates pass. Loss 0.3015 → 0.01038 (v20 0.01041); 96.6% of 638 targets within 10% (v20
  96.4%); CGT targets 61 of 65 within 10% (worst: tax by age 65–74 +14.0%, liability total +10.0%). ESS 4,907 (v20
  9,305). Calibrated liable mass 550,202 carrying £117.4bn of gains.
- Fence (calibrated frame): PASSED. The stock of crossers is 12,711 by 2025, 22,331 by 2026, 43,377 by 2027, 50,416
  by 2028, 54,321 by 2029 and 67,611 by 2030, against 73,000: about 13k a year, so the margin is one year of trend, and
  the demoted-donor decision below is what would restore it. Sub-exempt weighted persons 326,532, median £2,116, none
  at exactly £3,000.
- Pass-2 head-to-head against the incumbent (scored from the deferral tree, whose scoring-prepare fix 1f45bdf3 the
  eval script needs): 500–24 on 524 common targets, candidate loss 0.0103 against the incumbent's 0.2171; the CGT family
  73–2 (v20: 493–26 on 519, loss 0.0096 vs 0.211, CGT 73–2).

## What the fence found next

Of the 67,611 entrants in 2030, 61,219 are band donors and 6,392 are clones. The Table 3 redraw ranks every gainer in
an income band by its prior and fills the published cells from the top; it does not distinguish the band donors, so
119 of the 270 donors fall into the sub-exempt remainder, almost all in the £12.3k–£250k bands where the clones'
Advani-Summers tails outrank the band means (12.3k band 30 of 30, 25k band 29, 50k band 27, 100k band 27, 250k band
5, 1m band 1, none above). They carry 321,300 weighted persons before calibration and 291,099 after. The same
demotion existed on v20, where those donors sat at the £3,000 cap inside the 10.82 million. The donor stage's notes
describe the donors as support households and exclude bands below £12.3k because "the spline body already supplies
that support", so a demoted donor is redundant support that has kept its mass as a sub-exempt gainer. How to treat
them (protect donors in the redraw's cell selection, drop their mass, or stack donors only where the clone pool is
short) is a separate decision this branch does not take.

## Status

Code, manifests, fixtures, pins and the licensed receipts above are on the branch. The full `microcosm-build`
shard on the final tree passed (8,458 tests, 0 failures, 0 errors, 48 licensed or network-gated skips, 2 h 47 min).
Still open: the dashboard measurement (after the staged upload) and María's ruling on demoted donors.
