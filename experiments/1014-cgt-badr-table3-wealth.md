# #1014 receipts: BADR-qualifying gains, the Table 3 income margins and wealth-conditioned gain carriers

Plan: `repos/microcosm-1014-implementation-plan.md` (approved 2026-09-26). María's rulings: BADR is drawn
inside the asset-type stage and the claimant's main type follows the claim; Table 4.1 is bound as eight
bands by claimants and qualifying gains; Table 3's margins are measured on the engine's taxable income;
every new row sits in `hmrc_cgt`; one PR carries all three work packages, with wealth conditioning in its
last commits so its effect is measured on its own. Worktree `repos/populace-1014`, branch
`uk-cgt-badr-table3-1014`. The commits were built and measured on main `2bce66156`, rebased onto main
`9e5b0cee2` (#998's test layout and CI, no UK source change), then onto main `937aca4ec`, which merges
#1012 (the SPI-first order and the SPI housing shell), then onto main `6a70cd4ee`, which merges #901 (the UK
full-build graph: the spine driver moves into the package and two inert HMRC tail stages retire, 36 to 34
stages) and #1007 (typed calibration diagnostics). The commit hashes below are the pre-rebase ones.
Parts A to D measure c1 to c4 against main before #1012; Part E records how they carry over onto
#1012's base. Licensed evidence lives under `data/ukds/acceptance/1014-cgt-badr/` (aggregates only).

## Part A: the Chronicle landing and the re-pin (c1)

PolicyEngine/chronicle#287 asked for HMRC CGT statistics 2026 Table 4; Chronicle PR #288 landed it at
`505e0e72aa7c82b96dc06941d84fa797a629161c` as package `hmrc-cgt-badr-ir-2026` (126 facts: claimants,
qualifying gains and tax by band of qualifying gain, individuals, trusts and all taxpayers, 2021-22 to
2024-25). The feed was rebuilt from that commit with `chronicle build-bundle --suite uk` then
`build-consumer-artifact`:

- `consumer_facts.jsonl` sha256 `4a45c543553617d606ebf8ea11254b48c1e6e115c97840dd8757beafbac035d9`
- `manifest.json` sha256 `3dc9ed05ba00d30609c295cb7d34ebcd98476778bebf1da9a1e7f2e64b97caf7`
- 287,150 rows, `policyengine_ledger.consumer_artifact.v2`, consumer-fact schema `chronicle.consumer_fact.v4`
  (`cc9efb90…`, unchanged); 158 source packages
- a strict superset of the `5324aa2` artifact: every one of its 287,024 rows is present byte for byte,
  and the 126 new rows are all Table 4 (31, 31, 32 and 32 for 2021-22 to 2024-25)

The re-pin moves no compiled value on either surface: the national (1,124) and local (20,885) reference
files are byte-identical, the three compile-parity receipts are unchanged, and the vendored resources,
census, validation-level register and membership reports restate the feed identity; the coverage
manifest records the conditioning resource's new digest.

## Part B: BADR claims on the licensed spine (c2)

Spine `spine-badr` built at c2 `1ffa11762` with the PLAN_5 assessment relaxation in the measurement tree
only (never committed); every spine gate passes, including the extended `cgt_asset_type_summary` check.
The draw runs over liable gainers not flagged residential. The residential flag is drawn first on its
own seed and is byte-identical to the control spine's (202,669 people and GBP 12.82bn flagged, against
targets of 202,630 and GBP 12.24bn).

Per band, the logistic solves each published band exactly in expectation and the weighted systematic
walk lands within one pool weight of the count:

- GBP 0 to 9,999: 5,096 claimants against 5,000, GBP 33.1m against 32.0m
- GBP 10,000 to 24,999: 7,741 against 8,000, GBP 125.7m against 129.0m
- GBP 25,000 to 49,999: 6,235 against 6,000, GBP 233.1m against 224.0m
- GBP 50,000 to 99,999: 7,778 against 8,000, GBP 564.8m against 575.0m
- GBP 100,000 to 249,999: 10,933 against 11,000, GBP 1,886.7m against 1,866.0m
- GBP 250,000 to 499,999: 8,009 against 8,000, GBP 2,776.7m against 2,901.0m
- GBP 500,000 to 999,999: 8,028 against 8,000, GBP 5,941.7m against 5,929.0m
- GBP 1m and over (claims at the lifetime limit): 6,793 against 6,787, GBP 6,793.1m against 6,787.0m

In total 60,613 claimants and GBP 18.355bn of qualifying gains (published 61,000 and GBP 18.443bn; the
count target is 60,787 because the open band holds 6,787 claims at the limit against a rounded 7,000),
and the closed-form tax at the relief rate is GBP 1.819bn against the published GBP 1.821bn. All six
invariants are zero. Claimants' main types: unlisted shares 56.3k people (GBP 16.1bn qualifying), other
non-financial assets 2.3k, business land and buildings 2.0k. The restricted type fit converges in 31
iterations; claimants hold 32.5 percent of the non-residential gains against the eligible types' 63.5
percent Table 7 share.

The realized Table 7 composition, which is fenced and diagnostic, drifts further from its targets than
the control's: unlisted shares 66.6 percent against 55.3 (control 62.1), other financial assets 20.5
against 26.7 (control 26.9). The fit matches expected shares; the categorical draw over heavy-tailed rows
does not reproduce them, and confining claimants to the eligible types concentrates large gains in
unlisted shares.

## Part C: calibrations (c2 to c4)

All four arms ran pass 1 of `repos/uk-candidate-eval/scripts/run_uk_national_calibration.sh` (the
national release role: 1,500 epochs, `family_equal`), through a wrapper that never re-freezes the shared
scoring register. The control is `spine-ctl` calibrated as `spine-assessment-spifirst-ctl-main` (main
`8f628b1e7`; main has no UK change between it and `2bce66156`). The candidate spine `spine-badr` serves
c2, c3 and c4, since c3 and c4 change only targets and measures. Every arm stops at the terminal gate
on the CGT entrants fence (#970, out of scope here): 146,920, 146,507, 146,635 and 146,680 weighted
sub-exempt gainers against the bound of 73,000. The comparisons use `calibration_diagnostics.json`, the
run's design and final weights and the engine at the 2024 disposal year
(`scripts/cgt_phase_a.py`, output `phase-a/cgt-phase-a.json`).

Headline at final weights (control, c2, c3, c4):

- targets 1,062, 1,062, 1,078, 1,090; loss 0.0091, 0.0087, 0.0086, 0.0086; within 10 percent 97.4,
  97.8, 98.0, 98.0 percent; rows beyond 25 percent 2 in every arm (the same two non-CGT rows)
- effective sample size 4,529, 4,557, 4,526, 4,517; top 1 percent weight share 27.0, 26.9, 27.0, 27.0
- CGT liability GBP 25.38bn, 22.53bn, 23.31bn, 23.35bn against the published 22.503bn (+12.8, +0.1,
  +3.6, +3.8 percent); gains and taxpayers within 0.2 percent in every arm
- BADR claimants and qualifying gains: none; 82.6k and GBP 26.01bn; 61.0k and GBP 18.61bn; 60.9k and
  GBP 18.64bn (published 61.0k and GBP 18.443bn); relief-rate tax GBP 2.58bn at c2, 1.85bn at c3 and c4
- share of gains above GBP 125,140 of the engine's taxable income: 46.5, 44.1, 44.8, 55.4 percent
  (published 55.4; 56.6 at prior weights)
- OBR income tax -0.5 percent in every arm; the 33 SPI, ITL and regional rows above GBP 200,000 average
  0.82, 0.93, 0.83 and 0.75 percent absolute error

**c2 (BADR on the spine, no new rows).** Liability falls from +12.8 to +0.1 percent and every tax-by-age
row fits (the control's 35 to 84 rows sat 9 to 19 percent over). Unbound, calibration inflates the
claims to 82.6k and GBP 26.0bn: taxing more gains at the relief rate is the cheapest way to meet the
liability row.

**c3 (Table 4.1 bound).** Fifteen of the 16 rows land within 0.6 percent. The open band's pair cannot
both hold, because every claim there is exactly the limit (count 6,787 on the gains row against a
rounded 7,000): the solver leaves the count at -0.6 percent and the gains at +2.6 percent. With the
claims back at their published size, liability rises to +3.6 percent and the 45 to 74 tax-by-age rows
to +3.3, +4.0 and +7.7 percent.

**c4 (Table 3's margins bound on the engine's taxable income).** All 12 rows land within 0.4 percent and
the share above GBP 125,140 matches HMRC. At prior weights the spine matches the published taxpayer
counts by income band on the redraw's own proxy but not the gains: the GBP 125,140 to 199,999 band
holds GBP 21.0bn on the proxy (24.0bn on the engine's income) against 10.9bn published, and the 50,000
to 99,999 band 22.0bn against 17.2bn. The excess is not in the redraw's plan, which holds the joint's
cell means (GBP 119.05bn): the realized draw carries GBP 135.3bn of liable gains, almost all of the
difference in the GBP 5m-and-over band (GBP 62.7bn from 3,559 people against 48.5bn from 3,000
published). The control spine is identical here, so this predates #1014. Calibration removes the
excess by down-weighting the largest rows, and the Table 3 rows now decide which income columns give
it up. Liability ends at +3.8 percent and the 55 to 74 tax-by-age rows at +5.5
and +7.6 percent. The costs sit on the gain rows: the effective sample size of weighted gains falls
from 43 (control) to 35, the share of gain households whose weight more than doubles rises from 8.0
to 8.9 percent, and the heaviest row (a GBP 15.1m gain) carries 5.1 times its prior weight against 2.5
in the control.

Against the plan's acceptance: c2 meets the liability and tax-by-age bar; c3 holds Table 4.1 within
rounding except the open band's structural pair; c4 holds the Table 3 margins and the share above
GBP 125,140; the effective sample size and the entrants fence do not move beyond noise, and no other
family degrades. Two items remain open: liability at +3.8 percent with the older-age tax rows 5 to 8
percent over once the claims are pinned, and the concentration of the calibration's gain corrections
on a few large rows. Correcting the redraw's GBP 5m-and-over overshoot at the stage would take most of
that correction out of calibration; it is a follow-up, not part of this PR.

## Part D: tests

Each commit ran its targeted suites before landing (c1: 251 passed, and the national compile parity
regenerates on the pinned feed; c2: 814; c3: 517; c4: 560, including the engine-inversion lockstep for
the derived taxable income). The whole shard at c4 `a2bd0be23` (the `spine-uk`, `uk:build`, `uk:frame`
and `shared-spec` groups plus microcosm-data and microcosm-graph, 270 files, with the pinned feed
configured) ran 6,482 tests: no failures or errors, and 41 skips, all for optional or licensed local
inputs (the axiom rules engine, rulespec checkouts, licensed tabs, unmounted ladder artifacts).
`tools/ci_test_groups.py --verify` is clean.

After the rebases the moved tests sit in #998's layout (the observation-period tests split between
the engine-free and UK engine directories, their shared helpers in `test_support/`) and the layout
check is clean. On #1012's base each commit's generated surfaces (coverage manifest, H2 fixture, gate
digests) are regenerated from the merged tree, and the references and parity receipts regenerate
unchanged. PR CI runs every group on the rebased head.

## Part E: wealth conditioning (WP3)

Commits c5 (the wealth-blended ranking in the amounts redraw, weight 0.75) and c6 (the stock-conditioned
odds in the asset-type stage) are on the branch, María's pick of 2026-09-29 from the investigation below.
The investigation needed #1012's household wealth, so it ran on a local tree of #1012 (its head before the
final review round) and c1 to c4 (c4′), comparing seven arms of the plan's three candidates with c4′. c1 to c4 carry over onto that base: against #1012's own control, liability
moves from +12.8 to +3.7 percent, the Table 3 and Table 4.1 rows fit, and the effective sample size is
5,173 against 5,200. The recommendation, which María took: a wealth-blended ranking in the amounts
redraw and stock-conditioned odds in the asset-type stage, as two commits on this branch; hold
wealth-coupled incidence until the redraw can place heavy records in the open top bands without
overshooting them.

On c4′, gains are close to wealth-blind: within Table 3's income columns their Spearman correlation
with household investable wealth is 0.08, and 40 percent of GBP 1m-and-over gainers hold under GBP
250,000 of it. The ranking at weight 0.75 raises the correlation to 0.46 and leaves no such gainer.
With the ranking at 0.5, the stock-conditioned odds take the BADR claimants and unlisted-shares
gainers with neither corporate wealth nor self-employment income from about half to about 5 percent.
Each arm calibrates about as well as c4′ (loss and effective sample size within noise), though the
worst tax-by-age row rises from +7.7 percent to +9.1 with the pair at 0.5 and to +10.6 with the
ranking alone at 0.75.

The pair as committed (the ranking at 0.75 and the odds as declared, business land without a signal)
was then built and calibrated on c4′: the correlation holds at 0.46, no GBP 1m-and-over gainer holds
under GBP 250,000 of investable wealth, and BADR claimants and unlisted-shares gainers without corporate
wealth or self-employment income fall to 3.9 and 4.4 percent (listed-shares gainers without an ISA or
dividends 14.8, residential gainers without another residential property or property income 51.7, from
62.0 and 82.6). Calibration: effective sample size 5,176 against 5,173, liability +3.5 percent, Table 3
within 0.4 and Table 4.1 within 2.8 percent; the worst tax-by-age row is +9.6 percent against +7.7, and
the redraw realizes GBP 138.8bn against its GBP 119.05bn plan (c4′ 135.3bn).

On the merged #1012 (main `937aca4ec`) three arms were rebuilt and recalibrated: a control on that main,
the PR head (c1 to c4 with the #1012 review fixes) and the WP3 pair. The spines the PR head and the pair
build are identical, dataset for dataset, to those built on the pre-merge tree, and so are their
calibrations: #1012's final review round changed gates, the target-fit register and stage health, nothing
the spine or the solve reads. Against the control on the merged main, liability moves from +12.8 to +3.7
percent (GBP 25.38bn to 23.33bn against the published 22.503bn), the worst tax-by-age row from +18.7 to
+7.7 percent, the worst gains-by-age row from 10.8 to 0.3 percent, and the share of gains above GBP
125,140 of taxable income from 46.2 to 55.4 percent, with the effective sample size 5,173 against 5,200,
the loss 0.0084 against 0.0088 and the entrants fence 129,433 against 129,863. The pair on that base:
effective sample size 5,176, liability +3.5 percent, worst tax-by-age row +9.6 percent, fence 99,133, and
the coherence figures above unchanged. Runs `spine-assessment-1014-{ctl1012,c4m1012,wp3m1012}`; outputs
`phase-b/cgt-phase-b-m1012.json`, `wp3/coherence-m1012.json` and `wp3/calib-summary-m1012.txt` under the
evidence directory.

## Part F: the student-loans realization gate (microcosm#1049)

Every licensed spine in this lane was built with an uncommitted relaxation of the
`uk_stage_student_loans_realization` threshold, because since #1006 the gate refused every spine built
at main's head: PLAN_5's reported England count had risen to 9,183 of the 10,000 SLC liable stock, the
top-up's shortfall of 817 people was about one survey person's weight, and the Bernoulli draw realised
it as two rows of 1,718 people (+110 percent against the 1.0 rule) while the final count, 10,901, went
unchecked. The fix replaces the draw with the identity-keyed greedy walk the gas-connection imposition
uses and re-bases the gate on what the walk controls; a pool lighter than its shortfall is receipted as
exhausted rather than refused.

Spine `spine-fix1049` was built at the fix commit with no relaxation: every spine gate passes, the
student-loans gate included. PLAN_5's walk takes 19 rows for 817.1 people against the
817.2 shortfall (gap -0.13, lightest skipped weight 0.30), so England ends at
9,999.9 against the 10,000 stock. PLAN_2's pool of 2,240 rows and 2.04m people is lighter than
its 7.55m shortfall, so it is taken whole and recorded as exhausted at 38.3 percent of the 8.94m
stock, the same assignment as before. The calibration against the PR head's arm on the same base (which
differs only in the PLAN_5 top-up: 19 rows instead of the draw's two): loss 0.0084 against 0.0084,
effective sample size 5,165 against 5,173, CGT liability GBP 23.30bn against 23.33bn, the
worst tax-by-age row 7.5 against 7.7 percent, Table 3 within 0.4 and Table 4.1
within 2.6 percent, and the entrants fence 129,293 against 129,433.

On main `6a70cd4ee` the spine at the rebased head (`spine-m901`, built under #901's driver with no
relaxation) passes all 26 gates; its datasets are identical to `spine-fix1049`'s (the file differs only
in provenance bytes), and its calibration reads loss 0.0084 against 0.0084, effective sample size 5,165
against 5,165, liability GBP 23.30bn against 23.30bn and the entrants fence 129,293 against 129,293.

## Part G: the support split replaces the band donors (c7, c8)

Every licensed spine of this line failed `uk_cgt_projection_entrants` (98,600 to 129,300 against 73,000), and
the receipts placed the whole excess on one mechanism. `cgt_band_donors` stacked 30 copied households per
retained Table 2.1a band at published taxpayers over 30, so the four heavy bands' rows weighed 2,033 to 3,267
people against about 500 for a liable clone; the Table 3 redraw walks each income, age and region cell in
rank order placing whole rows, its second pass never runs on these spines (pass 1 overshoots every income
band), and every row a cell cannot place falls to the sub-exempt remainder, where the mapping hands the largest
priors the top stratum just under GBP 3,000. On `spine-wp3final` 204 of the 270 donors were demoted, carrying
333,833 of the 375,093 sub-exempt mass and 101,033 of the 108,563 crossing mass at design weights (92,371 of
98,611 calibrated); the clones contributed 6,000 to 7,500 on every arm. A replay of the redraw shows that
placement is a wealth cliff rather than a weight problem: the placed share of clone gainer mass is zero below
the eighth investable-wealth decile in every income band, 6 to 32 percent in the ninth and 32 to 45 percent in
the tenth, and c5's wealth-blended rank key raised donor demotion from 120 (m901) to 204 because a donor's
wealth is its source's. A mass-conserving redesign of the stack drawn by income propensity (the design first
approved on 2026-09-29) was therefore predicted to demote 300,000 to 360,000 and fail the fence again, and
María's ruling was to retire the stage.

The redraw owns every gain amount (the Table 3 joint, the Table 2.1a size bands, the age and region margins,
wealth rank within cells) and the calibration binds the twelve Table 2.1a bands at final weights, so the
donors' band-mean values were notional and their one remaining function was row support at the top of the
distribution. c7 supplies that support from the households the redraw already places: `cgt_support_split`,
a deterministic stage before the incidence clone, walks each Table 3 income column in descending investable
wealth until the cumulative weight reaches twice the headroom (2.0) times the published count of gainers at
or above GBP 250,000 in that column (17,000 / 5,000 / 11,000 / 3,000 / 6,000 / 16,000, so 68,000 / 20,000 /
44,000 / 12,000 / 24,000 / 64,000 before the clone, 232,000 in all, 0.8 percent of household mass) and splits
each selected household into ceil(weight / 60) copies at equal weight, ids offset by the clone stage's own
multiplier scheme, every other column unchanged; the graph policy is `conserve`, the coverage family
`mass_conserving`, and the stage has no draw, seed or salt. The graph kernel's lineage rule recovers copy k
from `id + k x offset`; the anchor's pairing needs only the clone flag; `mass_increasing_support` loses its
only user (the SPI income-band donors follow in microcosm#1063). On the H2 fixture the split seats one
household into 167 copies (444 households before the clone, 888 after, 34 stages as before) and the smoke
build runs in 35 seconds of stage time with the split at 0.5 seconds. c8 adds placement receipts to the
redraw, keyed by row type (support family or plain): rows placed and demoted per gain band, the heaviest row
placed per cell and per band, the pass-1 overshoot per income band and the open band against its published
count and gains; it holds no threshold, so the first licensed build measures the design with the redraw and
the anchor unchanged in code.

**Licensed arms.** Four arms settle the two parameters; every build ran at main's gate thresholds with no
relaxation, and every calibration is the national role's doctrine.

- `ch1` (277d95d3e; cap 60, headroom 2.0): all 26 spine gates pass. The split selects 299 households into
  3,764 copies (30,532 households before the clone, 61,064 after) at a conserved 29,422,433. In the redraw the
  support families bring 3,216 gainer rows, 2,173 placed carrying 63,101 people and 1,043 demoted carrying
  30,303, which the anchor then trims with every other sub-exempt clone to the A&S composition (40,738 on a
  40,738 target); the open band lands at 2,871 people and GBP 52.1bn on 94 rows with the heaviest at 254,
  against one row of about 900 and GBP 62.7bn before. The fence reads 7,466 at design weights and 6,771
  calibrated, all of it clones (the support families cross nothing), against the 73,000 bound; loss 0.0085,
  effective sample size 5,228, liability +3.8 percent, Table 3 within 0.3 and Table 4.1 within 3.0 percent.
  One terminal gate fails: `uk_weight_ratio` at 1,210.8 against the certified 1,151.25, the maximum weight
  unchanged (31,449 against 34,734 on the previous arm) and the positive median halved from 53.6 to 26.0
  because 8,126 light family rows joined the frame.
- `ch2` (cap 80, headroom 2.0; commit 817e13f3e, reverted): fails at the spine on the asset-type stage's
  residential gains bound, a gap of GBP 10.29bn against a bound of 5.38bn where every earlier arm, `ch1`
  included, sat at four to five percent of its bound. The split's lighter rows shrank that gate's three-sigma
  envelope from about 11bn to about 5.4bn while a few top-band families carry billions each, so a change of
  the copy granularity re-rolls the residential draw with a ten-billion stake; the cap stays at 60.
- `h15` (cap 60, headroom 1.5): all 26 spine gates pass (residential gap 8 percent of its bound); 223 households
  into 2,847 copies; fence 6,677; the ratio 1,161.0 misses the bound by under one percent, the maximum weight
  having moved to 33,025 and the median to 28.5.
- `h125` (cap 60, headroom 1.25; the configuration the branch carries from c306cc07d): all 26 spine gates and
  all 7 terminal gates pass. 189 households into 2,376 copies (29,144 households before the clone); the
  support families place 1,317 rows carrying 38,300 people and demote 714 carrying 20,732; the anchor sits
  on its 41,000 target; the open band lands at 2,643 people and GBP 47.7bn on 84 rows with the heaviest at
  254; the fence reads 7,513 at design and 6,711 calibrated, all clones; the ratio is 1,079.1 (maximum 33,975,
  median 31.5) against 1,151.25; loss 0.0084, effective sample size 5,198, liability +3.6 percent (GBP
  23.31bn), 550,033 taxpayers and GBP 119.1bn of gains against 549,961 and 119.4bn on `wp3final`, BADR
  61,016 claimants and GBP 18.65bn qualifying, the worst tax-by-age row 6.4 percent against 10.1, Table 3
  within 0.4 and Table 4.1 within 3.0 percent, the share of gains above GBP 125,140 at 55.3 percent, and the
  identity-stability tool passes with the split's E8 recompute green. The residential gap sits at 8 percent
  of its bound.
- `ch3` (c306cc07d, the pushed head; cap 60, headroom 1.25): rebuilt for provenance under the PR's own
  commit, its spine is identical to `h125` dataset for dataset (2,454 datasets, the files differ only in
  provenance bytes) and its calibration reproduces every figure above to the digit: all 26 spine gates and all
  7 terminal gates pass, fence 6,711, ratio 1,079.1, effective sample size 5,198.

The placement replay confirms the mechanism on every arm: the wealth cliff is unchanged for plain clones
(placed share near zero below the eighth investable-wealth decile, 11 percent in the eighth, 35 in the ninth
and 70 in the tenth on `ch1`), while the support families, being the wealthiest rows of their columns, are
placed at 99 percent by mass. The heaviest rows the walk still places in the GBP 250,000 to 1m bands are
plain clones of the SPI income-band donors at 1,496 (their stage stacks them at 2,992 and the clone halves
them), which microcosm#1063 will lighten; the split does not touch them.
