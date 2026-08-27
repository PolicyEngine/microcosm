> Historicized 2026-08-26: This is the adjudication packet and run-receipt history for the #630 UK gate adjudication. It is superseded as live state by the signed register and the battery; the content below is retained as historical evidence, not current operational status.

# microcosm#630 gate adjudication — measurement and verification receipts

Campaign runs for the #630 close-out PR (#706). All full-scale runs: certified input
`f17306cc…`, seed 42, `--qrf-estimators 100`, `--sample-seed 578`; rung runs at
`--sample-fraction 0.10`. Every value below is a ratio, share, count-of-columns,
tolerance, or digest per CD171 §5.2.1 — no unit-record weights or small cells.

## Adjudication history (why two measurement runs)

1. The original owner ruling on finding 2 proposed re-baselining the weight-ratio
   fence to the staged surface's measurement (1590.5346779161957, bit-stable across
   five reproductions).
2. **Flip (2026-08-17):** the adversarial review of #705 decomposed the post-SPI
   weight distribution: the excess was 11–12 SPI-synthetic clone rows from two donor
   lineages inheriting extreme calibrated weights — deterministic but concentrated.
   "Stable is not structural." The fence stays at the exact certified June bound
   (1151.2542195939373); the resolution is upstream data repair.
3. **#710** fixed the SPI prior allocation (per-stratum mass over sampled quota,
   no donor-weight propagation). The m4/m5 runs below are on the fixed surface;
   m2/m3's minted edges were superseded and re-minted.

## Run table

| run | code | scale | release id | logbook row | verdict |
|---|---|---|---|---|---|
| m1 (prior) | be405573 (pre-#691) | f100 | uk-654-m1-seed42-20260814 | `36d85d5151…` | blocked: weight_ratio + input_mass (the #630 findings) |
| m2 | 4159c444 (post-#691/#692) | f100 | uk-654-m2-seed42-20260817 | `5a622d6794…` | blocked as expected; its ratio measurement (1590.5346779161957, fifth bit-exact reproduction) became evidence in the #705 flip |
| b1 pre | d3c7d5e0 (pre-#691) | f010 | uk-654-b1-f010-seed42-20260817 | `03c1fb26b5…` | evidence side A |
| b1 post | 4159c444 | f010 | uk-654-b1-f010-seed42-20260817 | `2d0c532297…` | evidence side B; comparison PASS (below) |
| m3 | pre-flip PR branch | f100 | uk-654-m3-seed42-20260817 | `8ab1a78045…` | superseded: passed only under the withdrawn re-baseline; retained as chain history |
| m4 — measurement | 1131576b (post-#710 main) | f100 | uk-654-m4-seed42-20260817 | `f716aa9620…` | weight_ratio PASSES at the June bound; minted the receipted edges below |
| m5 — confirmation | PR #706 branch (post-#710) | f100 | uk-654-m5-seed42-20260817 | `8ee736948d…` | **full battery passed at the June bound** — superseded as the close-out receipt by m6 after the #709 rebase (14 of the current 16 gates) |
| m6 — confirmation | PR #706 branch, rebased onto post-#709 main | f100 | uk-654-m6-seed42-20260818 | `6492afe85e…` | **all 16 gates green** — the #630 close-out receipt (below) |

## Run m4 — post-#710 measurement (the receipted values)

- **`uk_weight_ratio` passes at the certified June bound**: measured ratio
  **681.5534900252635** (Max's #710 offline run printed the same value at 10
  significant figures), ESS fraction 0.131008, total mass bit-equal
  28840551.182180054. The fence held; no re-baseline.
- `uk_input_mass_parity.relative_tolerance` → **4.521811483823806** — the worst
  surviving |drift| (`is_enhanced_disabled_for_benefits`, the same edge column as
  the superseded pre-#710 measurement) after the sole breach
  (`charitable_investment_gifts`, +15,168.8% on the fixed surface,
  SPI-channel-exclusive) became the per-reference reviewed exclusion. Next
  drifts: 3.0587 (`sda_reported`), 2.1690 (`jsa_income_reported`).
- `minimum_reference_total` → **0.0** — measured at floor 0.0: columns_checked =
  128, columns_below_reference_floor = 3 (exact-zero reference columns, skipped
  at any floor); the surviving-drift edge is an economically meaningful column,
  so 0.0 costs nothing and maximizes coverage.
- `uk_qrf_tail_concentration.max_top_share` → **0.9970712395200448** — exact
  measured maximum top-share (`charitable_investment_gifts`); second-highest
  0.8859278040084307 (`hmrc_spi_taxable_termination_pay`).
- `min_nonzero_records` → **274** — the thinnest measured column
  (`hmrc_spi_taxable_termination_pay`); count-based, unchanged by the weight fix
  as predicted. `top_k` stays 100 (measurement-grid anchor).
- Only failure: input-mass on `charitable_investment_gifts` — resolved by the
  per-reference exclusion this PR lands. New #703 gate `uk_nonnegative_columns`
  passed.

## Run (b) — 10% rung bit-identity across the #691 typed-weights flip: PASS

Deferred verification from PR #691 ("10% rung bit-identity of gate evidence pre/post
the flip"). Identical commands, fresh checkpoint dirs, identical dependency versions.

Bit-identical between d3c7d5e0 (pre) and 4159c444 (post): every gate's
status/failures/details/reason; the full `evidence_sha256` map;
`gate_outcomes_sha256`; `pipeline_sha256`; run_config (minus `code_identity`);
every stage's content identity (per-table column lists, row counts, content sha256).

Expected movers, observed at exactly their recorded values: `policy_sha256`
(`2586535b…` → `b147b503…`), `gates_manifest_sha256` (`6308ee13…` → `6a989153…`),
`spec_fingerprint` (`48993fd9…` → `da0039af…`) — the #691 `evidence_absent_blocks`
flag riding the policy hash — plus the attestation signature over the moved digests.

One artifact-level difference, investigated and classified benign-by-design: the three
stage checkpoint container files differ in bytes while their content identities are
bit-equal. Probe result: the pre-side checkpoint's household table carries the
`household_weight` column and the post-side does not — precisely #691's declared
change (typed weights are the only in-build weight state). All remaining dataset-key
differences are positional renumbering from that one dropped column; the typed
`weights` payload metadata is bit-equal. Gate evidence is untouched by the flip.

## Runs m5/m6 — confirmation on the PR branch: the #630 close-out receipt

Full battery pass at the spec-armed receipted thresholds, weight_ratio at the
**certified June bound** — the first staging build to pass the fully-armed
terminal battery since the gates landed, on the #710-fixed surface. Three parity
gates (`uk_export_surface`/`uk_target_surface`/`uk_target_fit`) record
`evidence_absent` (no calibration on the staging path; non-candidate posture), so
`shippable: false` — this certifies gates, not a release.

**Re-confirmed as m6 after the #709 rebase.** #709 added two release-blocking
terminal entries (`uk_take_up_signal`, `uk_brma_enum_domain`), so m5's 14-gate
report no longer covered the battery; rather than let the "fully-armed" claim
decay, the confirmation was re-run on the rebased branch. m6 evaluates all
**16 gates: 13 passed, the same three parity gates `evidence_absent`, none
failed** — the two #709 entries pass on the national staging path. m6's staged
artifact is **payload-identical to m5's** (`compare_uk_h5_payload.py`:
`payload_identical: true`, zero value mismatches in every table, no root-attr
drift), which is itself the receipt that the rebase moved policy only and not
one byte of built data. The committed build record is re-cut from m6 and embeds
the 16-gate signed report.

Digest note, for exactness: m6 ran before the review-response prose edit to the
`uk_input_mass_parity` entry notes, so its report carries manifest
`6e08fe4c…` / fingerprint `9a9613eb…` while the committed spec now pins
`610512a5…` / `cb25537c…`. The **policy digest is identical either side**
(`609075af…`, verified by recomputation): notes ride the manifest digest, the
reviewed thresholds ride the policy digest, and no threshold moved after the
run.

Observed: weight_ratio 681.5534900252635 vs bound 1151.2542195939373; input-mass
passed against `efrs-post-calibration` with the `charitable_investment_gifts`
exclusion in force (stale/expired/premature all empty); QRF passed at
100 / 0.9970712395200448 / 274 with no thin columns; `uk_nonnegative_columns`
(#703) passed; report digests policy `5060cc95…` / manifest `1ce7d4c0…` /
fingerprint `ef567048…` (mirrored in `contract.py` by the producer-recomputed
pins); `household_weight_total` bit-equal 28840551.182180054.

Payload comparisons:

1. **m5 vs m3** (isolates #710, both post-#691 exports): differences are exactly
   `household.household_weight` on 200,000 rows — precisely the SPI-synthetic row
   set the #710 per-stratum allocation repairs — and `person.capital_gains` on
   15,170 rows (~1.3%), the CGT stage's weighted draw re-running on the repaired
   weights (#710's "post-CGT weights array-equal" receipt binds the weights; the
   drawn values legitimately follow them). No column-order or root-attr drift;
   benunit and time_period payload-equal.
2. **m5 vs the e726abd8 rebuild** (the #635 transitive target — the pinned June
   staging is `local_untracked` by charter; the retained rebuild is
   payload-adjudicated equivalent to it in the #612/#654 record): the same two
   named changes plus the pre/post-#691-era deltas already receipted on m3 —
   `person.capital_gains` on 183,151 rows (the #676/#693 CGT family postdating
   the July artifact), household column order only (sets and values equal), and
   the mass-log root attr. All other tables and columns payload-equal. Bytes
   moved for named reasons — the case the #635 charter re-cuts
   `national_staging_build_record.json` for; the committed record is re-cut
   verbatim from this run (schema 3).

## The input-mass tolerance's sensitivity floor (review question, #706)

Review question (vahid-ahmadi, 2026-08-18): the shared `relative_tolerance`
4.521811483823806 is minted honestly as the worst surviving drift, but its edge
column looks *explained* rather than noisy — so the fence may be granting every
other column a ±452% unexplained-regression budget. The receipt named only the
top three drifts, so the full distribution is recorded here.

**Full comparison surface, m5/m6 staged candidate vs the `efrs-post-calibration`
reference (128 columns compared at floor 0.0; |relative drift|):**

| band | columns | cumulative above band floor |
|---|---:|---:|
| ≥ 10 | 1 | 1 |
| 4 – 10 | 1 | 2 |
| 3 – 4 | 1 | 3 |
| 2 – 3 | 1 | 4 |
| 1.5 – 2 | 1 | 5 |
| 1 – 1.5 | 2 | 7 |
| 0.5 – 1 | 15 | 22 |
| 0.25 – 0.5 | 40 | 62 |
| 0.1 – 0.25 | 26 | 88 |
| 0.01 – 0.1 | 35 | 123 |
| < 0.01 | 5 | 128 |

Median 0.2419, p75 0.4217, p90 0.6259, p95 1.1509. The single column above 10
(`charitable_investment_gifts`, 151.69) is the per-reference reviewed exclusion.

**The decay is smooth — there is no natural cut point.** Ranks 2–8:
`is_enhanced_disabled_for_benefits` 4.5218, `sda_reported` 3.0587,
`jsa_income_reported` 2.1690, `lump_sum_income` 1.5237, `education_grants`
1.1557, `bsp_reported` 1.1509, `non_residential_property_value` 0.9026. Moving
explained drifts into the register and re-minting at the worst *unexplained*
survivor therefore buys tightening only in proportion to how many columns earn a
receipt:

| columns excluded | re-minted tolerance | tightening |
|---:|---:|---:|
| 1 (today: charitable only) | 4.5218 | — |
| 2 | 3.0587 | 1.48× |
| 4 | 1.5237 | 2.97× |
| 7 | 0.9026 | 5.01× |
| 20 | 0.5167 | 8.75× |
| 30 | 0.4401 | 10.27× |

So a 5–10× tightening requires receipting **7 to 30 columns**, not the three-to-five
the review sketched; conversely the register does *not* need to absorb a dense
1.5–4.5 band (only four columns sit above 1.5).

**Receipt status of the top drifts, today.** Only the edge column has a
documented explanation: microcosm#703's head-to-head receipt verified the
disability categories and flags agree row-for-row at 100.0000% with the
incumbent's own `create_frs` at the pinned revision, attributing the residual
delta against the frozen reference to the artifact's vintage (rev `655dd07e`
predates the disability-logic fixes) plus the SPI-synthetic composition (~37% of
persons), and schedules re-measurement at matching composition at E10 — which is
both a real receipt and a natural expiry anchor. `sda_reported` and
`jsa_income_reported` are plausibly the same compositional story (FRS-reported
legacy-benefit columns under a shifted benefit population) but carry no
verification of their own, and this register does not accept unevidenced reasons.
Excluding only the one receipted column re-mints the fence at 3.0587 — a 1.48×
gain that does not change the gate's practical sensitivity.

**Disposition.** The tolerance stays at the measured edge for this PR, with the
limitation recorded rather than silent: at 4.5218 this gate is a gross
mass-loss/explosion fence, not a regression detector, and a 2× mass move in a
mid-band column passes it. The tightening path is the per-reference register this
PR lands, and it is gated on evidence per column — tracked as follow-up work
anchored on the #703 receipt precedent and the E10 recomposition expiry.
