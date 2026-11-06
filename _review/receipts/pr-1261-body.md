Carry Schedule E line 32 by entity on the US PUF donor; retire the S-corp exact-zero universe 2026-10-31T16:08:17Z
Resolves #1258. **Stacked on #1251.** This PR targets `main` so CI runs, which means its diff also contains #1251's commits until #1251 merges. This PR's own commits start at `85eb94b7b`; to see only them, compare against `us-puf-line32-passive-1239`.

The US PUF donor now carries the partnership and S corporation shares of Schedule E line 32 as separate leaves:

- `partnership_income` = E25940 + E25980 − E25920 − E25960 − E26110
- `s_corp_income` = E26170 + E26190 − E26160 − E26180 − E26100

Before this PR, the whole line-32 total sat in `partnership_income`, and `s_corp_income` was an exact-zero universe enforced in four places. This PR retires that machinery. It also binds SOI Table 1.4's partnership and S corporation legs to each entity's own leaf.

Max ruled yes on d1489 and d2183 (see "Decision"). This PR lands last, after #1251 and #1259.

## Why

From TY2021, SOI Table 1.4 publishes partnership and S corporation net income and net loss separately. For TY2022:

| | Net income | Net loss |
|---|---|---|
| Partnership | $487.1B (3.31M returns) | $175.6B (2.09M) |
| S corporation | $812.7B (4.03M) | $93.0B (1.52M) |

A frame whose S-corp leaf is zero by construction can bind only their sum. The raw TY2015 PUF carries every field the split needs.

## The PUF fields

The roles come from the IRS SOI General Description Booklet for the 2014 Public Use Tax File (annotated Schedule E Part II). The same table appears in #1251.

| Line 28 column | Partnership | S corporation |
|---|---|---|
| (f) passive loss allowed | E25920 | E26160 |
| (g) passive income | E25940 | E26170 |
| (h) nonpassive loss | E25960 | E26180 |
| (i) section 179 expense deduction | E26110 | E26100 |
| (j) nonpassive income | E25980 | E26190 |

Each entity's share is (g) + (j) − (f) − (h) − (i) of its own columns. Checked on the pinned TY2015 file (`puf_2015.csv`, sha256 `0a7fd643…`, all 207,692 regular records; `out/entity_split_stats.txt`):

- **The entity shares partition the same fields as passive and nonpassive.** So partnership + S corp equals passive + nonpassive exactly.
- **They equal E26270 within the booklet's per-field rounding on every record.** E26270 differs from the field sum on 38,378 records (18.5%). The largest residual is 0.985 of the record's bound, and the weighted absolute residual is $0.135B on $621.9B net.
- **The donor carries the field sums, not E26270.** No field says which entity a rounding residual belongs to. Pushing E26270's residual into either entity would make 3,860 S-corp-only records partnership carriers, and 12,987 partnership-only records S-corp carriers.

| TY2015, regular records | Net | Income (returns) | Loss (returns) |
|---|---:|---:|---:|
| Partnership | $238.3B | $316.8B (3.29M) | $78.5B (1.91M) |
| S corporation | $383.6B | $434.1B (3.53M) | $50.5B (1.27M) |
| Line 32 (E26270) | $621.9B | $735.8B (6.15M) | $113.9B (2.69M) |

Other shares of regular returns:

- 25,315 records (1.17M weighted) carry both entities.
- 10,274 records (0.49M weighted) carry the two with opposite signs.

## The change

**Donor (`puf_source_schedule_e.py`, `puf_support.py`)**

- `schedule_e_partnership_s_corp_columns` returns each entity's share. The line-32 rounding identity is now checked against their sum.
- `source_year_puf_partnership_s_corp_income` returns `partnership_income`, `s_corp_income` and `passive_partnership_s_corp_income`.
- For the four disclosure aggregate rows, each entity's total, and the passive total, is allocated over the row's synthetic records in proportion to their processed nonpassive amounts. This is the same rule #1251 used for line 32: every bucket keeps its published totals by entity.
- The receipt adds per-entity legs and counts, and the stored-E26270 residual.
- The donor supplement must carry all three leaves. The processed nonpassive column no longer aliases `s_corp_income` to zero, so a processed donor that requests any entity leaf without the supplement now fails with "cannot derive".
- The raw-field path takes its entity legs from the same constants.
- Passive placement and zeroing refuse an imputation pass that lacks either component (a partial or misordered basis is refused before any QRF fit).

On the real inputs (211,677 processed tax units, TY2015 dollars on the processed weights; `out/real_donor_vectors.json`):

| Leaf | Net | Income (returns) | Loss (returns) |
|---|---:|---:|---:|
| Partnership | $250.6B | $340.2B (3.52M) | $89.6B (2.04M) |
| S corporation | $426.0B | $491.6B (3.78M) | $65.6B (1.36M) |

- The line-32 total is $676.562B, the same as #1251's to the dollar shown.
- The passive vector is byte-identical to #1251's (sha256 `6c76e077…`).

**Retired: the S-corp exact-zero universe**

- **Stacked spine:**
  - `_materialize_us_puf_s_corp_universe_zero`, `us_puf_s_corp_universe_zero_rule_identity` and `US_PUF_S_CORP_UNIVERSE_ZERO_RULE_ID`;
  - the `whole_pool_output_universes` doctrine, in the primary execution config, the PUF-pass receipt and the late-resource validator;
  - the pass's `s_corp_income_universe_zero` receipt.

  Native rows now stay null until the late transfer fills them, as `partnership_income`'s always have.
- **Spec:** the doctrine is gone from `imputation.yaml` (regenerated), `imputation.schema.json` and `imputation_semantics.py`.
- **Late ACS transfer:** `s_corp_income` is no longer excluded from `puf_tax_itemization`. It joins `partnership_income` in batch 3, so the passive draw conditions on both entities. The post-transfer passive reconciliation now requires both line-32 components rather than reading a missing S-corp leg as zero.
- **Late producer registry:** `s_corp_income` is a PUF-clone primary output like every other primary target.
- **Pool:** QBI reconciliation's `s_corp_income` input is provisioned by the pool input surface, like `partnership_income`. The `primary_puf_exact_zero_universe` provision is gone.
- **Fiscal refresh:** the reviewed degenerate-input exclusion for `s_corp_income` ("constant zero") is removed. The gate fails a stale exclusion, and the leaf is no longer constant.

The execution-config payload keeps `schema_version` 5, and the registry keeps schema 17. Receipts embed the full content, and the exact doctrine-equality check refuses a pre-#1258 receipt; a test covers this. Bumping the versions would have cascaded into worker-identity authentication for no added safety.

**Table 1.4 entity legs (`fiscal_targets.py`, `variable_labels.py`)**

The measure ids are Chronicle's (PolicyEngine/chronicle#180, as specified in chronicle#346):

| Measure ids | Concept | Leaf | Part |
|---|---|---|---|
| `partnership_net_income_{returns,amount}` | `partnership_income` | `partnership_income` | positive |
| `partnership_net_loss_{returns,amount}` | `partnership_losses` | `partnership_income` | negative-part magnitude |
| `s_corporation_net_income_{returns,amount}` | `s_corp_income` | `s_corp_income` | positive |
| `s_corporation_net_loss_{returns,amount}` | `s_corp_losses` | `s_corp_income` | negative-part magnitude |

The materializer sums each person leaf to the tax unit before taking the part, which is return grain, as for the estate/trust legs.

- **The pinned feed carries none of these ids today,** so the compiled registries are unchanged. On both #1251's head and this branch (`out/compile_*.json`):
  - full registry `f5bd921d6b21` (33,250 specs);
  - `national_state` `e1f9906bf0e0` (6,102);
  - the 1,098 HT2/CD A26270 rows are identical.
- **When the legs land, they bind.** I cloned the feed's own TY2023 all-returns Table 1.4 estate/trust facts into the eight entity-leg facts, with SOI's published TY2023 values (`scripts/compile_with_entity_legs.py`). All eight compile onto `national_state`, each binding its own leaf (`out/compile_with_entity_legs.json`).
- HT2/CD A26270 keeps binding the combined `tax_unit_partnership_s_corp_income`, which is consistent with the entity legs: line 32 is their sum.

**Re-pinned, every value re-derived by the repo's tools or from the observing code**

- `tools/generate_us_bundle_from_constants.py`: `imputation.yaml`, `battery.yaml` (s_corp sign-separated metric), `catalogs.yaml` (domain `puf_tax_itemization`), and its reviewed counts (late 19/72, the itemization split [8,8,8,8,7], graph inputs 2747, compiled outputs 230).
- `tools/repin_us_pool_engine_contracts.py`: the remaining-stage manifest digest and the engine ABI lock.
- `inventory_coverage.EXPECTED_HASHES` and counts, from `build_inventory_coverage` observations: authority, full checkpoint, graph nodes, late families, late resource semantics, late schedule, seed protocol and seed map.
- `field_usage` pins (battery, imputation family execution, imputation producer graph, totals) and the pointer-inventory digest. `tools/spec_engine_coverage.py --check` passes: 42,244/42,244 fields, 41/41 checks.
- `tools/graph_us_post_transfer_fixture.py`: the H3 fixture's pool digest line.
- The US spec digest `02c4227f…` (engine-workflow multispine tool test) and the loader golden vector.
- Counts in tests and docs, each moved by the one added transfer target:

| Count | Before | After |
|---|---:|---:|
| Late transfer targets | 71 | 72 |
| Transfer surface | 119 | 120 |
| Terminal surface | 135 | 136 |
| PUF-clone producer targets | 44 | 45 |
| Numeric late targets | 52 | 53 |
| Sign-separated battery metrics | 80 | 81 |
| Whole-pool primary outputs | 35 | 34 |

## Invariants

These hold for every input. The Hypothesis property tests run 300–500 examples each.

1. **Entity partition.** partnership + S corp == passive + nonpassive, bit for bit on whole-dollar fields. Both equal (g) + (j) − (f) − (h) − (i) summed over the two entities (`test_entity_shares_partition_line_32_exactly`).
2. **Line-32 identity.** |E26270 − (partnership + S corp)| ≤ the booklet's per-field rounding, after independent disclosure rounding of every field. A residual beyond the bound is refused (existing tests, now against the entity sum). It holds on every regular record of the real file.
3. **Entity support.** An entity whose five columns are zero carries exactly zero, even when E26270 carries a rounding residual (`test_an_entity_without_columns_carries_nothing`).
4. **Entity bounds.** −(own loss columns) ≤ share ≤ (own income columns).
5. **Passive unchanged.** The passive vector is byte-identical to #1251's on the real inputs (pinned hash), and passive's formula is untouched.
6. **Bucket conservation.** Each aggregate-row bucket's weighted partnership, S-corp and passive totals equal the row's population times its per-return value, and each synthetic record keeps its processed sign pattern.
7. **Differential.** `puf_support`'s raw-field derivation equals `puf_source_schedule_e`'s columns for all three leaves (`test_raw_array_path_agrees_with_the_source_year_columns`).
8. **Passive coherence.** No passive amount survives where the person's or unit's partnership + S-corp total is zero. This holds after the primary draw (Hypothesis, with the two entities cancelling) and after the transfer. An S-corp-only carrier keeps its passive amount. Both the primary pass and the transfer refuse a basis missing either component; the primary pass does so before fitting.
9. **Target partition.** For each entity, income leg − loss leg == the signed leaf, both legs are nonnegative, and the two return indicators partition nonzero units (`test_table_1_4_entity_legs_partition_each_signed_leaf`).
10. **Retirement.**
    - Native S-corp rows stay null after the primary pass (the late transfer owns them).
    - No retired symbol remains.
    - A pre-#1258 receipt that carries the universe doctrine is refused.

**Do the tests catch wrong code?** I seeded 13 mutants across the donor module, `puf_support`, the transfer reconciliation, the target map and the materializer's sign rule. All 13 fail, each first on the test aimed at it, and the unmutated code passes (`scripts/mutation_check.py`, `out/mutation.txt`; the Hypothesis example database is cleared before each mutant). The mutants:

- the partnership share drops section 179;
- the section 179 legs are swapped between the entities;
- S corp takes the E26270 residual;
- the identity is checked against the passive split only;
- the regular S-corp share is left at zero;
- the synthetic buckets put all of line 32 in partnership;
- the donor supplement drops `s_corp_income`;
- the raw path drops E26100;
- passive zeroing reads partnership only;
- the transfer tolerates a missing S-corp leg;
- the S-corp loss leg binds the positive part;
- the partnership legs bind the combined line 32;
- the loss legs take the positive part.

## Modal evidence

I built a PUF-support base on Modal: run `entity-split-1258-base-d239b0f77b7a`, rc 0, 8,644 s, about $7.81.

- **What it was built from:** evidence branch `us-puf-entity-split-1258-route-a` (d239b0f77). That is #1251's evidence commit d02fb2bca (Route A's 4b57d15a2 plus #1239), plus only this PR's donor-side files.
- **Control:** #1251's base `line32-1239-base-d02fb2bca843`. The two bases share every household, tax-unit and person id, so the only difference is the donor split.
- **The base's donor receipt matches the local real-data receipt to the dollar.**
- **Tooling:** `scripts/modal_compare_bases.py` → `out/compare_bases.json`.

**Imputed against the donor.** PUF-clone half (89.76M weight); donor scaled per unit weight (×89.76/161.18). Base initial weights:

| Leaf | Donor-scaled net / income / loss | Imputed | Ratio |
|---|---|---|---|
| Partnership | $139.5B / $189.4B / $49.9B | $134.6B / $194.4B / $59.8B | 0.96 / 1.03 / 1.20 |
| S corporation | $237.2B / $273.7B / $36.5B | $259.7B / $285.7B / $26.1B | 1.09 / 1.04 / 0.71 |
| Line 32 (P+S, tax unit) | $376.8B / $453.8B / $77.1B | $394.2B / $471.0B / $76.8B | 1.05 / 1.04 / 1.00 |
| Line 32, #1251 (one combined leaf) | same | $455.3B / $552.5B / $97.2B | 1.21 / 1.22 / 1.26 |
| Passive (signed-mass calibrated) | $46.2B / $58.9B / $12.7B | identical | 1.00 |

- **Returns.** Partnership income returns are 2.55M against 1.96M scaled, and loss returns 1.50M against 1.14M (1.3×). S corp returns are 2.17M against 2.10M and 0.81M against 0.76M.
- **Entity mix.** 0.79M tax units carry both entities, and 0.34M carry them with opposite signs. The donor shares are similar.
- **Coherence.** No passive amount sits under a zero line 32, at person or unit grain.
- **The split brings line 32 from 1.21× to 1.05× of the donor.** This is observed; I have not traced the mechanism.
- **The entity loss legs err in opposite directions.** Partnership loss mass is 1.20× (+$9.9B) with 1.31× loss returns; S-corp loss mass is 0.71× (−$10.5B). Line 32's 1.00 loss ratio is those two errors cancelling. Nothing binds the entity loss legs until chronicle#180 lands, so this PR leaves both leaves uncalibrated, like #1251's combined leaf. Whether to signed-mass calibrate them is a methodology call, and it must be settled before those legs bind; it is tracked in #1265.
- **Chained draws.** Other PUF leaves move through the chained draws (`person_columns_moved` in `out/compare_bases.json`).
  - At base weights:
    - `w2_wages_from_qualified_business`: −$54.3B (−30%)
    - `sstb_w2_wages_from_qualified_business`: −$33.7B (−40%)
    - SSTB self-employment: −$16.9B
    - self-employment: −$9.1B
    - SSTB property basis: +$7.6B
    - REIT/PTP: +$1.2B
  - Under the Route A final weights:
    - QBI property basis: −$77.4B
    - SSTB self-employment: −$47.0B
    - SSTB property basis: −$39.7B
    - passive: −$36.6B (−26%)
    - W-2 wages: −$34.2B
    - self-employment: −$18.0B
    - SSTB W-2 wages: +$14.7B (they fall at base weights and rise here)
  - Passive mass equals the donor's only at base weights. Its placement follows the new line-32 draws (positive returns 1.80M → 1.90M), so it moves under other weights. That accounts for the whole −$36.6B change in net investment income, and the NIIT row below.

**Microsimulation.** PolicyEngine-US 2.2.1, 2024, on both bases (`scripts/modal_microsim.py`, `scripts/microsim_totals.py` → `out/microsim_totals.json`). Treatment minus control:

| Weights | Line 32 | AGI | Income tax | NIIT | QBID |
|---|---:|---:|---:|---:|---:|
| Base initial | −$61.1B | −$102.4B | −$37.3B | −$0.2B | −$0.2B |
| Route A final (control-calibrated, not re-solved) | −$55.0B | −$161.6B | −$42.2B | −$1.5B | −$19.5B |

These are base-level effects at fixed weights. A release re-solves its weights against the targets (HT2/CD A26270, and the Table 1.4 entity legs once Chronicle publishes them), so they are not release figures.

Under the Route A final weights, line 32 is $1,000.8B, against HT2 A26270 TY2022 of $1,033.3B. The entity split is partnership $439.8B and S corp $561.0B, against Table 1.4 TY2022 nets of $311.5B and $719.7B. The donor's TY2015 mix (37/63) carries a lower S-corp share than TY2022's (30/70); calibrating to the entity legs is what corrects that.

## Review

- **Round 1** (Opus 5.5 lane, because the Codex lanes were full): APPROVE WITH NITS at aa47a3f3c. Response in `review-r1-response.md`:
  1. Calibration of the entity leaves: answered with the Modal evidence above.
  2. The primary pass accepted a partial line-32 basis: fixed in ba4dfa98b.
  3. The legacy schema-16 schedule is rebuilt from current anchors. That is pre-existing: every surface change since 2026-09-03 moves it, #1251's included.
  4. Possible double-binding with HT2/CD A26270 once the legs land: noted for the #1247/#1230 control selection.
  5. The real-data test skips in CI. It passes locally on the pinned files.
- **Round 2** (Opus 5.5 lane): APPROVE WITH NITS at ba4dfa98b. It confirmed:
  - the finding-2 fix closes every path;
  - the pins come from a single observed state;
  - every figure in the evidence, and the arithmetic.

  Addressed in 48046ebfa:
  - The basis check now also runs before any QRF fit, in both entry points.
  - The loss-leg claim is reworded, and the leg calibration is tracked as #1265.
  - The stale digest, the chained-draw list and two out-of-date descriptions in this body are corrected.
  - The exact A26270 conflict, below, is recorded on #1247.
- **Round 3** (Opus 5.5 lane): APPROVE WITH NITS at 48046ebfa. It confirmed the finding-4 fix in both entry points, that the re-pins agree with one another, and the corrected body. Its notes, and what was done:
  - The Route A mover list was incomplete: completed above.
  - The test's name and ordering: the test is renamed, and it now proves the check runs ahead of donor validation as well as the fit.
  - Chain workers do not re-check a resumed manifest: no change, because `_load_manifest` already refuses a production manifest whose target order differs from the current canonical order.
  - No evidence file recorded this round's observed digests: `out/pins-final.txt` now does.
- **Round 4** (GPT-6.1 Sol): APPROVE at `487243ee9`, covering the merges of #1251 and main.
  - It confirmed the donor formulas against Schedule E's columns and the IRS booklet, the order of the post-transfer steps, and that the merges lost no main change.
  - It replaced a net-share test with three checks: draw resolution, gross-delta concentration (largest unit at most 25%, top ten under 50%, at both weightings), and level concentration.
  - **Tail verdict:** #1261 on #1251 does not clear, because single draws remain outside line 32. #1251 + #1259 + #1261 clears at the observed seed.
  - Certifying a build also needs a split-leaf seed sweep, a merged-head engine comparison and calibration checks on the candidate. The sweep is done: see the table in #1259. It confirms the verdict, because single draws in this PR's leaves recur at other seeds without the bound. The other two run on the certified build.

## CI

Later commits fix count pins that CI and local sweeps found (f33302900, e95316734, 2126a5ac8). Two were inherited from #1251: its own CI is red on them at 45a5b9891.

- `test_puf_qrf_chain.py` locks the primary target order at 66, not 65.
- `test_us_post_export_scoring.py` has 19 smoke baseline keys, not 18.

Both PRs pass all 10 checks at their current heads.

At 2126a5ac8, `engine-free (3.13)` passes the whole engine-free suite.

## Not in this PR

- **Table 1.4 vs HT2/CD.** Once chronicle#180 lands, the Table 1.4 entity legs and HT2/CD's signed A26270 describe the same line 32. At one vintage, the signed net equals income less loss of the four legs exactly, so binding both nationally is an equality conflict. For TY2022, Table 1.4 implies $1,031.2B and HT2 says $1,033.3B. Which binds nationally is a choice for the HT2/Table 1.x control work (#1247, #1230); recorded there.

- **Dollar basis.** Both leaves keep the processed column's TY2015 dollars. #1248 (PR #1223) owns aging the donor and can age each entity with its own series.
- **When the legs bind.** Table 1.4 entity legs bind only once Chronicle publishes them (chronicle#180 / #346).
- **Legacy scoring.** Pre-#1258 attested artifacts need re-attestation for legacy scoring, as after #1251.

## Decision

Merging changes what the next US build carries: S-corp income on every PUF-clone and native row, instead of zero. Max ruled yes on d1489 (10/24): land #1251, then this PR, with a fresh review that clears the top-tail draws, then a certified US build. Round 4 clears them only with #1259, and Max ruled yes on d2183 (10/31), so the order is #1251, #1259, #1261.

axiom: n/a: microsim data donor, imputation and calibration-target mapping only; no policy rule changes

🤖 Generated with [Claude Code](https://claude.com/claude-code)


