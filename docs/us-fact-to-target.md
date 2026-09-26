# US: turning a Ledger fact into a calibration target

Ledger's side of the boundary is documented in its facts-only ADR: Ledger
stores what a source asserted, with provenance and a `concept_alignment`
*claim* about the matching PolicyEngine-US concept. Everything operative —
which variable a fact binds, in what mode, over which universe, whether it
binds at all — is a consumer decision and lives here. This is the procedure
for the US pipeline, in the order that avoids the known failure modes.

## What may never become a target

Before minting anything, apply the rule (doctrine, Max 2026-08-02):

**We may not calibrate against tax-benefit quantities from a survey —
reported or computed, any of them — or anything derived from such.** The
four quadrants:

| | administrative source | survey source |
|---|---|---|
| **tax-benefit quantity** | ✅ target (SOI claims, FNS counts, SSA payments, ACF dollars) | ❌ never (e.g. total SNAP from the CPS) |
| **raw quantity** | ✅ target | ✅ target (ACS population, demographics, income margins by geography) |

…and the "derived from such" clause extends the prohibition to everything
downstream of survey tax-benefit measurement: **SPM/OPM poverty rates above
all** (SPM resources embed survey-measured benefits and calculated
taxes), and other models' survey-based tax-benefit estimates
(TRIM3/ATTIS/DYNASIM outputs — comparators or seeds, never targets).

Rationale: microcosm replaces the survey's tax-benefit measurement with
imputed, computed, and admin-calibrated values — that is the product.
Fitting a survey-derived tax-benefit quantity launders the
measured-with-error version back in and destroys the held-out validation
signal (the scorecard's win column is held-out-only for the same reason).
Release gates may *fail* a certification on a held-out poverty regression;
*fitting* the statistic is categorically different and prohibited.

For raw survey margins, prefer an administrative source when one covers the
same cell and concept — e.g. congressional-district income binds from
`irs_soi.congressional_district_2022`, while ACS (the
`census_acs` source) supplies CD population and structure. Every US calibrated
artifact compiles the same national + state + congressional-district target
surface (`us_runtime/fiscal_targets.py::compile_us_fiscal_target_registry`);
artifact record count changes L0, not target membership. The pinned
`census_acs.acs1_2023` family remains a reviewed `survey_derived` exclusion
because all of its facts are state-grain, while the compiler admits ACS only at
congressional-district geography
(`tools/build_us_target_parity_manifest.py::_FAMILY_EXCLUSIONS`).

**Corollary: deviations from official poverty metrics are never inherently
problematic.** A model that corrects benefit underreporting should, all
else equal, sit below survey-based poverty rates; divergence from Census
numbers is expected by construction. Treat official statistics as
comparators — direction and composition anomalies are investigation flags,
not "misses".

## 0. Mint the fact (ledger repo)

Author the measure in the owning source package
(`ledger/packages/<source>/<table>/source_package.yaml`), commit the raw
workbook + manifest under `ledger/db/data/...`, and verify the export against
an oracle you read from the source yourself — the exported
`consumer_facts.jsonl` row must reproduce the published number exactly.
Include the `concept_alignment` evidence block (source concept, relation,
evidence URL/notes, `legal_vintage`): without it the fact cannot state which
engine concept it anchors. See ledger's `agent-source-package-harness.md`.

## 1. Splice the feed

The build consumes one sha-pinned `consumer_facts_*.jsonl`. A new feed
version is the previous feed plus the new rows, **deduplicated on
`lineage.source_record_id`** — re-exporting an id supersedes the old row
(record any deliberate supersession, e.g. a vintage-mislabeled legacy row, in
the release PROGRESS notes). Re-pin the sha everywhere it is enforced: the
launch scripts' `FACTS_SHA` guards, `--ledger-facts-sha256`, and the PROGRESS
doc. The builder refuses to start on a mismatch.

## 2. Run the support oracle BEFORE wiring anything

A target with no model support cannot bind; a target with thin support binds
by concentrating weight on the few carriers (the #445 keogh-crush class).
Before touching the maps, measure on the current certified artifact:

- **Input variables**: carrier count and weighted mass from the export
  (variables live in each entity's PyTables `table` dataset, e.g.
  `person/table` field `tip_income`).
- **Computed variables** (ALDs, credits): compute with the engine —
  `Microsimulation(dataset=...)` then `calculate(var, period)`; the export
  does not carry them.

Decision rule from the certified-M oracles: a ~3× mass stretch on real
carriers is bindable with a selection-mass protection; a ~34× stretch
(keogh ALD: $0.92B modeled vs $30.13B SOI) or a count target demanding
hundreds-fold weight concentration (tips returns: 549 carriers vs 6.04M) is
not — those wait for source-stage widening and are recorded honestly in the
exclusion register instead.

## 3. Wire the binding (`us_runtime/fiscal_targets.py`)

- Facts with **real measure ids** route through
  `SOI_AMOUNT_MEASURE_VARIABLES` / `SOI_RETURN_MEASURE_VARIABLES` (measure id
  → concept) and `SOI_VARIABLE_MAP` (concept → engine variable).
- Facts with **generic measure ids** (`amount`, `return_count` — the W-2
  item tables) route by layout in `_soi_layout_variable_override`, which
  runs before the unmapped early-return precisely so layout-only ids can
  bind. Keep override patterns narrow (exact groupby dimension + value
  frozensets) so no unintended fact matches.
- Universe restrictions (e.g. itemized-only) derive from the record set via
  `_soi_return_universe_from_record_set_id`; check the auto-stamp is right
  for the new record set.
- An unmapped fact is **inert by design** — shipping facts ahead of their
  wiring is safe and normal (the keogh ALD facts rode the feed unmapped for
  weeks).

### State AGI bands bind as shares of the state total (microcosm#940)

A state's total return count and AGI leave its top tail free, and the top
tail is what progressive state rate schedules tax: #940 measured Build P's
ACS-local release within 0.12% of Colorado's SOI returns and AGI while
holding 24% of Colorado's SOI AGI above $1M. IRS SOI Historic Table 2
publishes each state's returns and AGI in ten AGI classes, including
$500k–$1M and $1M+ separately. `_soi_reference_from_fact` rescues those
bands from the cross-period refusal and `_rebase_soi_state_agi_bands`
binds them:

- **Which rows.** `return_count` and `adjusted_gross_income` from the
  per-state HT2 AGI record sets (`irs_soi.<ty>.historic_table_2.state_agi.<st>`),
  all filing statuses, bands from
  `US_SOI_STATE_AGI_BAND_MINIMUM_LOWER_BOUND` ($100,000) up. The floor is the
  national size-of-AGI floor for the same reason: the SOI slice materializer
  counts every tax unit in a band, filer or not. The HT2 `us` rows never
  qualify; the national AGI shape belongs to Table 1.1.
- **The value.** `control x band / partition`. The partition is the sum of
  the state's published bands of the same vintage, measure and record set
  over the whole AGI line, negative AGI under $1 included; HT2 publishes
  amounts that add exactly and counts rounded to tens. The control is the
  state's latest HT2 all-returns total not after the build period
  (`irs_soi.<ty>.historic_table_2.state_broad.<st>.all.<measure>`), the row
  that already binds the state's total. Congressional-district `<st>_total`
  rows never anchor a state.
- **Periods.** The value lands at the control's period and ages with the
  state total: AGI on the CBO AGI series, counts never. The bands therefore
  stay the same share of the state total they bind beside, at every stage.
  The control may be older than the bands (TY2023 bands on the TY2022 state
  total): the newest published shares scale onto the level the state total
  binds at, rather than binding a second, differently aged level.
- **One vintage.** Per state and measure only the newest vintage's bands
  bind, so the TY2022 package's summed `500k_plus` row never binds beside
  TY2023's split rows. A gap in a partition drops that state's bands; an
  overlap raises.
- **Receipts.** Every rebased row carries `state_agi_band_share`,
  `uprating_factor` and the control's record id.
- **Release gate.** `irs_state_agi_top_tail` in
  `US_FISCAL_TARGET_COVERAGE_REQUIREMENTS` requires a $1M+ AGI row for all 51
  states, so a feed without the split bands fails the release rather than
  shipping without the constraint.
- **Agreement with Table 1.1.** On the pinned feed the states' $500k–$1M and
  $1M+ rows sum to 98–99% of Table 1.1's TY2023 classes aged the same way;
  Table 1.1 also counts returns filed from other areas and Puerto Rico
  (`test_pinned_feed_state_top_tail_agrees_with_table_1_1`).
- **Support.** The rows can bind only where the pool has records in the
  band. `experiments/940-state-agi-bands/` measures pre-calibration support
  per state and band on the #982 fix's offline outputs.

## 4. Keep the exclusion register honest

`US_FISCAL_TARGET_SUPPORT_EXCLUSIONS` is keyed by `source_record_id` and its
reasons are read by humans deciding what to trust. When support reality
changes (a source stage lands, carriers appear), update the reason in the
same PR — a stale "no support exists" claim over real-but-thin support is a
data-integrity bug of its own.

An entry excludes one vintage. Latest-vintage selection would hand its key
to any other vintage of the same cell the feed carries, so the compile
refuses such a fallback (microcosm#956). When a feed adds a vintage of an
excluded cell, decide which of two things it is. If the reason holds at
every vintage, as #564's concept mismatch does, add the key to
`US_FISCAL_TARGET_ALL_VINTAGE_SUPPORT_EXCLUSIONS`. If the other vintage
should calibrate, add its id to `US_FISCAL_TARGET_EXCLUSION_VINTAGE_BYPASSES`
with a reason. `us_source_coverage.json`'s `fiscal_target_exclusion_receipt`
lists the ids each rule dropped or allowed in a release.

## 5. Add a reform-coverage probe

Probes are **generated data**: author them in
`tools/build_us_release_input_coverage_manifest.py` and regenerate the
manifest with that tool — never hand-edit the JSON (a byte-sync test
enforces this). Probe discipline:

- **Measure the effect first** on the certified artifact; set
  `min_abs_effect` well under it (~25%). Never guess floors.
- **Probe the law year where the channel exists.** Attribution-only columns
  (tips, FLSA overtime premium) are contained in W-2 wages; their only tax
  channel is the OBBBA deductions, so a 2024-law probe measures exactly $0
  and the correct probe runs at 2026 law with a negative expected sign.
- State the measured value, the external anchor, and what a structural zero
  would mean in the `reason`.

## 6. Prove it end to end

Add a discriminating registry test (`test_us_fiscal_targets.py` pattern:
compile `compile_us_fiscal_target_registry` with a synthetic fact carrying
the new id/layout and assert the spec's variable, mode, and metadata — the
test must fail without the wiring). Run the fiscal-targets,
release-input-coverage, ledger-targets, and builder suites, then
`tools/preflight_us_release_gates.py` against the new feed before any
release launch. Log the registry version-hash / spec-count delta in the
release PROGRESS notes.

## 7. Launch pairing

Thin-support dollar targets ship with a matching
`--selection-mass-protection <variable>` on the launch script (the #446
pattern) so the pre-solve selection cannot crush the carriers the new target
needs.

Worked examples of the whole arc: microcosm#451 (anchors → support oracles →
ledger#105 facts → PR #465 wiring/probes), and the exclusion-register
corrections in that PR.
