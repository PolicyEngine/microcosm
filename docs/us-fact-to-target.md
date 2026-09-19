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

### Cross-period AGI slices bind only as shares

SOI publishes most size-of-AGI detail one or two tax years behind the build
period. `_is_untransformed_cross_period_agi_slice` therefore refuses any SOI
fact that carries an AGI bound and a tax year other than the build period: an
old nominal bin is not a target-year level. A family escapes that refusal only
when it has **both** a narrow rescue predicate in `_soi_reference_from_fact`
and a pass that rebases it onto an active national control. A rescued row
with no pass would compile flagged and ship as a stale hard target, which is
the microcosm#489 defect. The rescued families are:

| Family | Record set | Measures | Pass |
|---|---|---|---|
| EITC by AGI and qualifying children | `irs_soi.table_2_5.eitc_by_agi_children` | `eitc_total`, `eitc_returns` | `_uprate_cross_period_eitc_decompositions` |
| Taxable interest by AGI | `irs_soi.historic_table_2.*` | `taxable_interest_amount`, `taxable_interest_returns` | `_rebase_stale_soi_taxable_interest_distributions` |
| Size of AGI (microcosm#958) | `irs_soi.table_1_1` | `return_count`, `adjusted_gross_income` | `_rebase_stale_soi_agi_size_distributions` |

The size-of-AGI family is the only national anchor for the **shape** of the
upper income distribution; the all-returns totals bind its level and nothing
else. Its rules:

- National, all filing statuses, Publication 1304 Table 1.1 only. Table 1.2
  and Table 1.4 carry the same measure ids over other universes and are not
  rescued.
- Size classes bind from
  `US_SOI_AGI_SIZE_DISTRIBUTION_MINIMUM_LOWER_BOUND` ($100,000) up. The SOI
  slice materializer counts every tax unit in the AGI band with no filer
  filter, so a return count equals a tax-unit count only where filing is
  near-universal. Below the edge the EITC-by-AGI families anchor the
  distribution.
- Each class is a share of its own vintage's Table 1.1 national total, scaled
  to the latest eligible Table 1.1 national total and landed at that
  control's period. Target aging completes the chain, so class AGI ages on
  the CBO AGI series with the all-returns AGI row, class counts stay raw with
  the all-returns count, and the classes keep summing to the national rows.
- A class with no same-vintage total or no eligible control is dropped.
- **Support requirement.** The $5M–$10M and $10M+ rows need an own-tail
  stratum in the pool. Measured on certified
  `populace-us-2024-spm-20260915`, which has none: 9 and 3 records carry
  12.1k and about 1 weighted returns against SOI's 49.3k and 30.4k, a stretch
  no solve can make under `max_weight_ratio = 5`. Measured on main's base
  pool, the capital-gains tail stratum holds about 12k and 9k units at base
  weights, a 3x to 4x stretch.
- **Launch pairing.** Do not launch a release on this surface with a tail
  stratum that carries capital gains only. The family fixes the shape of AGI,
  and a capital-gains-only tail then fills the top classes with preferential
  income while the classes below lose the excess returns that had been
  carrying ordinary income. A prototype on the certified file
  (`experiments/958-us-top-income-tail-receipts.md`; a graft and an entropy
  reweighting, not a Microcosm build) scores a 37% to 39.6% top-rate reform
  at +$23.5B as certified, +$20.8B with no tail and this family, **+$18.8B**
  with a capital-gains-only tail and this family, +$28.4B with a full-vector tail and this family, and +$30.7B
  when SOI Table 1.4 wages and net capital gains by size of AGI bind as
  well. Pair the launch with the full-vector own-tail stratum (microcosm#958
  increment 2).

## 4. Keep the exclusion register honest

`US_FISCAL_TARGET_SUPPORT_EXCLUSIONS` is keyed by `source_record_id` and its
reasons are read by humans deciding what to trust. When support reality
changes (a source stage lands, carriers appear), update the reason in the
same PR — a stale "no support exists" claim over real-but-thin support is a
data-integrity bug of its own.

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
