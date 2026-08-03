# US: turning a Ledger fact into a calibration target

Ledger's side of the boundary is documented in its facts-only ADR: Ledger
stores what a source asserted, with provenance and a `concept_alignment`
*claim* about the matching PolicyEngine-US concept. Everything operative —
which variable a fact binds, in what mode, over which universe, whether it
binds at all — is a consumer decision and lives here. This is the procedure
for the US pipeline, in the order that avoids the known failure modes.

## What may never become a target

Before minting anything, apply the rule (doctrine, Max 2026-08-02):

**We may not calibrate against tax-benefit quantities measured in a survey,
or anything derived from such.** The four quadrants:

| | administrative source | survey source |
|---|---|---|
| **tax-benefit quantity** | ✅ target (SOI claims, FNS counts, SSA payments, ACF dollars) | ❌ never (e.g. total SNAP from the CPS) |
| **raw quantity** | ✅ target | ✅ target (ACS population, demographics, income margins by geography) |

…and the "derived from such" clause extends the prohibition to everything
downstream of survey tax-benefit measurement: **SPM/OPM poverty rates above
all** (SPM resources embed survey-measured benefits and calculated
taxes), and other models' survey-based tax-benefit estimates
(TRIM3/ATTIS/DYNASIM outputs — comparators or seeds, never targets).

Rationale: populace replaces the survey's tax-benefit measurement with
imputed, computed, and admin-calibrated values — that is the product.
Fitting a survey-derived tax-benefit quantity launders the
measured-with-error version back in and destroys the held-out validation
signal (the scorecard's win column is held-out-only for the same reason).
Release gates may *fail* a certification on a held-out poverty regression;
*fitting* the statistic is categorically different and prohibited.

For raw survey margins, prefer an administrative source when one covers the
same cell and concept — e.g. congressional-district income binds from
`irs_soi.congressional_district_2022`, while ACS (the `census_acs.acs`
family) supplies population and structure.

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

Worked examples of the whole arc: populace#451 (anchors → support oracles →
ledger#105 facts → PR #465 wiring/probes), and the exclusion-register
corrections in that PR.
