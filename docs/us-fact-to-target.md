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
  state's HT2 all-returns total
  (`irs_soi.<ty>.historic_table_2.state_broad.<st>.all.<measure>`) as the
  registry binds it: the row latest-vintage selection kept and the
  exclusions let through, read from the compiled registry, not from the
  facts. A band's share of the total it calibrates beside is therefore the
  share it records. A state whose total is excluded has no control and binds
  no bands of that measure; two bound totals for one state and measure
  raise. Congressional-district `<st>_total` rows never anchor a state.
- **Periods.** The value lands at the control's period and ages with the
  state total: AGI on the CBO AGI series, counts never. The bands therefore
  stay the same share of the state total they bind beside, at every stage.
  The control may be older than the bands (TY2023 bands on the TY2022 state
  total): the newest published shares scale onto the level the state total
  binds at, rather than binding a second, differently aged level.
- **One vintage.** State bands are the one family selected by vintage, not
  per row. Every eligible vintage (not after the build period) of a binding
  band stays a candidate through latest-vintage selection, and
  `_with_one_state_agi_band_vintage` then keeps, per state and measure, the
  newest vintage whose published partition is complete. The TY2022 package's summed `500k_plus` row never binds beside
  TY2023's split rows, and a gapped newest vintage falls back to the last
  complete one. The choice is made inside selection, so the exclusion vintage
  guard and the exclusion receipt see only the band ids that can bind.
- **Malformed partitions.** In a vintage that is examined, overlapping
  bands, a NaN bound, a value that is infinite, NaN or does not parse, or one
  source record id with two different rows raise; an identical re-emission of a sub-floor band counts once. A
  binding band's own fact is a target fact: like every other family, the
  compile refuses two facts for one source record id, identical or not.
  Vintages older than the one chosen are not examined.
- **The floor holds at every period.** A state band below $100k, or for a
  single filing status, is refused even in its own tax year, so a same-year
  vintage can never bind raw sub-floor levels.
- **Receipts.** Every rebased row carries `state_agi_band_share`,
  `uprating_factor` and the control's record id.
- **Release gate.** `US_FISCAL_TARGET_COVERAGE_REQUIREMENTS` carries one
  requirement per state and measure,
  `irs_state_agi_top_tail_<st>_<measure>` (102 in all), each matching that
  state's `[$1M, inf)` row by `state_fips` and both AGI bounds. A feed
  without the split bands, or a state whose partition or bound total is
  missing for either measure, fails the release and names the state, rather
  than shipping without the constraint; a second row for another state
  cannot stand in. Together the two rows pin each state's AGI above $1M.
- **Period contract.** Without aging, a band rebased onto an older state
  total holds that total's period's dollars; `find_period_contract_violations`
  reads `uprating_to_period` for rebased rows, the same period target aging
  ages them from.
- **Agreement with Table 1.1.** On the pinned feed the states' $500k–$1M and
  $1M+ rows sum to 98–99% of Table 1.1's TY2023 classes aged the same way;
  Table 1.1 also counts returns filed from other areas and Puerto Rico
  (`test_pinned_feed_state_top_tail_agrees_with_table_1_1`).
- **Support.** The rows can bind only where the pool has records in the
  band. `experiments/940-state-agi-bands/` measures pre-calibration support
  per state and band on the #982 fix's offline outputs.

### Federal Student Aid's portfolio binds the student loan balances (microcosm#1074)

Federal Student Aid publishes its federal student loan portfolio quarterly.
Chronicle stores each table as published (PolicyEngine/chronicle#310); the
alignment across tables is Microcosm's, in
`us_runtime/fsa_student_loan_targets.py`. The targets bind the person columns
the SCF education-debt stage writes (microcosm#1073):
`federal_student_loan_balance` (own loans) and
`federal_parent_plus_loan_balance`. A person's federal balance is the sum, and a
borrower is a person whose federal balance is positive.

- **Which rows.** The borrower-based tables: by location (the 50 states and DC,
  balance and borrowers), by age (five bands on the person's age) and by debt
  size (nine bands on the person's federal balance, lower edge inclusive). From
  the loan-type table, the Parent PLUS balance and the balance of every other
  loan type. Each fact's disposition, bound or not, is in
  `fsa_student_loan_target_receipt` and in a release's
  `us_source_coverage.json` (`fsa_student_loan_target_receipt`).
- **Diagnostic only.** Repayment plans: the SCF records only an income-driven
  yes/no, the build has no plan column, and FSA counts plan recipients loan by
  loan. The portfolio summary: recipient totals by loan program, which the
  balances do not record. Loan-type recipient counts: loan-level, so a person
  with two loan types counts twice. Stafford (Combined): a subtotal.
- **The reference date.** The latest quarter every bound table publishes, FY2026
  Q2 (31 March 2026) in the draft packages: the location workbook publishes
  only that date, so the age, debt-size and loan-type tables bind at it too, and
  every family describes one portfolio on one day. The facts are stocks the
  build carries as they stand at that date, unaged at the 2024 build period
  (`period_alignment = stock_at_source_reference_date`, factor policy 5 in
  `target_aging`): no income-growth series describes a loan stock, and the
  analysis reads the columns in 2026.
- **Not Reported, Puerto Rico and Other.** The location table has no total row.
  Its "Not Reported" row ($52.2B, 1.95M borrowers) holds FFEL and Perkins loans
  whose commercial lenders need not report an address. It is allocated to every
  reported location in proportion to that location's reported balance (or
  borrowers, for the count). Puerto Rico and "Other" (an address outside the
  listed states and territories) are outside the host universe, the 50 states
  and DC the CPS samples, and leave with their share. The 51 allocated state
  levels sum to the **in-scope control**: $1,679.1B and 44.31M borrowers at the
  draft packages, against $1,696.3B and 44.76M in the table.
- **One control.** FSA's tables disagree slightly (the borrower tables total
  $1,696.3B, the recipient-based loan-type table $1,723.8B). The age and
  debt-size bands and the two loan-type rows bind as their table's share of the
  in-scope control, `control x row / partition`, the state AGI band pattern. A
  table's own "not reported" age row is in no partition, which allocates it
  across the reported bands. So the state, age, debt-size and loan-type families
  each sum to the same total.
- **Opt-in.** `compile_us_fiscal_target_registry(student_loan_targets=True)`
  binds the facts; the release tool sets it with `--scf-education-debt`, the
  only builds that carry the columns. The release materializes the rows from
  the stored person columns (`fsa_student_loan_household_values`), because the
  pinned policyengine-us does not define them yet. The release's
  restated-filter guard accepts the age and balance bands only on that
  materializer. While the stage is opt-in the family stays off the one-surface
  target-parity contract (`_without_opt_in_student_loan_targets`).
- **Known limits, measured.** The receipt
  (`experiments/1074-fsa-calibration-receipts.md`) refits the published
  default with and without the family. The stage gives balances only to
  reference persons and spouses (and Parent PLUS to the reference person), so
  the host holds about half of FSA's borrowers ($0.86T and 21.2M at the
  published weights). Calibration closes that by reweighting: borrower
  households' weight doubles and their share of household weight goes from
  16.4% to 33.0%, against 18.2% of SCF families. Borrowers under 25 stay about
  40% short, because the host has few young reference persons with a balance.
  FSA's Parent PLUS row counts only unconsolidated Parent PLUS loans (a
  consolidated one sits in "Consolidation"), while the SCF's loans for a
  child's education include consolidated ones, so that target is a lower bound
  on the concept it binds; it pulls the Parent PLUS share from 16.8% to 6.8%.
- **When the facts reach the pin.** The US pin moves to a Chronicle commit that
  carries the six `packages/ed_fsa` packages: add their (record set, period)
  pairs to `us/chronicle_feed_scope.json` (build year 2026), rebuild the feed,
  re-pin `us/chronicle_feed.json`, and regenerate the two parity resources with
  the FSA families declared as reviewed exclusions that bind only with
  `--scf-education-debt`. The fixture
  `tests/fixtures/us_fsa_portfolio_chronicle_347_facts.jsonl.gz` is the draft
  export the tests pin; refresh it from the merged packages.

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
