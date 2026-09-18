# Byte-transport census: brief for every census and verification agent

Read this whole file before reading any module. Every line number you cite is
this tree's, at the head you read it. You are read-only: do not edit, format,
or create any file under `packages/`, `docs/` or `tools/`; write nothing
outside the scratchpad path you are given. Do not run pytest. Do not read
gated data; the only data inputs are the two files named in §4.

## 1. What a row is

One row per **byte bound**: a module-level constant or inline literal that
bounds a byte count (`MAX_*_BYTES`, `*_MAX_BYTES`, `_MAX_*`, a `_bounded_json`
/ `_encode` / `_json` limit argument, a `len(payload) <= N` check, a
`read(N + 1)` read cap, a struct width times a row ceiling, a JSON-token or
record cap). A bound that is defined in one module and enforced in another is
one row, keyed by its definition, listing every enforcement site.

Each row must carry, from the code you read at this head:

- `constant`: `module.py:line NAME` (or `module.py:line <inline>` for a literal).
- `value`: the expression and its integer value.
- `enforced_at`: every site `module.py:line`, with the refusal code literal and
  exception type raised there.
- `encodes`: what document the bound sits in front of. Say precisely: a
  per-row roster (which roster: ACS households, ACS persons, ASEC households,
  ASEC persons, stacked households, stacked persons, clone households, clone
  persons, selected serialnos, eligible children, tax units, groups), a
  fixed-shape receipt, a catalogue, an evidence list, a header, a numpy body,
  an upstream file, a single scalar/token.
- `shape`: `roster` (grows with the selection fraction), `fixed` (does not),
  `upstream` (an input file's own size, identical at every fraction),
  `width` (a fixed-width field or encoder width).
- `site_kind`: `materialises` (the bytes exist whole before or after the check)
  or `streams` (a digest or a token-at-a-time accumulation; no whole copy).
- `bytes_at_tenth`, `bytes_at_full_source`: for a `roster` row, the bytes the
  document meets at 1/10 and at full source **through the module's own
  encoder** at the counts in §3. Cite `byte-census-measurements.json` when it
  already measured the document; otherwise measure it yourself with the venv
  python (`.venv/bin/python`) on invented rows at full-source id widths, as a
  difference between two row counts so fixed preambles cancel, and put the
  script and its receipt in your scratchpad. For a `fixed`/`upstream`/`width`
  row say why the bytes do not move with the fraction and give the one number
  you have (the recovered 1/1000 artifact's, or the input file's).
- `binds_at_tenth`, `binds_at_full_source`: `true` only if the measured bytes
  exceed the value **at the value the base branch shipped**; record the base
  value and this head's value separately when they differ (`base_value`,
  `head_value`), because several bounds were lifted on this branch already.
- `reachable_from`: `19` (the nineteen-node financial graph), `45` (the
  forty-five-node pilot graph), `age` (`survey_age_calibration`), or `none`,
  with the call chain in one line (runner → function → module). §2 gives the
  rosters. A bound in a module the runners import but whose enforcing function
  no runner reaches is `none`, and you say which function would reach it.
- `classification`, one of:
  - `segment`: a whole-roster canonical stream that meets its bound; takes the
    transport lane's segmented shape (bytes unchanged, accumulation ceiling
    unchanged, explicit total).
  - `consumer`: a consumer-side cap on bytes a segmented producer issues; takes
    the producer's total.
  - `resource`: a single body that exists whole before the bound (a numpy
    buffer) and meets its bound; becomes a fixed multiple of full source.
  - `stays`: a roster-shaped document that does not meet its bound at full
    source (give the headroom multiple).
  - `must-not-move`: protects an encoder width, an upstream file's real size,
    a fixed-width field, or a token/scalar.
  - `not-a-transport`: a row count wearing a byte name (say which row-count
    rule applies), or a memory ceiling with no document behind it.
- `notes`: anything a reviewer needs; keep it to what the code says.

## 2. Reachability, from the code

**The nineteen-node financial graph** (`_recovered/pilot-runs/native19-required-20260912/run/financial-artifacts/graph.json`, run by `graph_atomic_survey_financial.run_atomic_survey_financial` with no property, tax, status or roles):

| node | kernel |
|---|---|
| `survey_population.create`, `.allocate`, `.observed_geography` | `graph_survey_population`, `graph_atomic_survey_population`, `survey_population_preparation` (source admission), `acs_native_coverage_binding`, `acs_person_coverage_authentication`, `acs_housing_universe_source`, `acs_population_catalogue`, `asec_2024_native_population`, `asec_population_catalogue`, `survey_catalogue_selection`, `survey_population_domains`, `survey_observed_age`, `native_household_origin` |
| `combined_survey_puf_support_clone`, `.owned` | `graph_combined_clone` |
| `geography.support.0`, `.assign`, `.derive`, `.gate` | `survey_atomic_geography`, `graph_geography`, `microcosm.graph.codecs` (`raw-bytes-v1`) |
| `survey_predictors.source_projection`, `.asec_design_donor`, `.asec_current_columns`, `.fit.000-002`, `.apply.000-002`, `.attach` | `graph_current_survey_predictors`, `current_survey_predictors`, `asec_current_money*`, `_asec_current_money_codec`, `microcosm.fit.model_input`, `microcosm.fit.qrf`, `microcosm.fit.graph_qrf*` |

After the graph returns, the runner replays and verifies: `survey_population_replay`, `graph_atomic_survey_financial` verification, `current_survey_predictors.verify_materialized_current_survey_predictors`.

**The forty-five-node pilot graph** (`_recovered/pilot-runs/native45-v5/`, run through `graph_survey_completion_host` with `property_income`, `rebase_property_taxes=True`, `person_status=False`, `household_roles=False`): the nineteen above, plus the property-income nodes of `graph_current_survey_property.current_survey_property_nodes` (`graph_property_income`, `graph_property_income_receipts`, `current_property_income_sources`, `current_asec_property_basis`, `property_income_constants`), the three property-tax nodes of `graph_property_tax_leaves`, and the ten completion nodes: `survey_completion.receiving` (`graph_survey_completion`), the six `child_property.*` nodes (`graph_child_property_income`, `current_child_property_income_source`, `current_property_completion_routing`, `microcosm.fit.graph_joint_empirical`, `microcosm.fit.joint_empirical`), and the tax trio again over the completed population. Household roles (`current_survey_household_roles`, `graph_current_survey_household_roles`) and person status (`graph_current_survey_person_status`) are reachable only from the 49/51-node variants: record them, mark `reachable_from: none` with that note.

**`survey_age_calibration`** is neither graph. It runs `survey_origin_budget.freeze_survey_origin_budget` and `.checked_view`, `graph_survey_budget`, `graph_survey_calibration`, `graph_survey_age_artifact`, `survey_financial_successor`, `demographic_calibration_graph`, `graph_national_age_counts`, `survey_calibration_diagnostics`. The brief that commissioned this census names the origin budget explicitly, so these are in scope under `reachable_from: age`.

**Not on any of these paths** (record with `reachable_from: none` if you meet them; do not spend measurement effort): `graph_survey_puf55`, `puf55_survey_recipients`, `graph_current_survey_puf_transfer`, `puf_diagnostic_consumer.current_survey_recipient_matrix`, `current_survey_amounts`, `graph_us_survey_enrichment`, `graph_fiscal_*`, `graph_acs_housing_universe` (the housing *graph*; the housing *source* `acs_housing_universe_source` is on the path through `acs_native_coverage_binding`).

## 3. The counts (measured, not extrapolated)

From the recovered 1/1000 artifact's catalogues, which are counts of the whole
upstream files and so are full-source counts (`experiments/native-row-ceilings/roster-census.json`):

| quantity | full source | at 1/10 | at 1/15 |
|---|---:|---:|---:|
| ACS households (selectable) | 1,531,614 | 153,161 | 102,107 |
| ACS persons | 3,422,888 | 342,288 | 228,192 |
| ASEC households | 55,762 | 5,576 | 3,717 |
| ASEC persons | 142,125 | 14,212 | 9,475 |
| stacked households (= selected = groups) | 1,587,376 | 158,737 | 105,825 |
| stacked persons | 3,565,013 | 356,501 | 237,667 |
| clone households | 3,174,752 | 317,475 | 211,650 |
| clone persons | 7,130,026 | 713,002 | 475,335 |

Entities other than person and household (tax units, families, marital units,
SPM units) are roster ratios from the 1/1000 artifact (1,584 households carry
2,128 tax units, 1,594 families, 2,774 marital units, 1,585 SPM units); say
`SCALED` when you use one.

## 4. Inputs you may read

- `experiments/native-byte-transports/byte-census-measurements.json`: this
  lane's measurements through each module's own encoder (origin budget,
  numeric bounds, age artifact, roles projection, predictor projection,
  diagnostic matrix, child draw, selection block, and the whole captured ACS
  archive: coverage body, pre-allocation charge, key list, serialno list).
- `experiments/native-row-ceilings/census.json`: the row-count lane's census
  (146 bounds). Its byte rows are a starting list to **verify at this head**,
  not a source of truth: several of its values and verdicts predate lifts on
  this branch, and it missed modules (the property family, the housing source's
  prepared receipt, consumer-side re-encodings).
- `_recovered/pilot-runs/native19-required-20260912/run/financial-artifacts/preparation.json`
  (the 1/1000 preparation receipt, 1,136,063 bytes) for row shapes.
- `docs/us-native-scale-transport.md` §2 and `docs/us-native-row-ceilings.md`
  for the two neighbouring arguments.

## 5. What has already moved on this branch (record base and head values)

- `acs_person_coverage_authentication`: `MAX_ROSTER_BYTES = 64 * MAX_BODY_BYTES`
  added; the coverage body, the issuance receipt and the key-list digest are
  segmented/streamed; `SELECTED_BODY_BUDGET` charges the NDJSON line bound.
- `acs_native_coverage_binding`: the serialno pre-check sizes without encoding
  under `coverage.MAX_ROSTER_BYTES`; the issuance receipt is `_json_roster`.
- `graph_survey_population`: `_canonical_pieces`, `_segmented_json`,
  `_json_sha256`, `_json_matches`; `PREPARATION_ROSTER_BYTES`.
- `survey_origin_budget`: `MAX_ROSTER_BYTES`; the budget document is
  segmented; `selection_sha256` streamed; `checked_view` streams its
  comparison.
- `graph_survey_budget`, `graph_survey_calibration` (`MAX_ROSTER_BYTES`,
  `MAX_ROWS`), `graph_survey_age_artifact` (`MAX_BYTES` 2 GiB),
  `current_survey_predictors` (segmented projection),
  `graph_child_property_income` (segmented `_json`, `MAX_ROSTER_BYTES`,
  `MAX_ORIGINALS`), `current_child_property_income_source`
  (`MAX_RECIPIENT_ROWS`), `microcosm.fit.graph_joint_empirical`
  (`MAX_DRAW_BYTES`).

Use `git diff origin/native-row-ceilings -- <path>` to see exactly what moved
in a module; the base value is the `-` side.

## 6. Verification agents

You receive one row and must try to refute it by re-reading the code: the
value, every enforcement site, the encoded document's shape, whether the site
streams or materialises, the reachability chain, the byte figures (recompute
one), and the classification. Return `agrees: true/false`, the exact
correction if false, and the line-cited evidence either way. Default to
`agrees: false` when you cannot verify a claim from the code.
