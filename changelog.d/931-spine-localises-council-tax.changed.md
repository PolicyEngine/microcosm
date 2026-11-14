Localise Great Britain council tax by each clone's local authority just before the UK spine's WAS chain, and carry each bill's provenance from stage 5 (#931). The council-tax localisation #931 group (d) added unwired is now declared: `council_tax_localise` runs after `private_rent` and before `was_wealth`, in a graph version of its own, so the packaged spine has 61 stages. Stage 5 (`frs_council_tax`) writes `council_tax_bill_provenance` beside the bill, and the localisation reads it. The localisation keeps the paid bill from before it as `council_tax_frs_bill`, which the WAS chain's bill predictor reads. The release boundary drops both columns.

- **Stages** (`sources.yaml`, then `source_stages.json` by the by-name merge):
  - `frs_council_tax` declares `derive` after `impute_cell_means`, as `frs_council_tax_provenance_operation_parameters()` returns it, and outputs `council_tax_bill_provenance` after `council_tax`. Its imputation and nonnegative outputs are unchanged, and its notes gain one sentence on the flag. The transform writes the flag (`StringDtype("python")`) from the same branches that choose the bill, which stays bit for bit as before. `checkpoint_metadata()` returns the provenance receipt, `council_tax_provenance_evidence`: distinct source households before the copies, through `rounded_count`. The stage's declaration check expects `read_tables`, `impute_cell_means` and `derive`, with both reviewed parameter sets.
  - `council_tax_localise` declares `council_tax_localise_operation_parameters()` verbatim and in order:
    - `verify_vendored_fact_resource` for Scotland's Band D at build period 2024;
    - `derive` for the kept bill;
    - `localise_council_tax`;
    - `record_mass_conservation_receipt`, coupled to the terminal family gate as `private_rent`'s is.

    Its two Band D resources are artifacts with `runtime_sha256_required`: `council_tax_band_d` (`council_tax_band_d_england_wales_2024_25.json`) and `scotland_band_d_facts` (`scotgov_band_d_council_tax_2024_25.json`). Its one output and nonnegative output is `council_tax_frs_bill`. It rewrites `council_tax`, `council_tax_rebate` and `council_tax_benefit_reported`. Its notes describe this behaviour. `was_wealth`'s and `was_lisa`'s declarations are unchanged.
- **Shared vocabulary**, for Max's review. The source-stage schema gains the strict `localise_council_tax` kind: every key is required and no other key is allowed. `ALLOWED_SOURCE_OPERATION_KINDS` gains the kind.
- **The WAS read.** `was_wealth.recipient_predictors`, which `was_lisa` calls, now takes its council-tax predictor from `council_tax_frs_bill`. That is the paid bill the localisation keeps from before it localises: `max(council_tax - council_tax_rebate, 0)` on the FRS bill, the expression the predictor computed before. The predictor frame keeps its columns and their order. A frame without the kept bill is refused, naming the column. So the bill predictor is bit for bit as before, and the drawn area does not enter it. The HBAI net income both stages read from the engine sees the localised bill and reduction.
- **The graph and the spine build.**
  - `council_tax_localise` opens its own graph version (`_READER_ISOLATION_BOUNDARIES`). It rewrites the housing shell's person council tax reduction, which `private_rent`'s finite reader anchors on, so without a version of its own the compile would close a cycle between the two. `private_rent` reads the shell's cells, and `was_wealth` and `was_lisa` read the localisation's. The spine graph gains the stage and `council_tax_localise.boundary`.
  - The stage consumes each household's draw inputs (geography key, region, IMD decile, council-tax band and band flag), its drawn area, system and local authority, the bill's provenance and the single-adult flag. It also consumes the support channel's branch of the copy lineage and its true source and clone index. Like the area draw, it reads the three conditioned supports as graph sources. Its cells are the kept bill and the three rewrites, all float64; stage 5's cells gain the flag, a string cell.
  - In `graph_kernels`, the stage's module is `spine_council_tax`. It is bound with the helpers its transform computes through, among them `atomic_support_expectations`, `bounded_amounts` and `council_tax_band_d`, and counts as a counting stage. It runs on the bitwise stage kernel every spine stage runs on (F10 is open with Max). The H2 fixture's registry builds it on the toy supports at the committed Band D.
  - `spine_build` builds the stage on the supports the area stages read, holding them to the spine's support pins first. So a support replaced after the spine was prepared is refused with the area stages' message before the stage builds anything. The support flags' help names the stage. The sidecar's stage records report no nonzero share of the flag or of the kept bill (`_UNREPORTED_SHARE_COLUMNS`), as for the root's area carriers.
  - The geography-first roster gains `COUNCIL_TAX_LOCALISE_STAGE`, and `spine_council_tax` reads its stage name from there. So the dense line's posture check, the spine CLI's release candidate and the national line refuse every spine built before this change, naming the stage alone.
- **The release boundary.** `UK_RELEASE_EXPORT_DROPPED_COLUMNS["household"]` gains `council_tax_bill_provenance` and `council_tax_frs_bill`, which no engine variable reads. Neither is a reviewed extra of the export surface (`UK_ALLOWED_EXTRA_EXPORT_COLUMNS`), so the dense and national exports carry neither.
- **Tools.**
  - `tools/graph_uk_spine_fixture.py` runs the localisation in its oracle (`UK_FIXTURE_STAGE_COUNT` 61).
  - `tools/build_uk_release_input_coverage_manifest.py` gains the stage's family, in `private_rent`'s form.
  - `tools/verify_uk_identity_stability.py`'s scope docstring now names the stage's in-place rewrites, which no receipted recompute reads.
- **Generated files.**
  - `source_stages.json` regenerates by the by-name merge. Stage 5's unchanged operations keep their committed key order.
  - The H2 fixture regenerates through the engine slot, and `--check` passes. Three files change:
    - `uk_spine.json` gains the stage and its boundary;
    - `sources/fixture.json` gains stage 5's new operation, output and notes sentence and the new stage;
    - `sources/cgt_asset_type_facts.json` changes, because the tool sizes it to the oracle's frame before `cgt_residential_split`, which reads the stages the localised bill reaches.

    `property_rental_facts.json`, `property_purchase_targets.json`, `NORMALIZATION.md` and `PRODUCED_BY.txt` are unchanged.
  - `release_input_coverage_manifest.json` regenerates without an H5, and `--check` passes. It gains the new family, and every family's source-manifest digest moves.
- **What moves.**
  - Values, at one clone. Before `council_tax_localise` every existing column keeps its values: stage 5's bill is as before, and the flag is a new column.
    - Great Britain's council tax, the reported council tax reduction and the rebate move by each household's ratio: its authority's 2024-25 Band D over the expected Band D of its own draw cell.
    - Missing bills, and bills netted of a reduction of unknown size, take the 2024-25 statute where the band was observed.
    - A reduction or rebate above the new bill is capped at it.
    - SPI copies and income band donors move at their region-level cell.
    - Northern Ireland's rows are unchanged.

    The WAS bill predictor is unchanged bit for bit. The HBAI income WAS and LISA read moves with the localised bill and reduction. So can their draws, the home values the WAS chain localises, and every later stage that reads wealth, council tax or the reduction. Above one clone, clone 0's FRS rows take one clone's council tax exactly, and the SPI-channel rows nest to the share class.
  - Keys.
    - Every spine node re-keys, and so does every dense and national node after the spine. Of the stage contracts only `frs_council_tax`'s moves, and `council_tax_localise`'s is new. The UK country and solve fingerprints move. No gate phase code moves.
    - Outside the UK one kernel re-keys: the shared transport gate battery (`gates.battery@1`). Its bindings' source closure reaches `source_manifest`, where `ALLOWED_SOURCE_OPERATION_KINDS` lives. So NZ's terminal gate and the export, readback and package nodes after it re-key (`nz.gates.terminal`, `nz.export.prepare`, `nz.export.readback`, `nz.package`), the cause María accepted on 12 Nov.
    - No other NZ, AM, BE or US key moves. In those four countries' bundles, the schema edit moves only the spec engine's grammar receipt.
- **Tests**, on invented data, the toy supports and H2.
  - Engine-free:
    - the 61 stages and their order, the localisation right after `private_rent` and right before `was_wealth`;
    - no stage from the area draw to the localisation writing a draw input but the housing shell's band, and no stage after it writing its columns;
    - stage 5's three operations and two outputs, the localisation's outputs, rewrites and mass receipt;
    - the packaged stages equal to the reviewed declarations, and stage 5's transform writing the flag and its receipt from a pinned invented tab;
    - the reader isolation and the cells each reader binds, the support readers, the consumes, and the counting stages;
    - a pin that the housing shell's consumes hold no household area column;
    - the roster refusing a spine without the stage, on the dense and the national lines;
    - the support re-pin refusal for the stage;
    - the release boundary dropping both columns on every line, and the sidecar reporting no share of either;
    - the WAS predictor reading the kept bill and refusing a frame without it;
    - the coverage family.
  - Engine tier, through the slot:
    - H2 parity;
    - the driver's stage records leaving out both columns' shares;
    - `required_build_stages` and the adapter's support readers gaining the stage;
    - the localisation on all three systems at one clone and at two, after the real-engine BRMA and copy stages. Northern Ireland, an SPI copy among its rows, is left bit for bit. Great Britain's ratio rows take their cell's ratio. Reductions stay within the bill, and the kept bill is the paid bill from before the stage.
  - After merge, H2's clone-first run at 61 stages:
    - each ratio row's bill is its cell's ratio times its bill, with one where the cell holds one toy authority and other than one for some London and Scotland region-level rows, SPI-channel rows among them in each;
    - household 12 and the households whose ids are multiples of 21 take the statute, and no other row does;
    - every reduction lies within its bill, and the kept bill is the paid bill, which the WAS bill predictor reads, bit for bit, on the stage's output;
    - clone 0's FRS rows at four clones are one clone's, bit for bit, and the stage's receipt is the same at every K but for the clone count.
