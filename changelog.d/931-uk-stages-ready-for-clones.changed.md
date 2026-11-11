The UK spine gains G1 and G1b, the steps that give it geographic clones (`931-uk-spine-geographic-expand` declares and wires them), and three of its stages are made ready for clones: salary sacrifice draws SPI bonuses within each clone, UC capital counts its donors in survey records, and the CGT support split reads the clone count from a module of its own (#931). No output moves: every spine is one clone today.

- **Clone-index names.** `ladder_clone_index_column`, `ARTIFACT_CLONE_INDEX_COLUMN` and `geographic_clone_count` move unchanged to `microcosm.build.uk_runtime.geographic_clones`, which imports nothing first-party. `rowwise_dataset` and `cgt_support` re-export them, so every importer keeps working. The module joins the spine's counting modules, so every stage that counts by lineage hashes it. The CGT support split, which reads the clone count, binds it as a helper. The full build's population and export kernels, which read the names, hash it too: `uk.full.expand` and the other population kernels, `uk.full.identity`, `uk.full.property.value`, `uk.full.private_rent` and both export descriptions.
- **Geographic expansion (G1).** `geographic_expand.expand_uk_geographic_clones(frame, clones=K)` and its stage transform, `UKGeographicExpandStageTransform(stage, clones)`, turn a spine frame into K clones of every FRS household, clone-major. They call the shared clone expansion core, as María ruled on 10 Nov, with the UK's declaration, `UK_GEOGRAPHIC_CLONES`:
  - clone `k` takes `id + k × 10^9` on all five id columns, in the copy-id table's `geographic_support` field, through two named functions that call the table (`mint_geographic_clone_ids`, `require_room_for_geographic_clones`); clone 0 keeps the frame's ids;
  - each table carries its clone index (`person_clone_index`, `benunit_clone_index`, `household_clone_index`), 0 to K - 1, with exactly K values;
  - each clone takes `w / K` of its household's design weight, the same float in every clone, and the mass log gains one record whose reason names no clone count;
  - each household carries the source columns the SPI support channel wrote before G1 took them over: `source_year` (the frame's time period) and `source_household_key` (`"{source_year}:{source_household_id}"`, the values that channel wrote, as Python strings in an object column, the dtype the legacy plan gives it; the graph casts it to its string dtype), written before the clones are made, and `source_household_id`, its clone-0 id, which the core writes.

  It refuses, by name or by count only:
  - a clone count outside 1 to 100 (`GEOGRAPHIC_CLONE_LIMIT`, the field's bound);
  - weights on any entity but the household, or of any kind but design;
  - a frame already carrying a clone index, a source column or a column of the UK copy lineage;
  - id or membership columns that are not int64, the dtype the table mints, at every clone count, so one clone takes the same frames as more;
  - a time period that is not a year;
  - incoming ids at or above the field's radix;
  - whatever the core refuses.

  Its evidence gives K, the copy-id field and its radix, the distinct source households rounded through `receipt_counts` at the floor of three, and the weight split and kind, with no row count, id or weight.
- **Geography identity (G1b).** `geographic_identity.key_uk_households_by_clone_path(frame)` and `UKGeographicIdentityStageTransform(stage)` give every household of G1's frame the key the dense line's `uk.full.identity` writes, `geography_household_key`. The key is `household_draw_key` over:
  - the source `frs`;
  - the FRS vintage frs_release.json pins for the household's `source_year`, which maps the release's survey year and its base year, the frame's time period, to its vintage;
  - its `source_household_id`;
  - its copy path, read from the lineage columns through `source_lineage.copy_paths`: empty for clone 0, and the `geographic_support` branch at `k` for clone `k`.

  So each clone keys as the dense identity keys that clone, and a larger K keeps every key. The key is in the graph's `string` dtype (`StringDtype("python")`). It refuses, by count or by name and never printing a key, a frame that already carries the key or lacks G1's int64 columns, a source year frs_release.json pins no vintage for, and repeated keys. Its evidence gives the distinct source households rounded through `receipt_counts`, that the keys are unique, the source, the vintages and the columns it reads.
- **SPI bonuses per clone (salary sacrifice).** `draw_spi_bonuses` takes `clone_indices`, each row's geographic clone. The stage passes each person's household `household_clone_index`, and a frame without the column is one clone.
  - The pay cells and the donor-source floor still pool every clone's donors, the floor counted in distinct source households.
  - Each SPI-channel adult now draws its bonus share from the donors of its own clone in its cell, in id order, on the same person-keyed uniform.
  - At one clone the draw is unchanged, value for value. Where the clones agree on the donors' pay, clone 0 of K clones draws on the one-clone pool at `w / K`, up to the 1/K split's rounding, which can move a pooled cell edge where a quantile falls exactly on a cumulative weight share.
  - A cell with no donor in an SPI adult's clone is refused, with no count, id or value in the message.
  - The declared draw text (`spi_bonus_draw`) changes with the declarations.
- **UC capital donors counted in survey records** (María's ruling of 10 Nov). The stage folds its donors through `fold_geographic_clones`, and a cell stops widening once it holds `minimum_cell_donors` survey records: the distinct replica keys (`ReplicaFold.key_group`) among its donor rows, not the rows.
  - A unit's geographic clones count once, whether they fold into one row or differ and donate apart. Any other copy, such as a high-value support copy, counts as a record of its own.
  - A donor variant donates from its lowest-id clone.
  - Without geographic clones every row is its own record, so the count and every draw are unchanged.
  - The stage's notes change with the declarations.
- **What moves.** No output, at one clone or on any spine: both stages give the parent commit's output bit for bit on invented frames, with or without a one-clone index. Node keys move:
  - every UK spine node. Every spine kernel hashes `graph_kernels`, which this commit edits, and the stages that count by lineage now also hash `geographic_clones`. Keys chain, so in the in-graph build every node after the spine moves too;
  - on a bound spine, whose checkpoint kernel hashes no edited module, the whole dense line after the checkpoint: its population kernels hash `rowwise_dataset`, `graph_population` and now `geographic_clones`;
  - on a bound spine, the national line from its problem on. The national problem and the calibration nodes hash `graph_population`, which this commit edits to bind `geographic_clones` into the population kernels' keys, and the national gates' registry modules include `uc_capital_coherence`. The national targets keep their key;
  - the full gates' terminal phase code, which hashes `uc_capital_coherence` and `graph_population`.

  No generated file, declaration or `gates.json` changes. No transport kernel and no US, NZ or BE node hashes an edited module; only the US PUF-support builder's whole-tree resume fingerprint moves, as on any edit to packaged source.
- **Tests.**
  - `test_uk_geographic_expand.py` holds G1 to the clone-first stand-in `clone_first_frame` bit for bit at K = 1, 2, 3, 7 and 15. It also checks the one-clone frame and the mass log it is handed, the layout at K = 3, the source key's dtype under either string inference, the mass record, the evidence, the declaration's named functions and each refusal, the int64 one at one clone and at two.
  - `test_uk_geographic_identity.py` holds every clone's key to the dense identity kernel's at K = 1 and 3. It also checks the clone path, uniqueness and nesting across K, the dtype, the vintage for the release's survey and base years, the evidence and each refusal.
  - `test_uk_clone_first_nesting.py` runs an invented frame through G1 and stages 16, 17 and 42 at K = 1, 2, 3, 7 and 15. Clone 0 equals the one-clone frame on its ids, G1's columns, stage 16's copies and stage 17's seats, and every frame and copy keeps the clone-index invariant. K × each clone-0 weight matches one clone's within stated rounding classes, stage 42 nests approximately, and through G1 rule B's correction lands on a copy at 7 and 13 clones.
  - The salary-sacrifice and UC capital tests cover the per-clone draw and its refusal, the stage at one and three clones, a one-clone index, clones that differ, and a high-value copy.
  - `test_uk_cgt_support.py` holds every spelling of the household clone index to the module's name, and its nesting scenario's data move to `test_support` for the nesting test to share. `test_uk_clone_expansion_core.py` takes G1's declaration, and `test_uk_copy_id_table.py` lists `geographic_expand` among the copy-id readers. `test_uk_graph.py` re-derives the binding tables, checks which nodes write the clone index columns, and checks that an edit to `geographic_clones` re-keys every counting stage.
