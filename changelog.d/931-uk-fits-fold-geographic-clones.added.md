The UK fits trained on the frame's FRS rows now fold geographic clones into each survey record before they fit (#931). This covers the SPI stage-2 and reference forests, the housing shell, the UC reporter redraw, the UC capital donor cells, the salary-sacrifice forest and the SPI employment shell.

- **How the fold works.** Each fit passes what it reads: its predictors and targets, or the columns they derive from. A record's clones that agree on it keep the record's first training row, with the weight of those clones summed exactly. Clones that differ, through a draw keyed on their own ids or a stage run per area, stay rows of their own, each variant with its own clones' summed weight. SPI, high-value and CGT copies stay rows of their own, as now.
- **No change today.** Today's spine has no geographic clones, so every fit's rows and weights stay as they were, value for value.
- **What changes once the spine clones before enrichment:**
  - the fits see each record once wherever its clones agree, and every clone's value at its share wherever they differ;
  - the reference transport meets no duplicate twins from agreeing clones;
  - the UC capital cells' donor minimum counts records, not identical clones.
- **The country-neutral pieces.** `microcosm.build.source_lineage` gains three functions:
  - `copy_paths`: every household's ordered copy path from a declared lineage, never inferred from ids, weights or values;
  - `replica_keys`: a key per record of any entity, shared only by its replicas in the named branches, with a member's rank in its household telling members apart;
  - `collapse_replicas`: the fold itself, which takes the fit's `inputs` and folds only replicas that agree on them.
- **The UK entry.** `household_lineage.collapse_geographic_clones` is the UK's entry and requires those inputs. `uk_household_clone_paths` now reads the clone path through `copy_paths` on `UK_COPY_LINEAGE`, with identical paths and draw keys; three identity-test messages name the shared reader's branches.
- **Declarations and tests.**
  - The collapsing stages declare the lineage live where they run, and a test refuses one whose surface would hide it.
  - Replicated-fixture tests hold each fit to the rows and weights it fits without clones.
  - A fixture with differing clones (a landlord's property income drawn apart) holds the UC reporter redraw to training on both variants at half the unit's weight each.
- **Hashing.** The SPI income stage also binds `spi_rank_transport` into its hash.
