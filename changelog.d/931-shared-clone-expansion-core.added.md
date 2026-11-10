A country-neutral clone expansion core, `microcosm.build.clone_expansion` (#931; shared now, as María ruled on 10 Nov). `expand_clones(frame, clones=K, declaration=...)` turns a frame into K clones of every record, clone-major. Clone 0 is the frame's own rows with their ids, and clone `k` repeats them with every id minted by the country's rule. The module names no country's columns, tables or constants: it reads each entity, id column, membership column and weighted entity from the frame, and the country passes the rest in a `CloneDeclaration`.

- **Declarations.**
  - A `CloneIdRule`: the most clones the rule numbers (`bound`), `mint(ids, k)` for clone `k`'s ids in the ids' own dtype, and `require_below(ids, what)`, which refuses incoming ids the mint has no room for.
  - A clone-index column for every entity.
  - The weight kinds the expansion accepts.
  - The mass records' reason, declared once, so it is the same at every K.
  - Optionally, a column per entity for each row's id before the expansion.

  A declaration compares and hashes by identity.
- **What the clones carry.**
  - Each entity's ids are minted for clones 1 to K - 1, and a person's membership in clone `k` is its unit's clone-`k` id. Every other column is copied unchanged. Ids and memberships keep their dtype at every K.
  - Each row carries its clone ordinal, with exactly K values on every entity, and, where declared, its clone-0 id.
  - Every weighted entity's weights become `numpy.tile(w / K, K)`: one float per row of the frame, the same in each of its clones, of the entity's kind. Strata repeat once per clone and metadata is kept.
  - The mass log gains one record per weighted entity, in the frame's entity order, with factor 1.0 and the declared reason. Each entity's clones' total must equal the frame's within a relative 1e-9.
  - At K = 1 the rows, ids and weights are the frame's own; only the declared columns and the records are added.
- **Refusals.** Each names entities, columns, dtypes, weight kinds and declared constants only, never an id or a weight. The core refuses:
  - a clone count that is not an integer in `[1, bound]`;
  - a declaration that names an entity the frame lacks, or gives one of its entities no clone-index column;
  - a frame that already carries a declared column, or carries a link table;
  - weights, on any entity, of a kind the declaration does not accept;
  - ids or memberships that are not numpy integers, memberships in another dtype than their unit's ids, and ids the rule's `require_below` refuses;
  - a mint that does not return one integer per id in the ids' own dtype, does not keep the ids' order, or does not put every clone's ids above the clone before's.
- **Not covered yet.** Link tables and a different number of clones per record. These wait for a caller that needs them.
- **Tests.**
  - `tests/engine_free/shared/test_clone_expansion.py` runs the core under neutral and US-shaped declarations on invented frames. The US-shaped frame weights households and people, each split with a record of its own, and its clones, labelled from their clone index, pass the shared two-channel invariant `outer_stage_runtime.assert_clone_expansion`. Narrow ids keep their dtype at every K, and each refusal, the dtype ones included, is checked to print no id or weight.
  - `tests/engine_free/uk/test_uk_clone_expansion_core.py` runs it under the UK's declarations: the copy-id table's `geographic_support` mint and `require_below`, the ladder's clone-index columns, `source_household_id`, and design weights on households. At K = 1, 2, 3, 7 and 15 it reproduces the clone-first stand-in `clone_first_frame`, which is written without the core, bit for bit on the ids, the household clone index, the source household id and the weights.
- **No caller yet.** No build module imports it yet, so no output and no node key moves.
