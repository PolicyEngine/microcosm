# populace-frame

The populace kernel, imported as `populace.frame`. One datatype — the
**`Frame`**, a weighted *sampling frame* of entity tables (the
survey-statistics sense: the list of units a sample is drawn from and the
thing weights refer back to) — and strict invariants the kernel enforces so
no operator ever re-derives structure or silently corrupts weights:

- **Entity tables** (person plus schema-declared group entities) with explicit
  linkage: `person_{group}_id` membership columns on the person table,
  `{group}_id` id columns on each group table, and globally unique column
  names across tables.
- **Typed weights** (`design | importance | calibrated`) that are validated on
  construction (finite, non-negative, not all zero), only move forward
  (design → importance → calibrated), and can be held to mass conservation.
- **Strata**: per-person provenance labels, so pool design is explicit survey
  design.
- **Links** (experimental placeholder): `LinkSpec` declares many-to-many
  associations between entities (e.g. a `jobs` link between persons and
  firms); the frame validates link tables on construction. Firm tables and
  person-firm links are experimental, and the full link operator comes later.
- **Weighted accounting** (`wsum`, `wmean`, `wquantile`, `wmedian`, `gini`,
  `groupby_wsum`) computed on the frame.
- **US unit structure**: `assign_us_unit_structure` builds the PolicyEngine
  entity systems (tax units delegated to `microunit`, install via
  `populace-frame[us]`) and returns a validated frame.
- **The `RulesEngine` protocol** plus a lazy `policyengine_us` adapter
  (install via `populace-frame[policyengine]`).

See the repository `DESIGN.md` for the charter and
`tests/test_contracts.py` for the behavioral guarantees the kernel makes.
