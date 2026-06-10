# microframe

The micro-stack kernel. One datatype — the **`WeightedBundle`** — and strict
invariants the kernel enforces so no operator ever re-derives structure or
silently corrupts weights:

- **Entity tables** (person plus schema-declared group entities) with explicit
  linkage: `person_{group}_id` membership columns on the person table,
  `{group}_id` id columns on each group table, and globally unique column
  names across tables.
- **Typed weights** (`design | importance | calibrated`) that are validated on
  construction (finite, non-negative, not all zero), only move forward
  (design → importance → calibrated), and can be held to mass conservation.
- **Strata**: per-person provenance labels, so pool design is explicit survey
  design.
- **Weighted accounting** (`wsum`, `wmean`, `wquantile`, `wmedian`, `gini`,
  `groupby_wsum`) computed on the bundle.
- **US unit structure**: `assign_us_unit_structure` builds the PolicyEngine
  entity systems (tax units delegated to `microunit`, install via
  `microframe[us]`) and returns a validated bundle.
- **The `RulesEngine` protocol** plus a lazy `policyengine_us` adapter
  (install via `microframe[policyengine]`).

See the repository `DESIGN.md` for the charter and
`tests/test_contracts.py` for the behavioral guarantees the kernel makes.
