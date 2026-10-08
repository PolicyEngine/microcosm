Prepare the Axiom adapter for graph nodes. `NZ_SCHEMA` declares persons in
households and families. `AxiomEngine(nesting=NZ_NESTING)` refuses a family
whose members sit in two households, from the person memberships alone, and an
explicit `{group}_{parent}_id` column must agree with those memberships. A
`periods={label: AxiomPeriod(...)}` mapping resolves labels such as `"2026-27"`
to explicit `tax_year` bounds and refuses any label it does not map.

`output_dtypes="graph"` casts outputs to graph-ownable dtypes without loss:
judgments become int64 codes, integers int64, decimals float64. It refuses text
and date outputs. The default output stays as before.

`axiom_engine_ref` builds an `engine_ref` from the engine commit and wheel
digest, the RuleSpec commit, the module and whole-root content digests, and the
adapter configuration, so a RuleSpec edit re-keys every node that names it. It
also pins the adapter, which then refuses to compile from a root whose bytes
have moved. A git-checkout root must hold exactly the declared commit.
`assert_no_relations` refuses relation-bearing modules.

The new `simulate.rules_by_ref@1` kernel (`microcosm.frame.rules_kernels`) runs
several rules engines in one graph run. It routes each node to the adapter its
`engine_ref` names, with the `simulate.rules@1` contract unchanged.
