# Declared transport units and binary relations

`transport.create@2` accepts a versioned `unit_rule` document, using the same
donor, fact, seed and digest parameters as `transport.create@1`:

```json
{
  "format": "microcosm.multi_unit_rule.v1",
  "units": [
    {
      "kind": "tax_unit",
      "rule": {
        "format": "microcosm.tax_unit_rule.v1",
        "entity": "tax_unit",
        "pairing_eligibility_column": "joint_assessment_eligible",
        "provenance": {"eligibility": "The declared source of eligibility"}
      }
    }
  ]
}
```

A declaration requires at least one entry and distinct physical entity names;
unknown kinds are refused. Each entry dispatches to a registered membership constructor. `benefit_unit`
uses the existing `BenefitUnitRule.to_dict()` format. The legacy CREATE reads
the unchanged NZ country rule and lifts it to a one-entry declaration. CREATE,
CLI inventory preparation, unit attributes, group encoding and the rules
registry share the declarations and their derived entity schema and nesting.

Tax units consume an explicit, non-null boolean eligibility column. Both
members must be eligible reciprocal co-resident partners to share an assessment;
every other person in a valid concept frame has a singleton assessment. Invalid
self, non-reciprocal or dangling pointers are refused by the shared concept
validation. Minors' assessment membership
does not allocate dependant relief. The smaller physical person ID is the
head (role `head`, spouse A); the other is `partner` (spouse B). This rule reads
no income and implements no Belgian legal eligibility test. Its eligibility
input must already exist in the supplied concept tables; CREATE's donor
boundary continues to carry only declared transport concepts. A country must
provide and document that input before activation.

All partitions use the same validation, household nesting and source-household
lineage. Units store no weights. `Frame.resolve_weights(unit)` inherits each
unit's household weight once. Household totals use the household table;
unit totals count the units explicitly. Changed membership changes CREATE's
population identity. Unchanged membership keeps IDs under row permutations and
income changes. Physical support clones keep distinct IDs and shared original
source-household lineage.

`transport.unit_attributes@1` accepts either declaration format. It can emit
`person.<entity>_role` for each partition when the node owns that output. Legacy
nodes keep their declared output inventory. Group encoding uses the sole
declared partition for legacy and one-entry declarations. With several
partitions, its node must name the physical `unit_entity`; `engine_entity`
separately names the logical engine entity. The person/household concept
mapping does not choose group partitions.

Binary Axiom bindings opt in with `relations` on the binding row. For example:

```json
{
  "predicate": "be:statutes/example#relation.spouse",
  "key": "be:statutes/example#relation.spouse:0:1",
  "current_slot": 0,
  "related_slot": 1,
  "arguments": ["TaxUnit", "Person"],
  "root_entity": "tax_unit",
  "related_entity": "person",
  "membership_column": "person_tax_unit_id",
  "role_column": "tax_unit_role",
  "roles": ["head", "partner"],
  "cardinality": "many",
  "period": "2026",
  "related_inputs": {"professional_income": "input_professional_income"}
}
```

The predicate and key use the compiler's canonical scoped names. Explicit tuple
coordinates are checked against compiled current/related slots and the source
relation declaration. The expected related input map must match the union of
compiled schemas sharing that key. `one` means exactly one selected person per
root; `many` permits zero. A role selector requires both its column and distinct
role values. This interface supports person members of genuine group partitions;
binding-local views and arbitrary links remain separate work.

Rules nodes declare whole-table slices for related physical IDs, partition
membership, selector columns and every mapped related input; composition refuses
a missing dependency. The adapter gathers per-root offsets and input arrays,
ordering people by physical identity and retaining distinct equal-valued people.
Data-derived batches are built at execution and never enter prepared adapter
configuration. Relation-enabled execution requires the native dense API with
`DenseRelationBatch.related_row_count`, including for relations with no related
input columns. Undeclared compiled relations continue to refuse.

Counterfactual and solve-zero bridges currently reject overrides of aliased
related inputs, where the engine input name differs from its physical column.
Either spelling of the alias refuses before native execution. Resolving these
overrides is deferred; ordinary relation materialization uses the declared map.

Relation declarations enter the binding and engine identity. An absent
declaration remains absent; an explicitly supplied empty declaration is distinct.
Bindings without relations retain their existing engine-reference serialization
and native execution call shape. Calibration roster counts use the same frame
unit framework, with frozen-kernel differential tests preserving NZ results.
