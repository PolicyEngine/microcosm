# US pre-clone relationship and eligibility graph

`microcosm.build.us_runtime.graph_person_preparation` exposes the two existing
deterministic ASEC person-preparation operations as graph nodes. Each family
declares two nodes under the actual outer phase
`prepare_multispine_source_inputs_for_clone`:

- `early_preparation.relationship.boundary` and
  `early_preparation.eligibility.boundary` validate the assembled pre-clone
  person evidence through the registered source owner, then retain every row.
- `early_preparation.relationship` and `early_preparation.eligibility` read the
  available raw and incumbent person cells, run the registered pool observer,
  and merge the family's outputs by person ID.

The declared pre-clone source order is unchanged: CPS-carried inputs, accepted
measured hours, prior-year income, relationship inputs, housing inputs,
eligibility inputs. `us_relationship_source_nodes` and
`us_eligibility_source_nodes` take `base` and `context_producer`, so the caller
stitches those neighbours. Relationship's producer is the prior-year node.
Eligibility's producer is whatever the caller binds in housing's place; the
housing adapter is authored separately and nothing in this module asserts that
a housing node ran.

## No gate node

Neither `with_us_relationship_inputs` nor `with_us_eligibility_inputs` imposes
a post-derivation scientific signal requirement on its own output, so neither
adapter declares a gate node. `us_relationship_inputs_signal_gate` and
`us_eligibility_inputs_signal_gate` remain available to release batteries that
already run them; the graph neither strengthens nor weakens that science. The
boundary node attests the source boundary and the derivation node attests
declaration completeness and output ownership — nothing else. The transitive
dependency on the accepted measured-hours gate is retained through the typed
context edge: the hours derivation's pending context has a different artifact
type and cannot satisfy an accepted-context input, so a failed hours gate
leaves every person-preparation node unreached on every resume mode.

## What stays in the registered owner

`observe_multispine_relationship_person` and
`observe_multispine_eligibility_person` own the CPS evidence selection, the
boundary role validation, the ephemeral role-metadata removal, the all-null
output omission, canonical `Weights` selection in the original person order,
and the full-pool output ID merge. The adapter module never names a source
evidence column, a support-role column, or the role-stripping helper.

Both observers call the public pass-through predicate
(`us_relationship_inputs_person_carries_signal`,
`us_eligibility_inputs_person_carries_signal`) before the always-recomputing
`prepare_us_*_person` helper. That preserves the Frame wrappers' order exactly:
a signal-carrying incumbent surface passes through without reaching the
helper's weight validation, without requiring raw ASEC columns, and without
running the source stage. The canonical person weights are still selected
before the predicate, because the direct chain's own Frame projection resolves
selected weights before it calls the operator.
Relationship pass-through requires all three leaves with more than one observed
value each; eligibility requires all five leaves and more than one observed
disability value, so a constant `is_blind` or `veterans_benefits` still passes
through. The legacy partial-null rule is preserved as-is.

Full entity counts and untouched descriptors come from the authenticated
`microcosm.us.frame_context` artifact; selected group counts use actual
memberships. No incomplete Frame carrying whole-population metadata is
constructed. The graph retains the original person index and requires the
context v1 contract's single explicit weight source and matching weight kind.

## Declared cells and dtypes

Relationship owns `is_household_head`, `is_separated` and
`is_surviving_spouse`. Eligibility owns `is_disabled`, `is_blind`,
`is_full_time_college_student`, `own_children_in_household` and
`veterans_benefits`.

The six boolean leaves are physical booleans in the helper's output and
nullable `boolean` after the full-pool merge, so they are declared `boolean`.
An incumbent for one of them that is not already nullable `boolean` is refused
at declaration time rather than cast. That is a deliberate narrowing: the
shared merge would accept a physical `bool` incumbent and materialize nullable
`boolean`, but then the declared token would disagree with the merged cell. It
mirrors the reviewed prior-year adapter.

`own_children_in_household` and `veterans_benefits` are `float64` when created
and otherwise adopt the incumbent's dtype exactly — a `float32`, `int64` or
nullable `Int64` incumbent is declared and merged at that dtype, matching the
direct chain.

Native non-CPS cells survive the declared merge: an ACS-native
`is_household_head` keeps its own values, and outputs the CPS derivation
cannot reach stay null.

A repeated person index label combined with an existing non-boolean incumbent
is refused early by name. The unchanged shared merge writes an existing
non-boolean output cell by index label (its boolean branch writes
positionally), so that combination reaches pandas as an opaque length error
late in the direct chain. The observer refuses it up front instead. The
outcome is unchanged — both paths refuse — and no merge semantics are altered.
Repeated labels on their own, and repeated labels with boolean incumbents,
stay supported.

Neither family's outputs are formula-owned, so each receipt reports an empty
`transient_outputs_carried_through_clone` and no formula-owned removal. Money
stays nominal; neither adapter uprates or aligns source-year prices.

## Implementation identity

`implementation_hash` binds the adapter module, both input modules, the shared
`_person_signal_summary` validator both of them import — the graph route runs
no gate, so that binding is conservative closure over the input modules rather
than a claim that this route executes the validator — the shared
Frame/graph/pool runtime closure, the `source_stages.json` resource, the
declared dependency versions, the live registered output family and
formula-owned projections, and the executed `POOL_RANDOM_SEED` /
`POOL_TIME_PERIOD` options. Both kernels live in one module, so an edit to
either family already invalidates the other; binding both input modules into
both scopes states that conservatism rather than implying a precision the
shared file does not have. The two families' identities still differ, because
each binds its own schema key, contract projection, reads and outputs.

## Evidence and limits

The fixtures are tiny synthetic multispine frames. The composed graph runs
CPS-carried, hours, prior year, relationship and eligibility, with a declared
population/context stand-in in housing's place, and matches the corresponding
direct operator subsequence byte-for-byte on values, ordered receipts and
context across cold, auto and required replay. A separate fixture runs the real
direct housing operator and exercises the eligibility node on the state it
leaves; that fixture re-declares housing's `str` tenure column as canonical
`string` storage, which is a housing-side storage seam and is out of scope
here. Repeated person index labels are covered three ways: alone, with boolean
incumbents, and with a numeric incumbent (where the early refusal fires and the
direct chain's inherited pandas error is reproduced alongside it). The complete
six-operator direct chain is pinned to its pre-edit byte digests measured on
base `ebb47671`; `us-person-source-author-oracle-capture.py` in the review
directory re-measures them and exits 0 at both base and head.

This is a development slice. Nothing here certifies a data artifact, a housing
implementation, or the scientific quality of either family's outputs.
