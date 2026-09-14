# Fiscal input policies

The declared fiscal measurement stage requires an input producer for every
leaf in its static model closure. Missing and nonfinite values still refuse
evaluation. The optional leaf policy records more precise intent without
weakening that default.

An explicit policy can describe producer ownership or a proposed literal
scenario assumption. **Assumption-bearing executions currently refuse with
`FISCAL_MEASUREMENT_ASSUMPTION_PARENT_ADMISSION_UNSUPPORTED`.** No US assumption
resource is enabled, and no dataset column or producer requirement changes.

## Exact policy identity

`load_fiscal_leaf_policy(payload, expected_sha256=...)` accepts bounded,
canonical JSON bytes and an explicit SHA-256 pin. It performs no file or
network I/O. The same `FiscalLeafPolicy` goes to `fiscal_measurement_node` and
`FiscalMeasurementKernel`. Its pin is rechecked when used; the parsed document
is detached. Omitting the argument, or passing `None`, preserves the previous
all-producer declaration shape and runtime behavior.

A policy has schema version 1, artifact kind
`microcosm.us.fiscal_leaf_policy`, one positive integer `period`, an exact
ordered list of model `roots`, and an `entries` mapping covering the static
closure exactly. Missing entries and stale entries refuse. Each entry's
entity must agree with the independently derived model metadata.

| Kind | Required entry fields | Behavior |
| --- | --- | --- |
| `producer` | `kind`, `entity`, `producer` | The declared input column must exist and every cell must be known and finite. |
| `assumption` | `kind`, `entity`, `value`, `interpretation`, `period`, `roots`, `affected_roots`, `affected_programs`, `rationale`, `reviewer`, `source_issue`, `reform_sensitivity` | Validated and inspectable; execution remains unsupported. |
| `inactive` | No supported shape | Always refused. |

An assumption's interpretation is explicitly `baseline_behavior` or
`scenario_parameter`. Neither category means the program is insensitive to
the choice. The affected roots must include every declared root whose static
closure contains that leaf, in declaration order. Scope must exactly match
the policy's period and roots; program names and review fields cannot be
empty. These fields describe claimed review and intent. They do not prove
reviewer authorization, establish a source observation, or make an observed
attribute an acceptable assumption.

V1 supports finite Boolean, signed 64-bit integer and floating-point literals
with explicit engine type checks. Strings, enums, null, containers and an
`engine default` reference refuse. The actual engine metadata and whether a
default is available (with its literal value when present) are recorded
beside the proposed value. The host never chooses a value by default lookup.
The exact policy and engine/default record enter `model_contract`, so policy,
scope, ownership, review or engine-default changes alter the declared node and
its cache key. The same declaration is embedded in a successful measurement
artifact. In this version, only producer policies can produce that artifact.

A column constant at an engine default remains admissible as a producer at
this boundary. The label does not establish observation or modeled variation.
The source owner and existing release input-coverage/variation gates retain
that responsibility. Behavioral take-up and filing propensities are not
automatically classified as innocuous scenario settings. No prior wages are
added by this infrastructure.

## Complete-parent admission still required

The graph kernel receives only its declared input slices. It cannot prove a
column is absent from the complete parent when a caller omitted that column
from the slice. Declaration checks reject an assumption name present on any
declared entity, but this is insufficient to enable execution.

The future country host must retain its independently checked complete
population and use `validate_fiscal_leaf_policy_population(frame, policy)`
before projection, around relevant I/O, and during cache admission. The
validator rejects any incumbent assumption name on any entity, including
wholly unknown or partially unknown columns. Passing a projection or a
caller-authored column roster is not absence proof. The validator itself
does not authenticate or issue a parent.

`private_fiscal_assumption_frame(frame, policy)` is a detached preparation
helper for that future integration. It repeats the collision check and
creates engine-only tables. It preserves the supplied frame's values,
membership, weights, strata and metadata. The current fiscal kernel does not
call it. A future host must validate the exact policy against the admitted
engine, retain source and complete-parent checks before and after I/O, bind
the engine/default record, evaluate only the private copy, and keep assumption
columns out of the population and measurement adapter. A cached artifact
cannot supply missing source or parent authority.

## Why inactivity is unsupported

`validation_input_coverage._provision_input_leaves` traces a single empty
one-person simulation. It observes the leaves read by that execution. A
conditional formula or `defined_for` branch may read additional leaves for
other households. Absence from that trace cannot establish universal
inactivity, even at the same period and roots.

Future inactivity support needs exhaustive engine-, period- and root-scoped
dependency proof, including conditional paths and declared state roots. The
static closure remains conservative; neither it nor a sampled trace supplies
this exemption today.

## Bounded implementation evidence

The local source/mock test run passed 91 cases, including the existing
invented fiscal measurement, calibration and observer graphs. The new controls
check strict-default compatibility, malformed policies, exact scope and
closure coverage, type mismatches, private-copy isolation, omitted-column
collisions, late policy mutation, graph key changes and required replay with
a forbidden model call. There were no country engine imports, network
connections or subprocess operations; no native population or source data
was used. This is infrastructure validation, not actual engine-policy
admission or candidate fiscal validation.

The source and selected test files were sealed before and after the run.
Receipt SHA-256:
`c7aed643333638460d7464091be0c068b910baba2eefbb76cfa1208af6f8265f`.
JUnit SHA-256:
`54215b02c7a8a85488db48b2fee339b0ddbad907a0985502019c185b9a1c3adb`.
