# Invented native tail structural expansion

`microcosm.build.us_runtime.graph_native_puf_tail_expand` implements a real graph
EXPAND for a bounded invented fixture. It copies selected whole clone1
households to clone2, splitting each selected parent's household IMPORTANCE
weight equally between the parent and copy. It does not place donor money.

The fixed policy is `native_tail_matched_parent_half_importance_v1`, with alpha
`(1, 2)`. Donor support weights remain descriptive donor-selection metadata;
they never become household weights. This is a support-expansion convention,
not a PUF return-to-household conversion or a reproduction of PR #964's
weighting. Alpha is fixed prospectively and is not a tuning parameter.

## Declaration and graph API

`InventedHouseholdDomain` describes each household's domain, statistical unit
and source-origin coordinates. Supply exactly one row for every incoming
household, including unselected households and group quarters. Domain units
come from `survey_population_domains.declaration`; the input cannot infer
housing status from a household ID or group size. Repeated origin coordinates
must have consistent domains across clones.

`InventedTailAssignment` supplies a unique donor ID, a unique selected parent
household ID and a finite positive donor support weight. The pairs are already
chosen: no filing-status, AGI-band, head/spouse or modeled person-allocation
qualification occurs in this API. `declare_invented_tail_expansion` validates
and snapshots these records in canonical order. The immutable result and its
digest establish descriptive consistency, not source authority. It always
reports `source_admission_issued=False` and `release_eligible=False`.

`native_puf_tail_expand_nodes(columns, base=..., declaration=...)` returns an
EXPAND and a same-version claimant for the six clone-index columns. Register
them with `register_native_puf_tail_expand_kernels`. The incoming population
has the US schema, IMPORTANCE household weights, and clone indices 0/1. This
contract does not require or invoke a legacy exact-two-clone validator.

The kernel reads the four maintained provenance columns for all six entities
through declared slices. The executor supplies their structural IDs and all
five person memberships. The complete domain roster must match the actual
household IDs and origin coordinates in that context. These are **fixture
declarations**; matching the context does not authenticate an actual survey.

## Structural and weight invariants

Only declared occupied-housing clone1 parents can be selected. Each must have
one tax unit and all groups must be household-closed. Every group row must have
members, and their source channel and clone index must agree. Unknown domains,
duplicate donors or parents, conflicting origins, group quarters and clone0
selection refuse. Existing clone2 rows also refuse; this is one expansion.

For every selected positive parent weight `W`, both descendants receive
`W / 2`. Halves must be finite, positive and sum to exactly `W` in float64.
Unrepresentable submitted splits refuse; there is no epsilon floor, clipping
or redistribution. Selection-level exclusion is a later contract. Clone0,
unselected clone1 and unselected zero-weight/GQ records keep their weights.

The kernel returns six new-ID-to-parent-ID lineage Series, six clone-index
overlays, and the full typed household weight vector. IDs are deterministic,
collision-free int64 values above each entity's current maximum; exhaustion
refuses. The real executor copies columns, remaps memberships, carries design
anchors and per-person strata, and checks `mass="conserve"`. It receives no
opaque final Frame from this kernel. Arm provenance does not create a new mass
stratum, and this slice declares no partition-ledger guarantee.

The opt-in executor parameter `expand_require_sole_weight_entity=True` requires
the declared EXPAND weight entity to be the sole explicitly stored weight
entity. It is an exact boolean, not a truthy control. The executor checks cold
input/output and cached input/output topology. This detects an extra person or
group vector even when it numerically equals inherited household weights.
Nodes omitting the flag retain the previous behavior; explicit `False` also
keeps that behavior.

The receipt binds the fixed policy, declaration digest, donor/parent/new-household
pairs, separate donor support and host IMPORTANCE weights, and all six lineages.
The graph key includes the parameters, declared provenance artifacts, kernel
implementation and complete base population identity. Required replay must
restore validated stored artifacts and cannot silently refit or rebuild.

## Verification scope and remaining work

The invented tests construct a small two-survey, six-entity, two-arm fixture
and use the actual graph executor. They check lineage, five memberships,
nullable/signed-zero storage, fixed-assignment donor-weight independence,
per-parent halves, effective entity and stratum mass, design anchors, cold and
required replay, and refusal paths. The graph-core tests cover the strict
topology flag on cold and cached paths. Test source alone is not a passing
result; execution evidence must be recorded separately.

This structural prototype carries existing columns verbatim to copied rows.
That does not qualify those values as final clone2 monetary inputs. The actual
source/domain issuer, donor selection and modeled role qualifier, native52
placement, SS/SCF and three late owners, and variable-clone receiving joins are
separate work. Public fixture dataclasses cannot substitute for their owners.

The full-spine host must place this expansion before clone-specific census
block assignment and derive larger geographies from the block. This primitive
does not validate or perform that host ordering: it must not be used to claim
that copied preexisting geography is a qualified clone2 draw. Actual-source
qualification, joint administrative-target feasibility, calibration, sparse
concentration/survival checks, engine compatibility and release acceptance
remain incomplete. Poverty is comparison-only, never a tuning, selection or
release criterion. Prior wages are excluded.
