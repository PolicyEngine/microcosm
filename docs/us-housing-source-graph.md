# US pre-clone housing source graph

`microcosm.build.us_runtime.graph_housing_source` exposes the existing
pre-clone housing stage as a fresh-CPS-source recipe over the assembled pool.
Every node declares the actual outer phase
`prepare_multispine_source_inputs_for_clone` and the explicit `fresh_source`
mode:

- `early_preparation.housing.source` is a genuine `FILTER` of the caller's full
  pool with `mass="free"`. It validates the assembled pre-clone person evidence
  through the registered source owner, selects exactly the rows the direct
  operator chain selects, and refuses observed CPS `pre_subsidy_rent` before any
  recipient work.
- `early_preparation.housing.measured` calls `derive_us_housing_tables` on the
  declared source person/household/SPM views and owns `household.tenure_type`
  and the three SPM leaves on that projection.
- The existing `early_preparation.rent.*` nodes run unchanged on the projection:
  the authenticated donor `CREATE`, the real fit, the recipient matrix and
  placement, the raw draw and apply, the assignment-integrity gate and the
  rent-only finalizer.
- `early_preparation.housing.source_outputs` exports the five completed leaves
  as immutable typed bytes bound to the projection, the rent final context and
  the completed rent validation.
- `early_preparation.housing.merge_boundary` is an all-row `FILTER` of the
  original full pool that depends on the completed export, so rewritten cells
  cannot create a source/merge cycle.
- `early_preparation.housing.merge` calls the registered ID merge once per
  affected entity and owns all five full-pool leaves.
- `early_preparation.housing.gate` re-derives the selection, re-checks the merged
  and native cells against the bound evidence, and only then publishes the
  accepted `US_FRAME_CONTEXT_TYPE`.

The production order is relationship, then housing, then eligibility. Pass the
relationship stage's population as `base` and its accepted context producer as
`context_producer` to `us_housing_source_nodes`. Eligibility must consume
`early_preparation.housing.gate`'s context edge, not merely the output columns,
so a withheld acceptance leaves it unreached. The merge node's pending context
has a different artifact type and cannot satisfy that accepted-context input.

The registered `multispine_pool` helpers own the source facts. Selection and
role validation come from `multispine_housing_source_selection`, which is the
same `_cps_person_evidence_mask` the direct chain applies; the merge and its
ordered receipt come from `merge_multispine_housing_source_outputs`, which
calls the unchanged `_merge_source_person_or_group_outputs` and
`_source_operator_receipt`. No selection semantics or merge formula is
rewritten here. Role columns ride along on the FILTER because a structural node
carries every column of its base; the measured node removes them from the
ephemeral scientific views exactly as `_source_available_projection` does.

The projected operator context has empty metadata and an empty legacy
`mass_log`, matching the original source-operator view. Parent authority — the
full-pool context producer key and digest, both identity sets for every actual
entity, the original person index, the weight-source descriptor and the live
output family — travels in the separate
`microcosm.us.housing_source_projection` artifact, which carries no
caller-authored pass flag. Selected group identities are derived from actual
membership vectors and checked against the parent descriptors before the rent
handoff on cold, auto and required replay.

This slice is `fresh_source` only. Selected CPS rent must be absent or entirely
null; partly or completely observed CPS rent refuses at the boundary, before
the donor is fitted and before any recipient work. The authenticated donor
`CREATE` is recipient-independent by design and may still load first; that is
the same preflight the caller already performed to build the descriptor. The complete-incumbent pass-through remains owned by
the unchanged direct `with_us_housing_inputs`, and this recipe does not claim
it. There is no post-clone housing-assistance QRF, no target activation change,
no source-period money adjustment, and no change to the rent science or its
gate thresholds; `source_year` money remains nominal.

The five outputs are exactly `person.pre_subsidy_rent`,
`household.tenure_type`, `spm_unit.receives_housing_assistance`,
`spm_unit.takes_up_housing_assistance_if_eligible` and
`spm_unit.spm_unit_tenure_type`, read live from the registered
`housing_inputs` output family rather than restated here. On the source
population the two assistance leaves are physical `bool` and both tenure leaves
are the canonical graph `string` (python storage, `pd.NA`); the registered
full-pool merge normalizes the booleans to nullable `boolean`, exactly as the
direct helper already does. Full-pool tenure incumbents must already use that
canonical string representation: an absent or numeric incumbent is refused at
declaration rather than silently widening the scope to object dtype. Native
non-CPS rent and tenure cells survive the merge unchanged, and the merge
evidence digests them before and after so the gate can attest it.

The typed `microcosm.us.housing_source_outputs` payload carries ordered int64
IDs and exactly five column buffers — little-endian float64, canonical
single-byte booleans, and UTF-8 offset payloads with an explicit null mask that
cannot hide bytes. It rejects schema, ordering, length and digest mismatches,
duplicate or negative IDs, empty tables, foreign string storage, and any
encoding, null count or byte width that disagrees with the declared
representation. Adjacent string offsets are compared rather than differenced,
so int64 wraparound cannot hide an overlapping or out-of-range range, and every
evidence digest is domain-separated by coordinate, dtype and row count. There
is no pickle transport and no arbitrary object dtype support.

Resources were measured, not estimated. On the eight-person mixed fixture the
stage compiles to fifteen nodes and five population versions
(`source`, `early_preparation.housing.source`, `early_preparation.rent.donor`,
`early_preparation.rent.boundary`, `early_preparation.housing.merge_boundary`),
a 2.6 MB content store dominated by the donor frame and the fitted forest, and
typed payloads of 2,512 B (projection), 2,633 B (five-leaf export) and 3,313 B
(merge evidence). The FILTER carries the source population's existing columns,
so the two extra structural versions are real physical copies; the logical
subset does not eliminate them. Record this profile again before any larger
rung. No cache GC or format redesign is proposed here.

The outer gate's scope is `source_housing_completion_integrity`: source
projection, exact ID merge and completion. It is deliberately **not** a new
mandatory post-fit housing signal gate. `us_housing_inputs_signal_gate` keeps
its original role of deciding complete-incumbent pass-through in the direct
helper, and the established terminal quality and release gates remain
authoritative. Making it mandatory here would change the legacy source recipe
and is a separate approved scope.

The inherited shared-merge label seam is refused, not worked around.
`_merge_source_person_or_group_outputs` writes a non-boolean output through
`target.loc[target.index[positions]]`, which reaches every row sharing a
repeated pandas index label. `assert_multispine_source_merge_labels_supported`
refuses that combination at this explicit scope boundary; fixing the shared
merge is a separate reviewed scope, and ordinary unique-label frames retain
exact original parity.

Each source receipt describes its single real operator with local
`order_index=0`. Root composition retains the authored
relationship/housing/eligibility order when assembling the full receipt; these
ordered source receipts and the preserved legacy `mass_log` in the typed context
are distinct from the executor's structural mass ledger.

Implementation identity binds this adapter, the strict artifact codec, the
existing rent adapter and donor preparation, the actual housing helpers, the
registered pool and operator-boundary modules, the shared Frame/graph runtime
closure, the source-stage resource and stage declaration, dependency versions,
the live registered output family, the declared source and merge
representations, and the resolved pandas string-runtime configuration. This
conservative module hash also rekeys the shared donor preparation on helper
edits; it does not promise adapter-only invalidation. No source-provenance
exemption and no frozen graph interface change is introduced.

Full graph composition is not claimed. The additive relationship graph is still
under review, so the landed integration test supplies the true direct
relationship-produced Frame as `base`, and the eligibility consumer is a stub
that only proves the accepted-context edge is required. Wiring the real
preceding and following graph stages is separate work.
