# Property completion routing

`current_property_completion_routing.build_property_completion_routing(qualified, clone_frame)` describes the remaining ordinary-interest and dividend work over the original ACS/ASEC source population. It assigns no amounts, fits no model and changes no Frame. Its `PropertyCompletionRouting` return value is descriptive; constructing it grants no source, complete-parent or release authority.

The inputs are the existing `QualifiedPropertyIncomeSources` description and the complete initial clone Frame. The operation checks original source order, integer identities, source-native agreement, source ages, component knownness, the exact two clone roles and original household DESIGN weights. It preserves the complete input physical descriptions. Actual source custody remains external: a country host must retain and requalify the preparation and source issuers around relevant I/O. Neither the description's type nor a separately supplied DESIGN vector authenticates that custody.

## Routes and retained reasons

Each original person receives one route:

| Route | Meaning |
| --- | --- |
| `unsupported_under15_measurement` | The source age is below15. No child amount or adult extrapolation is supplied. |
| `source_review_required` | A required source amount/receipt or ACS anchor is malformed, outside its published domain, inconsistent or has an invalid adjustment. |
| `carry_known_components` | Both independent ASEC components are known. Joint-fit exclusion does not erase them. |
| `existing_acs_anchor_decomposition` | The adult ACS aggregate anchor is observed and qualifies for the existing decomposition branch. The separate components remain unresolved at this stage. |
| `acs_anchor_completion_review` | An adult ACS anchor remains missing. This operation supplies no fallback. |
| `asec_component_completion_review` | At least one adult ASEC component remains unresolved. Any independently known other component is preserved. |

Under15 is the first route; malformed, positive-outside-universe and contradictory observations remain visible as overlapping reasons. Adult source errors precede generic missingness. Joint-donor exclusions remain separately named reasons and never by themselves turn otherwise known O/D into unknowns. Unknown new source statuses refuse classification instead of silently entering a generic completion branch.

The long component table preserves amount, individual knownness and the exact reporting/literal statuses. `observed` denotes a qualified observed amount; `derived` distinguishes a qualified known-nonreceipt zero; `unresolved` retains missing amounts. An observed ACS aggregate is recorded separately from its unresolved individual components. No component receives a `modeled` origin here; only a later operation consuming actual checked model outputs can assign that origin. Allocation and disclosure flags remain source provenance rather than donor filters.

## Outputs and support

- `person`: one row per original, with source/native identity, source age, original household, independent O/D knownness, separate ACS anchor and exclusive route.
- `components`: two rows per original for the two required component families.
- `reasons`: all overlapping required-source and joint-fit diagnostics.
- `clones`: exact original/native/source identity and clone roles mapped to the initial receiving person IDs.
- `summary`: raw original-person and unique-household counts, household-inherited DESIGN person mass and union-household DESIGN mass for each route/reason. Clones do not multiply these summaries. Zero-weight support remains visible in raw counts.
- `payload`: deterministic JSON with the tables and explicit statements that it neither assigns amounts nor authenticates a source or complete parent. Integer IDs remain integers, including those above2⁵³; unknown amounts serialize as null.

These are support diagnostics, not prevalence estimates or calibrated totals. Overlapping reason masses must not be added as if the reasons were disjoint.

## Integration boundary

`property_completion_artifact_output()` supplies the typed `completion_routing` artifact declaration. This additive slice does not connect a new node or change the accepted38-node host. The next host integration should emit the payload at the existing property source-projection boundary, bind the actual retained source and clone descriptions, and recompute/compare it during requalification and reconstruction. That host change requires its own source/graph identity and observation checks.

This operation does not create a child-income measurement model, fill an unknown adult input, assert that the PUF covers a missing source population or relax the existing completeness gate.

## Opt-in host artifact

`PropertyIncomeOptions(completion_routing=True, ...)` requests a fourth typed
artifact, `completion_routing`, from the existing
`survey_property.source_projection` node. The default remains disabled and
retains the previous canonical option payload. Neither setting changes the
numerical model declarations, source universes, clone attachment or tax split.
An implementation source edit does change executable cache identities.

The private artifact contains the complete original-person, component, reason
and initial-clone tables. The source receipt contains an explicitly allowlisted
aggregate summary and its private artifact digest. The actual graph HTML and
text views expose receipts, so private identifiers and per-person values must
never enter this summary. Aggregate DESIGN support is descriptive; it is not
calibrated representation, a release gate or a disclosure certificate.

Source execution reuses its existing qualification and builds routing once.
Independent host reconstruction checks exact diagnostic bytes and the expected
aggregate receipt on cold execution and required replay. Ordinary branch checks
retain their existing three source-artifact checks and do not recompute routing.
The host retains all declared artifact hashes and canonical options; its checked
parent/source lifetime remains the authority boundary. The pure routing result
and the artifact grant no source or complete-parent authority.

The bounded integration controls exercise qualified invented source descriptions,
option/declaration behavior, tamper refusal and the real public serializers.
Actual 38-node host acceptance is a separate test scope; this feature does not
claim native completion, PUF readiness or a released dataset.

The repository's existing 38-node tax-host fixture now enables completion
routing on both its cold and required calls. Its controls retain full population
and unknown-input checks, verify the private typed artifact and aggregate receipt
on replay, and refuse changes to the retained artifact bytes. The local native
harness predicates can share those fixtures through a separately pinned acceptance
wrapper; they are not a machine-specific dependency of repository tests.
