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
