# Country expansion playbook

Expand countries through three parallel evidence lanes, then join them at a
population-build gate. A country registry entry, a compiled rules module, a
calibrated population, and a published replication are different milestones.
Record each milestone separately.

## Start three lanes in parallel

| Lane | Owner and output | Completion receipt |
| --- | --- | --- |
| Official facts and calibration candidates | Chronicle stores source cells, observed facts, periods, geographies, units, universes, uncertainty, and provenance. Microcosm authors the later target bindings. | Immutable fact-package revision, source hashes, cell identities, coverage inventory, and explicit unavailable families. |
| Full Axiom coverage | Axiom maintains the jurisdiction's source and policy coverage frontier, RuleSpec, real-runtime tests, and the root-input/output ABI. | RuleSpec and engine commits, artifact hashes, coverage report, passing runtime fixtures, required inputs, permitted defaults, and formula-owned export exclusions. |
| External score registry | Scorecard records official Budget or fiscal-office scores independently of any replication. Prioritize reforms whose complete scored scope can be represented. | Official publication and table/cell, reform definition, baseline, score components, currency, sign, period, and an explicit replication-readiness assessment. |

Chronicle is facts-only. Legacy Microcosm interfaces still use the name
`Ledger`, but that does not move target profiles or observed values into this
repository. Do not embed observed calibration values in country packages.
Official model projections and Budget reform scores remain validation evidence,
not targets selected to make the model reproduce its comparator.

For each official score, separate legal entitlement from cash timing, accruals,
debt impairment, administration, behavioural assumptions, and interactions. A
reform is fully scoreable only when the whole published quantity is covered or
the registered comparison explicitly selects a separately published component.
Registering the external score does not imply that replication is complete.

## Join at the population-input gate

Choose a native survey, a public donor population, or an explicitly licensed
combination. Before binding a build, record:

- Country, tax/fiscal periods, population universe, and geography vintages.
- Exact source revision, filename, SHA-256, byte size, and licence. No mutable
  `main` or `latest` fallback may substitute for a pinned donor.
- Frame entities, stable identifiers, memberships, links, and typed weights.
  Keep one explicit calibrated weight source and use `Frame.resolve_weights`
  for another entity. Do not persist an ambiguous second weight vector.
- The rules engine's actual ABI: entity names, input dtypes and units, required
  fields, legal/data bridges, allowed padding, output periods, and export
  exclusions. Missing substantive inputs must fail closed.
- Cell-pinned Chronicle references and reviewed candidate-column bindings.
  A table selector is an authoring contract, not an active target; it must not
  silently fan out or choose the first matching fact.
- Separate source-period, currency, geography, family-definition, and
  entitlement-to-payment bridges. Do not infer a joint distribution from
  marginal agreement.

Keep country folders under `packages/microcosm-build/src/microcosm/build/<cc>/`
spec-only. Use `load_country_spec`, the shared spec compiler, and real shared
operators. A contract-only kernel proves that the compiler can express the
country; it is not an executable implementation. `country_stage_plan` must
refuse missing operators rather than synthesize a fallback.

## Build, calibrate, and expose diagnostics

Run the same logical pipeline at reduced scale first, then at release scale:

1. Authenticate sources and construct the Frame. Preserve donor country and
   support-stratum provenance. When transporting a donor, receipt the change
   from source-population mass to destination-population mass.
2. Bind the real rules engine and prove its entity/input contract. Do not
   replace a missing runtime stage with handwritten policy calculations.
3. Compile the target matrix from the pinned facts and reviewed bindings.
   Reconcile totals and components; do not count both as independent evidence.
4. Calibrate household weights, then evaluate support, per-family fit,
   reference coverage, effective sample size, weight concentration, input
   coverage, and formula-owned export gates. Set country thresholds before
   inspecting a candidate. An absent threshold is unfinished work, not a pass.
5. Emit release and build manifests, structured calibration diagnostics,
   source coverage, demographics where supported, and held-out reform
   validation. Persist failures with their evidence and owners.

Fast mode reduces records, epochs, or draws; it must not replace rules, targets,
or gating logic. Compile/refusal tests are appropriate while shared operators
are missing. Label them as scaffold tests, not as a population calibration run.

Recipient counts can constrain an explicitly modelled receipt/take-up layer.
They must not force full-entitlement outputs to match observed expenditure.
Hold out official Budget scores and independent model results from calibration.

## Publish and connect the calibration dashboard

The country release contract declares the public/private repository boundary,
required artifacts, immutable release identity, and staging destination.
Publication remains a separate authorized action after gates pass.

The dashboard consumes a real `latest.json` pointer, the referenced
`release_manifest.json`, and `calibration_diagnostics.json`. Country labels,
geography, units, and source publishers must come from structured metadata,
not filename parsing or US-specific assumptions. The producer and the dashboard
registry must agree on the `release_manifest.json.country` block and supported
capabilities. A registered country with no valid release shows an unavailable
state; it must not display fabricated charts or another country's data.

For NZ, the agreed block is:

```json
{
  "code": "nz",
  "label": "New Zealand",
  "geography_id": null,
  "geography_label": "New Zealand",
  "repository_visibility": "public",
  "capabilities": ["calibration", "targets", "compare"]
}
```

The producer contract uses `policyengine/populace-nz` for releases and
`policyengine/populace-nz-staging` for build telemetry. These identifiers alone
do not create repositories or publish a release.

## Replicate scores and attach results

After the population and rules legs pass their gates, execute baseline and
reform against the same immutable population. Record the engine, RuleSpec or
model version, data build, target snapshot, reform definition, periods,
accounting bridges, output units, and static/behavioural assumptions.

Use the PolicyEngine interface for PolicyEngine replications. That interface
may run a real Axiom backend; a separate legacy country package is not a
prerequisite. Until the interface and input surface are wired, label direct
Axiom executions as Axiom and leave the PolicyEngine replication pending.
Never relabel an unexecuted or handwritten result as a PolicyEngine run.

Attach the result to the existing external Scorecard claim with a run receipt,
the like-for-like amount, residual difference, and known exclusions. Report a
partial entitlement comparison as partial; do not fill missing operating-cost
components with zero. Keep independent official scores immutable when adding
replications.

## Track country milestones

| Milestone | Evidence required before marking complete |
| --- | --- |
| Registered | Country and external-score records exist; missing data is visible. |
| Compile-ready | Typed country package, exact ABI and source references, reproducible spec hash, and refusal tests pass. |
| Build-ready | Required source facts, population fields, shared operators, licences, and country thresholds are bound. |
| Calibrated | A real solve completed and the complete gate report and diagnostics exist. |
| Live | An immutable release and valid pointer load successfully in the deployed dashboard. |
| Replicated | Real engine runs cover the official score's stated scope and Scorecard links their receipts. |

Belgium and NZ share the same Frame/Axiom seam but not their data-access
posture: Belgium's SILC package names a private artifact repository; NZ's first
transport package names a public US donor and public NZ aggregates. Neither
country may borrow another country's thresholds or claim that donor records
were observed locally. See the [NZ plan](nz-take-up-engine-plan.md) and
[NZ target inventory](nz-calibration-targets.md) for the country-specific gates.
