# Chronicle dual-acceptance (consumer side) — chronicle#143

Branch `chronicle-dual-accept`. Journal for this lane. Root journals are
history, not state (see CLAUDE.md); this one is current only while the branch
is open.

## Goal

Microcosm must accept BOTH ledger-era and chronicle-era Chronicle identities
*before* Chronicle flips emit:

- schema ids: `policyengine_ledger.consumer_artifact.v1` **and**
  `policyengine_chronicle.consumer_artifact.v2`; `ledger.consumer_fact.v1`
  **and** `chronicle.consumer_fact.v2`
- hash domains: `ledger.<x>.v2` **and** `chronicle.<x>.v3` (same canonical
  payload, new domain string)
- env names: `CHRONICLE_*` preferred, legacy honored with a once-per-process
  deprecation warning

Frozen (microcosm#639): nothing on disk or in artifacts renames. Diagnostic
field names (`ledger_aggregate_fact_key`), H5 attrs, `populace_*` ids, fact
keys, goldens and fixtures stay at v1.

## State

- [ ] not started

## Done

(nothing yet)

## Next

1. Read the primary files, run the epoch-sensitivity audit.
2. Land the shared epoch module.
3. Dual-accept the two consumer-artifact loaders.
4. Env dual-read helper.
5. Tests + full verification.
