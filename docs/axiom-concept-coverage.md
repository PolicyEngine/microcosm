# Axiom input concept-coverage diagnostic

This is a **diagnostic, not a population schema, calibration plan, or
certification gate**. It makes unavailable contracts visible before a population
build tries to use them. It introduces no country-specific Python operator and
does not change Chronicle source facts.

The generic `InputInventoryProvider` protocol is optional; it does not expand the
required `RulesEngine` protocol. `AxiomEngine.input_inventory()` compiles a fresh
program through the real Axiom dense runtime and records root-input names per
mapped entity, canonical request addresses, and accepted aliases. It excludes
derived outputs. A relation-bearing module fails closed because this adapter
does not yet wire related-entity batches. Its scope is the mapped root entities
and all versions in the module, not every Belgian policy or a selected output's
minimal dependency closure.

`entity` records the operational frame table; `engine_entity` records its dense
root. These are not evidence for a source publisher's statistical entity or
universe. Input `dtype`, `unit`, `period`, `definition`, semantic `concept_id`, and
`required` remain null where the runtime does not supply them. In particular,
being a root input does not establish that an input is mandatory in every case.
Some runtime inputs are configuration or externally supplied derived quantities,
not observed microdata: the Belgian pilot includes communal/agglomeration tax
rates, a supplied tax amount, and a tax-base flag. This inventory does not decide
which fields to store, derive, impute, or supply as configuration.
The complete typed-input contract remains
[axiom-rules-engine#62](https://github.com/TheAxiomFoundation/axiom-rules-engine/issues/62).

## Run it

Install the real Axiom Python wrapper and native dense extension into the local
environment, then pass explicit absolute canonical `rulespec-<country>` roots:

```bash
uv run --no-sync python tools/inventory_axiom_concepts.py \
  --module /path/to/rulespec-be/be/statutes/income_tax/individual/pilot_worker_oracle_pipeline.yaml \
  --rulespec-root /path/to/rulespec-be \
  --group-entity household
```

The command prints a closed v1 JSON manifest. It does not open a microdata file,
materialize taxes, infer metadata from spelling, or create semantic matches by
column name. Missing Axiom software is an error, never a fallback evaluator.

## Evidence and dataset assessment

`microcosm.build.concept_coverage.build_concept_coverage` also accepts an optional
Frame, explicit `column_bindings`, and consumer-authored `fact_bindings`.
Its string-only manifest refuses non-string Frame column labels rather than
coercing distinct labels into apparent matches.

- No Frame: `not_supplied`; every input column status is `unassessed`.
- Frame but no explicit binding: still `unassessed`, even if names match.
- Explicit binding plus Frame: `present` or `absent` according to column names.
  Values, distributions, missingness, units, and observed/imputed origin are not
  assessed. The schema digest fingerprints column names, **not dataset content**.
- Extra columns remain permitted. Demography, wealth, predictors, future reforms,
  and presently unencoded concepts need not be current-law inputs to belong in
  the population.

Every binding pins both the Frame entity and the native engine entity, along
with the runtime slot and canonical address, plus the inventory's
engine/module/root fingerprints. A binding cannot move between native root
entities even when files and request addresses are unchanged. Stale pins fail; non-canonical
aliases must be replaced by the catalog's canonical address. Conflicting target
concepts on one runtime slot fail instead of merging.

Every fact binding independently records source concept, fact ID, source artifact
digest/vintage, target concept/legal vintage, both statistical scopes, a declared
transformation, and evidence (URI, digest, locator, claim). The scopes include
statistic, entity, universe, unit, geography, period, stock/flow classification,
stock reference date, and income-year versus assessment-year accounting basis.
Entity and universe definitions require document pins for non-unresolved
assertions. An asserted exact match requires equal complete statistical scope
and an identity transformation. Names do not create bindings.

The builder preserves `asserted_relationship` separately from
`effective_relationship`. Missing target semantics produce effective
`unresolved` with reason `target_semantics_unavailable`, never a proxy invented
from ignorance. Even with future complete metadata, v0 emits unresolved with
`semantic_equivalence_unverified`: it does not fetch documents or adjudicate
legal/statistical equivalence. Authoring an exact/proxy assertion does not change
engine metadata, create a coverage score, or promote a certification flag.

## Provenance and limits

The manifest fingerprints the entry module, every YAML file plus toolchain pin
in each explicit RuleSpec root, actual imported wrapper/native files, adapter,
and diagnostic builder. Relative paths and bytes determine tree fingerprints;
relocating an identical root does not change identity. Compilation uses a fresh
adapter, and changes between before/after fingerprints abort discovery. Keep
the source checkouts quiescent during the run, as the Axiom loader requires.

Per-entity discovery records complete enumeration and its runtime input count,
or `no_derived_program` with a null count. The latter is not successful zero-input
enumeration. Missing discovery, unsupported relations, failed compilation, or
inconsistent counts refuse an artifact. The Belgian pilot enumerates seven
Person inputs; it has no Household program. `blocking_gaps` exposes unavailable
metadata, unassessed dataset columns, and unresolved evidence without claiming
that this module is all of Belgian law.

Inputs sort by canonical address, entity, and slot. The content hash uses
`microcosm-json-v1`: sorted JSON object keys, UTF-8, no NaN, compact separators,
and exclusion of `content_sha256` itself. Runtime versions and platform are
informational context. This diagnostic is unsigned; do not use a corpus release
key to sign it.

These are reproducibility fingerprints, not signed source authenticity or
release certification. They neither establish the native binary's source-build
attestation nor replace the corpus's signature chain. PR CI tests synthetic
contract fixtures; it does not certify a population or publish any artifact.

Belgium is the first inventory example, not the first proven cross-constellation
concordance. The [pilot snapshot](evidence/be-concept-coverage/README.md) records
the exact runtime/source pins and the separate scoped target-inventory audit.
No evidence bindings or dataset are supplied in that snapshot. Linking concepts
exposes missing evidence; it does not manufacture it.
