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
field names (`ledger_aggregate_fact_key`, `ledger_commit`), H5 attrs,
`populace_*` ids, fact keys, goldens and fixtures stay at v1.

## State

Implementation and tests landed; verification run recorded in `out.md`.

## Done

- `microcosm/build/chronicle_epoch.py` — the single epoch authority. Epoch
  detection is **structural** (the namespace segment of a key domain), not a
  lookup in a frozen domain list, because chronicle#143 declares the `v3`
  spelling only for the aggregate and semantic fact families. Only identity
  strings the spec names explicitly are pinned as literals.
- `microcosm/build/chronicle_env.py` — the env dual-read window, one helper,
  one `DeprecationWarning` per process per legacy name.
- `ledger_artifact.py` — manifest `schema_version` is a membership test over
  both eras; a per-row `schema_version` is validated when present (Chronicle
  rows have never carried one) and never demanded; `provenance()` records the
  observed id, `schema_epoch`, and `fact_key_epochs`.
- `us_trade/import_entry_facts.py` — emission stays ledger-era (bytes are
  pinned; these rows are minted from Census/CBP bytes, so there is no source
  epoch to inherit), but the declared id is now an argument checked against
  both eras.
- Chronicle-era aliases for the four `LEDGER_*` module constants that only
  look like env vars to a grep.
- Tests: `test_chronicle_epoch.py`, `test_chronicle_env.py`, plus mixed-epoch
  cases in `test_ledger_targets.py`, epoch-independence of the minted
  `microcosm.derived_fact.*` keys in
  `test_us_congressional_district_vintage.py`, and dual-era emission and
  acceptance in `test_us_trade_facts.py`.

## Audit result

An 8-surface / 42-agent adversarial audit of every site that compares,
parses, or mints a Chronicle fact key or schema id found exactly three
hard-coded epoch literals in non-test source — `ledger_artifact.py:35`/`:126`
and `import_entry_facts.py:141`/`:142` — all handled here. `ledger_targets.py`
carries keys opaquely and contains no epoch literal at all;
`congressional_district_vintage.py` mints into Microcosm-owned namespaces that
sit outside both eras; the UK runtime is clean; no golden embeds a Chronicle
domain and no H5 attribute name embeds `ledger`. Every other candidate site
was refuted on verification.

## Next

Nothing outstanding on this branch. Chronicle's own acceptance half lands in
the parallel lane; the emit flip is a separate, later cutover.
