# Chronicle dual-acceptance (consumer side) — chronicle#143

Branch `chronicle-dual-accept`. Journal for this lane. Root journals are
history, not state (see CLAUDE.md); this one is current only while the branch
is open.

> **Corrected 2026-09-02 by the round-2 review pass.** Several identity and
> design claims below were accurate to the first pass and are wrong now. The
> accepted artifact set is three ids, not two; the chronicle-era successor of
> the artifact id is `.v3`; epoch resolution is a declared registry, not
> namespace parsing; and the env half is `LOGBOOK_*`, not `CHRONICLE_*`. Each
> is annotated in place below. See `PROGRESS-chronicle-dual-accept-round2.md`
> and the PR's "Review fixes (round 2)" section for what is actually in the
> branch.

## Goal

Microcosm must accept BOTH ledger-era and chronicle-era Chronicle identities
*before* Chronicle flips emit:

- schema ids: `policyengine_ledger.consumer_artifact.v1` **and**
  `policyengine_chronicle.consumer_artifact.v2`; `ledger.consumer_fact.v1`
  **and** `chronicle.consumer_fact.v2`

  *(Corrected 2026-09-02: wrong at both ends. Chronicle's `main` emits
  `policyengine_ledger.consumer_artifact.v2` today, so that pair rejected
  every artifact Chronicle publishes; and the chronicle-era successor of the
  v2 id is `policyengine_chronicle.consumer_artifact.v3`. The branch now
  accepts all three.)*
- hash domains: `ledger.<x>.v2` **and** `chronicle.<x>.v3` (same canonical
  payload, new domain string)

  *(Corrected 2026-09-02: the version numbers differ by family. The
  source-side domains are ledger `v1` → chronicle `v2`; only the derived
  families are `v2` → `v3`.)*
- env names: `CHRONICLE_*` preferred, legacy honored with a once-per-process
  deprecation warning

  *(Corrected on the PR before merge, and again here: these are the
  **Logbook** store's credentials, so the preferred spellings are
  `LOGBOOK_*`. See the PR review comment of 2026-09-01.)*

Frozen (microcosm#639): nothing on disk or in artifacts renames. Diagnostic
field names (`ledger_aggregate_fact_key`, `ledger_commit`), H5 attrs,
`populace_*` ids, fact keys, goldens and fixtures stay at v1.

**Acceptance widens; nothing narrows.** This is the rule the lane's own first
pass broke and the second pass restored — see "The regression that mattered".

## State

Implementation and tests landed; verification run recorded in the PR body and
in the lane's report. PR #849, open, do not merge.

## Done

- `microcosm/build/chronicle_epoch.py` — the single epoch authority. Epoch
  detection is **structural** (the namespace segment of a key domain), not a
  lookup in a frozen domain list, because chronicle#143 declares the `v3`
  spelling only for the aggregate and semantic fact families. Only identity
  strings the spec names explicitly are pinned as literals.

  *(Corrected 2026-09-02: structural detection reported
  `chronicle.<anything>.vN` as Chronicle-**issued** identity, which is a
  guess dressed as a witness. Resolution is now a declared registry keyed by
  `(namespace, family, version)`; an undeclared spelling in a Chronicle
  namespace is reported as `undeclared`. The claim that chronicle#143
  "declares the `v3` spelling" for two families also overstated the issue,
  which lists `dual-hash window or v3 domains` as open options.)*
- `microcosm/build/chronicle_env.py` — the env dual-read window, one helper,
  one `DeprecationWarning` per process per legacy name.

  *(Renamed before merge to `microcosm/build/logbook_env.py`; there is no
  `chronicle_env.py` on the branch.)*
- `ledger_artifact.py` — the manifest `schema_version` is a membership test
  over both eras; per-row schema ids and fact keys are carried as published;
  `provenance()` records the observed manifest id, `schema_epoch`,
  `fact_key_epochs`, and `fact_schema_versions`.
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

## The regression that mattered

The lane's first pass added a per-row `schema_version` check to
`_load_fact_rows` that rejected any id outside the two chronicle#143 names —
validation `main` never performed, on a field `main` never read. Real feeds do
not honor that set. The pinned US fiscal-refresh feed
`consumer_facts_buildn_v9_4.jsonl`, which the release path loads through this
loader, declares `arch.consumer_fact.v1` on 37,006 of its first 200,000 rows
and `ledger.consumer_fact.v1` on 399. The check failed the build closed on its
own pinned input, and **PR CI could not see it**: the feed is gated data
outside PR CI. Reproduced directly, fixed, and pinned by a test.

The same pass witnessed fact-key epochs from only the four fields targets
resolve a fact by. Published rows carry eleven key-bearing paths; the pinned
feed exercises all of them. A row straddling the cutover — ledger-era
aggregate key, chronicle-era source-release key — was reported as pure
ledger-era. The inventory is now complete and pinned to the captured feed
fixture.

## Audit result

Every site that compares, parses, or mints a Chronicle fact key or schema id
was inventoried. After this branch, no non-test source file hard-codes a
single epoch in a validator. `ledger_targets.py` carries keys opaquely and
contains no epoch literal at all; `congressional_district_vintage.py` and
`us_trade/import_entry_facts.py` mint into Microcosm-owned namespaces from
digest payloads that contain no Chronicle key, so a source row's epoch cannot
move a minted key; the UK runtime is clean; no golden embeds a Chronicle
domain and no H5 attribute name embeds `ledger`. `require_pins` does not exist
in this repo.

## Open question for the Chronicle side

The pinned feeds carry an `arch.*` key namespace alongside `ledger.*` — 37,006
rows against 399 in `consumer_facts_buildn_v9_4.jsonl`, across all eleven
families. Microcosm treats those keys opaquely, so nothing here depends on
what `arch` is, and the epoch module reports it as outside both declared eras.
Whether chronicle#143's cutover is meant to re-epoch `arch.*` rows too is a
question for the Chronicle lane; this branch does not guess.

## Next

Nothing outstanding on this branch. Chronicle's own acceptance half lands in
the parallel lane; the emit flip is a separate, later cutover.

## Correction (2026-09-02): the env half was misnamed

PR #849 review (Fable, main) caught that `POPULACE_LEDGER_URL` /
`_KEY` / `_API_KEY` / `_EXPORT_KEY` are **Logbook** store credentials
(Supabase `logbook` schema, `logbook_writer` / `logbook_exporter` roles —
see `logbook.py`'s docstring and `logbook/README.md`), not Chronicle fact-
store ones. "Ledger" there was the generic build-ledger sense renamed to
Logbook on 2026-08-08 (microcosm#632) specifically to stop colliding with
Chronicle. Naming the preferred spellings `CHRONICLE_*` would have recreated
that exact collision.

The epoch half above (`chronicle_epoch.py`, `ledger_artifact.py`,
`import_entry_facts.py`, `ledger_targets.py`) is unaffected and was approved
as-is. The env half only: `chronicle_env.py` renamed to `logbook_env.py`,
`CHRONICLE_*_ENV`/`chronicle_env`/`chronicle_env_names`/
`describe_chronicle_env`/`reset_chronicle_env_deprecation_warnings` renamed
to `LOGBOOK_*_ENV`/`logbook_env`/`logbook_env_names`/`describe_logbook_env`/
`reset_logbook_env_deprecation_warnings`, and every caller (`logbook.py`,
`tools/logbook.py`, `build/__init__.py` exports, `logbook/README.md`, the
changelog fragment, this file, and the module-path comments in
`firm_generation.py`/`source_coverage.py`) updated to match. The
`POPULACE_LEDGER_*` legacy names and the once-per-process
`DeprecationWarning` behavior are unchanged; the warning text now cites
microcosm#632 instead of chronicle#143, since this window is a Logbook
naming cleanup riding along on this branch, not part of the chronicle#143
epoch migration. `CHRONICLE_*_BANDS` and
`CHRONICLE_US_SOURCE_COVERAGE_CONTRACT_COMMIT` are untouched — those really
do translate Chronicle ids and pin a Chronicle commit.
