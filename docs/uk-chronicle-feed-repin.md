# Re-pin the UK Chronicle consumer feed

One reviewed Chronicle artifact identity, `uk/chronicle_feed.json`, governs
every UK consumer of the feed: the national references and membership, the
local census, the local references, the local validation levels and the
vendored per-concern fact resources. The target contract
(`uk/uk_population_targets.json`) is likewise one file for both grains. A
re-pin is therefore one reviewed change; the surfaces regenerate from the same
artifact and cannot drift apart silently (the local census and the validation
register restate the pin and are drift-gated against it).

Rebuild the complete UK bundle and consumer artifact in
`PolicyEngine/chronicle` at the declared commit. Keep the resulting
`consumer_facts.jsonl` and `manifest.json` together; do not commit either
file. The untracked default location is `.codex-work/consumer_facts_uk.jsonl`
+ `.codex-work/consumer_facts_uk_manifest.json` (also mirrored as the artifact
directory `.codex-work/uk-artifact/` for the calibration runner); the hermetic
regeneration tests accept a `CHRONICLE_UK_FACTS` override and skip only when
neither is present.

Verify both SHA-256 digests and the manifest's `facts_sha256`, row count, and
schema version, then update `uk/chronicle_feed.json` and regenerate. Vendor
the per-concern fact resources (step 5) before the national generator: the
generator refuses to compile against a vendored resource whose feed identity
differs from the pin, so on a fresh pin step 5 has to run first. The rest go in
this order:

1. the national references and membership with
   `tools/generate_uk_target_references.py` (pass the stable
   `--source-fact-feed` label recorded in the membership);
2. the local census with `uv run --no-sync python tools/census_uk_local_targets.py`
   (it restates the pin from the shared declaration);
3. the local references and membership with
   `tools/generate_uk_local_target_references.py` (same label);
4. the local validation-level register's `source_feed` block
   (`uk/local_validation_levels.json`), which the loader checks against the pin;
5. the vendored fact resources with `tools/vendor_uk_ledger_facts.py`
   (`uk/ledger_fact_vendor_selections.json` names them; each records the feed
   identity it was taken from);
6. the signed compile parity receipts with
   `tools/build_uk_ledger_compile_parity_signed_differences.py --surface all`.

Verify the complete compiled target diff on both surfaces, including targets
outside the intended policy area, and record the value moves in the changelog
fragment.

Two-level (country + region) contract targets fan out over the region tier
(`UK_REGION_TIER` in `microcosm.calibrate.geography_constants`), one reference per area
(microcosm#905); their cells resolve Chronicle's region- and country-stamped
facts, so a re-pin must carry all twelve areas or the national generator
refuses. The cross-grain legs of English constituencies and authorities come
from `region_code_by_area` in `local_area_crosswalk.json`, regenerated from
the sha-pinned ladder with `tools/generate_uk_local_area_crosswalk.py`.

The national calibration runner refuses a feed whose facts or manifest digest
differs from the committed pin. `--allow-unpinned-feed` is an explicit
diagnostic override recorded in the run manifest; it is not a re-pin procedure.

History: the `ec7169b` re-pin (#887/#900) moved census household targets onto
the same Chronicle compile path as every other bound UK local family; the
`c6f9361` re-pin (#890) added the chronicle #254/#255 and #257/#258 transport
and energy packages and unified the national and local pins into this one
declaration; the `474a0ae` re-pin (#904, chronicle #263) moved the rows to
`chronicle.consumer_fact.v2`, which carries the dimension and value labels the
schema-8 target hierarchy completes from (141,400 rows, including chronicle #260's
Universal Credit packages); the `df35af7` re-pin (#929, chronicle #264 and #267)
moved the rows to `chronicle.consumer_fact.v3`, which adds the publisher's
`geography.name` to every fact (the label the hierarchy needs for constituencies
and local authorities, microcosm#920) and brought the council taxbase packages
for England, Wales and Scotland (266,390 rows); the `ec20085` re-pin (#890 PR-S, chronicle #269/#270 via PR #271) brought the domestic energy facts the energy stage levels and prices against (DESNZ Energy Trends domestic consumption, subnational consumption and meter counts, QEP average prices paid, NEED 2024, ONS 04.5 sub-classes) and the census central-heating tables (275,698 rows).
The `c5e5bf8` re-pin (microcosm#725, chronicle #273, on top of `ec20085`) then brought the HMRC CGT
Tables 7, 8 and 9 (asset type, residential property, carried interest), 276,205 rows, with no
compiled value moving on either surface.
