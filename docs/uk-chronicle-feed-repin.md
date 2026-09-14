# Re-pin the UK Chronicle consumer feed

The national and local target surfaces have independently reviewed Chronicle
artifact pins. The full build reads `uk/national_chronicle_feed.json` for
national targets and the separate local declarations for local targets and
validation. Updating one pin does not authorize changes to the other surface's
membership or values.

Rebuild the complete UK bundle and consumer artifact in
`PolicyEngine/chronicle` at the declared commit. Keep the resulting
`consumer_facts.jsonl` and `manifest.json` together; do not commit either file.

Verify both SHA-256 digests and the manifest's `facts_sha256`, row count, and
schema version. For a national update, update
`uk/national_chronicle_feed.json` and regenerate the national references and
membership with `tools/generate_uk_target_references.py`. Verify the complete
compiled target diff, including targets outside the intended policy area.
The hermetic national regeneration test accepts `CHRONICLE_UK_FACTS`.

For a separately reviewed local update, update `_LEDGER_FACT_FEED_PIN` in
`uk_runtime/local_target_census.py`, the local validation-level pin, and their
tests together. Regenerate the local census with
`uv run --no-sync python tools/census_uk_local_targets.py`.

Regenerate the local reference surface with
`tools/generate_uk_local_target_references.py`, then rebuild the signed compile
parity receipts affected by that update with
`tools/build_uk_ledger_compile_parity_signed_differences.py`. The hermetic
local regeneration test accepts either the default `.codex-work` files or a
`CHRONICLE_UK_LOCAL_FACTS` override and skips only when neither is present.

The canonical `tools/build_uk_full.py` command refuses a feed whose facts or
manifest digest differs from either reviewed declaration. Omitting explicit
CLI hashes uses the committed pins. A country-only target selector retains
those source checks. There is no `--allow-unpinned-feed` bypass in the full
build; changes require the reviewed re-pin procedure above.

After the `ec7169b` re-pin, census household targets use the same Chronicle
compile path as every other bound UK local family. The OA ladder now supplies
geography assignment and diagnostic household dispersion only; it no longer
creates calibration targets or a non-contract reconciliation surface.
