# Re-pin the UK Chronicle consumer feed

The national and local target surfaces must always use one reviewed Chronicle
consumer artifact. Rebuild the UK bundle and consumer artifact in
`PolicyEngine/arch-data`, then copy the resulting `consumer_facts.jsonl` and
`manifest.json` into `.codex-work/consumer_facts_uk.jsonl` and
`.codex-work/consumer_facts_uk_manifest.json`. Do not commit either
file.

Verify both SHA-256 digests and the manifest's `facts_sha256`, row count, and
schema version. Update `_LEDGER_FACT_FEED_PIN` in
`uk_runtime/local_target_census.py`, the local validation-level pin, and their
tests together. Regenerate the local census with
`uv run --no-sync python tools/census_uk_local_targets.py`.

Regenerate the national and local reference surfaces from the same fact file
with `tools/generate_uk_target_references.py` and
`tools/generate_uk_local_target_references.py`, then rebuild the signed compile
parity receipts with
`tools/build_uk_ledger_compile_parity_signed_differences.py`. The hermetic
regeneration test accepts either the default `.codex-work` files or a
`CHRONICLE_UK_FACTS` override and skips only when neither is present.

The calibration runner refuses a feed whose facts digest differs from the
committed pin. `--allow-unpinned-feed` is an explicit diagnostic override and
is recorded in the run manifest; it is not a re-pin procedure.
