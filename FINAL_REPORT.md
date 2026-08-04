# Final report: populace #462 register alignment

## Outcome

Completed the split-PR remediation on `loss-contract-alignment`, based on
`origin/main` at `7b6e10b`. The change is now register alignment only: one
shared critical-target register, one shared congressional-district classifier,
two consumers, builder contract-row gating, and behavioral containment of the
publish contract.

The critical-row loss multiplier was removed entirely per
[populace#492](https://github.com/PolicyEngine/populace/issues/492). There is no
constant, CLI option, validation, loss overlay, telemetry, diagnostics/scorer
provenance, or historical replay pin left. `_fiscal_target_loss_weights` is
source-identical to `origin/main`, and its output therefore preserves main's
bit-level behavior for the same registry and family multipliers.

## Sol round-1 findings

1. **Table 1.4 selector parity:** removed the builder-only
   `accepted_name_prefixes=("irs_soi.",)` constraint. The adapter now has
   exactly the shared requirement's substring and suffix selectors. The
   outside-prefix reproduction is builder-rejected.
2. **Congressional-district parity:** added exported, stdlib-only
   `is_congressional_district_target(name, metadata)` and made the publisher
   and builder classifiers thin wrappers. It ORs layout dimension, source-id
   token, geography level, geography scope, truthy CD GEOID, and name token.
   The builder's exact/semantic, Table 1.4, and zero-support paths now see the
   same registry metadata.
3. **Recorded relative-error shape:** a matched row with missing/`None`
   `relative_error` now fails with the publish-contract message instead of
   silently passing after recomputation. Existing non-numeric and stale-value
   checks remain.
4. **Behavioral anti-drift:** the load-bearing test now runs adversarial rows
   through both consumers for exact-name, family+role, Table pattern,
   missing/non-finite values, and a disallowed incumbent escape at the 0.25
   hard stop. A production Ledger compile supplies six separate CD evidence
   rows; builder and publisher exclude identical six-name sets and counts.
   Field comparisons remain as fast checks, and any added conjunctive prefix
   is proven to trip the guard.

The [#490](https://github.com/PolicyEngine/populace/issues/490) medical 0.25
adjudication tolerance and its adjacent comment in `us_critical_targets.py`
remain byte-for-byte unchanged, as required.

## Reproduction receipts

The Table 1.4 prefix reproduction now returns:

```text
SOI Table 1.4 national dollar fit failed: other.table_1_4.all.bad_amount@2024: relative_error=1 exceeds 0.25 for SOI Pub 1304 Table 1.4 national dollar rows (soi_table_1_4_national_dollar_rows); target=100.0, final_estimate=200.0.
```

The missing-relative-error reproduction now returns:

```text
SOI Table 1.4 national dollar fit failed: irs_soi.ty2023.table_1_4.all.adversarial_amount@2024: missing recorded relative_error; the publish contract requires a numeric value.
```

The CD reproduction has the owner-mandated exclusion result:

```text
builder_excluded=True
publisher_excluded=True
builder_failures=[]
```

Calling that row "rejected" would contradict the required OR-union exclusion
semantics. The two malformed critical rows are rejected; the CD row is
symmetrically excluded by both consumers.

## Verification

The requested suite ran with `UV_NO_SYNC=1` to use the already-synced workspace
environment in the network-restricted sandbox:

```text
uv run --package populace-build --extra us --group dev python -m pytest packages/populace-data/tests packages/populace-build/tests/test_us_fiscal_refresh_builder.py packages/populace-build/tests/test_us_state_files_scorer.py -q
264 passed, 3 skipped (267 collected)
```

Additional receipts:

- Complete `test_gates.py`: passed.
- Required multiplier grep: zero Python hits.
- Ruff check: clean on all ten touched Python files.
- Ruff format check: clean on the eight non-exempt touched Python files; the
  two historical experiment files were not reformatted, as instructed.
- `git diff --check`: clean.
- The medical adjudication block compares byte-for-byte equal to pre-fix
  commit `068854d`.
- Pytest emitted non-failing macOS temporary-directory cleanup warnings; no
  test failed.

## Remediation commits

- `5077f95` — start populace#462 Sol remediation progress.
- `c48ba37` — remove the populace#462 loss multiplier per populace#492.
- `afa910a` — fix Sol finding 1 selector parity.
- `89f74f4` — fix Sol finding 2 CD classifier parity.
- `77040fb` — fix Sol finding 3 relative-error shape.
- `bad7145` — fix Sol finding 4 behavioral containment.
- `3c96514` — apply the finding-2 classifier's required Ruff formatting.

Nothing was pushed at the time of this report; the branch was subsequently
pushed and merged as #491 (2026-07-22).

The sandbox rejected writing
`/Users/maxghenis/PolicyEngine/_reviews/sol-491-fix-out.md` with `Operation not
permitted`; the full completion report is therefore committed here and will be
printed to stdout as the requested fallback.
