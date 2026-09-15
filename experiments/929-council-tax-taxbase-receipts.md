# microcosm#929 — council tax stock family on the councils' taxbase returns: receipts

Branch `uk-council-tax-taxbase-929`, cut from #906's head `fbeb4045` (region-tier activation) on
2026-09-15; re-targeted to `main` when #906 merges. Plan: `repos/uk-council-tax-family-plan.md`
(local). Every measurement below names the artifact it ran on.

## Part A — re-pin to chronicle `df35af7` (consumer_fact.v3; closes #920)

**Artifact.** Built from `PolicyEngine/chronicle` main `df35af7e7ccf689ad2a5b6e47ce33e99b8c9d3fb`
(tree `7973f020…`, identical to the `arch-data-265` worktree it was built in) with
`build-bundle --suite uk` → `build-consumer-artifact` (14 min): 266,390 rows, every row
`chronicle.consumer_fact.v3` (schema sha `bdb51e2a…`), facts `3e7d5a4f…`, manifest `51aab341…`.
Against the `474a0ae` pin (141,400 rows) the artifact adds chronicle #264's council taxbase
packages: MHCLG CTB 2023/2024/2025 (38,610 rows each: 14 lines × 297 rows × bands A-, A–H +
total), StatsWales CT1 FY2023–FY2026 (8,278 rows, bands A–I, `period_type: fiscal_year`,
integer opening year), CTAXBASE 2023/2024/2025 councils (32 S12 rows per measure per year).
All 244,226 sub-national rows carry `geography.name` (chronicle #266 / #267).

**Microcosm side.** `build.chronicle_epoch` declares `chronicle.consumer_fact.v3` beside v2
(the per-row schema id was never gated, but the identity registry is the authority on what
counts as a Chronicle spelling). `uk/chronicle_feed.json` moves commit, schema version, schema
sha, both digests and the row count. The six-step runbook ran with one ordering correction now
written into `docs/uk-chronicle-feed-repin.md`: the national generator refuses to compile against
vendored resources whose feed identity differs from the pin, so `tools/vendor_uk_ledger_facts.py`
runs before it on a fresh pin.

**Local surface (`tools/generate_uk_local_target_references.py`, 32 s).** 20,430 active,
1,586 `no_fact_for_area`, 513 signed deferred, 1 signed compile error — the #906 counts. Every
one of the 20,430 rows is byte-identical to the committed row; only the membership's feed label
and the register header (description text, the hierarchy providers/categories catalog the
current generator emits) move. The #920 refusal ("`E14001063` has no display label") is gone:
the hierarchy labels every constituency and authority from the fact's `geography.name`.

**National surface (`tools/generate_uk_target_references.py`, 84 s).** 595 active, 7
`no_fact_at_or_before_period`, 7 signed excluded — #906's counts; all 595 references
byte-identical, no value or period moves. One defect surfaced and is fixed in the same commit:
the nine `scotgov.council_tax_stock.*` rows first came back `multi_fact` (3 matches each)
because chronicle spells the vintage into the record-set id without a separator
(`scotgov.ctaxbase2025.chargeable_dwellings.scotland`; MHCLG does the same with `ctb2025`),
which the series-invariant key did not strip, so the 2023, 2024 and 2025 workbooks looked like
three series and latest-not-after refused. `_normalized_record_set_part` now strips a vintage
year glued to a word (`ctaxbase2025` → `ctaxbase`), the way it already reads `fy2025` and
`september2025`; the nine rows resolve `2025-09` again (values unchanged) with
`matched_fact_count_overall` 1 → 3. Unit tests pin the normalization and the three-vintage
resolution.

**Census, vendored resources, validation levels.** `uk_local_target_census.json` restates the
pin (44 lines, identity only, 11 sources). The nine vendored resources regenerate with identity
headers only (rows unchanged). `uk/local_validation_levels.json` `source_feed` restated by hand
(three values).

**Compile-parity receipts (6 min, both surfaces).** Incumbent 2025 national and local registers:
unchanged. Production 2023 register: 337 → 346 compiled, +9 `ledger_only` — the nine Scottish
stock rows now compile at 2023 from the September 2023 CTAXBASE workbook chronicle #264 added.

**Not yet on this branch.** The taxbase facts bind nothing until the family commits below
(no contract target selects `mhclg.council_taxbase.*`, `welshgov.council_tax_dwellings.*` or the
Scottish council record set). The baseline measurement on this re-pinned surface is Part B.
