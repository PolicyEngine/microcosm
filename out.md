# Armenia country package final report (#814)

Date: 2026-08-28

Branch: `armenia-country-package-814`

## Outcome

The third greenfield country package is complete as a schema-valid,
engine-free walking skeleton. `am/` consumes an authenticated pre-built
`populace-us` artifact as US donor support, assigns 2022-census-vintage Armenian
marz from public margins, then declares uniform community cloning within marz.
No record is represented as Armenian microdata, no rules engine is required,
and missing facts or runtime bindings fail closed.

The closed package inventory contains the same six typed resources and five
generation-zero JSON resources as Belgium. It declares:

- a two-stage donor-load and target-derived marz source contract;
- the 10-marzes-plus-Yerevan / 71-community geography posture without inventing
  a code list;
- 14 Ledger-only target references covering demography, household structure,
  consumption, labour, earnings, pensions, family benefits, and SNA validation;
- greenfield aggregate, family-fit, coverage, macro-realism, support, audit,
  nonzero/nonnegative, ESS, and weight-ratio gates, with unknown Armenia
  thresholds explicitly non-applicable rather than copied or fabricated;
- a public 2024 `populace_am_{year}.h5` release contract; and
- contract-only compiler identities for the unimplemented donor, marz, and
  community stages, with a deliberate runtime-refusal test.

The accompanying Chronicle handoff has exactly one harvest row per target key.
Architecture notes preserve both Belgium solver lessons, the permanent holdout
doctrine, the donor-provenance warning, the `populace#263/#265` dependencies,
and the unresolved US-dollar-to-AMD amount bridge. A separate TODO records the
issue-supplied Armenian rule facts for a future Axiom-backed `rulespec-am`.

## Verification

All verification ran offline with the repository-pinned NumPy/Torch versions.
The engine-free lanes omitted country engines; the separate engine-only closure
used the exact lock-required PolicyEngine-US package from the local uv cache.

- Focused package/golden/compiler suite:
  `uv run pytest packages/microcosm-build/tests/test_spec_only_country_packages.py packages/microcosm-build/tests/test_country_spec.py packages/microcosm-build/tests/test_spec_engine_country_bundles.py -p no:cacheprovider`
  — **99 passed** in 28.88 seconds.
- Authoritative `shared-spec` CI group:
  `python3 tools/ci_test_groups.py --list shared-spec | xargs uv run pytest -p no:cacheprovider`
  — **1,332 passed, 42 skipped** in 421.33 seconds.
- Requested package-wide scope:
  `uv run pytest packages/microcosm-build`
  — **6,102 passed, 191 skipped, 14 failed** in 1,562.36 seconds.
  All 14 failures were missing-`policyengine-us` imports/metadata in
  `test_us_multispine_pool_tool.py` and
  `test_us_release_head_to_head_scorer.py`, the repository's two explicitly
  engine-only files. No Armenia or shared/spec test failed.
- Engine-only closure using the exact lock-required
  `policyengine-us==1.819.0` from the local uv cache:
  `uv run pytest packages/microcosm-build/tests/test_us_multispine_pool_tool.py packages/microcosm-build/tests/test_us_release_head_to_head_scorer.py -p no:cacheprovider`
  — **202 passed** in 312.62 seconds. This rerun includes every one of the 14
  package-wide failures, so no test remains red; no network access was used.
- `python3 tools/ci_test_groups.py --verify` — **verification=ok** (313 tracked
  test files).
- `ruff check .` — **passed**.
- `git diff --check` — **passed**.

## Deviations from Belgium

1. Armenia v1 reweights a pre-built public US donor artifact instead of loading
   restricted native SILC records. The current `support_spine.json` vocabulary
   only describes raw ASEC pool construction, so declaring one here would be
   false; donor authentication and support coverage live in the source/release
   contracts.
2. The package is engine-free. A future `rulespec-am` may enter only through the
   Frame `RulesEngine` protocol and Axiom adapter; Belgium's active rules leg is
   not copied.
3. Marz is target-assigned in its own stage before community cloning, while the
   typed donor channel honestly remains observed only at country level.
4. There is no incumbent, so parity/export/target-surface gates are absent.
   CEQ and World Bank comparisons remain documentation-only band candidates,
   and unsupplied national-account and weight-health thresholds are explicit
   activation blockers.
5. Armenia's release contract is public, subject to exact ArmStat and donor
   licence verification; the package contains no invented community roster or
   fabricated observed value.

## Top five maintainer questions

1. Which immutable `populace-us` revision/file, hash, licence, and column
   inventory certify the donor input?
2. What reviewed pre-built destination-unit bridge makes US donor amount
   columns comparable with AMD facts without introducing an engine?
3. Which exact Statbank, ILCS, LFS, SRC, pension, benefit, and SNA tables/cells
   define the 2024 Ledger profile and its hierarchies?
4. What ex-ante Armenia fit, macro-realism, ESS, and weight-ratio thresholds and
   shared evaluator bindings should activate the currently deferred gates?
5. What is the authoritative 2022 marz/community roster and distribution, and
   when will the country-neutral clone/coverage contracts land under
   `populace#263/#265`?

No network fetch, push, PR, artifact build, release, or publication was
performed. The repository-root `PROGRESS.md` was not touched.
