# Armenia country package final report (#814)

Date: 2026-08-28

Branch: `armenia-country-package-814`

## Outcome

The Armenia package is complete as a schema-valid, spec-only, engine-free
greenfield contract. It is not represented as an executable build: the exact
`populace-us` artifact, Armenian facts, generated cell bindings, community
distribution, gate policy, and shared runtime kernels remain explicit
prerequisites.

The closed `am/` inventory mirrors Belgium's six typed resources and five
generation-zero JSON projections. It declares:

- a two-stage load of a to-be-authenticated public `populace-us` artifact and
  target-derived marz assignment; every record remains a US donor support
  record and is never described as Armenian microdata;
- a 2022-census-vintage community spine constrained to the assigned marz, with
  10 marzes plus Yerevan and 71 consolidated communities documented but no
  invented code roster;
- eight Ledger-only, count/indicator calibration authoring contracts spanning
  demography, household structure, consumption, labour, earnings, pensions,
  and family benefits;
- real-resolver refusal of unexpanded multi-cell tables: Chronicle must generate
  cell-pinned Ledger references and direct/pre-built candidate bindings before
  runtime activation;
- wage/payment, raw-income diagnostic, and national-accounts facts outside the
  solver manifest until their validation role or AMD-compatible pre-built bridge
  is enforceable;
- greenfield aggregate-admin, per-family-fit, target-coverage, macro-realism,
  support, weight-audit, ESS, and ratio gate declarations, plus active
  release-blocking reference-coverage/support/output checks; and
- a public 2024 release contract for `populace_am_{year}.h5`, with ArmStat open
  dissemination stated and exact ArmStat/donor licence text left as a mandatory
  verification item.

`HARVEST.md` has exactly one solver worklist row per live target-reference key,
plus separate deferred validation/amount, geography, source-authentication, and
external-oracle worklists. `NOTES.md` records the Belgium solver lessons,
permanent survey tax-benefit holdouts, the amount/currency boundary,
`populace#263/#265`, and the future Axiom-backed `rulespec-am` boundary.

Package identity after review:

- CountrySpec fingerprint:
  `64f50fa39e68e9ba6c451e3a47a2f2adeaba5a5ccb147cb80297f942de433ca8`
- Typed spec SHA-256:
  `659b6baf5ebbd71fb7786ec4c4d49df565b2bddabeb868a9385ed226c56880f9`

## Verification

All commands ran offline. The package-wide command used the exact lock-required
`policyengine-us==1.819.0` already present in the local uv cache, so the two
engine-only test files ran inside the same aggregate rather than failing for a
missing optional dependency.

Exact requested package-wide command:

```sh
UV_NO_SYNC=1 UV_PROJECT_ENVIRONMENT=/tmp/armenia-uv-env.ME3TCl UV_CACHE_DIR=/tmp/uv-cache-armenia-814 PYTHONPATH=packages/microcosm-build/src:packages/microcosm-calibrate/src:packages/microcosm-frame/src:packages/microcosm-fit/src:packages/microcosm-data/src:/Users/maxghenis/.cache/uv/archive-v0/ewqqbcYNhWejPQ-OfsFxl:/tmp/armenia-no-engine-site.vZ5Kv8:/Users/maxghenis/PolicyEngine/chronicle/.venv/lib/python3.14/site-packages uv run pytest packages/microcosm-build
```

Result: **6,556 passed, 45 skipped, 2,351 warnings, 0 failed** in 3,671.74
seconds (1:01:11). The warnings are existing numerical, pandas copy/fragmentation,
and PolicyEngine-US runtime warnings; none is Armenia-specific.

Additional checks:

- Focused package/golden/compiler suite:
  `uv run pytest packages/microcosm-build/tests/test_spec_only_country_packages.py packages/microcosm-build/tests/test_country_spec.py packages/microcosm-build/tests/test_spec_engine_country_bundles.py -p no:cacheprovider`
  — **101 passed** in 29.82 seconds.
- Authoritative `shared-spec` CI group:
  `python3 tools/ci_test_groups.py --list shared-spec | xargs uv run pytest -p no:cacheprovider`
  — **1,334 passed, 42 skipped, 1 warning** in 418.20 seconds.
- `ruff check .` — **passed**.
- `python3 tools/ci_test_groups.py --verify` — **verification=ok**, 313 test
  files tracked.
- `git diff --check` — **passed**.

## Deviations from Belgium and why

1. Armenia consumes a pre-built public US donor pool instead of native,
   restricted SILC. No `support_spine.json` is present because the current
   vocabulary describes raw ASEC pool construction, not an existing artifact.
2. Armenia is engine-free. No target, gate, or release file requires
   `rulespec-am`; any later rules leg must use Frame's `RulesEngine` protocol
   through the Axiom adapter.
3. Marz is target-assigned before community cloning. The clone factor is the
   compile-safe minimum of one, not Belgium's 20: collision-avoiding fanout must
   wait for the 71-to-11 roster and within-marz support evidence.
4. There is no incumbent, so parity/export/target-surface gates are absent.
   National accounts back a deferred macro-realism band; CEQ and World Bank
   estimates remain documentation-only band candidates until harvested.
5. The live solver manifest contains eight count/indicator series contracts,
   not guessed scalar cells or AMD amount targets. Those series will expand to
   the reviewed cell-level profile after Chronicle harvest; the eventual
   10–16-margin selection is not fabricated in this package.

## Top five maintainer questions

1. Which immutable `populace-us` revision/file, hash, licence, and column
   inventory certify the donor input?
2. Which exact Statbank, ILCS, LFS, SRC, pension, and benefit tables/cells define
   the 2024 profile, and what generated cell-reference/binding artifact owns
   their fanout?
3. Which scale-free indicator or reviewed AMD-compatible pre-built bridge makes
   donor consumption bands and any future amount rows conceptually comparable?
4. What ex-ante Armenia aggregate-fit, family-fit, macro-realism, ESS, and
   weight-ratio thresholds—and which shared evaluators—activate the deferred
   gate declarations?
5. What is the authoritative 2022 marz/community roster and assignment table,
   and when will the shared geography/source-coverage runtimes land under
   `populace#263/#265`?

No network fetch, push, PR, artifact build, release, or publication was
performed. The repository-root `PROGRESS.md` was not touched.
