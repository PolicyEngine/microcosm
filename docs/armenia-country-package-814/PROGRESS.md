# Armenia country package progress

## State

The complete spec-only Armenia package, Chronicle/rules handoff, changelog, and
regression coverage are committed. All #814-focused and authoritative
shared/spec CI tests pass. The requested package-wide run completed, and every
engine-only failure from its engine-free environment passed on rerun against
the exact lock-required `policyengine-us==1.819.0` found in the local uv cache.

## Done

- Read the repository `AGENTS.md` and `CLAUDE.md` operating rules.
- Confirmed that the branch starts from `origin/main` with no existing `am/`
  package changes.
- Read the complete Belgium spec-only package, the country-spec schema and
  golden tests, both New Zealand plan documents, and the calibration/holdout
  doctrine.
- Moved this journal under `docs/`, following the New Zealand planning-doc
  convention. Country directories are closed spec inventories: in-package
  Markdown would make `am/` fail both its loader and spec-only tests.
- Confirmed that New Zealand has planning documents but no committed package.
  Armenia will derive the populace-US donor posture from that plan and say so.
- Decided to omit `support_spine.json`: its only supported method constructs
  raw current/prior ASEC pools and cannot truthfully describe consuming an
  existing populace-US support artifact.
- Identified the shared compiler additions needed for honest contract-only
  Armenia source and geography kernel IDs. These IDs will compile but remain
  explicitly non-executable.
- Confirmed that survey-measured poverty and tax-benefit quantities remain
  permanent holdouts; the ILCS poverty snapshot and external CEQ/World Bank
  results cannot become calibration targets.
- Drafted the complete closed `am/` resource inventory: six typed schema
  resources plus five generation-zero JSON projections, mirroring Belgium.
- Split donor loading from Armenia marz assignment so the typed spine honestly
  declares country-level observed geography and target-derived marz codes.
- Added contract-only compiler identities for the unimplemented Armenia donor,
  marz, and community stages; no runtime fallback or rules engine is present.
- Confirmed the draft resolves through `load_country_spec("am")` with both
  source stages and the community geography contract.
- Added a one-row-per-reference Chronicle harvest contract, the architecture
  and solver notes, the verified-facts-only future rules boundary, and the
  changelog fragment.
- Added Armenia to the deterministic country golden, shared-core compile proof,
  contract-only registry closure, spec-only discovery checks, and runtime
  refusal tests.
- Passed the three focused country/spec files: **99 passed**.
- Passed the authoritative `shared-spec` CI group: **1,332 passed, 42 skipped**.
- Ran the requested full `packages/microcosm-build` scope: **6,102 passed, 191
  skipped, 14 failed**. Every failure is a missing-`policyengine-us` error in
  `test_us_multispine_pool_tool.py` or
  `test_us_release_head_to_head_scorer.py`; both files are classified
  engine-only.
- Reran both engine-only files against the exact cached lock version, with no
  network access: **202 passed**. This includes and closes all 14 failures from
  the engine-free package-wide run; no test remains red.
- Passed full-repository Ruff, `tools/ci_test_groups.py --verify`, and
  `git diff --check`; the repository-root `PROGRESS.md` remains untouched.

## Next

1. Chronicle must harvest and publish the 14 referenced fact families, source
   artifact pins/licences, and the 2022 marz/community tables.
2. Maintainers must approve the destination-unit amount bridge and Armenia gate
   thresholds before runtime activation.
3. `populace#263` and `populace#265` must land the shared geography and source
   coverage runtime contracts; #814 must decide future `rulespec-am` ownership.
4. Network-capable CI remains the authoritative full engine/wheel certification
   lane; this offline task performed no artifact build or release.
