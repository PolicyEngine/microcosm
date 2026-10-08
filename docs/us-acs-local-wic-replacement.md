# Replace incomplete ACS local-area WIC participation

## Scope and ownership

[Issue #1154](https://github.com/PolicyEngine/microcosm/issues/1154) remains open
until a dataset owner rebuilds and publishes the replacement, `.py` certifies
it, and the temporary assumption is removed. [Issue #1155](https://github.com/PolicyEngine/microcosm/issues/1155)
covers this code repair only. Merging the code does not change existing H5
files or certify a population.

The temporary `.py` change is [PR #562](https://github.com/PolicyEngine/policyengine.py/pull/562).
Its separate removal draft, [PR #563](https://github.com/PolicyEngine/policyengine.py/pull/563),
must remain blocked until the actual replacement is published and certified.
It contains a deliberately failing check against the known incomplete pin.

This work does not publish a dataset, modify the national default/latest
pointer, deploy services, or add environment variables or database changes.

## Generation order

The supported ACS local-area build calls `build_optional_acs_multispine`:

1. Require complete boolean `takes_up_wic_if_eligible` on the donor population.
   This is the participation decision conditional on eligibility, not the
   separate `receives_wic` reported-receipt input. Do not recreate donor draws.
2. Load ACS source records and map their native age, sex, income, and family
   membership. Keep `source_year`, `source_household_id`, and `source_person_id`.
3. Complete the demographic inputs through the existing transfer plan,
   including pregnancy and own children. Missing required inputs remain errors.
4. On ACS records only, run `with_acs_wic_claim_input`, which requires complete
   source identity and delegates to `with_us_wic_claim_input`. Use the existing
   build seed and ACS source year. The existing generator owns category order,
   sourced FNS CY2022 rates, and deterministic source-person hash draws. It
   assigns pregnant, postpartum, infant, child, or no-category probabilities;
   it does not invent a breastfeeding observation or change eligibility rules.
5. Combine ACS records with donors, preserving donor decisions exactly. Require
   complete boolean participation over the combined population. Record the
   generator seed, source year, output column, row count, and rate-source URL
   in the existing orchestration provenance.

The country engine still determines income and demographic eligibility and
benefit amounts. The generator does not assume every eligible ACS person
claims WIC. Reordering records cannot change a person's seeded decision.

## Release validation

The local release tool rejects missing or malformed participation before
sampling/materialization, before consumer export, and before packaging.
`fill_reviewed_nulls` checks WIC before loading the reviewed-null register:
even an explicit register entry cannot fill participation from the engine's
`True` default. Other reviewed-null handling remains unchanged.

Packaging retains the existing stored-input-name checks, including rejection
of obsolete `would_claim_wic`. It additionally checks the actual participation
column in every person row and records the checked row count in the existing
stored-input report, bound to the packaged H5 hash. H5 reads request batches
of 65,536 rows. Table-format files project only the participation column;
fixed-format files require all columns and pandas may unpickle an entire
object block despite the requested row bounds. This is not a claim of a strict
memory bound for historical fixed-format files.

## Dataset-owner steps (not performed by this PR)

1. Choose a qualified donor release with complete current-name participation
   and every other required donor input. Preserve its immutable identity.
   Existing donor qualification, receipt, hours, geography, and model checks
   still apply; an incompatible donor must fail, not receive an implicit fill.
2. Use a new output directory and rebuild ACS staging with the supported
   `tools/build_us_acs_multispine_base.py` entry point. It is deprecated for
   new general multispine builds but still supports this local-area release
   path. Use pinned ACS archives, the PUMA mapping, donor inputs, and explicit
   build configuration. A small smoke build may validate the setup but cannot
   be published as the full replacement.
3. Check `orchestration.provenance.wic_claim` in the new staging summary and
   complete boolean participation in the H5. Compare donor decisions to the
   selected parent; inspect ACS rates by demographic category. Never edit the
   existing published artifact in place.
4. Rebuild materialization, calibration, consumer export, QA, finalization, and
   packaging using `tools/build_us_acs_local_release.py --stage all` and the
   recorded full-scale configuration. Do not reuse old target matrices,
   calibrated weights, consumer evidence, prepared years, or QA reports:
   changing participation can change WIC and dependent outputs. Satisfy every
   existing release requirement, not just the new participation check.
5. Review and publish the new immutable **non-default local-area** release with
   the existing `microcosm-publish-release <release-dir> --no-latest` route.
   Do not change the national latest pointer or default dataset.
6. Supply the actual revision, artifact hash, and regional manifest to `.py`
   PR #563. Use its existing certification process and model-compatibility
   requirements; do not invent replacement metadata or bypass validation.
7. After the compatibility parent merges, rebase and retarget the removal PR
   to `main`, certify the replacement, run its checks, and only then merge it.
   Keep ordinary legacy-name mapping for other datasets. Close #1154 only
   after publication, certification, and removal are all complete.

## Focused tests

Existing CI discovers the new tests through the test-directory registry:

- `engine_free/us/test_us_acs_wic.py`: deterministic generator delegation,
  reordered identities, required source inputs, strict participation validation.
- `engine_free/us/test_us_acs_multispine.py`: stage order, donor preservation,
  early donor validation, and unchanged disabled behavior.
- `engine_free/us/test_us_acs_local_release_tool.py`: no WIC default fills,
  export refusal, packaging refusal, and hash-bound row-count evidence.
- `engine_workflow/us/test_us_acs_wic.py`: H5 round-trip, live country-engine
  consumption, fixed/table validation, and reads over multiple batches.

These small synthetic tests do not rebuild, publish, or certify the real dataset.
