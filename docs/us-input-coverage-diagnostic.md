# US population input-coverage diagnostic

Select `USInputProfile.NATIVE_NATIONAL_CD` explicitly for the approved native
scope. Its 159 names are the existing 161-name national/CD profile minus
`self_employment_income_last_year` and `previous_year_income_available`.
`employment_income_last_year` was already absent. The report's
`scope_excluded_inputs` records all three canonical prior-year income names,
whether present or absent in the supplied population; it does not remove
physical columns or classify missing data as non-applicable. Other required
names, assigned-block diagnostics and unresolved applicability stay unchanged.
The historical 163-name profile and default 161-name diagnostic retain their
existing behavior and report no native scope exclusions. This explicit name
profile does not resolve the separate native scientific/source gates or alter
historical release requirements.

[`diagnose_us_input_coverage`](../packages/microcosm-build/src/microcosm/build/us_runtime/population_input_coverage.py) takes a `Population`, its `CompiledGraph` and attached `RunManifest`, and returns a typed `PopulationInputCoverage` report. Its default `USInputProfile.NATIONAL_CD` describes 161 national/congressional-district input names; [`required_us_inputs()`](../packages/microcosm-build/src/microcosm/build/us_runtime/input_coverage_profile.py) keeps the historical 163-name default. The national/CD profile omits only engine `block_geoid` and `tract_geoid`, while separately reporting assigned `census_block_geoid`; neither profile adds `employment_income_last_year`. The diagnostic checks complete declared current ownership and replayed storage, then reports actual entity grains, source-channel/clone groups, null and invalid values, typed weights, and masked-writer versus carried rows. Missing grain and applicability remain unresolved. It verifies consistency of supplied graph objects; source ancestry and source requalification remain the host's responsibility. [Twenty-five frozen controls](../experiments/us-input-coverage25-20260910.json) passed, including actual cold/required replay and late-mutation refusals, with 515 physical source/owned checks and zero model/resource inputs. This is scoped evidence, not maintained-CI validation: four eager calibration imports differ from the frozen run and retain their newer maintained versions. The report establishes neither complete input coverage, statistical signal, applicability, native build acceptance nor release eligibility.
