# Native survey release entry

`tools/build_us_fiscal_refresh_release.py` exposes an in-process native entry
beside the [development handoff](us-native-survey-development-handoff.md):

```python
from tools.build_us_fiscal_refresh_release import build_native_survey_release

# enriched_run: the live, issued SurveyEnrichmentRun from the native continuation.
result = build_native_survey_release(
    enriched_run,
    argv=[
        "--out", "/owned/out",
        "--release-id", "populace-us-2024-native-<id>",
        "--ledger-facts", "/inputs/consumer_facts.jsonl",
        "--ledger-facts-sha256", "<64 hex>",
        "--no-staging",
        "--no-target-materialization-cache",
        "--no-target-frame-checkpoint",
    ],
    declaration=projection_declaration,   # NativeSurveyEngineProjectionSpec
    engine=PolicyEngineUSEngine(contract=..., spm={...}),
    consumer_manifest=expected_consumer_identity,
)
```

`main`/`_main` remain the legacy H5/pool entry. The native entry never runs the
legacy source block: base or pool loading, frozen-support selection, value
repairs, donor/QRF stages, take-up and benefit assignment, or the ACS join. It
never downloads, never starts staging telemetry and never writes the legacy
`release_manifest.json`. Its output is a candidate with measured results. It is
**never release-eligible**, and publication stays a separate, human,
preflight-gated step.

## Order of checks

Each step refuses with a `NATIVE_RELEASE_*` code (a
`NativeSurveyReleaseRefusalError`, a `ValueError`) before the next one starts.
Diagnostics carry codes, declared names, counts and digests only.

1. **Owner.** `check_survey_enrichment_run(run)` runs first. A Frame, a
   development checkpoint or its report, a projection, a descriptive checked
   view or a forged `SurveyEnrichmentRun` refuses with `UNISSUED_RUN` before any
   option is parsed or any consumer method, download, legacy stage or writer is
   reached.
2. **Options.** The shared `_parse_args` parses `argv` unchanged. Every option
   the native entry does not consume must keep the parser's own default:
   `--base-h5`, pool and exact-k options, selection sources, warm starts,
   legacy source paths, SSI/QRF/SCF/SIPP/ORG inputs, evidence tiers, every
   `--skip-*`/`--allow-*` switch, `--no-age-targets`, checkpoint and cache
   locations, and staging options all refuse. New legacy options refuse by
   default. `--input-mass-reference-h5` and `--incumbent-diagnostics` refuse
   because this entry does not yet run those comparisons, not because they
   are H5 or JSON. `--release-id` (a safe `populace-us-` name),
   `--ledger-facts-sha256`, `--no-staging`, `--no-target-materialization-cache`
   and `--no-target-frame-checkpoint` are required. The consumed options are the
   output location and release ID, the Ledger feed and its pins, the
   congressional-district crosswalk and gating, the dense/L0 solve settings,
   target-family loss multipliers and the microsimulation batch size.
3. **Output directory.** `<out>/native-releases/<release-id>` must not exist.
4. **Consumer.** `engine` must be exactly `PolicyEngineUSEngine`, with no
   defaults, the declaration's closed export contract and an explicit SPM
   selection equal to the declaration's. The declaration's period must be the
   integer 2024 and it must name a consumer ID dtype. `consumer_manifest` must
   equal the live identity: adapter and constructor source digests, the
   `RECORD` digests and versions of `policyengine-us`, `policyengine-core`,
   `spm-calculator`, `numpy`, `pandas` and `tables`, the export contract, the
   explicit SPM selection and the effective SPM settings. The static part is
   compared before the country system is built. The effective settings come
   from `policyengine_us.spm.spm_config` on the adapter's own system, so a
   partial caller mapping is not enough. A match describes this process's
   consumer. It is not root admission of that runtime, which stays outstanding.
5. **Projection.** `prepare_native_survey_engine_input(run, declaration=...,
   consumer=engine)` selects the declared cells from the live owner and runs the
   consumer's representation check.
6. **Targets.** `_compile_fiscal_release_target_registry` loads the pinned
   Ledger feed and runs the target-parity manifest gate that the legacy entry
   uses.
7. **Input gate.** Every name on the maintained release roster (pool input
   surface, release source columns and historical input profile, without the
   prior-year income family) must be selected and complete. No native profile
   of permissible unknowns has been reviewed yet, so unknown values fail
   closed and nothing is filled. The selected cells must also fit the H5
   codec with complete SPM roles and canonical scope, and carry `state_fips`
   and, when district targets exist, `congressional_district_geoid`. Presence
   is not a source-signal or applicability qualification.
8. **Materialization.** Only now is the output directory created. Targets are
   materialized from the projected Frame with the admitted consumer's dataset,
   simulation and system constructors, a copy of the explicit SPM selection,
   the adapter as formula-ownership metadata, and target caches disabled.
9. **Solve.** `_calibrate_native_input_frame` runs the shared dense or L0/refit
   solve and attaches weights only to the projected input cells.
10. **Fit gates.** `_release_gate_failures` evaluates the solve against the
    compiled surface: unmaterialized or skipped targets, zero support, critical
    target fit, national SOI Table 1.4 dollar fit and loss. Legacy source-stage
    gates are not passed as `None` to look green; their native successors are
    listed as outstanding. `native_calibration_diagnostics.json` is always
    written. A failure writes no H5.
11. **Export.** `write_verified_policyengine_h5_export` writes
    `native_candidate_populace_us_2024.h5` once with the adapter and compares
    the logical readback. Its binding must equal the calibration attachment's.
12. **Final checks.** The consumer identity is derived again, the owner is
    checked again, and the declaration, projection, report and owner population
    are compared after that last owner I/O. Only then is
    `native_release_manifest.json` written.

## Manifest

`native_release_manifest.json` records the owner receipt digest; the
declaration and projection digests; the consumer identity and its digest
(`root_admitted: false`); the input-gate report; Ledger provenance, registry
version, crosswalk digest and parity result; solver options, loss weights and
the default-dataset summary; the calibration specification and export binding
digests; the H5 digest and codec normalizations; the builder source; and
timing. It sets `release_eligible`, `certified` and `publication_authorized`
to `false`, lists `evaluated_gate_groups`, and lists these
`outstanding_qualifications`:

- a positive integration test with a genuine issued owner;
- root admission of the consumer runtime;
- a closed native input profile with reviewed unknowns;
- native source-signal and applicability gates;
- native take-up and benefit-assignment successors;
- owner-bound geography and district-vintage evidence;
- export input-mass and reference parity;
- reform coverage and reform validation on the written file;
- demographics and source-coverage diagnostics;
- the national and district matched-incumbent scorecard;
- publication preflight.

Survey poverty is comparison-only. This entry never uses it as a gate, target,
tuning signal or selection criterion.

## Tests

`packages/microcosm-build/tests/test_us_native_release_entry.py` is
engine-free. It shows that forged owners refuse before every later step; that
each unconsumed option refuses; how consumer admission orders its refusals; the
input gate's codes and name-only diagnostics; that the private composition
passes the admitted constructors, SPM copy and metadata to materialization,
attaches weights only to input cells, refuses on the real fit gate, and writes
only after a verified logical readback; and that the final owner check guards
the manifest. The composition and wiring tests use invented constructors, a
byte writer and a stand-in owner. They do not show that a genuine public native
build succeeds. `test_us_policyengine_h5_readback.py` adds wrong-parent
refusals before the writer. It also documents that the pure comparator cannot
tell apart parents that differ only outside the retained scope. The native
entry therefore passes the projected parent object it retains, binds the owner
and whole-projection digests in the parent reference, and after the final owner
check compares the whole projection stamp and the projection against the
owner's source frame again.
