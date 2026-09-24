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
repairs, donor/QRF stages, take-up and benefit assignment, or the ACS join. Its
own code calls no download path, never starts staging telemetry and never
writes the legacy `release_manifest.json`. Whether the consumer's own runtime
fetches anything is part of that runtime's qualification. The engine-free
wiring test makes socket connects, sends and name lookups fail, but it stubs
materialization and never imports the country engine, so it shows only that
the entry's own wiring opens no connection. Its output is a candidate with
measured results. It is **never release-eligible**, and publication stays a separate,
human, preflight-gated step.

## Order of checks

Each step refuses before the next one starts. The entry's own refusals raise
`NativeSurveyReleaseRefusalError`, a `ValueError`, with a `NATIVE_RELEASE_*`
code. The maintained helpers it reuses keep their own errors: the owner check
(for example `UNISSUED_RUN`), the projection (`NATIVE_ENGINE_PROJECTION_*`),
the calibration attachment (`NATIVE_CALIBRATION_*`) and the H5 comparison
(`H5_*` and retained-export codes) raise `ValueError`s. Ledger loading and the
target-parity gate keep their existing errors; the parity gate raises
`RuntimeError`. Writer and HDF library errors become
`NATIVE_RELEASE_EXPORT_WRITE`, since they can quote cells. No error is followed
by a manifest. Diagnostics carry codes, declared names, counts and digests.
The consumer file and import-origin refusals also name installed file paths
as `RECORD` lists them and loaded module names; none carries a cell value.

Run with exclusive control of the output tree. `native-releases` and the
release directory are created and opened relative to held directory
descriptors, and the final manifest is published through them. `<out>` and any
missing parents are created and opened by path and then checked for their
canonical location, so a concurrently swapped ancestor is refused only after
`<out>` may have been created at the link's target. The final reopen of the
release directory likewise detects an ancestor swap but does not prevent it.
Diagnostics and H5 writers still use ordinary paths and can be redirected by
concurrent directory replacement. Final checks refuse detected path changes,
but do not prevent those earlier writes. Portable `mkdir` followed by `open`
also cannot prove that an uncooperative actor did not replace the new directory
between those calls. The retained inode is the first one opened after `mkdir`
succeeds.

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
   target-family loss multipliers and the microsimulation batch size. The
   solve settings are checked here rather than after materialization: positive
   epochs and learning rate, a finite `--max-weight-ratio` of at least 1 (the
   conserved total cannot be met when every weight is capped below its
   initial value), finite non-negative L2 penalties and L0 share, and a
   non-negative seed.
3. **Output directory.** `<out>/native-releases/<release-id>` must not exist,
   and `<out>/native-releases` must not be a symlink or a file. A platform
   without descriptor-relative file calls (`O_NOFOLLOW`, `dir_fd`) refuses
   with `NATIVE_RELEASE_PLATFORM`. The directory is created later (step 8).
4. **Consumer.** `engine` must be exactly `PolicyEngineUSEngine`, with no
   defaults, the declaration's closed export contract and an explicit SPM
   selection equal to the declaration's. The declaration's period must be the
   integer 2024 and it must name a consumer ID dtype. `consumer_manifest` must
   equal the live identity: adapter and constructor source digests; the
   versions, `RECORD` digests and verified file counts of `policyengine-us`,
   `policyengine-core`, `spm-calculator`, `numpy`, `pandas` and `tables`; the
   export contract; the (empty) input defaults; the explicit SPM selection;
   and the effective SPM settings. Each hashed `RECORD` row is re-hashed on
   disk, so an edited, resized or removed installed file refuses with
   `NATIVE_RELEASE_CONSUMER_FILES`. An editable install refuses with
   `NATIVE_RELEASE_CONSUMER_EDITABLE`, and a legacy install without `RECORD`
   (for example an egg link) with `NATIVE_RELEASE_CONSUMER_RECORD`. Files not
   listed in `RECORD` are not covered. The check fails closed: if a tool
   rewrites installed files after installation (as relocating an environment
   can do to script shebangs), the runtime refuses. Every file-backed module
   already loaded under a listed distribution's import packages must come
   from one of its verified files. A namespace package without `__file__` is
   accepted only when every search location, resolved, ends in the module's
   own name and is either an ancestor of a resolved verified file or a
   directory that a verified `RECORD` row of that distribution names under the
   installation root. The second form covers installs that symlink each file
   into a cache (uv's `--link-mode symlink`); a directory that is neither an
   ancestor of a resolved verified file nor named by a verified row refuses. Its file-backed descendants are checked separately. A
   checkout earlier on `sys.path` or an unverified namespace location refuses
   with `NATIVE_RELEASE_CONSUMER_IMPORT_ORIGIN`. A distribution that is not
   installed is recorded as `null`, and only while none of its import
   packages is loaded. Expected identities must be recorded against this
   schema, including `input_defaults`; none is checked in. The static part is
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
   is not a source-signal or applicability qualification. A
   `--target-family-loss-multiplier` that names no compiled family also
   refuses here, before any output.
8. **Materialization.** Only now is the output directory created, by path
   together with any missing parents. It is opened without following a final
   symlink and must still be canonical; `native-releases` and the release
   directory are created and opened relative to those descriptors. A
   replacement detected at those components (symlink or directory), or an
   opened release directory no longer at its canonical path, refuses with
   `NATIVE_RELEASE_DIRECTORY`; an existing release directory refuses with
   `NATIVE_RELEASE_DIRECTORY_EXISTS` and is left alone. An output that cannot
   hold hard links refuses with `NATIVE_RELEASE_OUTPUT_FILESYSTEM`. Failure
   leaves directory paths in place because cleanup by name could remove
   another actor's replacement. Inspect retained paths before cleaning them or
   choosing a new release ID. Targets are
   materialized from the projected Frame with the admitted consumer's dataset,
   simulation and system constructors, a copy of the explicit SPM selection,
   the adapter as formula-ownership metadata, and target caches disabled.
9. **Solve.** `_calibrate_native_input_frame` runs the shared dense or L0/refit
   solve and attaches weights only to the projected input cells.
10. **Fit gates.** `_release_gate_failures` evaluates the solve against the
    compiled surface: unmaterialized or skipped targets, zero support, critical
    target fit, national SOI Table 1.4 dollar fit and loss. The legacy
    source-stage gate arguments stay at their `None` defaults because their
    inputs come from legacy stages; the manifest lists their native successors
    as outstanding, never as passed. `native_calibration_diagnostics.json` is
    written whenever the solve returns, including when a fit gate fails; a
    failure writes no H5. A refusal inside the solve itself
    (`NATIVE_CALIBRATION_*`) leaves the release directory without
    diagnostics. Any refusal after step 8 keeps the directory, so a rerun
    needs a new release ID or the directory removed.
11. **Export.** `write_verified_policyengine_h5_export` writes
    `native_candidate_populace_us_2024.h5` once with the adapter and compares
    the logical readback. Its binding must equal the calibration attachment's.
12. **Final checks.** The consumer identity, including installed files,
    loaded-module origins and input defaults, is derived again; a refusal
    there becomes `NATIVE_RELEASE_CONSUMER_CHANGED` with the original code as
    its `cause`. The owner is then checked again. After that last owner I/O
    the declaration, projection, report and owner population are compared,
    and so is the consumer's in-memory state: no defaults, the admitted SPM
    selection and export contract, and the same constructor objects. The
    release directory is reopened and must match the inode first opened in
    step 8 (`NATIVE_RELEASE_DIRECTORY_CHANGED` otherwise); the dataset and
    diagnostics bytes are hashed through it. Only then is
    `native_release_manifest.json` written in that directory. It is linked
    into place from a complete, synced temporary file, so an existing
    manifest is never replaced; a stale temporary file refuses with
    `NATIVE_RELEASE_OUTPUT_EXISTS` and other write errors with
    `NATIVE_RELEASE_OUTPUT_WRITE`. Temporary-file cleanup is best-effort after
    both successful and failed writes. A cleanup failure can leave the
    temporary file; a successfully linked manifest remains complete.

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

The engine-free tests in
`packages/microcosm-build/tests/test_us_native_release_entry.py` show that:

- forged owners refuse before every later step, including `main`/`_main`,
  downloads, legacy source stages, writers and staging;
- each unconsumed option and each unusable solve setting refuses at parsing;
- consumer admission orders its refusals, and the installed-file check
  refuses edited, resized, removed and editable installs of an invented
  distribution, an unparseable `RECORD`, and loaded modules that come from a
  checkout, have neither a verified file nor authenticated namespace search
  locations, or belong to a distribution that is not installed; a namespace
  in an install whose files are symlinks into another tree is accepted, while
  mirrored, unrecorded and hash-less directories in that layout still refuse;
- the input gate reports codes and name-only counts, including unknown
  values, and a projection report that does not bind the owner and consumer
  refuses;
- the private composition passes the admitted constructors, SPM copy and
  metadata to materialization, attaches weights only to input cells, refuses
  on the real fit gate, on an unknown loss family, on a mismatched export
  binding and on writer errors, and writes only after a verified logical
  readback;
- the public wiring runs for dense and L0 solves with socket connects, sends
  and lookups failing; a redirected output root, a symlink swapped in while
  the release directory is created, a concurrent creation, an unwritable
  root and an output without hard links each refuse before the solve;
  failed directory setup retains directory paths for inspection; and a failed
  final owner check, replaced dataset or diagnostics bytes, a release directory
  replaced before the manifest, or a consumer whose SPM settings, input defaults, installed files
  or in-memory state changed each leave no manifest, which is never replaced
  once written. Manifest write failures are coded, and temporary-file cleanup
  is best-effort.

The composition and wiring tests use invented constructors, a byte writer and a
stand-in owner. They do not show that a genuine public native build succeeds.
For speed, most tests hash one small installed distribution; one test checks
the whole maintained roster that is installed.

`test_us_policyengine_h5_readback.py` adds wrong-parent refusals before the
writer. It also documents that the pure comparator cannot tell apart parents
that differ only outside the retained scope. The native entry therefore passes
the projected parent object it retains, binds the owner and whole-projection
digests in the parent reference, and after the final owner check compares the
whole projection stamp and the projection against the owner's source frame
again.
