# US compatibility engineering pilot

Run `tools/probe_us_compatibility_recertification.py` to check native construction and household scoring on at most 32 complete source households. The tool keeps the original source weights and writes a partial engineering receipt. It supplies no population estimate or compatibility certification.

## Inputs and invocation

Prepare a pending ordinary compatibility-recertification candidate as the [recertification runbook](us-compatibility-recertification.md) describes. This pilot accepts only a schema 1 calibration parent and the existing 2024 scoring period. It rejects source-enrichment parents and passed candidates.

Use the task's authenticated US environment and the three original official wheel files for its installed country model, Core and SPM calculator. The ordinary `scoring_runtime` check verifies the installed trees against those wheels and queries PyPI for the matching public wheel digests before the pilot loads the country engine.

From the checkout root, run:

```bash
uv run --no-sync python tools/probe_us_compatibility_recertification.py \
  --release-dir /absolute/candidate/microcosm-us-2024-national \
  --artifact-root /absolute/candidate/artifacts \
  --compatibility-wheel /absolute/wheels/policyengine_us-VERSION-py3-none-any.whl \
  --compatibility-wheel /absolute/wheels/policyengine_core-VERSION-py3-none-any.whl \
  --compatibility-wheel /absolute/wheels/spm_calculator-VERSION-py3-none-any.whl \
  --max-households 32 \
  --output /absolute/evidence/engineering-pilot.json
```

Replace the placeholder versions and paths with the task's actual authenticated artifacts. Create the output directory first, outside the release and artifact directories. The command refuses an existing output, including a symlink. The script creates a fresh bytecode cache before it imports Microcosm when the process lacks one; the existing bytecode checks still govern every run.

Follow the applicable resource reservation and heavy-run rules before launching this command. Use one process, retain the configured numerical-thread limits, and record the native exit, wall time and peak memory beside the receipt. Full membership scans and model construction can still consume substantial memory; the household bound limits retained source rows, not a measured memory ceiling.

## Source preservation

The pilot first runs the ordinary candidate validator, which authenticates the carried parent manifests, release files and root artifacts. It then scans the household IDs and compact person memberships in chunks and selects the first source households in stored row order. This deterministic selection supplies an engineering fixture, not a statistical sample.

The reader rejects duplicate or dangling IDs, incomplete selected households and group units that cross the selection boundary. It retains every selected person's referenced tax, SPM, family and marital unit, with the source row order, column order, values and dtypes. It neither rescales weights nor rewrites the H5. The reader requires all six entity tables, an explicit 2024 period and a supported table codec.

The existing reviewed-column register governs every newly read or ignored model-named column. A known reviewed change can coexist with a raw stored-input refusal. The receipt records both `reviewed_changes_accepted` and the unchanged raw `h5_verdict`, including its `passed`, `refused` and failure fields. An unreviewed change or a direction mismatch stops the pilot. The tool adds no aliases, replacement care costs or register exceptions.

## Native calculation and receipt

The pilot constructs a native single-year dataset, then calls the public `extend_single_year_dataset(..., end_year=2024)` function. That base-year branch makes a one-year container without uprating the source inputs. The pilot checks every selected column before native construction and after scoring.

One genuine native `Microsimulation` uses the parent's SPM configuration. The existing `_BatchFrames` and `_BatchScorer` calculate only metrics whose definitions declare household batching, with their original source weights. The receipt lists all excluded metrics. The guard refuses any known population or state aggregate cache after construction and before and after each metric, including branch caches. This check covers the existing aggregate-variable list; it does not establish universal separability across every model path.

The tool verifies loaded module bytes, isolated bytecode, carried input hashes and selected input fingerprints before writing an exclusive JSON file. Nonfinite totals stop the run. Its receipt uses kind `compatibility_recertification_engineering_pilot`, records `population_estimate: false` and `compatibility_certification: "not_attested"`, and includes aggregate outcomes without raw household IDs or records.

The ordinary score consumer still requires full household and state passes, the complete metric set and its existing runtime and source bindings. It rejects this pilot receipt. Qualification still needs the ordinary baseline, tested and withheld-input scores, drift and claim checks, publisher and consumer checks, and any wrapper or scientific-adapter work. A successful pilot supplies no default-dataset promotion, historical-law qualification or study-adoption decision.

## Tests

`engine_free/us/test_compatibility_recertification_pilot.py` checks selection closure, source preservation, limits, refusal order, reviewed-column accounting and ordinary-score rejection with pure synthetic fixtures. The `engine_workflow/us` module runs one tiny synthetic H5 through the real native constructor and scorer. That workflow uses explicit test-local parent, runtime and bytecode metadata stubs; it authenticates no real release. Both modules use existing registered test categories and add no CI job.
