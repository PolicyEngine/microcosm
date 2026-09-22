# US annual static-aging candidates

`microcosm.build.us_annual_static_aging.build_annual_static_aging` creates a local
candidate containing one native single-year H5 for each year from the source year
through the requested end year (2035 by default). It reuses `static_aging` and
`multi_year_dataset`; it does not run the base calibration graph or publish data.

The inputs are an exact base H5 SHA256 and parent release identifier, an exact SSA
population CSV SHA256, and the version, full commit, and clean source checkout of
the imported PolicyEngine-US model. The caller supplies the parent release claim;
release certification must authenticate that release and its base SHA separately.
The numerical environment must contain the reviewed model and Microcosm packages.

After setting the input paths and reviewed pins, use the CLI in that environment:

```sh
uv run --no-sync python -m microcosm.build.us_annual_static_aging \
  --base-h5 "$BASE_H5" --base-sha256 "$BASE_SHA256" \
  --parent-release "$PARENT_RELEASE" \
  --ssa-csv "$SSA_CSV" --ssa-sha256 "$SSA_SHA256" \
  --model-source "$MODEL_SOURCE" --model-commit "$MODEL_COMMIT" \
  --model-version "$MODEL_VERSION" --output-dir "$NEW_OUTPUT_DIR" \
  --base-year 2024 --end-year 2035
```

The builder uses the frame-anchored demographic targets, SSA ages 80–84 pooled at
80 and 85+ pooled at 85, 300 calibration epochs, seed 0, and a maximum weight ratio
of 5. It records all settings in the manifest. Every projected year starts from
the original frame; years are calculated and serialized individually to avoid
retaining the entire budget window in memory.

The source-year H5 is copied byte for byte. Projected files preserve all six
entities, columns, rows, IDs, and unscaled inputs. Household weights and mapped
monetary inputs change according to the existing static-aging operator. Signed
income inputs retain their separate positive and negative scale factors. Each
annual H5 uses root entity tables and an explicit `_time_period` equal to its
year. Logical and native-loader round trips compare all table values. Consumers
must select the exact annual file and year; its inputs are already projected.
The current publication contract requires table-format storage with direct HDF
fields. Fixed-format inputs, including missing nullable booleans that require
that storage format, are rejected before building and need a separate certified
layout contract.

The output directory must be new. A failed attempt retains its partial files and
`build_status.json`; rerunning requires another directory. `annual_manifest.json`
is written last, after all years finish and source/input hashes are rechecked.
It contains:

- `base`: source dataset, year, parent release claim, path, and SHA256.
- `inputs`, `model`, and `runtime`: SSA pin, model commit and source hashes, actual
  implementation hashes, and dependency versions.
- `metadata.dataset_years`: a map from the source dataset to year-keyed annual
  dataset names, including the preserved source year.
- `artifacts`: annual names mapped to relative H5 paths, hashes, years, entity row
  counts, ordered columns, and round-trip receipts.
- Per-year projection receipt paths and hashes, containing the actual factors,
  parameter series, demographic targets and achievements, and calibration fit.

A complete candidate is not a certified release. Release preparation must add
normal immutable repository/revision pins, authenticate the parent, and run
independent demographic, monetary, identity, and runtime acceptance checks.
That process supplies the separate annual projection acceptance report and
publication decision; this module supplies no certification override.

## Qualification and publication order

1. Qualify the base release under its existing contract with the intended
   country, Core, wrapper, and calculator wheels. Source-enrichment bases also
   require the exact original parent H5 and their producer-source evidence.
   Merely downloading an older bundle and rerunning compatibility does not
   update its producer-source pins. If those pins no longer qualify, reproduce
   the base bundle with reviewed producer code and prove preservation before
   adding annual artifacts.
2. Run independent checks on the saved annual files. Record schema, source
   identity, year, demographics, input aggregates, and runtime acceptance for
   every declared year, including the base year. Runtime checks must exercise
   the intended wrapper and model with the earlier annual inputs that policy
   formulas need. Bind the acceptance report to the candidate manifest hash,
   model commit and source hash, and country/Core versions.
3. Add `metadata.dataset_years`, `metadata.annual_projection_manifest`, and
   `metadata.annual_projection_acceptance` to the qualified release manifest.
   The latter two values name artifact keys for the candidate manifest and
   the separate schema-1 `us_annual_projection_acceptance` report. Keep H5
   paths at the repository root; give both reports and every per-year
   projection receipt their exact `releases/<base_release>/filename.json`
   paths. The gate verifies each per-year receipt's hash and years against
   the candidate manifest and requires every year from the base through the
   last declared year, so policy lookbacks retain their annual inputs.
   Pin all artifacts to one
   `<base_release>-annual-<YYYYMMDDTHHMMSSZ>-<hex8>` tag and their exact hashes.
   Complete source-enrichment certification before this step: that operation
   writes base-tag compatibility metadata.
4. Run normal publisher preparation with the artifact root, annual tag, and
   `update_latest=False`. For source enrichment, also supply the original
   parent H5 and compatibility wheels. The annual extension adds checks; the
   publisher still enforces every original base-release gate. Publish with
   `tag_only=True` to preserve the existing main-branch files as well as both
   latest pointers.
5. Certify the wrapper against the explicit annual revision and release
   manifest, then release the wrapper and update consumers. Certification
   must retain the complete year map and exact annual artifact pins.

The annual tag cannot become `latest.json` through this route. It augments the
same accepted base population; it does not promote another agent's new graph
candidate. A later graph release can supply a new base only after passing its
own qualification, followed by a fresh annual build and acceptance.
