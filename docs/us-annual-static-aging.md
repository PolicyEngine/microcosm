# US annual static-aging candidates

`microcosm.build.us_annual_static_aging.build_annual_static_aging` creates a local
candidate containing one native single-year H5 for each year from the source year
through the requested end year (2035 by default). It reuses `static_aging` and
`multi_year_dataset`; it does not run the base calibration graph or publish data.

The inputs are an exact base H5 SHA256 and the ID of the base build it comes
from (`--parent-release`), the base build's dataset key (`--base-dataset`), an
exact SSA population CSV SHA256, and the version, full commit, and clean source
checkout of the imported PolicyEngine-US model. The caller supplies the base
build claim; assembly and publication authenticate that build and its base SHA
separately. The numerical environment must contain the reviewed model and
Microcosm packages.

After setting the input paths and reviewed pins, use the CLI in that environment:

```sh
uv run --no-sync python -m microcosm.build.us_annual_static_aging \
  --base-h5 "$BASE_H5" --base-sha256 "$BASE_SHA256" \
  --parent-release "$BASE_BUILD" --base-dataset "$BASE_DATASET" \
  --ssa-csv "$SSA_CSV" --ssa-sha256 "$SSA_SHA256" \
  --model-source "$MODEL_SOURCE" --model-commit "$MODEL_COMMIT" \
  --model-version "$MODEL_VERSION" --output-dir "$NEW_OUTPUT_DIR" \
  --base-year 2024 --end-year 2035
```

`$BASE_DATASET` is the key of the base build's dataset in its release manifest
(`default_datasets.national`): `microcosm_us_2024` for a new national build,
which is the default, or `populace_us_2024` for one of the builds the US switch
moved into folders. The base year's H5 and the family take that key. Each
projected year's H5 is `microcosm_us_<year>`, written as
`microcosm_us_<year>.h5`.

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

- `base`: source dataset, year, base build ID (`parent_release`), path, and
  SHA256.
- `inputs`, `model`, and `runtime`: SSA pin, model commit and source hashes, actual
  implementation hashes, and dependency versions.
- `metadata.dataset_years`: a map from the source dataset to year-keyed annual
  dataset names, including the preserved source year.
- `artifacts`: annual names mapped to relative H5 paths, hashes, years, entity row
  counts, ordered columns, and round-trip receipts.
- Per-year projection receipt paths and hashes, containing the actual factors,
  parameter series, demographic targets and achievements, and calibration fit.

A complete candidate is not a certified release. It is published only as a
build of its own release, after a separate acceptance report passes and the
assembler below checks that the candidate, the report and the base build
belong together. This module supplies no certification override.

## Qualification and publication order

An annual family is published as a build of the release
`microcosm-us-2024-annual` in the US repository, `policyengine/populace-us`.
That release has the role `national` in `releases.json`, but it is never the
repository's default release, so publishing or promoting an annual build never
changes what `microcosm.data.load("us")` loads.

1. Qualify the base build under its existing contract with the intended
   country, Core, policyengine.py and calculator wheels. The base is a
   calibrated build of `microcosm-us-2024-national` that is already published
   in its folder, `releases/microcosm-us-2024-national/<base build>/`; the
   annual build runs the same contract on the base's files again. The
   published builds the switch moved into folders fail the current national
   contract, so in practice the base is a build made since. A source
   enrichment cannot be a base yet: its contract accepts an enrichment build
   only in the national release, so it refuses one in the annual release.
2. Build the candidate from the base build's H5 with the command above, giving
   the base build's ID as `--parent-release` and its dataset key as
   `--base-dataset`.
3. Run independent acceptance checks on the candidate's files. For every
   declared year, the base year included, record schema, source identity,
   year, demographics, input aggregates, and runtime acceptance. The runtime
   checks must use the intended policyengine.py and model, and must give the
   policy formulas the earlier years' annual inputs they read. Write the
   result as a schema-1 `us_annual_projection_acceptance` report with status
   `passed`, bound to the SHA-256 of the candidate's `annual_manifest.json`,
   to the model's commit and source hash, and to the country and Core
   versions.
4. Assemble the build. Download the base build's folder, with its H5, to a
   local directory, then run:

   ```sh
   uv run --no-sync python tools/assemble_us_annual_release_dir.py \
     --base-build-dir "$BASE_BUILD_DIR" --candidate-dir "$NEW_OUTPUT_DIR" \
     --acceptance "$ACCEPTANCE_REPORT" --out "$ASSEMBLY_DIR"
   ```

   The assembler refuses a candidate made from another build, with another
   base dataset key or another base SHA-256, and an acceptance report bound to
   another candidate manifest. It writes
   `$ASSEMBLY_DIR/releases/microcosm-us-2024-annual/`, holding the base
   build's files unchanged except its two manifests, plus the candidate
   manifest (`annual_manifest.json`), the acceptance report
   (`annual_acceptance.json`) and every per-year receipt. The H5 files go to
   `$ASSEMBLY_DIR/artifacts/`: the base-year H5 under the base build's file
   name, and one `microcosm_us_<year>.h5` for each projected year. Every file
   is a bare file name in the build's folder, and nothing pins a revision.
   The build ID is `<base build>-annual-<YYYYMMDDTHHMMSSZ>-<hex8>`, where the
   time is when the assembler ran, in UTC, and `<hex8>` is the first eight
   hex digits of the candidate manifest's SHA-256. The release manifest
   records the year map, the keys of the candidate manifest and the
   acceptance report, and `metadata.annual_projection_base`, which names the
   base build by `repo_id`, `release`, `build`, `path` and `sha256`. The
   assembler checks the result with the release contract, which runs the base
   build's own checks on the carried files and the annual checks on the added
   ones, requires every year from the base year through the last declared
   year so that policy lookbacks have their annual inputs, and moves it into
   place only if it passes. It prints the build, the release directory and the
   two commands to run next.
5. Publish with the printed `publish_command`. Before anything is handed to
   the emitter, the command reads the base build's `release_manifest.json` on
   the Hub, from its folder at `main`, and refuses unless its dataset has the
   base-year H5's file name and SHA-256. Publishing does not make the build
   current. To make it the annual release's current build, run the printed
   `promote_command` after review, or add `--make-current` to the publish
   command. Either changes only the annual release.
6. Certify policyengine.py against the annual build, naming its release,
   `microcosm-us-2024-annual`, and its build. Certification must keep the
   complete year map and each year's exact file and SHA-256. Then release
   policyengine.py and update its consumers.

This route never makes an annual build the default. It augments the
same accepted base population; it does not promote another agent's new graph
candidate. A later graph release can supply a new base only after passing its
own qualification, followed by a fresh annual build and acceptance.

**Loading a year's file.** `microcosm.data.load` and `microcosm.data.download`
do not load annual files. They select a dataset through the registry by country
and year, the US has only a 2024 entry, which reads
`microcosm-us-2024-national`, and they always load a build's
`default_datasets.national` file, which in an annual build is the base year's.
Read a projected year from the build's folder instead: download
`releases/microcosm-us-2024-annual/<build>/microcosm_us_<year>.h5` from
`policyengine/populace-us` at `main`, check its SHA-256 against that folder's
`release_manifest.json`, install a model and Core version that manifest lists
as compatible, and pass the file's path to `USSingleYearDataset`. A build
folder never changes once published, so the path names the same bytes for
good. Once policyengine.py is certified against the annual build, it makes
this selection for its users.
