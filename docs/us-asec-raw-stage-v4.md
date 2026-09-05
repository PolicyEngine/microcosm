# ASEC measured-coverage source restoration

Raw checkpoint version 4 restores seven measured at-interview coverage recodes
that frozen 2022 and 2023 source files omitted: `NOW_NONM`, `NOW_MCAID`,
`NOW_CHAMPVA`, `NOW_MIL`, `NOW_VACARE`, `NOW_OTHMT`, and `NOW_IHSFLG`.
`NOW_GRP` and `NOW_MRK` remain native observations. Restoration joins the official
ASEC person members by income year and exact 22-digit `PERIDNUM`; it never imputes
missing coverage. Redundant source identities, when present, must agree. Existing
nonmissing values must match the source, including in years already carrying them.

Call `load_asec_raw_stage_checkpoint_v4` explicitly for v4. The existing
`load_asec_raw_stage_checkpoint` and `ASEC_RAW_STAGE_SCHEMA_VERSION = 3` retain
the historical v3 contract. Neither codec silently accepts the other version.
V4 requires operator-untouched US input, household design weights, all seven
complete 1/2 recodes, exact registered archive/member pins for every pooled year,
and full-member counts and coverage audit receipts. Full-member survey-weighted
audit shares are source diagnostics; they are not calibrated population estimates.

The separate local producer leaves the historical PUF support builder unchanged:

```bash
uv run --no-sync python -m microcosm.build.us_runtime.asec_raw_stage_v4 \
  --input /absolute/path/to/v3.checkpoint.h5 \
  --expected-sha256 <reviewed-input-file-sha256> \
  --coverage 2022=/absolute/path/to/pppub23.csv \
  --coverage 2023=/absolute/path/to/pppub24.csv \
  --coverage 2024=/absolute/path/to/pppub25.csv \
  --output-dir /absolute/path/to/new-v4-bundle
```

Supply exactly the pooled years and authenticated local official CSV members or
ZIP files. The producer performs no download or model calculation. It verifies
input bytes, restores all seven fields, validates the serialized v4 result, and
creates a new directory containing the checkpoint and `restoration.receipt.json`.
The destination must not exist. Input artifacts remain unchanged. Peak memory
includes the loaded v3 frame, restored person table, and serialized validation
frame; profile a small local case before scaling. No resource bound or release
fitness is inferred from the schema version.

The receipt binds input/output file hashes, source pins, restoration implementation
hashes and serialization runtime versions. The inherited `pipeline_sha256` records
the original producer lineage; the separate receipt identifies the added source
restoration. These module hashes and dependency versions are a restoration receipt,
not a complete hermetic environment attestation; orchestration must also pin its
source tree and environment. Structural Frame identity does not bind values: downstream callers
must pin and authenticate the whole output file and its trusted producer receipt.
Registered pins inside arbitrary metadata alone do not prove the values came from
those files.

No country model runs here, so changing a model version cannot authorize reusing
a fitted or evaluated artifact. A source-byte or restoration implementation change
requires a new restoration receipt and downstream identity review; source/model changes must
invalidate their downstream graph dependencies and rerun the applicable checks.
The measured coverage inputs remain observed evidence, never a calibration target
or a model-generated replacement. V4 source validation does not certify a national
or congressional-district release, or repair an excluded historical candidate.
