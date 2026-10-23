# US compatibility recertification

A `compatibility_recertification` release lets a published, certified US
calibration release carry a measured `compatible_model_packages` claim for a
newer policyengine-us without a rebuild. The child republishes byte for byte,
under a new id and tag, every artifact the parent's release manifest declares
(the H5, the calibration weights and every declared release file) and both
parent manifests. A file the parent tag carries without a manifest entry is
not bound by the manifest and is not carried. The child transforms no data
and never recalibrates. What it adds is evidence about the newer engine:

- **a loader replay** under the tested country, Core and calculator wheels;
- **the engine's column changes**: what the newer engine ignores that the
  built-with one read, and what it reads that no engine read before;
- **the drift**: the 18 exact critical rows, the scorable national SOI Table
  1.4 dollar rows and the headline totals, scored on the same H5 under both
  engines;
- **a claim**, declared by a named person after reading the drift.

**The wrapper loader is not attested here.** policyengine.py imports its
bundled country model at `import policyengine`, and certifies the bundle's
data release against the installed country version as it does.
policyengine 6.10.1 under policyengine-us 2.60.4 raises "Data release
manifest is not certified for the runtime model version 2.60.4" before any
loader exists. So the first wrapper that can load this child is one whose
bundle certifies it, which is the downstream policyengine.py certification
step; that step runs the wrapper's own suite on the pinned pair. The receipt
records `wrapper_loader: not_attested` with this reason.

The module is `microcosm.data.compatibility_recertification`; the tool is
`tools/build_us_compatibility_recertification.py`.

## Parent identity

The parent is never a code constant. The child carries the parent's published
`release_manifest.json` and `build_manifest.json` byte for byte
(`parent_release_manifest.json`, `parent_build_manifest.json`) and records
their SHA-256 in `compatibility_recertification.json`. That manifest pins
every parent artifact, so it is the whole lineage. Offline, the validator:

- rebuilds the parent from the carried files and requires the report's
  `parent` block to equal what they describe;
- requires every parent artifact in the child with the parent's digest, at
  the child's revision;
- replays today's release contract on the rebuilt parent (`parent_contract`).

Publication adds the online anchor (`require_parent_publication`). After the
Hub client exists, and before any branch, commit or upload, `publish_release`
checks two things at the parent's tag:

- both parent manifests are byte for byte the carried ones;
- each parent artifact is published exactly where the child will publish it,
  with the pinned digest: each release file under `releases/<parent>/`, each
  root artifact at its root path. It compares an LFS file's SHA-256, or the
  git blob id of the carried bytes.

Offline, which artifacts count as release files rests on where the files sit
locally. This check is what stops a root artifact moved into the release
directory from publishing at the wrong path. Each artifact must sit at exactly
one of the two places: a parent that publishes a file at both its root and its
release-file location is refused as ambiguous. Offline, the validator also
replays the publisher's own upload plan on the child directory, and requires it
to write every carried artifact to its manifest path.

Preflight stays offline. A parent artifact is refused outright when:

- its key is one the child uses for its own evidence (`build_manifest`,
  `compatibility_drift`, ...); or
- its path is not one the publisher writes (anything under `releases/`, or
  not a clean relative path).

The parent must be a US calibration release (`release_type` absent or
`calibration`) with `dataset_role` `national_default`. A recertification of a
recertification or of a source enrichment is refused, so every child is one
hop from the build its calibration evidence describes. Recertify the
calibration release again for a later engine.

### Parents that predate microcosm#1034

A parent published before the stored-input evidence contract carries no
`stored_inputs.json`, so today's contract refuses it for that reason alone.
The recertification measures the same rule itself, against the current
register, under both engines: the baseline score records the built-with
engine's verdict and the loader replay grades the tested engine. Such a parent
passes as `passed_with_stored_input_evidence_replaced`, the replaced failure
lines are recorded verbatim, and certification requires the baseline verdict
to pass. The substitution covers absent evidence only. It applies when the
parent carries no stored-input evidence anywhere and its two manifests agree
on every other `gate_evidence` field (compared as written: `[]` and `{}`
disagree). Any other contract failure refuses the child.

## What the newer engine reads differently

`stored_column_changes` classifies every stored model-named column against
the tested engine and the stored-input register:

- `ignored`: not a variable of the tested engine and not a register entry.
  The stored-input rule refuses exactly these. The parent passed that rule
  under its built-with engine, so that engine read them.
- `read`: a register entry (a column the register says no engine reads) that
  the tested engine defines, so it now reads the column as an input. No
  existing gate sees this direction.

Each change needs an entry in `RECERTIFICATION_COLUMN_CHANGES` with its
direction and reason, or the replay refuses. The report records the changes
with their reasons. Measured against policyengine-us 2.60.4 on the Route A
release `populace-us-2024-0fb05b6-4b57d15a287c-20260930T150401Z`:

| Column | Change | Why |
| --- | --- | --- |
| `first_home_mortgage_interest`, `second_home_mortgage_interest` | ignored | removed by policyengine-us#9605; interest comes from the stored person-level `home_mortgage_interest` |
| `tax_unit_role_input` | read | policyengine-us#9617: heads and spouses follow the stored roles, not age |
| `parent_1_id`, `parent_2_id` | read | policyengine-us#9406: Medicaid MAGI household composition |

`filing_status_input` is a register entry that both engines ignore; both
derive `filing_status`. The drift records the stated-by-derived filing-status
table under each engine.

## Drift

The tool scores the H5 in household batches of the country loader's tables:
2,000 households by default, every group unit nested in one household. Each
engine is built with the SPM selection the parent's post-export scoring
recorded. The weight of every unit is its household's `household_weight`
from the H5, so both engines weigh the same rows identically. The rules:

- **Critical rows** use the release tool's SOI semantics: the signed
  component (positive part, except the identity and loss sources), the
  nonrefundable-CTC rule and the itemizer mask.
- **Population aggregates are refused.** A batch engine that computed one of
  the release tool's population-aggregate formulas refuses
  (`POPULATION_AGGREGATE_VARIABLES`).
- **Medicaid is scored in batches of whole states.** Its two state sums are
  `sum_by_state`, so whole-state batches compute them exactly.

Three score runs feed one drift document:

1. `--variant as_stored` under the built-with engine (the baseline);
2. `--variant as_stored` under the tested engine;
3. `--variant without_newly_read_inputs` under the tested engine, with the
   `read` columns withheld. It is required whenever there are any, and it
   splits each row's move into `newly_read_inputs_effect` (for example, the
   role-driven move in income tax, EITC and CTC) and
   `other_engine_changes_effect`.

The drift document embeds all three, and validation recomputes every row from
them. Every score document must:

- cover all households of the H5, in both its household and its state pass;
- score the carried `calibration_diagnostics.json` (by digest) under the
  parent's post-export SPM selection;
- run on a runtime proven against the three wheels:
  - every installed file is included (parameter YAML, not only Python);
  - the files on disk under each package are exactly the recorded ones, so an
    unrecorded module cannot ride along;
  - each import resolves to the installed package, not to something earlier
    on the path;
  - each wheel's SHA-256 matches the digest PyPI publishes for that version.

The scorer and the loader replay also compile the packages from source into a
fresh, empty bytecode cache (`sys.pycache_prefix`), so a replaced `.pyc` in the
install tree is never run. After the run they check that every loaded module
came from the verified tree and was compiled into that cache.

The tested runs must be exactly the runtime the loader replay certified:
version, content digest and wheel digest of all three packages. The baseline
run must be the parent's built-with policyengine-us and policyengine-core.

## Steps

Every step up to preflight is local. Scoring runs population microsims:
check `python3 ~/reviews/fleet-ops/tools/reservation.py allowed <caller>`
first, and stay out of reserved windows. Run everything from one clean,
reviewed checkout. Publication authenticates the producer files against the
recorded commit.

1. **Fetch the parent** at its tag:

   ```bash
   uv run --no-sync python tools/build_us_compatibility_recertification.py fetch-parent \
     --release-id PARENT_ID --output-dir /path/parent
   ```

2. **Build the candidate.** Root artifacts are hard-linked; compatibility is
   `pending`:

   ```bash
   uv run --no-sync python tools/build_us_compatibility_recertification.py build \
     --parent-release-dir /path/parent/releases/PARENT_ID \
     --parent-artifact-root /path/parent/artifacts \
     --output-dir /path/candidate --release-id CHILD_ID
   ```

3. **Score** in two isolated environments, each with this checkout's
   `microcosm-data` installed editable. One has the parent's built-with
   wheels, the other the three tested wheels:

   ```bash
   BASE/bin/python tools/build_us_compatibility_recertification.py score \
     --release-dir /path/candidate/releases/CHILD_ID \
     --artifact-root /path/candidate/artifacts --output baseline.json
   TESTED/bin/python tools/build_us_compatibility_recertification.py score ... --output tested.json
   TESTED/bin/python tools/build_us_compatibility_recertification.py score ... \
     --variant without_newly_read_inputs --output counterfactual.json
   ```

   Each run also takes the three `--compatibility-wheel` files its environment
   was installed from. Before building any engine, the scorer proves the
   install against them and checks them against PyPI.

4. **Put the drift in front of the declarer.** Whether drift blocks the
   claim is the declarer's call, not this lane's. Queue it with
   `cos add-decision`.

5. **Certify** in the tested environment with the declarer's ruling:

   ```bash
   TESTED/bin/python -m microcosm.data.compatibility_recertification --certify \
     --release-dir /path/candidate/releases/CHILD_ID \
     --artifact-root /path/candidate/artifacts \
     --output-dir /path/certified/releases/CHILD_ID \
     --compatibility-wheel policyengine_us-X.whl --compatibility-wheel policyengine_core-Y.whl \
     --compatibility-wheel policyengine-Z.whl --compatibility-wheel spm_calculator-W.whl \
     --baseline-scores baseline.json --tested-scores tested.json \
     --counterfactual-scores counterfactual.json \
     --compatible-model-specifier 'policyengine-us==X' \
     --compatibility-claim-declared-by 'NAME (cos dNNNN)'
   ```

   The certified manifest keeps every parent entry. It adds one
   `publisher_claim` model entry in one of two bounded shapes:

   - `==T`, the tested version; or
   - `>=L,<U`, where the built-with version < L <= T < U <= the next minor
     release after T.

   Wildcards, `~=`, exclusion lists and open ranges are refused. The
   source-enrichment guards only probe bounds. It also adds the tested Core as an exact `publisher_claim` pin,
   unless a parent entry already covers that Core. `build.built_with_*`
   stays the parent's.

6. **Preflight**, offline, and then **publish** on the owner's go:

   ```bash
   microcosm-publish-release /path/certified/releases/CHILD_ID \
     --repo-id policyengine/populace-us \
     --artifact-root /path/candidate/artifacts \
     --compatibility-wheel ... (three) \
     --no-latest --tag-only --preflight-only
   ```

   Publication is the same command through `tools/publish_release.sh` without
   `--preflight-only`. Whether the child should also become `latest.json` is
   a separate decision; the commands above leave the pointer alone.

7. **Certify the bundle downstream.** policyengine.py certifies the child on
   its publisher claim (`compatibility_basis: legacy_compatible_model_package`,
   with a warning naming the built-with version).

## Limits

- **The regional release.** policyengine.py's US certification also checks
  the regional (ACS local-area) release against the pinned model. On
  2026-10-23 the routed regional release
  `populace-us-2024-buildo-acs-local-766b7c7-20261017T030317Z` was built with
  policyengine-us 2.2.1 and claimed nothing wider. Certifying 2.60.4 then
  needs either `--drop-regional-release` (state and district simulations fall
  back to the national file), a local-area recertification, or a fresh
  local-area build on 2.60.x. Local-area parents are a follow-up: the identity
  mechanism does not depend on the role, but the child must replay the
  local-area contract.
- **One measured version.** The claim's `claim_measurement` names the one
  version the loader replay and the drift measured. A range admits versions
  nothing measured.
- **Not numerical acceptance.** The replay proves loading; the drift reports
  movement; neither is a calibration to the new engine.

## Local checks

```bash
uv run --no-sync pytest \
  packages/microcosm-data/tests/engine_free/shared/test_compatibility_recertification.py \
  packages/microcosm-build/tests/engine_free/us/test_us_compatibility_recertification_parity.py
python3 tools/ci_test_plan.py verify
```

These tests use invented releases and score documents. Passing them does not
certify a real artifact.
