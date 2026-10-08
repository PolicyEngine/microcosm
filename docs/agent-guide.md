# Microcosm agent guide

Agent-operational notes only. The [README](../README.md) covers usage and release
operations; [DESIGN.md](../DESIGN.md) is the architectural authority. On any
conflict, executable configuration (pyproject.toml, `.github/workflows/`,
`tools/`) and DESIGN.md win over this file.

## Layout

uv workspace monorepo. Shards live in `packages/microcosm-<x>/` and import as
the PEP 420 namespace `microcosm.<x>`: `frame`, `fit`, `calibrate`, `build`,
`data`. There is no top-level `microcosm` package directory — never add an
`__init__.py` to the namespace root.

## Commands

```bash
uv sync --all-packages   # set up the whole workspace
uv sync --all-packages --locked --extra us  # US engine environment
uv sync --all-packages --locked --extra uk  # UK engine environment
bash tools/run_engine_free_tests.sh  # complete engine-free test category
uv run pytest <path>     # focused test while developing
uv run ruff check .      # lint
```

PR CI (`.github/workflows/test.yml`) has `lint`, `engine-free`, `engine-us`,
`engine-uk`, `integration-uk`, and `wheels` jobs, plus the
`select-countries` orchestration job.
`tools/ci_test_plan.py` is the only authority for test-directory ownership, CI
job assignment, country ownership, US timing-report categories, and
changed-path country selection. Its `TEST_GROUPS` registry defines every valid
test directory and its job, country, engine dependency, integration status, and
optional timing category. Shell scripts, Python executors, and pytest collection
must query that module; they must not maintain another directory, job, category,
or country mapping.

Each ordinary behavioral job has a Python 3.13/3.14 matrix and reports the 25
slowest tests. The engine-free job installs no country extra, always runs every
engine-free test, installs its locked JavaScript test dependencies, and
distributes files across two pytest workers with `--dist loadfile`.
Changed-file selection only controls the more resource-intensive country jobs.
A documentation-only pull request selects neither country; a US-only or UK-only
pull request selects that country; shared, mixed, unknown, or empty changed-path
sets select both. Main pushes select both without querying the pull-request API.
The UK integration job follows the same UK selection as the ordinary UK engine
job.

The US engine job runs contract and scenario categories in small pytest
processes with at most two processes active at once, then runs each complete
workflow module in its own fresh pytest process with at most two processes
active at once. Process exit releases country-engine state after every workflow
file rather than retaining it across the suite. The UK engine job and UK
integration job remain serial. Shell orchestration lives in versioned scripts
under `tools/`; the workflow invokes those scripts directly. Country jobs
install only their own extra and run only their registered directories. Native
numerical libraries receive one thread per process. The US runner writes JSON
and Markdown timing reports with overall, category, process, file, and
individual-test timings; CI publishes the Markdown report in the job summary
and uploads both files as artifacts. The wheels job builds each wheel once and
compares its archive with its source tree; it does not repeat behavioral tests.
Ordinary behavioral jobs pass `-v --tb=short --maxfail=1 --durations=25`, so each job log
names tests as they run, prints a concise first-failure traceback, and reports
its 25 slowest tests.

Every behavioral or cross-language compatibility assertion must be a pytest
test in a directory registered by `TEST_GROUPS` and must run through that
category's existing workflow job. Supporting programs and data may live under
`tools/`, but they do not receive separate workflow jobs. The test-plan verifier
allows only registered test jobs and the explicitly declared orchestration,
lint, and wheel-building jobs. Add a test category to `TEST_GROUPS` before its
executor; do not create a separate selection system, test workflow, or
single-purpose behavioral test job.

The Orrery parser compatibility assertion lives in
`packages/microcosm-graph/tests/engine_free/shared/test_graph_orrery.py`. The
engine-free runner installs the supported public Orrery range and locked Node
dependency set under `tools/orrery-contract/`; the pytest test generates a
document through Microcosm's public Python API and requires Orrery's public
parser to accept it. It performs no browser rendering.

Workspace tests use an in-memory telemetry emitter by default. The shared
fixture isolates Hugging Face credentials and cache paths and blocks collector
HTTP requests outside loopback. Tests of actual emitter startup must request
`real_local_telemetry` and supply an explicit loopback development collector;
that opt-in retains credential isolation. Other build tests can inspect
`fake_telemetry_emitters` without creating sockets or service processes.

New commits to a PR cancel older unfinished CI runs for that same PR.
Each main-push run has a unique concurrency group, so all main-push runs
remain independent and can finish validating their merged changes.

`load_country_spec("<code>")` loads each packaged country spec once per
process and hands every caller the same immutable object; a `Path` argument is
re-read on every call. A test that patches something the loader itself runs and
then loads a packaged spec by code must call
`country_spec._load_packaged_country_spec.cache_clear()` before and after that
load, or it silently receives the spec an earlier test cached.

**Adding or moving tests.** Every `test_*.py` module must live directly in one
of these directories below its package's `tests/` directory:

- `engine_free/shared/`: does not import a country engine and is country-neutral.
- `engine_free/us/` or `engine_free/uk/`: does not import a country engine but
  tests country-specific code or data.
- `engine_contract/us/`: imports the US engine to validate variable, parameter,
  schema, graph, or package contracts without running a representative household
  calculation or a complete data workflow.
- `engine_scenario/us/`: runs one or more small household or entity-level US
  engine calculations without reading or writing a representative-population H5
  file.
- `engine_workflow/us/`: performs H5 I/O, population-scale adaptation, static
  aging, fiscal refresh, release preparation, scoring, or another complete data
  workflow. Every module in this category runs in a fresh pytest process.
- `engine/uk/`: imports or executes the UK engine.
- `integration/uk/`: runs serially through the `integration-uk` job with the
  explicit `--run-integration` option.

Classify a mixed US module by its most resource-intensive test, or split it when
the tests already have cleanly separable helpers. The directory is the execution
authority. Do not create `both/`, `engine/shared/`, another category, or a
module-local country-engine
`importorskip`, `find_spec`, skip alias, or `requires_*` decorator. Pytest
excludes unavailable engine directories before module import and adds the
registered `requires_us`, `requires_uk`, and `integration` markers from paths.
If one source module contains tests for different environments, split its test
functions between files with the same basename in the appropriate directories.
Move fixtures and helper functions shared by those files to
`test_support/<shard>/`; do not copy their implementations. Use
`test_support.paths.paths_for()` for repository, package, and fixture locations
instead of deriving them from a test module's `__file__`. Keep fixture data in
the package's root `tests/fixtures/` or `tests/golden/` directory.

`tools/ci_test_plan.py` recursively enumerates and validates this layout. Run
`python3 tools/ci_test_plan.py verify` after adding, moving, or splitting tests,
and use `python3 tools/ci_test_plan.py describe` to inspect the generated group,
directory, job, country, engine-dependency, integration, and timing-category
table. `--import-mode=importlib`
permits the same basename in more than one
category. Spec identities
(`spec_sha256` pins, seed digests) attest kernel source and locked
RNG-library versions, so they legitimately move when main changes an attested
module or dependency. CI tests the merge ref, so merge main and re-pin rather
than hunting for an environment leak. Editable installs hide packaging breaks;
if you touch packaging, build wheels locally before pushing.

The `integration-uk` job in `.github/workflows/test.yml` runs the
`integration/uk/` directory serially on Python 3.13 whenever the unified
changed-path selection includes the UK, and on every push to `main`. It skips
documentation-only and US-only pull requests. Integration tests do not run in
the ordinary UK job; they require the explicit `--run-integration` option. The
current test runs the real spine command against the complete committed
synthetic fixture with seed 42, uses `--smoke --staging-local-only`, and writes
only below the runner's temporary directory. The job has
`contents: read`, does not persist checkout credentials, does not reference a
protected GitHub environment, and receives no external writer credential. Fork
pull requests run the synthetic test without secrets. An optional
repository-level `HF_STAGING_READ_TOKEN` permits a separate private-repository
access check. The job invokes `tools/run_integration_tests.sh`, so its shell
logic remains locally executable. Run it locally without the optional
repository access check with:

```bash
HF_STAGING_READ_TOKEN= bash tools/run_integration_tests.sh
```

## The PR-CI / certification boundary

PR CI never receives external writer credentials or touches restricted
microdata. Same-repository runs may receive the optional read-only
`HF_STAGING_READ_TOKEN` in the `integration-uk` job; fork pull requests receive
no secret. Green PR checks mean the code contracts hold — they do **not**
certify data artifacts.
Builds, calibrations, and releases run outside PR CI, need gated Hugging Face
data and credentials, and cannot run from forks. Release publication is a
deliberate human step (`tools/publish_release.sh` →
`microcosm-publish-release`), gated by `tools/preflight_us_release_gates.py`;
reviewed line promotion is a separate deliberate call to the same CLI with
`--promote-line`. See README "Releasing & alerts". Before launching a US
release, dry-run it with `tools/dry_run_us_release_gates.py` (or the release
tool's `--dry-run-gates-report`). It replays the release's own input stages and
grades every waiver register on the staged frame before the solve; see README
"Dry-running the release's registers". Publication also refuses a
release whose build recorded staging telemetry that never reached its repo
(`--allow-missing-staging` overrides); a build that declared `--no-staging`
publishes without the flag. Never publish or promote artifacts as a side
effect of another task. A UK rowwise run's **staged** bundle
(`staged/<run_id>/` in the private repository, written by the build itself)
is inspection evidence, not a release: it never moves `releases/` or
`latest.json` and is not loadable through the certified loader. The build's
default is to upload that bundle (hundreds of megabytes of licensed microdata)
to the private repository; when you run `microcosm-build-uk`
(`tools/build_uk_full.py`, or its stub `tools/build_uk_rowwise_candidate.py`)
yourself, pass `--staging-local-only` unless the operator asked for a staged
upload.

The versioned Route A driver is `tools/route_a/route_a.sh`; see its
[runbook](../tools/route_a/README.md) for configuration, preserved admission
and release gates, and the Modal-base hand-off. Its release stage enables
staging telemetry by default, obtains the HF credential only in a runtime
wrapper, and accepts `ROUTE_A_STAGING=0` as the opt-out. It runs only the
publisher's offline `--preflight-only` check and leaves publication to Max.

The US fiscal-refresh builder scores its written H5 in household batches.
Before a release rerun, run the small-H5 guard sweep described in
[the release build rule](us-release-build-rule.md#post-export-scoring).
That fixture check is separate from full-export timing and release
certification.
To check the post-export stages on a written export without a full rerun,
subsample it and probe it as
[the release build rule](us-release-build-rule.md#probing-an-export-on-a-household-subsample)
describes (`tools/sample_us_export_households.py`,
`tools/probe_us_post_export.py`). The probe's verdicts are diagnostics, not
certification. Its sampler tests must stay off `microcosm.build.us_runtime`:
importing that package builds the policyengine-us tax-benefit system wherever
the engine is installed.

The US native-SPM-role source-enrichment lane is a separate release type:
`tools/build_us_spm_role_enrichment.py` creates a local candidate from the exact
reviewed BuildP parent, preserving original variables and inherited schema-5
calibration evidence. It does not run calibration or relax schema 6 for ordinary
releases. `microcosm.data.source_enrichment` validates candidates and records
actual native-loader compatibility in a separate bundle. The regular publisher
requires `--parent-h5` and the four tested country/Core/wrapper/calculator wheels; `--preflight-only`
runs the same contract and local publisher preparation (file paths, artifact
hashes, revision/tag pins and latest-pointer eligibility), without constructing
a Hub client or publishing. Supplying `--parent-h5` or `--compatibility-wheel`
for a release that is not a source enrichment is an error, including preflight
and evidence-tier requests. See
[the source-enrichment runbook](us-native-spm-role-source-enrichment.md).
Root's canonical-model acceptance and publication authorization remain separate
from this producer-native-input receipt. The same release type has a second
reviewed operation, `add_reported_receipt_inputs`. It packages the donor
receipt qualification's child of the pinned national default
(`populace-us-2024-spm-20260915`) with the qualification receipt as its source
evidence. `tools/build_us_receipt_enrichment_release.py` builds the local
candidate, and `source_enrichment.json`'s `operation` selects the lineage. The
contract replays the shared verifier in `microcosm.data.h5_boolean_append`
against both H5 files. The publisher refuses to point `latest.json` at this
child, so publish it with `--no-latest --tag-only`. See
[the reported-receipt runbook](us-reported-receipt-source-enrichment.md).

Three US release seams refuse a stored column that is lowercase snake_case,
not a variable of the engine the release is certified against, and not in the
reviewed register `microcosm.data.stored_inputs.US_STORED_NON_VARIABLE_COLUMNS`
(microcosm#1026): the fiscal-refresh tool in its batched pre-export gates
(the exact-k ladder lane runs through it), the source-enrichment probe at
certification and on every replay, and the ACS local-area chain's package
stage (`tools/build_us_acs_local_release.py`). A new provenance column needs a
register entry with its reason, bound to its producer in
`test_us_stored_input_register.py`; a renamed engine input needs its live
name. A new US lane that writes a release H5 must run the check where it
records `build.built_with_model_package`.

A US release or release-gate preflight that receives a multispine pool through
`--base-h5` must authenticate its sibling terminal manifest. A current stacked
pool whose terminal battery is red remains fail-closed unless the operator
passes `--allow-gate-failed-base-pool`; that opt-in carries the full red verdict
into `release_manifest.json` for a separate human publication decision. It does
not weaken the exact-k manifest arm or authorize publication by itself.
A sealed deny-list in `microcosm.build.us_runtime.h5_io` overrides this opt-in
for known-excluded publications while preserving their scoring-only diagnostic
path.

`tools/build_us_acs_donor_receipt_qualification.py` is a third local,
non-publishing lane. It takes one of two exact pinned Build P lineage parents
and appends the three reported-receipt inputs current main's ACS transfer
families require (`person.receives_wic`, `spm_unit.receives_snap`,
`spm_unit.receives_tanf`), derived only through the maintained
`us_runtime.cps_carried` producers and the pinned `PAW_TYP` restore. It
replaces `person/table` and `spm_unit/table` with wider record types that keep
every existing field's bytes, adds five attributes per new column, and
rewrites the four pandas column-registration attributes on those two groups;
every other HDF object and attribute is proven exact. It writes a local H5 and
an aggregate receipt and cannot publish, stage or calibrate; its receipt is
build evidence, not certification. Its byte-preservation verifier lives in
`microcosm.data.h5_boolean_append`, so the reported-receipt release contract
can replay it. See
[the qualification note](us-acs-donor-receipt-qualification.md).

The independent US annual static-aging candidate builder lives in
`microcosm.build.us_annual_static_aging`; it consumes a pinned published parent
and writes local annual H5 files without running the base graph or publishing.
See [the annual candidate guide](us-annual-static-aging.md). Its completion
manifest is build evidence, not release certification.
Optional annual release metadata invokes additional artifact, identity, and
acceptance checks within the normal release gates. Annual cuts use one pinned
`<base_release>-annual-<YYYYMMDDTHHMMSSZ>-<hex8>` tag and cannot update latest
pointers. Qualify source-enrichment bases before adding annual metadata; use
the candidate guide's qualification order and tag-only publication route.

## Concept schema and engine mappings

`microcosm.frame.concepts` is the engine-neutral content layer; each adapter's
`concept_mapping()` maps it onto that engine's inputs (see
[the ADR](concept-schema-transport-adr.md)). Builds do not use it yet.
Each PolicyEngine mapping (`adapters/policyengine_{us,uk}_concepts.py`) names
the engine version it was reviewed against, and the engine tests compare it
with the committed coverage golden in
`packages/microcosm-frame/tests/golden/concept-coverage/`. After bumping
policyengine-us or policyengine-uk, review the mapping against the new engine,
update its `engine_version`, and regenerate the golden and the readable report
(`docs/concept-coverage/`) in that engine's environment with
`uv run --no-sync python tools/refresh_concept_coverage.py --engine <engine>`.
Axiom mappings are data (`adapters/axiom_concept_mappings/<country>.json`),
checked against the engine-generated input surfaces in
`packages/microcosm-frame/tests/fixtures/axiom_input_surfaces/` (regenerate with
`tools/refresh_axiom_input_surface.py`; it needs a real Axiom engine build).

## Root journals are history, not state

The root `PROGRESS*.md`, `FINAL_REPORT.md`, `*_COVERAGE_PROGRESS.md`, and
similar files are session-handoff journals: accurate when written, historical
afterward. Do not treat their "State"/"Next" sections as current truth —
check git/GitHub instead. When a branch carrying such a journal merges,
historicize any currency claims in it ("nothing was pushed", "do not merge",
"in progress") in place with a dated note, so the file cannot mislead later
readers. Adjudicated verdicts belong in `experiments/` or the tracking issue,
with the journal pointing to them.

## Shared constants

Before adding a module-local mapping, enumeration, identifier, or display
label, search for an existing definition and follow
[`docs/shared-constants.md`](shared-constants.md). Human contributors and
AI assistants must import shared static data from its domain-specific constants
module instead of copying it or reconstructing alternate views in consumers.

## Review this guide

Update this guide in the same PR whenever the workspace layout, test
commands, or release flow change. If you find it contradicting the repo,
trust the repo and fix this file.

UK size experiments use `microcosm-build-uk --release-role dense --dataset-households`
(`tools/build_uk_full.py`; `tools/build_uk_rowwise_candidate.py` is a stub over it)
with the same pool inputs as the dense candidate. The flag changes exported
support, not clone K. Sizes remain candidate-only until their matched comparison
and promotion scorecard are adjudicated; see
[the size plan](uk-dataset-size-plan-355.md).
