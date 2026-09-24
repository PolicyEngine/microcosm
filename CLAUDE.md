# Microcosm — agent guide

Agent-operational notes only. The [README](README.md) covers usage and release
operations; [DESIGN.md](DESIGN.md) is the architectural authority. On any
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
uv run pytest            # engine-free tests; integration tests remain excluded
uv run ruff check .      # lint
```

PR CI (`.github/workflows/test.yml`) has `lint`, `engine-free`, `engine-us`,
`engine-uk`, `integration-uk`, and `wheels` jobs.
`tools/classify_ci_changes.py` classifies the complete changed-path inventory
as shared, US, or UK. Each ordinary behavioral job has only a Python 3.13/3.14
matrix and reports the 25 slowest tests. The engine-free job distributes files
across two pytest workers with `--dist loadfile`; country-engine jobs run
serially so their process-local engine state remains within hosted-runner
memory. The engine-free job
installs no country extra and always runs shared tests plus the affected
countries' engine-free tests. The country
jobs install only their own extra and run only their country directory. Main
pushes run every environment. Native numerical libraries receive one thread per
process. The wheels job builds each wheel once and compares its archive with its
source tree; it does not repeat behavioral tests. The integration job runs the
UK staging smoke test serially on Python 3.13. `ci-ok` requires every selected
ordinary job and the integration job to pass. Ordinary behavioral jobs pass
`--durations=25`, so each job log reports its 25 slowest tests.

Every automated test must run from `.github/workflows/test.yml`. Add new test
jobs to that workflow and include their results in `ci-ok`; do not create a
separate test workflow.

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
- `engine/us/` or `engine/uk/`: imports or executes that country engine.
- `integration/uk/`: runs serially through the `integration-uk` job with the
  explicit `--run-integration` option.

The directory is the execution authority. Do not create `both/`,
`engine/shared/`, another category, or a module-local country-engine
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

`tools/ci_test_groups.py` recursively enumerates and validates this layout.
Run `python3 tools/ci_test_groups.py --verify` after adding, moving, or splitting
tests. `--import-mode=importlib` permits the same basename in more than one
category. Spec identities
(`spec_sha256` pins, seed digests) attest kernel source and locked
RNG-library versions, so they legitimately move when main changes an attested
module or dependency. CI tests the merge ref, so merge main and re-pin rather
than hunting for an environment leak. Editable installs hide packaging breaks;
if you touch packaging, build wheels locally before pushing.

The `integration-uk` job in `.github/workflows/test.yml` runs the
`integration/uk/` directory serially on Python 3.13 for every pull request to
`main` and every push to `main`. Integration tests do not run in the ordinary
UK job; they require the explicit `--run-integration` option. The current test
runs the real spine command against the complete committed synthetic fixture
with seed 42, uses `--smoke --staging-local-only`, and writes only below the
runner's temporary directory. `ci-ok` requires this job to pass. The job has
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
`--promote-line`. See README "Releasing & alerts". Publication also refuses a
release whose build recorded staging telemetry that never reached its repo
(`--allow-missing-staging` overrides); a build that declared `--no-staging`
publishes without the flag. Never publish or promote artifacts as a side
effect of another task. A UK rowwise run's **staged** bundle
(`staged/<run_id>/` in the private repository, written by the build itself)
is inspection evidence, not a release: it never moves `releases/` or
`latest.json` and is not loadable through the certified loader. The build's
default is to upload that bundle (hundreds of megabytes of licensed microdata)
to the private repository; when you run `tools/build_uk_rowwise_candidate.py`
yourself, pass `--staging-local-only` unless the operator asked for a staged
upload.

The US fiscal-refresh builder scores its written H5 in household batches.
Before a release rerun, run the small-H5 guard sweep described in
[the release build rule](docs/us-release-build-rule.md#post-export-scoring).
That fixture check is separate from full-export timing and release
certification.

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
[the source-enrichment runbook](docs/us-native-spm-role-source-enrichment.md).
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
[the reported-receipt runbook](docs/us-reported-receipt-source-enrichment.md).

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
[the qualification note](docs/us-acs-donor-receipt-qualification.md).

The independent US annual static-aging candidate builder lives in
`microcosm.build.us_annual_static_aging`; it consumes a pinned published parent
and writes local annual H5 files without running the base graph or publishing.
See [the annual candidate guide](docs/us-annual-static-aging.md). Its completion
manifest is build evidence, not release certification.
Optional annual release metadata invokes additional artifact, identity, and
acceptance checks within the normal release gates. Annual cuts use one pinned
`<base_release>-annual-<YYYYMMDDTHHMMSSZ>-<hex8>` tag and cannot update latest
pointers. Qualify source-enrichment bases before adding annual metadata; use
the candidate guide's qualification order and tag-only publication route.

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
[`docs/shared-constants.md`](docs/shared-constants.md). Human contributors and
AI assistants must import shared static data from its domain-specific constants
module instead of copying it or reconstructing alternate views in consumers.

## Review this file

Update this guide in the same PR whenever the workspace layout, test
commands, or release flow change. If you find it contradicting the repo,
trust the repo and fix this file.

UK size experiments use `tools/build_uk_rowwise_candidate.py --release-role dense --dataset-households`
with the same pool inputs as the dense candidate. The flag changes exported
support, not clone K. Sizes remain candidate-only until their matched comparison
and promotion scorecard are adjudicated; see
[the size plan](docs/uk-dataset-size-plan-355.md).
