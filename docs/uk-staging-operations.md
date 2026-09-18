# UK staging operations

This document records the operational ownership and lifecycle decisions for
UK Microcosm staging telemetry. It does not authorize production publication.

## Maintainer responsibilities

- PolicyEngine Hugging Face organization administrators review individual
  access requests and administer `policyengine/populace-uk-staging`. The
  administrator performing initial provisioning becomes the recorded primary
  owner; another organization administrator must be recorded as backup before
  remote telemetry is enabled.
- Administrators of `PolicyEngine/microcosm` administer the GitHub `staging`
  environment. The initial required-reviewer candidates are `@anth-volk` and
  `@MaxGhenis`; environment creation must confirm both accounts still have the
  required repository access before applying that configuration.
- Calibration Diagnostics deployment maintainers administer the server-side
  read credential and deploy consumer support before UK remote output is
  enabled.

The current GitHub repository-administrator list was inspected on 2026-09-08.
It included `@anth-volk`, `@MaxGhenis`, `@nikhilwoodruff`, `@nwoodruff-co`,
`@PavelMakarchuk`, `@policyengine-auto`, and `@vahid-ahmadi`. This dated list is
evidence, not a permanent authorization list; provisioning must query current
permissions again.

The GitHub `staging` environment did not exist when inspected on 2026-09-08.
It was then created with required reviewers `@anth-volk` and `@MaxGhenis`,
self-review prevention enabled, and no environment secrets. The general
integration-test workflow does not reference this protected environment, so
these reviewers do not delay its synthetic tests. The environment is audited
or reconciled idempotently with:

```bash
uv run python tools/configure_github_staging_environment.py
uv run python tools/configure_github_staging_environment.py --apply
```

The setup permits no environment secret except `HF_STAGING_READ_TOKEN`. The
general integration-test workflow instead reads that optional name as a
repository-level Actions secret, where it may inspect the private repository's
settings. Fork pull requests do not receive it. The integration build itself
receives no external service writer credential.

## Contract fixture ownership

Microcosm is the canonical source for staging contract fixtures because it is
the producer and validator of the repository files. Canonical fixtures live
under `packages/microcosm-build/tests/fixtures/staging/` and have a checked-in
SHA-256 manifest. Calibration Diagnostics keeps a byte-identical copy and
checks the same manifest digest. A contract change is incomplete until both
repositories accept the new fixture manifest in their respective feature
branches.

Version 1 fixtures describe current US output and remain fixed during the UK
implementation. Version 2 fixtures include successful spine, calibration,
sanitized failure, delivery-failure, and incompatible-version cases.

The implementation separates versioned file serialization from country
configuration. `microcosm.build.staging_storage` owns repository access and
the repeated-upload failure policy used by both version 1 and version 2.
`microcosm.build.staging_cli` adds and validates common command-line options
from a supplied repository configuration. The UK repository identifier and
environment-variable name live in `microcosm.build.uk_runtime.staging`; the
shared modules contain no UK repository or environment defaults. The two
serializers remain separate because version 1 and version 2 intentionally
write different files.

Regenerate and verify the canonical version 2 bytes with:

```bash
uv run python tools/generate_staging_contract_fixtures.py
uv run python tools/generate_staging_contract_fixtures.py --check
```

## Repository provisioning

An authorized Hugging Face organization administrator provisions or reconciles
the repository with:

```bash
uv run python tools/provision_uk_staging_repository.py --apply
uv run python tools/provision_uk_staging_repository.py --verify-access
uv run python tools/provision_uk_staging_repository.py --verify-access --verify-write
```

The procedure creates the dataset repository with private visibility first,
then applies the Hugging Face setting `gated="manual"`, which means each access
request requires individual approval. If the settings update fails, the
recovery operation selects private visibility again and never selects public
visibility. The write probe is optional, uses only
`verification/operator-write-probe.json`, and deletes that probe after a
successful check.

The procedure was run successfully on 2026-09-08 while authenticated as
`anth-volk`, a member of the `policyengine` organization. The resulting
`policyengine/populace-uk-staging` dataset reported `private: true` and
`gated: "manual"`; in concrete terms, the repository is not publicly readable
and individual access requests require explicit approval. Anonymous repository
inspection was refused, authenticated repository inspection succeeded, and the
operator write probe succeeded and was removed. This confirms that the
organization supports the required combination of private visibility and
individual manual approval.

The bootstrap credential was then replaced on 2026-09-08 by the local token
named `microcosm-uk-staging-local-writer`. Hugging Face reported its role as
`fineGrained`; authenticated repository inspection and the temporary write
probe both succeeded, and the probe was removed. The token was configured by
the operator for this dataset only; its secret value was not printed or written
to the worktree. This establishes the scoped local-writer check, but the
complete access-control task still requires a separate authenticated user that
has not been approved to demonstrate refusal. Calibration Diagnostics and the
optional GitHub repository access check each require their own fine-grained
read-only credential.

## Command modes and files

The three UK commands (`tools/build_uk_frs_spine.py`,
`tools/calibrate_uk_national_dataset.py` and
`tools/build_uk_rowwise_candidate.py`) support these staging modes:

- Default: local version 2 files plus best-effort delivery to
  `policyengine/populace-uk-staging`.
- `--staging-local-only`: the same validated files without constructing a
  remote client.
- `--no-staging`: no telemetry files, with a version 2 opt-out object written
  into build evidence.

Common options are `--staging-dir`, `--staging-repo-id`,
`--staging-run-id`, `--staging-candidate-id`,
`--staging-upload-interval-seconds`, and `--staging-read-back`. An empty
repository identifier is invalid in remote mode. Authenticated read-back is
valid only in remote mode.

Version 2 storage uses the fixed repository prefix `runs/`. Individual runs
cannot select another prefix, so consumers can enumerate that subtree without
traversing the rest of the repository.

Each local or remote run contains only files below its own run directory:

```text
runs/<run_id>/run_manifest.json
runs/<run_id>/progress.json
runs/<run_id>/events.ndjson
runs/<run_id>/calibration_progress.json  # calibration runs only
```

Every JSON document and event declares its schema name and version 2. Unknown
schema names or versions are incompatible data. Only reviewed aggregate JSON
artifacts are permitted; population H5 files, NumPy archives, source survey
tables, row-level extracts, archives, credentials, and environment data are
rejected before remote storage is called.

## Staged datasets

Telemetry is not the dataset. The rowwise candidate command (dense K=15 and
exact-count `--dataset-households` runs alike) also stages the bundle its
manifest vouches for, so a run can be inspected by the team without being
published. The two destinations share one run id:

```text
policyengine/populace-uk-staging   runs/<run_id>/...          telemetry (above)
policyengine/populace-uk-private   staged/<run_id>/...        the bundle
```

The bundle is every file `rowwise_candidate_manifest.json` registers under
`outputs` (the H5, `calibration_diagnostics.json`, the gate report, the CSVs
and the registry), the manifest as built, `staged_manifest.json` (inventory
with digests, summary fields and the telemetry run reference) and
`sha256sums.txt`. Nothing else in the run directory is eligible: logs, the
Logbook spool, checkpoints and evaluation trees stay local. The bundle is
verified from disk against the manifest's own digests and written in one
commit; the commit is recorded as the bundle's revision. `releases/` and
`latest.json` are never touched, no tag is created, and the release contract is
not consulted: a staged bundle is not a release and cannot be loaded as one.

Modes follow the staging switch. The default uploads telemetry and the bundle;
`--staging-local-only` keeps both on disk (the sidecars are still written);
`--no-staging` disables both and records the opt-out; `--no-staged-dataset`
runs telemetry alone, with the bundle neither inventoried nor uploaded. The
repository is `--staged-dataset-repo-id` (environment
`POPULACE_UK_STAGED_DATASET_REPO_ID`). Because the upload closes a multi-hour
run, the command refuses to start a remote dataset stage without an ambient
Hub credential (`HF_TOKEN`) that can see the repository **and** write it: a
read token, or a fine-grained token scoped to another owner, is refused up
front rather than by the Hub's 403 hours later (a fine-grained token needs
`repo.write` on the repository or on the `policyengine` organisation). The
telemetry stays best-effort with no pre-flight. Forwarded epochs are thinned to
at most 2,400 rows per run, whatever `--epochs` says, so the telemetry files
stay under the contract's 5 MiB cap; a content refusal from the telemetry is
reported once and stops the forwarding without touching the solve.

The manifest gains two evidence blocks after the bundle is on disk:
`staging_delivery`, the validated version 2 telemetry receipt, and
`staged_dataset` (`mode`, `repository`, `prefix`, `run_id`, `revision`,
`status` in `uploaded`, `already_staged`, `failed`, `skipped`, a reviewed
`error_code`, and the `files` inventory as a mapping). A failed upload is
recorded with its code, warned on stderr, and never changes the build's exit
code or Logbook disposition. The telemetry run declares two reviewed
artifacts, `artifacts/staged_dataset.json` (the same block) and
`artifacts/fit_summary.json` (loss, fit by family, gate verdicts, the size
receipt without its per-row arrays), so a run in the dashboard points at its
bundle.

Re-stage a finished directory, including runs built before this lane existed
or whose upload failed, with:

```bash
uv run python tools/stage_uk_rowwise_candidate.py --run-dir <run directory>
```

It is idempotent on the outputs' digests: a directory whose manifest already
records the same outputs as uploaded is left untouched (the driver's revision
stands); an identical remote bundle the manifest does not know about is
recorded as `already_staged` with the bundle's own commit; a different bundle
under the same run id is refused (`REMOTE_DIFFERS`). Fetch a bundle for a scorecard, an evaluation leg or the
dashboard's local-directory mode with:

```bash
uv run python tools/fetch_uk_staged_dataset.py --run-id <run_id> --dest <dir>
```

Every file is checked against `sha256sums.txt`; `--h5-only` fetches the
dataset alone. Rollback is configuration-first here too: `--no-staged-dataset`
or `--staging-local-only` for new runs, and deleting `staged/<run_id>/` on the
Hub for a bundle that must not remain (its manifest keeps the record). The
dense release assembler applies the national rule to this command's runs: it
requires the manifest's `staging_delivery` receipt and copies it into
`build_manifest.json` as `staging`, where publication reads it. A dense run
built before this lane carries no receipt; `--allow-missing-staging` assembles
it with a recorded disabled-staging opt-out naming the override, the same
posture publication's `--allow-missing-staging` grants.

## Smoke verification

The UK spine command keeps fractional input sampling for scale tests. The
`--smoke` option marks its H5, sidecar, and staging records as non-release. It
does not invoke national calibration, release certification, release assembly,
or publication.

Continuous integration runs all current UK spine transformations against the
complete deterministic synthetic fixture. It does not reduce the fixture by a
source-family count. Every output stays below the runner's temporary directory,
and the workflow receives no external writer credential.

Run the integration command locally with:

```bash
uv run python tools/build_uk_frs_spine.py \
  --synthetic-fixture-dir packages/microcosm-graph/tests/fixtures/parity/uk_spine/sources \
  --spine-h5 <temporary-directory>/uk-smoke.h5 \
  --sample-fraction 1.0 \
  --sample-seed 42 \
  --smoke \
  --staging-local-only \
  --staging-dir <temporary-directory>/staging \
  --staging-run-id ci-uk-smoke-full-s42
```

The workflow `.github/workflows/integration-tests.yml` runs on manual dispatch
and every pull request to `main`, without a path filter. Its commands live in
`tools/run_integration_tests.sh`. The test reports total elapsed time and the
elapsed time for each transformation.

An authorized operator can verify the current remote layout with the same
synthetic fixture by omitting `--staging-local-only`, adding
`--staging-read-back`, and supplying unique `--staging-run-id` and
`--staging-candidate-id` values. The command still requires `--smoke`, so its
H5 file, sidecar, and aggregate staging records are marked non-release. The H5
file and fixture source tables remain local because the staging content policy
permits only the reviewed aggregate JSON files.

An authenticated staging transport check completed on 2026-09-09 using the
earlier source-family-count interface and the superseded shared-file layout. It
uploaded version 2 JSON records for `uk-smoke-h0100-s578-20260909T125305Z` to
`policyengine/populace-uk-staging`; it did not upload the H5 dataset. Its
authenticated read-back does not verify the current run-scoped-only layout.
That obsolete sampling option has also been removed because it selected source
families before construction and therefore did not guarantee a requested final
household count.

A current-layout authenticated check completed on 2026-09-14 using the full
deterministic synthetic fixture. It recorded
`uk-smoke-full-s42-20260914T172919Z` under its own `runs/` directory, completed
15 of 15 upload attempts, and passed authenticated read-back. Repository
inspection found exactly `run_manifest.json`, `progress.json`, and
`events.ndjson` in that directory. The Calibration Diagnostics PR preview then
listed and loaded the completed non-release run through its server API while
reporting the obsolete 2026-09-09 run separately as incompatible.

## Monitoring authentication

Calibration Diagnostics already loads staging files inside Next.js API routes,
and its Hugging Face authorization header is constructed only in server-side
library code. UK support will retain that request boundary and use separate
server deployment variables:

- `POPULACE_UK_STAGING_HF_REPO`
- `POPULACE_UK_STAGING_HF_REVISION`
- `POPULACE_UK_STAGING_HF_TOKEN`

The browser calls the Calibration Diagnostics API routes and never calls the
private Hugging Face repository with a credential. Responses and error text
must not contain the token. The UK token is a fine-grained read credential for
`policyengine/populace-uk-staging`; it is not shared with the local writer.

The version-aware consumer implementation was published for review on
2026-09-08 as
[`PolicyEngine/calibration-diagnostics#181`](https://github.com/PolicyEngine/calibration-diagnostics/pull/181).
Its feature-branch checks passed locally with TypeScript compilation and all
328 tests before publication; its GitHub tests, TypeScript/build check, and
Vercel preview deployment also passed. Review was requested from `@MaxGhenis`.
UK remote output remains disabled until that change is reviewed, merged,
deployed with the separate read credential, and verified against both contract
fixture versions.

## Access-control verification

Hugging Face currently documents both `private=True` and `gated="manual"` on
the repository-settings API, and documents individual approval for gated
datasets. This establishes API support, but not the effective behavior of the
PolicyEngine organization. Before remote output is enabled, the provisioning
procedure must verify the combined settings with an organization writer and
then prove anonymous or unapproved refusal, approved authenticated access, and
narrowly scoped writer access. If any check fails, the repository remains
private and UK remote output remains disabled.

References:

- <https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.update_repo_settings>
- <https://huggingface.co/docs/hub/datasets-gated>
- <https://huggingface.co/docs/hub/repositories-settings>

## Monitoring, rollback, and publication

Consumers discover runs by enumerating `runs/*/run_manifest.json`, then derive
the latest run from manifest `updated_at` timestamps. Monitor each run's
`progress.json` for current status and `events.ndjson` for ordered stage
durations. A remote verification run downloads and validates only its own run
manifest and progress document. Upload failures are recorded with counts and
reviewed error codes while local recording continues; unrestricted remote
exception text is not serialized.

Rollback is configuration-first: use `--staging-local-only` to retain local
evidence or `--no-staging` for a deliberate opt-out, disable the GitHub
workflow, and revoke its optional read credential. Revoke the local writer
credential separately. Leave existing private run evidence intact for audit
and do not alter production repository references.

National calibration build records carry the validated version 2 delivery
object. Release assembly copies that object unchanged into
`build_manifest.json`. Normal publication refuses local-only evidence, invalid
or unknown delivery versions, and remote mode with zero successful uploads. It
retains the existing explicit missing-staging override and accepts a valid
deliberate opt-out. Production publication remains a separate operator action.
