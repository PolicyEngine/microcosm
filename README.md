# populace

The population stack: one kernel datatype — the **`Frame`**, a weighted
sampling frame of entity tables — and packages as operators on it. One PEP 420
`populace` namespace, shipped as shard distributions; a `populace` metapackage
will pin the constellation.

| package | import | role | succeeds |
|---|---|---|---|
| `populace-frame` | `populace.frame` | the kernel: Frame, typed weights, strata, links, weighted accounting, unit structure, rules-engine protocol | microdf, microunit |
| `populace-fit` | `populace.fit` | conditional models (weight-aware by construction) | ad hoc imputation scripts |
| `populace-calibrate` | `populace.calibrate` | representation: targets → calibrated weights (APG / L0) | microcalibrate |
| `populace-build` | `populace.build` | population build plans, donor graphs, release gates, and country build stages | one-off build drivers |
| `populace-data` | `populace.data` | published population registry and lazy engine loaders | country-specific data packages |

See [DESIGN.md](DESIGN.md) for the charter: why the rebuild, the kernel
semantics, the RulesEngine protocol (policyengine-us today, Axiom rulespec-us
next), longitudinal design (one weight per trajectory), and the process rules
(behavioral contract tests, constellation versioning, environment-carrying
artifacts).

Incumbent comparisons and historical replacement benchmarks live outside this
repo. The live Populace repo owns the library, build contracts, published
population registry, and acceptance gates.

## Development

```bash
uv sync --all-packages   # workspace install (all members + dev groups)
uv run pytest            # all packages, incl. behavioral contract tests
uv run ruff check .
```

## Staging build telemetry

US fiscal refresh builds emit pre-release staging telemetry **by default**:
progress JSON is uploaded to `policyengine/populace-us-staging` while the build
runs (best-effort — a missing token or failed upload never fails the build), so
every candidate shows up on the staging dashboard before it is published.
Disable with `--no-staging`, or point elsewhere with `--staging-repo-id` /
`POPULACE_STAGING_REPO_ID`. The build manifest records the staging run id, and
`populace-publish-release` warns when publishing a release that has none:

```bash
python tools/build_us_fiscal_refresh_release.py \
  --ledger-facts consumer_facts.jsonl \
  --out /tmp/populace-build \
  --staging-repo-id policyengine/populace-us-staging
```

This writes `progress.json`, `events.ndjson`, `calibration_progress.json`, and
final candidate diagnostics under `runs/<run_id>/` without updating production
`latest.json`.

See [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md) for the measured memory,
disk, and CPU footprint of developing and building locally (and what to budget
on a build machine — RAM is the binding constraint).

## Releasing & alerts

Publishing uploads the locally built `releases/<id>/` artifacts to the Hugging
Face dataset, tags the release, and updates `latest.json`. It runs on the build
machine (it needs the freshly built H5), so it isn't a CI step:

```bash
tools/publish_release.sh releases/<id> --repo-id policyengine/populace-us
```

`tools/publish_release.sh` is a thin wrapper around `populace-publish-release`
(all arguments pass straight through). The moment `latest.json` goes live, the
publish CLI posts a release alert to Slack — `#populace-us` or `#populace-uk`,
chosen from the repo id.

The alert is a **no-op unless the channel's incoming-webhook URL is set**, so
configure it once on the build machine:

```bash
cp tools/release.env.example tools/release.env   # then paste the webhook URLs
```

`tools/release.env` is gitignored; the wrapper loads it (or you can just export
`SLACK_WEBHOOK_POPULACE_US` / `SLACK_WEBHOOK_POPULACE_UK` in your shell) and
warns if neither is set. After that, every release publishes with an automatic
Slack alert.
