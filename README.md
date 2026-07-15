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

Firm support is **experimental**. The frame kernel can declare firm entity
tables and validate person-firm `jobs` link tables, but link-aware operators,
firm calibration targets, and firm release pipelines are not production
surfaces yet.

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
  --out /tmp/populace-build
```

This writes `progress.json`, `events.ndjson`, `calibration_progress.json`, and
final candidate diagnostics under `runs/<run_id>/` without updating production
`latest.json`.

See [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md) for the measured memory,
disk, and CPU footprint of developing and building locally (and what to budget
on a build machine — RAM is the binding constraint).

## Release-gate preflight

Several release gates fail on facts that are already determined by the base
pool, the frozen selection, and the target/coverage registry — no calibration
solve needed to see them. `tools/preflight_us_release_gates.py` recovers those
signals in **minutes** so a two-hour release launch is not the first place a
knowable defect surfaces:

```bash
uv run python tools/preflight_us_release_gates.py \
  --base-h5 out/base-m/base_populace_us_2024_puf_support.h5 \
  --selection-source-manifest inputs/buildm_keogh_swap_selection_source.json \
  --export-input-mass-reference-h5 forensics/populace_us_2024.h5
```

It is read-only against the H5 artifacts and reports, per check, `PASS` /
`FAIL` / `AT-RISK` with the measured numbers (exit `1` on any FAIL, `2` on
AT-RISK only, `0` clean):

1. **Selection carryover** — the frozen selection-source manifest maps cleanly
   onto the base pool (the frozen-support recovery contract, run pre-solve).
2. **Zero-support preview** — compiled positive fiscal targets whose
   materialized support is ~0 under the selection at base weights stay a
   structural zero after the solve. Direct-column targets are checked;
   engine-derived measures are marked *not statically checkable* (pass
   `--ledger-facts` to compile the target surface).
3. **Export-mass parity risk** — each export-mass column's pool mass at *base*
   weights against its reference band, honoring the release tool's
   `US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS` register (reused, never
   re-declared). A column out of band pre-solve is flagged for review.
4. **Smoke-probe support audit** — every reform-coverage probe leaf's pool vs
   *selected* nonzero support and pool sign-leg decomposition. A leaf with pool
   support but zero selected support **fails** (the input the frozen selection
   cannot express); a thin selection or a signed leaf whose net sign
   contradicts the probe's `expected_sign` is AT-RISK.

**Run it** at base-build exit, before any release launch, and after any change
to the selection-source manifest or the target/coverage registry. The
synthetic-fixture unit tests
(`packages/populace-build/tests/test_us_release_gate_preflight.py`) run in the
normal `uv run pytest` suite; the real-H5 mode above is a local/runbook step.

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
