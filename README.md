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

Long-running US fiscal refresh builds can emit pre-release telemetry before the
candidate is published as the production Hugging Face release. Pass
`--staging-dir` to write local staging artifacts, and optionally
`--staging-repo-id policyengine/populace-us-staging` to upload small progress
JSON files while the build runs:

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
