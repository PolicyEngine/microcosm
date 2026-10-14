# UK dense line release assembly runbook (#762)

The dense line `microcosm-uk-2024-25-dense` is the spine cloned K=15 times
through the atomic output-area assignment and calibrated to the national and local
target surfaces in one solve (`microcosm-build-uk --release-role dense`, the
graph driver `tools/build_uk_full.py`; `tools/build_uk_rowwise_candidate.py`
is a stub over it). It ships
on the **inspect lane only**: a constant release id, an immutable per-cut tag,
`dataset_role: non_default_local_area`, an empty `default_datasets` map, and
`--no-latest` at publication, so it can never displace the default artifact.
It is registered as `("uk", 2025, "dense")` in the private repo
`policyengine/populace-uk-private`. Publication is a separate human step.
Since microcosm#1124 (ruling of 2026-10-13) the line may ship the
100,000-household exact-k selection from that pool instead of the whole pool,
under the same requirements; see [Sized release](#sized-release-microcosm1124).

The R16/R17 release verdicts recorded in the historical receipts used the
previous gate policy. They do not satisfy the current contract: four quality
gates now block release, and the incumbent-surface evaluation is mandatory.
No historical run was re-signed or recalibrated by the PR #870 review fixes.

## Prerequisites

- The four pins the run stood on (`spine`, `ladder`, `facts`, `manifest`) and
  the signing key in `MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY` (base64, 32 bytes).
- The Ledger consumer artifact, the spine H5 with its sidecar, the OA ladder.
- The incumbent extraction (`tools/extract_uk_local_incumbent_surface.py`) for
  the head-to-head score.
- The licensed input-mass reference the national line keeps with its runs
  (`--input-mass-reference`). `uk_input_mass_parity` is release-blocking on
  this line too and records `evidence_absent` without it. The sidecar's totals
  digest must match the registry's `totals_sha256`; the totals depend on the
  installed policyengine-uk, so re-emitting them under a newer engine does not
  reproduce the pinned file.

## 1. Pre-flight the environment

```bash
uv run --no-sync python tools/preflight_uk_local_release_candidate.py --env \
  --pins <pins.txt> --spine-h5 <spine.h5> --ladder build/uk/uk_oa_ladder_2021.npz \
  --ledger-facts <chronicle-uk-artifact-dir>
```

Fails closed and by name on a missing or malformed key, a missing pin, a
digest mismatch, or doctrine constants that are not the ruled ones.

## 2. Run the release candidate

```bash
uv run --no-sync python tools/build_uk_full.py --release-role dense \
  --release-candidate --input-h5 <spine.h5> --input-sha256 <spine> \
  --ladder build/uk/uk_oa_ladder_2021.npz --ladder-sha256 <ladder> \
  --ledger-facts <chronicle-uk-artifact-dir> --ledger-facts-sha256 <facts> \
  --ledger-manifest-sha256 <manifest> --seed 42 \
  --input-mass-reference <licensed-input-mass-reference> \
  --logbook-prev-row-digest <previous row> --out <candidate-dir>
```

`--release-role dense` is required (microcosm#823): the role supplies the
solve defaults (K=15, seed 42, 2,000 epochs, learning rate 0.15,
`nation_grain_family_equal_sqrt_count`), names the outputs `microcosm_uk_2024_25_local.h5` and
`microcosm_uk_2024_25_local.local_gates.json`, records itself in the
manifest, and refuses the national role's flags. `--release-candidate` pins
the doctrine (bound 10, `nation_grain_family_equal_sqrt_count`, K=15, 2000
epochs) and runs the
rotated holdout. The engine may resolve in a single block or per clone
(`--engine-blocks K`): each block's engine weights are scaled to the pool so
weight-share formulas see the pool's denominator, and the measures receipt
records whether the blocks were identical copies (an exact representation);
a per-block run is releasable only when it was. Best-effort staging
telemetry uploads every 300 s by default on this driver (the Hub allows about
128 commits per hour per repository).
For scale: at K=25 with `--engine-blocks 25` the graph driver's dense solve took
about 3 hours (2,000 epochs over 1.6 million households, M1 Max), each of three
size-search probes about as long, and the build peaked at 24.7 GiB (the
2026-10-11 100,000-household candidate). At K=15 the pool is 60 % of that size,
and the rowwise tool measured about 3.5 hours and 10 GB for an unsized build.
A release candidate also runs the rotated holdout, which repeats the dense
solve in each of its five folds, and in a sized build the size search and
refit too. Since microcosm#901 the
dense role runs through the [full-build graph](uk-full-build-graph.md): the
candidate directory also holds `candidate.json`, `certification.json`,
`operations.json` and the `build.json` completion marker, the graph store sits
at `<candidate-dir>/.graph-store` unless `--graph-store` says otherwise, and
`--input-h5` must be a spine whose gate manifest matches the branch's gate
declarations.

### Sized release (microcosm#1124)

A release candidate may add `--dataset-households 100000` to ship the exact-k
selection instead of the whole pool. It must carry the local size doctrine
(`local_doctrine.uk_local_size_doctrine`): 100,000 households, an exact-count
draw at π_hi 0.5, the refit baseline floored at 0.5 and a chi-square refit L2
penalty toward uniform weights at λ 0.0216, drawn with the solve seed. The role
fills those values when the request leaves them unset and refuses any other.
`--selection-initial-lambda` warm-starts the budget search, and every holdout
fold's search starts from the same value. The requirements are the dense
line's (ruling of 2026-10-13; no matched-size scorecard): every
release-blocking gate passed, the rotated holdout, the pre-flight, which also
checks the size receipt, and the incumbent-surface evaluation on one engine
over the 100,000-household file. The release identity is unchanged.

## 3. Pre-flight the finished run, then score it

```bash
uv run --no-sync python tools/preflight_uk_local_release_candidate.py --candidate-dir <candidate-dir>
uv run --no-sync python tools/score_uk_local_candidate.py ... --output-json <candidate-dir>/score_vs_incumbent.json
```

The scorer's `--registry-json` is the candidate's target selection,
`<candidate-dir>/microcosm_uk_2024_25_local.target_selection.json` (the
manifest's `outputs.target_registry`); it scores that selection's local rows,
which must be the 21,596 compiled local references of the #1123 surface
less the 208 cells #1278's signed small-cell rules defer, and
must all appear in the candidate's diagnostics, so scoring needs the full rung.

The pre-flight checks the manifest and the signed gate report for everything
the contract will demand: release posture attested, shippable, every
release-blocking gate passed, an exact engine population (single block, or
per-block with identical copies scaled to the pool), the doctrine values (for
a sized run, the size doctrine and a realized count equal to the request), the
A15 census household uprating and A17 tenure application, the measure
exclusions and their windows, the holdout, the
Logbook row, the artifact digest.

Measure the full pinned incumbent surface before assembly:

```bash
uv run --no-sync python tools/evaluate_uk_incumbent_surface.py \
  --candidate-h5 <candidate-dir>/microcosm_uk_2024_25_local.h5 \
  --candidate-manifest <candidate-dir>/rowwise_candidate_manifest.json \
  --ledger-facts <chronicle-uk-artifact-dir> --ledger-facts-sha256 <facts> \
  --ledger-manifest-sha256 <manifest> --engine-blocks 1 \
  --incumbent-manifest <incumbent-dir>/incumbent_local_surface_manifest.json \
  --incumbent-metrics-csv <incumbent-dir>/household_metrics.csv \
  --incumbent-weights-csv <incumbent-dir>/wide_weights.csv \
  --out-json <candidate-dir>/incumbent_surface_evaluation.json \
  --out-md <candidate-dir>/incumbent_surface_evaluation.md
```

This evaluation is also where the released file's fit is gated on the engine
a consumer runs. The build's release-blocking fit gates read the measures it
resolved, per clone when `--engine-blocks K`; the build records that
representation as exact only for the weight-share scaling
(`engine_population_representation.exact_scope`), and engine draws that depend
on a record's position can differ between per-clone and single-block runs. The
evaluator re-measures the released file on one engine, which the data contract
requires, and its assessment fails release for any national or local row more
than 25% from its target, or any family of five or more rows with fewer than
half within 25%. Signed deferrals and measure exclusions do not exempt a row.

Use the actual metrics and weights filenames from the extraction manifest.
The evaluator remains diagnostic: missing optional incumbent inputs and poor
fit produce a failed assessment, not permission to publish. Only one engine
block is accepted. Assembly requires the complete authenticated evaluation,
including finite candidate measurements on every national and local row and
finite realized incumbent estimates on every local row. National comparisons
use the pinned incumbent targets; they do not claim realized incumbent fit.
Signed deferrals stay in this evaluation. A missing or unmeasurable row blocks
release until its measurement is supplied.

The same existing absolute quality limits apply to this surface: every row
within 25%, and at least half each family's rows within 25% when the family
has at least five rows. The within-10% family share remains diagnostic.
The candidate's fitted score uses uniform rows on its active local surface.
Its holdout uses the separately recorded weighting rule over held local
grains. Their shared cap does not make the losses directly comparable; no
ranking of fitted versus holdout losses is reported.

## 4. Assemble the release directory

```bash
uv run --no-sync python tools/assemble_uk_dense_release_dir.py \
  --candidate-dir <candidate-dir> --spine-h5 <spine.h5> \
  --incumbent-manifest <incumbent-dir>/incumbent_local_surface_manifest.json \
  --out-dir releases
```

Assembly requires `release_role: "dense"` in the candidate manifest (as does
the pre-flight; a candidate built before the role existed is rebuilt, never
grandfathered), verifies the hash join (every manifest output against its
bytes, the spine against its pin, the gate report against the Logbook build
id), requires
the manifest's `staging_delivery` receipt (the run's version 2 staging
telemetry evidence, copied into `build_manifest.json` as `staging` so
publication can apply its undelivered-staging refusal, as on the national
lane), re-runs the candidate pre-flight, mints the cut tag
`microcosm-uk-2024-25-dense-<YYYYMMDDTHHMMSSZ>-<uuid8>` from the run's attempt
id, clones the H5 beside itself as `microcosm_uk_2024_25_dense.h5`, stages
`build_manifest.json`, `release_manifest.json`, `calibration_diagnostics.json`,
`gate_summary.json`, `uk_source_coverage.json`, the signed `uk_local_gates.json`,
`score_vs_incumbent.json`, `incumbent_surface_evaluation.json`, the original
`rowwise_candidate_manifest.json`, `source_calibration_diagnostics.json`,
`incumbent_manifest.json`, and `sha256sums.txt`, validates the directory with
`microcosm.data.contract.validate_release_dir`, and only then renames it into
`releases/microcosm-uk-2024-25-dense/`. Re-assembling requires removing the
previous directory first. The JSON summary prints the publication command.

Assembly and every later directory validation require measured clean code
(`code.git_dirty` exactly `false`) and full measure-exclusion provenance.
Approval and expiry dates must be valid ISO dates and in force on the current
validation date; expiry-day validation is allowed, the following day is not.
The upload path invokes this validator again before uploading bytes. Separate support
and binding adjudications keep their own policies. The gate thresholds and
existing approvals have not been widened or renewed.

## 5. Publish for inspection (human step)

Run the printed command. Its shape is:

```bash
uv run python -m microcosm.data.publish_cli releases/microcosm-uk-2024-25-dense \
  --repo-id policyengine/populace-uk-private --artifact-root <candidate-dir> \
  --no-latest --tag-name microcosm-uk-2024-25-dense-<timestamp>-<uuid8>
```

`--no-latest` is mandatory and enforced: publication refuses to move
`latest.json` for a non-default role. The artifact is reachable by its tag and
by the registry key `("uk", 2025, "dense")` only.

## Promotion is a separate change

Making the dense line (or a sparse successor via the L0 penalty, #762 I10) a
default dataset is a registry and contract change with its own review; nothing
in this runbook promotes anything.
