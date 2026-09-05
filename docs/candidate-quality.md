# Local candidate evidence

`microcosm.build.candidate_quality` seals existing aggregate JSON evidence for
inspection. It never reads population H5 files, runs a simulation, uploads,
changes release defaults, or writes production/staging pointers. It has no
dependency on the model-artifact graph PR.

Run from the locked workspace:

```sh
uv run --no-sync python tools/build_candidate_quality.py \
  --metadata candidate.json --attachments attachments.json \
  --out /absolute/path/to/candidate-quality
```

The output root is dedicated to one country's candidate evidence. A single
writer updates its optional `index.json`; `--no-index` writes only immutable
artifacts. Consumers can select an exact candidate and manifest SHA without
reading the index. Completed manifests live at `manifests/<sha256>.json`.
Available aggregate files retain their exact source bytes under
`artifacts/<sha256>.json`. No scientific eligibility is inferred from a
successful copy or a completed run.

Metadata example (illustrative identities, not a real population):

```json
{
  "country": "us",
  "candidate_release_id": "candidate-example",
  "run_id": "build-example",
  "created_at": "2026-09-05T00:00:00Z",
  "execution_status": "completed",
  "release_eligibility": {
    "status": "not_evaluated",
    "reason": "Independent acceptance has not completed."
  },
  "population": {
    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "period": "2024",
    "entity": "household"
  }
}
```

`execution_status` is `pending`, `running`, `completed`, `failed` or `cancelled`.
`release_eligibility.status` is independently `not_evaluated`, `eligible`,
`ineligible` or `excluded`, with a mandatory reason. Eligibility is an explicit
attestation from the authoritative acceptance producer, not a decision of this
transport. `eligible` requires a completed run and a population identity.
Historical excluded artifacts remain diagnostic references even if their run
completed. `population` can be null only for pending/failed/cancelled runs that
have not produced a population.

Optional `provenance` contains known string identities (Microcosm commit, model
package/version/commit, target surface SHA, Chronicle snapshot). Optional
`geography` contains declared `levels` and `vintage`. Do not insert guessed
identities to make an attachment compatible.

The attachment map uses names such as `calibration_diagnostics`, `support`,
`distributions`, `build_manifest`, `release_manifest`, `cross_dataset`,
`validation_reach`, `reform_validation` and `demographics`. An available entry
contains `status`, `source` (local JSON path relative to the map file), `scope`
and `identity`. A nonavailable entry contains only `status` (`pending`,
`missing`, `failed`) and a nonempty `reason`. Expected omitted attachments become
`missing`; no zero metrics are fabricated.

- `population` scope requires exact `candidate_release_id` and
  `population_sha256`; supplied known model/target/snapshot identities must agree.
- `model` scope requires the actual `model_package` and `model_version`. It
  cannot claim a candidate or population; model coverage is not population
  validation.
- `run` scope requires exact `candidate_release_id` and `run_id` and cannot
  claim a population. Use it for a source-load failure or execution profile
  before population construction.

Standalone aggregate JSON is capped at 64 MiB. Supply existing calibrated
schema-7 diagnostics unchanged. Support and distribution attachments must be
actual aggregate outputs using their declared weight and geography definitions;
this producer does not calculate them. `summarize_pool_diagnostics` projects
existing pool receipts without loading/re-scoring H5: it labels failure-string
counts precisely and never interprets them as counts of failed comparison IDs.

## Cross-dataset bundles

An available `cross_dataset` entry points to the existing
`cross_dataset.frontend_bundle.v1` manifest. Its identity additionally requires
`run_id`, `snapshot_id` and `source_id`; the evaluation run and snapshot may
differ from the population build run and calibration Chronicle snapshot. The
selected source and country must occur in the bundle.
The producer verifies all declared partition hashes/run/snapshot identities and
copies the unchanged manifest to `artifacts/<sha256>/manifest.json` with all
partitions at their original safe relative paths. Bundle content is bounded at
256 MiB. Missing partitions, collisions, unsafe paths or mismatched identities
fail before writing. The existing dashboard reader remains authoritative for
the bundle's evaluation semantics.

## Geography and acceptance are declarations, not decisions made here

This transport does not decide, compute or grant release eligibility, and it
grants no geographic eligibility in particular — national, state or
congressional district. It has no acceptance-policy surface: nothing here reads
or writes a release-scope configuration, and it adds no flag to any build CLI.

`release_eligibility` is an attestation copied from a separate authoritative
acceptance producer, which owns the scientific gate. This module only checks two
consistency rules on that attestation — `eligible` requires a completed run, and
`eligible` requires an actual population identity — and both are refusals, not
grants. A completed run with a population is not thereby eligible; only the
acceptance producer's own `eligible` attestation makes it so, and sealing that
attestation does not verify it.

`geography` is likewise a producer declaration of the levels and vintage a
candidate's aggregates partition. A declared level names a partition; it
establishes no source authority, no nesting between levels or vintages, and no
certification that analysis at that level is sound. Congressional-district
aggregates are inspectable evidence here whether or not a reviewed CD fit and
held-out protocol exists; if one does not, that is a fact about the candidate
that must be recorded in `release_eligibility.reason` by the acceptance
producer, because this transport cannot detect it.
