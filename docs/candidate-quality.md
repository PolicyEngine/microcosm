# Local candidate evidence

`microcosm.build.candidate_quality` seals existing aggregate JSON evidence for
inspection. It never reads population H5 files, runs a simulation, uploads,
changes release defaults, or writes production/staging pointers. It imports
only the Python standard library and depends on no other Microcosm module.

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

## Acceptance, eligibility and geographic claims

`release_eligibility` is recorded, never computed. This transport has no
acceptance policy, no geographic scope configuration and no command-line
surface beyond the three arguments above. It copies the status an authoritative
acceptance producer has already decided, and refuses only the combinations that
would contradict themselves: `eligible` without a completed run, and `eligible`
without a population identity. A successful copy, a completed run, a present
attachment or a matching digest is transport integrity and producer
attestation. None of them is scientific certification, and nothing here can be
read as one.

Nothing in this module grants congressional-district eligibility, or eligibility
at any other geographic level. A CD-scoped aggregate is carried exactly like any
other attachment: its declared identity is checked against the candidate,
population or run it names, and its bytes are preserved unchanged. That
establishes the file is the file it claims to be. It does not establish that a
CD fit was reviewed or that a held-out protocol was run, and this transport
cannot be used to assert either. Declared `geography.levels` and
`geography.vintage` are copied descriptors of what a producer says it computed,
not verified coverage.
