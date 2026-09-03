# UK dense line release assembly runbook (#762)

The dense line `microcosm-uk-2024-25-dense` is the spine cloned K=15 times
through the OA geography ladder and calibrated to the national and local
target surfaces in one solve (`tools/build_uk_rowwise_candidate.py`). It ships
on the **inspect lane only**: a constant release id, an immutable per-cut tag,
`dataset_role: non_default_local_area`, an empty `default_datasets` map, and
`--no-latest` at publication, so it can never displace the default artifact.
It is registered as `("uk", 2025, "dense")` in the private repo
`policyengine/populace-uk-private`. Publication is a separate human step.

## Prerequisites

- The four pins the run stood on (`spine`, `ladder`, `facts`, `manifest`) and
  the signing key in `MICROCOSM_UK_TERMINAL_GATE_SIGNING_KEY` (base64, 32 bytes).
- The Ledger consumer artifact, the spine H5 with its sidecar, the OA ladder.
- The incumbent extraction (`tools/extract_uk_local_incumbent_surface.py`) for
  the head-to-head score.

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
uv run --no-sync python tools/build_uk_rowwise_candidate.py --release-candidate \
  --input-h5 <spine.h5> --input-sha256 <spine> \
  --ladder build/uk/uk_oa_ladder_2021.npz --ladder-sha256 <ladder> \
  --ledger-facts <chronicle-uk-artifact-dir> --ledger-facts-sha256 <facts> \
  --ledger-manifest-sha256 <manifest> --seed 42 \
  --logbook-prev-row-digest <previous row> --out <candidate-dir>
```

`--release-candidate` pins the doctrine (bound 10, `grain_equal`, K=15, 1500
epochs), resolves the engine in a single block, and runs the rotated holdout.
Expect about 3.5 hours and 10 GB at K=15.

## 3. Pre-flight the finished run, then score it

```bash
uv run --no-sync python tools/preflight_uk_local_release_candidate.py --candidate-dir <candidate-dir>
uv run --no-sync python tools/score_uk_local_candidate.py ... --output-json <candidate-dir>/score_vs_incumbent.json
```

The pre-flight checks the manifest and the signed gate report for everything
the contract will demand: release posture attested, shippable, every
release-blocking gate passed, single-block engine, the doctrine values, the
A15/A17 uprating, the measure exclusions and their windows, the holdout, the
Logbook row, the artifact digest.

## 4. Assemble the release directory

```bash
uv run --no-sync python tools/assemble_uk_dense_release_dir.py \
  --candidate-dir <candidate-dir> --spine-h5 <spine.h5> \
  --incumbent-manifest <incumbent-dir>/incumbent_local_surface_manifest.json \
  --out-dir releases
```

Assembly verifies the hash join (every manifest output against its bytes, the
spine against its pin, the gate report against the Logbook build id), re-runs
the candidate pre-flight, mints the cut tag
`microcosm-uk-2024-25-dense-<YYYYMMDDTHHMMSSZ>-<uuid8>` from the run's attempt
id, clones the H5 beside itself as `microcosm_uk_2025_dense.h5`, stages
`build_manifest.json`, `release_manifest.json`, `calibration_diagnostics.json`,
`gate_summary.json`, `uk_source_coverage.json`, the signed `uk_local_gates.json`,
`score_vs_incumbent.json` and `sha256sums.txt`, validates the directory with
`microcosm.data.contract.validate_release_dir`, and only then renames it into
`releases/microcosm-uk-2024-25-dense/`. Re-assembling requires removing the
previous directory first. The JSON summary prints the publication command.

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
