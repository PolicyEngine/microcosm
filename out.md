# F1 portable worker identity — Sol gate round 1 report

Date: 2026-09-04

Branch: `f1-portable-worker-identity`

## State

Implementation is in progress. The four findings have been reproduced with
new regression tests on the `b26708a1` implementation. The only preceding
branch change was the committed progress-journal initialization.

## Environment preparation

- `uv sync --all-packages --locked --extra us --extra uk` exited 2 because the
  runner exports `UV_FROZEN`, which `uv` rejects together with `--locked`.
- `env -u UV_FROZEN uv sync --all-packages --locked --extra us --extra uk`
  exited 2 because the sandbox does not permit uv to initialize its default
  cache under `/Users/maxghenis/.cache/uv`.
- `env -u UV_FROZEN UV_CACHE_DIR=/private/tmp/microcosm-uv-cache uv sync
  --all-packages --locked --extra us --extra uk` exited 0: 125 packages
  resolved and 103 packages checked.
- All pytest commands below likewise set the writable `UV_CACHE_DIR` and use
  the required `uv run --no-sync` mode.

## Finding 1 — schema-9 stacked-envelope bypass

Reproduction test:
`test_scoring_loader_requires_complete_schema_nine_stacked_envelope`.

Command:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-uv-cache uv run --no-sync pytest -q packages/microcosm-build/tests/test_us_multispine_pool_h5_io.py::test_scoring_loader_accepts_legacy_worker_alias_relocation_only packages/microcosm-build/tests/test_us_multispine_pool_h5_io.py::test_scoring_loader_requires_complete_schema_nine_stacked_envelope
```

Result: exit 1; 1 passed and 1 failed. The new regression showed that missing
and wrong `pipeline` values were accepted. Omitting `operator_order`,
`sampling`, `stack_manifest`, `geography_assignment`, or `stage_receipts`
reached later validators instead of the common stacked-envelope refusal. The
existing valid schema-9 path passed and restored both stacked frame-metadata
anchors.

Fix: pending.

Commit SHA: pending.

## Finding 2 — unbound loaded runtime and stdlib

Reproduction tests:
`test_primary_qrf_worker_identity_binds_loaded_runtime_bytes` and
`test_primary_qrf_worker_identity_binds_imported_stdlib_source`.

The four identity/resource reproduction nodes were run together:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-uv-cache uv run --no-sync pytest -q packages/microcosm-build/tests/test_us_stacked_spine.py::test_primary_qrf_worker_identity_binds_loaded_runtime_bytes packages/microcosm-build/tests/test_us_stacked_spine.py::test_primary_qrf_worker_identity_binds_imported_stdlib_source packages/microcosm-build/tests/test_us_stacked_spine.py::test_worker_identity_refuses_unapproved_torch_backend_provider_before_import packages/microcosm-build/tests/test_us_stacked_spine.py::test_worker_transitive_source_identity_binds_actual_imported_package_resource
```

Result: exit 1; 4 failed. For finding 2, changing the temporary runtime
library left both absent `runtime_binary` fields equal (`None == None`), and
changing the temporary imported `argparse.py` left both absent
`stdlib_imports_sha256` fields equal.

Fix: pending.

Commit SHA: pending.

## Finding 3 — Torch backend entry-point autoload

Reproduction tests:
`test_worker_identity_refuses_unapproved_torch_backend_provider_before_import`
and `test_primary_qrf_worker_launch_forces_torch_backend_autoload_off`.

The first ran in the four-node command above and failed because a synthetic,
unapproved distribution declaring a `torch.backends` entry point was accepted
(`DID NOT RAISE RuntimeError`). The launcher test ran separately:

```sh
UV_CACHE_DIR=/private/tmp/microcosm-uv-cache uv run --no-sync pytest -q packages/microcosm-build/tests/test_puf_qrf_chain.py::test_primary_qrf_worker_launch_forces_torch_backend_autoload_off
```

Result: exit 1; 1 failed. With both the inherited and caller-supplied value set
to `1`, the child environment retained `TORCH_DEVICE_BACKEND_AUTOLOAD=1`
instead of forcing `0`.

Fix: pending.

Commit SHA: pending.

## Finding 4 — incomplete import-time resource closure

Reproduction test:
`test_worker_transitive_source_identity_binds_actual_imported_package_resource`.

It ran in the four-node identity command above. Result: exit 1 as part of the
4-failure run. The manual two-file closure contained no row for
`soi_table_2_1_interest_components_ty2015.json` (`len(target_rows) == 0`). A
separate read-only audit-hook probe of a clean worker import found 24 Microcosm
non-code resources, including that asset.

Fix: pending.

Commit SHA: pending.

## Pins moved

Pending implementation and verification.

## Final verification

Pending. No final verification result is claimed yet.

## Deliberately not done

- No network access, push, branch creation, stash, artifact build, release,
  publication, or graph-interface/acceptance-lock edit was performed.
