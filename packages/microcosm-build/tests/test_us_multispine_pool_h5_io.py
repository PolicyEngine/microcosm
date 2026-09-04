from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.acs_transfer as acs_transfer_module
import microcosm.build.us_runtime.h5_io as h5_io
import microcosm.build.us_runtime.post_transfer_calibration as post_transfer_calibration_runtime
import microcosm.build.us_runtime.stacked_spine as stacked_spine_module
import microcosm.build.us_runtime.worker_identity as worker_identity_module
from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.serialization_dtypes import (
    CANONICAL_STRING_DTYPE,
    canonicalize_frame_string_dtypes,
    canonicalize_table_string_dtypes,
)
from microcosm.build.us_runtime.congressional_district_vintage import (
    CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
    CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
)
from microcosm.build.us_runtime.h5_io import (
    LEGACY_NULLABLE_STAGING_ARTIFACT_KIND,
    US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND,
    US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
    US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION,
    US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
    US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION,
    US_STACKED_POOL_OPERATOR_ORDER,
    AuthenticatedPoolH5,
    AuthenticatedPoolH5MismatchError,
    identify_us_multispine_pool_manifest,
    load_authenticated_us_multispine_pool_for_release,
    load_authenticated_us_multispine_pool_for_scoring,
    load_simulation_ready_us_multispine_pool,
    require_authenticated_us_multispine_pool_h5,
    us_multispine_pool_release_receipt,
    write_nullable_us_h5,
)
from microcosm.build.us_runtime.support_provenance import (
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


@pytest.fixture(scope="module", autouse=True)
def _reuse_real_worker_identity_for_manifest_checks() -> object:
    """Reuse real identities while this module mutates only artifact payloads."""

    original_binding = worker_identity_module.primary_qrf_worker_execution_binding
    original_semantic = worker_identity_module.primary_qrf_worker_semantic_identity

    def controls() -> tuple[str | None, str | None, int | None]:
        return (
            worker_identity_module.os.environ.get("POPULACE_FIT_N_JOBS"),
            worker_identity_module.os.environ.get("POPULACE_FIT_PREDICT_WORKERS"),
            worker_identity_module.os.cpu_count(),
        )

    @cache
    def cached_semantic(
        uv_lock_sha256: str | None,
        _fit_jobs: str | None,
        _predict_workers: str | None,
        _cpu_count: int | None,
    ) -> dict[str, object]:
        return original_semantic(uv_lock_sha256=uv_lock_sha256)

    def semantic_identity(*, uv_lock_sha256: str | None = None) -> dict[str, object]:
        return deepcopy(cached_semantic(uv_lock_sha256, *controls()))

    @cache
    def cached_binding(
        _fit_jobs: str | None,
        _predict_workers: str | None,
        _cpu_count: int | None,
    ) -> dict[str, object]:
        return original_binding()

    def binding() -> dict[str, object]:
        return deepcopy(cached_binding(*controls()))

    # Each lock/control combination still constructs a real identity. Keep the
    # cached expected values private and return independent copies so a re-signed
    # malicious manifest or attestation cannot mutate the validation baseline.
    # Source/cache mutation regressions live outside this module; production
    # identity generation remains uncached after this fixture is torn down.
    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        worker_identity_module,
        "primary_qrf_worker_semantic_identity",
        semantic_identity,
    )
    patcher.setattr(
        worker_identity_module, "primary_qrf_worker_execution_binding", binding
    )
    yield
    patcher.undo()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pool_frame() -> Frame:
    ids = np.asarray([10, 20, 30], dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            **{
                US_SCHEMA.membership_column(entity): ids
                for entity in US_SCHEMA.group_entities
            },
            "nullable_input": np.asarray([True, None, False], dtype=object),
        }
    )
    tables = {
        "person": person,
        **{
            entity: pd.DataFrame({US_SCHEMA.id_column(entity): ids})
            for entity in US_SCHEMA.group_entities
        },
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([2.0, 3.0, 5.0]),
                WeightKind.IMPORTANCE,
            )
        },
    )


def _pool_frame_with_object_strings_on_every_entity(
    *,
    stacked: bool = False,
) -> Frame:
    """Match the assembled pool's object-backed source-string shape."""

    frame = _pool_frame()
    tables = {}
    for entity in frame.entities:
        table = frame.table(entity).copy()
        column = "PERIDNUM" if entity == "person" else f"{entity}_source_label"
        table.insert(
            0,
            column,
            pd.Series(
                [f"{entity}-0", None, f"{entity}-2"],
                index=table.index,
                dtype=object,
            ),
        )
        tables[entity] = table
    if stacked:
        household = tables["household"].copy()
        household["puma"] = ["0600101", "0600102", "0600102"]
        household["congressional_district_geoid"] = np.asarray(
            [601, 601, 601],
            dtype=np.int64,
        )
        household["county_fips"] = ["06001", "06001", "06001"]
        household[support_source_id_column("household")] = np.asarray(
            [10, 20, 20], dtype=np.int64
        )
        household[support_clone_index_column("household")] = np.asarray(
            [0, 0, 1], dtype=np.int64
        )
        tables["household"] = household
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _semantic_string_columns(table: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in table.columns
        if isinstance(table[column].dtype, pd.StringDtype)
        or (
            pd.api.types.is_object_dtype(table[column].dtype)
            and pd.api.types.infer_dtype(table[column], skipna=True) == "string"
        )
    )


def test_string_canonicalization_reuses_unchanged_numeric_storage() -> None:
    source = pd.DataFrame(
        {
            "PERIDNUM": pd.Series(["1", None, "3"], dtype=object),
            "employment_income": np.asarray([10.0, 20.0, 30.0]),
        }
    )

    canonical = canonicalize_table_string_dtypes(
        source,
        boundary="fixture checkpoint load",
        table_name="person",
    )

    assert canonical is not source
    assert source["PERIDNUM"].dtype == np.dtype(object)
    assert canonical["PERIDNUM"].dtype == CANONICAL_STRING_DTYPE
    assert np.shares_memory(
        canonical["employment_income"].to_numpy(),
        source["employment_income"].to_numpy(),
    )


def test_in_place_frame_canonicalization_reuses_all_numeric_storage() -> None:
    frame = _pool_frame_with_object_strings_on_every_entity()
    numeric_storage = {
        entity: frame.table(entity)[US_SCHEMA.entity_id_column(entity)].to_numpy()
        for entity in US_SCHEMA.entities
    }

    canonical = canonicalize_frame_string_dtypes(
        frame,
        boundary="fixture checkpoint load",
        in_place=True,
    )

    assert canonical is frame
    for entity in US_SCHEMA.entities:
        table = canonical.table(entity)
        assert np.shares_memory(
            table[US_SCHEMA.entity_id_column(entity)].to_numpy(),
            numeric_storage[entity],
        )
        assert all(
            table[column].dtype == CANONICAL_STRING_DTYPE
            for column in _semantic_string_columns(table)
        )


def test_in_place_frame_canonicalization_is_atomic_on_ambiguity() -> None:
    frame = _pool_frame_with_object_strings_on_every_entity()
    frame.table("household")["household_source_label"] = pd.Series(
        ["household-0", 2, None],
        dtype=object,
    )

    with pytest.raises(TypeError, match="household.household_source_label"):
        canonicalize_frame_string_dtypes(
            frame,
            boundary="fixture checkpoint load",
            in_place=True,
        )

    assert frame.table("person")["PERIDNUM"].dtype == np.dtype(object)


def test_object_string_simulated_checkpoint_resume_exports_canonical_strings(
    tmp_path: Path,
) -> None:
    """Regress the exact production shape while proving loader symmetry."""

    pytest.importorskip("h5py")
    pytest.importorskip("tables")
    fresh = _pool_frame_with_object_strings_on_every_entity()
    assert fresh.table("person").columns[0] == "PERIDNUM"
    checkpoint_path = tmp_path / "simulated.checkpoint.h5"
    write_frame_checkpoint(
        checkpoint_path,
        fresh,
        metadata={"stage": "simulated"},
    )
    resumed = load_frame_checkpoint(checkpoint_path).frame

    for entity in US_SCHEMA.entities:
        string_columns = _semantic_string_columns(fresh.table(entity))
        assert string_columns
        for column in string_columns:
            assert fresh.table(entity)[column].dtype == np.dtype(object)
            assert resumed.table(entity)[column].dtype == np.dtype(object)

    for label, frame in (("fresh", fresh), ("resumed", resumed)):
        output = tmp_path / f"{label}.pool.h5"
        write_nullable_us_h5(
            frame,
            output,
            period=2024,
            artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
            publication_run_id=f"{label}-fixture-publication",
        )
        # Fixed-format HDF persists the logical ``str`` dtype but not the
        # pandas storage backend; a raw read resolves storage from the
        # environment (python without pyarrow installed, pyarrow with).  Pin
        # the persisted logical dtype, then prove the load boundary restores
        # the exact canonical dtype in either environment.
        with pd.HDFStore(output, mode="r") as store:
            for entity in US_SCHEMA.entities:
                stored = store[entity]
                string_columns = _semantic_string_columns(stored)
                assert string_columns
                for column in string_columns:
                    dtype = stored[column].dtype
                    assert isinstance(dtype, pd.StringDtype)
                    assert dtype.na_value is np.nan
                canonical = canonicalize_table_string_dtypes(
                    stored,
                    boundary="raw pool store load",
                    table_name=entity,
                )
                assert all(
                    canonical[column].dtype == CANONICAL_STRING_DTYPE
                    for column in string_columns
                )
        with pd.option_context("mode.string_storage", "python"):
            with pd.HDFStore(output, mode="r") as store:
                for entity in US_SCHEMA.entities:
                    stored = store[entity]
                    assert all(
                        stored[column].dtype == CANONICAL_STRING_DTYPE
                        for column in _semantic_string_columns(stored)
                    )
        for entity in US_SCHEMA.entities:
            string_columns = _semantic_string_columns(frame.table(entity))
            assert all(
                frame.table(entity)[column].dtype == np.dtype(object)
                for column in string_columns
            )


def test_pool_export_rejects_ambiguous_object_strings_before_replacement(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    frame = _pool_frame_with_object_strings_on_every_entity()
    frame.table("person")["PERIDNUM"] = pd.Series(
        ["person-0", 2, None],
        dtype=object,
    )
    output = tmp_path / "existing.pool.h5"
    output.write_bytes(b"previous-good-pool")

    with pytest.raises(
        TypeError,
        match=(
            "nullable US H5 export.*person.PERIDNUM.*"
            "offending value types.*builtins.int"
        ),
    ):
        write_nullable_us_h5(
            frame,
            output,
            period=2024,
            artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
        )

    assert output.read_bytes() == b"previous-good-pool"


@pytest.mark.parametrize("materializer_version", (True, 0, -1, 1.5, "2"))
def test_pool_export_rejects_invalid_materializer_versions_before_replacement(
    tmp_path: Path,
    materializer_version: object,
) -> None:
    output = tmp_path / "existing.pool.h5"
    output.write_bytes(b"previous-good-pool")

    with pytest.raises(ValueError, match="positive integer"):
        write_nullable_us_h5(
            _pool_frame(),
            output,
            period=2024,
            artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
            materializer_version=materializer_version,  # type: ignore[arg-type]
        )

    assert output.read_bytes() == b"previous-good-pool"


def test_pool_export_rejects_untyped_all_missing_objects_before_replacement(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    frame = _pool_frame_with_object_strings_on_every_entity()
    frame.table("household")["household_source_label"] = pd.Series(
        [None] * len(frame.table("household")),
        dtype=object,
    )
    output = tmp_path / "existing.pool.h5"
    output.write_bytes(b"previous-good-pool")

    with pytest.raises(
        TypeError,
        match=(
            "nullable US H5 export.*household.household_source_label.*"
            "no observed values.*declare an explicit dtype"
        ),
    ):
        write_nullable_us_h5(
            frame,
            output,
            period=2024,
            artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
        )

    assert output.read_bytes() == b"previous-good-pool"


def test_pool_export_canonicalizes_explicit_all_missing_strings(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    frame = _pool_frame_with_object_strings_on_every_entity()
    frame.table("household")["household_source_label"] = pd.Series(
        pd.NA,
        index=frame.table("household").index,
        dtype=pd.StringDtype(storage="python", na_value=pd.NA),
    )
    output = tmp_path / "typed-all-missing.pool.h5"

    write_nullable_us_h5(
        frame,
        output,
        period=2024,
        artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
    )

    with pd.HDFStore(output, mode="r") as store:
        stored = store["household"]["household_source_label"]
    assert isinstance(stored.dtype, pd.StringDtype)
    assert stored.dtype.na_value is np.nan
    assert stored.isna().all()
    with pd.option_context("mode.string_storage", "python"):
        with pd.HDFStore(output, mode="r") as store:
            repinned = store["household"]["household_source_label"]
    assert repinned.dtype == CANONICAL_STRING_DTYPE
    assert repinned.isna().all()


def test_pool_h5_load_boundary_canonicalizes_under_pyarrow_default(
    tmp_path: Path,
) -> None:
    """Raw read-back storage is environment-resolved; the load boundary owns
    the exact canonical dtype even under a pyarrow string-storage default."""

    pytest.importorskip("tables")
    pytest.importorskip("pyarrow")
    frame = _pool_frame_with_object_strings_on_every_entity()
    output = tmp_path / "pyarrow-default.pool.h5"
    write_nullable_us_h5(
        frame,
        output,
        period=2024,
        artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
    )
    with pd.option_context("mode.string_storage", "pyarrow"):
        with pd.HDFStore(output, mode="r") as store:
            stored = store["person"]
        assert stored["PERIDNUM"].dtype != CANONICAL_STRING_DTYPE
        canonical = canonicalize_table_string_dtypes(
            stored,
            boundary="pyarrow-default pool load",
            table_name="person",
        )
    assert canonical["PERIDNUM"].dtype == CANONICAL_STRING_DTYPE
    assert canonical["PERIDNUM"].iloc[0] == "person-0"
    assert pd.isna(canonical["PERIDNUM"].iloc[1])


def _fixture_geography_declaration() -> dict[str, object]:
    return {
        "anchor": "puma",
        "order": "before_gap_fill",
        "kernels": {
            "assign": "kernel:assign_us_puma_ladder",
            "validate": "kernel:us_puma_ladder_gate",
        },
        "draw": {
            "asec": {
                "universe": "puma_within_state",
                "weight": "puma_population_2020",
            },
            "congressional_district": {
                "universe": "congressional_district_within_puma",
                "weight": "block_population_overlap",
            },
            "county": {
                "universe": "county_within_puma",
                "weight": "block_population_overlap",
            },
        },
        "derive": ["puma", "congressional_district_geoid", "county_fips"],
        "assertions": [
            "observed_acs_puma_preserved",
            "geography_state_prefix_consistent",
        ],
        "ladder_source": "source:us_puma_ladder_2020",
        "congressional_district_vintage_crosswalk": {
            "source_ref": (
                "source:us_congressional_district_vintage_crosswalk_117_to_119"
            ),
            "source_vintage": "vintage:cd_117",
            "target_vintage": "vintage:cd_119",
        },
        "seed": "stream:geography_legacy",
        "default_seed": 0,
        "assign_tract": False,
        "layer_vintages": {
            "congressional_district": "vintage:cd_119",
            "county": "vintage:census_2020",
            "puma": "vintage:puma_2020",
            "tract": "vintage:census_2020",
        },
        "validation": ["puma_ladder_gate", "vintage_refusal"],
    }


def _fixture_geography_assignment(
    household: pd.DataFrame,
) -> dict[str, object]:
    clone_column = support_clone_index_column("household")
    native = household.loc[household[clone_column].eq(0)]
    household_ids = native["household_id"].to_numpy(dtype="<i8", copy=False)
    order_digest = hashlib.sha256()
    order_digest.update(b"populace-ordered-household-id-int64-le-v1\0")
    order_digest.update(
        len(household_ids).to_bytes(8, byteorder="little", signed=False)
    )
    order_digest.update(household_ids.tobytes(order="C"))

    geography_arrays = (
        household_ids,
        native["puma"].astype(np.int64).to_numpy(dtype="<i8", copy=False),
        native["congressional_district_geoid"].to_numpy(
            dtype="<i8",
            copy=False,
        ),
        native["county_fips"]
        .astype(np.int64)
        .to_numpy(
            dtype="<i8",
            copy=False,
        ),
    )
    geography_digest = hashlib.sha256()
    geography_digest.update(
        b"populace-ordered-household-geography-column-major-int64-le-v1\0"
    )
    geography_digest.update(len(native).to_bytes(8, byteorder="little", signed=False))
    for values in geography_arrays:
        geography_digest.update(values.tobytes(order="C"))

    return {
        "artifact_kind": "populace_us_stacked_household_geography_assignment",
        "schema_version": 1,
        "contract": {
            "declaration": _fixture_geography_declaration(),
            "algorithm": {
                "id": "assign_us_puma_ladder.population_weighted_overlap.v1",
                "kernel": "assign_us_puma_ladder",
                "operator": "assign_us_puma_ladder",
                "order": "before_gap_fill",
                "assign_tract": False,
            },
            "authorities": {
                "puma_ladder": {
                    "input_role": "puma_ladder",
                    "source_ref": "source:us_puma_ladder_2020",
                    "sha256": (
                        "39a2ab2abeab07a88362af7ab2940e0e1d50a297c919e4bbc6fb65bab51147d8"
                    ),
                },
                "congressional_district_vintage_crosswalk": {
                    "input_role": "congressional_district_vintage_crosswalk",
                    "source_ref": (
                        "source:us_congressional_district_vintage_crosswalk_117_to_119"
                    ),
                    "sha256": (
                        "c7cb040b1f57ca2ea2adcbfe60cc2b250ca23acbc4b640cd421e766fa54c1aec"
                    ),
                    "source_vintage_ref": "vintage:cd_117",
                    "source_vintage": "117th_congress",
                    "target_vintage_ref": "vintage:cd_119",
                    "target_vintage": "119th_congress",
                },
            },
            "seed": {
                "site": "legacy_puma_ladder",
                "stream": "geography_legacy",
                "value_source": "run_request.build_model_seed",
                "value": 0,
            },
        },
        "pre_assignment_household_order": {
            "column": "household_id",
            "codec": "int64_little_endian.v1",
            "row_count": len(native),
            "sha256": order_digest.hexdigest(),
        },
        "assigned_household_geography": {
            "columns": [
                "household_id",
                "puma",
                "congressional_district_geoid",
                "county_fips",
            ],
            "codec": "column_major_int64_little_endian.v1",
            "row_count": len(native),
            "sha256": geography_digest.hexdigest(),
        },
        "target_universe": {
            "district_count": 1,
            "geoids_sha256": "d" * 64,
        },
        "output": {
            "household_rows": len(native),
            "positive_congressional_district_rows": len(native),
            "unique_congressional_district_values": int(
                native["congressional_district_geoid"].nunique()
            ),
        },
        "summary": {"applied": True, "household_rows": len(native)},
        "gate": {
            "passed": True,
            "gates": {"us_puma_ladder": {"passed": True}},
        },
    }


def _fixture_stacked_sampling(
    sample_fraction: float,
) -> tuple[dict[str, object], dict[str, object]]:
    fraction_token = {
        0.01: "f001",
        0.04: "f004",
        0.10: "f010",
        0.25: "f025",
        1.00: "f100",
    }[sample_fraction]
    realized = {"asec": 2, "acs": 1}
    stack_manifest: dict[str, object] = {
        "version": 4,
        "sample_fraction": sample_fraction,
        "sample_seed": 578,
        "survey_samples": {
            channel: {
                "fraction": sample_fraction,
                "seed": 578,
                "realized_household_count": count,
            }
            for channel, count in realized.items()
        },
    }
    sampling = {
        "sample_fraction": sample_fraction,
        "fraction_token": fraction_token,
        "sample_seed": 578,
        "realized_households": realized,
        "stack_manifest_sha256": _json_sha256(stack_manifest),
    }
    return sampling, stack_manifest


def _write_ready_pool(
    tmp_path: Path,
    *,
    stacked: bool = False,
    sample_fraction: float = 1.0,
) -> Path:
    run_id = "fixture-publication"
    pool_path = tmp_path / "pool.h5"
    diagnostics_path = tmp_path / "pool.agreement.json"
    manifest_path = tmp_path / "pool.manifest.json"
    agreement_gate = {
        "passed": True,
        "gates": {
            "us_spine_agreement": {
                "passed": True,
                "failures": [],
                "details": {"fixture": True},
            }
        },
    }
    schema_version = US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION if stacked else 4
    pool_frame = _pool_frame_with_object_strings_on_every_entity(stacked=stacked)
    geography_assignment = (
        _fixture_geography_assignment(pool_frame.table("household"))
        if stacked
        else None
    )
    write_nullable_us_h5(
        pool_frame,
        pool_path,
        period=2024,
        artifact_kind=US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
        publication_run_id=run_id,
        materializer_version=(
            US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION if stacked else None
        ),
        root_attributes=(
            {
                CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR: (
                    geography_assignment["contract"]["authorities"][
                        "congressional_district_vintage_crosswalk"
                    ]["sha256"]
                ),
                CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR: (
                    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
                ),
            }
            if geography_assignment is not None
            else None
        ),
    )
    diagnostics = {
        "artifact_kind": (US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND),
        "schema_version": schema_version,
        "simulation_ready": True,
        "publication_run_id": run_id,
        "agreement_gate": agreement_gate,
    }
    if stacked:
        diagnostics.update(
            {
                "pipeline": "us-stacked-pool",
                "semantic_kind": "stacked_terminal_gates",
                "terminal_gates": agreement_gate,
            }
        )
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest = {
        "artifact_kind": US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
        "schema_version": schema_version,
        "status": "simulation_ready",
        "simulation_ready": True,
        "publication_run_id": run_id,
        "period": 2024,
        "operator_order": [
            "assemble",
            "clone",
            "impute",
            "derive",
            "seed",
            "simulate",
            "agreement",
        ],
        "stage_receipts": {
            stage: {"operator": stage}
            for stage in ("impute", "derive", "seed", "simulate")
        },
        "stage_checkpoints": {
            "artifact_kind": "populace_us_multispine_pool_checkpoint_provenance",
            "schema_version": 1,
            "materializer_version": 3 if not stacked else 5,
            "enabled": False,
            "agreement": {
                "source": "always_fresh",
                "cached": False,
                "terminal_verdict_persisted": False,
            },
        },
        "agreement_gate": agreement_gate,
        "provenance_counts": {"household": {"rows": 3}},
        "pool_h5": {
            "path": str(pool_path.resolve()),
            "sha256": _sha256(pool_path),
            "size_bytes": pool_path.stat().st_size,
            "artifact_kind": US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
            "publication_run_id": run_id,
        },
        "agreement_diagnostics": {
            "path": str(diagnostics_path.resolve()),
            "sha256": _sha256(diagnostics_path),
            "publication_run_id": run_id,
        },
    }
    if stacked:
        dag = _canonical_stacked_late_dag_receipt()
        assert geography_assignment is not None
        sampling, stack_manifest = _fixture_stacked_sampling(sample_fraction)
        transition_authority = (
            stacked_spine_module._late_producer_transition_authority_receipt(dag)
        )
        primary = next(
            row
            for row in dag["execution"]
            if row["producer"] == stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
        )
        config = primary["available_input_receipts"][
            "tax_unit.@primary_puf_execution_config"
        ]["binding"]
        worker_execution_authentication = (
            worker_identity_module.current_worker_execution_authentication_receipt(
                config["qrf"]["worker_execution"],
                manifest_schema_version=schema_version,
                execution_config_schema_version=config["schema_version"],
                boundary="stacked pool fixture worker",
            )
        )
        manifest.update(
            {
                "pipeline": "us-stacked-pool",
                "random_seed": 0,
                "sampling": sampling,
                "stack_manifest": stack_manifest,
                "geography_assignment": geography_assignment,
                "provenance_pins": {
                    role: {
                        "path": f"/fixture/{role}",
                        "expected_sha256": authority["sha256"],
                        "actual_sha256": authority["sha256"],
                        "size_bytes": 1,
                    }
                    for role, authority in geography_assignment["contract"][
                        "authorities"
                    ].items()
                },
                "late_producer_transition_authority_sha256": (
                    transition_authority["sha256"]
                ),
                "terminal_gates": agreement_gate,
                "operator_order": list(US_STACKED_POOL_OPERATOR_ORDER),
                "stage_receipts": {
                    "geography_assignment": geography_assignment,
                    "impute": {
                        "source_operator_chain": {
                            "late_dag_completion": dag["source_completion"],
                        },
                        "stacked_late_producer_dag": dag,
                        "stacked_post_puf_transfer": dag["post_puf_transfer"],
                    },
                },
                "worker_execution_authentication": (worker_execution_authentication),
            }
        )
        diagnostics["worker_execution_authentication"] = worker_execution_authentication
        diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
        manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
        manifest["pool_h5"]["materializer_version"] = (
            US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION
        )
    manifest_path.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return manifest_path


def _write_gate_failed_pool(tmp_path: Path) -> Path:
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    failed_gate = {
        "passed": False,
        "gates": {
            "us_spine_agreement": {
                "passed": False,
                "failures": ["fixture terminal failure"],
                "details": {"fixture": False},
            }
        },
    }
    manifest.update(
        {
            "status": "gate_failed",
            "simulation_ready": False,
            "agreement_gate": failed_gate,
            "terminal_gates": failed_gate,
        }
    )
    diagnostics.update(
        {
            "simulation_ready": False,
            "agreement_gate": failed_gate,
            "terminal_gates": failed_gate,
        }
    )
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _canonical_late_calibration_owner_receipt(
    spec: post_transfer_calibration_runtime.PostTransferCalibrationSpec,
) -> dict[str, object]:
    values = np.asarray(
        [10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    )
    weights = np.asarray([2.0, 3.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    entity_ids = np.arange(1, len(values) + 1)
    reference = np.asarray([True] * 5 + [False] * 5)
    recipient = ~reference
    constrained = spec.special_constraint != "none"
    result = post_transfer_calibration_runtime.calibrate_post_transfer_values(
        values,
        weights,
        entity_ids,
        spec=spec,
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
        allowed_carrier_rows=recipient if constrained else None,
        addition_candidate_rows=recipient if constrained else None,
    )
    calibration = result.receipt
    scope = calibration["scope"]
    constraint: dict[str, object] = {"constraint": spec.special_constraint}
    if spec.special_constraint == "adult_care_qualifying_one_per_tax_unit":
        constraint.update(
            {
                "qualifying_mutable_rows": scope["allowed_carrier_rows"],
                "one_per_empty_tax_unit_addition_candidates": scope[
                    "addition_candidate_rows"
                ],
            }
        )
    elif spec.special_constraint == "weeks_requires_positive_unemployment_compensation":
        constraint["positive_unemployment_mutable_rows"] = scope["allowed_carrier_rows"]
    owner: dict[str, object] = {
        "stage": "late_transfer",
        "reference_selection": "asec_origin_clone_0",
        "recipient_selection": "acs_origin_clone_0",
        "mutable_selection": "recipient_null_before_nonnull_after",
        "reference_rows": scope["reference_rows"],
        "recipient_rows": scope["recipient_rows"],
        "mutable_rows": scope["mutable_rows"],
        "constraint": constraint,
        "context_binding": {
            "scope": dict(scope),
            "weights_sha256": calibration["weights"]["sha256"],
            "live_output": {
                "reference_rows": int(reference.sum()),
                "recipient_rows": int(recipient.sum()),
                "reference_entity_ids_sha256": (
                    stacked_spine_module._post_transfer_entity_ids_sha256(
                        entity_ids[reference]
                    )
                ),
                "recipient_entity_ids_sha256": (
                    stacked_spine_module._post_transfer_entity_ids_sha256(
                        entity_ids[recipient]
                    )
                ),
                "reference_output_values_sha256": (
                    stacked_spine_module._post_transfer_float64_sha256(
                        result.values[reference],
                        boundary="synthetic reference calibration output",
                    )
                ),
                "recipient_output_values_sha256": (
                    stacked_spine_module._post_transfer_float64_sha256(
                        result.values[recipient],
                        boundary="synthetic recipient calibration output",
                    )
                ),
                "reference_weights_sha256": (
                    stacked_spine_module._post_transfer_float64_sha256(
                        weights[reference],
                        boundary="synthetic reference calibration weights",
                    )
                ),
                "recipient_weights_sha256": (
                    stacked_spine_module._post_transfer_float64_sha256(
                        weights[recipient],
                        boundary="synthetic recipient calibration weights",
                    )
                ),
            },
        },
        "calibration": calibration,
    }
    if spec.special_constraint == "adult_care_qualifying_one_per_tax_unit":
        owner["post_reconciliation"] = {"status": "verified_no_op"}
    return owner


def _canonical_pregnancy_structural_receipt() -> dict[str, object]:
    policy = acs_transfer_module.acs_transfer_execution_contract_identity(
        targets=("is_pregnant",),
        derive_schedule_d=False,
    )["structural_target_policies"]["is_pregnant"]
    return {
        "policy_sha256": policy["sha256"],
        "source_person_key": "person_source_id",
        "source_persons_checked": 1,
        "physical_rows_checked": 1,
        "clone_rows_checked": 0,
        "donor_rows_checked": 1,
        "qrf_draw_source_persons": 0,
        "qrf_draw_rows": 0,
        "qrf_fanout_rows": 0,
        "preexisting_value_fanout_rows": 0,
        "ineligible_rows_assigned_false": 0,
        "donor_preexisting_domain_violation_rows": 0,
        "recipient_preexisting_domain_violation_rows": 0,
        "preexisting_clone_disagreement_source_persons": 0,
        "inconsistent_eligibility_source_persons": 0,
        "maximum_clones_per_source_person": 1,
        "final_incomplete_rows": 0,
        "final_domain_violation_rows": 0,
        "final_clone_disagreement_source_persons": 0,
        "status": "verified",
    }


@cache
def _cached_canonical_stacked_late_dag_receipt(
    _fit_jobs: str | None,
    _predict_workers: str | None,
    _cpu_count: int | None,
) -> dict[str, object]:
    """Build a signed fixture receipt over the live canonical contracts."""

    schedule = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_SCHEDULE
    schedule_receipt = stacked_spine_module._json_ready(
        stacked_spine_module.us_late_producer_schedule_receipt()
    )
    source_order = [
        producer.removeprefix("source:")
        for producer in schedule.order
        if producer.startswith("source:")
    ]
    source_receipts = {
        operator: {
            "phase": "post_clone",
            "operator_order": [operator],
            "cps_source_evidence": None,
            "suboperators": [{"operator": operator}],
        }
        for operator in source_order
    }
    source_completion = {
        "phase": "post_clone",
        "operator_order": source_order,
        "cps_source_evidence": None,
        "suboperators": [
            {"operator": operator, "order_index": index}
            for index, operator in enumerate(source_order)
        ],
        "deferred_transfer_inputs": {
            "inputs": {
                column: {}
                for column in (
                    "bank_account_assets",
                    "bond_assets",
                    "stock_assets",
                )
            }
        },
    }
    late_specs = {
        spec.key: spec
        for spec in post_transfer_calibration_runtime.POST_TRANSFER_CALIBRATION_SPECS.values()
        if spec.stage == "late_transfer"
    }
    policy_sha256 = (
        post_transfer_calibration_runtime.post_transfer_calibration_policy_identity()[
            "sha256"
        ]
    )
    group_receipts: dict[str, object] = {}
    for group in stacked_spine_module.CANONICAL_US_LATE_TRANSFER_GROUPS:
        group_targets = {
            f"{group.entity}/{group.family}/{target}": {
                "authorized_null_rows": 0,
                "imputed_rows": 0,
                "unmodeled_rows": 0,
                "residual_null_rows": 0,
            }
            for target in group.targets
        }
        pregnancy_key = f"{group.entity}/{group.family}/is_pregnant"
        if pregnancy_key in group_targets:
            group_targets[pregnancy_key]["structural_policy"] = (
                _canonical_pregnancy_structural_receipt()
            )
        calibrated_keys = sorted(set(group_targets) & set(late_specs))
        for key in calibrated_keys:
            group_targets[key]["post_transfer_calibration"] = (
                _canonical_late_calibration_owner_receipt(late_specs[key])
            )
        group_receipts[group.name] = {
            "producer": group.name,
            "entity": group.entity,
            "family": group.family,
            "ordered_targets": list(group.targets),
            "targets": group_targets,
            "post_transfer_calibration": {
                "policy_sha256": policy_sha256,
                "target_count": len(calibrated_keys),
                "targets": calibrated_keys,
            },
        }
    group_by_name = {
        group.name: group
        for group in stacked_spine_module.CANONICAL_US_LATE_TRANSFER_GROUPS
    }
    canonical_family = {
        (entity, target): family
        for entity, families in (
            stacked_spine_module.CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE.items()
        )
        for family, targets in families.items()
        for target in targets
    }
    aggregate_targets = {
        f"{group.entity}/{canonical_family[(group.entity, target)]}/{target}": (
            group_receipts[group.name]["targets"][
                f"{group.entity}/{group.family}/{target}"
            ]
        )
        for group in stacked_spine_module.CANONICAL_US_LATE_TRANSFER_GROUPS
        for target in group.targets
    }
    transfer = {
        "authority": dict(stacked_spine_module.stacked_spine_authority_receipt()),
        "producer_schedule": schedule_receipt,
        "producer_execution_order": [
            producer
            for producer in schedule.order
            if producer != stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
        ],
        "groups": group_receipts,
        "targets": aggregate_targets,
        "completion": {
            "status": "complete",
            "group_count": 19,
            "target_count": 70,
            "residual_null_rows": 0,
        },
    }
    input_frame_sha256 = "1" * 64
    previous_sha256 = stacked_spine_module._late_execution_genesis_sha256(
        producer_schedule_sha256=schedule_receipt["payload_sha256"],
        input_frame_sha256=input_frame_sha256,
    )
    execution = []
    for index, producer_name in enumerate(schedule.order):
        contract = stacked_spine_module.CANONICAL_US_LATE_PRODUCER_REGISTRY[
            producer_name
        ]
        if contract.kind == "acs_earnings_universe":
            available = (
                stacked_spine_module._late_acs_earnings_universe_resource_receipts()
            )
        elif contract.kind == "primary_puf":
            available = stacked_spine_module.stacked_late_primary_resource_receipts(
                pd.DataFrame({"fixture_donor": [1.0]}),
                primary_qrf_checkpoint_identity_sha256="5" * 64,
                clone_attachment_fraction=1.0,
                clone_attachment_seed=578,
                seed=0,
                n_estimators=100,
                fit_records_enabled=True,
                tail_bound_diagnostics_enabled=True,
            )
        elif contract.kind == "post_clone_source":
            available = stacked_spine_module._late_source_resource_receipts(
                producer_name=producer_name,
            )
        elif contract.kind == "source_finalizer":
            available = {
                f"person.@source_receipt:{operator}": (
                    stacked_spine_module._late_available_input_receipt(
                        producer=producer_name,
                        entity="person",
                        column=f"@source_receipt:{operator}",
                        rows=1,
                        binding={
                            "resource_kind": "source_operator_receipt",
                            "schema_version": 1,
                            "source_operator": operator,
                            "source_receipt_sha256": (
                                stacked_spine_module._canonical_sha256(source_receipt)
                            ),
                        },
                    )
                )
                for operator, source_receipt in source_receipts.items()
            }
            available.update(
                stacked_spine_module._late_source_finalizer_resource_receipts()
            )
        elif contract.kind == "late_transfer":
            group = group_by_name[producer_name]
            available = stacked_spine_module._late_transfer_resource_receipts(
                group_name=group.name,
                entity=group.entity,
                family=group.family,
                targets=group.targets,
                seed=0,
                n_estimators=100,
                max_targets_per_fit=(
                    stacked_spine_module.DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
                ),
                target_bank=None,
            )
        else:
            available = {}
        declared_inputs = []
        for requirement in contract.inputs:
            alternatives = []
            for alternative in requirement.alternatives:
                physical_evidence = []
                for column in alternative:
                    is_virtual = (
                        column.column.startswith("@")
                        and column.column != "@resolved_weight"
                        and column.entity != "frame"
                    )
                    key = f"{column.entity}.{column.column}"
                    resource_receipt = available.get(key) if is_virtual else None
                    present = not is_virtual or resource_receipt is not None
                    physical_evidence.append(
                        {
                            "entity": column.entity,
                            "column": column.column,
                            "value_kind": column.value_kind,
                            "required_scope": requirement.required_scope,
                            "scope_rows": 1,
                            "missing_rows": 0 if present else 1,
                            "invalid_rows": 0,
                            "status": "present" if present else "absent",
                            "content_sha256": (
                                stacked_spine_module._canonical_sha256(resource_receipt)
                                if resource_receipt is not None
                                else "2" * 64
                            ),
                            **(
                                {"weight_kind": "household_weight"}
                                if column.column == "@resolved_weight"
                                else {}
                            ),
                        }
                    )
                alternatives.append(physical_evidence)
            evidence = {"alternatives": alternatives}
            evidence["sha256"] = stacked_spine_module._canonical_sha256(evidence)
            declared_inputs.append(
                {
                    "entity": requirement.entity,
                    "column": requirement.column,
                    "required_scope": requirement.required_scope,
                    "producing_stage": requirement.producing_stage,
                    "unfilled_rows": 0,
                    "invalid_rows": 0,
                    "evidence": evidence,
                }
            )
        output_surface = [
            {
                "entity": output.entity,
                "column": output.column,
                "coverage_scope": output.coverage_scope,
                "status": "present",
                "content_sha256": "3" * 64,
                **({} if output.entity == "frame" else {"scope_rows": 1}),
                **(
                    {"weight_kind": "household_weight"}
                    if output.column == "@resolved_weight"
                    else {}
                ),
            }
            for output in contract.outputs
        ]
        if contract.kind == "acs_earnings_universe":
            producer_receipt = {"fixture": "acs_earnings_universe"}
        elif contract.kind == "primary_puf":
            producer_receipt = {
                "primary_resource_receipts_sha256": (
                    stacked_spine_module._canonical_sha256(available)
                )
            }
        elif contract.kind == "post_clone_source":
            producer_receipt = source_receipts[producer_name.removeprefix("source:")]
        elif contract.kind == "source_finalizer":
            producer_receipt = source_completion
        elif contract.kind == "late_transfer":
            producer_receipt = group_receipts[group_by_name[producer_name].name]
        else:
            producer_receipt = {}
        for output in output_surface:
            if output["column"].startswith("@source_receipt:"):
                output["content_sha256"] = stacked_spine_module._canonical_sha256(
                    producer_receipt
                )
        row = {
            "execution_index": index,
            "producer": producer_name,
            "kind": contract.kind,
            "declared_inputs": declared_inputs,
            "declared_absence_receipts": {},
            "available_input_receipts": available,
            "input_surface_sha256": stacked_spine_module._canonical_sha256(
                declared_inputs
            ),
            "output_surface": output_surface,
            "output_surface_sha256": stacked_spine_module._canonical_sha256(
                output_surface
            ),
            "producer_receipt": producer_receipt,
            "producer_receipt_sha256": stacked_spine_module._canonical_sha256(
                producer_receipt
            ),
            "previous_execution_sha256": previous_sha256,
            "status": "complete",
        }
        row["sha256"] = stacked_spine_module._canonical_sha256(row)
        previous_sha256 = row["sha256"]
        execution.append(row)
    receipt = {
        "version": stacked_spine_module.US_LATE_PRODUCER_RECEIPT_SCHEMA_VERSION,
        "producer_schedule": schedule_receipt,
        "input_frame_sha256": input_frame_sha256,
        "output_frame_sha256": "4" * 64,
        "execution_chain_sha256": previous_sha256,
        "execution": execution,
        "source_completion": source_completion,
        "post_puf_transfer": transfer,
    }
    receipt["sha256"] = stacked_spine_module._canonical_sha256(receipt)
    stacked_spine_module.validate_stacked_late_producer_receipt(
        receipt,
        boundary="canonical stacked H5 fixture",
    )
    return receipt


def _canonical_stacked_late_dag_receipt() -> dict[str, object]:
    return deepcopy(
        _cached_canonical_stacked_late_dag_receipt(
            worker_identity_module.os.environ.get("POPULACE_FIT_N_JOBS"),
            worker_identity_module.os.environ.get("POPULACE_FIT_PREDICT_WORKERS"),
            worker_identity_module.os.cpu_count(),
        )
    )


def _rewrite_as_legacy_relocated_worker_pool(
    manifest_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Re-sign the tiny fixture with the frozen schema-9 worker binding."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dag = manifest["stage_receipts"]["impute"]["stacked_late_producer_dag"]
    primary = next(
        row
        for row in dag["execution"]
        if row["producer"] == stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
    )
    available = primary["available_input_receipts"]
    config_receipt = available["tax_unit.@primary_puf_execution_config"]
    config = config_receipt["binding"]
    recorded_executable = (
        "/Users/maxghenis/PolicyEngine/_worktrees/microcosm-c26-build/.venv/bin/python"
    )
    try:
        from microcosm.build.us_runtime import worker_identity
    except ImportError:
        live_legacy_worker = (
            stacked_spine_module._late_primary_qrf_worker_execution_binding()
        )
        semantic_identity = {
            "worker_module": {"name": live_legacy_worker["module"]},
            "argv_template": [
                "{python_interpreter}",
                *live_legacy_worker["argv_template"][1:],
            ],
            "environment": live_legacy_worker["environment"],
            "transitive_environment_code_sha256": "0" * 64,
        }
    else:
        semantic_identity = worker_identity.primary_qrf_worker_semantic_identity(
            uv_lock_sha256=(
                "27f47e385cfa35e2644a37410d1804b361ad9aee123577551c8421547bda65ee"
            )
        )
    legacy_environment = (
        worker_identity_module.legacy_primary_qrf_worker_execution_binding(
            semantic_identity=semantic_identity
        )["environment"]
    )
    recorded_worker = {
        "module": semantic_identity["worker_module"]["name"],
        "argv_template": [
            recorded_executable,
            *semantic_identity["argv_template"][1:],
        ],
        "interpreter": {
            "executable": recorded_executable,
            "resolved_executable": str(Path(sys.executable).resolve()),
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "version": list(sys.version_info[:3]),
        },
        "environment": legacy_environment,
    }
    config["schema_version"] = 4
    config["qrf"]["worker_execution"] = recorded_worker
    config_receipt["binding_sha256"] = _json_sha256(config)
    config_receipt_sha256 = _json_sha256(config_receipt)

    for declared_input in primary["declared_inputs"]:
        evidence = declared_input["evidence"]
        changed = False
        for alternative in evidence["alternatives"]:
            for column in alternative:
                if (
                    column["entity"] == "tax_unit"
                    and column["column"] == "@primary_puf_execution_config"
                ):
                    column["content_sha256"] = config_receipt_sha256
                    changed = True
        if changed:
            evidence["sha256"] = _json_sha256(
                {"alternatives": evidence["alternatives"]}
            )
    primary["producer_receipt"]["primary_resource_receipts_sha256"] = _json_sha256(
        available
    )

    schedule = dag["producer_schedule"]
    schedule["schema_version"] = 16
    schedule["execution_receipt_contract"]["version"] = 3
    schedule["execution_receipt_contract"]["transition_authority"]["version"] = 1
    schedule_payload = {
        key: value
        for key, value in schedule.items()
        if key
        not in {
            "payload_sha256",
            "producer_count",
            "source_producer_count",
            "transfer_group_count",
            "transfer_target_count",
            "status",
        }
    }
    schedule["payload_sha256"] = _json_sha256(schedule_payload)
    dag["post_puf_transfer"]["producer_schedule"] = json.loads(json.dumps(schedule))
    dag["post_puf_transfer"]["authority"] = (
        stacked_spine_module._legacy_stacked_authority_receipt()
    )
    dag["version"] = 3
    previous_sha256 = _json_sha256(
        {
            "receipt_schema_version": 3,
            "producer_schedule_sha256": dag["producer_schedule"]["payload_sha256"],
            "input_frame_sha256": dag["input_frame_sha256"],
        }
    )
    for row in dag["execution"]:
        row["input_surface_sha256"] = _json_sha256(row["declared_inputs"])
        row["output_surface_sha256"] = _json_sha256(row["output_surface"])
        row["producer_receipt_sha256"] = _json_sha256(row["producer_receipt"])
        row["previous_execution_sha256"] = previous_sha256
        row.pop("sha256", None)
        row["sha256"] = stacked_spine_module._canonical_sha256(row)
        previous_sha256 = row["sha256"]
    dag["execution_chain_sha256"] = previous_sha256
    dag.pop("sha256", None)
    dag["sha256"] = stacked_spine_module._canonical_sha256(dag)
    manifest["stage_receipts"]["impute"]["stacked_post_puf_transfer"] = json.loads(
        json.dumps(dag["post_puf_transfer"])
    )
    transition_authority = {
        "authority_id": stacked_spine_module.US_LATE_PRODUCER_TRANSITION_AUTHORITY_ID,
        "version": 1,
        "receipt_sha256": dag["sha256"],
        "producer_schedule_sha256": dag["producer_schedule"]["payload_sha256"],
        "input_frame_sha256": dag["input_frame_sha256"],
        "output_frame_sha256": dag["output_frame_sha256"],
        "execution_chain_sha256": dag["execution_chain_sha256"],
    }
    transition_authority["sha256"] = _json_sha256(transition_authority)
    manifest["late_producer_transition_authority_sha256"] = transition_authority[
        "sha256"
    ]
    manifest["schema_version"] = 9
    manifest.pop("worker_execution_authentication", None)

    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["schema_version"] = 9
    diagnostics.pop("worker_execution_authentication", None)
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return recorded_worker, semantic_identity


def _write_legacy_worker_attestation(
    manifest_path: Path,
    *,
    recorded_worker: dict[str, object],
    semantic_identity: dict[str, object],
    sealed_pool_h5_sha256: str | None = None,
) -> Path:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attestation = {
        "artifact_kind": "populace_us_worker_identity_compatibility_attestation",
        "schema_version": 1,
        "plan_signature": {
            "gate": "owner-authorization:c27-root-cause:2026-09-03",
            "plan_sha256": (
                "0a3409cfe1560d56a78ecc9acf012abaeb32621af278d745b674ebf1bee32cf6"
            ),
            "prompt_sha256": (
                "9c1e4508f24d0915c1f3a2942723d3c219c990679227c7d0a315295d5e76efa2"
            ),
            "checklist_sha256": (
                "5ee1f5fb40387cb690c2e85b32b6bd5abed78200f367c253e517a3917c417238"
            ),
            "evidence_sha256": (
                "85345eae623d0081354d746a118c9dc5ddaa89a641238e546d8c8e9f7aabbb44"
            ),
        },
        "purpose": "scoring_only",
        "sealed_manifest_sha256": _sha256(manifest_path),
        "sealed_pool_h5_sha256": (
            sealed_pool_h5_sha256 or manifest["pool_h5"]["sha256"]
        ),
        "campaign_tree_sha": "b8819b3f",
        "uv_lock_sha256": (
            "27f47e385cfa35e2644a37410d1804b361ad9aee123577551c8421547bda65ee"
        ),
        "installed_transitive_environment_code_sha256": semantic_identity[
            "transitive_environment_code_sha256"
        ],
        "recorded_worker_execution": recorded_worker,
        "semantic_identity": semantic_identity,
        "semantic_identity_sha256": _json_sha256(semantic_identity),
        "permitted_mismatches": [
            "argv_template[0]",
            "interpreter.executable",
        ],
    }
    attestation_path = manifest_path.with_name("worker-identity-attestation.json")
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    return attestation_path


def _resign_worker_manifest(
    manifest_path: Path,
    mutate_worker: Callable[[dict[str, object]], None],
    *,
    current_worker_receipt: bool,
) -> dict[str, object]:
    """Re-hash the tiny fixture after a deliberate worker-binding mutation."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    impute = manifest["stage_receipts"]["impute"]
    dag = impute["stacked_late_producer_dag"]
    primary = next(
        row
        for row in dag["execution"]
        if row["producer"] == stacked_spine_module.US_LATE_PRIMARY_PUF_STAGE
    )
    available = primary["available_input_receipts"]
    config_receipt = available["tax_unit.@primary_puf_execution_config"]
    config = config_receipt["binding"]
    worker = config["qrf"]["worker_execution"]
    mutate_worker(worker)
    config_receipt["binding_sha256"] = _json_sha256(config)
    config_receipt_sha256 = _json_sha256(config_receipt)
    for declared_input in primary["declared_inputs"]:
        evidence = declared_input["evidence"]
        changed = False
        for alternative in evidence["alternatives"]:
            for column in alternative:
                if (
                    column["entity"] == "tax_unit"
                    and column["column"] == "@primary_puf_execution_config"
                ):
                    column["content_sha256"] = config_receipt_sha256
                    changed = True
        if changed:
            evidence["sha256"] = _json_sha256(
                {"alternatives": evidence["alternatives"]}
            )
    primary["producer_receipt"]["primary_resource_receipts_sha256"] = _json_sha256(
        available
    )
    previous_sha256 = stacked_spine_module._late_execution_genesis_sha256(
        producer_schedule_sha256=dag["producer_schedule"]["payload_sha256"],
        input_frame_sha256=dag["input_frame_sha256"],
        receipt_schema_version=dag["version"],
    )
    for row in dag["execution"]:
        row["input_surface_sha256"] = _json_sha256(row["declared_inputs"])
        row["output_surface_sha256"] = _json_sha256(row["output_surface"])
        row["producer_receipt_sha256"] = _json_sha256(row["producer_receipt"])
        row["previous_execution_sha256"] = previous_sha256
        row.pop("sha256", None)
        row["sha256"] = stacked_spine_module._canonical_sha256(row)
        previous_sha256 = row["sha256"]
    dag["execution_chain_sha256"] = previous_sha256
    dag.pop("sha256", None)
    dag["sha256"] = stacked_spine_module._canonical_sha256(dag)
    transition_authority = (
        stacked_spine_module._late_producer_transition_authority_receipt(dag)
    )
    manifest["late_producer_transition_authority_sha256"] = transition_authority[
        "sha256"
    ]
    if current_worker_receipt:
        authentication = (
            worker_identity_module.current_worker_execution_authentication_receipt(
                worker,
                manifest_schema_version=manifest["schema_version"],
                execution_config_schema_version=config["schema_version"],
                boundary="mutated stacked fixture worker",
            )
        )
        manifest["worker_execution_authentication"] = authentication
        diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        diagnostics["worker_execution_authentication"] = authentication
        diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
        manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return worker


def test_ready_pool_loader_preserves_importance_weights_and_nullable_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    expected_manifest_sha256 = _sha256(manifest_path)
    original_read_bytes = Path.read_bytes
    manifest_reads = 0

    def replace_after_pinned_read(path: Path) -> bytes:
        nonlocal manifest_reads
        raw = original_read_bytes(path)
        if path == manifest_path:
            manifest_reads += 1
            replacement = json.loads(raw)
            replacement["publication_run_id"] = "replacement-publication"
            path.write_text(json.dumps(replacement), encoding="utf-8")
        return raw

    monkeypatch.setattr(Path, "read_bytes", replace_after_pinned_read)

    frame, manifest, authenticated_pool_h5 = load_simulation_ready_us_multispine_pool(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )

    weights = frame.weights_for("household")
    assert weights.kind is WeightKind.IMPORTANCE
    np.testing.assert_array_equal(weights.values, [2.0, 3.0, 5.0])
    assert frame.table("person")["nullable_input"].tolist() == [True, None, False]
    assert frame.n("household") == 3
    for entity in US_SCHEMA.entities:
        string_columns = _semantic_string_columns(frame.table(entity))
        assert string_columns
        assert all(
            frame.table(entity)[column].dtype == CANONICAL_STRING_DTYPE
            for column in string_columns
        )
    assert manifest["publication_run_id"] == "fixture-publication"
    assert authenticated_pool_h5.path == Path(manifest["pool_h5"]["path"])
    assert authenticated_pool_h5.sha256 == manifest["pool_h5"]["sha256"]
    assert authenticated_pool_h5.size_bytes == manifest["pool_h5"]["size_bytes"]
    assert authenticated_pool_h5.publication_run_id == "fixture-publication"
    assert authenticated_pool_h5.manifest_sha256 == expected_manifest_sha256
    assert "terminal_gates" not in manifest
    assert manifest_reads == 1
    assert json.loads(manifest_path.read_text())["publication_run_id"] == (
        "replacement-publication"
    )


def test_ready_legacy_pool_loader_accepts_pre_653_schema_four_envelope(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(written_manifest["agreement_diagnostics"]["path"])
    written_diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))

    frame, loaded_manifest, _authenticated_h5 = (
        load_simulation_ready_us_multispine_pool(manifest_path)
    )

    assert written_manifest["schema_version"] == 4
    assert written_diagnostics["schema_version"] == 4
    assert loaded_manifest["schema_version"] == 4
    assert "materializer_version" not in written_manifest["pool_h5"]
    assert "materializer_version" not in h5_io.read_nullable_us_h5_metadata(
        written_manifest["pool_h5"]["path"]
    )
    assert frame.n("household") == 3


def test_ready_legacy_pool_loader_rejects_current_stacked_envelope(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION
    diagnostics["schema_version"] = US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="ambiguous stacked envelope"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_pool_loader_rejects_a_false_h5_size_receipt(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pool_h5"]["size_bytes"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="pool_h5 size_bytes .* does not match"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_authenticated_pool_h5_copy_rejects_a_raced_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    _, _, authenticated_pool_h5 = load_simulation_ready_us_multispine_pool(
        manifest_path
    )
    replacement = bytearray(authenticated_pool_h5.path.read_bytes())
    replacement[0] ^= 1
    original_copy = h5_io._copy_file_bytes

    def replace_then_copy(source: Path, destination: Path) -> None:
        source.write_bytes(replacement)
        original_copy(source, destination)

    monkeypatch.setattr(h5_io, "_copy_file_bytes", replace_then_copy)
    destination = tmp_path / "audit" / "base_pool.h5"

    with pytest.raises(
        AuthenticatedPoolH5MismatchError,
        match="builder final local-audit copy.*copied bytes",
    ):
        authenticated_pool_h5.copy_verified_to(
            destination,
            consumer="builder final local-audit copy",
        )

    assert not destination.exists()
    assert not list(destination.parent.glob(".*.tmp"))


def test_ready_pool_loader_reconciles_manifest_and_h5_household_counts(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["provenance_counts"]["household"]["rows"] = 4
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="household row count 4.*H5 count 3"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_pool_loader_requires_explicitly_green_agreement_receipt(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["agreement_gate"]["passed"] = False
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["agreement_gate"]["passed"] = False
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="no passing agreement-gate verdict"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_pool_loader_binds_diagnostics_agreement_verdict(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["agreement_gate"]["gates"]["us_spine_agreement"]["details"] = {
        "fixture": False
    }
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="verdict does not match"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_stacked_pool_loader_binds_terminal_gate_aliases(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)

    frame, manifest, _ = load_simulation_ready_us_multispine_pool(manifest_path)

    assert manifest["terminal_gates"] == manifest["agreement_gate"]
    assert manifest["schema_version"] == US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION
    assert (
        manifest["pool_h5"]["materializer_version"]
        == US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION
    )
    assert (
        h5_io.read_nullable_us_h5_metadata(manifest["pool_h5"]["path"])[
            "materializer_version"
        ]
        == US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION
    )
    transition_authority = frame.metadata[
        stacked_spine_module.US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY
    ]
    assert (
        transition_authority["sha256"]
        == manifest["late_producer_transition_authority_sha256"]
    )
    assert (
        stacked_spine_module._json_ready(
            frame.metadata[stacked_spine_module.STACKED_SPINE_MANIFEST_KEY]
        )
        == (manifest["stack_manifest"])
    )


def test_ready_stacked_pool_loader_restores_sampled_rung_manifest(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(
        tmp_path,
        stacked=True,
        sample_fraction=0.25,
    )

    frame, manifest, _ = load_simulation_ready_us_multispine_pool(manifest_path)

    stack_manifest = frame.metadata[stacked_spine_module.STACKED_SPINE_MANIFEST_KEY]
    assert (
        stacked_spine_module._json_ready(stack_manifest) == manifest["stack_manifest"]
    )
    assert stack_manifest["version"] == 4
    assert stack_manifest["sample_fraction"] == 0.25


def test_ready_stacked_pool_loader_rejects_inconsistent_sampling_factor(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(
        tmp_path,
        stacked=True,
        sample_fraction=0.25,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sampling"]["sample_fraction"] = 0.10
    manifest["sampling"]["fraction_token"] = "f010"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="sample_fraction differs"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_stacked_pool_loader_rejects_inconsistent_arm_sampling(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(
        tmp_path,
        stacked=True,
        sample_fraction=0.25,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stack_manifest"]["survey_samples"]["asec"]["fraction"] = 0.10
    manifest["sampling"]["stack_manifest_sha256"] = _json_sha256(
        manifest["stack_manifest"]
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="asec survey-sample fraction differs"):
        load_simulation_ready_us_multispine_pool(manifest_path)


@pytest.mark.parametrize("sample_fraction", [True, 0, 0.25])
def test_ready_stacked_pool_loader_rejects_malformed_sampling_receipt(
    tmp_path: Path,
    sample_fraction: object,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sample_fraction == 0.25:
        del manifest["stack_manifest"]["sample_fraction"]
        manifest["sampling"]["stack_manifest_sha256"] = _json_sha256(
            manifest["stack_manifest"]
        )
    else:
        manifest["sampling"]["sample_fraction"] = sample_fraction
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="sample_fraction.*finite float"):
        load_simulation_ready_us_multispine_pool(manifest_path)


@pytest.mark.parametrize(
    ("mutation", "error_match"),
    (
        ("stack_seed_bool", "same non-negative integer"),
        ("arm_fraction_bool", "survey-sample fraction differs"),
        ("arm_seed_bool", "survey-sample seed differs"),
        ("top_realized_bool", "realized-household count is malformed"),
    ),
)
def test_ready_stacked_pool_loader_rejects_boolean_sampling_aliases(
    tmp_path: Path,
    mutation: str,
    error_match: str,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stack = manifest["stack_manifest"]
    samples = stack["survey_samples"]
    if mutation == "stack_seed_bool":
        manifest["sampling"]["sample_seed"] = 1
        stack["sample_seed"] = True
        for sample in samples.values():
            sample["seed"] = 1
    elif mutation == "arm_fraction_bool":
        samples["asec"]["fraction"] = True
    elif mutation == "arm_seed_bool":
        manifest["sampling"]["sample_seed"] = 1
        stack["sample_seed"] = 1
        samples["asec"]["seed"] = True
        samples["acs"]["seed"] = 1
    else:
        samples["asec"]["realized_household_count"] = 1
        manifest["sampling"]["realized_households"]["asec"] = True
    manifest["sampling"]["stack_manifest_sha256"] = _json_sha256(stack)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=error_match):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_stacked_pool_loader_binds_h5_cd_vintage_attrs(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool_path = Path(manifest["pool_h5"]["path"])
    with pd.HDFStore(pool_path, mode="a") as store:
        store.get_node("/")._v_attrs[CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR] = (
            "117th_congress"
        )
    manifest["pool_h5"]["sha256"] = _sha256(pool_path)
    manifest["pool_h5"]["size_bytes"] = pool_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="root attributes do not match"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_stacked_pool_loader_binds_h5_household_geography_digest(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool_path = Path(manifest["pool_h5"]["path"])
    with pd.HDFStore(pool_path, mode="a") as store:
        household = h5_io.read_frame_table(store, "household")
        household.loc[household.index[0], "congressional_district_geoid"] = 602
        store.put("household", household, format="fixed")
    manifest["pool_h5"]["sha256"] = _sha256(pool_path)
    manifest["pool_h5"]["size_bytes"] = pool_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="household geography differs"):
        load_simulation_ready_us_multispine_pool(manifest_path)


@pytest.mark.parametrize(
    "column",
    (
        support_source_id_column("household"),
        support_clone_index_column("household"),
    ),
)
def test_ready_stacked_pool_loader_requires_h5_clone_lineage(
    tmp_path: Path,
    column: str,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool_path = Path(manifest["pool_h5"]["path"])
    with pd.HDFStore(pool_path, mode="a") as store:
        household = h5_io.read_frame_table(store, "household").drop(columns=column)
        store.put("household", household, format="fixed")
    manifest["pool_h5"]["sha256"] = _sha256(pool_path)
    manifest["pool_h5"]["size_bytes"] = pool_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="missing household clone-lineage"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_stacked_pool_loader_rejects_divergent_clone_geography(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool_path = Path(manifest["pool_h5"]["path"])
    clone_column = support_clone_index_column("household")
    with pd.HDFStore(pool_path, mode="a") as store:
        household = h5_io.read_frame_table(store, "household")
        clone_row = household[clone_column].eq(1)
        assert int(clone_row.sum()) == 1
        household.loc[clone_row, "puma"] = "0600103"
        store.put("household", household, format="fixed")
    manifest["pool_h5"]["sha256"] = _sha256(pool_path)
    manifest["pool_h5"]["size_bytes"] = pool_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="cloned household rows disagree.*puma"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_scoring_pool_loader_authenticates_failed_stacked_terminal_receipt(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_gate_failed_pool(tmp_path)

    with pytest.raises(ValueError, match="not simulation-ready"):
        load_simulation_ready_us_multispine_pool(manifest_path)
    with pytest.raises(ValueError, match="not simulation-ready"):
        load_authenticated_us_multispine_pool_for_release(
            manifest_path,
            allow_terminal_gate_failure=False,
        )

    frame, loaded_manifest, authenticated_h5 = (
        load_authenticated_us_multispine_pool_for_scoring(manifest_path)
    )
    release_frame, release_manifest, release_h5 = (
        load_authenticated_us_multispine_pool_for_release(
            manifest_path,
            allow_terminal_gate_failure=True,
        )
    )

    assert frame.n("household") == 3
    assert loaded_manifest["status"] == "gate_failed"
    assert loaded_manifest["simulation_ready"] is False
    assert loaded_manifest["terminal_gates"]["passed"] is False
    assert authenticated_h5.sha256 == loaded_manifest["pool_h5"]["sha256"]
    assert release_frame.n("household") == frame.n("household")
    assert release_manifest == loaded_manifest
    assert release_h5 == authenticated_h5

    receipt = us_multispine_pool_release_receipt(
        loaded_manifest,
        authenticated_h5,
        allow_gate_failed_base_pool=True,
    )
    assert (
        receipt["content_identity_sha256"] == authenticated_h5.content_identity_sha256
    )
    assert receipt["status"] == "gate_failed"
    assert receipt["simulation_ready"] is False
    assert receipt["allow_gate_failed_base_pool"] is True
    assert (
        receipt["worker_execution_authentication"]
        == loaded_manifest["worker_execution_authentication"]
    )
    assert receipt["agreement_gate_reference"] == {
        "battery_status": "red",
        "passed": False,
        "gates_json_sha256": loaded_manifest["agreement_diagnostics"]["sha256"],
        "failure_count": 1,
        "failures": [
            {
                "gate": "us_spine_agreement",
                "message": "fixture terminal failure",
            }
        ],
        "verdict": loaded_manifest["agreement_gate"],
    }


def test_current_worker_authentication_ignores_audit_alias_relocation(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    original = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool_sha256 = original["pool_h5"]["sha256"]
    relocated = "/relocated/worktree/.venv/bin/python"

    worker = _resign_worker_manifest(
        manifest_path,
        lambda value: value.__setitem__(
            "audit_aliases",
            {
                "sys_executable": relocated,
                "sys_prefix": "/relocated/worktree/.venv",
                "argv_template_0": relocated,
            },
        ),
        current_worker_receipt=True,
    )
    frame, manifest, authenticated = load_simulation_ready_us_multispine_pool(
        manifest_path
    )

    assert frame.n("household") == 3
    assert authenticated.sha256 == pool_sha256
    assert (
        manifest["worker_execution_authentication"]["audit_aliases"]
        == worker["audit_aliases"]
    )


def test_scoring_loader_rejects_semantic_worker_change(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_gate_failed_pool(tmp_path)

    def mutate(worker: dict[str, object]) -> None:
        semantic = worker["semantic_identity"]
        semantic["interpreter"]["bytes_sha256"] = "0" * 64
        worker["semantic_identity_sha256"] = _json_sha256(semantic)

    _resign_worker_manifest(
        manifest_path,
        mutate,
        current_worker_receipt=True,
    )

    with pytest.raises(ValueError, match="semantic worker identity changed"):
        load_authenticated_us_multispine_pool_for_scoring(manifest_path)


def test_scoring_loader_accepts_legacy_worker_alias_relocation_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tables")
    monkeypatch.delenv("POPULACE_FIT_N_JOBS", raising=False)
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "18")
    manifest_path = _write_gate_failed_pool(tmp_path)
    recorded_worker, semantic_identity = _rewrite_as_legacy_relocated_worker_pool(
        manifest_path
    )
    manifest_sha256 = _sha256(manifest_path)
    attestation_path = _write_legacy_worker_attestation(
        manifest_path,
        recorded_worker=recorded_worker,
        semantic_identity=semantic_identity,
    )

    with pytest.raises(ValueError, match="requires.*compatibility attestation"):
        load_authenticated_us_multispine_pool_for_scoring(manifest_path)

    frame, manifest, authenticated = load_authenticated_us_multispine_pool_for_scoring(
        manifest_path,
        expected_manifest_sha256=manifest_sha256,
        worker_identity_attestation=attestation_path,
    )

    expected_authentication = {
        "manifest_schema_version": 9,
        "execution_config_schema_version": 4,
        "worker_execution_schema_version": 0,
        "semantic_identity_sha256": _json_sha256(semantic_identity),
        "audit_aliases": {
            "sys_executable": recorded_worker["interpreter"]["executable"],
            "argv_template_0": recorded_worker["argv_template"][0],
        },
        "compatibility_attestation_sha256": _sha256(attestation_path),
        "purpose": "scoring_only",
    }
    assert frame.n("household") == 3
    assert manifest["status"] == "gate_failed"
    assert manifest["simulation_ready"] is False
    assert authenticated.sha256 == manifest["pool_h5"]["sha256"]
    assert authenticated.manifest_sha256 == manifest_sha256
    assert manifest["worker_execution_authentication"] == expected_authentication
    assert authenticated.worker_execution_authentication == expected_authentication
    assert _sha256(manifest_path) == manifest_sha256
    assert (
        stacked_spine_module._json_ready(
            frame.metadata[stacked_spine_module.STACKED_SPINE_MANIFEST_KEY]
        )
        == manifest["stack_manifest"]
    )
    assert (
        frame.metadata[stacked_spine_module.US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY][
            "sha256"
        ]
        == manifest["late_producer_transition_authority_sha256"]
    )
    with pytest.raises(ValueError, match="Scoring-only.*release receipt"):
        us_multispine_pool_release_receipt(
            manifest,
            authenticated,
            allow_gate_failed_base_pool=True,
        )


def test_scoring_loader_requires_complete_schema_nine_stacked_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tables")
    monkeypatch.delenv("POPULACE_FIT_N_JOBS", raising=False)
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "18")
    manifest_path = _write_gate_failed_pool(tmp_path)
    recorded_worker, semantic_identity = _rewrite_as_legacy_relocated_worker_pool(
        manifest_path
    )
    complete_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authentication_calls: list[Path] = []

    def authenticate_without_recomputing_identity(
        attestation_path: str | Path,
        **_kwargs: object,
    ) -> worker_identity_module.LegacyWorkerIdentityAuthentication:
        path = Path(attestation_path)
        authentication_calls.append(path)
        return worker_identity_module.LegacyWorkerIdentityAuthentication(
            attestation_sha256=_sha256(path),
            campaign_tree_sha="b8819b3f",
            recorded_worker_execution=recorded_worker,
            semantic_identity=semantic_identity,
            semantic_identity_sha256=_json_sha256(semantic_identity),
        )

    monkeypatch.setattr(
        worker_identity_module,
        "authenticate_legacy_worker_identity_attestation",
        authenticate_without_recomputing_identity,
    )
    cases = (
        ("missing_pipeline", "pipeline"),
        ("wrong_pipeline", None),
        ("missing_operator_order", "operator_order"),
        ("missing_sampling", "sampling"),
        ("missing_stack_manifest", "stack_manifest"),
        ("missing_geography_assignment", "geography_assignment"),
        ("missing_stage_receipts", "stage_receipts"),
    )
    mismatches: list[str] = []
    for case, missing_field in cases:
        manifest = json.loads(json.dumps(complete_manifest))
        if missing_field is None:
            manifest["pipeline"] = "changed-stacked-pipeline"
            expected_missing: list[str] = []
        else:
            manifest.pop(missing_field)
            expected_missing = [missing_field]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        attestation_path = _write_legacy_worker_attestation(
            manifest_path,
            recorded_worker=recorded_worker,
            semantic_identity=semantic_identity,
        )

        try:
            load_authenticated_us_multispine_pool_for_scoring(
                manifest_path,
                worker_identity_attestation=attestation_path,
            )
        except ValueError as exc:
            message = str(exc)
            expected_fragments = (
                "ambiguous stacked envelope",
                f"missing={expected_missing!r}",
            )
            if any(fragment not in message for fragment in expected_fragments):
                mismatches.append(f"{case}: ValueError: {message}")
        else:
            mismatches.append(f"{case}: accepted")

    assert mismatches == [], "\n".join(mismatches)
    assert authentication_calls == []


def test_scoring_loader_rejects_mismatched_legacy_worker_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tables")
    monkeypatch.delenv("POPULACE_FIT_N_JOBS", raising=False)
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "18")
    manifest_path = _write_gate_failed_pool(tmp_path)
    recorded_worker, semantic_identity = _rewrite_as_legacy_relocated_worker_pool(
        manifest_path
    )
    attestation_path = _write_legacy_worker_attestation(
        manifest_path,
        recorded_worker=recorded_worker,
        semantic_identity=semantic_identity,
        sealed_pool_h5_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="attestation.*H5|H5.*attestation"):
        load_authenticated_us_multispine_pool_for_scoring(
            manifest_path,
            worker_identity_attestation=attestation_path,
        )


def test_legacy_manifest_cannot_supply_its_own_worker_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tables")
    monkeypatch.delenv("POPULACE_FIT_N_JOBS", raising=False)
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "18")
    manifest_path = _write_gate_failed_pool(tmp_path)
    recorded_worker, semantic_identity = _rewrite_as_legacy_relocated_worker_pool(
        manifest_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["worker_execution_authentication"] = {"forged": True}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    attestation_path = _write_legacy_worker_attestation(
        manifest_path,
        recorded_worker=recorded_worker,
        semantic_identity=semantic_identity,
    )

    with pytest.raises(ValueError, match="unauthenticated worker receipt"):
        load_authenticated_us_multispine_pool_for_scoring(
            manifest_path,
            worker_identity_attestation=attestation_path,
        )


def test_legacy_worker_authentication_cannot_be_stripped_for_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tables")
    monkeypatch.delenv("POPULACE_FIT_N_JOBS", raising=False)
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "18")
    manifest_path = _write_gate_failed_pool(tmp_path)
    recorded_worker, semantic_identity = _rewrite_as_legacy_relocated_worker_pool(
        manifest_path
    )
    attestation_path = _write_legacy_worker_attestation(
        manifest_path,
        recorded_worker=recorded_worker,
        semantic_identity=semantic_identity,
    )
    _, manifest, authenticated = load_authenticated_us_multispine_pool_for_scoring(
        manifest_path,
        worker_identity_attestation=attestation_path,
    )
    stripped_manifest = json.loads(json.dumps(manifest))
    stripped_manifest.pop("worker_execution_authentication")
    stripped_manifest["schema_version"] = 4
    stripped_authenticated = replace(
        authenticated,
        worker_execution_authentication=None,
        _legacy_worker_authentication=None,
    )

    with pytest.raises(ValueError, match="manifest payload changed"):
        us_multispine_pool_release_receipt(
            stripped_manifest,
            stripped_authenticated,
            allow_gate_failed_base_pool=True,
        )


@pytest.mark.parametrize(
    "changed_field",
    (
        "manifest",
        "purpose",
        "plan_signature",
        "campaign_tree",
        "lock",
        "transitive_environment_code",
        "recorded_worker",
        "semantic_digest",
        "permitted_mismatches",
    ),
)
def test_scoring_loader_rejects_changed_legacy_attestation_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_field: str,
) -> None:
    pytest.importorskip("tables")
    monkeypatch.delenv("POPULACE_FIT_N_JOBS", raising=False)
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "18")
    manifest_path = _write_gate_failed_pool(tmp_path)
    recorded_worker, semantic_identity = _rewrite_as_legacy_relocated_worker_pool(
        manifest_path
    )
    attestation_path = _write_legacy_worker_attestation(
        manifest_path,
        recorded_worker=recorded_worker,
        semantic_identity=semantic_identity,
    )
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if changed_field == "manifest":
        attestation["sealed_manifest_sha256"] = "0" * 64
    elif changed_field == "purpose":
        attestation["purpose"] = "release"
    elif changed_field == "plan_signature":
        attestation["plan_signature"]["gate"] = "unapproved"
    elif changed_field == "campaign_tree":
        attestation["campaign_tree_sha"] = "b8819b3f" + "1" * 32
    elif changed_field == "lock":
        attestation["uv_lock_sha256"] = "0" * 64
    elif changed_field == "transitive_environment_code":
        attestation["installed_transitive_environment_code_sha256"] = "0" * 64
    elif changed_field == "recorded_worker":
        attestation["recorded_worker_execution"]["module"] = "changed.worker"
    elif changed_field == "semantic_digest":
        attestation["semantic_identity_sha256"] = "0" * 64
    else:
        attestation["permitted_mismatches"] = ["interpreter.executable"]
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")

    with pytest.raises(ValueError, match="attest|mismatch|campaign"):
        load_authenticated_us_multispine_pool_for_scoring(
            manifest_path,
            worker_identity_attestation=attestation_path,
        )


@pytest.mark.parametrize("mismatch_count", (1, 3))
def test_scoring_loader_requires_exact_legacy_alias_mismatch_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch_count: int,
) -> None:
    pytest.importorskip("tables")
    monkeypatch.delenv("POPULACE_FIT_N_JOBS", raising=False)
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "18")
    manifest_path = _write_gate_failed_pool(tmp_path)
    _recorded_worker, semantic_identity = _rewrite_as_legacy_relocated_worker_pool(
        manifest_path
    )

    def mutate(worker: dict[str, object]) -> None:
        if mismatch_count == 1:
            worker["argv_template"][0] = str(Path(sys.executable))
        else:
            worker["interpreter"]["resolved_executable"] = "/changed/python"

    recorded_worker = _resign_worker_manifest(
        manifest_path,
        mutate,
        current_worker_receipt=False,
    )
    attestation_path = _write_legacy_worker_attestation(
        manifest_path,
        recorded_worker=recorded_worker,
        semantic_identity=semantic_identity,
    )

    with pytest.raises(ValueError, match="mismatch set changed"):
        load_authenticated_us_multispine_pool_for_scoring(
            manifest_path,
            worker_identity_attestation=attestation_path,
        )


def test_release_loader_ignores_legacy_worker_attestation_and_refuses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tables")
    monkeypatch.delenv("POPULACE_FIT_N_JOBS", raising=False)
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "18")
    manifest_path = _write_gate_failed_pool(tmp_path)
    recorded_worker, semantic_identity = _rewrite_as_legacy_relocated_worker_pool(
        manifest_path
    )
    _write_legacy_worker_attestation(
        manifest_path,
        recorded_worker=recorded_worker,
        semantic_identity=semantic_identity,
    )

    with pytest.raises(ValueError, match="unsupported artifact binding"):
        load_authenticated_us_multispine_pool_for_release(
            manifest_path,
            allow_terminal_gate_failure=True,
        )


def test_scoring_loader_refuses_ready_legacy_worker_even_with_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tables")
    monkeypatch.delenv("POPULACE_FIT_N_JOBS", raising=False)
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "18")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    recorded_worker, semantic_identity = _rewrite_as_legacy_relocated_worker_pool(
        manifest_path
    )
    attestation_path = _write_legacy_worker_attestation(
        manifest_path,
        recorded_worker=recorded_worker,
        semantic_identity=semantic_identity,
    )

    with pytest.raises(ValueError, match="unsupported artifact binding"):
        load_authenticated_us_multispine_pool_for_scoring(
            manifest_path,
            worker_identity_attestation=attestation_path,
        )


def test_denied_gate_failed_pool_is_available_only_for_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_gate_failed_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest["publication_run_id"]
    reason = "fixture pool is excluded from the certifiable line"
    reference = "microcosm#856; fixture-plan-gate"
    monkeypatch.setattr(
        h5_io,
        "DENIED_POOL_PUBLICATIONS",
        {
            run_id: h5_io.DeniedPoolPublication(
                manifest_sha256="0" * 64,
                pool_h5_sha256="1" * 64,
                content_identity_sha256="2" * 64,
                release_id="fixture-release",
                reason=reason,
                reference=reference,
            )
        },
        raising=False,
    )

    frame, loaded_manifest, _ = load_authenticated_us_multispine_pool_for_scoring(
        manifest_path
    )
    assert frame.n("household") == 3
    assert loaded_manifest["status"] == "gate_failed"

    refused_loaders = (
        (
            "manifest-only",
            lambda: h5_io.load_simulation_ready_us_multispine_pool_manifest(
                manifest_path
            ),
        ),
        (
            "simulation-ready",
            lambda: load_simulation_ready_us_multispine_pool(manifest_path),
        ),
        (
            "release-strict",
            lambda: load_authenticated_us_multispine_pool_for_release(
                manifest_path,
                allow_terminal_gate_failure=False,
            ),
        ),
        (
            "release-opt-in",
            lambda: load_authenticated_us_multispine_pool_for_release(
                manifest_path,
                allow_terminal_gate_failure=True,
            ),
        ),
    )
    for loader_name, loader in refused_loaders:
        with pytest.raises(ValueError) as error:
            loader()
        message = str(error.value)
        assert run_id in message, loader_name
        assert reason in message, loader_name
        assert reference in message, loader_name


def test_pool_deny_list_matches_manifest_sha256_without_matching_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "sha-only.manifest.json"
    observed_run_id = "different-observed-publication"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
                "publication_run_id": observed_run_id,
            }
        ),
        encoding="utf-8",
    )
    denied_run_id = "sha-matched-denied-publication"
    reason = "fixture manifest digest is excluded"
    reference = "microcosm#856; sha-fixture-plan-gate"
    monkeypatch.setattr(
        h5_io,
        "DENIED_POOL_PUBLICATIONS",
        {
            denied_run_id: h5_io.DeniedPoolPublication(
                manifest_sha256=_sha256(manifest_path),
                pool_h5_sha256="1" * 64,
                content_identity_sha256="2" * 64,
                release_id="fixture-sha-release",
                reason=reason,
                reference=reference,
            )
        },
        raising=False,
    )
    assert observed_run_id != denied_run_id

    with pytest.raises(ValueError) as error:
        h5_io.load_simulation_ready_us_multispine_pool_manifest(manifest_path)

    message = str(error.value)
    assert observed_run_id in message
    assert denied_run_id in message
    assert reason in message
    assert reference in message


def test_pool_deny_list_contains_candidate_26_identity() -> None:
    denied = h5_io.DENIED_POOL_PUBLICATIONS["2ab3f5a136bf4033be876bf150a6fbb4"]
    assert denied.manifest_sha256 == (
        "2a06fc2b1b73b006bb1bae7d13daeef813a4645c989374408eceaed0ef321cbd"
    )
    assert denied.release_id == (
        "populace-us-2024-stacked-f025-s578-asec42213-acs382903-"
        "20260831T162338Z-e14b24e8"
    )
    assert denied.pool_h5_sha256 == (
        "45f401735d7c5dc75da78be01bec4db7bf49ef074f69cecf39a1d5b1d77d7b9b"
    )
    assert denied.content_identity_sha256 == (
        "f5a5023bb9a74003d433abf04c796c96da0a34c6a7caff78b70fee421c4a7b2c"
    )
    assert "\n" not in denied.reason
    assert denied.reference == (
        "microcosm#856; plan gate 20260902-220844-plan-532dab66"
    )


def test_pool_deny_list_matches_pool_h5_sha256_without_other_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repackaged manifest around the same H5 bytes is still refused."""
    manifest_path = tmp_path / "h5-only.manifest.json"
    denied_h5 = "a" * 64
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
                "publication_run_id": "repackaged-publication",
                "pool_h5": {"sha256": denied_h5},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        h5_io,
        "DENIED_POOL_PUBLICATIONS",
        {
            "h5-matched-denied-publication": h5_io.DeniedPoolPublication(
                manifest_sha256="0" * 64,
                pool_h5_sha256=denied_h5,
                content_identity_sha256="2" * 64,
                release_id="fixture-h5-release",
                reason="fixture H5 bytes are excluded",
                reference="microcosm#856; h5-fixture-plan-gate",
            )
        },
        raising=False,
    )
    with pytest.raises(ValueError, match="pool H5 SHA-256"):
        h5_io.load_simulation_ready_us_multispine_pool_manifest(manifest_path)


def test_denied_pool_h5_digest_is_refused_on_generic_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stripping the sidecar and metadata row must not reopen a generic path."""
    denied_h5 = "b" * 64
    monkeypatch.setattr(
        h5_io,
        "DENIED_POOL_PUBLICATIONS",
        {
            "stripped-denied-publication": h5_io.DeniedPoolPublication(
                manifest_sha256="0" * 64,
                pool_h5_sha256=denied_h5,
                content_identity_sha256="2" * 64,
                release_id="fixture-stripped-release",
                reason="fixture bytes are excluded",
                reference="microcosm#856; stripped-fixture-plan-gate",
            )
        },
        raising=False,
    )
    h5_io.refuse_denied_pool_h5_digest("c" * 64, consumer="fixture consumer")
    with pytest.raises(ValueError) as error:
        h5_io.refuse_denied_pool_h5_digest(denied_h5, consumer="fixture consumer")
    message = str(error.value)
    assert "fixture consumer" in message
    assert "stripped-denied-publication" in message
    assert "even without pool identity metadata" in message


def test_scoring_evidence_of_a_denied_pool_cannot_become_a_release_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scoring exception must not launder a denied pool into release evidence."""
    pytest.importorskip("tables")
    manifest_path = _write_gate_failed_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest["publication_run_id"]
    monkeypatch.setattr(
        h5_io,
        "DENIED_POOL_PUBLICATIONS",
        {
            run_id: h5_io.DeniedPoolPublication(
                manifest_sha256="0" * 64,
                pool_h5_sha256="1" * 64,
                content_identity_sha256="2" * 64,
                release_id="fixture-release",
                reason="fixture pool is excluded from the certifiable line",
                reference="microcosm#856; fixture-plan-gate",
            )
        },
        raising=False,
    )
    _frame, loaded_manifest, authenticated_h5 = (
        load_authenticated_us_multispine_pool_for_scoring(manifest_path)
    )
    with pytest.raises(ValueError, match="cannot become a release receipt"):
        us_multispine_pool_release_receipt(
            loaded_manifest,
            authenticated_h5,
            allow_gate_failed_base_pool=True,
        )


def test_denied_pool_bytes_are_refused_by_every_generic_ingress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stripped of sidecar and metadata, a denied pool must fail at each ingress."""
    from microcosm.build.us_runtime import l0_refit_export

    stripped = tmp_path / "stripped.h5"
    stripped.write_bytes(b"denied pool bytes with no identity metadata")
    denied_h5 = hashlib.sha256(stripped.read_bytes()).hexdigest()
    monkeypatch.setattr(
        h5_io,
        "DENIED_POOL_PUBLICATIONS",
        {
            "stripped-denied-publication": h5_io.DeniedPoolPublication(
                manifest_sha256="0" * 64,
                pool_h5_sha256=denied_h5,
                content_identity_sha256="2" * 64,
                release_id="fixture-stripped-release",
                reason="fixture bytes are excluded",
                reference="microcosm#856; stripped-fixture-plan-gate",
            )
        },
        raising=False,
    )
    with pytest.raises(ValueError, match="even without pool identity metadata"):
        l0_refit_export.load_us_frame(stripped)
    with pytest.raises(ValueError, match="even without pool identity metadata"):
        h5_io.load_legacy_calibrated_us_h5(stripped)
    # The L0/refit export reads the base through load_us_frame first, so the
    # refusal precedes any use of the weights file, which need not exist.
    with pytest.raises(ValueError, match="even without pool identity metadata"):
        l0_refit_export.export_us_l0_refit_h5(
            base_h5=stripped,
            weights_npz=tmp_path / "absent.npz",
            output_h5=tmp_path / "out.h5",
        )


def test_h5_read_is_refused_when_the_file_changes_under_it(tmp_path: Path) -> None:
    """The refusal check and the read are bound by a post-read digest."""
    path = tmp_path / "base.h5"
    path.write_bytes(b"benign bytes")
    sha256 = h5_io.refuse_denied_pool_h5(path, consumer="fixture consumer")
    h5_io.assert_h5_unchanged(path, sha256, consumer="fixture consumer")
    path.write_bytes(b"replaced bytes")
    with pytest.raises(ValueError, match="changed while being read"):
        h5_io.assert_h5_unchanged(path, sha256, consumer="fixture consumer")


def _tool_module(relative: str, name: str):
    import importlib.util

    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_a_repackaged_denied_pool_is_refused_by_every_generic_ingress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping the metadata table and re-serializing changes the byte digest
    and removes every pool marker; the content identity survives and every
    generic ingress still refuses the file."""
    pytest.importorskip("tables")
    from microcosm.build.us_runtime import l0_refit_export

    manifest_path = _write_gate_failed_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_h5 = Path(manifest["pool_h5"]["path"])
    repackaged_dir = tmp_path / "repackaged"
    repackaged_dir.mkdir()
    repackaged = repackaged_dir / "plain.h5"  # no sidecar manifest beside it
    with pd.HDFStore(original_h5, mode="r") as source:
        with pd.HDFStore(repackaged, mode="w") as target:
            for key in source.keys():
                if key.lstrip("/") == "_populace_staging_metadata":
                    continue
                table = source.get(key)
                if key.lstrip("/") == "household" and isinstance(table, pd.DataFrame):
                    # Launder harder: drop optional provenance columns and
                    # relabel the household ids consistently.
                    table = table.drop(
                        columns=[
                            c
                            for c in (
                                "household_support_channel",
                                "household_support_clone_index",
                            )
                            if c in table.columns
                        ]
                    )
                    table = table.assign(household_id=table["household_id"] + 1000)
                target.put(key.lstrip("/"), table)

    assert h5_io.identify_us_multispine_pool_manifest(repackaged) is None
    assert _sha256(repackaged) != _sha256(original_h5)
    identity = h5_io.pool_h5_content_identity(repackaged)
    assert identity == h5_io.pool_h5_content_identity(original_h5)

    monkeypatch.setattr(
        h5_io,
        "DENIED_POOL_PUBLICATIONS",
        {
            manifest["publication_run_id"]: h5_io.DeniedPoolPublication(
                manifest_sha256=_sha256(manifest_path),
                pool_h5_sha256=_sha256(original_h5),
                content_identity_sha256=identity,
                release_id="fixture-release",
                reason="fixture pool is excluded from the certifiable line",
                reference="microcosm#856; fixture-plan-gate",
            )
        },
        raising=False,
    )
    fiscal = _tool_module("tools/build_us_fiscal_refresh_release.py", "fiscal_tool")
    puf_base = _tool_module("tools/build_us_puf_support_base.py", "puf_base_tool")
    acs_base = _tool_module(
        "tools/_legacy/build_us_acs_multispine_base.py", "acs_base_tool"
    )
    ingresses = (
        ("load_us_frame", lambda: l0_refit_export.load_us_frame(repackaged)),
        ("legacy loader", lambda: h5_io.load_legacy_calibrated_us_h5(repackaged)),
        ("fiscal builder", lambda: fiscal._load_frame(repackaged)),
        ("puf support base", lambda: puf_base._load_frame(repackaged)),
        ("legacy acs base", lambda: acs_base._load_base_frame(repackaged)),
    )
    for label, ingress in ingresses:
        with pytest.raises(ValueError, match="repackaging") as error:
            ingress()
        assert manifest["publication_run_id"] in str(error.value), label


def test_content_identity_ignores_ids_columns_and_order() -> None:
    """Version 2 hashes the sorted weight multiset with the count, nothing else."""
    weights = np.asarray([3.0, 1.5, 2.25], dtype=np.float64)
    identity = h5_io.content_identity_of_household_weights(weights)
    assert identity == h5_io.content_identity_of_household_weights(weights[::-1])
    assert identity != h5_io.content_identity_of_household_weights(weights[:2])
    assert identity != h5_io.content_identity_of_household_weights(weights + 1e-9)


def test_loaded_frame_of_a_denied_pool_is_refused_even_if_the_disk_file_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ABA: benign bytes on disk for both hashes, denied tables actually read."""
    pytest.importorskip("tables")
    manifest_path = _write_gate_failed_pool(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    denied_h5 = Path(manifest["pool_h5"]["path"])
    with pd.HDFStore(denied_h5, mode="r") as store:
        denied_tables = {
            key.lstrip("/"): store.get(key)
            for key in store.keys()
            if key.lstrip("/") != "_populace_staging_metadata"
        }
    identity = h5_io.content_identity_of_household_weights(
        denied_tables["household"]["household_weight"].to_numpy(dtype=np.float64)
    )
    monkeypatch.setattr(
        h5_io,
        "DENIED_POOL_PUBLICATIONS",
        {
            manifest["publication_run_id"]: h5_io.DeniedPoolPublication(
                manifest_sha256="0" * 64,
                pool_h5_sha256="1" * 64,
                content_identity_sha256=identity,
                release_id="fixture-release",
                reason="fixture pool is excluded from the certifiable line",
                reference="microcosm#856; fixture-plan-gate",
            )
        },
        raising=False,
    )
    benign = tmp_path / "benign.h5"
    with pd.HDFStore(benign, mode="w") as store:
        for key, table in denied_tables.items():
            if key == "household":
                table = table.assign(household_weight=table["household_weight"] * 0.5)
            store.put(key, table)
    # Both on-disk checks see the benign file; the read is swapped to the
    # denied tables, as a pathname replacement between check and read would do.
    monkeypatch.setattr(
        h5_io, "read_frame_table", lambda store, entity: denied_tables[entity].copy()
    )
    with pytest.raises(ValueError, match="loaded frame"):
        h5_io.load_legacy_calibrated_us_h5(benign)


def test_selection_manifest_provenance_is_required_for_v2_and_denied_for_any(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from microcosm.build.us_runtime import warm_start_selection as wss

    def write(version: int, source: dict) -> Path:
        path = tmp_path / f"selection-v{version}-{len(list(tmp_path.iterdir()))}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": version,
                    "join_key": list(wss.DEFAULT_SELECTION_JOIN_KEY),
                    "source": source,
                    "n_selected": 0,
                    "identities_sha256": wss._identities_digest(
                        tuple(wss.DEFAULT_SELECTION_JOIN_KEY), []
                    ),
                    "identities": [],
                }
            ),
            encoding="utf-8",
        )
        return path

    # A legacy v1 manifest with loose provenance still loads.
    assert (
        wss.load_selection_source_from_manifest(write(1, {"kind": "h5"})).n_identities
        == 0
    )
    # A v2 manifest must carry canonical byte and content provenance.
    with pytest.raises(ValueError, match="must record source.sha256"):
        wss.load_selection_source_from_manifest(write(2, {"kind": "h5"}))
    with pytest.raises(ValueError, match="must record source.sha256"):
        wss.load_selection_source_from_manifest(
            write(
                2,
                {"kind": "h5", "sha256": "A" * 64, "content_identity_sha256": "b" * 64},
            )
        )
    # The public writer cannot emit an unloadable v2 manifest.
    with pytest.raises(ValueError, match="must record source.sha256"):
        wss.write_selection_source_manifest(
            wss.SelectionSource(
                join_key=tuple(wss.DEFAULT_SELECTION_JOIN_KEY),
                identities=[],
                provenance={"kind": "h5"},
            ),
            tmp_path / "unwritable.json",
        )

    denied_h5 = "d" * 64
    monkeypatch.setattr(
        h5_io,
        "DENIED_POOL_PUBLICATIONS",
        {
            "denied-selection-publication": h5_io.DeniedPoolPublication(
                manifest_sha256="0" * 64,
                pool_h5_sha256=denied_h5,
                content_identity_sha256="2" * 64,
                release_id="fixture-release",
                reason="fixture pool is excluded",
                reference="microcosm#856; fixture-plan-gate",
            )
        },
        raising=False,
    )
    # Denied provenance is refused on either schema, byte or content, any case.
    with pytest.raises(ValueError, match="denied publication"):
        wss.load_selection_source_from_manifest(
            write(1, {"kind": "h5", "sha256": denied_h5.upper()})
        )
    with pytest.raises(ValueError, match="content identity"):
        wss.load_selection_source_from_manifest(
            write(
                2,
                {"kind": "h5", "sha256": "e" * 64, "content_identity_sha256": "2" * 64},
            )
        )


def test_fiscal_builder_binds_its_late_base_load_to_the_recorded_digest(
    tmp_path: Path,
) -> None:
    fiscal = _tool_module(
        "tools/build_us_fiscal_refresh_release.py", "fiscal_tool_bind"
    )
    base = tmp_path / "base.h5"
    base.write_bytes(b"benign base bytes")
    with pytest.raises(ValueError, match="not the base dataset whose identity"):
        fiscal._load_frame(base, expected_sha256="f" * 64)


def test_scoring_only_loader_is_not_reachable_from_release_paths() -> None:
    """The deny-list's only exception must stay a scoring-only ingress.

    Every non-test source file that names the scoring loader is listed here;
    a release builder, preflight, or pool tool that started calling it would
    reopen the release ingress the deny-list closes.
    """
    root = Path(__file__).resolve().parents[3]
    name = "load_authenticated_us_multispine_pool_for_scoring"
    allowed = {
        "packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py",
        "tools/score_us_release_head_to_head.py",
    }
    found = set()
    for folder in ("tools", "packages/microcosm-build/src"):
        for path in sorted((root / folder).rglob("*.py")):
            if name in path.read_text(encoding="utf-8"):
                found.add(path.relative_to(root).as_posix())
    assert found == allowed


def test_gate_failed_release_opt_in_is_rejected_for_a_ready_pool(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    _frame, manifest, authenticated_h5 = (
        load_authenticated_us_multispine_pool_for_release(
            manifest_path,
            allow_terminal_gate_failure=False,
        )
    )

    with pytest.raises(ValueError, match="override is valid only"):
        us_multispine_pool_release_receipt(
            manifest,
            authenticated_h5,
            allow_gate_failed_base_pool=True,
        )


@pytest.mark.parametrize(
    ("nested_passed", "failures"),
    ((False, []), (True, ["contradictory failure"])),
)
def test_release_receipt_rejects_incoherent_nested_gate_verdict(
    tmp_path: Path,
    nested_passed: bool,
    failures: list[str],
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    _frame, manifest, authenticated_h5 = (
        load_authenticated_us_multispine_pool_for_release(
            manifest_path,
            allow_terminal_gate_failure=False,
        )
    )
    first_gate = next(iter(manifest["agreement_gate"]["gates"].values()))
    first_gate["passed"] = nested_passed
    first_gate["failures"] = failures
    authenticated_h5 = replace(
        authenticated_h5,
        manifest_payload_sha256=_json_sha256(manifest),
    )

    with pytest.raises(ValueError, match="incoherent passed verdict"):
        us_multispine_pool_release_receipt(
            manifest,
            authenticated_h5,
            allow_gate_failed_base_pool=False,
        )


def test_pool_h5_identity_requires_its_missing_manifest_sidecar(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    pool_path = tmp_path / "pool.h5"
    manifest_path.unlink()

    assert identify_us_multispine_pool_manifest(pool_path) == manifest_path


def test_pool_sidecar_identity_requires_authentication_even_without_h5_stamp(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    pool_path = tmp_path / "pool.h5"
    write_nullable_us_h5(
        _pool_frame(),
        pool_path,
        period=2024,
        artifact_kind=LEGACY_NULLABLE_STAGING_ARTIFACT_KIND,
    )
    manifest_path = pool_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps({"artifact_kind": US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND}),
        encoding="utf-8",
    )

    assert identify_us_multispine_pool_manifest(pool_path) == manifest_path


def test_non_pool_h5_has_no_required_pool_manifest(tmp_path: Path) -> None:
    pytest.importorskip("tables")
    h5_path = tmp_path / "staging.h5"
    write_nullable_us_h5(
        _pool_frame(),
        h5_path,
        period=2024,
        artifact_kind=LEGACY_NULLABLE_STAGING_ARTIFACT_KIND,
    )

    assert identify_us_multispine_pool_manifest(h5_path) is None


def test_pool_sidecar_must_authenticate_the_requested_h5(tmp_path: Path) -> None:
    requested = tmp_path / "requested.h5"
    authenticated = tmp_path / "authenticated.h5"
    identity = AuthenticatedPoolH5(
        path=authenticated,
        sha256="a" * 64,
        size_bytes=1,
        publication_run_id="fixture-publication",
        manifest_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="authenticates a different H5"):
        require_authenticated_us_multispine_pool_h5(
            requested,
            identity,
            consumer="fixture consumer",
        )


@pytest.mark.parametrize(
    ("status", "simulation_ready"),
    (("simulation_ready", False), ("gate_failed", True), ("unknown", False)),
)
def test_scoring_pool_loader_rejects_incoherent_publication_status(
    tmp_path: Path,
    status: str,
    simulation_ready: bool,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest["simulation_ready"] = simulation_ready
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="not simulation-ready"):
        load_authenticated_us_multispine_pool_for_scoring(manifest_path)


@pytest.mark.parametrize("location", ("manifest", "h5"))
@pytest.mark.parametrize("value", (None, 1, True))
def test_ready_stacked_pool_loader_requires_exact_h5_materializer_binding(
    tmp_path: Path,
    location: str,
    value: object,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if location == "manifest":
        if value is None:
            del manifest["pool_h5"]["materializer_version"]
        else:
            manifest["pool_h5"]["materializer_version"] = value
    else:
        pool_path = Path(manifest["pool_h5"]["path"])
        with pd.HDFStore(pool_path, mode="a") as store:
            metadata = json.loads(str(store["_populace_staging_metadata"].iloc[0]))
            if value is None:
                del metadata["materializer_version"]
            else:
                metadata["materializer_version"] = value
            store.put(
                "_populace_staging_metadata",
                pd.Series([json.dumps(metadata, sort_keys=True)]),
                format="table",
            )
        manifest["pool_h5"]["sha256"] = _sha256(pool_path)
        manifest["pool_h5"]["size_bytes"] = pool_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="current H5 materializer version"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_stacked_pool_loader_requires_current_late_dag_proof(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["stage_receipts"]["impute"]["stacked_late_producer_dag"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="has no late-producer DAG receipt"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_stacked_pool_loader_rejects_schema_four_envelope(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 4
    diagnostics["schema_version"] = 4
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy envelope carries stacked-only"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_stacked_pool_cannot_be_downgraded_to_legacy(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 4
    manifest.pop("pipeline")
    manifest.pop("terminal_gates")
    diagnostics["schema_version"] = 4
    diagnostics.pop("pipeline")
    diagnostics.pop("semantic_kind")
    diagnostics.pop("terminal_gates")
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="legacy envelope carries stacked-only"):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_stacked_pool_cannot_be_stripped_into_legacy_shape(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 4
    for field in (
        "pipeline",
        "release_id",
        "sampling",
        "clone_attachment",
        "geography_assignment",
        "input_pins_digest",
        "late_producer_transition_authority_sha256",
        "stack_manifest",
        "terminal_gates",
        "operator_order",
        "stage_receipts",
        "worker_execution_authentication",
    ):
        manifest.pop(field, None)
    diagnostics["schema_version"] = 4
    for field in ("pipeline", "semantic_kind", "release_id", "terminal_gates"):
        diagnostics.pop(field, None)
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    # A lazy strip that misses the pool_h5 receipt's materializer_version is
    # caught earliest, by the stacked-only-marker refusal.
    with pytest.raises(ValueError, match="stacked-only field"):
        load_simulation_ready_us_multispine_pool(manifest_path)

    # Even a complete strip must still refuse at the canonical-envelope check.
    manifest["pool_h5"].pop("materializer_version", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical legacy envelope"):
        load_simulation_ready_us_multispine_pool(manifest_path)


@pytest.mark.parametrize("authority", [None, "0" * 64])
def test_ready_stacked_pool_loader_rejects_late_authority_mismatch(
    tmp_path: Path,
    authority: str | None,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if authority is None:
        del manifest["late_producer_transition_authority_sha256"]
    else:
        manifest["late_producer_transition_authority_sha256"] = authority
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="independently carried late-producer transition authority",
    ):
        load_simulation_ready_us_multispine_pool(manifest_path)


@pytest.mark.parametrize("document", ["manifest", "diagnostics"])
def test_ready_stacked_pool_loader_rejects_divergent_terminal_gate_alias(
    tmp_path: Path,
    document: str,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document == "manifest":
        manifest["terminal_gates"]["passed"] = False
    else:
        diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        diagnostics["terminal_gates"]["passed"] = False
        diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
        manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="terminal_gates do not match agreement_gate"):
        load_simulation_ready_us_multispine_pool(manifest_path)


@pytest.mark.parametrize("document", ["manifest", "diagnostics"])
def test_ready_stacked_pool_loader_requires_both_terminal_gate_aliases(
    tmp_path: Path,
    document: str,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document == "manifest":
        del manifest["terminal_gates"]
    else:
        diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        del diagnostics["terminal_gates"]
        diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
        manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="terminal_gates must be an object"):
        load_simulation_ready_us_multispine_pool(manifest_path)
