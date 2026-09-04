from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.acs_transfer as acs_transfer_runtime
import microcosm.build.us_runtime.h5_io as h5_io
import microcosm.build.us_runtime.immigration as immigration_runtime
import microcosm.build.us_runtime.post_transfer_calibration as post_transfer_calibration_runtime
import microcosm.build.us_runtime.stacked_spine as stacked_spine_module
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
    BASE_ASEC_SUPPORT_CHANNEL,
    spine_assembly_manifest,
    spine_provenance_counts,
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


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


def _pool_frame_with_object_strings_on_every_entity() -> Frame:
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
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _stacked_pool_frame_with_live_immigration(
    *,
    include_zero_weight_clone: bool = False,
) -> tuple[Frame, dict[str, object]]:
    """Build a persisted-shape final stack and its real immigration receipt."""

    controls = immigration_runtime.us_immigration_controls()
    positive_draws = tuple(draw for draw in controls.humanitarian if draw.target > 0)
    acs_evidence = {
        "paroled_one_year:afghanistan": (200, 2021, "NONE", "UNDOCUMENTED"),
        "paroled_one_year:ukraine": (164, 2022, "NONE", "UNDOCUMENTED"),
        "paroled_one_year:nicaragua": (315, 2023, "NONE", "UNDOCUMENTED"),
        "paroled_one_year:venezuela": (373, 2022, "NONE", "UNDOCUMENTED"),
        "refugee": (
            412,
            2022,
            "OTHER_NON_CITIZEN",
            "LEGAL_PERMANENT_RESIDENT",
        ),
        "asylee": (
            207,
            2020,
            "OTHER_NON_CITIZEN",
            "LEGAL_PERMANENT_RESIDENT",
        ),
        "tps:venezuela": (373, 2015, "NONE", "UNDOCUMENTED"),
        "tps:el_salvador": (312, 2001, "NONE", "UNDOCUMENTED"),
        "tps:honduras": (314, 1998, "NONE", "UNDOCUMENTED"),
        "tps:nicaragua": (315, 1998, "NONE", "UNDOCUMENTED"),
        "tps:nepal": (229, 2015, "NONE", "UNDOCUMENTED"),
        "tps:other_designated": (248, 2020, "NONE", "UNDOCUMENTED"),
    }
    expected_labels = {draw.label for draw in positive_draws}
    if set(acs_evidence) != expected_labels:
        raise AssertionError(
            "Stacked H5 immigration evidence must exactly cover the positive "
            "canonical humanitarian draws."
        )

    row_count = 1 + len(positive_draws) + int(include_zero_weight_clone)
    ids = np.arange(1, row_count + 1, dtype=np.int64)
    channels = np.asarray(
        [BASE_ASEC_SUPPORT_CHANNEL]
        + [stacked_spine_module.ACS_STACKED_SUPPORT_CHANNEL] * len(positive_draws)
        + ([BASE_ASEC_SUPPORT_CHANNEL] if include_zero_weight_clone else []),
        dtype=object,
    )
    clone_indices = np.zeros(row_count, dtype=np.int64)
    source_ids = ids.copy()
    if include_zero_weight_clone:
        clone_indices[-1] = 1
        source_ids[-1] = ids[0]

    person_rows: list[dict[str, object]] = [
        {
            "PRCITSHP": 1,
            "PENATVTY": 57,
            "PEINUSYR": 0,
            "CIT": np.nan,
            "POBP": np.nan,
            "YOEP": np.nan,
            "A_AGE": 50,
            "age": 50,
            "ssn_card_type": "CITIZEN",
            "immigration_status_str": "CITIZEN",
        }
    ]
    for draw in positive_draws:
        birth_country, arrival_year, ssn_card_type, immigration_status = acs_evidence[
            draw.label
        ]
        person_rows.append(
            {
                "PRCITSHP": np.nan,
                "PENATVTY": np.nan,
                "PEINUSYR": np.nan,
                "CIT": 5,
                "POBP": birth_country,
                "YOEP": arrival_year,
                "A_AGE": np.nan,
                "age": 50,
                "ssn_card_type": ssn_card_type,
                "immigration_status_str": immigration_status,
            }
        )
    if include_zero_weight_clone:
        person_rows.append(dict(person_rows[0]))

    person = pd.DataFrame(person_rows)
    person.insert(0, "PERIDNUM", pd.Series([f"person-{value}" for value in ids]))
    person.insert(1, "person_id", ids)
    for entity in US_SCHEMA.group_entities:
        person[US_SCHEMA.membership_column(entity)] = ids
    person["nullable_input"] = np.resize(
        np.asarray([True, None, False], dtype=object),
        row_count,
    )
    person["is_incapable_of_self_care"] = True
    person["tax_unit_role_input"] = "DEPENDENT"
    person[support_channel_column("person")] = channels
    person[support_source_id_column("person")] = source_ids
    person[support_clone_index_column("person")] = clone_indices

    household_weights = np.asarray(
        [1.0]
        + [float(draw.target) for draw in positive_draws]
        + ([0.0] if include_zero_weight_clone else []),
        dtype=np.float64,
    )
    mutable = channels == stacked_spine_module.ACS_STACKED_SUPPORT_CHANNEL
    reconciled_person, reconciliation = (
        immigration_runtime.reconcile_us_immigration_humanitarian_transfer(
            person,
            weights=household_weights,
            mutable_rows=mutable,
            seed=0,
            time_period=2024,
            controls=controls,
        )
    )

    tables = {"person": reconciled_person}
    for entity in US_SCHEMA.group_entities:
        table = pd.DataFrame({US_SCHEMA.id_column(entity): ids})
        table.insert(
            0,
            f"{entity}_source_label",
            pd.Series([f"{entity}-{value}" for value in ids], dtype=object),
        )
        table[support_channel_column(entity)] = channels
        table[support_source_id_column(entity)] = source_ids
        table[support_clone_index_column(entity)] = clone_indices
        tables[entity] = table

    household = tables["household"]
    household["puma"] = "0600101"
    household["congressional_district_geoid"] = np.full(
        row_count,
        601,
        dtype=np.int64,
    )
    household["county_fips"] = "06001"

    for index, spec in enumerate(
        post_transfer_calibration_runtime.POST_TRANSFER_CALIBRATION_SPECS.values()
    ):
        tables[spec.entity][spec.target] = (
            10.0 + index + np.arange(row_count, dtype=np.float64)
        )

    assembly_tables = {
        entity: table.loc[table[support_clone_index_column(entity)].eq(0)]
        for entity, table in tables.items()
    }
    metadata = spine_assembly_manifest(
        assembly_tables,
        channels=(
            BASE_ASEC_SUPPORT_CHANNEL,
            stacked_spine_module.ACS_STACKED_SUPPORT_CHANNEL,
        ),
    )
    frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                household_weights,
                WeightKind.IMPORTANCE,
            )
        },
        metadata=metadata,
    )
    return frame, reconciliation


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
    include_zero_weight_clone: bool = False,
    sample_fraction: float = 1.0,
) -> Path:
    run_id = "fixture-publication"
    pool_path = tmp_path / "pool.h5"
    diagnostics_path = tmp_path / "pool.agreement.json"
    manifest_path = tmp_path / "pool.manifest.json"
    gate_names = (
        (
            "us_stacked_completeness",
            "us_by_origin_battery",
            "immigration_composition",
        )
        if stacked
        else ("us_spine_agreement",)
    )
    agreement_gate = {
        "passed": True,
        "gates": {
            name: {
                "passed": True,
                "failures": [],
                "details": {"fixture": True},
            }
            for name in gate_names
        },
    }
    if include_zero_weight_clone and not stacked:
        raise ValueError("Only the stacked H5 fixture may include a clone row.")
    schema_version = US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION if stacked else 4
    if stacked:
        pool_frame, immigration_reconciliation = (
            _stacked_pool_frame_with_live_immigration(
                include_zero_weight_clone=include_zero_weight_clone,
            )
        )
        dag = _canonical_stacked_late_dag_receipt(
            pool_frame,
            immigration_reconciliation=immigration_reconciliation,
        )
        provenance_counts = spine_provenance_counts(
            pool_frame,
            boundary="stacked H5 fixture provenance counts",
        )
    else:
        pool_frame = _pool_frame_with_object_strings_on_every_entity()
        dag = None
        provenance_counts = {"household": {"rows": pool_frame.n("household")}}
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
        "provenance_counts": provenance_counts,
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
        assert dag is not None
        assert geography_assignment is not None
        sampling, stack_manifest = _fixture_stacked_sampling(sample_fraction)
        transition_authority = (
            stacked_spine_module._late_producer_transition_authority_receipt(dag)
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
            }
        )
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
    frame: Frame,
    spec: post_transfer_calibration_runtime.PostTransferCalibrationSpec,
) -> dict[str, object]:
    table = frame.table(spec.entity)
    channel = table[support_channel_column(spec.entity)].astype(str)
    clone_index = pd.to_numeric(
        table[support_clone_index_column(spec.entity)],
        errors="raise",
    )
    reference = (channel.eq(BASE_ASEC_SUPPORT_CHANNEL) & clone_index.eq(0)).to_numpy(
        dtype=bool
    )
    recipient = (
        channel.eq(stacked_spine_module.ACS_STACKED_SUPPORT_CHANNEL) & clone_index.eq(0)
    ).to_numpy(dtype=bool)
    weights = np.asarray(
        frame.resolve_weights(spec.entity).values,
        dtype=np.float64,
    )
    entity_ids = table[frame.schema.entity_id_column(spec.entity)].to_numpy(copy=False)
    constrained = spec.special_constraint != "none"
    application = post_transfer_calibration_runtime.apply_post_transfer_calibration(
        frame,
        entity=spec.entity,
        family=spec.family,
        target=spec.target,
        reference_rows=reference,
        recipient_rows=recipient,
        mutable_rows=recipient,
        allowed_carrier_rows=recipient if constrained else None,
        addition_candidate_rows=recipient if constrained else None,
    )
    result_values = application.frame.table(spec.entity)[spec.target].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    table[spec.target] = result_values
    calibration = application.receipt
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
                        result_values[reference],
                        boundary="synthetic reference calibration output",
                    )
                ),
                "recipient_output_values_sha256": (
                    stacked_spine_module._post_transfer_float64_sha256(
                        result_values[recipient],
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
    policy = acs_transfer_runtime.acs_transfer_execution_contract_identity(
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


def _canonical_stacked_late_dag_receipt(
    frame: Frame,
    *,
    immigration_reconciliation: dict[str, object],
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
    immigration_reconciliation = stacked_spine_module._json_ready(
        immigration_reconciliation
    )
    immigration_recipient_rows = int(immigration_reconciliation["mutable_rows"])
    immigration_producer_rows = int(immigration_reconciliation["immutable_rows"])
    group_receipts: dict[str, object] = {}
    for group in stacked_spine_module.CANONICAL_US_LATE_TRANSFER_GROUPS:
        is_immigration_group = (
            group.entity == "person" and group.family == "source_operator_immigration"
        )
        group_targets = {
            f"{group.entity}/{group.family}/{target}": {
                **(
                    {
                        "producer_roles": ["asec_source"],
                        "producer_rows": immigration_producer_rows,
                    }
                    if is_immigration_group
                    else {}
                ),
                "authorized_null_rows": (
                    immigration_recipient_rows if is_immigration_group else 0
                ),
                "imputed_rows": (
                    immigration_recipient_rows if is_immigration_group else 0
                ),
                "unmodeled_rows": 0,
                "residual_null_rows": 0,
            }
            for target in group.targets
        }
        if is_immigration_group:
            required_predictors, _optional_predictors = (
                stacked_spine_module._acs_pattern_predictor_authority(
                    entity=group.entity,
                    family_targets=group.targets,
                )
            )
            pattern = acs_transfer_runtime.AcsTransferPattern(
                name=acs_transfer_runtime._pattern_name(0, ()),
                observed_optional_predictors=(),
                predictors=required_predictors,
                seed=0,
                weight_kind="design",
                donor_rows=immigration_producer_rows,
                recipient_rows=immigration_recipient_rows,
                target_regimes=tuple(
                    (model_target, "positive_only")
                    for model_target in acs_transfer_runtime._model_target_names(
                        group.targets
                    )
                ),
            )
            for target in group.targets:
                key = f"{group.entity}/{group.family}/{target}"
                record = acs_transfer_runtime.AcsImputedInput(
                    column=target,
                    entity=group.entity,
                    family=group.family,
                    donor_spine="synthetic_stacked_h5_fixture",
                    donor_channel="asec",
                    predictors=pattern.predictors,
                    seed=pattern.seed,
                    weight_kind=pattern.weight_kind,
                    patterns=(pattern,),
                    imputed_recipient_rows=immigration_recipient_rows,
                    reconciliation=immigration_reconciliation,
                )
                group_targets[key]["qrf_pattern_evidence"] = (
                    stacked_spine_module._acs_imputed_pattern_evidence(record)
                )
                group_targets[key]["post_transfer_reconciliation"] = dict(
                    immigration_reconciliation
                )
        pregnancy_key = f"{group.entity}/{group.family}/is_pregnant"
        if pregnancy_key in group_targets:
            group_targets[pregnancy_key]["structural_policy"] = (
                _canonical_pregnancy_structural_receipt()
            )
        calibrated_keys = sorted(set(group_targets) & set(late_specs))
        for key in calibrated_keys:
            group_targets[key]["post_transfer_calibration"] = (
                _canonical_late_calibration_owner_receipt(frame, late_specs[key])
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

    positive_draws = tuple(
        draw
        for draw in immigration_runtime.us_immigration_controls().humanitarian
        if draw.target > 0
    )
    expected_rows = 1 + len(positive_draws)
    expected_weights = np.asarray(
        [1.0, *(float(draw.target) for draw in positive_draws)],
        dtype=np.float64,
    )
    assert all(frame.n(entity) == expected_rows for entity in US_SCHEMA.entities)
    np.testing.assert_array_equal(
        frame.weights_for("household").values,
        expected_weights,
    )
    person = frame.table("person")
    assert person.loc[0, "PRCITSHP"] == 1
    assert pd.isna(person.loc[0, "CIT"])
    assert person.loc[0, "immigration_status_str"] == "CITIZEN"
    assert person.loc[1:, "CIT"].eq(5).all()
    assert person.loc[1:, ["POBP", "YOEP"]].notna().all().all()
    for entity in US_SCHEMA.group_entities:
        np.testing.assert_array_equal(
            person[US_SCHEMA.membership_column(entity)],
            frame.table(entity)[US_SCHEMA.id_column(entity)],
        )
    for draw in positive_draws:
        emitted = immigration_runtime.us_immigration_humanitarian_draw_mask(
            frame,
            draw,
        )
        assert int(emitted.sum()) == 1
        assert float(frame.resolve_weights("person").values[emitted].sum()) == float(
            draw.target
        )
    for entity in US_SCHEMA.entities:
        assert manifest["provenance_counts"][entity] == {
            "rows": expected_rows,
            "by_source_channel": {
                BASE_ASEC_SUPPORT_CHANNEL: 1,
                stacked_spine_module.ACS_STACKED_SUPPORT_CHANNEL: len(positive_draws),
            },
            "by_clone_index": {"0": expected_rows},
            "by_source_channel_and_clone_index": {
                BASE_ASEC_SUPPORT_CHANNEL: {"0": 1},
                stacked_spine_module.ACS_STACKED_SUPPORT_CHANNEL: {
                    "0": len(positive_draws)
                },
            },
        }
    assert manifest["terminal_gates"] == manifest["agreement_gate"]
    assert manifest["schema_version"] == 10
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


def test_ready_stacked_pool_loader_replays_immigration_status_against_h5(
    tmp_path: Path,
) -> None:
    """An ordinary H5 re-hash cannot forge the sealed live immigration proof."""

    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pool_path = Path(manifest["pool_h5"]["path"])
    with pd.HDFStore(pool_path, mode="a") as store:
        person = h5_io.read_frame_table(store, "person")
        parole_rows = person["immigration_status_str"].eq("PAROLED_ONE_YEAR")
        assert parole_rows.any()
        person.loc[parole_rows.idxmax(), "immigration_status_str"] = (
            "LEGAL_PERMANENT_RESIDENT"
        )
        h5_io.put_frame_table(
            store,
            "person",
            person,
            preferred_format="fixed",
        )

    # Re-sign only the ordinary byte-level artifact envelope. The immutable
    # late-transition receipt still describes the untampered live population.
    manifest["pool_h5"]["sha256"] = _sha256(pool_path)
    manifest["pool_h5"]["size_bytes"] = pool_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="differs from the live ASEC/ACS weighted population",
    ):
        load_simulation_ready_us_multispine_pool(manifest_path)


def test_ready_stacked_pool_loader_requires_immigration_terminal_gate(
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")
    manifest_path = _write_ready_pool(tmp_path, stacked=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    diagnostics_path = Path(manifest["agreement_diagnostics"]["path"])
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    for payload in (manifest, diagnostics):
        payload["terminal_gates"]["gates"].pop("immigration_composition")
        payload["agreement_gate"]["gates"].pop("immigration_composition")
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    manifest["agreement_diagnostics"]["sha256"] = _sha256(diagnostics_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical terminal gate set"):
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
    manifest_path = _write_ready_pool(
        tmp_path,
        stacked=True,
        include_zero_weight_clone=True,
    )
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

    expected_rows = 1 + sum(
        draw.target > 0
        for draw in immigration_runtime.us_immigration_controls().humanitarian
    )
    assert frame.n("household") == expected_rows
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
    assert receipt["agreement_gate_reference"] == {
        "battery_status": "red",
        "passed": False,
        "gates_json_sha256": loaded_manifest["agreement_diagnostics"]["sha256"],
        "failure_count": 1,
        "failures": [
            {
                "gate": "us_stacked_completeness",
                "message": "fixture terminal failure",
            }
        ],
        "verdict": loaded_manifest["agreement_gate"],
    }


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
