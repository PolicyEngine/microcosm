import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from populace.build.us_runtime.base_pool import (
    ACS_2024_1YR_SPINE,
    ASEC_PUF_SPINE,
    spine_column,
    with_optional_acs_spine,
)
from populace.build.us_runtime.puf_support import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)
from populace.frame import US_SCHEMA, EntitySchema, Frame, WeightKind, Weights

BASE_HOUSEHOLD_MASS = 400.0
ACS_RAW_HOUSEHOLD_MASS = 50.0
DEFAULT_ACS_SHARE = 0.5
QUARTER_ACS_SHARE = 0.25
MEMORY_BENCHMARK_ROWS = 10_000
MEMORY_BENCHMARK_COLUMNS_PER_SPINE = 32
MEMORY_AMPLIFICATION_LIMIT = 12
MEMORY_FIXED_ALLOWANCE_BYTES = 128 * 1024 * 1024


def _base_frame(*, support_metadata: bool = True) -> Frame:
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "person_household_id": [1, 1, 2],
                "person_tax_unit_id": [10, 10, 20],
                "person_spm_unit_id": [100, 100, 200],
                "person_family_id": [1_000, 1_000, 2_000],
                "person_marital_unit_id": [10_000, 10_000, 20_000],
                "base_only_income": [10.0, 20.0, 30.0],
            }
        ),
        "household": pd.DataFrame({"household_id": [1, 2]}),
        "tax_unit": pd.DataFrame({"tax_unit_id": [10, 20]}),
        "spm_unit": pd.DataFrame({"spm_unit_id": [100, 200]}),
        "family": pd.DataFrame({"family_id": [1_000, 2_000]}),
        "marital_unit": pd.DataFrame({"marital_unit_id": [10_000, 20_000]}),
    }
    if support_metadata:
        _add_support_metadata(tables, channel="asec")
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([100.0, 300.0]),
                WeightKind.DESIGN,
            )
        },
        pd.Series(["asec_2024", "asec_2024", "asec_2023"]),
    )


def _acs_frame(
    *,
    support_metadata: bool = False,
    conflicting_support_metadata: bool = False,
) -> Frame:
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_household_id": [1, 1],
                "person_tax_unit_id": [10, 10],
                "person_spm_unit_id": [100, 100],
                "person_family_id": [1_000, 1_000],
                "person_marital_unit_id": [10_000, 10_000],
                "acs_only_income": [40.0, 50.0],
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": [1],
                "puma": [101],
            }
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": [10]}),
        "spm_unit": pd.DataFrame({"spm_unit_id": [100]}),
        "family": pd.DataFrame({"family_id": [1_000]}),
        "marital_unit": pd.DataFrame({"marital_unit_id": [10_000]}),
    }
    if support_metadata or conflicting_support_metadata:
        _add_support_metadata(tables, channel=ACS_2024_1YR_SPINE)
    if conflicting_support_metadata:
        tables["person"][support_source_id_column("person")] += 10_000
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([ACS_RAW_HOUSEHOLD_MASS]),
                WeightKind.DESIGN,
            )
        },
    )


def _one_person_household_frame(
    weights: np.ndarray,
    *,
    extra_prefix: str | None = None,
    n_extra_columns: int = 0,
) -> Frame:
    ids = np.arange(1, len(weights) + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
        }
    )
    if extra_prefix is not None:
        for index in range(n_extra_columns):
            person[f"{extra_prefix}_{index}"] = ids.astype(np.float64)
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame({"household_id": ids}),
            "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
            "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
            "family": pd.DataFrame({"family_id": ids}),
            "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
        },
        US_SCHEMA,
        {"household": Weights(weights, WeightKind.DESIGN)},
    )


def _add_support_metadata(
    tables: dict[str, pd.DataFrame],
    *,
    channel: str,
) -> None:
    for entity, table in tables.items():
        entity_id = US_SCHEMA.entity_id_column(entity)
        table[support_source_id_column(entity)] = table[entity_id]
        table[support_channel_column(entity)] = channel
        table[support_clone_index_column(entity)] = 0


def _frame_digest(frame: Frame) -> str:
    payload = (
        [(entity, frame.table(entity)) for entity in frame.entities],
        [
            (
                entity,
                frame.weights_for(entity).values,
                frame.weights_for(entity).kind.value,
            )
            for entity in frame.weighted_entities
        ],
        frame.strata,
        frame.mass_log,
    )
    return hashlib.sha256(pickle.dumps(payload, protocol=5)).hexdigest()


def test__given_no_acs__then_base_object_and_bytes_are_unchanged() -> None:
    # Given
    base = _base_frame()
    digest_before = _frame_digest(base)

    # When
    result = with_optional_acs_spine(base, None)

    # Then
    assert result is base
    assert _frame_digest(result) == digest_before
    for entity in base.entities:
        assert spine_column(entity) not in base.table(entity)


def test__given_acs__then_every_entity_is_tagged_and_links_remain_valid() -> None:
    # Given
    base = _base_frame()
    acs = _acs_frame()

    # When
    result = with_optional_acs_spine(base, acs)

    # Then
    for entity in result.entities:
        assert set(result.table(entity)[spine_column(entity)]) == {
            ASEC_PUF_SPINE,
            ACS_2024_1YR_SPINE,
        }

    acs_people = result.table("person")[
        result.table("person")[spine_column("person")] == ACS_2024_1YR_SPINE
    ]
    assert acs_people["person_id"].tolist() == [4, 5]
    assert result.strata.iloc[-2:].tolist() == [
        ACS_2024_1YR_SPINE,
        ACS_2024_1YR_SPINE,
    ]
    for group in US_SCHEMA.group_entities:
        acs_group_ids = set(
            result.table(group).loc[
                result.table(group)[spine_column(group)] == ACS_2024_1YR_SPINE,
                US_SCHEMA.id_column(group),
            ]
        )
        assert set(acs_people[US_SCHEMA.membership_column(group)]) == acs_group_ids


def test__given_source_specific_columns__then_other_spine_receives_missing_values() -> (
    None
):
    # Given
    base = _base_frame()
    acs = _acs_frame()

    # When
    result = with_optional_acs_spine(base, acs)

    # Then
    person = result.table("person")
    base_people = person[person[spine_column("person")] == ASEC_PUF_SPINE]
    acs_people = person[person[spine_column("person")] == ACS_2024_1YR_SPINE]
    assert base_people["acs_only_income"].isna().all()
    assert acs_people["base_only_income"].isna().all()
    assert not base_people["acs_only_income"].eq(0).any()
    assert not acs_people["base_only_income"].eq(0).any()


def test__given_supported_base__then_acs_support_provenance_uses_source_ids() -> None:
    # Given
    base = _base_frame(support_metadata=True)
    acs = _acs_frame(support_metadata=True)
    original_acs_ids = {
        entity: acs.table(entity)[US_SCHEMA.entity_id_column(entity)].tolist()
        for entity in acs.entities
    }

    # When
    result = with_optional_acs_spine(base, acs)

    # Then
    for entity in result.entities:
        table = result.table(entity)
        acs_rows = table[table[spine_column(entity)] == ACS_2024_1YR_SPINE]
        assert (
            acs_rows[support_source_id_column(entity)].tolist()
            == original_acs_ids[entity]
        )
        assert set(acs_rows[support_channel_column(entity)]) == {ACS_2024_1YR_SPINE}
        assert set(acs_rows[support_clone_index_column(entity)]) == {0}


def test__given_conflicting_acs_support_metadata__then_validation_fails() -> None:
    # Given
    base = _base_frame(support_metadata=True)
    acs = _acs_frame(conflicting_support_metadata=True)
    original_source_ids = acs.table("person")[support_source_id_column("person")].copy()

    # When / Then
    with pytest.raises(ValueError, match="ACS support metadata conflicts"):
        with_optional_acs_spine(base, acs)
    pd.testing.assert_series_equal(
        acs.table("person")[support_source_id_column("person")],
        original_source_ids,
    )


@pytest.mark.parametrize(
    "acs_share,expected_base_mass,expected_acs_mass",
    [
        (DEFAULT_ACS_SHARE, 200.0, 200.0),
        (QUARTER_ACS_SHARE, 300.0, 100.0),
    ],
)
def test__given_acs_share__then_original_mass_is_allocated_and_recorded(
    acs_share: float,
    expected_base_mass: float,
    expected_acs_mass: float,
) -> None:
    # Given
    base = _base_frame()
    acs = _acs_frame()

    # When
    result = with_optional_acs_spine(base, acs, acs_share=acs_share)

    # Then
    household = result.table("household")
    weights = pd.Series(result.weights_for("household").values, index=household.index)
    base_mask = household[spine_column("household")] == ASEC_PUF_SPINE
    acs_mask = household[spine_column("household")] == ACS_2024_1YR_SPINE
    assert result.weights_for("household").total == pytest.approx(BASE_HOUSEHOLD_MASS)
    assert weights.loc[base_mask].sum() == pytest.approx(expected_base_mass)
    assert weights.loc[acs_mask].sum() == pytest.approx(expected_acs_mass)
    assert len(result.mass_log) == 2
    assert result.mass_log[0].new_total == pytest.approx(expected_base_mass)
    assert result.mass_log[1].new_total == pytest.approx(expected_acs_mass)
    assert "ASEC-by-PUF" in result.mass_log[0].reason
    assert "ACS" in result.mass_log[1].reason


def test__given_adversarial_weights__then_household_mass_is_exact() -> None:
    # Given
    rng = np.random.default_rng(2024)
    base = _one_person_household_frame(
        rng.uniform(100_000_000.0, 1_000_000_000_000.0, 4_097)
    )
    acs = _one_person_household_frame(rng.uniform(1.0, 1_000_000.0, 4_093))
    original_mass = base.weights_for("household").total

    # When
    result = with_optional_acs_spine(base, acs)

    # Then
    assert result.weights_for("household").total == original_mass


def test__given_rounding_boundary_weights__then_pool_assembly_succeeds() -> None:
    # Given
    base = _one_person_household_frame(
        np.asarray(
            [
                float.fromhex("0x1.1b6fc8a1fb5d7p+16"),
                float.fromhex("0x1.5fd6d9b527953p+13"),
            ]
        )
    )
    acs = _one_person_household_frame(
        np.asarray(
            [
                float.fromhex("0x1.d3e48ce1ae317p+15"),
                float.fromhex("0x1.8a6db6475255fp+14"),
            ]
        )
    )
    original_mass = base.weights_for("household").total

    # When
    result = with_optional_acs_spine(base, acs)

    # Then
    assert result.weights_for("household").total == original_mass


def test__given_representative_wide_frames__then_peak_memory_is_bounded() -> None:
    # Given
    test_file = str(Path(__file__).resolve())
    benchmark = f"""
import json
import resource
import runpy
import sys

import numpy as np

namespace = runpy.run_path({test_file!r})
make_frame = namespace["_one_person_household_frame"]
combine = namespace["with_optional_acs_spine"]
rows = {MEMORY_BENCHMARK_ROWS}
columns = {MEMORY_BENCHMARK_COLUMNS_PER_SPINE}

base = make_frame(
    np.full(rows, 100.0),
    extra_prefix="base_only",
    n_extra_columns=columns,
)
acs = make_frame(
    np.full(rows, 50.0),
    extra_prefix="acs_only",
    n_extra_columns=columns,
)
input_bytes = sum(
    int(frame.table(entity).memory_usage(index=True, deep=True).sum())
    for frame in (base, acs)
    for entity in frame.entities
)

def peak_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)

before = peak_bytes()
result = combine(base, acs)
after = peak_bytes()
print(json.dumps({{
    "input_bytes": input_bytes,
    "peak_growth_bytes": max(0, after - before),
    "n_person": result.n("person"),
}}))
"""

    # When
    completed = subprocess.run(
        [sys.executable, "-c", benchmark],
        check=True,
        capture_output=True,
        text=True,
    )
    measurement = json.loads(completed.stdout.strip().splitlines()[-1])

    # Then
    assert measurement["n_person"] == 2 * MEMORY_BENCHMARK_ROWS
    assert measurement["peak_growth_bytes"] <= (
        MEMORY_AMPLIFICATION_LIMIT * measurement["input_bytes"]
        + MEMORY_FIXED_ALLOWANCE_BYTES
    )


@pytest.mark.parametrize(
    "invalid_share",
    [0.0, 1.0, -0.1, np.nan, np.inf, True, "half"],
)
def test__given_invalid_acs_share__then_validation_fails(invalid_share: object) -> None:
    # Given
    base = _base_frame()
    acs = _acs_frame()

    # When / Then
    with pytest.raises(ValueError, match="acs_share"):
        with_optional_acs_spine(base, acs, acs_share=invalid_share)


def test__given_non_us_frame__then_schema_validation_fails() -> None:
    # Given
    schema = EntitySchema(group_entities=("household",))
    non_us = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": [1],
                    "person_household_id": [1],
                }
            ),
            "household": pd.DataFrame({"household_id": [1]}),
        },
        schema,
        {"household": Weights(np.asarray([1.0]), WeightKind.DESIGN)},
    )

    # When / Then
    with pytest.raises(ValueError, match="US entity schema"):
        with_optional_acs_spine(non_us, _acs_frame())
