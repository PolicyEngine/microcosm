"""Byte-level parity for the rules-engine graph kernel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import (
    US_SCHEMA,
    EntitySchema,
    ExportContract,
    Frame,
    VariableMetadata,
    WeightKind,
    Weights,
)
from microcosm.frame.kernels import RulesKernel, SimulateRulesKernel
from microcosm.graph import (
    Determinism,
    Kernel,
    KernelContext,
    Node,
    Numeric,
    Owned,
    SeedSource,
    Slice,
    StructuralDelta,
    source_hash,
)

_STUB_SCHEMA = EntitySchema(group_entities=("household",))
_STUB_VARIABLES = ("net_earnings", "housing_allowance")


class _StubRulesEngine:
    """Small deterministic adapter exercising two output entities."""

    def __init__(self) -> None:
        self.materialize_calls = 0

    def variable_metadata(self, name: str) -> VariableMetadata:
        metadata = {
            "net_earnings": VariableMetadata(
                name="net_earnings", entity="person", dtype="float", period="year"
            ),
            "housing_allowance": VariableMetadata(
                name="housing_allowance",
                entity="household",
                dtype="float",
                period="year",
            ),
        }
        try:
            return metadata[name]
        except KeyError as error:
            raise ValueError(f"Unknown stub variable {name!r}.") from error

    def variables(self) -> Sequence[str]:
        return ("earnings", "housing_cost")

    def entity_schema(self) -> EntitySchema:
        return _STUB_SCHEMA

    def materialize(
        self,
        bundle: Frame,
        variables: Sequence[str],
        period: int | str,
    ) -> Mapping[str, np.ndarray]:
        self.materialize_calls += 1
        year = int(str(period)[:4])
        earnings = bundle.table("person")["earnings"].to_numpy(dtype=np.float64)
        housing_cost = bundle.table("household")["housing_cost"].to_numpy(
            dtype=np.float64
        )
        available = {
            "net_earnings": earnings * np.float64(0.8) + np.float64(year - 2025),
            "housing_allowance": np.maximum(
                np.float64(12_000.0) - housing_cost, np.float64(0.0)
            ),
        }
        return {variable: available[variable] for variable in variables}

    def export_contract(self) -> ExportContract:
        return ExportContract.empty()

    def write_dataset(
        self,
        bundle: Frame,
        path: str | Path,
        period: int | str,
    ) -> None:
        return None


def _stub_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.asarray([11, 12, 21, 22], dtype=np.int64),
            "person_household_id": np.asarray([100, 100, 200, 200], dtype=np.int64),
            "earnings": np.asarray([50_000.0, 25_000.0, 80_000.0, 0.0]),
        },
        index=pd.Index([101, 103, 107, 109]),
    )
    household = pd.DataFrame(
        {
            "household_id": np.asarray([100, 200], dtype=np.int64),
            "housing_cost": np.asarray([9_600.0, 15_000.0]),
        },
        index=pd.Index([501, 509]),
    )
    return Frame(
        {"person": person, "household": household},
        _STUB_SCHEMA,
        {"household": Weights(np.asarray([125.0, 275.0]), WeightKind.DESIGN)},
        strata=pd.Series(
            ["survey", "survey", "synthetic", "synthetic"],
            index=person.index,
        ),
    )


def _context(
    frame: Frame,
    node: Node,
    *,
    weighted_entities: tuple[str, ...],
) -> KernelContext:
    return KernelContext(
        node=node,
        tables={entity: frame.table(entity) for entity in frame.entities},
        weights={entity: frame.resolve_weights(entity) for entity in weighted_entities},
        strata=frame.strata,
        params=node.params,
        rng=np.random.default_rng(719),
    )


def _assert_byte_parity(
    frame: Frame,
    engine: _StubRulesEngine | object,
    expected: Mapping[str, np.ndarray],
    actual: Mapping[tuple[str, str], pd.Series],
    variables: tuple[str, ...],
) -> None:
    for variable in variables:
        metadata = engine.variable_metadata(variable)  # type: ignore[attr-defined]
        values = np.asarray(expected[variable])
        series = actual[(metadata.entity, variable)]
        assert series.shape == values.shape
        assert series.dtype == values.dtype
        assert series.to_numpy(copy=False).tobytes() == values.tobytes()
        id_column = frame.schema.entity_id_column(metadata.entity)
        expected_ids = frame.table(metadata.entity)[id_column]
        assert series.index.name == id_column
        assert series.index.tolist() == expected_ids.tolist()


def test_stub_rules_kernel_has_byte_parity_shape_ids_and_capabilities() -> None:
    frame = _stub_frame()
    expected_engine = _StubRulesEngine()
    expected = expected_engine.materialize(frame, _STUB_VARIABLES, period=2025)
    engine = _StubRulesEngine()
    kernel = SimulateRulesKernel("stub", engine)
    node = Node(
        id="simulate",
        kernel=kernel.ref,
        inputs=(
            Slice("person", ("earnings",)),
            Slice("household", ("housing_cost",)),
        ),
        outputs=(
            Owned("person", "net_earnings", "float64"),
            Owned("household", "housing_allowance", "float64"),
        ),
        params={
            "engine_ref": "stub",
            "variables": _STUB_VARIABLES,
            "period": 2025,
        },
    )

    result = kernel.run(
        _context(frame, node, weighted_entities=("person", "household"))
    )

    assert engine.materialize_calls == 1
    assert set(result.columns) == {
        ("person", "net_earnings"),
        ("household", "housing_allowance"),
    }
    _assert_byte_parity(frame, engine, expected, result.columns, _STUB_VARIABLES)
    assert isinstance(kernel, Kernel)
    assert RulesKernel is SimulateRulesKernel
    assert kernel.capabilities.determinism is Determinism.DETERMINISTIC
    assert kernel.capabilities.numeric is Numeric.BITWISE
    assert kernel.capabilities.seed_source is SeedSource.NONE
    assert kernel.capabilities.structural is StructuralDelta.NONE
    assert kernel.capabilities.consumes_se is False
    assert kernel.capabilities.dependencies == ()
    assert kernel.implementation_hash() == source_hash(
        SimulateRulesKernel, _StubRulesEngine
    )
    assert result.receipt == {
        "engine_ref": "stub",
        "period": 2025,
        "variables": _STUB_VARIABLES,
        "output_rows": (
            ("net_earnings", "person", 4),
            ("housing_allowance", "household", 2),
        ),
    }


def _twenty_household_us_frame() -> Frame:
    count = 20
    ids = np.arange(1, count + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids.copy(),
            "person_household_id": ids.copy(),
            "person_tax_unit_id": ids.copy(),
            "person_spm_unit_id": ids.copy(),
            "person_family_id": ids.copy(),
            "person_marital_unit_id": ids.copy(),
            "age": np.arange(25, 25 + count, dtype=np.int64),
            "employment_income_before_lsr": np.linspace(
                0.0, 190_000.0, count, dtype=np.float64
            ),
        },
        index=pd.RangeIndex(1_000, 1_000 + count),
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": ids.copy(),
                "state_fips": np.resize(np.asarray([6, 36], dtype=np.int64), count),
            },
            index=pd.RangeIndex(2_000, 2_000 + count),
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids.copy()}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids.copy()}),
        "family": pd.DataFrame({"family_id": ids.copy()}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids.copy()}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.linspace(750.0, 1_250.0, count, dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


@pytest.mark.requires_us
def test_policyengine_us_twenty_household_kernel_matches_direct_materialize() -> None:
    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    frame = _twenty_household_us_frame()
    engine = PolicyEngineUSEngine()
    variables = ("employment_income", "household_net_income")
    expected = engine.materialize(frame, variables, period=2024)
    kernel = SimulateRulesKernel(
        "policyengine-us", engine, dependencies=("policyengine-us",)
    )
    node = Node(
        id="simulate-us",
        kernel=kernel.ref,
        inputs=(
            Slice("person", ("age", "employment_income_before_lsr")),
            Slice("household", ("state_fips",)),
        ),
        outputs=(
            Owned("person", "employment_income", "float32"),
            Owned("household", "household_net_income", "float32"),
        ),
        params={
            "engine_ref": "policyengine-us",
            "variables": variables,
            "period": 2024,
        },
    )

    result = kernel.run(
        _context(frame, node, weighted_entities=("person", "household"))
    )

    _assert_byte_parity(frame, engine, expected, result.columns, variables)
    assert kernel.capabilities.dependencies == ("policyengine-us",)
    assert len(kernel.implementation_hash()) == 64
