"""Mechanical metadata checks with invented registries, never model results."""

import datetime
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine
from microcosm.frame.units import US_SCHEMA


class InventedEnum:
    pass


class InventedStatus(InventedEnum, Enum):
    INCLUDED = 0
    UNRESOLVED = 1


def _variable(entity, value_type=int, dtype="int32", **kwargs):
    return SimpleNamespace(
        entity=SimpleNamespace(key=entity),
        value_type=value_type,
        dtype=np.dtype(dtype),
        definition_period=kwargs.pop("definition_period", "year"),
        is_neutralized=False,
        end=None,
        get_formula=lambda period: None,
        **kwargs,
    )


@pytest.fixture
def metadata_case(monkeypatch):
    # Only parser/Enum metadata stand-ins: no country package or model evaluator.
    core = ModuleType("policyengine_core")
    periods = ModuleType("policyengine_core.periods")
    enums = ModuleType("policyengine_core.enums")
    enums.Enum = InventedEnum

    def parse_period(value):
        year = int(value)
        return SimpleNamespace(
            unit="year", size=1, start=SimpleNamespace(date=datetime.date(year, 1, 1))
        )

    periods.period = parse_period
    monkeypatch.setitem(sys.modules, "policyengine_core", core)
    monkeypatch.setitem(sys.modules, periods.__name__, periods)
    monkeypatch.setitem(sys.modules, enums.__name__, enums)
    spm = ModuleType("invented_country.spm")
    spm.DATASET_SOURCE_INPUTS = frozenset({"role", "status"})
    spm.REJECTED_DATASET_INPUTS = frozenset({"rejected_output"})
    monkeypatch.setitem(sys.modules, spm.__name__, spm)
    tables = {"person": pd.DataFrame({"person_id": [10, 11]})}
    registry = {"person_id": _variable("person")}
    for entity in US_SCHEMA.group_entities:
        tables[entity] = pd.DataFrame({entity + "_id": [1, 2]})
        tables["person"]["person_" + entity + "_id"] = [1, 2]
        registry[entity + "_id"] = _variable(entity)
        registry["person_" + entity + "_id"] = _variable("person")
    tables["person"]["amount"] = [-5.0, 0.1]
    tables["person"]["role"] = pd.array([True, False], dtype="boolean")
    tables["person"]["count"] = pd.array([1, 2], dtype="Int64")
    tables["spm_unit"]["status"] = ["INCLUDED", "UNRESOLVED"]
    tables["household"]["code"] = ["001", "002"]
    registry.update(
        amount=_variable("person", float, "float32"),
        role=_variable("person", bool, "bool", definition_period="eternity"),
        count=_variable("person"),
        status=_variable(
            "spm_unit",
            InventedEnum,
            "int16",
            possible_values=InventedStatus,
        ),
        code=_variable("household", str, "object"),
        household_weight=_variable("household", float, "float32"),
        rejected_output=_variable("person", float, "float32"),
    )
    for name in ("person_id", "role", "status"):
        registry[name].get_formula = lambda period: object()
    frame = Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([1.0, 2.0]), WeightKind.IMPORTANCE)},
    )
    engine = PolicyEngineUSEngine(defaults={"unobserved": 999})
    engine._system = SimpleNamespace(variables=registry)
    monkeypatch.setattr(
        engine,
        "_import_policyengine_us",
        lambda: SimpleNamespace(__name__="invented_country"),
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Metadata validation must not construct, calculate, default or write"
        )

    for name in ("_build_dataset", "write_dataset", "materialize", "default_values"):
        monkeypatch.setattr(engine, name, forbidden)
    return engine, frame, registry, spm, periods


@pytest.mark.parametrize("period", [2024, "2024"])
def test_representation_success_is_read_only_and_keeps_unresolved(
    metadata_case, period
):
    engine, frame, _, _, _ = metadata_case
    before = {entity: frame.table(entity).copy(deep=True) for entity in frame.entities}
    weight = frame.weights_for("household").values.tobytes()
    assert engine.validate_input_representation(frame, period=period) is None
    for entity in frame.entities:
        pd.testing.assert_frame_equal(frame.table(entity), before[entity])
    assert frame.weights_for("household").values.tobytes() == weight
    assert engine._defaults == {"unobserved": 999}
    assert "household_weight" not in frame.table("household")


@pytest.mark.parametrize(
    "period", [True, 2024.0, "year:2024:2", "2024-01", "ETERNITY", "x", 0]
)
def test_request_period_refuses_non_calendar_year(metadata_case, period):
    engine, frame, *_ = metadata_case
    with pytest.raises(ValueError, match="INPUT_REPRESENTATION_PERIOD"):
        engine.validate_input_representation(frame, period=period)


@pytest.mark.parametrize("change", ["unit", "size", "start"])
def test_checks_effective_period_parser_result(metadata_case, change):
    engine, frame, _, _, periods = metadata_case
    parsed = periods.period(2024)
    setattr(
        parsed,
        change,
        {
            "unit": "month",
            "size": 2,
            "start": SimpleNamespace(date=datetime.date(2024, 2, 1)),
        }[change],
    )
    periods.period = lambda period: parsed
    with pytest.raises(ValueError, match="INPUT_REPRESENTATION_PERIOD"):
        engine.validate_input_representation(frame, period=2024)


@pytest.mark.parametrize(
    "change,reason",
    [
        ("unknown", "UNKNOWN"),
        ("entity", "ENTITY"),
        ("month", "PERIOD"),
        ("day", "PERIOD"),
        ("neutralized", "NEUTRALIZED"),
        ("expired", "EXPIRED"),
        ("missing_dtype", "DTYPE"),
        ("custom_type", "TYPE"),
        ("rejected", "OWNERSHIP"),
        ("compat", "OWNERSHIP"),
    ],
)
def test_live_metadata_refusals(metadata_case, change, reason):
    engine, frame, registry, _, _ = metadata_case
    if change == "unknown":
        del registry["amount"]
    elif change == "entity":
        registry["amount"].entity.key = "household"
    elif change in ("month", "day"):
        registry["amount"].definition_period = change
    elif change == "neutralized":
        registry["amount"].is_neutralized = True
    elif change == "expired":
        registry["amount"].end = datetime.date(2023, 12, 31)
    elif change == "missing_dtype":
        del registry["amount"].dtype
    elif change == "custom_type":
        registry["amount"].value_type = datetime.date
    else:
        name = "rejected_output" if change == "rejected" else "social_security"
        frame.person[name] = [1.0, 2.0]
        registry[name] = _variable("person", float, "float32")
    with pytest.raises(ValueError, match=reason):
        engine.validate_input_representation(frame, period=2024)


def test_selected_year_formula_ownership(metadata_case):
    engine, frame, registry, _, _ = metadata_case
    registry["amount"].get_formula = lambda period: (
        object() if int(period) >= 2025 else None
    )
    engine.validate_input_representation(frame, period=2024)
    with pytest.raises(ValueError, match="OWNERSHIP"):
        engine.validate_input_representation(frame, period=2025)


@pytest.mark.parametrize("change", ["missing", "mutable", "overlap", "unknown"])
def test_actual_source_declarations_are_required(metadata_case, change):
    engine, frame, _, spm, _ = metadata_case
    if change == "missing":
        del spm.DATASET_SOURCE_INPUTS
    elif change == "mutable":
        spm.DATASET_SOURCE_INPUTS = {"role"}
    elif change == "overlap":
        spm.REJECTED_DATASET_INPUTS |= {"role"}
    else:
        spm.DATASET_SOURCE_INPUTS |= {"unknown"}
    with pytest.raises(RuntimeError):
        engine.validate_input_representation(frame, period=2024)


@pytest.mark.parametrize(
    "column,dtype,values,target,accepted",
    [
        ("person_id", "int64", [-(2**31), 2**31 - 1], "int32", True),
        ("person_id", "int64", [2**31, 2**31 + 1], "int32", False),
        ("person_id", "uint64", [2**63, 2**63 + 1], "int64", False),
        ("person_id", "uint64", [2**63, 2**63 + 1], "uint64", True),
        ("person_household_id", "int64", [1, 2], "int8", True),
        ("person_household_id", "int64", [1, 2], "bool", False),
        ("person_id", "float64", [1.0, 2.5], "int64", False),
        ("person_id", "object", ["1", "2"], "int64", False),
        ("count", "int64", [-1, 1], "uint32", False),
    ],
)
def test_integer_losslessness_uses_each_live_dtype(
    metadata_case, column, dtype, values, target, accepted
):
    engine, frame, registry, _, _ = metadata_case
    frame.person[column] = pd.Series(values, dtype=dtype)
    registry[column].dtype = np.dtype(target)
    before = frame.person[column].copy(deep=True)
    if accepted:
        engine.validate_input_representation(frame, period=2024)
    else:
        with pytest.raises(ValueError, match="INPUT_REPRESENTATION"):
            engine.validate_input_representation(frame, period=2024)
    pd.testing.assert_series_equal(frame.person[column], before)


@pytest.mark.parametrize(
    "entity,column,values,dtype,reason",
    [
        ("person", "role", [True, None], "boolean", "NULL"),
        ("person", "count", [1, None], "Int64", "NULL"),
        ("person", "amount", [1.0, np.nan], "float64", "NULL"),
        ("household", "code", ["001", None], "object", "NULL"),
        ("spm_unit", "status", ["INCLUDED", None], "object", "NULL"),
        ("spm_unit", "status", ["INCLUDED", "ROW_SENTINEL"], "object", "ENUM"),
        ("spm_unit", "status", [0, 1], "int16", "ENUM"),
        ("person", "role", [0, 1], "int64", "BOOL"),
        ("person", "amount", [1.0, 1e100], "float64", "FLOAT"),
        ("person", "amount", ["ROW_SENTINEL", "2"], "object", "FLOAT"),
        ("household", "code", ["001", 2], "object", "STRING"),
    ],
)
def test_values_refuse_without_leaking_examples(
    metadata_case, entity, column, values, dtype, reason
):
    engine, frame, *_ = metadata_case
    frame.table(entity)[column] = pd.Series(values, dtype=dtype)
    before = frame.table(entity)[column].copy(deep=True)
    with pytest.raises(ValueError, match=reason) as error:
        engine.validate_input_representation(frame, period=2024)
    assert "ROW_SENTINEL" not in str(error.value)
    assert error.value.__cause__ is None
    pd.testing.assert_series_equal(frame.table(entity)[column], before)


@pytest.mark.parametrize("dtype", ["U2", "S2"])
def test_fixed_strings_cannot_truncate(metadata_case, dtype):
    engine, frame, registry, _, _ = metadata_case
    registry["code"].dtype = np.dtype(dtype)
    with pytest.raises(ValueError, match="STRING"):
        engine.validate_input_representation(frame, period=2024)


def test_representable_infinity_is_not_scientific_qualification(metadata_case):
    engine, frame, *_ = metadata_case
    frame.person["amount"] = [np.inf, -np.inf]
    engine.validate_input_representation(frame, period=2024)


@pytest.mark.parametrize(
    "values,accepted",
    [
        ([InventedStatus.INCLUDED, InventedStatus.UNRESOLVED], True),
        ([InventedStatus.INCLUDED, "UNRESOLVED"], False),
        ([SimpleNamespace(name="INCLUDED"), SimpleNamespace(name="UNRESOLVED")], False),
        ([b"INCLUDED", b"UNRESOLVED"], True),
        ([b"INCLUDED", b"\xffROW_SENTINEL"], False),
    ],
)
def test_only_supported_named_or_member_enum_inputs(metadata_case, values, accepted):
    engine, frame, *_ = metadata_case
    frame.table("spm_unit")["status"] = pd.Series(values, dtype=object)
    if accepted:
        engine.validate_input_representation(frame, period=2024)
    else:
        with pytest.raises(ValueError) as error:
            engine.validate_input_representation(frame, period=2024)
        assert "ROW_SENTINEL" not in str(error.value)


def test_group_identifier_uses_its_live_dtype(metadata_case):
    engine, frame, registry, _, _ = metadata_case
    frame.table("household")["household_id"] = [128, 129]
    frame.person["person_household_id"] = [128, 129]
    registry["household_id"].dtype = np.dtype("int8")
    with pytest.raises(ValueError, match="INTEGER_RANGE: household.household_id"):
        engine.validate_input_representation(frame, period=2024)


def test_end_in_selected_year_is_not_expired(metadata_case):
    engine, frame, registry, _, _ = metadata_case
    registry["amount"].end = datetime.date(2024, 1, 1)
    engine.validate_input_representation(frame, period=2024)


@pytest.mark.parametrize("dtype,accepted", [(object, False), ("S2", True)])
def test_enum_bytes_follow_core_array_normalization(metadata_case, dtype, accepted):
    engine, frame, registry, _, _ = metadata_case
    # Core decodes fixed-width bytes as UTF-8, but casts object arrays to str.
    # The latter cannot decode non-ASCII UTF-8 bytes. Names alone are insufficient.
    registry["status"].possible_values = Enum("UnicodeStatus", {"é": 0})
    frame.table("spm_unit")["status"] = pd.Series([b"\xc3\xa9"] * 2, dtype=dtype)
    if accepted:
        engine.validate_input_representation(frame, period=2024)
    else:
        with pytest.raises(ValueError, match="UNSUPPORTED"):
            engine.validate_input_representation(frame, period=2024)


@pytest.mark.parametrize(
    "dtype,reason",
    [
        ("float64", "ENUM"),
        ("int16", "ENUM"),
        ("object", None),
        ("S2", None),
        ("U2", None),
    ],
)
def test_empty_enum_still_requires_supported_physical_kind(
    metadata_case, dtype, reason
):
    from microcosm.frame.adapters.policyengine_us import _input_representation_reason

    _, _, registry, _, _ = metadata_case
    assert (
        _input_representation_reason(
            pd.Series(dtype=dtype), registry["status"], InventedEnum
        )
        == reason
    )
