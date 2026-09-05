"""Synthetic current-dollar contract; no source rows, engine or microunit call."""

import json
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime._asec_current_money_codec import (
    decode_current_money,
    encode_current_money,
)
from microcosm.build.us_runtime.asec_current_money import (
    AsecMoneyScope,
    AsecMoneyViews,
    CodebookStatus,
    MoneyRefusalError,
    SyntheticAsecSource,
    classify_asec_money,
    compile_asec_current_money_spec,
    require_complete_current_money,
    restate_asec_current_money,
)


def digest(label):
    return sha256(label.encode()).hexdigest()


@pytest.fixture
def resources():
    # Explicit test injection: fixtures do not need an installed optional package.
    from microcosm.build.us_runtime.asec_current_money_resources import (
        packaged_current_money_resources,
    )

    return packaged_current_money_resources(
        inspected_version="0.1.0",
        inspected_module_sha256="79007065113f9600601433dc2633f4392c675e28b87a7383ab8a1f7c6279d47e",
    )


@pytest.fixture
def fixture(resources):
    source = SyntheticAsecSource(
        parent_sha256=digest("invented v4"),
        attachment_sha256=digest("invented attachment"),
        cohort_sha256=tuple((y, digest(str(y))) for y in (2022, 2023, 2024)),
        sidecar_sha256=(
            ("ED_VAL.archive", digest("archive")),
            ("ED_VAL.member", digest("member")),
        ),
        selection_sha256=digest("whole invented scope"),
    )
    spec = compile_asec_current_money_spec(resources, source)
    scope = AsecMoneyScope(
        person_ids=(30, 10, 20, 60, 40, 50),
        household_ids=(100, 200, 300),
        person_household_ids=(100, 100, 200, 200, 300, 300),
        person_spm_ids=(5, 5, 7, 7, 9, 9),
        person_years=(2022, 2022, 2023, 2023, 2024, 2024),
        household_years=(2022, 2023, 2024),
        person_native_keys=tuple("p" + str(i) for i in range(6)),
        household_native_keys=("h1", "h2", "h3"),
    )
    person = pd.DataFrame(
        {f.column: np.zeros(6) for f in spec.fields if f.entity == "person"},
        index=[8, 8, 1, 5, 5, 3],
    )
    household = pd.DataFrame({"asec_HTOTVAL": np.zeros(3)}, index=[91, 18, 91])
    return spec, scope, AsecMoneyViews(person, household, scope)


def run(fixture):
    spec, scope, views = fixture
    decoded = classify_asec_money(views, spec, scope)
    restated = restate_asec_current_money(decoded, spec)
    return decoded, restated, require_complete_current_money(restated, spec)


def test_scalar_factor_oracle_niu_zero_and_immutable_roundtrip(fixture):
    spec, scope, views = fixture
    views.person["WSAL_VAL"] = [12345, 11, 12345, 11, 12345, 11]
    views.person["SEMP_VAL"] = [-1, -9999, -1, -9999, -1, -9999]
    views.person["ANN_VAL"] = -1.0
    views.person["PNSN_VAL"] = 100.0
    views.person["CSP_VAL"] = -0.0
    raw_person = views.person.copy(deep=True)
    raw_household = views.household.copy(deep=True)
    decoded, restated, ready = run(fixture)
    factors = [
        float("174.4") / float(str(y))
        for y in (163.6, 163.6, 170.0, 170.0, 174.4, 174.4)
    ]
    np.testing.assert_array_equal(
        ready.field("WSAL_VAL").amounts,
        [v * f for v, f in zip([12345, 11] * 3, factors, strict=True)],
    )
    np.testing.assert_array_equal(
        ready.field("SEMP_VAL").amounts,
        [v * f for v, f in zip([-1, -9999] * 3, factors, strict=True)],
    )
    assert ready.field("ANN_VAL").amount_bytes == bytes(6 * 8)
    assert set(ready.field("ANN_VAL").statuses) == {CodebookStatus.DECLARED_NIU}
    assert set(ready.field("CSP_VAL").statuses) == {CodebookStatus.ZERO_NONE_OR_NIU}
    assert set(ready.field("CHSP_VAL").statuses) == {CodebookStatus.DECLARED_NIU}
    assert set(ready.field("CSP_VAL").zero_origin) == {1}
    assert set(ready.field("ANN_VAL").zero_origin) == {0}
    assert (
        ready.field("WSAL_VAL").amount_bytes[-16:]
        == decoded.field("WSAL_VAL").amount_bytes[-16:]
    )
    for value in (decoded, restated, ready):
        with pytest.raises(ValueError):
            value.field("WSAL_VAL").amounts.setflags(write=True)
    payload = encode_current_money(ready)
    replay = decode_current_money(payload, ready.bindings)
    assert encode_current_money(replay) == payload
    assert json.loads(replay.header)["source_authentication"] == "synthetic_unverified"
    assert not hasattr(restated, "nominal")
    pd.testing.assert_frame_equal(views.person, raw_person)
    pd.testing.assert_frame_equal(views.household, raw_household)


@pytest.mark.parametrize(
    "field", ["SEMP_VAL", "FRSE_VAL", "RNT_VAL", "PTOTVAL", "HTOTVAL"]
)
@pytest.mark.parametrize("amount", [-1, -9999])
def test_signed_amounts_are_not_codes(fixture, field, amount):
    spec, scope, views = fixture
    domain = next(f for f in spec.fields if f.name == field)
    getattr(views, domain.entity)[domain.column] = float(amount)
    assert set(run(fixture)[0].field(field).amounts) == {amount}
    assert set(run(fixture)[0].field(field).statuses) == {CodebookStatus.AMOUNT_NONZERO}


@pytest.mark.parametrize("field", ["WSAL_VAL", "CAP_VAL"])
@pytest.mark.parametrize("amount", [-1, -9999])
def test_invalid_negative_is_not_hidden(fixture, field, amount):
    fixture[2].person[field] = float(amount)
    with pytest.raises(MoneyRefusalError, match=field + ": OUTSIDE_CODEBOOK_RANGE"):
        run(fixture)


@pytest.mark.parametrize(
    "amount,reason",
    [
        (0, None),
        (10000, None),
        (10001, "DISPUTED"),
        (99999, "DISPUTED"),
        (-1, "OUTSIDE"),
        (100000, "OUTSIDE"),
    ],
)
def test_energy_conflict_refusal(fixture, amount, reason):
    fixture[2].person["SPM_ENGVAL"] = float(amount)
    if reason:
        with pytest.raises(MoneyRefusalError, match=reason):
            run(fixture)
    else:
        assert set(run(fixture)[0].field("SPM_ENGVAL").amounts) == {amount}


@pytest.mark.parametrize(
    "dtype", ["float32", "Float32", "bool", "boolean", "object", "string", "complex128"]
)
def test_unsupported_dtypes_refuse_without_coercion(fixture, dtype):
    fixture[2].person["PTOTVAL"] = pd.Series(
        [0] * 6, dtype=dtype, index=fixture[2].person.index
    )
    with pytest.raises(MoneyRefusalError, match="UNSUPPORTED_DTYPE"):
        run(fixture)


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float64",
        "Int8",
        "UInt8",
        "Int16",
        "UInt16",
        "Int32",
        "UInt32",
        "Int64",
        "UInt64",
        "Float64",
    ],
)
def test_closed_numeric_types(fixture, dtype):
    fixture[2].person["PTOTVAL"] = pd.Series(
        [1] * 6, dtype=dtype, index=fixture[2].person.index
    )
    assert set(run(fixture)[0].field("PTOTVAL").amounts) == {1}


@pytest.mark.parametrize("value", [np.inf, -np.inf, 1.5])
def test_nonfinite_fractional_refuse(fixture, value):
    fixture[2].person["PTOTVAL"] = value
    with pytest.raises(MoneyRefusalError, match="NONFINITE|FRACTIONAL"):
        run(fixture)


@pytest.mark.parametrize("dtype", ["float64", "Float64", "Int64"])
def test_null_retains_invalidity_and_cannot_be_ready(fixture, dtype):
    spec, scope, views = fixture
    views.person["PTOTVAL"] = pd.Series(
        [None, 1, 1, 1, 1, 1], dtype=dtype, index=views.person.index
    )
    decoded = classify_asec_money(views, spec, scope)
    amount = decoded.field("PTOTVAL")
    assert amount.validity[0] == 0 and amount.amount_bytes[:8] == bytes(8)
    assert amount.statuses[0] == CodebookStatus.MISSING_NULL
    with pytest.raises(MoneyRefusalError, match="MISSING_REQUIRED_AMOUNT"):
        require_complete_current_money(restate_asec_current_money(decoded, spec), spec)


def test_every_domain_boundary_and_year(fixture):
    spec, scope, views = fixture
    for field in spec.fields:
        table = getattr(views, field.entity)
        original = table[field.column].copy()
        for amount in (field.minimum, field.maximum):
            table[field.column] = float(amount)
            assert len(run(fixture)[0].field(field.name).amounts) == len(table)
        for amount in (field.minimum - 1, field.maximum + 1):
            table[field.column] = float(amount)
            with pytest.raises(MoneyRefusalError, match=field.name):
                run(fixture)
        table[field.column] = original


def test_missing_column_address_and_typed_states(fixture):
    spec, scope, views = fixture
    _, restated, ready = run(fixture)
    with pytest.raises(MoneyRefusalError, match="TYPED_STATE"):
        classify_asec_money(restated, spec, scope)
    with pytest.raises(MoneyRefusalError, match="TYPED_STATE"):
        restate_asec_current_money(restated, spec)
    with pytest.raises(MoneyRefusalError, match="SYNTHETIC_NOT_PRODUCTION"):
        require_complete_current_money(restated, spec, production=True)
    del views.household["asec_HTOTVAL"]
    views.person["HTOTVAL"] = 0.0
    with pytest.raises(MoneyRefusalError, match="HTOTVAL: MISSING_COLUMN"):
        run(fixture)


def test_group_year_and_repetition_require_actual_agreement(fixture):
    spec, scope, views = fixture
    views.person["SPM_CHILDCAREXPNS"] = [0, 1, 0, 0, 0, 0]
    with pytest.raises(MoneyRefusalError, match="INCONSISTENT_SPM_AMOUNT"):
        run(fixture)
    with pytest.raises(MoneyRefusalError, match="MIXED_HOUSEHOLD_YEAR"):
        replace(scope, person_years=(2022, 2023, 2023, 2023, 2024, 2024))
    with pytest.raises(MoneyRefusalError, match="DUPLICATE_PERSON_ID"):
        replace(scope, person_ids=(1, 1, 2, 3, 4, 5))
    other = replace(scope, person_ids=(10, 30, 20, 60, 40, 50))
    with pytest.raises(MoneyRefusalError, match="SCOPE_MISMATCH"):
        classify_asec_money(views, spec, other)


def test_additive_graph_consumer_status_preserves_source_recipe(resources):
    from microcosm.build.us_runtime.asec_current_money import RESOURCE_PINS
    from microcosm.build.us_runtime.asec_current_money_graph_resources import (
        load_graph_current_money_consumers,
    )
    from microcosm.build.us_runtime.asec_current_money_units import (
        MICROUNIT_CURRENT_MONEY_COLUMNS,
    )
    from microcosm.build.us_runtime.asec_engine_evaluation import ADMITTED_ROOTS
    from microcosm.build.us_runtime.cps_carried_current import (
        CPS_CARRIED_CURRENT_MONEY_FIELDS,
        CPS_CARRIED_CURRENT_PERSON_LEAVES,
        CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES,
    )

    source = json.loads(resources.consumers)
    status = load_graph_current_money_consumers()
    assert sha256(resources.consumers).hexdigest() == RESOURCE_PINS[2]
    assert status["source_declarations"]["sha256"] == RESOURCE_PINS[2]
    assert source["implementation_status"] == (
        "declarations_only_live_nominal_consumers_unchanged"
    )
    assert "addresses" not in status
    routes = status["wired_routes"]
    assert set(routes) == {"microunit", "asec_prepared_current_money_graph"}
    graph = routes["asec_prepared_current_money_graph"]
    assert tuple(graph["money_fields"]) == CPS_CARRIED_CURRENT_MONEY_FIELDS
    assert tuple(graph["person_leaves"]) == CPS_CARRIED_CURRENT_PERSON_LEAVES
    assert tuple(graph["spm_unit_leaves"]) == CPS_CARRIED_CURRENT_SPM_UNIT_LEAVES
    assert tuple(graph["engine_outputs"]) == ADMITTED_ROOTS
    assert set(routes["microunit"]["money_fields"]) == set(
        MICROUNIT_CURRENT_MONEY_COLUMNS
    )
    assert set(status["pending_nominal_consumer_routes"]) == {
        consumer for address in source["addresses"] for consumer in address["consumers"]
    } - {"microunit"}
    assert status["all_current_money_consumers_wired"] is False
    assert status["release_eligible"] is False
    assert status["engine_defaults_allowlist"] == []


@pytest.mark.parametrize("target", ["graph", "parent"])
def test_additive_graph_consumer_resource_and_parent_drift_refuse(
    tmp_path, monkeypatch, target
):
    from types import SimpleNamespace

    from microcosm.build.us_runtime import asec_current_money_graph_resources as graph

    source_package = graph.resources.files(graph.__package__)
    names = (graph.GRAPH_CONSUMERS_RESOURCE, "asec_current_money_consumers_v1.json")
    for name in names:
        (tmp_path / name).write_bytes(source_package.joinpath(name).read_bytes())
    path = tmp_path / names[target == "parent"]
    path.write_bytes(path.read_bytes() + b"\n")
    monkeypatch.setattr(graph, "resources", SimpleNamespace(files=lambda _: tmp_path))
    with pytest.raises(ValueError, match="GRAPH_CONSUMERS.*FINGERPRINT"):
        graph.load_graph_current_money_consumers()
