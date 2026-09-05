"""Closed resource, replay, scope and installed-code refusal contracts."""

import json
import struct
import sys
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import asec_current_money as money
from microcosm.build.us_runtime import asec_current_money_resources as loader
from microcosm.build.us_runtime._asec_current_money_codec import (
    MAGIC,
    decode_current_money,
    encode_current_money,
)
from microcosm.build.us_runtime.asec_current_money import (
    AsecMoneyScope,
    AsecMoneyViews,
    AuthenticatedAsecSource,
    MoneyBindings,
    MoneyRefusalError,
    ReadyCurrentMoney,
    SyntheticAsecSource,
    classify_asec_money,
    compile_asec_current_money_spec,
    require_complete_current_money,
    restate_asec_current_money,
)


def fixture():
    resources = loader.packaged_current_money_resources(
        inspected_version="0.1.0", inspected_module_sha256=money.MICROUNIT_SHA256
    )
    pin = "a" * 64
    source = SyntheticAsecSource(
        pin,
        pin,
        ((2022, pin), (2023, pin), (2024, pin)),
        (("ED_VAL.archive", pin), ("ED_VAL.member", pin)),
        pin,
    )
    spec = compile_asec_current_money_spec(resources, source)
    scope = AsecMoneyScope(
        (4, 1, 9, 5),
        (7, 8),
        (7, 7, 8, 8),
        (3, 3, 6, 6),
        (2022, 2022, 2024, 2024),
        (2022, 2024),
        ("a", "b", "c", "d"),
        ("g", "h"),
    )
    person = pd.DataFrame(
        {f.column: np.zeros(4) for f in spec.fields if f.entity == "person"},
        index=[2, 2, 7, 5],
    )
    household = pd.DataFrame({"asec_HTOTVAL": np.zeros(2)}, index=[91, 91])
    views = AsecMoneyViews(person, household, scope)
    return spec, scope, views


def ready(spec, scope, views):
    return require_complete_current_money(
        restate_asec_current_money(classify_asec_money(views, spec, scope), spec), spec
    )


def edit_resource(resources, name, edit):
    value = json.loads(getattr(resources, name))
    edit(value)
    return replace(resources, **{name: json.dumps(value).encode()})


@pytest.mark.parametrize(
    "name,edit",
    [
        ("domains", lambda x: x["fields"].pop()),
        ("domains", lambda x: x["fields"][0].update(minimum=-2)),
        ("domains", lambda x: x["fields"][0].update(zero_semantics="invented")),
        (
            "domains",
            lambda x: x["fields"][13].update(entity="person", column="HTOTVAL"),
        ),
        ("domains", lambda x: x["source_years"].pop()),
        ("domains", lambda x: x.update(schema_version=2)),
        ("domains", lambda x: x.update(unknown="must refuse")),
        ("price", lambda x: x["cells"].update({"2022": "163.7"})),
        ("price", lambda x: x["cells"].pop("2023")),
        ("price", lambda x: x.update(target_year=2025)),
        ("consumers", lambda x: x["addresses"][0].update(cpi_owner="unowned")),
        ("consumers", lambda x: x["addresses"].append(x["addresses"][0])),
        (
            "consumers",
            lambda x: x["addresses"][0].update(raw_alternates=["nominal_ANN_VAL"]),
        ),
        (
            "consumers",
            lambda x: x["addresses"][0].update(
                resulting_surfaces=["prior_year_employment_income"]
            ),
        ),
        (
            "consumers",
            lambda x: x["addresses"].__setitem__(0, {"field": "unknown_amount"}),
        ),
    ],
)
def test_closed_resource_amendments_refuse(name, edit):
    spec, _, _ = fixture()
    with pytest.raises(MoneyRefusalError, match="RESOURCE_FINGERPRINT"):
        compile_asec_current_money_spec(
            edit_resource(spec.resources, name, edit), spec.source
        )


@pytest.mark.parametrize(
    "document",
    [
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b"[]",
        b"{",
        b"{}" + b" " * money.RESOURCE_MAX_BYTES,
    ],
)
def test_strict_resource_json(document):
    spec, _, _ = fixture()
    with pytest.raises(MoneyRefusalError):
        compile_asec_current_money_spec(
            replace(spec.resources, domains=document), spec.source
        )


def test_oversized_json_integer_has_a_sanitized_refusal():
    with pytest.raises(MoneyRefusalError, match="MALFORMED_JSON"):
        money._parse(b'{"value":' + b"9" * 5000 + b"}")


def test_pinned_consumers_have_closed_current_money_ownership():
    spec, _, _ = fixture()
    document = json.loads(spec.resources.consumers)
    addresses = document["addresses"]
    assert len(addresses) == 33
    assert {item["field"] for item in addresses} == set(money.FIELDS)
    assert all(item["raw_alternates"] == [] for item in addresses)
    assert all(item["cpi_owner"] == money.RECIPE for item in addresses)
    assert all(item["input_basis"] == "target_current_2024" for item in addresses)
    assert all(
        "prior_year_employment_income" not in item["resulting_surfaces"]
        for item in addresses
    )


def test_33_exact_fields_addresses_vintages_and_zero_mapping():
    spec, _, _ = fixture()
    data = json.loads(spec.resources.domains)
    assert tuple(f.name for f in spec.fields) == money.FIELDS
    assert (
        len(spec.fields) == 33 and sum(len(f["vintages"]) for f in data["fields"]) == 99
    )
    assert set(f["zero_semantics"] for f in data["fields"]) == set(money._ZERO_STATUS)
    assert all(
        [(v["income_year"], v["survey_year"]) for v in f["vintages"]]
        == [(2022, 2023), (2023, 2024), (2024, 2025)]
        for f in data["fields"]
    )
    h = next(f for f in spec.fields if f.name == "HTOTVAL")
    assert (h.entity, h.column, h.grain) == ("household", "asec_HTOTVAL", "household")
    assert all(f.entity == "person" for f in spec.fields if f.name != "HTOTVAL")
    for year, expression, bits in spec.factors:
        cell = {2022: "163.6", 2023: "170.0", 2024: "174.4"}[year]
        assert expression == "174.4/" + cell
        assert bits == struct.pack("<d", float("174.4") / float(cell)).hex()


def test_no_public_authentication_or_production_constructor():
    spec, scope, views = fixture()
    with pytest.raises(MoneyRefusalError):
        AuthenticatedAsecSource()
    with pytest.raises(MoneyRefusalError):
        ReadyCurrentMoney()
    with pytest.raises(MoneyRefusalError, match="SYNTHETIC_SOURCE_REQUIRED"):
        compile_asec_current_money_spec(spec.resources, {"namespace": "verified"})
    with pytest.raises(TypeError):
        replace(spec.source, namespace="verified")
    with pytest.raises(MoneyRefusalError):
        replace(spec.source, source_kind="authenticated")
    with pytest.raises(MoneyRefusalError):
        replace(spec.source, zero_origin_policy="observed_nonblank_zero")
    value = ready(spec, scope, views)
    data = json.loads(value.header)
    data["source_authentication"] = "verified"
    with pytest.raises(MoneyRefusalError, match="HEADER_BINDING"):
        MoneyBindings(spec, money._json(data))


def test_source_roster_requires_ptotval_and_three_income_years():
    spec, _, _ = fixture()
    with pytest.raises(MoneyRefusalError):
        replace(
            spec.source, field_roster=tuple(f for f in money.FIELDS if f != "PTOTVAL")
        )
    with pytest.raises(MoneyRefusalError):
        replace(
            spec.source,
            cohort_sha256=((2023, "a" * 64), (2024, "a" * 64), (2025, "a" * 64)),
        )


def test_year_bytes_cannot_be_replaced_after_classification():
    spec, scope, views = fixture()
    decoded = classify_asec_money(views, spec, scope)
    forged = replace(
        decoded,
        person_year_bytes=np.full(4, 2024, dtype="<i8").tobytes(),
        _token=money._STATE_TOKEN,
    )
    with pytest.raises(MoneyRefusalError, match="YEAR_BINDING"):
        restate_asec_current_money(forged, spec)


def test_input_representation_scope_source_and_execution_invalidate_replay():
    spec, scope, views = fixture()
    original = ready(spec, scope, views)
    payload = encode_current_money(original)
    variants = []
    views.person["WSAL_VAL"] = views.person["WSAL_VAL"].astype("int64")
    variants.append(ready(spec, scope, views).bindings)
    views.person["WSAL_VAL"] = views.person["WSAL_VAL"].astype("float64")
    for changed in (
        replace(scope, person_ids=(1, 4, 9, 5)),
        replace(scope, person_native_keys=("z", "b", "c", "d")),
        replace(scope, person_spm_ids=(8, 8, 6, 6)),
        replace(
            scope, person_years=(2023, 2023, 2024, 2024), household_years=(2023, 2024)
        ),
    ):
        variants.append(ready(spec, changed, replace(views, scope=changed)).bindings)
    for source in (
        replace(spec.source, selection_sha256="b" * 64),
        replace(spec.source, parent_sha256="b" * 64),
        replace(spec.source, attachment_sha256="b" * 64),
        replace(
            spec.source,
            sidecar_sha256=(("ED_VAL.archive", "a" * 64), ("ED_VAL.member", "b" * 64)),
        ),
    ):
        new = compile_asec_current_money_spec(spec.resources, source)
        variants.append(ready(new, scope, views).bindings)
    execution = json.loads(spec.resources.execution_identity)
    execution["modules"]["asec_current_money.py"] = "b" * 64
    new = compile_asec_current_money_spec(
        replace(spec.resources, execution_identity=money._json(execution)), spec.source
    )
    variants.append(ready(new, scope, views).bindings)
    for binding in variants:
        with pytest.raises(MoneyRefusalError, match="REPLAY_BINDING"):
            decode_current_money(payload, binding)


def test_row_permutation_is_positional_and_pandas_labels_are_preserved():
    spec, scope, views = fixture()
    views.person["WSAL_VAL"] = [1, 2, 3, 4]
    views.person.attrs["untouched"] = {"metadata": "nominal"}
    original = ready(spec, scope, views)
    order = (3, 0, 2, 1)
    changed = replace(
        scope,
        **{
            name: tuple(getattr(scope, name)[i] for i in order)
            for name in (
                "person_ids",
                "person_household_ids",
                "person_spm_ids",
                "person_years",
                "person_native_keys",
            )
        },
    )
    shuffled = replace(views, person=views.person.iloc[list(order)], scope=changed)
    result = ready(spec, changed, shuffled)
    assert result.bindings.header != original.bindings.header
    np.testing.assert_array_equal(
        result.field("WSAL_VAL").amounts,
        original.field("WSAL_VAL").amounts[list(order)],
    )
    assert list(shuffled.person.index) == [5, 2, 7, 2]
    assert views.person.attrs == {"untouched": {"metadata": "nominal"}}
    assert list(views.person.index) == [2, 2, 7, 5]


def test_missing_status_is_not_hidden_by_group_collapse():
    spec, scope, views = fixture()
    views.person["SPM_CHILDCAREXPNS"] = [np.nan, 0, 0, 0]
    with pytest.raises(MoneyRefusalError, match="INCONSISTENT_SPM_AMOUNT"):
        classify_asec_money(views, spec, scope)
    # Household years can agree while an SPM erroneously crosses cohorts.
    with pytest.raises(MoneyRefusalError, match="MIXED_SPM_YEAR"):
        replace(scope, person_spm_ids=(3, 3, 3, 3))


def test_index_refusal_is_early_and_inputs_survive_failure():
    spec, scope, views = fixture()
    original = views.person.copy(deep=True)
    bad = replace(views, person=views.person.set_axis(["a", "b", "c", "d"]))
    with pytest.raises(MoneyRefusalError, match="UNSUPPORTED_INDEX"):
        ready(spec, scope, bad)
    views.person.loc[:, "WSAL_VAL"] = -1.0
    failed = views.person.copy(deep=True)
    with pytest.raises(MoneyRefusalError):
        ready(spec, scope, views)
    pd.testing.assert_frame_equal(views.person, failed)
    views.person.loc[:, "WSAL_VAL"] = 0.0
    pd.testing.assert_frame_equal(views.person, original)


def rechecksum(payload):
    return payload[:-32] + sha256(payload[:-32]).digest()


@pytest.mark.parametrize(
    "mutation",
    [
        "truncated",
        "trailing",
        "checksum",
        "version",
        "header_size",
        "status",
        "validity",
        "origin",
        "negative_zero",
        "nan",
        "duplicate_header_key",
        "wrong_dtype",
        "wrong_shape",
        "unknown_field",
    ],
)
def test_payload_refuses_malformed_bytes_even_with_recomputed_checksum(mutation):
    spec, scope, views = fixture()
    value = ready(spec, scope, views)
    payload = encode_current_money(value)
    start = len(MAGIC) + 4
    size = struct.unpack_from("<I", payload, len(MAGIC))[0]
    body = start + size
    raw = bytearray(payload)
    if mutation == "truncated":
        raw = raw[:-1]
    elif mutation == "trailing":
        raw.extend(b"x")
    elif mutation == "checksum":
        raw[-1] ^= 1
    elif mutation == "version":
        raw[0] ^= 1
    elif mutation == "header_size":
        struct.pack_into("<I", raw, len(MAGIC), money.HEADER_MAX_BYTES + 1)
    elif mutation == "status":
        raw[body + 4 * 8] = 255
    elif mutation == "validity":
        raw[body + 4 * 9] = 0
    elif mutation == "origin":
        raw[body + 4 * 10] = 0
    elif mutation == "negative_zero":
        raw[body : body + 8] = struct.pack("<d", -0.0)
    elif mutation == "nan":
        raw[body : body + 8] = struct.pack("<d", np.nan)
    else:
        header = json.loads(value.header)
        if mutation == "wrong_dtype":
            header["dtype"] = "<f4"
        elif mutation == "wrong_shape":
            header["person_rows"] = 5
        elif mutation == "unknown_field":
            header["fields"][0] = "unknown"
        encoded = money._json(header)
        if mutation == "duplicate_header_key":
            encoded = b'{"schema_version":1,' + encoded[1:]
        raw = bytearray(
            MAGIC + struct.pack("<I", len(encoded)) + encoded + payload[body:]
        )
    corrupt = bytes(raw)
    if mutation not in ("truncated", "trailing", "checksum"):
        corrupt = rechecksum(corrupt)
    with pytest.raises(MoneyRefusalError):
        decode_current_money(corrupt, value.bindings)


def test_all_artifact_arrays_are_immutable_and_ready_is_revalidated():
    spec, scope, views = fixture()
    value = ready(spec, scope, views)
    field = value.field("ANN_VAL")
    for a in (field.amounts, field.statuses, field.validity, field.zero_origin):
        with pytest.raises(ValueError):
            a.setflags(write=True)
        with pytest.raises(ValueError):
            a[0] = 1
    malformed = replace(field, amount_bytes=bytearray(field.amount_bytes))
    with pytest.raises(MoneyRefusalError, match="FIELD_BUFFER_SHAPE"):
        encode_current_money(
            replace(
                value,
                fields=(malformed,) + value.fields[1:],
                _token=money._STATE_TOKEN,
            )
        )


@pytest.mark.parametrize("change", ["version", "bytes", "inventory", "size"])
def test_installed_microunit_drift_refuses_without_import(
    tmp_path, monkeypatch, change
):
    path = tmp_path / "tax_unit_construction.py"
    path.write_bytes(b"raise RuntimeError('must never execute')")
    distribution = SimpleNamespace(
        version="0.1.0",
        files=["microunit/tax_unit_construction.py"],
        locate_file=lambda _: path,
    )
    if change == "version":
        distribution.version = "0.2.0"
    elif change == "inventory":
        distribution.files = []
    elif change == "size":
        path.write_bytes(bytes(1024 * 1024 + 1))
    monkeypatch.setattr(loader.metadata, "distribution", lambda _: distribution)
    before = set(sys.modules)
    with pytest.raises(MoneyRefusalError, match="MICROUNIT"):
        loader.load_current_money_resources()
    assert "microunit" not in set(sys.modules) - before


def test_resource_loader_only_hashes_pinned_module_and_paths_are_not_identity(
    tmp_path, monkeypatch
):
    # Fake distribution payload validates successful loader control flow; the real
    # reviewed installed pin is checked separately in the author's local receipt.
    module = b"raise RuntimeError('must not import')"
    monkeypatch.setattr(loader, "MICROUNIT_SHA256", sha256(module).hexdigest())
    values = []
    for folder in ("one", "two"):
        path = tmp_path / folder
        path.mkdir()
        code = path / "tax_unit_construction.py"
        code.write_bytes(module)
        dist = SimpleNamespace(
            version="0.1.0",
            files=["microunit/tax_unit_construction.py"],
            locate_file=lambda _, code=code: code,
        )
        monkeypatch.setattr(loader.metadata, "distribution", lambda _, dist=dist: dist)
        values.append(loader.load_current_money_resources())
    assert values[0] == values[1]


def test_pure_module_contains_no_file_discovery_or_optional_execution():
    # Source-level boundary check complements actual hostile-to-import fixtures.
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(money))
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not imports.intersection(
        {"pathlib", "importlib", "microunit", "policyengine_us"}
    )
    calls = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert not calls.intersection({"open", "eval", "exec"})


def test_nominal_arrays_cannot_be_relabelled_through_public_state_constructors():
    spec, scope, views = fixture()
    decoded = classify_asec_money(views, spec, scope)
    for cls in (money.RestatedAsecMoney, money.SyntheticReadyCurrentMoney):
        with pytest.raises(MoneyRefusalError, match="STATE_CONSTRUCTOR_UNAVAILABLE"):
            cls(decoded.bindings, decoded.fields)
    with pytest.raises(MoneyRefusalError, match="TYPED_STATE"):
        require_complete_current_money(decoded, spec)
