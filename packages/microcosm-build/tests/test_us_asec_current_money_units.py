"""Real microunit on tiny invented, explicitly test-pinned source checkpoints."""

import importlib.util
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import _asec_current_money_codec as codec
from microcosm.build.us_runtime import asec_current_money as money
from microcosm.build.us_runtime import asec_current_money_units as units
from microcosm.build.us_runtime.asec_current_money_source import (
    load_authenticated_current_money_source,
)
from microcosm.frame import units as frame_units


def _fixture(tmp_path, monkeypatch, *, child_wages=None):
    pytest.importorskip("microunit")
    spec = importlib.util.spec_from_file_location(
        "unit_money_source_fixtures",
        Path(__file__).with_name("test_us_asec_current_money_source.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent, attachment, *_ = module.invented_checkpoints(tmp_path, monkeypatch)
    if child_wages is not None:
        from microcosm.build.us_runtime import (
            asec_current_money_source as source_module,
        )

        # Rewrite only invented fixture bytes and their PRIVATE test pins.
        for path in (parent, attachment):
            loaded = source_module.checkpoint.load_frame_checkpoint(path)
            loaded.frame.person.iloc[
                3, loaded.frame.person.columns.get_loc("WSAL_VAL")
            ] = child_wages
            if path == attachment:
                pin = money._sha(parent.read_bytes())
                loaded.metadata["parent_checkpoint_sha256"] = pin
                loaded.metadata["household_observations"]["input_checkpoint_sha256"] = (
                    pin
                )
            source_module.checkpoint.write_frame_checkpoint(
                path, loaded.frame, metadata=loaded.metadata
            )
        monkeypatch.setattr(
            source_module,
            "_SOURCE_PINS",
            (
                money._sha(parent.read_bytes()),
                money._sha(attachment.read_bytes()),
                source_module._SOURCE_PINS[2],
            ),
        )
    return load_authenticated_current_money_source(parent, attachment)


def test_real_current_basis_changes_filing_status_and_preserves_every_other_entity(
    tmp_path, monkeypatch
):
    source = _fixture(tmp_path, monkeypatch)
    ready = source.ready()
    seen = []
    original = frame_units._construct_tax_units

    def observe(construct, person, **kwargs):
        seen.append(person.copy(deep=True))
        return original(construct, person, **kwargs)

    monkeypatch.setattr(frame_units, "_construct_tax_units", observe)
    result = units.reconstruct_current_money_tax_units(source, ready)
    assert len(seen) == 1
    view = seen[0]
    assert tuple(view.columns) == units.MICROUNIT_CURRENT_INPUT_COLUMNS
    np.testing.assert_array_equal(
        view.WSAL_VAL.to_numpy(), ready.field("WSAL_VAL").amounts
    )
    assert view.ANN_VAL.tolist() == [0.0] * 6
    np.testing.assert_array_equal(
        view.SEMP_VAL.to_numpy(), ready.field("SEMP_VAL").amounts
    )
    assert view.SEMP_VAL.iloc[1] < 0 and view.SEMP_VAL.iloc[3] < 0
    assert isinstance(view.index, pd.RangeIndex) and view.index.is_unique
    assert result.frame.n("tax_unit") == 3
    assert source.frame.n("tax_unit") == 3
    assert result.frame.person.tax_unit_role_input.tolist() == ["HEAD", "DEPENDENT"] * 3
    # Actual 0.1.0 policyengine behavior: the under-15 fallback retains the
    # partition; qualifying-relative gross income changes the filing status.
    assert result.frame.table("tax_unit").filing_status_input.tolist() == [
        "SINGLE",
        "HEAD_OF_HOUSEHOLD",
        "HEAD_OF_HOUSEHOLD",
    ]
    assert result.money is ready
    result.validate()
    assert (
        result.receipt["year"] == 2024 and result.receipt["release_eligible"] is False
    )
    for entity in source.frame.entities:
        if entity not in ("person", "tax_unit"):
            pd.testing.assert_frame_equal(
                source.frame.table(entity), result.frame.table(entity)
            )
    untouched = [c for c in source.frame.person if c not in units.TAX_PERSON_OUTPUTS]
    pd.testing.assert_frame_equal(
        source.frame.person[untouched], result.frame.person[untouched]
    )
    pd.testing.assert_series_equal(source.frame.strata, result.frame.strata)
    assert (
        source.frame.metadata == result.frame.metadata
        and source.frame.mass_log == result.frame.mass_log
    )
    assert (
        source.frame.weights_for("household").values.tobytes()
        == result.frame.weights_for("household").values.tobytes()
    )
    source.validate()


def test_transient_view_excludes_every_undeclared_raw_and_harmonized_alternate(
    tmp_path, monkeypatch
):
    source = _fixture(tmp_path, monkeypatch)
    ready = source.ready()
    view = units.current_money_microunit_view(source, ready)
    assert (
        "ED_VAL" not in view and "SPM_ENGVAL" not in view and "source_year" not in view
    )
    assert not set(view).intersection(
        {
            "wage_income",
            "employment_income_before_lsr",
            "total_person_income",
            "source_row_id",
            "person_id",
        }
    )
    assert set(c for c in view if c.endswith("_VAL")) == set(
        units.MICROUNIT_CURRENT_MONEY_COLUMNS
    ) - {"PTOTVAL"}


def test_actual_nominal_comparator_retains_the_original_three_tax_units(
    tmp_path, monkeypatch
):
    source = _fixture(tmp_path, monkeypatch)
    import microunit

    person = (
        source.frame.person.loc[:, units.MICROUNIT_CURRENT_INPUT_COLUMNS]
        .copy()
        .reset_index(drop=True)
    )
    nominal, tax_unit = frame_units._construct_tax_units(
        microunit.construct_tax_units, person, year=2024, mode="policyengine"
    )
    assert nominal.TAX_ID.nunique() == 3
    assert nominal.tax_unit_role_input.tolist() == ["HEAD", "DEPENDENT"] * 3
    assert tax_unit.filing_status_input.tolist() == ["HEAD_OF_HOUSEHOLD"] * 3


@pytest.mark.parametrize(
    "changed_name", ["rule_helpers.py", "dependent_gross_income_limit.yaml"]
)
def test_changed_rule_helpers_or_parameter_file_refuses_before_engine(
    tmp_path, monkeypatch, changed_name
):
    source = _fixture(tmp_path, monkeypatch)
    ready = source.ready()
    original = units._runtime_file_bytes

    def changed(path):
        value = original(path)
        return value + b"\nchanged" if str(path).endswith(changed_name) else value

    monkeypatch.setattr(units, "_runtime_file_bytes", changed)
    with pytest.raises(money.MoneyRefusalError, match="MICROUNIT_EXECUTION_PIN"):
        units.reconstruct_current_money_tax_units(source, ready)


def test_unit_adapter_refuses_a_post_load_structure_change_before_engine(
    tmp_path, monkeypatch
):
    source = _fixture(tmp_path, monkeypatch)
    ready = source.ready()
    source.frame.person.iloc[0, source.frame.person.columns.get_loc("A_AGE")] = 99
    with pytest.raises(money.MoneyRefusalError, match="SOURCE_CHANGED"):
        units.reconstruct_current_money_tax_units(source, ready)


def test_tax_result_rebind_refuses_later_changes_and_cannot_run_as_raw_source(
    tmp_path, monkeypatch
):
    source = _fixture(tmp_path, monkeypatch)
    ready = source.ready()
    result = units.reconstruct_current_money_tax_units(source, ready)
    with pytest.raises(money.MoneyRefusalError, match="AUTHENTICATED_MONEY_REQUIRED"):
        units.reconstruct_current_money_tax_units(result, ready)
    with pytest.raises(
        money.MoneyRefusalError, match="TAX_RESULT_CONSTRUCTOR_UNAVAILABLE"
    ):
        units.CurrentMoneyTaxUnitResult(result.frame, ready, result._receipt)
    result.frame.person.iloc[0, result.frame.person.columns.get_loc("WSAL_VAL")] = 1.0
    with pytest.raises(money.MoneyRefusalError, match="TAX_RESULT_CHANGED"):
        result.validate()


@pytest.mark.parametrize(
    "column,change", [("A_ENRLW", "missing"), ("PEDISDRS", "boolean")]
)
def test_authenticated_missing_or_wrongly_typed_controls_refuse_without_engine_call(
    tmp_path, monkeypatch, column, change
):
    spec = importlib.util.spec_from_file_location(
        "unit_bad_control_fixtures",
        Path(__file__).with_name("test_us_asec_current_money_source.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent, attachment, *_ = module.invented_checkpoints(tmp_path, monkeypatch)
    from microcosm.build.us_runtime import asec_current_money_source as source_module

    for path in (parent, attachment):
        loaded = source_module.checkpoint.load_frame_checkpoint(path)
        if change == "missing":
            loaded.frame.person.drop(columns=[column], inplace=True)
        else:
            loaded.frame.person[column] = True
        if path == parent:
            identity = source_module.frame_identity(loaded.frame).to_payload()
            loaded.metadata["identity"] = identity
            loaded.metadata["source_construction_identity"] = identity
        if path == attachment:
            pin = money._sha(parent.read_bytes())
            loaded.metadata["parent_checkpoint_sha256"] = pin
            loaded.metadata["household_observations"]["input_checkpoint_sha256"] = pin
        source_module.checkpoint.write_frame_checkpoint(
            path, loaded.frame, metadata=loaded.metadata
        )
    monkeypatch.setattr(
        source_module,
        "_SOURCE_PINS",
        (
            money._sha(parent.read_bytes()),
            money._sha(attachment.read_bytes()),
            source_module._SOURCE_PINS[2],
        ),
    )
    source = load_authenticated_current_money_source(parent, attachment)
    ready = source.ready()

    def unexpected(*args, **kwargs):
        raise AssertionError("Engine must not run before control validation")

    monkeypatch.setattr(frame_units, "_construct_tax_units", unexpected)
    with pytest.raises(
        money.MoneyRefusalError, match="MICROUNIT_INPUT_ROSTER|MICROUNIT_CONTROL_DTYPE"
    ):
        units.reconstruct_current_money_tax_units(source, ready)


def test_reconstructed_tax_filing_status_retains_canonical_source_dtype(
    tmp_path, monkeypatch
):
    from microcosm.build.serialization_dtypes import CANONICAL_STRING_DTYPE

    source = _fixture(tmp_path, monkeypatch)
    result = units.reconstruct_current_money_tax_units(source, source.ready())
    incumbent = source.frame.table("tax_unit").filing_status_input
    actual = result.frame.table("tax_unit").filing_status_input
    assert incumbent.dtype == CANONICAL_STRING_DTYPE
    assert actual.dtype == incumbent.dtype
    assert actual.dtype.storage == incumbent.dtype.storage
    assert actual.tolist() == ["SINGLE", "HEAD_OF_HOUSEHOLD", "HEAD_OF_HOUSEHOLD"]


def test_tax_result_receipt_survives_actual_checkpoint_and_string_boundary(
    tmp_path, monkeypatch
):
    from microcosm.build.frame_checkpoint import (
        load_frame_checkpoint,
        write_frame_checkpoint,
    )
    from microcosm.build.serialization_dtypes import canonicalize_frame_string_dtypes

    source = _fixture(tmp_path, monkeypatch)
    result = units.reconstruct_current_money_tax_units(source, source.ready())
    output = tmp_path / "invented-tax-result.h5"
    write_frame_checkpoint(
        output, result.frame, metadata={"tax_result": result.receipt}
    )
    loaded = load_frame_checkpoint(output)
    assert loaded.metadata["tax_result"][
        "money_content_sha256"
    ] == codec.current_money_content_sha256(result.money)
    assert loaded.metadata["tax_result"]["money_header_sha256"] == money._sha(
        result.money.header
    )
    expected = loaded.metadata["tax_result"]["output_frame_sha256"]
    assert units._frame_signature(loaded.frame) == expected
    canonicalize_frame_string_dtypes(
        loaded.frame, boundary="test tax result reload", in_place=True
    )
    assert units._frame_signature(loaded.frame) == expected
    for entity in result.frame.entities:
        pd.testing.assert_frame_equal(
            result.frame.table(entity), loaded.frame.table(entity), check_exact=True
        )


def test_tax_result_rejects_changed_frame_core_implementation(tmp_path, monkeypatch):
    from microcosm.frame import schema

    source = _fixture(tmp_path, monkeypatch)
    result = units.reconstruct_current_money_tax_units(source, source.ready())
    result.validate()
    core_path = Path(schema.__file__).resolve()
    original_read = Path.read_bytes

    def changed_core_bytes(path):
        value = original_read(path)
        return (
            value + b"\n# changed core behavior"
            if path.resolve() == core_path
            else value
        )

    monkeypatch.setattr(Path, "read_bytes", changed_core_bytes)
    with pytest.raises(money.MoneyRefusalError, match="SOURCE_IMPLEMENTATION_CHANGED"):
        result.validate()


def _changed_fields(ready, mutation="nonzero"):
    name = (
        "ANN_VAL"
        if mutation == "status_only"
        else "CAP_VAL"
        if mutation == "zero_to_amount"
        else "WSAL_VAL"
    )
    position = money.FIELDS.index(name)
    field = ready.fields[position]
    amounts, statuses, origin = (
        field.amounts.copy(),
        field.statuses.copy(),
        field.zero_origin.copy(),
    )
    row = 0 if mutation == "zero_to_amount" else 3
    if mutation == "status_only":
        # Both zero-dollar and declared-NIU states are valid for ANN_VAL.
        # Change only its meaning and zero provenance, leaving amounts intact.
        assert amounts[row] == 0.0
        assert statuses[row] == money.CodebookStatus.DECLARED_NIU
        assert origin[row] == 0
        statuses[row] = money.CodebookStatus.ZERO_DOLLARS_AS_CODED
        origin[row] = 1
    elif mutation == "amount_to_zero":
        amounts[row] = 0.0
        statuses[row] = money._ZERO_STATUS[
            ready.bindings.spec.fields[position].zero_semantics
        ]
        origin[row] = 1
    else:
        amounts[row] = 1_000_000.0 if mutation == "nonzero" else 100.0
        statuses[row] = money.CodebookStatus.AMOUNT_NONZERO
        origin[row] = 0
    changed = replace(
        field,
        amount_bytes=amounts.tobytes(),
        status_bytes=statuses.tobytes(),
        zero_origin_bytes=origin.tobytes(),
    )
    money._validate_field(
        changed, ready.bindings.spec.fields[position], len(amounts), nominal=False
    )
    if mutation == "status_only":
        assert changed.amount_bytes == field.amount_bytes
        assert changed.validity_bytes == field.validity_bytes
        assert changed.status_bytes != field.status_bytes
        assert changed.zero_origin_bytes != field.zero_origin_bytes
    fields = list(ready.fields)
    fields[position] = changed
    return tuple(fields)


def _changed_payload(ready, fields):
    """Change public transport bytes and recompute their public checksum."""
    payload = bytearray(codec.encode_current_money(ready))
    cursor = len(codec.MAGIC) + 4 + len(ready.header)
    for field in fields:
        for part in (
            field.amount_bytes,
            field.status_bytes,
            field.validity_bytes,
            field.zero_origin_bytes,
        ):
            payload[cursor : cursor + len(part)] = part
            cursor += len(part)
    payload[-32:] = sha256(payload[:-32]).digest()
    return bytes(payload)


def test_replay_bindings_alone_cannot_authorize_rechecksummed_changed_wages(
    tmp_path, monkeypatch
):
    source = _fixture(tmp_path, monkeypatch)
    ready = source.ready()
    changed = _changed_payload(ready, _changed_fields(ready))
    with pytest.raises(money.MoneyRefusalError, match="EXPECTED_CONTENT_REQUIRED"):
        codec.decode_current_money(changed, ready.bindings)


@pytest.mark.parametrize(
    "mutation", ["nonzero", "zero_to_amount", "amount_to_zero", "status_only"]
)
def test_replay_refuses_invariant_valid_changed_content_before_state_issuance(
    tmp_path, monkeypatch, mutation
):
    source = _fixture(tmp_path, monkeypatch)
    ready = source.ready()
    changed = _changed_payload(ready, _changed_fields(ready, mutation))

    def unexpected(*args, **kwargs):
        raise AssertionError("No state may be issued for changed replay content")

    monkeypatch.setattr(codec, "RestatedAsecMoney", unexpected)
    with pytest.raises(money.MoneyRefusalError, match="REPLAY_CONTENT"):
        codec.decode_current_money(changed, ready.bindings, expected=ready)


def test_replay_expected_content_is_anchored_and_deterministic(tmp_path, monkeypatch):
    source = _fixture(tmp_path, monkeypatch)
    ready = source.ready()
    independent = source.ready()
    assert independent == ready
    payload = codec.encode_current_money(ready)
    replay = codec.decode_current_money(payload, ready.bindings, expected=independent)
    second = codec.decode_current_money(payload, ready.bindings, expected=replay)
    assert type(second) is money.ReadyCurrentMoney
    assert second == ready and codec.encode_current_money(second) == payload
    assert codec.current_money_content_sha256(
        second
    ) == codec.current_money_content_sha256(ready)
    assert (
        codec.current_money_content_sha256(ready) != sha256(payload[:-32]).hexdigest()
    )


@pytest.mark.parametrize("kind", ["absent", "object", "synthetic"])
def test_replay_refuses_wrong_expected_type(tmp_path, monkeypatch, kind):
    source = _fixture(tmp_path, monkeypatch)
    ready = source.ready()
    # Private test-only construction supplies a wrong runtime type, not authority.
    expected = {
        "absent": None,
        "object": object(),
        "synthetic": money.SyntheticReadyCurrentMoney(
            ready.bindings, ready.fields, _token=money._STATE_TOKEN
        ),
    }[kind]
    with pytest.raises(money.MoneyRefusalError, match="EXPECTED_CONTENT_REQUIRED"):
        codec.decode_current_money(
            codec.encode_current_money(ready), ready.bindings, expected=expected
        )


def test_replay_refuses_expected_from_different_source(tmp_path, monkeypatch):
    first_path, second_path = tmp_path / "first", tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    first = _fixture(first_path, monkeypatch).ready()
    second = _fixture(second_path, monkeypatch).ready()
    assert first.bindings != second.bindings
    with pytest.raises(money.MoneyRefusalError, match="EXPECTED_CONTENT_BINDING"):
        codec.decode_current_money(
            codec.encode_current_money(first), first.bindings, expected=second
        )


@pytest.mark.parametrize(
    "mutation", ["nonzero", "zero_to_amount", "amount_to_zero", "status_only"]
)
def test_consumer_rejects_private_test_forged_ready_before_engine(
    tmp_path, monkeypatch, mutation
):
    source = _fixture(tmp_path, monkeypatch)
    ready = source.ready()
    forged = money.ReadyCurrentMoney(
        ready.bindings, _changed_fields(ready, mutation), _token=money._STATE_TOKEN
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("Engine must not run on a changed money artifact")

    monkeypatch.setattr(frame_units, "_construct_tax_units", unexpected)
    with pytest.raises(money.MoneyRefusalError, match="MONEY_CONTENT_MISMATCH"):
        units.reconstruct_current_money_tax_units(source, forged)


@pytest.mark.parametrize("mutation", ["nonzero", "status_only"])
def test_result_receipt_binds_money_body_not_only_header(
    tmp_path, monkeypatch, mutation
):
    source = _fixture(tmp_path, monkeypatch)
    ready = source.ready()
    result = units.reconstruct_current_money_tax_units(source, ready)
    forged = money.ReadyCurrentMoney(
        ready.bindings, _changed_fields(ready, mutation), _token=money._STATE_TOKEN
    )
    assert forged.header == ready.header
    replaced = replace(result, money=forged, _token=units._RESULT_TOKEN)
    with pytest.raises(money.MoneyRefusalError, match="TAX_RESULT_CHANGED"):
        replaced.validate()
    assert (
        codec.current_money_content_sha256(forged)
        != result.receipt["money_content_sha256"]
    )


def test_reconstruction_reclassifies_exactly_once_and_validates_source_twice(
    tmp_path, monkeypatch
):
    from microcosm.build.us_runtime import asec_current_money_source as source_module

    source = _fixture(tmp_path, monkeypatch)
    ready = source.ready()
    actual_classify = source_module.classify_asec_money
    actual_validate = type(source).validate
    calls = {"classify": 0, "validate": 0}

    def classify(*args, **kwargs):
        calls["classify"] += 1
        return actual_classify(*args, **kwargs)

    def validate(self):
        calls["validate"] += 1
        return actual_validate(self)

    monkeypatch.setattr(source_module, "classify_asec_money", classify)
    monkeypatch.setattr(type(source), "validate", validate)
    units.reconstruct_current_money_tax_units(source, ready)
    assert calls == {"classify": 1, "validate": 2}


def test_real_different_money_results_persist_different_content_receipts(
    tmp_path, monkeypatch
):
    from microcosm.build.frame_checkpoint import (
        load_frame_checkpoint,
        write_frame_checkpoint,
    )

    receipts = []
    statuses = []
    for index, wages in enumerate((4800.0, 1_000_000.0)):
        directory = tmp_path / str(index)
        directory.mkdir()
        source = _fixture(directory, monkeypatch, child_wages=wages)
        result = units.reconstruct_current_money_tax_units(source, source.ready())
        result.validate()
        path = directory / "result.h5"
        write_frame_checkpoint(
            path, result.frame, metadata={"tax_result": result.receipt}
        )
        replay = load_frame_checkpoint(path)
        receipt = replay.metadata["tax_result"]
        assert receipt["money_content_sha256"] == codec.current_money_content_sha256(
            result.money
        )
        receipts.append(receipt)
        statuses.append(replay.frame.table("tax_unit").filing_status_input.iloc[1])
    assert receipts[0]["money_content_sha256"] != receipts[1]["money_content_sha256"]
    assert statuses == ["HEAD_OF_HOUSEHOLD", "SINGLE"]
