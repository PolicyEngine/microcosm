"""Real microunit on tiny invented, explicitly test-pinned source checkpoints."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import asec_current_money as money
from microcosm.build.us_runtime import asec_current_money_units as units
from microcosm.build.us_runtime.asec_current_money_source import (
    load_authenticated_current_money_source,
)
from microcosm.frame import units as frame_units


def _fixture(tmp_path, monkeypatch):
    pytest.importorskip("microunit")
    spec = importlib.util.spec_from_file_location(
        "unit_money_source_fixtures",
        Path(__file__).with_name("test_us_asec_current_money_source.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent, attachment, *_ = module.invented_checkpoints(tmp_path, monkeypatch)
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
