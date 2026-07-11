from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import populace.build.us_runtime.acs_transfer as acs_transfer_module
from populace.build.us_runtime.acs_transfer import (
    ACS_DONOR_CHANNEL_AUTO,
    ACS_GROUP_TRANSFER_PREDICTORS,
    ACS_NATIVE_PERSON_INPUTS,
    ACS_OPTIONAL_PERSON_TRANSFER_PREDICTORS,
    ACS_PERSON_TRANSFER_PREDICTORS,
    AcsTransferResult,
    default_acs_transfer_target_families,
    transfer_acs_inputs,
)
from populace.frame import EntitySchema, Frame, WeightKind, Weights

SCHEMA = EntitySchema(group_entities=("household", "tax_unit"))


def _donor_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 9, dtype=np.int64),
            "person_household_id": np.asarray(
                [100, 100, 200, 200, 300, 300, 400, 400], dtype=np.int64
            ),
            "person_tax_unit_id": np.asarray(
                [10, 10, 20, 20, 30, 30, 40, 40], dtype=np.int64
            ),
            "age": [45.0, 43.0, 30.0, 8.0, 68.0, 66.0, 39.0, 17.0],
            "is_female": [False, True, True, False, False, True, True, False],
            "is_household_head": [True, False, True, False, True, False, True, False],
            "employment_income_before_lsr": [
                80_000.0,
                np.nan,
                35_000.0,
                0.0,
                0.0,
                0.0,
                120_000.0,
                4_000.0,
            ],
            "self_employment_income_before_lsr": [
                np.nan,
                5_000.0,
                0.0,
                0.0,
                1_000.0,
                0.0,
                20_000.0,
                0.0,
            ],
            "taxable_interest_income": [
                500.0,
                100.0,
                50.0,
                0.0,
                2_000.0,
                1_000.0,
                5_000.0,
                10.0,
            ],
            "qualified_dividend_income": [
                900.0,
                200.0,
                80.0,
                20.0,
                3_000.0,
                1_500.0,
                8_000.0,
                40.0,
            ],
            "pre_subsidy_rent": [
                0.0,
                0.0,
                18_000.0,
                0.0,
                0.0,
                0.0,
                24_000.0,
                0.0,
            ],
            "real_estate_taxes": [
                6_000.0,
                0.0,
                0.0,
                0.0,
                2_000.0,
                0.0,
                9_000.0,
                0.0,
            ],
            "takes_up_medicaid_if_eligible": [
                False,
                False,
                True,
                True,
                False,
                True,
                False,
                True,
            ],
        },
        index=pd.Index(np.arange(101, 109), name="donor_person_row"),
    )
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": np.asarray([10, 20, 30, 40], dtype=np.int64),
            "first_home_mortgage_balance": [250_000.0, 120_000.0, 5_000.0, 400_000.0],
            "takes_up_eitc": [False, True, False, True],
        },
        index=pd.Index([201, 202, 203, 204], name="donor_tax_unit_row"),
    )
    household = pd.DataFrame(
        {
            "household_id": np.asarray([100, 200, 300, 400], dtype=np.int64),
            "state_fips": [6, 6, 36, 36],
            "tenure_type": [
                "OWNED_WITH_MORTGAGE",
                "RENTED",
                "OWNED_OUTRIGHT",
                "RENTED",
            ],
        },
        index=pd.Index([301, 302, 303, 304], name="donor_household_row"),
    )
    return Frame(
        {"person": person, "household": household, "tax_unit": tax_unit},
        SCHEMA,
        {
            "household": Weights(
                np.asarray([100.0, 200.0, 150.0, 250.0]),
                WeightKind.CALIBRATED,
            )
        },
        pd.Series(
            "asec_puf",
            index=person.index,
            dtype=object,
        ),
    )


def _recipient_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": np.arange(101, 107, dtype=np.int64),
            "person_household_id": np.asarray(
                [1_100, 1_100, 1_200, 1_300, 1_300, 1_300], dtype=np.int64
            ),
            "person_tax_unit_id": np.asarray(
                [110, 110, 120, 130, 130, 130], dtype=np.int64
            ),
            "age": [41.0, 40.0, 27.0, 72.0, 69.0, 15.0],
            "is_female": [False, True, True, False, True, False],
            "is_household_head": [True, False, True, True, False, False],
            "employment_income_before_lsr": [
                65_000.0,
                np.nan,
                48_000.0,
                0.0,
                0.0,
                2_000.0,
            ],
            "self_employment_income_before_lsr": [
                np.nan,
                3_000.0,
                10_000.0,
                0.0,
                0.0,
                0.0,
            ],
            # Native ACS mapping wins even though the donor offers this PUF leaf.
            "taxable_interest_income": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            "native_label": ["a", "b", "c", "d", "e", "f"],
        },
        index=pd.Index(np.arange(501, 507), name="acs_person_row"),
    )
    tax_unit = pd.DataFrame(
        {
            "tax_unit_id": np.asarray([110, 120, 130], dtype=np.int64),
            "native_tax_unit_marker": [1.0, 2.0, 3.0],
        },
        index=pd.Index([601, 602, 603], name="acs_tax_unit_row"),
    )
    household = pd.DataFrame(
        {
            "household_id": np.asarray([1_100, 1_200, 1_300], dtype=np.int64),
            "state_fips": [6, 36, 36],
            "tenure_type": ["OWNED_WITH_MORTGAGE", "RENTED", "RENTED"],
        },
        index=pd.Index([701, 702, 703], name="acs_household_row"),
    )
    return Frame(
        {"person": person, "household": household, "tax_unit": tax_unit},
        SCHEMA,
        {"household": Weights(np.asarray([50.0, 60.0, 70.0]), WeightKind.DESIGN)},
        pd.Series("acs_2024_1yr", index=person.index, dtype=object),
    )


def _replace_column(
    frame: Frame,
    entity: str,
    column: str,
    values: object,
) -> Frame:
    tables = {name: frame.table(name).copy() for name in frame.entities}
    tables[entity][column] = values
    return Frame(
        tables,
        frame.schema,
        {name: frame.weights_for(name) for name in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _drop_column(frame: Frame, entity: str, column: str) -> Frame:
    tables = {name: frame.table(name).copy() for name in frame.entities}
    tables[entity] = tables[entity].drop(columns=[column])
    return Frame(
        tables,
        frame.schema,
        {name: frame.weights_for(name) for name in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _with_columns(
    frame: Frame,
    entity: str,
    columns: dict[str, object],
) -> Frame:
    tables = {name: frame.table(name).copy() for name in frame.entities}
    for column, values in columns.items():
        tables[entity][column] = values
    return Frame(
        tables,
        frame.schema,
        {name: frame.weights_for(name) for name in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _conditioning_frames() -> tuple[Frame, Frame]:
    donor = _with_columns(
        _donor_frame(),
        "person",
        {
            "social_security_retirement": [
                100.0,
                0.0,
                0.0,
                0.0,
                800.0,
                700.0,
                0.0,
                0.0,
            ],
            "social_security_disability": [0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "social_security_dependents": [0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0],
            "social_security_survivors": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 10.0],
            "taxable_private_pension_income": [
                0.0,
                0.0,
                0.0,
                0.0,
                900.0,
                800.0,
                0.0,
                0.0,
            ],
            "tax_exempt_private_pension_income": [
                0.0,
                0.0,
                0.0,
                0.0,
                100.0,
                50.0,
                0.0,
                0.0,
            ],
            "taxable_ira_distributions": [0.0, 0.0, 0.0, 0.0, 200.0, 100.0, 0.0, 0.0],
            "tax_exempt_interest_income": [0.0, 0.0, 0.0, 0.0, 50.0, 20.0, 0.0, 0.0],
            "non_qualified_dividend_income": [
                50.0,
                10.0,
                0.0,
                0.0,
                100.0,
                50.0,
                200.0,
                0.0,
            ],
            "rental_income": [0.0, 0.0, 0.0, 0.0, 300.0, 0.0, 500.0, 0.0],
            "estate_income": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, np.nan],
        },
    )
    recipient = _with_columns(
        _recipient_frame(),
        "person",
        {
            "acs_social_security_income": [100.0, np.nan, 50.0, 0.0, np.nan, 10.0],
            "acs_retirement_income": [0.0, np.nan, 0.0, 1_200.0, np.nan, 0.0],
            "acs_interest_dividend_rental_income": [
                -100.0,
                np.nan,
                80.0,
                2_000.0,
                np.nan,
                0.0,
            ],
        },
    )
    return donor, recipient


def _channel_copy(
    frame: Frame,
    *,
    channel: str,
    offset: int,
    target_value: float,
) -> Frame:
    tables = {name: frame.table(name).copy() for name in frame.entities}
    person = tables["person"]
    person["person_id"] = person["person_id"] + offset
    person["person_household_id"] = person["person_household_id"] + offset
    person["person_tax_unit_id"] = person["person_tax_unit_id"] + offset
    person["person_support_channel"] = channel
    person["qualified_dividend_income"] = target_value
    tables["household"]["household_id"] = tables["household"]["household_id"] + offset
    tables["tax_unit"]["tax_unit_id"] = tables["tax_unit"]["tax_unit_id"] + offset
    return Frame(
        tables,
        frame.schema,
        {name: frame.weights_for(name) for name in frame.weighted_entities},
        pd.Series(channel, index=person.index, dtype=object),
        mass_log=frame.mass_log,
    )


class _MeanQRF:
    calls: list[dict[str, object]] = []

    def __init__(self, *, n_estimators: int, seed: int) -> None:
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(
        self,
        frame: Frame,
        predictors: list[str],
        targets: list[str],
        *,
        weights: str,
    ) -> _MeanFitted:
        entity = frame.column_entity(targets[0])
        table = frame.table(entity)
        weight_kind = frame.resolve_weights(entity).kind.value
        assert weights == weight_kind
        call = {
            "entity": entity,
            "predictors": tuple(predictors),
            "features": table.loc[:, predictors].copy(),
            "targets": table.loc[:, targets].copy(),
            "seed": self.seed,
            "weight_kind": weight_kind,
        }
        self.calls.append(call)
        means = {target: float(table[target].mean()) for target in targets}
        return _MeanFitted(means, weight_kind)


class _MeanFitted:
    def __init__(self, means: dict[str, float], weight_kind: str) -> None:
        self.means = means
        self.weight_kind = weight_kind

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                target: np.full(len(frame), value, dtype=np.float64)
                for target, value in self.means.items()
            },
            index=frame.index,
        )


def test_default_transfer_preserves_native_fields_and_registers_added_inputs() -> None:
    donor = _donor_frame()
    recipient = _recipient_frame()
    native_person = recipient.table("person").copy(deep=True)
    native_tax_unit = recipient.table("tax_unit").copy(deep=True)

    result = transfer_acs_inputs(
        recipient,
        donor,
        donor_spine="asec_puf_test",
        seed=17,
        n_estimators=5,
    )

    person = result.frame.table("person")
    tax_unit = result.frame.table("tax_unit")
    pd.testing.assert_series_equal(
        person["taxable_interest_income"],
        native_person["taxable_interest_income"],
    )
    pd.testing.assert_series_equal(
        person["native_label"], native_person["native_label"]
    )
    pd.testing.assert_series_equal(
        tax_unit["native_tax_unit_marker"],
        native_tax_unit["native_tax_unit_marker"],
    )
    assert set(person.columns) - set(native_person.columns) == {
        "pre_subsidy_rent",
        "qualified_dividend_income",
        "takes_up_medicaid_if_eligible",
    }
    assert set(tax_unit.columns) - set(native_tax_unit.columns) == {
        "first_home_mortgage_balance",
        "takes_up_eitc",
    }
    assert person["takes_up_medicaid_if_eligible"].dtype == np.dtype(bool)
    assert tax_unit["takes_up_eitc"].dtype == np.dtype(bool)
    returned_columns = set().union(
        *(set(result.frame.table(entity).columns) for entity in result.frame.entities)
    )
    assert not any(column.startswith("__acs_transfer_") for column in returned_columns)

    provenance = {entry.column: entry for entry in result.imputed_inputs}
    assert set(provenance) == {
        "pre_subsidy_rent",
        "qualified_dividend_income",
        "takes_up_medicaid_if_eligible",
        "first_home_mortgage_balance",
        "takes_up_eitc",
    }
    assert all(entry.donor_spine == "asec_puf_test" for entry in provenance.values())
    assert all(entry.donor_channel is None for entry in provenance.values())
    assert all(entry.weight_kind == "calibrated" for entry in provenance.values())
    assert all(entry.patterns for entry in provenance.values())
    assert all(
        pattern.weight_kind == "calibrated"
        for entry in provenance.values()
        for pattern in entry.patterns
    )
    assert provenance["qualified_dividend_income"].entity == "person"
    assert provenance["first_home_mortgage_balance"].entity == "tax_unit"
    assert set(ACS_GROUP_TRANSFER_PREDICTORS).issubset(
        provenance["first_home_mortgage_balance"].predictors
    )
    housing = provenance["pre_subsidy_rent"]
    assert all(
        "__acs_transfer_is_household_head" in pattern.predictors
        and "__acs_transfer_tenure_code" in pattern.predictors
        for pattern in housing.patterns
    )
    assert {record.weight_kind for record in result.fit_records} == {"calibrated"}
    fit_names = {record.fit_name for record in result.fit_records}
    for prefix in (
        "acs_transfer:person:benefit_participation",
        "acs_transfer:person:housing",
        "acs_transfer:person:puf_tax_itemization",
        "acs_transfer:tax_unit:benefit_participation",
        "acs_transfer:tax_unit:puf_tax_itemization",
    ):
        assert any(name.startswith(prefix) for name in fit_names)


def test_default_families_include_puf_leaves_and_dynamic_take_up_columns() -> None:
    families = default_acs_transfer_target_families(_donor_frame())

    assert "qualified_dividend_income" in families["person"]["puf_tax_itemization"]
    assert families["person"]["housing"] == ("pre_subsidy_rent",)
    assert ACS_NATIVE_PERSON_INPUTS.isdisjoint(
        families["person"]["puf_tax_itemization"]
    )
    assert families["person"]["benefit_participation"] == (
        "takes_up_medicaid_if_eligible",
    )
    assert (
        "first_home_mortgage_balance" in (families["tax_unit"]["puf_tax_itemization"])
    )
    assert families["tax_unit"]["benefit_participation"] == ("takes_up_eitc",)


def test_default_families_read_full_required_inventory_without_geography(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor = _with_columns(
        _donor_frame(),
        "person",
        {
            "has_esi": [True, False, True, False, True, False, True, False],
            "is_blind": [False, False, False, False, True, False, False, False],
            "ssn_card_type": ["CITIZEN"] * 6 + ["NONE", "NONE"],
        },
    )
    donor = _with_columns(
        donor,
        "household",
        {"county_fips": [1, 3, 5, 7]},
    )
    monkeypatch.setattr(
        acs_transfer_module,
        "load_release_input_coverage_manifest",
        lambda: SimpleNamespace(
            required_columns=frozenset(
                {
                    "age",
                    "county_fips",
                    "employment_income_before_lsr",
                    "has_esi",
                    "is_blind",
                    "ssn_card_type",
                }
            )
        ),
    )

    families = default_acs_transfer_target_families(donor)

    assert families["person"]["model_required_boolean"] == (
        "has_esi",
        "is_blind",
    )
    assert families["person"]["model_required_categorical"] == ("ssn_card_type",)
    declared = {
        target
        for entity_families in families.values()
        for targets in entity_families.values()
        for target in targets
    }
    assert "county_fips" not in declared
    assert "employment_income_before_lsr" not in declared
    assert "age" not in declared


def test_default_transfer_adds_every_donor_observed_required_model_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor = _with_columns(
        _donor_frame(),
        "person",
        {
            "has_esi": [True, False, True, False, True, False, True, False],
            "is_blind": [False, False, False, False, True, False, False, False],
            "other_medical_expenses": [10.0, 0.0, 5.0, 0.0, 8.0, 1.0, 0.0, 2.0],
            "ssn_card_type": ["CITIZEN"] * 6 + ["NONE", "NONE"],
            "immigration_status_str": ["CITIZEN"] * 6
            + ["UNDOCUMENTED", "UNDOCUMENTED"],
        },
    )
    required = {
        "has_esi",
        "is_blind",
        "other_medical_expenses",
        "ssn_card_type",
        "immigration_status_str",
    }
    monkeypatch.setattr(
        acs_transfer_module,
        "load_release_input_coverage_manifest",
        lambda: SimpleNamespace(required_columns=frozenset(required)),
    )
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    _MeanQRF.calls = []

    result = transfer_acs_inputs(
        _recipient_frame(),
        donor,
        seed=4,
        n_estimators=1,
        max_targets_per_fit=1,
    )

    person = result.frame.person
    assert required.issubset(person.columns)
    assert {entry.column for entry in result.imputed_inputs}.issuperset(required)
    required_families = {
        entry.family.split("__batch_", 1)[0]
        for entry in result.imputed_inputs
        if entry.column in required
    }
    assert required_families == {
        "model_required_boolean",
        "model_required_categorical",
        "model_required_numeric",
    }
    assert set(person["ssn_card_type"]) <= {"CITIZEN", "NONE"}
    observed_pairs = set(
        zip(
            donor.person["ssn_card_type"],
            donor.person["immigration_status_str"],
            strict=True,
        )
    )
    imputed_pairs = set(
        zip(
            person["ssn_card_type"],
            person["immigration_status_str"],
            strict=True,
        )
    )
    assert imputed_pairs <= observed_pairs
    joint_calls = [
        call
        for call in _MeanQRF.calls
        if "__acs_transfer_immigration_status_pair" in call["targets"]
    ]
    assert joint_calls
    assert all(
        "ssn_card_type" not in call["targets"]
        and "immigration_status_str" not in call["targets"]
        for call in joint_calls
    )


def test_large_target_family_is_split_to_bound_retained_qrf_forests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = tuple(f"wide_target_{index:02d}" for index in range(11))
    donor = _with_columns(
        _donor_frame(),
        "person",
        {
            target: np.arange(8, dtype=np.float64) + index
            for index, target in enumerate(targets)
        },
    )
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    _MeanQRF.calls = []

    result = transfer_acs_inputs(
        _recipient_frame(),
        donor,
        target_families={"person": {"wide_numeric": targets}},
        n_estimators=1,
        max_targets_per_fit=3,
    )

    assert set(targets).issubset(result.frame.person.columns)
    assert max(len(call["targets"].columns) for call in _MeanQRF.calls) <= 3
    families = {entry.family for entry in result.imputed_inputs}
    assert families == {
        "wide_numeric__batch_1",
        "wide_numeric__batch_2",
        "wide_numeric__batch_3",
        "wide_numeric__batch_4",
    }


def test_discrete_year_predictions_snap_to_observed_donor_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor = _with_columns(
        _donor_frame(),
        "tax_unit",
        {
            "first_home_mortgage_origination_year": [
                2008.0,
                2018.0,
                2008.0,
                2018.0,
            ]
        },
    )
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    _MeanQRF.calls = []

    result = transfer_acs_inputs(
        _recipient_frame(),
        donor,
        target_families={
            "tax_unit": {
                "itemization": ("first_home_mortgage_origination_year",),
            }
        },
        seed=2,
        n_estimators=1,
    )

    values = result.frame.table("tax_unit")["first_home_mortgage_origination_year"]
    assert set(values).issubset({2008.0, 2018.0})
    assert pd.api.types.is_integer_dtype(values.dtype)


def test_engine_boolean_metadata_restores_bool_from_float_h5_donor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor = _with_columns(
        _donor_frame(),
        "person",
        {"has_esi": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]},
    )
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    _MeanQRF.calls = []

    result = transfer_acs_inputs(
        _recipient_frame(),
        donor,
        target_families={"person": {"health": ("has_esi",)}},
        seed=3,
        n_estimators=1,
    )

    values = result.frame.person["has_esi"]
    assert pd.api.types.is_bool_dtype(values.dtype)
    assert set(values) <= {False, True}


def test_auto_channel_excludes_artificial_asec_zeros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ACS_DONOR_CHANNEL_AUTO == "auto"
    base = _donor_frame()
    asec = _channel_copy(
        base,
        channel="asec",
        offset=0,
        target_value=0.0,
    )
    puf = _channel_copy(
        base,
        channel="puf_tax_detail",
        offset=10_000,
        target_value=100.0,
    )
    donor = asec.concat(puf)
    recipient = _drop_column(
        _drop_column(_recipient_frame(), "person", "employment_income_before_lsr"),
        "person",
        "self_employment_income_before_lsr",
    )
    families = {"person": {"tax_detail": ("qualified_dividend_income",)}}
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)

    _MeanQRF.calls = []
    auto = transfer_acs_inputs(
        recipient,
        donor,
        target_families=families,
        seed=5,
        n_estimators=1,
    )

    assert auto.frame.table("person")["qualified_dividend_income"].eq(100.0).all()
    assert auto.resolved_donor_channel == "puf_tax_detail"
    assert auto.imputed_inputs[0].donor_channel == "puf_tax_detail"
    assert auto.imputed_inputs[0].patterns[0].donor_rows == len(base.person)
    assert all(
        call["targets"]["qualified_dividend_income"].eq(100.0).all()
        for call in _MeanQRF.calls
    )

    _MeanQRF.calls = []
    whole = transfer_acs_inputs(
        recipient,
        donor,
        target_families=families,
        donor_channel=None,
        seed=5,
        n_estimators=1,
    )
    assert whole.frame.table("person")["qualified_dividend_income"].eq(50.0).all()
    assert whole.resolved_donor_channel is None
    assert whole.imputed_inputs[0].donor_channel is None
    assert whole.imputed_inputs[0].patterns[0].donor_rows == 2 * len(base.person)


def test_pattern_fits_use_observed_native_and_complete_donor_analogs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor, recipient = _conditioning_frames()
    native_columns = {
        column: recipient.person[column].copy()
        for column in (
            "employment_income_before_lsr",
            "self_employment_income_before_lsr",
            "acs_social_security_income",
            "acs_retirement_income",
            "acs_interest_dividend_rental_income",
        )
    }
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    _MeanQRF.calls = []

    result = transfer_acs_inputs(
        recipient,
        donor,
        target_families={
            "person": {"tax_detail": ("qualified_dividend_income",)},
        },
        seed=23,
        n_estimators=1,
    )

    provenance = result.imputed_inputs[0]
    assert set(ACS_OPTIONAL_PERSON_TRANSFER_PREDICTORS).issubset(provenance.predictors)
    assert len(provenance.patterns) > 1
    assert len({pattern.seed for pattern in provenance.patterns}) == len(
        provenance.patterns
    )
    assert all(pattern.weight_kind == "calibrated" for pattern in provenance.patterns)
    assert all(
        np.isfinite(call["features"].to_numpy(dtype=np.float64)).all()
        for call in _MeanQRF.calls
    )
    social_calls = [
        call
        for call in _MeanQRF.calls
        if "__acs_transfer_social_security_income" in call["predictors"]
    ]
    assert social_calls
    assert any(
        call["features"]["__acs_transfer_social_security_income"].eq(100.0).any()
        for call in social_calls
    )
    investment_calls = [
        call
        for call in _MeanQRF.calls
        if "__acs_transfer_interest_dividend_rental_income" in call["predictors"]
    ]
    assert investment_calls
    assert all(len(call["features"]) < len(donor.person) for call in investment_calls)
    for column, before in native_columns.items():
        pd.testing.assert_series_equal(result.frame.person[column], before)


def test_housing_rows_without_tenure_remain_unmodeled() -> None:
    recipient = _replace_column(
        _recipient_frame(),
        "household",
        "tenure_type",
        ["OWNED_WITH_MORTGAGE", "RENTED", np.nan],
    )

    result = transfer_acs_inputs(
        recipient,
        _donor_frame(),
        target_families={"person": {"housing": ("pre_subsidy_rent",)}},
        seed=8,
        n_estimators=2,
    )

    missing_tenure = result.frame.person["person_household_id"] == 1_300
    assert result.frame.person.loc[missing_tenure, "pre_subsidy_rent"].isna().all()
    provenance = result.imputed_inputs[0]
    assert provenance.unmodeled_recipient_rows == 3
    assert all(
        "__acs_transfer_is_household_head" in pattern.predictors
        and "__acs_transfer_tenure_code" in pattern.predictors
        for pattern in provenance.patterns
    )


def test_native_income_blanks_remain_absent_and_partition_transfer_patterns() -> None:
    donor = _donor_frame()
    recipient = _recipient_frame()
    employment_before = recipient.table("person")["employment_income_before_lsr"].copy()
    self_employment_before = recipient.table("person")[
        "self_employment_income_before_lsr"
    ].copy()

    result = transfer_acs_inputs(
        recipient,
        donor,
        target_families={
            "person": {"tax_detail": ("qualified_dividend_income",)},
            "tax_unit": {"itemization": ("first_home_mortgage_balance",)},
        },
        seed=33,
        n_estimators=3,
    )

    pd.testing.assert_series_equal(
        result.frame.table("person")["employment_income_before_lsr"],
        employment_before,
    )
    pd.testing.assert_series_equal(
        result.frame.table("person")["self_employment_income_before_lsr"],
        self_employment_before,
    )
    assert employment_before.isna().any()
    assert self_employment_before.isna().any()
    person_provenance = next(
        entry for entry in result.imputed_inputs if entry.entity == "person"
    )
    assert set(ACS_PERSON_TRANSFER_PREDICTORS).issubset(person_provenance.predictors)
    patterns = person_provenance.patterns
    assert sum(pattern.recipient_rows for pattern in patterns) == len(employment_before)
    assert any(
        "__acs_transfer_employment_income" in pattern.observed_optional_predictors
        for pattern in patterns
    )
    assert any(
        "__acs_transfer_employment_income" not in pattern.observed_optional_predictors
        for pattern in patterns
    )
    assert any(
        "__acs_transfer_self_employment_income" in pattern.observed_optional_predictors
        for pattern in patterns
    )
    assert any(
        "__acs_transfer_self_employment_income"
        not in pattern.observed_optional_predictors
        for pattern in patterns
    )
    group_provenance = next(
        entry for entry in result.imputed_inputs if entry.entity == "tax_unit"
    )
    assert set(ACS_GROUP_TRANSFER_PREDICTORS).issubset(group_provenance.predictors)


def test_explicit_family_is_seed_deterministic_and_preserves_recipient_index() -> None:
    donor = _replace_column(
        _donor_frame(),
        "person",
        "person_support_channel",
        ["puf_tax_detail"] * 8,
    )
    recipient = _recipient_frame()
    families = {
        "person": {
            "tax_detail": ("qualified_dividend_income",),
        }
    }

    first = transfer_acs_inputs(
        recipient,
        donor,
        target_families=families,
        donor_channel="puf_tax_detail",
        seed=91,
        n_estimators=5,
    )
    second = transfer_acs_inputs(
        recipient,
        donor,
        target_families=families,
        donor_channel="puf_tax_detail",
        seed=91,
        n_estimators=5,
    )

    assert first.frame.table("person").index.equals(recipient.table("person").index)
    pd.testing.assert_series_equal(
        first.frame.table("person")["qualified_dividend_income"],
        second.frame.table("person")["qualified_dividend_income"],
    )
    assert first.imputed_inputs == second.imputed_inputs
    assert first.imputed_inputs[0].donor_channel == "puf_tax_detail"


def test_formula_owned_target_is_refused_before_fit() -> None:
    with pytest.raises(ValueError, match="formula-owned.*interest_deduction"):
        transfer_acs_inputs(
            _recipient_frame(),
            _donor_frame(),
            target_families={
                "tax_unit": {"bad": ("interest_deduction",)},
            },
            n_estimators=2,
        )


def test_non_finite_donor_target_is_refused_without_zero_fill() -> None:
    donor = _replace_column(
        _donor_frame(),
        "person",
        "qualified_dividend_income",
        [900.0, 200.0, np.nan, 20.0, 3_000.0, 1_500.0, 8_000.0, 40.0],
    )

    with pytest.raises(
        ValueError,
        match="donor targets.*non-finite.*qualified_dividend_income",
    ):
        transfer_acs_inputs(
            _recipient_frame(),
            donor,
            target_families={
                "person": {"tax_detail": ("qualified_dividend_income",)},
            },
            n_estimators=2,
        )


def test_non_finite_recipient_predictor_is_refused_without_zero_fill() -> None:
    recipient = _replace_column(
        _recipient_frame(),
        "person",
        "age",
        [41.0, 40.0, np.inf, 72.0, 69.0, 15.0],
    )

    with pytest.raises(
        ValueError,
        match="recipient person predictors.*non-finite.*age",
    ):
        transfer_acs_inputs(
            recipient,
            _donor_frame(),
            target_families={
                "person": {"tax_detail": ("qualified_dividend_income",)},
            },
            n_estimators=2,
        )


def test_missing_donor_target_is_refused_without_zero_fill() -> None:
    donor = _drop_column(_donor_frame(), "person", "qualified_dividend_income")

    with pytest.raises(
        ValueError,
        match="donor entity 'person' is missing target.*qualified_dividend_income",
    ):
        transfer_acs_inputs(
            _recipient_frame(),
            donor,
            target_families={
                "person": {"tax_detail": ("qualified_dividend_income",)},
            },
            n_estimators=2,
        )


def test_missing_recipient_predictor_is_refused_without_zero_fill() -> None:
    recipient = _drop_column(_recipient_frame(), "person", "age")

    with pytest.raises(
        ValueError,
        match="recipient person predictors missing column.*age",
    ):
        transfer_acs_inputs(
            recipient,
            _donor_frame(),
            target_families={
                "person": {"tax_detail": ("qualified_dividend_income",)},
            },
            n_estimators=2,
        )


def test_missing_recipient_state_predictor_is_refused_without_national_fill() -> None:
    recipient = _drop_column(_recipient_frame(), "household", "state_fips")

    with pytest.raises(
        ValueError,
        match="recipient person predictors missing.*state_fips",
    ):
        transfer_acs_inputs(
            recipient,
            _donor_frame(),
            target_families={
                "person": {"tax_detail": ("qualified_dividend_income",)},
            },
            n_estimators=2,
        )


@pytest.mark.parametrize(
    "families",
    [
        {},
        {"person": {"already_native": ("taxable_interest_income",)}},
    ],
)
def test_no_missing_targets_returns_identical_recipient_object(
    families: dict[str, dict[str, tuple[str, ...]]],
) -> None:
    recipient = _recipient_frame()

    result = transfer_acs_inputs(
        recipient,
        _donor_frame(),
        target_families=families,
        n_estimators=2,
    )

    assert isinstance(result, AcsTransferResult)
    assert result.frame is recipient
    assert result.imputed_inputs == ()
    assert result.fit_records == ()
