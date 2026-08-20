from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.acs_transfer as acs_transfer_module
import microcosm.build.us_runtime.acs_transfer_bank as acs_transfer_bank_module
from microcosm.build.frame_checkpoint import write_frame_checkpoint
from microcosm.build.serialization_dtypes import CANONICAL_STRING_DTYPE
from microcosm.build.us_runtime.acs_transfer import (
    ACS_DEFERRED_GEOGRAPHY_INPUTS,
    ACS_DONOR_CHANNEL_AUTO,
    ACS_GROUP_TRANSFER_PREDICTORS,
    ACS_NATIVE_PERSON_INPUTS,
    ACS_OPTIONAL_PERSON_TRANSFER_PREDICTORS,
    ACS_PERSON_TRANSFER_PREDICTORS,
    AcsTransferResult,
    acs_adult_care_qualifying_rows,
    declared_acs_transfer_target_families,
    default_acs_transfer_target_families,
    transfer_acs_inputs,
)
from microcosm.build.us_runtime.acs_transfer_bank import (
    AcsTransferTargetBankStore,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.fit import Regime
from microcosm.frame import US_SCHEMA, EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSVariableMetadataIndex,
)

SCHEMA = EntitySchema(group_entities=("household", "tax_unit"))

_BANK_FAMILY = "fixture_banked_three_target"
_BANK_TARGETS = (
    "qualified_dividend_income",
    "pre_subsidy_rent",
    "takes_up_medicaid_if_eligible",
)


def _bank_identity(name: str = "current") -> dict[str, object]:
    return {
        "artifact_kind": "fixture_acs_transfer_target_bank_identity",
        "fixture_version": 1,
        "name": name,
    }


def _bank_store(
    root: Path,
    *,
    identity_name: str = "current",
) -> AcsTransferTargetBankStore:
    return AcsTransferTargetBankStore(
        root,
        identity=_bank_identity(identity_name),
    )


def _bank_paths(bank: AcsTransferTargetBankStore) -> tuple[Path, ...]:
    return tuple(
        bank.target_path(index, target) for index, target in enumerate(_BANK_TARGETS)
    )


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
            # Pinned to the canonical storage: bare lists infer the
            # environment-default string storage (pyarrow when installed),
            # which would force the no-op canonicalization rewrite these
            # tests assert never happens.
            "native_label": pd.array(
                ["a", "b", "c", "d", "e", "f"], dtype=CANONICAL_STRING_DTYPE
            ),
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
            "tenure_type": pd.array(
                ["OWNED_WITH_MORTGAGE", "RENTED", "RENTED"],
                dtype=CANONICAL_STRING_DTYPE,
            ),
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


def _with_metadata(frame: Frame, metadata: dict[str, object]) -> Frame:
    tables = {name: frame.table(name).copy() for name in frame.entities}
    tables.update({name: frame.link(name).copy() for name in frame.links})
    return Frame(
        tables,
        frame.schema,
        {name: frame.weights_for(name) for name in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=metadata,
    )


def _with_full_us_schema(frame: Frame) -> Frame:
    """Promote the compact transfer fixture to all PolicyEngine-US grains."""

    tables = {name: frame.table(name).copy() for name in frame.entities}
    person = tables["person"]
    household_ids = person["person_household_id"].to_numpy(copy=True)
    for entity in ("spm_unit", "family", "marital_unit"):
        person[f"person_{entity}_id"] = household_ids
        tables[entity] = pd.DataFrame(
            {f"{entity}_id": np.unique(household_ids)},
        )
    return Frame(
        tables,
        US_SCHEMA,
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
        assert all(dtype == np.dtype("float64") for dtype in frame.dtypes)
        assert np.isfinite(frame.to_numpy()).all()
        return pd.DataFrame(
            {
                target: np.full(len(frame), value, dtype=np.float64)
                for target, value in self.means.items()
            },
            index=frame.index,
        )


def _run_bank_fixture(
    target_bank: AcsTransferTargetBankStore | None = None,
) -> AcsTransferResult:
    return transfer_acs_inputs(
        _recipient_frame(),
        _donor_frame(),
        target_families={"person": {_BANK_FAMILY: _BANK_TARGETS}},
        seed=37,
        n_estimators=1,
        target_bank=target_bank,
    )


def _assert_transfer_results_exact(
    actual: AcsTransferResult,
    expected: AcsTransferResult,
) -> None:
    assert actual.frame.schema == expected.frame.schema
    assert actual.frame.entities == expected.frame.entities
    for entity in expected.frame.entities:
        pd.testing.assert_frame_equal(
            actual.frame.table(entity),
            expected.frame.table(entity),
            check_dtype=True,
            check_exact=True,
        )
    for entity in expected.frame.weighted_entities:
        actual_weights = actual.frame.weights_for(entity)
        expected_weights = expected.frame.weights_for(entity)
        assert actual_weights.kind is expected_weights.kind
        np.testing.assert_array_equal(actual_weights.values, expected_weights.values)
    pd.testing.assert_series_equal(
        actual.frame.strata,
        expected.frame.strata,
        check_dtype=True,
        check_exact=True,
    )
    assert actual.frame.mass_log == expected.frame.mass_log
    assert actual.frame.metadata == expected.frame.metadata
    assert actual.imputed_inputs == expected.imputed_inputs
    assert actual.fit_records == expected.fit_records
    assert actual.deferred_inputs == expected.deferred_inputs
    assert actual.resolved_donor_channel == expected.resolved_donor_channel


def _lock_bank_fixture_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")


def _assert_bank_receipt(
    bank: AcsTransferTargetBankStore,
    *,
    sources: tuple[str, ...],
    load_statuses: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    targets = bank.receipt()["targets"]
    assert tuple(targets[str(index)]["source"] for index in range(3)) == sources
    assert (
        tuple(targets[str(index)]["load_status"] for index in range(3)) == load_statuses
    )
    return targets


def test_default_transfer_preserves_native_fields_and_registers_added_inputs() -> None:
    donor = _donor_frame()
    recipient = _recipient_frame()
    native_person = recipient.table("person").copy(deep=True)
    native_tax_unit = recipient.table("tax_unit").copy(deep=True)

    result = transfer_acs_inputs(
        recipient,
        donor,
        target_families={
            "person": {
                "puf_tax_itemization": ("qualified_dividend_income",),
                "housing": ("pre_subsidy_rent",),
                "benefit_participation": ("takes_up_medicaid_if_eligible",),
            },
            "tax_unit": {
                "puf_tax_itemization": ("first_home_mortgage_balance",),
                "benefit_participation": ("takes_up_eitc",),
            },
        },
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


def test_default_families_exclude_runtime_owned_take_up_columns() -> None:
    families = default_acs_transfer_target_families(_donor_frame())

    assert "qualified_dividend_income" in families["person"]["puf_tax_itemization"]
    assert families["person"]["housing"] == ("pre_subsidy_rent",)
    assert ACS_NATIVE_PERSON_INPUTS.isdisjoint(
        families["person"]["puf_tax_itemization"]
    )
    assert "benefit_participation" not in families["person"]
    assert (
        "first_home_mortgage_balance" in (families["tax_unit"]["puf_tax_itemization"])
    )
    assert "benefit_participation" not in families["tax_unit"]


def test_explicit_declared_plan_preserves_deferred_geography(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = {
        "person": {
            "puf_tax_itemization": ("qualified_dividend_income",),
        }
    }
    monkeypatch.setattr(
        acs_transfer_module,
        "declared_acs_transfer_target_families",
        lambda: plan,
    )
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)

    result = transfer_acs_inputs(
        _recipient_frame(),
        _donor_frame(),
        target_families=plan,
        donor_channel=None,
        seed=3,
        n_estimators=1,
    )

    assert result.deferred_inputs == tuple(sorted(ACS_DEFERRED_GEOGRAPHY_INPUTS))


def test_declared_families_are_independent_of_release_coverage_surface() -> None:
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

    families = default_acs_transfer_target_families(donor)

    assert families["person"]["model_required_boolean"] == (
        "has_esi",
        "is_blind",
    )
    declared = {
        target
        for entity_families in families.values()
        for targets in entity_families.values()
        for target in targets
    }
    assert "county_fips" not in declared
    assert "employment_income_before_lsr" not in declared
    assert "age" not in declared
    production_declared = declared_acs_transfer_target_families()
    production_targets = {
        target
        for entity_families in production_declared.values()
        for targets in entity_families.values()
        for target in targets
    }
    assert production_targets.isdisjoint(
        {
            "immigration_status_str",
            "selected_marketplace_plan_benchmark_ratio",
            "ssn_card_type",
            "takes_up_aca_if_eligible",
            "takes_up_eitc",
            "takes_up_medicaid_if_eligible",
            "takes_up_snap_if_eligible",
            "takes_up_tanf_if_eligible",
            "weekly_hours_worked_before_lsr",
        }
    )
    assert "has_esi" in production_declared["person"]["model_required_boolean"]
    assert "receives_wic" in production_declared["person"]["model_required_boolean"]


def test_declared_plan_carries_the_23_stage_base_surface() -> None:
    """The M/N/O donor repairs must reach the ACS spine through the plan."""

    person = declared_acs_transfer_target_families()["person"]
    itemization = set(person["puf_tax_itemization"])
    # The 23-stage base observes partnership SE earnings (buildl's exclusion
    # is stale there) but still carries s_corp_income as an all-zero schema
    # column, so only the SE-earnings leg returns to the plan.
    assert "partnership_self_employment_net_earnings" in itemization
    assert "s_corp_income" not in itemization
    # The Schedule D capital-gain-distributions memo leg is DERIVED from its
    # transferred parents (route exclusivity at the packaged share), never
    # fitted: an independent fit would break the base's accounting identity.
    from microcosm.build.us_runtime.acs_transfer import (
        ACS_DERIVED_TRANSFER_INPUTS,
    )

    assert "capital_gain_details" not in person
    assert "schedule_d_capital_gain_distributions" not in itemization
    assert ACS_DERIVED_TRANSFER_INPUTS == ("schedule_d_capital_gain_distributions",)
    # The CDCC adult-care leg: qualifying-person flag + expense carrier.
    assert person["adult_care"] == (
        "is_incapable_of_self_care",
        "pre_subsidy_care_expenses",
    )
    # The 23-stage base zeroed the second-home mortgage legs; the plan must
    # not declare all-engine-default donor targets.
    tax_unit = set(
        declared_acs_transfer_target_families()["tax_unit"]["puf_tax_itemization"]
    )
    assert "first_home_mortgage_balance" in tax_unit
    assert tax_unit.isdisjoint(
        {
            "second_home_mortgage_balance",
            "second_home_mortgage_interest",
            "second_home_mortgage_origination_year",
        }
    )
    assert declared_acs_transfer_target_families()["spm_unit"][
        "model_required_boolean"
    ] == ("is_tanf_enrolled", "receives_snap")


def test_explicit_transfer_adds_requested_model_inputs(
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
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    _MeanQRF.calls = []

    result = transfer_acs_inputs(
        _recipient_frame(),
        donor,
        target_families={
            "person": {
                "model_required_boolean": ("has_esi", "is_blind"),
                "model_required_categorical": (
                    "ssn_card_type",
                    "immigration_status_str",
                ),
                "model_required_numeric": ("other_medical_expenses",),
            }
        },
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


def test_pattern_provenance_records_ordered_exact_donor_target_regimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = (
        "fixture_positive_only",
        "fixture_zero_inflated_positive",
        "fixture_three_sign",
    )
    donor = _with_columns(
        _donor_frame(),
        "person",
        {
            targets[0]: np.arange(1.0, 9.0),
            targets[1]: [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            targets[2]: [-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0],
        },
    )
    recipient = _recipient_frame()
    for column in (
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
    ):
        donor = _drop_column(donor, "person", column)
        recipient = _drop_column(recipient, "person", column)
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    _MeanQRF.calls = []

    result = transfer_acs_inputs(
        recipient,
        donor,
        target_families={"person": {"fixture_regimes": targets}},
        n_estimators=1,
    )

    expected = (
        (targets[0], Regime.POSITIVE_ONLY),
        (targets[1], Regime.ZERO_INFLATED_POSITIVE),
        (targets[2], Regime.THREE_SIGN),
    )
    assert len(result.imputed_inputs[0].patterns) == 1
    assert all(
        item.patterns[0].target_regimes == expected for item in result.imputed_inputs
    )


def test_target_bank_cold_output_matches_unbanked_monolith(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _lock_bank_fixture_threads(monkeypatch)
    monolithic = _run_bank_fixture()
    bank = _bank_store(tmp_path / "cold-bank")

    banked = _run_bank_fixture(bank)

    _assert_transfer_results_exact(banked, monolithic)
    ordinary_patterns = monolithic.imputed_inputs[0].patterns
    banked_patterns = banked.imputed_inputs[0].patterns
    assert ordinary_patterns
    assert tuple(pattern.target_regimes for pattern in banked_patterns) == tuple(
        pattern.target_regimes for pattern in ordinary_patterns
    )
    assert all(
        tuple(target for target, _regime in pattern.target_regimes) == _BANK_TARGETS
        for pattern in ordinary_patterns
    )
    _assert_bank_receipt(
        bank,
        sources=("rebuilt", "rebuilt", "rebuilt"),
        load_statuses=("missing", "missing", "missing"),
    )
    assert all(path.is_file() for path in _bank_paths(bank))


def test_target_bank_materializer_v1_artifacts_fail_closed_with_named_receipts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _lock_bank_fixture_threads(monkeypatch)
    bank_root = tmp_path / "materializer-v1-bank"

    with monkeypatch.context() as legacy:
        legacy.setattr(
            acs_transfer_bank_module,
            "ACS_TRANSFER_TARGET_BANK_MATERIALIZER_VERSION",
            1,
        )
        legacy_bank = _bank_store(bank_root)
        baseline = _run_bank_fixture(legacy_bank)
        legacy_paths = _bank_paths(legacy_bank)
        assert all(path.is_file() for path in legacy_paths)
        for path in legacy_paths:
            metadata, _raw_bits = acs_transfer_bank_module._read_checkpoint(path)
            assert metadata["materializer_version"] == 1

    assert acs_transfer_bank_module.ACS_TRANSFER_TARGET_BANK_MATERIALIZER_VERSION == 2
    current_bank = _bank_store(bank_root)
    rebuilt = _run_bank_fixture(current_bank)

    _assert_transfer_results_exact(rebuilt, baseline)
    targets = _assert_bank_receipt(
        current_bank,
        sources=("rebuilt",) * len(_BANK_TARGETS),
        load_statuses=("invalid_rebuild",) * len(_BANK_TARGETS),
    )
    for record in targets.values():
        invalid = record["invalid_checkpoint"]
        assert invalid["reason"] == "checkpoint_validation_failed"
        assert invalid["message"] == (
            "ACS transfer target checkpoint has an unsupported binding."
        )


def test_target_bank_fsyncs_file_then_parent_directory_after_each_rename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _lock_bank_fixture_threads(monkeypatch)
    events: list[tuple[str, str]] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracked_fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(("fsync", kind))
        real_fsync(descriptor)

    def tracked_replace(source: Path, destination: Path) -> None:
        events.append(("replace", Path(destination).name))
        real_replace(source, destination)

    monkeypatch.setattr(acs_transfer_bank_module.os, "fsync", tracked_fsync)
    monkeypatch.setattr(acs_transfer_bank_module.os, "replace", tracked_replace)
    bank = _bank_store(tmp_path / "durable-bank")

    _run_bank_fixture(bank)

    assert len(events) == 3 * len(_BANK_TARGETS)
    for index, target in enumerate(_BANK_TARGETS):
        assert events[index * 3 : (index + 1) * 3] == [
            ("fsync", "file"),
            ("replace", bank.target_path(index, target).name),
            ("fsync", "directory"),
        ]


def test_target_bank_resumes_joint_immigration_codec_as_one_model_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _lock_bank_fixture_threads(monkeypatch)
    donor = _with_columns(
        _donor_frame(),
        "person",
        {
            "ssn_card_type": ["CITIZEN"] * 6 + ["NONE", "NONE"],
            "immigration_status_str": ["CITIZEN"] * 6
            + ["UNDOCUMENTED", "UNDOCUMENTED"],
        },
    )
    families = {
        "person": {
            "source_operator_immigration": (
                "ssn_card_type",
                "immigration_status_str",
            )
        }
    }

    def run(
        target_bank: AcsTransferTargetBankStore | None = None,
    ) -> AcsTransferResult:
        return transfer_acs_inputs(
            _recipient_frame(),
            donor,
            target_families=families,
            seed=37,
            n_estimators=1,
            target_bank=target_bank,
        )

    monolithic = run()
    bank_root = tmp_path / "joint-bank"
    cold_bank = _bank_store(bank_root)
    cold = run(cold_bank)
    _assert_transfer_results_exact(cold, monolithic)

    warm_bank = _bank_store(bank_root)
    resumed = run(warm_bank)
    _assert_transfer_results_exact(resumed, monolithic)
    for result in (monolithic, cold, resumed):
        person = result.frame.table("person")
        assert all(
            person[target].dtype == CANONICAL_STRING_DTYPE
            for target in ("ssn_card_type", "immigration_status_str")
        )

    targets = warm_bank.receipt()["targets"]
    assert set(targets) == {"0"}
    assert targets["0"]["source"] == "checkpoint"
    assert targets["0"]["descriptor"]["model_target"] == (
        "__acs_transfer_immigration_status_pair"
    )
    assert targets["0"]["descriptor"]["exported_targets"] == [
        "ssn_card_type",
        "immigration_status_str",
    ]


def test_target_bank_interrupt_resumes_durable_prefix_byte_identically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _lock_bank_fixture_threads(monkeypatch)
    uninterrupted_bank = _bank_store(tmp_path / "uninterrupted-bank")
    uninterrupted = _run_bank_fixture(uninterrupted_bank)

    interrupted_bank = _bank_store(tmp_path / "interrupted-bank")
    durable_write = interrupted_bank.write_target

    def interrupt_after_first_durable_target(checkpoint) -> None:
        durable_write(checkpoint)
        if checkpoint.target_index == 0:
            raise RuntimeError("fixture interrupt after durable ACS target 0")

    monkeypatch.setattr(
        interrupted_bank,
        "write_target",
        interrupt_after_first_durable_target,
    )
    with pytest.raises(
        RuntimeError,
        match="fixture interrupt after durable ACS target 0",
    ):
        _run_bank_fixture(interrupted_bank)

    interrupted_paths = _bank_paths(interrupted_bank)
    assert interrupted_paths[0].is_file()
    assert not interrupted_paths[1].exists()
    assert not interrupted_paths[2].exists()

    resumed_bank = _bank_store(tmp_path / "interrupted-bank")
    resumed = _run_bank_fixture(resumed_bank)
    _assert_transfer_results_exact(resumed, uninterrupted)

    uninterrupted_path = tmp_path / "uninterrupted.frame.h5"
    resumed_path = tmp_path / "resumed.frame.h5"
    write_frame_checkpoint(uninterrupted_path, uninterrupted.frame)
    write_frame_checkpoint(resumed_path, resumed.frame)
    assert resumed_path.read_bytes() == uninterrupted_path.read_bytes()

    _assert_bank_receipt(
        resumed_bank,
        sources=("checkpoint", "rebuilt", "rebuilt"),
        load_statuses=("resumed", "missing", "missing"),
    )


def test_target_bank_identity_mismatch_rebuilds_only_stale_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _lock_bank_fixture_threads(monkeypatch)
    bank_root = tmp_path / "identity-bank"
    cold_bank = _bank_store(bank_root)
    baseline = _run_bank_fixture(cold_bank)

    stale_bank = _bank_store(
        tmp_path / "stale-identity-bank",
        identity_name="stale",
    )
    _run_bank_fixture(stale_bank)
    stale_path = stale_bank.target_path(1, _BANK_TARGETS[1])
    cold_bank.target_path(1, _BANK_TARGETS[1]).write_bytes(stale_path.read_bytes())

    warm_bank = _bank_store(bank_root)
    rebuilt = _run_bank_fixture(warm_bank)

    _assert_transfer_results_exact(rebuilt, baseline)
    targets = _assert_bank_receipt(
        warm_bank,
        sources=("checkpoint", "rebuilt", "checkpoint"),
        load_statuses=("resumed", "identity_mismatch", "resumed"),
    )
    assert targets["1"]["load_status"] == "identity_mismatch"
    assert targets["1"]["ignored_checkpoint"]["reason"] == "identity_mismatch"


def test_target_bank_torn_files_fail_closed_and_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _lock_bank_fixture_threads(monkeypatch)
    bank_root = tmp_path / "torn-bank"
    cold_bank = _bank_store(bank_root)
    baseline = _run_bank_fixture(cold_bank)

    truncated, missing_final, _intact = _bank_paths(cold_bank)
    truncated.write_bytes(b"truncated ACS target checkpoint")
    missing_final.unlink()
    orphan = missing_final.with_name(f".{missing_final.name}.fixture.tmp")
    orphan.write_bytes(b"orphan target temporary")

    warm_bank = _bank_store(bank_root)
    rebuilt = _run_bank_fixture(warm_bank)

    _assert_transfer_results_exact(rebuilt, baseline)
    targets = _assert_bank_receipt(
        warm_bank,
        sources=("rebuilt", "rebuilt", "checkpoint"),
        load_statuses=("invalid_rebuild", "invalid_rebuild", "resumed"),
    )
    assert targets["0"]["load_status"] == "invalid_rebuild"
    assert (
        targets["0"]["invalid_checkpoint"]["reason"] == "checkpoint_validation_failed"
    )
    assert targets["1"]["load_status"] == "invalid_rebuild"
    assert targets["1"]["invalid_checkpoint"]["reason"] == "incomplete_checkpoint"


def test_target_bank_mixed_hole_rebuilds_only_missing_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _lock_bank_fixture_threads(monkeypatch)
    bank_root = tmp_path / "mixed-bank"
    cold_bank = _bank_store(bank_root)
    baseline = _run_bank_fixture(cold_bank)
    _bank_paths(cold_bank)[1].unlink()

    warm_bank = _bank_store(bank_root)
    resumed = _run_bank_fixture(warm_bank)

    _assert_transfer_results_exact(resumed, baseline)
    _assert_bank_receipt(
        warm_bank,
        sources=("checkpoint", "rebuilt", "checkpoint"),
        load_statuses=("resumed", "missing", "resumed"),
    )


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


def test_engine_boolean_metadata_restores_primary_qrf_float_h5_donor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("tables")  # pandas HDF backend
    # The bool restore reads the installed engine source metadata; without the
    # us extra the adapter's documented dtype fallback applies instead.
    try:
        PolicyEngineUSVariableMetadataIndex()
    except ImportError:
        pytest.skip("requires the policyengine-us [us] extra")
    # Retiring primary-PUF artifacts physically stored QBI boolean-count
    # outputs as floats, and the supported legacy ACS builder can still load
    # them from HDF as its transfer donor. Pin that compatibility representation
    # rather than using an unrelated model-required boolean.
    source = tmp_path / "legacy-boolean-donor.h5"
    pd.DataFrame({"business_is_sstb": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]}).to_hdf(
        source, key="person", format="fixed"
    )
    round_tripped = pd.read_hdf(source, key="person")["business_is_sstb"]
    assert round_tripped.dtype == np.dtype("float64")

    donor = _with_columns(
        _donor_frame(),
        "person",
        {"business_is_sstb": round_tripped.to_numpy()},
    )
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    _MeanQRF.calls = []

    result = transfer_acs_inputs(
        _recipient_frame(),
        donor,
        target_families={"person": {"puf_tax_itemization": ("business_is_sstb",)}},
        seed=3,
        n_estimators=1,
    )

    values = result.frame.person["business_is_sstb"]
    assert pd.api.types.is_bool_dtype(values.dtype)
    assert set(values) <= {False, True}


@pytest.mark.parametrize(
    ("values", "offending_type"),
    [
        pytest.param(
            [True, 0.0] * 4,
            "builtins.float",
            id="sol-exact-bool-float",
        ),
        pytest.param(
            [1.0, 0.0] * 4,
            "builtins.float",
            id="uniform-object-float-zero-one",
        ),
        pytest.param([True, 0] * 4, "builtins.int", id="bool-int"),
        pytest.param([True, "false"] * 4, "builtins.str", id="bool-str"),
        pytest.param(
            [np.bool_(True), np.int64(0)] * 4,
            "numpy.int64",
            id="numpy-bool-int",
        ),
        pytest.param(
            [np.bool_(True), np.float64(0.0)] * 4,
            "numpy.float64",
            id="numpy-bool-float",
        ),
        pytest.param(
            [np.bool_(True), "false"] * 4,
            "builtins.str",
            id="numpy-bool-str",
        ),
    ],
)
def test_known_boolean_metadata_rejects_mixed_object_donor_values(
    monkeypatch: pytest.MonkeyPatch,
    values: list[object],
    offending_type: str,
) -> None:
    donor = _with_columns(
        _donor_frame(),
        "person",
        {"is_blind": np.asarray(values, dtype=object)},
    )
    real_metadata = acs_transfer_module._engine_variable_metadata

    def metadata(target: str):
        if target == "is_blind":
            return SimpleNamespace(dtype="bool", entity="person")
        return real_metadata(target)

    monkeypatch.setattr(acs_transfer_module, "_engine_variable_metadata", metadata)
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    _MeanQRF.calls = []

    with pytest.raises(TypeError) as exc_info:
        transfer_acs_inputs(
            _recipient_frame(),
            donor,
            target_families={"person": {"model_required_boolean": ("is_blind",)}},
            seed=3,
            n_estimators=1,
        )

    message = str(exc_info.value)
    assert "boolean target 'is_blind'" in message
    assert "offending value types" in message
    assert offending_type in message
    assert _MeanQRF.calls == []


def test_known_boolean_metadata_accepts_python_and_numpy_boolean_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        acs_transfer_module,
        "_engine_variable_metadata",
        lambda _target: SimpleNamespace(dtype="bool"),
    )
    series = pd.Series([True, np.bool_(False)], dtype=object)

    encoding = acs_transfer_module._target_encoding(series, target="is_blind")

    assert encoding.kind == "boolean"
    np.testing.assert_array_equal(encoding.model_values, np.asarray([1.0, 0.0]))


def test_semantic_boolean_predictor_encoding_preserves_missingness() -> None:
    predictors = pd.DataFrame(
        {
            "is_female": pd.Series(
                [True, None, np.bool_(False)],
                dtype=object,
            )
        }
    )

    acs_transfer_module._validate_optional_numeric_frame(
        predictors,
        context="fixture predictors",
    )
    encoded = acs_transfer_module._encoded_predictor_frame(
        predictors,
        predictors=("is_female",),
    )
    complete = acs_transfer_module._complete_predictor_mask(
        predictors,
        predictors=("is_female",),
    )

    assert encoded["is_female"].dtype == np.dtype("float64")
    np.testing.assert_allclose(
        encoded["is_female"].to_numpy(),
        np.asarray([1.0, np.nan, 0.0]),
        equal_nan=True,
    )
    np.testing.assert_array_equal(complete, np.asarray([True, False, True]))
    acs_transfer_module._validate_required_numeric_frame(
        predictors,
        context="fixture required predictors",
    )


def test_predictor_validation_rejects_infinity() -> None:
    predictors = pd.DataFrame({"age": [40.0, np.inf]})

    with pytest.raises(ValueError, match="infinite.*age"):
        acs_transfer_module._validate_required_numeric_frame(
            predictors,
            context="fixture required predictors",
        )


def test_required_semantic_boolean_nulls_are_complete_case_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor = _replace_column(
        _donor_frame(),
        "person",
        "is_female",
        pd.Series(
            [None, True, True, False, False, True, True, False],
            index=_donor_frame().person.index,
            dtype=object,
        ),
    )
    recipient = _replace_column(
        _recipient_frame(),
        "person",
        "is_female",
        pd.Series(
            [None, True, True, False, True, False],
            index=_recipient_frame().person.index,
            dtype=object,
        ),
    )
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    _MeanQRF.calls = []

    result = transfer_acs_inputs(
        recipient,
        donor,
        target_families={
            "person": {"tax_detail": ("qualified_dividend_income",)},
        },
        n_estimators=1,
    )

    assert result.frame.person["qualified_dividend_income"].isna().tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    provenance = result.imputed_inputs[0]
    assert provenance.imputed_recipient_rows == 5
    assert provenance.unmodeled_recipient_rows == 1
    assert _MeanQRF.calls
    assert all(
        np.isfinite(call["features"].to_numpy()).all() for call in _MeanQRF.calls
    )


def test_group_required_predictors_preserve_missing_state_for_masking() -> None:
    recipient = _replace_column(
        _recipient_frame(),
        "household",
        "state_fips",
        [6.0, np.nan, 36.0],
    )

    surface = acs_transfer_module._transfer_feature_surface(
        recipient,
        recipient,
        entity="tax_unit",
        targets=("first_home_mortgage_balance",),
    )
    complete = acs_transfer_module._complete_predictor_mask(
        surface.recipient,
        predictors=surface.required,
    )

    np.testing.assert_allclose(
        surface.recipient["__acs_transfer_state_fips"].to_numpy(),
        np.asarray([6.0, np.nan, 36.0]),
        equal_nan=True,
    )
    np.testing.assert_array_equal(complete, np.asarray([True, False, True]))


@pytest.mark.parametrize(
    "values",
    [
        [True, 0.0],
        [np.bool_(False), 1],
        [True, "False"],
    ],
)
def test_mixed_object_predictor_is_not_a_semantic_boolean(
    values: list[object],
) -> None:
    predictors = pd.DataFrame(
        {"is_female": pd.Series(values, dtype=object)},
    )

    with pytest.raises(TypeError, match="numeric/boolean.*is_female"):
        acs_transfer_module._validate_optional_numeric_frame(
            predictors,
            context="fixture predictors",
        )


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


def test_auto_role_selects_puf_clones_from_assembled_two_spine_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ACS transfer routes by clone role, never assembled source channel."""

    base = _with_full_us_schema(_donor_frame())
    assembled = assemble_spines(
        {"asec": base, "acs": base},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    donor = clone_us_frame_for_puf_support(assembled)
    source_channels = donor.table("person")["person_support_channel"]
    assert set(source_channels) == {"acs", "asec"}
    assert "puf_tax_detail" not in set(source_channels)

    compact_recipient = _drop_column(
        _drop_column(_recipient_frame(), "person", "employment_income_before_lsr"),
        "person",
        "self_employment_income_before_lsr",
    )
    recipient = _with_full_us_schema(compact_recipient)
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    _MeanQRF.calls = []

    result = transfer_acs_inputs(
        recipient,
        donor,
        target_families={
            "person": {"tax_detail": ("qualified_dividend_income",)},
        },
        seed=5,
        n_estimators=1,
    )

    assert result.resolved_donor_channel == "puf_tax_detail"
    assert result.imputed_inputs[0].donor_channel == "puf_tax_detail"
    assert result.imputed_inputs[0].patterns[0].donor_rows == assembled.n("person")
    assert donor.n("person") == 2 * assembled.n("person")


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


def test_nullable_recipient_target_fills_only_nulls_and_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipient = _with_columns(
        _recipient_frame(),
        "person",
        {
            "qualified_dividend_income": pd.array(
                [111.0, pd.NA, 333.0, pd.NA, 555.0, 666.0],
                dtype="Float64",
            ),
            "takes_up_medicaid_if_eligible": pd.array(
                [True, pd.NA, False, pd.NA, True, False],
                dtype="boolean",
            ),
        },
    )
    recipient = _with_metadata(
        recipient,
        {
            "assembly_receipt": {
                "channels": ("asec", "acs"),
                "id_bound": 10_000,
            }
        },
    )
    before = {
        target: recipient.person[target].copy()
        for target in (
            "qualified_dividend_income",
            "takes_up_medicaid_if_eligible",
        )
    }
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    _MeanQRF.calls = []

    result = transfer_acs_inputs(
        recipient,
        _donor_frame(),
        target_families={
            "person": {
                "tax_detail": (
                    "qualified_dividend_income",
                    "takes_up_medicaid_if_eligible",
                ),
            },
        },
        n_estimators=1,
    )

    for target, target_before in before.items():
        after = result.frame.person[target]
        observed = target_before.notna()
        pd.testing.assert_series_equal(
            after.loc[observed],
            target_before.loc[observed],
        )
        assert after.loc[~observed].notna().all()
        assert recipient.person[target].isna().sum() == 2
    assert result.frame.metadata == recipient.metadata
    provenance = {item.column: item for item in result.imputed_inputs}
    assert set(provenance) == set(before)
    assert all(item.imputed_recipient_rows == 2 for item in provenance.values())
    assert all(item.unmodeled_recipient_rows == 0 for item in provenance.values())
    assert all(
        sum(pattern.recipient_rows for pattern in item.patterns) == 2
        for item in provenance.values()
    )


def test_donor_family_fit_uses_rows_complete_for_every_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor = _with_columns(
        _donor_frame(),
        "person",
        {
            "nullable_target_a": [
                np.nan,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ],
            "nullable_target_b": [
                10.0,
                np.nan,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
            ],
        },
    )
    recipient = _recipient_frame()
    for column in (
        "employment_income_before_lsr",
        "self_employment_income_before_lsr",
    ):
        donor = _drop_column(donor, "person", column)
        recipient = _drop_column(recipient, "person", column)
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    _MeanQRF.calls = []

    result = transfer_acs_inputs(
        recipient,
        donor,
        target_families={
            "person": {
                "nullable_pair": ("nullable_target_a", "nullable_target_b"),
            },
        },
        n_estimators=1,
    )

    assert len(_MeanQRF.calls) == 1
    fitted_targets = _MeanQRF.calls[0]["targets"]
    assert len(fitted_targets) == 6
    assert np.isfinite(fitted_targets.to_numpy(dtype=np.float64)).all()
    assert {item.patterns[0].donor_rows for item in result.imputed_inputs} == {6}


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


def test_weeks_worked_formula_owned_target_reproduces_pool_run_3_failure() -> None:
    try:
        PolicyEngineUSVariableMetadataIndex()
    except ImportError:
        pytest.skip("requires the policyengine-us [us] extra")

    with pytest.raises(
        ValueError,
        match=(
            r"ACS transfer targets must be PolicyEngine input leaves, not "
            r"formula-owned outputs: \['weeks_worked'\]\."
        ),
    ):
        transfer_acs_inputs(
            _recipient_frame(),
            _donor_frame(),
            target_families={
                "person": {"source_operator_hours_worked": ("weeks_worked",)},
            },
            n_estimators=2,
        )


def test_strict_leaf_audit_reports_missing_us_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from microcosm.frame.adapters import policyengine_us as adapter_module

    class _MissingMetadataIndex:
        def __init__(self) -> None:
            raise ImportError("policyengine-us is absent")

    monkeypatch.setattr(
        adapter_module,
        "PolicyEngineUSVariableMetadataIndex",
        _MissingMetadataIndex,
    )

    with pytest.raises(
        RuntimeError,
        match="Strict ACS transfer leaf classification requires policyengine-us",
    ):
        acs_transfer_module.assert_acs_transfer_targets_are_input_leaves(
            {"employment_income"},
            require_known=True,
        )


def test_all_missing_donor_target_is_refused_without_zero_fill() -> None:
    donor = _replace_column(
        _donor_frame(),
        "person",
        "qualified_dividend_income",
        [np.nan] * 8,
    )

    with pytest.raises(
        ValueError,
        match="no donor rows complete for every target.*qualified_dividend_income",
    ):
        transfer_acs_inputs(
            _recipient_frame(),
            donor,
            target_families={
                "person": {"tax_detail": ("qualified_dividend_income",)},
            },
            n_estimators=2,
        )


def test_schedule_d_post_transfer_fills_only_newly_imputed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor = _with_columns(
        _donor_frame(),
        "person",
        {
            "long_term_capital_gains_before_response": [
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
                700.0,
                800.0,
            ],
            "non_sch_d_capital_gains": [0.0] * 8,
        },
    )
    recipient = _with_columns(
        _recipient_frame(),
        "person",
        {
            "long_term_capital_gains_before_response": [
                1_000.0,
                np.nan,
                2_000.0,
                np.nan,
                3_000.0,
                np.nan,
            ],
            "non_sch_d_capital_gains": [
                0.0,
                np.nan,
                500.0,
                np.nan,
                0.0,
                np.nan,
            ],
            "schedule_d_capital_gain_distributions": [
                777.0,
                np.nan,
                222.0,
                888.0,
                333.0,
                np.nan,
            ],
        },
    )
    cgd_before = recipient.person["schedule_d_capital_gain_distributions"].copy()
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    _MeanQRF.calls = []

    result = transfer_acs_inputs(
        recipient,
        donor,
        target_families={
            "person": {
                "capital_gain_details": (
                    "long_term_capital_gains_before_response",
                    "non_sch_d_capital_gains",
                ),
            },
        },
        n_estimators=1,
    )

    cgd = result.frame.person["schedule_d_capital_gain_distributions"]
    measured = cgd_before.notna()
    pd.testing.assert_series_equal(cgd.loc[measured], cgd_before.loc[measured])
    assert cgd.iloc[[1, 5]].gt(0.0).all()
    derived = next(
        item
        for item in result.imputed_inputs
        if item.column == "schedule_d_capital_gain_distributions"
    )
    assert derived.imputed_recipient_rows == 2

    targets = (
        "long_term_capital_gains_before_response",
        "non_sch_d_capital_gains",
    )
    bound_contract = acs_transfer_module.acs_transfer_execution_contract_identity(
        targets=targets,
        derive_schedule_d=False,
    )
    suppressed = transfer_acs_inputs(
        recipient,
        donor,
        target_families={"person": {"capital_gain_details": targets}},
        n_estimators=1,
        derive_schedule_d=False,
        execution_contract=bound_contract,
    )
    pd.testing.assert_series_equal(
        suppressed.frame.person["schedule_d_capital_gain_distributions"],
        cgd_before,
    )
    assert all(
        item.column != "schedule_d_capital_gain_distributions"
        for item in suppressed.imputed_inputs
    )


def test_adult_care_reconciliation_changes_only_imputed_expenses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    donor = _with_columns(
        _donor_frame(),
        "person",
        {
            "is_incapable_of_self_care": [True] * 8,
            "pre_subsidy_care_expenses": [
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
                700.0,
                800.0,
            ],
        },
    )
    recipient = _with_columns(
        _recipient_frame(),
        "person",
        {
            "is_incapable_of_self_care": [True] * 6,
            "pre_subsidy_care_expenses": [
                900.0,
                np.nan,
                0.0,
                np.nan,
                0.0,
                np.nan,
            ],
            "tax_unit_role_input": ["DEPENDENT"] * 6,
        },
    )
    before = recipient.person["pre_subsidy_care_expenses"].copy()
    monkeypatch.setattr(acs_transfer_module, "QRF", _MeanQRF)
    _MeanQRF.calls = []

    result = transfer_acs_inputs(
        recipient,
        donor,
        target_families={
            "person": {
                "adult_care": (
                    "is_incapable_of_self_care",
                    "pre_subsidy_care_expenses",
                ),
            },
        },
        n_estimators=1,
    )

    expenses = result.frame.person["pre_subsidy_care_expenses"]
    measured = before.notna()
    pd.testing.assert_series_equal(expenses.loc[measured], before.loc[measured])
    assert expenses.iloc[1] == 0.0
    assert expenses.iloc[3] > 0.0
    assert expenses.iloc[5] == 0.0
    provenance = next(
        item
        for item in result.imputed_inputs
        if item.column == "pre_subsidy_care_expenses"
    )
    assert provenance.imputed_recipient_rows == 3
    assert provenance.reconciliation == {
        "cleared_ineligible_carriers": 0,
        "cleared_multi_carrier_rows": 2,
        "remaining_carriers": 2,
    }


def test_non_finite_recipient_predictor_is_refused_without_zero_fill() -> None:
    recipient = _replace_column(
        _recipient_frame(),
        "person",
        "age",
        [41.0, 40.0, np.inf, 72.0, 69.0, 15.0],
    )

    with pytest.raises(
        ValueError,
        match="recipient person predictors.*infinite.*age",
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


@pytest.mark.parametrize(
    "families",
    [
        {},
        {"person": {"already_native": ("taxable_interest_income",)}},
    ],
)
def test_no_op_transfer_canonicalizes_object_strings(
    families: dict[str, dict[str, tuple[str, ...]]],
) -> None:
    recipient = _recipient_frame()
    recipient.table("person")["native_label"] = recipient.table("person")[
        "native_label"
    ].astype(object)

    result = transfer_acs_inputs(
        recipient,
        _donor_frame(),
        target_families=families,
        n_estimators=2,
    )

    assert result.frame is not recipient
    assert recipient.table("person")["native_label"].dtype == np.dtype(object)
    assert result.frame.table("person")["native_label"].dtype == CANONICAL_STRING_DTYPE


def test_schedule_d_cgd_derivation_enforces_route_exclusivity() -> None:
    from microcosm.build.us_runtime.acs_transfer import (
        derive_acs_schedule_d_capital_gain_distributions,
    )
    from microcosm.build.us_runtime.capital_gain_distributions import (
        load_capital_gain_distribution_shares,
    )

    share = load_capital_gain_distribution_shares()
    ratio = float(share.schedule_d_cgd_share_of_lt_net_gains)
    person = pd.DataFrame(
        {
            # eligible / other-route blocks / no gains / negative gains
            "long_term_capital_gains_before_response": [10_000.0, 8_000.0, 0.0, -5.0],
            "non_sch_d_capital_gains": [0.0, 1_200.0, 0.0, 0.0],
        }
    )

    values, provenance = derive_acs_schedule_d_capital_gain_distributions(person)

    assert values[0] == pytest.approx(ratio * 10_000.0)
    assert values[1] == 0.0  # mutually exclusive route present
    assert values[2] == 0.0
    assert values[3] == 0.0
    assert provenance["eligible_rows"] == 1
    # The identities an independent fit cannot guarantee:
    both_routes = (values > 0) & (person["non_sch_d_capital_gains"].to_numpy() > 0)
    assert not both_routes.any()
    assert (
        values
        <= ratio
        * person["long_term_capital_gains_before_response"].clip(lower=0.0).to_numpy()
        + 1e-9
    ).all()


def test_schedule_d_cgd_derivation_rejects_incomplete_parents() -> None:
    from microcosm.build.us_runtime.acs_transfer import (
        derive_acs_schedule_d_capital_gain_distributions,
    )

    person = pd.DataFrame(
        {
            "long_term_capital_gains_before_response": [1.0, np.nan],
            "non_sch_d_capital_gains": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="complete transferred parents"):
        derive_acs_schedule_d_capital_gain_distributions(person)


def test_adult_care_reconciliation_enforces_statute_structure() -> None:
    from microcosm.build.us_runtime.acs_transfer import reconcile_acs_adult_care

    person = pd.DataFrame(
        {
            "is_incapable_of_self_care": [
                True,  # dependent carrier, kept
                True,  # second carrier in unit 1 (smaller), cleared
                False,  # unflagged expense, cleared
                True,  # unmarried head, not qualifying, cleared
                True,  # spouse in married unit, kept
            ],
            "pre_subsidy_care_expenses": [900.0, 400.0, 250.0, 800.0, 600.0],
            "tax_unit_role_input": [
                "DEPENDENT",
                "DEPENDENT",
                "DEPENDENT",
                "HEAD",
                "SPOUSE",
            ],
            "person_tax_unit_id": [1, 1, 1, 2, 3],
        }
    )
    # Unit 3 married via its spouse row; unit 2 has no spouse row.
    person.loc[len(person)] = [False, 0.0, "HEAD", 3]

    expenses, counts = reconcile_acs_adult_care(person)

    assert expenses.tolist() == [900.0, 0.0, 0.0, 0.0, 600.0, 0.0]
    assert counts == {
        "cleared_ineligible_carriers": 2,
        "cleared_multi_carrier_rows": 1,
        "remaining_carriers": 2,
    }


def test_adult_care_qualifying_rows_enforces_role_and_marriage_rules() -> None:
    person = pd.DataFrame(
        {
            "is_incapable_of_self_care": [
                True,
                False,
                True,
                True,
                True,
                True,
            ],
            "tax_unit_role_input": [
                "DEPENDENT",
                "DEPENDENT",
                "HEAD",
                "SPOUSE",
                "HEAD",
                "OTHER",
            ],
            "person_tax_unit_id": [1, 1, 2, 2, 3, 3],
        }
    )

    qualifying = acs_adult_care_qualifying_rows(person)

    assert qualifying.tolist() == [True, False, True, True, False, False]


@pytest.mark.parametrize(
    "missing_column",
    [
        "is_incapable_of_self_care",
        "tax_unit_role_input",
        "person_tax_unit_id",
    ],
)
def test_adult_care_qualifying_rows_fails_closed_on_missing_structure(
    missing_column: str,
) -> None:
    person = pd.DataFrame(
        {
            "is_incapable_of_self_care": [True],
            "tax_unit_role_input": ["DEPENDENT"],
            "person_tax_unit_id": [1],
        }
    ).drop(columns=missing_column)

    with pytest.raises(ValueError, match=missing_column):
        acs_adult_care_qualifying_rows(person)
