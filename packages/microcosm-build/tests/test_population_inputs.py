"""Closed, fail-closed population-input and scheme-mapping contracts."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build import (
    PopulationInputContract,
    PopulationInputNotReadyError,
    PopulationInputProfile,
    SchemePopulationMapping,
    validate_population_input_frame,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "country": "xx",
        "profile_id": "benefit_population_inputs",
        "activation": "explicit_only",
        "description": "Synthetic value-free population input contract.",
        "inputs": [
            {
                "input_id": "regular_payment_recipient",
                "column": "regular_payment_recipient_indicator",
                "entity": "person",
                "dtype": "bool",
                "nullable": False,
                "semantic_kind": "receipt",
                "data_kind": "latent",
                "owner": "Microcosm",
                "consumer": "PolicyEngine",
                "mechanics_owner": "PolicyEngine",
                "axiom_role": "none",
                "description": "Microcosm population status supplied to an adapter.",
            }
        ],
        "mappings": [
            {
                "mapping_id": "regular_payment_scheme_population",
                "target_reference": "publisher_regular_payment_recipients",
                "input_id": "regular_payment_recipient",
                "chronicle_source_record_id": (
                    "publisher.regular_payment.month2025_01.all.beneficiaries"
                ),
                "chronicle_entity": "person",
                "chronicle_entity_role": "regular_payment_beneficiary",
                "chronicle_geography_level": "statistical_scope",
                "chronicle_geography_id": "XX-SCHEME",
                "chronicle_geography_vintage": "SCHEME_ADMIN_SCOPE_2025",
                "chronicle_period_type": "month",
                "chronicle_period": "2025-01",
                "microcosm_entity": "person",
                "microcosm_geography_level": "statistical_scope",
                "microcosm_geography_id": "XX-SCHEME",
                "microcosm_geography_vintage": "SCHEME_ADMIN_SCOPE_2025",
                "microcosm_period_type": "month",
                "microcosm_period": "2025-01",
                "input_readiness": "ready",
                "mapping_readiness": "ready",
                "period_readiness": "ready",
                "notes": "Exact publisher scheme population and snapshot.",
            }
        ],
    }


def _profile(payload: dict[str, object] | None = None) -> PopulationInputProfile:
    return PopulationInputProfile.from_mapping(payload or _payload(), country="xx")


def _frame(values=(True, False, True), ids=(11, 12, 13)) -> Frame:
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": [1, 1, 2],
            "regular_payment_recipient_indicator": values,
        }
    )
    household = pd.DataFrame({"household_id": [1, 2]})
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.array([2.0, 1.0]), WeightKind.DESIGN)},
    )


def test_profile_parses_typed_microcosm_to_policyengine_boundary():
    profile = _profile()

    assert isinstance(profile.inputs[0], PopulationInputContract)
    assert isinstance(profile.mappings[0], SchemePopulationMapping)
    assert profile.inputs[0].owner == "Microcosm"
    assert profile.inputs[0].consumer == "PolicyEngine"
    assert profile.inputs[0].mechanics_owner == "PolicyEngine"
    assert profile.inputs[0].axiom_role == "none"
    assert profile.mappings[0].chronicle_entity_role == ("regular_payment_beneficiary")
    assert profile.mappings[0].blockers == ()


@pytest.mark.parametrize(
    ("path", "key", "value"),
    [
        ((), "automatic_activation", True),
        (("inputs", 0), "formula", "eligible and random_draw"),
        (("inputs", 0), "axiom_concept", "takes_up_if_eligible"),
        (("mappings", 0), "geography_crosswalk", "statistical_scope_to_nuts1"),
    ],
)
def test_every_schema_level_rejects_unknown_bypass_fields(path, key, value):
    payload = copy.deepcopy(_payload())
    location = payload
    for part in path:
        location = location[part]
    location[key] = value

    with pytest.raises(ValueError, match="unknown"):
        PopulationInputProfile.from_mapping(payload, country="xx")


@pytest.mark.parametrize(
    ("path", "key"),
    [
        ((), "activation"),
        (("inputs", 0), "owner"),
        (("inputs", 0), "semantic_kind"),
        (("mappings", 0), "chronicle_source_record_id"),
        (("mappings", 0), "chronicle_entity_role"),
        (("mappings", 0), "mapping_readiness"),
        (("mappings", 0), "period_readiness"),
    ],
)
def test_required_contract_fields_cannot_be_omitted(path, key):
    payload = copy.deepcopy(_payload())
    location = payload
    for part in path:
        location = location[part]
    del location[key]

    with pytest.raises(ValueError, match="missing"):
        PopulationInputProfile.from_mapping(payload, country="xx")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dtype", "int64", "dtype"),
        ("nullable", True, "nullable"),
        ("semantic_kind", "take_up", "semantic_kind"),
        ("data_kind", "modeled_behavior", "data_kind"),
        ("owner", "PolicyEngine", "owner"),
        ("consumer", "Microcosm", "consumer"),
        ("mechanics_owner", "Axiom", "mechanics_owner"),
        ("axiom_role", "behavior_input", "axiom_role"),
    ],
)
def test_input_contract_refuses_wrong_types_or_ownership(field, value, message):
    payload = copy.deepcopy(_payload())
    payload["inputs"][0][field] = value

    with pytest.raises(ValueError, match=message):
        PopulationInputProfile.from_mapping(payload, country="xx")


@pytest.mark.parametrize(
    "column",
    [
        "takes_up_grant_if_eligible",
        "grant_take_up",
        "grant_takeup_propensity",
        "labor_supply_elasticity",
    ],
)
def test_behavioral_or_eligibility_variable_synthesis_is_refused(column):
    payload = copy.deepcopy(_payload())
    payload["inputs"][0]["column"] = column

    with pytest.raises(ValueError, match="looks behavioral"):
        PopulationInputProfile.from_mapping(payload, country="xx")


def test_statistical_scope_cannot_masquerade_as_nuts_geography():
    payload = copy.deepcopy(_payload())
    mapping = payload["mappings"][0]
    mapping["microcosm_geography_level"] = "nuts1"
    mapping["microcosm_geography_id"] = "XX1"
    mapping["microcosm_geography_vintage"] = "NUTS_2024"
    mapping["mapping_readiness"] = "required_missing"

    with pytest.raises(ValueError, match="statistical_scope"):
        PopulationInputProfile.from_mapping(payload, country="xx")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("microcosm_entity", "household", "entity/geography identity"),
        ("microcosm_geography_id", "XX-RESIDENTS", "entity/geography identity"),
        ("microcosm_period", "2025-02", "exact period identity"),
    ],
)
def test_ready_mapping_requires_exact_entity_geography_and_period(
    field, value, message
):
    payload = copy.deepcopy(_payload())
    payload["mappings"][0][field] = value

    with pytest.raises(ValueError, match=message):
        PopulationInputProfile.from_mapping(payload, country="xx")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_readiness", "pending"),
        ("mapping_readiness", "inferred"),
        ("period_readiness", "projected"),
    ],
)
def test_unknown_readiness_values_are_refused(field, value):
    payload = copy.deepcopy(_payload())
    payload["mappings"][0][field] = value

    with pytest.raises(ValueError, match=field):
        PopulationInputProfile.from_mapping(payload, country="xx")


def test_profile_rejects_orphan_inputs_unknown_links_and_duplicate_columns():
    orphaned = copy.deepcopy(_payload())
    extra_input = copy.deepcopy(orphaned["inputs"][0])
    extra_input["input_id"] = "orphaned_application"
    extra_input["column"] = "orphaned_application_indicator"
    extra_input["semantic_kind"] = "application"
    orphaned["inputs"].append(extra_input)
    with pytest.raises(ValueError, match="no scheme-population mapping"):
        PopulationInputProfile.from_mapping(orphaned, country="xx")

    unknown = copy.deepcopy(_payload())
    unknown["mappings"][0]["input_id"] = "omitted_input"
    with pytest.raises(ValueError, match="unknown input"):
        PopulationInputProfile.from_mapping(unknown, country="xx")

    duplicate = copy.deepcopy(_payload())
    second = copy.deepcopy(duplicate["inputs"][0])
    second["input_id"] = "same_column_choice"
    second["semantic_kind"] = "choice"
    duplicate["inputs"].append(second)
    duplicate_mapping = copy.deepcopy(duplicate["mappings"][0])
    duplicate_mapping["mapping_id"] = "same_column_choice_mapping"
    duplicate_mapping["input_id"] = "same_column_choice"
    duplicate["mappings"].append(duplicate_mapping)
    with pytest.raises(ValueError, match="duplicate entity/column"):
        PopulationInputProfile.from_mapping(duplicate, country="xx")


def test_nonready_contract_fails_before_touching_a_frame():
    payload = copy.deepcopy(_payload())
    payload["mappings"][0]["input_readiness"] = "required_missing"
    profile = _profile(payload)

    class FrameAccessTrap:
        def table(self, entity):  # pragma: no cover - must never run
            raise AssertionError(f"Frame was accessed for {entity}")

    with pytest.raises(PopulationInputNotReadyError, match="input_readiness"):
        validate_population_input_frame(
            FrameAccessTrap(),
            profile,
            mapping_id="regular_payment_scheme_population",
        )


def test_ready_boolean_column_emits_deterministic_row_value_identity_receipt():
    profile = _profile()
    receipt = validate_population_input_frame(
        _frame(),
        profile,
        mapping_id="regular_payment_scheme_population",
    )
    repeated = validate_population_input_frame(
        _frame(),
        profile,
        mapping_id="regular_payment_scheme_population",
    )

    assert receipt == repeated
    assert receipt["n_rows"] == 3
    assert receipt["n_true"] == 2
    assert receipt["n_false"] == 1
    assert receipt["chronicle_source_record_id"] == (
        "publisher.regular_payment.month2025_01.all.beneficiaries"
    )
    assert receipt["chronicle_geography_level"] == "statistical_scope"
    for key in (
        "contract_sha256",
        "row_ids_sha256",
        "values_sha256",
        "row_values_sha256",
        "receipt_sha256",
    ):
        assert len(receipt[key]) == 64
    assert "row_ids" not in receipt
    assert "values" not in receipt

    changed_values = validate_population_input_frame(
        _frame(values=(False, False, True)),
        profile,
        mapping_id="regular_payment_scheme_population",
    )
    assert changed_values["row_ids_sha256"] == receipt["row_ids_sha256"]
    assert changed_values["values_sha256"] != receipt["values_sha256"]
    assert changed_values["row_values_sha256"] != receipt["row_values_sha256"]
    assert changed_values["receipt_sha256"] != receipt["receipt_sha256"]

    changed_ids = validate_population_input_frame(
        _frame(ids=(21, 22, 23)),
        profile,
        mapping_id="regular_payment_scheme_population",
    )
    assert changed_ids["row_ids_sha256"] != receipt["row_ids_sha256"]
    assert changed_ids["values_sha256"] == receipt["values_sha256"]
    assert changed_ids["row_values_sha256"] != receipt["row_values_sha256"]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((True, None, False), "missing values"),
        ((1, 0, 1), "boolean values"),
        (("yes", "no", "yes"), "boolean values"),
    ],
)
def test_ready_input_refuses_incomplete_or_proxy_vectors(values, message):
    with pytest.raises(ValueError, match=message):
        validate_population_input_frame(
            _frame(values=values),
            _profile(),
            mapping_id="regular_payment_scheme_population",
        )


def test_ready_input_refuses_missing_column_and_mapping_omission():
    frame = _frame()
    missing = Frame(
        {
            "person": frame.table("person").drop(
                columns=["regular_payment_recipient_indicator"]
            ),
            "household": frame.table("household"),
        },
        frame.schema,
        {"household": frame.weights_for("household")},
    )
    with pytest.raises(ValueError, match="missing Frame column"):
        validate_population_input_frame(
            missing,
            _profile(),
            mapping_id="regular_payment_scheme_population",
        )
    with pytest.raises(KeyError, match="Unknown scheme-population mapping"):
        validate_population_input_frame(
            frame,
            _profile(),
            mapping_id="undeclared_bypass",
        )


def test_receipt_is_json_safe_without_exposing_microdata_rows():
    receipt = validate_population_input_frame(
        _frame(ids=(910001, 910002, 910003)),
        _profile(),
        mapping_id="regular_payment_scheme_population",
    )
    rendered = json.dumps(receipt, sort_keys=True, allow_nan=False)

    assert "910001" not in rendered
    assert "regular_payment_recipient_indicator" in rendered
