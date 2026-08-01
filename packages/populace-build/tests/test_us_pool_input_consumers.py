from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TypedDict

import pytest

from populace.build.us_runtime.multispine_pool import (
    PoolInputSurfaceEntry,
    pool_input_surface,
)
from populace.frame.adapters.policyengine_us import (
    PolicyEngineUSVariableMetadataIndex,
)


class NonEngineConsumerClass(StrEnum):
    """Reviewed substantive consumers outside PolicyEngine-US."""

    POPULACE_REGISTER = "populace_register"
    POPULACE_GATE = "populace_gate"
    POPULACE_DERIVATION = "populace_derivation"
    CLIENT_SURFACE = "client_surface"


class _ReviewedNonEngineConsumer(TypedDict):
    consumer_class: NonEngineConsumerClass
    consumer: str
    justification: str


NON_ENGINE_CONSUMER_ALLOWLIST: dict[str, _ReviewedNonEngineConsumer] = {
    "previous_year_income_available": {
        "consumer_class": NonEngineConsumerClass.POPULACE_GATE,
        "consumer": (
            "populace.build.us_runtime.prior_year_income."
            "us_prior_year_income_signal_gate"
        ),
        "justification": (
            "Persists adjacent-year provenance and is validated for "
            "non-default signal, channel support, and clone agreement."
        ),
    },
    "qualified_bdc_income": {
        "consumer_class": NonEngineConsumerClass.POPULACE_GATE,
        "consumer": ("populace.build.us_runtime.qbi_inputs.us_qbi_inputs_signal_gate"),
        "justification": (
            "Consumed by QBI reconciliation and the qualified-BDC exposure invariant."
        ),
    },
}


try:
    _ENGINE_INDEX = PolicyEngineUSVariableMetadataIndex()
except ImportError:
    pytest.skip(
        "requires the policyengine-us [us] extra",
        allow_module_level=True,
    )


_POOL_INPUT_SURFACE = pool_input_surface()


def _validate_allowlist(
    surface: Sequence[PoolInputSurfaceEntry],
    allowlist: Mapping[str, Mapping[str, object]],
    engine_index: PolicyEngineUSVariableMetadataIndex,
) -> None:
    surface_names = {entry.variable for entry in surface}
    expected_fields = {"consumer_class", "consumer", "justification"}

    for variable, review in sorted(allowlist.items()):
        assert variable in surface_names, (
            f"{variable}: stale non-engine consumer allowlist entry is not in "
            "the current pool input surface"
        )
        assert set(review) == expected_fields, (
            f"{variable}: reviewed non-engine consumer entry must contain "
            f"exactly {sorted(expected_fields)}"
        )
        assert isinstance(
            review["consumer_class"],
            NonEngineConsumerClass,
        ), (
            f"{variable}: consumer_class must be a member of the closed "
            "NonEngineConsumerClass enum"
        )
        for field in ("consumer", "justification"):
            value = review[field]
            assert isinstance(value, str) and value.strip(), (
                f"{variable}: {field} must be a non-empty string"
            )
        assert not engine_index.consumer_receipts(variable), (
            f"{variable}: redundant non-engine consumer allowlist entry; "
            "PolicyEngine-US now has an external consumer receipt"
        )


def _assert_surface_entry_has_consumer(
    entry: PoolInputSurfaceEntry,
    allowlist: Mapping[str, Mapping[str, object]],
    engine_index: PolicyEngineUSVariableMetadataIndex,
) -> None:
    receipts = engine_index.consumer_receipts(entry.variable)
    reviewed = entry.variable in allowlist
    if receipts:
        assert not reviewed, (
            f"{entry.variable}: redundant non-engine consumer allowlist entry; "
            "PolicyEngine-US now has an external consumer receipt"
        )
        return
    assert reviewed, (
        f"{entry.variable} (family={entry.family}) has no external "
        "PolicyEngine-US consumer receipt and no reviewed non-engine consumer"
    )


def _assert_consumer_guard(
    surface: Sequence[PoolInputSurfaceEntry],
    allowlist: Mapping[str, Mapping[str, object]],
    engine_index: PolicyEngineUSVariableMetadataIndex,
) -> None:
    _validate_allowlist(surface, allowlist, engine_index)
    for entry in surface:
        _assert_surface_entry_has_consumer(entry, allowlist, engine_index)


def _mutable_allowlist() -> dict[str, dict[str, object]]:
    return {
        variable: dict(review)
        for variable, review in NON_ENGINE_CONSUMER_ALLOWLIST.items()
    }


def _review_fixture() -> dict[str, object]:
    return {
        "consumer_class": NonEngineConsumerClass.POPULACE_GATE,
        "consumer": "tests.reviewed_consumer",
        "justification": "Mutation-authentication fixture.",
    }


def test_non_engine_consumer_allowlist_is_exact_and_current() -> None:
    assert set(NON_ENGINE_CONSUMER_ALLOWLIST) == {
        "previous_year_income_available",
        "qualified_bdc_income",
    }
    _validate_allowlist(
        _POOL_INPUT_SURFACE,
        NON_ENGINE_CONSUMER_ALLOWLIST,
        _ENGINE_INDEX,
    )


@pytest.mark.parametrize(
    "entry",
    _POOL_INPUT_SURFACE,
    ids=lambda entry: entry.variable,
)
def test_pool_input_has_engine_or_reviewed_non_engine_consumer(
    entry: PoolInputSurfaceEntry,
) -> None:
    _assert_surface_entry_has_consumer(
        entry,
        NON_ENGINE_CONSUMER_ALLOWLIST,
        _ENGINE_INDEX,
    )


@pytest.mark.parametrize(
    "entry",
    (
        PoolInputSurfaceEntry(
            variable="has_marketplace_health_coverage",
            entity="person",
            family="model_required_boolean",
            provenance=("mutation_reinjection",),
        ),
        PoolInputSurfaceEntry(
            variable="medicare_part_b_premiums_reported",
            entity="person",
            family="source_operator_cps_carried",
            provenance=("mutation_reinjection",),
        ),
    ),
    ids=lambda entry: entry.variable,
)
def test_guard_rejects_reinjected_consumerless_variable(
    entry: PoolInputSurfaceEntry,
) -> None:
    with pytest.raises(
        AssertionError,
        match=rf"^{re.escape(entry.variable)} .*family={entry.family}",
    ):
        _assert_consumer_guard(
            (*_POOL_INPUT_SURFACE, entry),
            NON_ENGINE_CONSUMER_ALLOWLIST,
            _ENGINE_INDEX,
        )


@pytest.mark.parametrize(
    "justification",
    ("", " \t\n"),
    ids=(
        "previous_year_income_available-empty",
        "previous_year_income_available-whitespace",
    ),
)
def test_guard_rejects_blank_allowlist_justification(
    justification: str,
) -> None:
    allowlist = _mutable_allowlist()
    allowlist["previous_year_income_available"]["justification"] = justification

    with pytest.raises(
        AssertionError,
        match=(
            r"^previous_year_income_available: justification must be a "
            r"non-empty string"
        ),
    ):
        _assert_consumer_guard(_POOL_INPUT_SURFACE, allowlist, _ENGINE_INDEX)


@pytest.mark.parametrize(
    "consumer",
    ("", " \t\n"),
    ids=(
        "previous_year_income_available-empty",
        "previous_year_income_available-whitespace",
    ),
)
def test_guard_rejects_blank_allowlist_consumer(consumer: str) -> None:
    allowlist = _mutable_allowlist()
    allowlist["previous_year_income_available"]["consumer"] = consumer

    with pytest.raises(
        AssertionError,
        match=(
            r"^previous_year_income_available: consumer must be a "
            r"non-empty string"
        ),
    ):
        _assert_consumer_guard(_POOL_INPUT_SURFACE, allowlist, _ENGINE_INDEX)


def test_guard_rejects_invalid_allowlist_consumer_class() -> None:
    allowlist = _mutable_allowlist()
    allowlist["previous_year_income_available"]["consumer_class"] = "carrier"

    with pytest.raises(
        AssertionError,
        match=(
            r"^previous_year_income_available: consumer_class must be a member "
            r"of the closed NonEngineConsumerClass enum"
        ),
    ):
        _assert_consumer_guard(_POOL_INPUT_SURFACE, allowlist, _ENGINE_INDEX)


def test_guard_rejects_stale_allowlist_entry() -> None:
    allowlist = _mutable_allowlist()
    allowlist["not_a_pool_input"] = _review_fixture()

    with pytest.raises(
        AssertionError,
        match=r"^not_a_pool_input: stale non-engine consumer allowlist entry",
    ):
        _assert_consumer_guard(_POOL_INPUT_SURFACE, allowlist, _ENGINE_INDEX)


def test_guard_rejects_redundant_allowlist_entry() -> None:
    variable = "hours_worked_last_week"
    allowlist = _mutable_allowlist()
    allowlist[variable] = _review_fixture()

    with pytest.raises(
        AssertionError,
        match=rf"^{variable}: redundant non-engine consumer allowlist entry",
    ):
        _assert_consumer_guard(_POOL_INPUT_SURFACE, allowlist, _ENGINE_INDEX)


@pytest.mark.parametrize(
    "variable",
    (
        "first_home_mortgage_balance",
        "first_home_mortgage_interest",
        "first_home_mortgage_origination_year",
        "second_home_mortgage_balance",
        "second_home_mortgage_interest",
        "second_home_mortgage_origination_year",
    ),
    ids=lambda variable: variable,
)
def test_colocated_mortgage_consumer_is_discoverable(variable: str) -> None:
    assert any(
        receipt.consumer == "deductible_mortgage_interest_tax_unit"
        and receipt.path.endswith("/mortgage_interest_structure.py")
        and receipt.line > 0
        and receipt.kind == "entity_call"
        for receipt in _ENGINE_INDEX.consumer_receipts(variable)
    )


@pytest.mark.parametrize(
    ("variable", "consumer", "filename"),
    (
        (
            "has_champva_health_coverage_at_interview",
            "qualifying_non_marketplace_health_coverage_type_count_at_interview",
            "qualifying_non_marketplace_health_coverage_type_count_at_interview.py",
        ),
        (
            "domestic_production_ald",
            "above_the_line_deductions",
            "above_the_line_deductions.py",
        ),
    ),
    ids=(
        "has_champva_health_coverage_at_interview",
        "domestic_production_ald",
    ),
)
def test_parameter_consumer_is_discoverable(
    variable: str,
    consumer: str,
    filename: str,
) -> None:
    assert any(
        receipt.consumer == consumer
        and receipt.path.endswith(f"/{filename}")
        and receipt.line > 0
        and receipt.kind == "parameter_adds"
        for receipt in _ENGINE_INDEX.consumer_receipts(variable)
    )


@pytest.mark.parametrize(
    "variable",
    (
        "estate_income_would_be_qualified",
        "farm_operations_income_would_be_qualified",
        "farm_rent_income_would_be_qualified",
        "partnership_s_corp_income_would_be_qualified",
        "rental_income_would_be_qualified",
        "self_employment_income_would_be_qualified",
    ),
    ids=lambda variable: variable,
)
def test_constructed_qbi_consumer_is_discoverable(variable: str) -> None:
    assert any(
        receipt.consumer == "qbid_amount"
        and receipt.path.endswith("/qbid_amount.py")
        and receipt.line > 0
        and receipt.kind == "constructed_parameter_entity_call"
        for receipt in _ENGINE_INDEX.consumer_receipts(variable)
    )
