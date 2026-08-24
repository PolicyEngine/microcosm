from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Protocol, TypedDict

import pytest

from microcosm.build.us_runtime.cps_carried import WIC_CARRIER_ADJUDICATION_URL
from microcosm.build.us_runtime.multispine_pool import (
    PoolInputSurfaceEntry,
    pool_input_surface,
)
from microcosm.frame.adapters.policyengine_us import (
    ConsumerReceipt,
    PolicyEngineUSVariableMetadataIndex,
)
from microcosm.frame.schema import VariableMetadata


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
            "microcosm.build.us_runtime.prior_year_income."
            "us_prior_year_income_signal_gate"
        ),
        "justification": (
            "Persists adjacent-year provenance and is validated for "
            "non-default signal, channel support, and clone agreement."
        ),
    },
    "qualified_bdc_income": {
        "consumer_class": NonEngineConsumerClass.POPULACE_GATE,
        "consumer": ("microcosm.build.us_runtime.qbi_inputs.us_qbi_inputs_signal_gate"),
        "justification": (
            "Consumed by QBI reconciliation and the qualified-BDC exposure invariant."
        ),
    },
}

_REVIEWED_WIC_CONSUMERS = {
    # WICYN identifies the adult-female reporter/carrier, not the beneficiary.
    # These targets are reviewed from the six PolicyEngine-US 1.819.0 formulas;
    # output entity alone is insufficient because Pell aggregates by tax unit
    # and Virginia reads people inside an SPM-unit aggregation. Values pin the
    # direct receiver and the enclosing aggregation entity, in that order.
    ("ca_care_categorically_eligible", "parameter_add", "household"): (
        "household",
        "household",
    ),
    ("is_acp_eligible", "parameter_add", "spm_unit"): ("spm_unit", "spm_unit"),
    ("md_ccs_weekly_copay", "add", "spm_unit"): ("spm_unit", "spm_unit"),
    (
        "pell_grant_simplified_formula_applies",
        "parameter_add",
        "person",
    ): ("tax_unit", "tax_unit"),
    (
        "tx_dart_reduced_fare_program_eligible",
        "parameter_adds",
        "person",
    ): ("person", "person"),
    ("va_ccsp_income_test_waived", "entity_call", "spm_unit"): (
        "person",
        "spm_unit",
    ),
}


class _EngineConsumerIndex(Protocol):
    def consumer_receipts(self, name: str) -> tuple[ConsumerReceipt, ...]: ...

    def variable_metadata(self, name: str) -> VariableMetadata: ...


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
    engine_index: _EngineConsumerIndex,
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


def _assert_reviewed_engine_consumers(
    variable: str,
    receipts: Sequence[ConsumerReceipt],
    engine_index: _EngineConsumerIndex,
) -> None:
    if variable != "receives_wic":
        return

    actual = {
        (
            receipt.consumer,
            receipt.kind,
            engine_index.variable_metadata(receipt.consumer).entity,
            receipt.receiver_entity,
            receipt.aggregation_entity,
        )
        for receipt in receipts
    }
    expected = {
        (*signature, receiver_entity, aggregation_entity)
        for signature, (
            receiver_entity,
            aggregation_entity,
        ) in _REVIEWED_WIC_CONSUMERS.items()
    }
    assert actual == expected, (
        "receives_wic: PolicyEngine-US consumer signatures changed; preserve "
        "the reporting-adult carrier semantics and re-adjudicate every added, "
        "removed, or changed consumer before adapting: "
        f"{WIC_CARRIER_ADJUDICATION_URL}; expected={sorted(expected)!r}; "
        f"actual={sorted(actual)!r}"
    )


def _assert_surface_entry_has_consumer(
    entry: PoolInputSurfaceEntry,
    allowlist: Mapping[str, Mapping[str, object]],
    engine_index: _EngineConsumerIndex,
) -> None:
    receipts = engine_index.consumer_receipts(entry.variable)
    reviewed = entry.variable in allowlist
    if receipts:
        assert not reviewed, (
            f"{entry.variable}: redundant non-engine consumer allowlist entry; "
            "PolicyEngine-US now has an external consumer receipt"
        )
        _assert_reviewed_engine_consumers(
            entry.variable,
            receipts,
            engine_index,
        )
        return
    assert reviewed, (
        f"{entry.variable} (family={entry.family}) has no external "
        "PolicyEngine-US consumer receipt and no reviewed non-engine consumer"
    )


def _assert_consumer_guard(
    surface: Sequence[PoolInputSurfaceEntry],
    allowlist: Mapping[str, Mapping[str, object]],
    engine_index: _EngineConsumerIndex,
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


def test_receives_wic_consumers_are_exactly_reviewed() -> None:
    receipts = _ENGINE_INDEX.consumer_receipts("receives_wic")

    actual = {
        (
            receipt.consumer,
            receipt.kind,
            _ENGINE_INDEX.variable_metadata(receipt.consumer).entity,
            receipt.receiver_entity,
            receipt.aggregation_entity,
        )
        for receipt in receipts
    }
    assert actual == {
        (*signature, receiver_entity, aggregation_entity)
        for signature, (
            receiver_entity,
            aggregation_entity,
        ) in _REVIEWED_WIC_CONSUMERS.items()
    }
    _assert_reviewed_engine_consumers(
        "receives_wic",
        receipts,
        _ENGINE_INDEX,
    )


class _UnreviewedWICConsumerMutation:
    """Inject one future WIC consumer into the live AST index."""

    _CONSUMER = "synthetic_person_wic_consumer"

    def __init__(self, delegate: _EngineConsumerIndex) -> None:
        self._delegate = delegate

    def consumer_receipts(self, name: str) -> tuple[ConsumerReceipt, ...]:
        receipts = self._delegate.consumer_receipts(name)
        if name != "receives_wic":
            return receipts
        return (
            *receipts,
            ConsumerReceipt(
                consumer=self._CONSUMER,
                path="variables/synthetic_person_wic_consumer.py",
                line=19,
                kind="entity_call",
                receiver_entity="person",
                aggregation_entity="person",
            ),
        )

    def variable_metadata(self, name: str) -> VariableMetadata:
        if name == self._CONSUMER:
            return VariableMetadata(
                name=name,
                entity="person",
                dtype="bool",
                period="month",
            )
        return self._delegate.variable_metadata(name)


def test_receives_wic_guard_rejects_unreviewed_consumer_mutation() -> None:
    mutated_index = _UnreviewedWICConsumerMutation(_ENGINE_INDEX)

    with pytest.raises(
        AssertionError,
        match=(
            r"^receives_wic: PolicyEngine-US consumer signatures changed; "
            r"preserve the reporting-adult carrier semantics and re-adjudicate "
            r"every added, removed, or changed consumer.*"
        ),
    ):
        _assert_consumer_guard(
            _POOL_INPUT_SURFACE,
            NON_ENGINE_CONSUMER_ALLOWLIST,
            mutated_index,
        )


class _ChangedWICAggregationMutation:
    """Change only one reviewed consumer's aggregation grain."""

    _CONSUMER = "va_ccsp_income_test_waived"

    def __init__(self, delegate: _EngineConsumerIndex) -> None:
        self._delegate = delegate

    def consumer_receipts(self, name: str) -> tuple[ConsumerReceipt, ...]:
        receipts = self._delegate.consumer_receipts(name)
        if name != "receives_wic":
            return receipts
        return tuple(
            ConsumerReceipt(
                consumer=receipt.consumer,
                path=receipt.path,
                line=receipt.line,
                kind=receipt.kind,
                receiver_entity=receipt.receiver_entity,
                aggregation_entity=(
                    "household"
                    if receipt.consumer == self._CONSUMER
                    else receipt.aggregation_entity
                ),
            )
            for receipt in receipts
        )

    def variable_metadata(self, name: str) -> VariableMetadata:
        return self._delegate.variable_metadata(name)


def test_receives_wic_guard_rejects_changed_aggregation_entity() -> None:
    mutated_index = _ChangedWICAggregationMutation(_ENGINE_INDEX)

    with pytest.raises(
        AssertionError,
        match=r"^receives_wic: PolicyEngine-US consumer signatures changed;",
    ):
        _assert_reviewed_engine_consumers(
            "receives_wic",
            mutated_index.consumer_receipts("receives_wic"),
            mutated_index,
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
    ("variable", "consumer", "kind"),
    (
        (
            "first_home_mortgage_balance",
            "deductible_mortgage_interest_tax_unit",
            "entity_call",
        ),
        (
            "first_home_mortgage_interest",
            "home_mortgage_interest_tax_unit",
            "add",
        ),
        (
            "first_home_mortgage_origination_year",
            "deductible_mortgage_interest_tax_unit",
            "entity_call",
        ),
        (
            "second_home_mortgage_balance",
            "deductible_mortgage_interest_tax_unit",
            "entity_call",
        ),
        (
            "second_home_mortgage_interest",
            "home_mortgage_interest_tax_unit",
            "add",
        ),
        (
            "second_home_mortgage_origination_year",
            "deductible_mortgage_interest_tax_unit",
            "entity_call",
        ),
    ),
    ids=(
        "first_home_mortgage_balance",
        "first_home_mortgage_interest",
        "first_home_mortgage_origination_year",
        "second_home_mortgage_balance",
        "second_home_mortgage_interest",
        "second_home_mortgage_origination_year",
    ),
)
def test_colocated_mortgage_consumer_is_discoverable(
    variable: str,
    consumer: str,
    kind: str,
) -> None:
    assert any(
        receipt.consumer == consumer
        and receipt.path.endswith("/mortgage_interest_structure.py")
        and receipt.line > 0
        and receipt.kind == kind
        for receipt in _ENGINE_INDEX.consumer_receipts(variable)
    )


@pytest.mark.parametrize(
    ("variable", "consumer", "filename"),
    (
        (
            "is_disabled",
            "il_pi_has_developmental_delay",
            "il_pi_has_developmental_delay.py",
        ),
        (
            "taxable_pension_income",
            "ok_pension_subtraction",
            "ok_pension_subtraction.py",
        ),
    ),
    ids=("is_disabled", "taxable_pension_income"),
)
def test_group_member_consumer_is_discoverable(
    variable: str,
    consumer: str,
    filename: str,
) -> None:
    assert any(
        receipt.consumer == consumer
        and receipt.path.endswith(f"/{filename}")
        and receipt.line > 0
        and receipt.kind == "entity_call"
        for receipt in _ENGINE_INDEX.consumer_receipts(variable)
    )


def test_model_api_reference_helper_consumer_is_discoverable() -> None:
    assert any(
        receipt.consumer == "is_ssi_aged_blind_disabled"
        and receipt.path.endswith("/is_ssi_aged_blind_disabled.py")
        and receipt.line > 0
        and receipt.kind == "helper_call"
        for receipt in _ENGINE_INDEX.consumer_receipts("is_blind")
    )


def test_strike_benefits_market_income_consumer_is_discoverable() -> None:
    assert any(
        receipt.consumer == "market_income"
        and receipt.path.endswith("/market_income.py")
        and receipt.line > 0
        for receipt in _ENGINE_INDEX.consumer_receipts("strike_benefits")
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
