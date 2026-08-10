"""Late-stage producer DAG doctrine regressions (microcosm#653)."""

from __future__ import annotations

from collections import OrderedDict

import pytest

from microcosm.build.us_runtime.late_producer_dag import (
    ProducerContract,
    ProducerInput,
    ProducerOutput,
    derive_producer_schedule,
    run_producer_when_ready,
)


def _contract(name: str, *dependencies: str) -> ProducerContract:
    return ProducerContract(
        name=name,
        kind="fixture",
        inputs=tuple(
            ProducerInput(
                entity="person",
                column=f"{dependency}_output",
                required_scope="whole_pool",
                producing_stage=dependency,
            )
            for dependency in dependencies
        ),
        outputs=(
            ProducerOutput(
                entity="person",
                column=f"{name}_output",
                coverage_scope="whole_pool",
            ),
        ),
    )


def test_unfilled_late_input_refuses_before_producer_runs() -> None:
    requirement = ProducerInput(
        entity="person",
        column="late_input",
        required_scope="cps_projection",
        producing_stage="transfer:person/puf_tax_itemization__batch_5",
    )
    consumer = ProducerContract(
        name="with_fixture_consumer",
        kind="source_operator",
        inputs=(requirement,),
        outputs=(),
    )
    invoked = False

    def callback() -> None:
        nonlocal invoked
        invoked = True

    with pytest.raises(
        ValueError,
        match=(
            r"with_fixture_consumer.*person\.late_input.*1 unfilled.*"
            r"cps_projection.*transfer:person/puf_tax_itemization__batch_5"
        ),
    ):
        run_producer_when_ready(
            consumer,
            callback,
            unfilled_rows={requirement: 1},
            absence_receipts={},
        )

    assert invoked is False


def test_synthetic_producer_cycle_is_rejected_with_named_cycle() -> None:
    registry = {
        "alpha": _contract("alpha", "charlie"),
        "bravo": _contract("bravo", "alpha"),
        "charlie": _contract("charlie", "bravo"),
    }

    with pytest.raises(
        RuntimeError,
        match=r"alpha -> bravo -> charlie -> alpha",
    ):
        derive_producer_schedule(registry)


def test_derived_schedule_is_byte_stable_under_registry_iteration_order() -> None:
    contracts = (
        _contract("alpha"),
        _contract("bravo"),
        _contract("charlie", "alpha", "bravo"),
        _contract("delta", "charlie"),
    )
    forward = OrderedDict((contract.name, contract) for contract in contracts)
    reverse = OrderedDict(
        (contract.name, contract) for contract in reversed(contracts)
    )

    forward_schedule = derive_producer_schedule(forward)
    reverse_schedule = derive_producer_schedule(reverse)

    assert forward_schedule.order == reverse_schedule.order
    assert forward_schedule.waves == reverse_schedule.waves
    assert forward_schedule.edges == reverse_schedule.edges
    assert forward_schedule.canonical_json == reverse_schedule.canonical_json
    assert forward_schedule.sha256 == reverse_schedule.sha256
