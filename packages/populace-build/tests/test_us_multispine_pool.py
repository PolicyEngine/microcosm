from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from populace.build.gates import GateResult
from populace.build.us_runtime.multispine_pool import (
    POOL_OPERATOR_ORDER,
    PoolStageOutput,
    pool_transfer_target_families,
    run_multispine_pool_path,
)
from populace.build.us_runtime.puf_support import (
    PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
)
from populace.build.us_runtime.spine_agreement import (
    SpineAgreementSpec,
    spine_agreement_gate,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def _source_frame(*, offset: float = 0.0) -> Frame:
    ids = np.asarray([1, 2], dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
            "age": np.asarray([30.0, 50.0]),
            "measured": np.asarray([1.0, 2.0]) + offset,
        }
    )
    tables = {
        "person": person,
        **{
            entity: pd.DataFrame({f"{entity}_id": ids})
            for entity in US_SCHEMA.group_entities
        },
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([2.0, 2.0]),
                WeightKind.DESIGN,
            )
        },
        pd.Series(["fixture", "fixture"], dtype=object),
    )


def _replace_person(
    frame: Frame, person: pd.DataFrame, *, metadata: bool = True
) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata if metadata else None,
    )


def _operator(
    name: str,
    order: list[str],
    transform: Callable[[pd.DataFrame], None],
) -> Callable[[Frame], PoolStageOutput]:
    def apply(frame: Frame) -> PoolStageOutput:
        order.append(name)
        person = frame.table("person").copy()
        transform(person)
        return PoolStageOutput(
            _replace_person(frame, person),
            {"operator": name},
        )

    return apply


def _fixture_registry() -> tuple[SpineAgreementSpec, ...]:
    return (
        SpineAgreementSpec("person", "imputed", ("transferred",)),
        SpineAgreementSpec("person", "derived", ("derived",)),
        SpineAgreementSpec("person", "take_up", ("seeded",)),
        SpineAgreementSpec("person", "simulated_output", ("ssi",)),
    )


def test_full_operator_path_is_ordered_and_keeps_simulation_out_of_pool() -> None:
    order: list[str] = []
    impute = _operator(
        "impute",
        order,
        lambda person: person.__setitem__("transferred", person["age"]),
    )
    derive = _operator(
        "derive",
        order,
        lambda person: person.__setitem__("derived", person["transferred"] * 2),
    )
    seed = _operator(
        "seed",
        order,
        lambda person: person.__setitem__("seeded", person["age"] >= 40),
    )
    simulate = _operator(
        "simulate",
        order,
        lambda person: person.__setitem__("ssi", person["derived"]),
    )

    result = run_multispine_pool_path(
        _source_frame(),
        _source_frame(),
        impute=impute,
        derive=derive,
        seed=seed,
        simulate=simulate,
        agreement_gate=lambda frame: spine_agreement_gate(
            frame,
            registry=_fixture_registry(),
        ),
    )

    assert tuple(["assemble", "clone", *order, "agreement"]) == POOL_OPERATOR_ORDER
    assert order == ["impute", "derive", "seed", "simulate"]
    assert result.agreement_gate.passed
    assert result.simulation_ready
    assert "ssi" not in result.frame.table("person")
    assert result.assembly_receipt["channels"] == ["asec", "acs"]
    assert result.assembly_receipt["native_row_counts"]["person"] == {
        "asec": 2,
        "acs": 2,
    }
    assert result.provenance_counts["person"] == {
        "rows": 8,
        "by_source_channel": {"asec": 4, "acs": 4},
        "by_clone_index": {"0": 4, "1": 4},
        "by_source_channel_and_clone_index": {
            "asec": {"0": 2, "1": 2},
            "acs": {"0": 2, "1": 2},
        },
    }
    assert result.stage_receipts == {
        "impute": {"operator": "impute"},
        "derive": {"operator": "derive"},
        "seed": {"operator": "seed"},
        "simulate": {"operator": "simulate"},
    }


def test_agreement_failures_remain_batched_in_terminal_result() -> None:
    order: list[str] = []

    def imputed(person: pd.DataFrame) -> None:
        person["transferred"] = person["measured"]

    def derived(person: pd.DataFrame) -> None:
        person["derived"] = person["transferred"]

    def seeded(person: pd.DataFrame) -> None:
        person["seeded"] = person["measured"] > 0

    def simulated(person: pd.DataFrame) -> None:
        person["ssi"] = person["measured"]

    result = run_multispine_pool_path(
        _source_frame(),
        _source_frame(offset=99.0),
        impute=_operator("impute", order, imputed),
        derive=_operator("derive", order, derived),
        seed=_operator("seed", order, seeded),
        simulate=_operator("simulate", order, simulated),
        agreement_gate=lambda frame: spine_agreement_gate(
            frame,
            registry=_fixture_registry(),
        ),
    )

    assert not result.agreement_gate.passed
    assert not result.simulation_ready
    assert len(result.agreement_gate.failures) >= 3
    assert result.agreement_gate.details["tolerances"] == {
        "incidence_ratio_bounds": [0.8, 1.25],
        "max_quantile_envelope_distance": 0.25,
    }
    assert order == ["impute", "derive", "seed", "simulate"]


def test_operator_metadata_drop_surfaces_assembly_receipt_error() -> None:
    def no_op(frame: Frame) -> PoolStageOutput:
        return PoolStageOutput(frame)

    def drop_receipt(frame: Frame) -> PoolStageOutput:
        person = frame.table("person").copy()
        return PoolStageOutput(_replace_person(frame, person, metadata=False))

    with pytest.raises(
        ValueError,
        match="multispine pool derive output:.*no assembly manifest",
    ):
        run_multispine_pool_path(
            _source_frame(),
            _source_frame(),
            impute=no_op,
            derive=drop_receipt,
            seed=no_op,
            simulate=no_op,
            agreement_gate=lambda _frame: GateResult("fixture", True),
        )


def test_clone_safe_id_violation_surfaces_assembly_error() -> None:
    oversized = PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID + 1
    asec = _source_frame()
    tables = {entity: asec.table(entity).copy() for entity in asec.entities}
    tables["person"].loc[0, "person_id"] = oversized
    asec = Frame(
        tables,
        asec.schema,
        {"household": asec.weights_for("household")},
        asec.strata,
    )

    def unreachable(_frame: Frame) -> PoolStageOutput:
        raise AssertionError("Assembly violations must precede every operator.")

    with pytest.raises(ValueError, match="Spine 'asec'.*clone-safe bound"):
        run_multispine_pool_path(
            asec,
            _source_frame(),
            impute=unreachable,
            derive=unreachable,
            seed=unreachable,
            simulate=unreachable,
        )


def test_pool_transfer_plan_covers_every_take_up_without_duplicates() -> None:
    families = pool_transfer_target_families()
    ownership: dict[str, tuple[str, str]] = {}
    for entity, by_family in families.items():
        for family, columns in by_family.items():
            for column in columns:
                assert column not in ownership
                ownership[column] = (entity, family)

    expected = {
        "takes_up_snap_if_eligible",
        "takes_up_tanf_if_eligible",
        "takes_up_eitc",
        "takes_up_medicaid_if_eligible",
        "takes_up_chip_if_eligible",
        "takes_up_basic_health_program_if_eligible",
        "takes_up_medicare_if_eligible",
        "takes_up_ssi_if_eligible",
        "takes_up_dc_ptc",
        "takes_up_head_start_if_eligible",
        "takes_up_early_head_start_if_eligible",
        "takes_up_housing_assistance_if_eligible",
        "takes_up_aca_if_eligible",
    }
    assert expected <= set(ownership)
