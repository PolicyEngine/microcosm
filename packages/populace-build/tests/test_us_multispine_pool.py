from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace.build.gates import GateResult
from populace.build.us_runtime.acs_transfer import (
    declared_acs_transfer_target_families,
)
from populace.build.us_runtime.multispine_pool import (
    POOL_OPERATOR_ORDER,
    PoolStageOutput,
    materialize_multispine_agreement_outputs,
    pool_transfer_target_families,
    run_multispine_pool_path,
    seed_multispine_pool_inputs,
)
from populace.build.us_runtime.puf_support import (
    PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
)
from populace.build.us_runtime.spine_agreement import (
    SpineAgreementSpec,
    spine_agreement_gate,
)
from populace.build.us_runtime.spine_assembly import assemble_spines
from populace.build.us_runtime.support_provenance import (
    support_clone_index_column,
    support_source_id_column,
)
from populace.build.us_runtime.take_up_contract import load_take_up_contract
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


def test_pool_transfer_plan_is_the_fixed_declared_qrf_surface() -> None:
    assert pool_transfer_target_families() == declared_acs_transfer_target_families()


class _FakeEngine:
    def __init__(self) -> None:
        self.materialized_person_ids: list[list[int]] = []

    def default_values(self, names: list[str]) -> dict[str, object]:
        programs = load_take_up_contract().program_map()
        return {name: programs[name].default for name in names}

    def materialize(
        self,
        bundle: Frame,
        variables: list[str],
        period: int,
    ) -> dict[str, np.ndarray]:
        assert variables == ["ssi"]
        assert period == 2024
        person = bundle.table("person")
        self.materialized_person_ids.append(person["person_id"].astype(int).tolist())
        return {"ssi": person["age"].to_numpy(dtype=np.float64)}


def _assembled_cloned_with_partial_take_up() -> Frame:
    asec = _source_frame()
    tables = {entity: asec.table(entity).copy() for entity in asec.entities}
    tables["spm_unit"]["takes_up_tanf_if_eligible"] = [True, False]
    tables["spm_unit"]["takes_up_housing_assistance_if_eligible"] = [True, False]
    tables["person"]["takes_up_medicare_if_eligible"] = [False, True]
    asec = Frame(
        tables,
        asec.schema,
        {"household": asec.weights_for("household")},
        asec.strata,
    )
    acs = _source_frame()
    tables = {entity: acs.table(entity).copy() for entity in acs.entities}
    tables["spm_unit"]["takes_up_housing_assistance_if_eligible"] = [False, True]
    acs = Frame(
        tables,
        acs.schema,
        {"household": acs.weights_for("household")},
        acs.strata,
    )
    assembled = assemble_spines(
        {"asec": asec, "acs": acs},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    from populace.build.us_runtime.puf_support import (
        clone_us_frame_for_puf_support,
    )

    return clone_us_frame_for_puf_support(assembled)


def test_pool_seed_stage_preserves_inputs_and_receipts_disclosed_defaults() -> None:
    frame = _assembled_cloned_with_partial_take_up()
    before_person = frame.table("person")
    before_spm = frame.table("spm_unit")
    engine = _FakeEngine()

    result = seed_multispine_pool_inputs(frame, engine=engine)

    after_person = result.frame.table("person")
    after_spm = result.frame.table("spm_unit")
    measured_person = before_person["takes_up_medicare_if_eligible"].notna()
    measured_spm = before_spm["takes_up_tanf_if_eligible"].notna()
    assert (
        after_person.loc[measured_person, "takes_up_medicare_if_eligible"].tolist()
        == before_person.loc[measured_person, "takes_up_medicare_if_eligible"].tolist()
    )
    assert (
        after_spm.loc[measured_spm, "takes_up_tanf_if_eligible"].tolist()
        == before_spm.loc[measured_spm, "takes_up_tanf_if_eligible"].tolist()
    )

    contract = load_take_up_contract()
    for program in contract.programs:
        assert not result.frame.table(program.entity)[program.variable].isna().any()
    tanf = result.receipt["programs"]["takes_up_tanf_if_eligible"]
    assert tanf["provenance_kind"] == "administrative_seed_or_preserved_input"
    medicare = result.receipt["programs"]["takes_up_medicare_if_eligible"]
    assert medicare["provenance_kind"] == (
        "preserved_input_or_disclosed_engine_default"
    )
    assert medicare["defaulted_rows"] == 4

    spm = result.frame.table("spm_unit")
    source_id = support_source_id_column("spm_unit")
    clone_index = support_clone_index_column("spm_unit")
    for _source, rows in spm.groupby(source_id):
        assert set(rows[clone_index]) == {0, 1}
        assert rows["takes_up_tanf_if_eligible"].nunique() == 1


def test_simulated_ssi_lives_only_on_receipt_preserving_gate_view() -> None:
    frame = seed_multispine_pool_inputs(
        _assembled_cloned_with_partial_take_up(),
        engine=_FakeEngine(),
    ).frame
    engine = _FakeEngine()

    result = materialize_multispine_agreement_outputs(frame, engine=engine)

    assert "ssi" not in frame.table("person")
    assert "ssi" in result.frame.table("person")
    assert result.frame.metadata == frame.metadata
    assert result.receipt["persisted_to_pool"] is False
    assert result.receipt["formula_outputs"]["ssi"]["rows"] == frame.n("person")
    assert sum(map(len, engine.materialized_person_ids)) == frame.n("person")


def test_simulation_defaults_are_disposable_and_receipted() -> None:
    frame = seed_multispine_pool_inputs(
        _assembled_cloned_with_partial_take_up(),
        engine=_FakeEngine(),
    ).frame
    person = frame.table("person").copy()
    person.loc[person.index[0], "age"] = np.nan
    frame = _replace_person(frame, person)

    class ProjectionEngine(_FakeEngine):
        def variables(self) -> list[str]:
            return ["age"]

        def variable_metadata(self, name: str) -> object:
            assert name == "age"
            return SimpleNamespace(entity="person")

        def default_values(self, names: list[str]) -> dict[str, object]:
            assert names == ["age"]
            return {"age": 0.0}

        def materialize(
            self,
            bundle: Frame,
            variables: list[str],
            period: int,
        ) -> dict[str, np.ndarray]:
            assert not bundle.table("person")["age"].isna().any()
            return super().materialize(bundle, variables, period)

    result = materialize_multispine_agreement_outputs(
        frame,
        engine=ProjectionEngine(),
    )

    assert result.frame.table("person")["age"].isna().sum() == 1
    assert result.frame.table("person").loc[person.index[0], "ssi"] == 0.0
    assert result.receipt["simulation_projection_default_fills"]["age"] == {
        "entity": "person",
        "rows": 1,
        "value": 0.0,
        "persisted_to_pool": False,
    }
