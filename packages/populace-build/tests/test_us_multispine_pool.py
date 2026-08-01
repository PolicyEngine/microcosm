from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from collections import Counter
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from populace.build.gates import GateResult
from populace.build.source_runtime import SourceRuntimeError
from populace.build.us_runtime import acs_transfer as acs_transfer_module
from populace.build.us_runtime import housing_inputs as housing_inputs_module
from populace.build.us_runtime import multispine_pool as multispine_pool_module
from populace.build.us_runtime import prior_year_income as prior_year_income_module
from populace.build.us_runtime import puf_support as puf_support_module
from populace.build.us_runtime.acs_transfer import (
    declared_acs_transfer_target_families,
)
from populace.build.us_runtime.multispine_pool import (
    POOL_DEFERRED_TRANSFER_INPUTS,
    POOL_DERIVE_OPERATOR_ORDER,
    POOL_OPERATOR_CONTRACTS,
    POOL_OPERATOR_ORDER,
    POOL_POST_CLONE_SOURCE_OPERATOR_ORDER,
    POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER,
    POOL_SOURCE_OPERATOR_CONTRACTS,
    POOL_SOURCE_OPERATOR_ORDER,
    POOL_SPINE_AGREEMENT_REGISTRY,
    MultispinePoolResult,
    PoolStageOutput,
    _complete_schedule_d_input,
    materialize_multispine_agreement_outputs,
    materialize_pool_deferred_transfer_inputs,
    pool_transfer_target_families,
    prepare_multispine_puf_predictors,
    prepare_multispine_source_inputs_for_clone,
    run_multispine_pool_path,
    seed_multispine_pool_inputs,
)
from populace.build.us_runtime.operator_boundary import (
    PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
)
from populace.build.us_runtime.prior_year_income import (
    with_us_prior_year_income_inputs,
)
from populace.build.us_runtime.puf_support import (
    PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
    clone_us_frame_for_puf_support,
)
from populace.build.us_runtime.spine_agreement import (
    SpineAgreementSpec,
    default_spine_agreement_registry,
    spine_agreement_gate,
    validate_spine_agreement_registry,
)
from populace.build.us_runtime.spine_assembly import assemble_spines
from populace.build.us_runtime.support_provenance import (
    support_clone_index_column,
    support_source_id_column,
)
from populace.build.us_runtime.take_up_contract import load_take_up_contract
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

_EXPECTED_POOL_SOURCE_OPERATOR_ORDER = (
    "derive_us_cps_carried_inputs",
    "with_us_hours_worked_inputs",
    "with_us_prior_year_income_inputs",
    "with_us_relationship_inputs",
    "with_us_medicare_take_up_input",
    "with_us_housing_inputs",
    "with_us_eligibility_inputs",
    "with_us_pregnancy_inputs",
    "with_us_wic_claim_input",
    "impute_us_housing_assistance_to_puf_support",
    "with_us_child_support_inputs",
    "with_us_disability_benefits",
    "with_us_workers_compensation",
    "with_us_weeks_unemployed",
    "with_us_childcare_inputs",
    "with_us_adult_care_inputs",
    "with_us_energy_subsidy_input",
    "with_us_retirement_contribution_inputs",
    "with_us_retirement_distribution_inputs",
    "with_us_immigration_inputs",
    "with_us_education_inputs",
)

_EXPECTED_PRE_CLONE_SOURCE_OPERATOR_ORDER = (
    "derive_us_cps_carried_inputs",
    "with_us_hours_worked_inputs",
    "with_us_prior_year_income_inputs",
    "with_us_relationship_inputs",
    "with_us_housing_inputs",
    "with_us_eligibility_inputs",
)

_EXPECTED_POST_CLONE_SOURCE_OPERATOR_ORDER = (
    "with_us_prior_year_income_inputs",
    "with_us_medicare_take_up_input",
    "with_us_pregnancy_inputs",
    "with_us_wic_claim_input",
    "impute_us_housing_assistance_to_puf_support",
    "with_us_child_support_inputs",
    "with_us_disability_benefits",
    "with_us_workers_compensation",
    "with_us_weeks_unemployed",
    "with_us_childcare_inputs",
    "with_us_adult_care_inputs",
    "with_us_energy_subsidy_input",
    "with_us_retirement_contribution_inputs",
    "with_us_retirement_distribution_inputs",
    "with_us_immigration_inputs",
    "with_us_education_inputs",
)


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


def _prior_year_source_frame() -> Frame:
    ids = np.arange(1, 5, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids,
            "person_spm_unit_id": ids,
            "person_family_id": ids,
            "person_marital_unit_id": ids,
            "source_year": [2023, 2024, 2023, 2024],
            "PERIDNUM": ["A", "A", "B", "B"],
            "WSAL_VAL": [100.0, 200.0, 300.0, 400.0],
            "SEMP_VAL": [-20.0, 30.0, 40.0, -50.0],
            "I_ERNVAL": [0, 0, 0, 0],
            "I_SEVAL": [0, 0, 0, 0],
            "age": [30.0, 31.0, 40.0, 41.0],
            "is_female": [False, False, True, True],
            "has_esi": [True, True, False, False],
            "tax_unit_role_input": ["PRIMARY"] * 4,
            "employment_income_before_lsr": [100.0, 200.0, 300.0, 400.0],
            "self_employment_income_before_lsr": [-20.0, 30.0, 40.0, -50.0],
            "SS_VAL": [0.0, 0.0, 10.0, 10.0],
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": ids,
                "filing_status_input": ["SINGLE"] * 4,
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(4, dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


class _PriorYearFitted:
    def predict(self, test: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        rows = np.arange(len(test), dtype=np.float64)
        return pd.DataFrame(
            {
                "employment_income_last_year": 1_000.0 + rows,
                "self_employment_income_last_year": -10.0 + rows,
            },
            index=test.index,
        )


class _PriorYearQRF:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def fit(self, *_args: object, **_kwargs: object) -> _PriorYearFitted:
        return _PriorYearFitted()


def _real_pre_clone_source_frame() -> Frame:
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [1, 1, 2, 2],
            "person_tax_unit_id": [101, 101, 102, 102],
            "person_spm_unit_id": [201, 201, 202, 202],
            "person_family_id": [301, 301, 302, 302],
            "person_marital_unit_id": [401, 402, 403, 404],
            "source_year": [2023, 2023, 2024, 2024],
            "PERIDNUM": ["parent", "child", "parent", "child"],
            "WSAL_VAL": [40_000.0, 0.0, 50_000.0, 0.0],
            "SEMP_VAL": [100.0, 0.0, 200.0, 0.0],
            "I_ERNVAL": [0, 0, 0, 0],
            "I_SEVAL": [0, 0, 0, 0],
            "A_AGE": [40, 10, 41, 11],
            "A_SEX": [1, 2, 1, 2],
            "HRSWK": [40, 0, 35, 0],
            "A_HRS1": [42, 0, 30, 0],
            "WKSWORK": [52, 0, 48, 0],
            "OI_VAL": [0.0, 0.0, 0.0, 0.0],
            "OI_OFF": [0, 0, 0, 0],
            "PH_SEQ": [10, 10, 20, 20],
            "P_SEQ": [1, 2, 1, 2],
            "A_MARITL": [7, 7, 7, 7],
            "A_LINENO": [1, 2, 1, 2],
            "PEPAR1": [-1, 1, -1, 1],
            "PEPAR2": [-1, -1, -1, -1],
            "PEDISDRS": [2, 2, 2, 2],
            "PEDISEAR": [2, 2, 2, 2],
            "PEDISEYE": [2, 2, 2, 2],
            "PEDISOUT": [2, 2, 2, 2],
            "PEDISPHY": [2, 2, 2, 2],
            "PEDISREM": [2, 2, 2, 2],
            "A_HSCOL": [0, 0, 0, 0],
            "A_FTPT": [0, 0, 0, 0],
            "VET_VAL": [0.0, 0.0, 0.0, 0.0],
            "SSI_VAL": [0.0, 0.0, 0.0, 0.0],
            "SPM_CAPHOUSESUB": [0.0, 0.0, 0.0, 0.0],
            "SPM_TENMORTSTATUS": [3, 3, 3, 3],
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": [1, 2],
                "state_fips": [6, 36],
                "H_TENURE": [2, 2],
            }
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": [101, 102],
                "filing_status_input": ["SINGLE", "SINGLE"],
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": [201, 202]}),
        "family": pd.DataFrame({"family_id": [301, 302]}),
        "marital_unit": pd.DataFrame({"marital_unit_id": [401, 402, 403, 404]}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(2, dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def _rent_donor() -> pd.DataFrame:
    rows = np.arange(60, dtype=np.float64)
    donor = pd.DataFrame(
        {
            predictor: rows + position
            for position, predictor in enumerate(
                housing_inputs_module.ACS_RENT_PREDICTORS
            )
        }
    )
    donor["is_household_head"] = 1.0
    donor["tenure_type"] = np.resize(
        np.array(["NONE", "OWNED_WITH_MORTGAGE", "RENTED"]),
        len(donor),
    )
    donor["state_code_str"] = np.resize(
        np.array(["06", "36", "48"]),
        len(donor),
    )
    rented = donor["tenure_type"].eq("RENTED")
    donor["rent"] = np.where(rented, 12_000.0, 0.0)
    donor["rent_is_allocated"] = False
    donor["real_estate_taxes"] = np.where(rented, 0.0, 4_000.0)
    donor["real_estate_taxes_is_allocated"] = False
    donor["household_weight"] = np.linspace(1.0, 2.0, len(donor))
    return donor


class _RowSensitiveRentFitted:
    def predict(self, test: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        return pd.DataFrame(
            {"rent": 1_000.0 + np.arange(len(test), dtype=np.float64)},
            index=test.index,
        )


class _RowSensitiveRentQRF:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def fit(self, *_args: object, **_kwargs: object) -> _RowSensitiveRentFitted:
        return _RowSensitiveRentFitted()


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


def _operator_mapping_structure(
    entrypoint: Callable[..., object],
) -> tuple[tuple[str, ...], set[str]]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(entrypoint)))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    mappings: list[ast.Dict] = []
    for node in ast.walk(function):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "operators"
            and isinstance(node.value, ast.Dict)
        ):
            mappings.append(node.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_source_operator_chain"
        ):
            mappings.extend(
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "operators" and isinstance(keyword.value, ast.Dict)
            )
    assert len(mappings) == 1

    def call_name(call: ast.Call) -> str:
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        raise AssertionError(f"Unclassifiable pool call: {ast.dump(call.func)}")

    operator_names: list[str] = []
    mapped_calls: set[int] = set()
    mapping = mappings[0]
    for key, value in zip(mapping.keys, mapping.values, strict=True):
        assert isinstance(key, ast.Constant) and isinstance(key.value, str)
        operator_name = key.value
        operator_names.append(operator_name)
        calls = [node for node in ast.walk(value) if isinstance(node, ast.Call)]
        if isinstance(value, ast.Name):
            assert value.id == operator_name
            assert not calls
        else:
            assert isinstance(value, ast.Lambda)
            assert [call_name(call) for call in calls] == [operator_name]
        mapped_calls.update(id(call) for call in calls)

    orchestration_calls = {
        call_name(node)
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and id(node) not in mapped_calls
    }
    return tuple(operator_names), orchestration_calls


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
        "max_categorical_total_variation_distance": 0.25,
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


def test_pool_transfer_plan_extends_legacy_except_receipted_asset_deferrals() -> None:
    legacy = declared_acs_transfer_target_families()
    pool = pool_transfer_target_families()
    deferred = frozenset(POOL_DEFERRED_TRANSFER_INPUTS)

    assert deferred == {
        "bank_account_assets",
        "bond_assets",
        "stock_assets",
    }

    for entity, families in legacy.items():
        for family, columns in families.items():
            assert pool[entity][family] == tuple(
                column for column in columns if column not in deferred
            )

    owners: dict[str, tuple[str, str]] = {}
    for entity, families in pool.items():
        for family, columns in families.items():
            for column in columns:
                assert column not in owners, (
                    f"{column} is duplicated by {owners[column]} and {(entity, family)}"
                )
                owners[column] = (entity, family)

    assert owners["takes_up_medicare_if_eligible"] == (
        "person",
        "source_operator_medicare_take_up",
    )
    assert owners["receives_housing_assistance"] == (
        "spm_unit",
        "source_operator_housing_inputs",
    )
    assert owners["immigration_status_str"] == (
        "person",
        "source_operator_immigration",
    )
    assert owners["hours_worked_last_week"] == (
        "person",
        "model_required_numeric",
    )
    assert owners["weekly_hours_worked_before_lsr"] == (
        "person",
        "source_operator_hours_worked",
    )
    assert owners["weeks_worked"] == (
        "person",
        "source_operator_hours_worked",
    )


class _ProducerDtypeFittedQRF:
    weight_kind = "design"

    def __init__(
        self,
        outcomes: tuple[str, ...],
        *,
        calls: Counter[str],
        owner: str,
    ) -> None:
        self.outcomes = outcomes
        self.calls = calls
        self.owner = owner

    def predict(self, test: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        self.calls[f"{self.owner}.predict"] += 1
        rows = len(test)
        return pd.DataFrame(
            {
                outcome: 1.0 + np.arange(rows, dtype=np.float64)
                for outcome in self.outcomes
            },
            index=test.index,
        )


class _ProducerDtypeQRF:
    """Tiny deterministic model beneath the real source-producer wrappers."""

    def __init__(self, *, calls: Counter[str], owner: str) -> None:
        self.calls = calls
        self.owner = owner

    def fit(
        self,
        _frame: object,
        *args: object,
        **_kwargs: object,
    ) -> _ProducerDtypeFittedQRF:
        self.calls[f"{self.owner}.fit"] += 1
        selected = _kwargs.get("targets")
        if selected is None and len(args) >= 2:
            selected = args[1]
        if selected is None:
            raise AssertionError(
                f"Could not resolve QRF outputs from args={args!r}, kwargs={_kwargs!r}."
            )
        return _ProducerDtypeFittedQRF(
            tuple(str(value) for value in selected),
            calls=self.calls,
            owner=self.owner,
        )


def _producer_dtype_qrf_factory(
    calls: Counter[str],
    *,
    owner: str,
) -> Callable[..., _ProducerDtypeQRF]:
    def build(**_kwargs: object) -> _ProducerDtypeQRF:
        return _ProducerDtypeQRF(calls=calls, owner=owner)

    return build


_PRODUCER_DTYPE_QRF_MODULES = (
    "prior_year_income",
    "housing_inputs",
    "child_support",
    "disability_benefits",
    "workers_compensation",
    "weeks_unemployed",
    "childcare",
    "energy_subsidy",
    "retirement_contributions",
    "retirement_distributions",
)


def _producer_dtype_source_frame() -> Frame:
    """Small ASEC fixture carrying every real pool producer's raw inputs."""

    frame = _real_pre_clone_source_frame()
    person = frame.table("person").copy()
    person["tax_unit_role_input"] = ["HEAD", "DEPENDENT", "HEAD", "DEPENDENT"]
    person["source_household_id"] = [10, 10, 20, 20]
    person["source_person_id"] = [1, 2, 1, 2]
    person["MCARE"] = [1, 2, 1, 2]
    person["CSP_VAL"] = [0.0, 100.0, 0.0, 200.0]
    person["CHSP_VAL"] = [50.0, 0.0, 75.0, 0.0]
    person["DIS_VAL1"] = [3_600.0, 0.0, 0.0, 0.0]
    person["DIS_SC1"] = [2, 0, 0, 0]
    person["DIS_VAL2"] = 0.0
    person["DIS_SC2"] = 0
    person["WC_VAL"] = [1_200.0, 0.0, 0.0, 0.0]
    person["LKWEEKS"] = [0, 2, 4, 6]
    person["SPM_CHILDCAREXPNS"] = [1_000.0, 1_000.0, 0.0, 0.0]
    person["SPM_ENGVAL"] = [600.0, 600.0, 0.0, 0.0]
    person["RETCB_VAL"] = [2_000.0, 0.0, 1_000.0, 0.0]
    for suffix in ("1", "2", "1_YNG", "2_YNG"):
        person[f"DST_SC{suffix}"] = 0
        person[f"DST_VAL{suffix}"] = 0.0
    person["DST_SC1"] = [1, 2, 3, 0]
    person["DST_VAL1"] = [100.0, 200.0, 300.0, 0.0]
    person["PRCITSHP"] = [1, 5, 1, 5]
    person["PEINUSYR"] = [0, 24, 0, 24]
    person["PENATVTY"] = [57, 303, 57, 303]
    person["A_SPOUSE"] = 0
    person["CAID"] = 2
    person["IHSFLG"] = 2
    person["CHAMPVA"] = 2
    person["MIL"] = 2
    person["PEN_SC1"] = 0
    person["PEN_SC2"] = 0
    person["RESNSS1"] = 0
    person["RESNSS2"] = 0
    person["SS_YN"] = 2
    person["SSI_YN"] = 2
    person["PEIO1COW"] = 0
    person["A_MJOCC"] = 0
    person["PEAFEVER"] = 2
    person["ED_VAL"] = [0.0, 500.0, 0.0, 1_000.0]
    person["qualified_tuition_expenses"] = [0.0, 1_000.0, 0.0, 2_000.0]
    return _replace_person(frame, person)


def _primary_puf_dtype_donor() -> pd.DataFrame:
    """Minimal donor accepted by the actual primary-PUF producer path."""

    columns = {
        *puf_support_module.PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
        *puf_support_module.PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
        *puf_support_module.PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    }
    donor = pd.DataFrame(
        {column: np.arange(1.0, 5.0, dtype=np.float64) for column in sorted(columns)}
    )
    donor["weight"] = 1.0
    return donor


def _run_pool_transfer_dtype_producers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    calls: Counter[str],
) -> Frame:
    """Execute the small production producer path used by the dtype guard."""

    for module_name in _PRODUCER_DTYPE_QRF_MODULES:
        module = importlib.import_module(f"populace.build.us_runtime.{module_name}")
        monkeypatch.setattr(
            module,
            "QRF",
            _producer_dtype_qrf_factory(calls, owner=module_name),
        )
    monkeypatch.setattr(
        puf_support_module,
        "QRF",
        _producer_dtype_qrf_factory(calls, owner="primary_puf_qrf"),
    )

    for operator_name in POOL_SOURCE_OPERATOR_ORDER:
        producer = getattr(multispine_pool_module, operator_name)

        def observe(
            *args: object,
            _name: str = operator_name,
            _producer: Callable[..., object] = producer,
            **kwargs: object,
        ) -> object:
            calls[_name] += 1
            return _producer(*args, **kwargs)

        monkeypatch.setattr(multispine_pool_module, operator_name, observe)

    assembled = assemble_spines(
        {
            "asec": _producer_dtype_source_frame(),
            "acs": _source_frame(offset=100.0),
        },
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    prepared = prepare_multispine_source_inputs_for_clone(
        assembled,
        acs_rent_donor=_rent_donor(),
    )
    cloned = clone_us_frame_for_puf_support(prepared.frame)
    primary = puf_support_module.impute_us_puf_tax_detail_support(
        cloned,
        _primary_puf_dtype_donor(),
        n_estimators=1,
        seed=0,
        tail_bound_diagnostics=[],
    )
    return multispine_pool_module.complete_multispine_source_inputs(primary).frame


def _assert_pool_transfer_produced_encodings(frame: Frame) -> set[tuple[str, str]]:
    plan = pool_transfer_target_families()
    donor, role = acs_transfer_module.resolve_acs_donor_channel(
        frame,
        acs_transfer_module.ACS_DONOR_CHANNEL_AUTO,
    )
    assert role == "puf_tax_detail"
    audited: set[tuple[str, str]] = set()
    for entity, families in plan.items():
        table = donor.table(entity)
        for targets in families.values():
            acs_transfer_module._validate_donor_targets(
                donor,
                entity=entity,
                targets=targets,
            )
            complete = acs_transfer_module._complete_target_mask(
                table,
                targets=targets,
            )
            assert complete.any(), (entity, targets)
            encodings = acs_transfer_module._complete_case_target_encodings(
                table,
                targets=targets,
                complete=complete,
            )
            assert set(encodings) == set(targets)
            audited.update((entity, target) for target in targets)
    return audited


def test_every_pool_transfer_family_accepts_its_produced_physical_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: Counter[str] = Counter()
    produced = _run_pool_transfer_dtype_producers(monkeypatch, calls=calls)

    audited = _assert_pool_transfer_produced_encodings(produced)

    assert len(audited) == 117
    assert len(POOL_DEFERRED_TRANSFER_INPUTS) == 3
    assert len(audited) + len(POOL_DEFERRED_TRANSFER_INPUTS) == 120
    assert set(POOL_SOURCE_OPERATOR_ORDER) <= set(calls)
    assert all(calls[name] > 0 for name in POOL_SOURCE_OPERATOR_ORDER)
    assert calls["with_us_prior_year_income_inputs"] == 2
    assert calls["primary_puf_qrf.fit"] > 0
    assert calls["primary_puf_qrf.predict"] > 0
    assert sum(calls[name] for name in POOL_SOURCE_OPERATOR_ORDER) == 22


def test_pool_transfer_dtype_guard_observes_hours_producer_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: Counter[str] = Counter()
    producer = multispine_pool_module.with_us_hours_worked_inputs

    def emit_object_strings(*args: object, **kwargs: object) -> Frame:
        calls["mutated_hours_producer"] += 1
        outcome = producer(*args, **kwargs)
        person = outcome.table("person").copy()
        person["hours_worked_last_week"] = person["hours_worked_last_week"].map(str)
        return _replace_person(outcome, person)

    monkeypatch.setattr(
        multispine_pool_module,
        "with_us_hours_worked_inputs",
        emit_object_strings,
    )
    produced = _run_pool_transfer_dtype_producers(monkeypatch, calls=calls)

    with pytest.raises(TypeError, match="hours_worked_last_week"):
        _assert_pool_transfer_produced_encodings(produced)

    assert calls["with_us_hours_worked_inputs"] > 0
    assert calls["mutated_hours_producer"] > 0
    assert all(calls[name] > 0 for name in POOL_SOURCE_OPERATOR_ORDER)
    assert calls["primary_puf_qrf.fit"] > 0
    assert calls["primary_puf_qrf.predict"] > 0


def test_every_pool_transfer_target_has_a_registered_pool_producer() -> None:
    producer_families = {
        "primary_puf_qrf",
        *(
            POOL_OPERATOR_CONTRACTS[operator_name].family
            for operator_name in POOL_SOURCE_OPERATOR_ORDER
        ),
    }
    produced = {
        (entity, column)
        for family in producer_families
        for entity, columns in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[family].items()
        for column in columns
    }
    transferred = {
        (entity, column)
        for entity, families in pool_transfer_target_families().items()
        for columns in families.values()
        for column in columns
    }

    assert not transferred - produced


def test_pool_agreement_registry_exactly_covers_expanded_pool_charter() -> None:
    target_families = pool_transfer_target_families()

    assert POOL_SPINE_AGREEMENT_REGISTRY == default_spine_agreement_registry(
        target_families
    )
    assert (
        validate_spine_agreement_registry(
            POOL_SPINE_AGREEMENT_REGISTRY,
            target_families=target_families,
        )
        == POOL_SPINE_AGREEMENT_REGISTRY
    )
    registered = {
        (spec.entity, column)
        for spec in POOL_SPINE_AGREEMENT_REGISTRY
        for column in spec.columns
    }
    transferred = {
        (entity, column)
        for entity, families in target_families.items()
        for columns in families.values()
        for column in columns
    }
    take_up = {
        (program.entity, program.variable)
        for program in load_take_up_contract().programs
    }
    assert transferred | take_up | {("person", "ssi")} <= registered

    immigration_spec = next(
        spec
        for spec in POOL_SPINE_AGREEMENT_REGISTRY
        if (spec.entity, spec.family) == ("person", "source_operator_immigration")
    )
    assert immigration_spec.columns == (
        "immigration_status_str",
        "ssn_card_type",
    )
    assert immigration_spec.joint_categorical_groups == (
        ("ssn_card_type", "immigration_status_str"),
    )


def test_production_path_passes_fixed_pool_registry_to_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    def no_op(frame: Frame) -> PoolStageOutput:
        return PoolStageOutput(frame)

    def gate(
        _frame: Frame,
        *,
        registry: object,
    ) -> GateResult:
        captured.append(registry)
        return GateResult("fixture", True)

    monkeypatch.setattr(multispine_pool_module, "spine_agreement_gate", gate)

    result = run_multispine_pool_path(
        _source_frame(),
        _source_frame(),
        impute=no_op,
        derive=no_op,
        seed=no_op,
        simulate=no_op,
    )

    assert result.simulation_ready
    assert captured == [POOL_SPINE_AGREEMENT_REGISTRY]


def test_pool_source_operator_order_is_the_full_legacy_chain() -> None:
    assert POOL_SOURCE_OPERATOR_ORDER == _EXPECTED_POOL_SOURCE_OPERATOR_ORDER


def test_every_source_operator_has_an_executable_clone_phase_contract() -> None:
    assert POOL_SOURCE_OPERATOR_CONTRACTS is POOL_OPERATOR_CONTRACTS
    assert tuple(POOL_OPERATOR_CONTRACTS) == (
        *POOL_SOURCE_OPERATOR_ORDER,
        *POOL_DERIVE_OPERATOR_ORDER,
    )
    assert POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER == (
        _EXPECTED_PRE_CLONE_SOURCE_OPERATOR_ORDER
    )
    assert POOL_POST_CLONE_SOURCE_OPERATOR_ORDER == (
        _EXPECTED_POST_CLONE_SOURCE_OPERATOR_ORDER
    )
    assert {
        name
        for name, contract in POOL_OPERATOR_CONTRACTS.items()
        if len(contract.phases) == 2
    } == {"with_us_prior_year_income_inputs"}
    assert all(contract.mechanism for contract in POOL_OPERATOR_CONTRACTS.values())
    assert {
        name
        for name, contract in POOL_OPERATOR_CONTRACTS.items()
        if contract.execution_scope == "whole_pool"
    } == set(POOL_DERIVE_OPERATOR_ORDER)


def test_production_operator_invocations_are_total_and_guarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structural_expectations = (
        (
            multispine_pool_module.prepare_multispine_puf_predictors,
            ("derive_us_cps_carried_inputs",),
            {"_run_source_operator_chain"},
        ),
        (
            multispine_pool_module.prepare_multispine_source_inputs_for_clone,
            POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER,
            {"_run_source_operator_chain"},
        ),
        (
            multispine_pool_module.complete_multispine_source_inputs,
            POOL_POST_CLONE_SOURCE_OPERATOR_ORDER,
            {
                "_assert_formula_owned_source_outputs_absent",
                "_run_source_operator_chain",
                "materialize_pool_deferred_transfer_inputs",
                "PoolStageOutput",
            },
        ),
        (
            multispine_pool_module.derive_multispine_pool_inputs,
            POOL_DERIVE_OPERATOR_ORDER,
            {"PoolStageOutput", "_run_source_operator_chain", "list"},
        ),
    )
    for (
        entrypoint,
        expected_operators,
        expected_orchestration,
    ) in structural_expectations:
        operators, orchestration = _operator_mapping_structure(entrypoint)
        assert operators == expected_operators
        assert orchestration == expected_orchestration

    observed: list[tuple[str, tuple[str, ...]]] = []

    def fail_direct_call(_frame: Frame, **_kwargs: object) -> Frame:
        raise AssertionError("operator kernel bypassed the phase-checked runner")

    for operator_name in POOL_OPERATOR_CONTRACTS:
        monkeypatch.setattr(
            multispine_pool_module,
            operator_name,
            fail_direct_call,
        )

    def observe_guarded_chain(
        frame: Frame,
        *,
        phase: str,
        operator_names: tuple[str, ...],
        operators: dict[str, Callable[[Frame], Frame]],
        **_kwargs: object,
    ) -> PoolStageOutput:
        assert tuple(operators) == operator_names
        observed.append((phase, operator_names))
        return PoolStageOutput(
            frame,
            {
                "phase": phase,
                "operator_order": list(operator_names),
                "suboperators": [
                    {"operator": name, "kernel_receipt": {}} for name in operator_names
                ],
            },
        )

    monkeypatch.setattr(
        multispine_pool_module,
        "_run_source_operator_chain",
        observe_guarded_chain,
    )
    frame = _source_frame()
    multispine_pool_module.prepare_multispine_source_inputs_for_clone(
        frame,
        acs_rent_donor=pd.DataFrame(),
    )
    completed = multispine_pool_module.complete_multispine_source_inputs(frame)
    multispine_pool_module.derive_multispine_pool_inputs(frame)

    assert set(completed.receipt["deferred_transfer_inputs"]["inputs"]) == set(
        POOL_DEFERRED_TRANSFER_INPUTS
    )

    assert observed == [
        ("pre_clone", POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER),
        ("post_clone", POOL_POST_CLONE_SOURCE_OPERATOR_ORDER),
        ("post_clone", POOL_DERIVE_OPERATOR_ORDER),
    ]
    observed_placements = {
        (name, phase) for phase, operator_names in observed for name in operator_names
    }
    registered_placements = {
        (name, phase)
        for name, contract in POOL_OPERATOR_CONTRACTS.items()
        for phase in contract.phases
    }
    assert observed_placements == registered_placements
    assert len({name for name, _phase in observed_placements}) == 23
    assert len(observed_placements) == 24


def test_derive_stage_rejects_preclone_pool_before_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assembled = assemble_spines(
        {"asec": _source_frame(), "acs": _source_frame()},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    calls: list[str] = []

    def unexpected_kernel(frame: Frame) -> Frame:
        calls.append("called")
        return frame

    monkeypatch.setattr(
        multispine_pool_module,
        "_complete_schedule_d_input",
        unexpected_kernel,
    )
    monkeypatch.setattr(
        multispine_pool_module,
        "with_us_qbi_input_reconciliation",
        unexpected_kernel,
    )

    with pytest.raises(ValueError, match="post_clone.*incompatible clone provenance"):
        multispine_pool_module.derive_multispine_pool_inputs(assembled)
    assert not calls


def test_derive_stage_keeps_whole_pool_qbi_reconciliation() -> None:
    assembled = assemble_spines(
        {"asec": _source_frame(), "acs": _source_frame()},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    frame = clone_us_frame_for_puf_support(assembled)
    person = frame.table("person").copy()
    person["long_term_capital_gains_before_response"] = 100.0
    person["non_sch_d_capital_gains"] = 0.0
    for column in multispine_pool_module.US_QBI_OUTPUT_COLUMNS:
        person[column] = 0.0
    person["self_employment_income_before_lsr"] = 10.0
    person["sstb_self_employment_income_before_lsr"] = 5.0
    frame = _replace_person(frame, person)

    result = multispine_pool_module.derive_multispine_pool_inputs(frame)
    derived = result.frame.table("person")

    assert result.receipt["operator_order"] == list(POOL_DERIVE_OPERATOR_ORDER)
    assert derived["schedule_d_capital_gain_distributions"].notna().all()
    assert derived["self_employment_income_before_lsr"].eq(15.0).all()
    assert derived["sstb_self_employment_income_before_lsr"].eq(0.0).all()


def test_prior_year_contract_fails_on_raw_clone_and_succeeds_in_clone_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prior_year_income_module, "QRF", _PriorYearQRF)
    asec = _prior_year_source_frame()
    acs = _source_frame(offset=100.0)
    assembled = assemble_spines(
        {"asec": asec, "acs": acs},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    cloned_raw = clone_us_frame_for_puf_support(assembled)
    prior_operator = {
        "with_us_prior_year_income_inputs": lambda frame: (
            with_us_prior_year_income_inputs(frame, seed=0, time_period=2024)
        )
    }

    with pytest.raises(
        SourceRuntimeError,
        match="must be derived before support cloning",
    ):
        multispine_pool_module._run_source_operator_chain(
            cloned_raw,
            phase="post_clone",
            operator_names=("with_us_prior_year_income_inputs",),
            operators=prior_operator,
        )

    def prepare_clone(frame: Frame) -> PoolStageOutput:
        return multispine_pool_module._run_source_operator_chain(
            frame,
            phase="pre_clone",
            operator_names=("with_us_prior_year_income_inputs",),
            operators=prior_operator,
        )

    def impute(frame: Frame) -> PoolStageOutput:
        return multispine_pool_module._run_source_operator_chain(
            frame,
            phase="post_clone",
            operator_names=("with_us_prior_year_income_inputs",),
            operators=prior_operator,
        )

    def no_op(frame: Frame) -> PoolStageOutput:
        return PoolStageOutput(frame)

    def run() -> MultispinePoolResult:
        return run_multispine_pool_path(
            asec,
            acs,
            prepare_clone=prepare_clone,
            impute=impute,
            derive=no_op,
            seed=no_op,
            simulate=no_op,
            agreement_gate=lambda _frame: GateResult("fixture", True),
        )

    first = run()
    second = run()
    for entity in first.frame.entities:
        pd.testing.assert_frame_equal(
            first.frame.table(entity),
            second.frame.table(entity),
        )

    person = first.frame.table("person")
    cps = person["PERIDNUM"].notna()
    assert "employment_income_last_year" not in person
    assert person.loc[cps, "self_employment_income_last_year"].notna().all()
    assert person.loc[~cps, "self_employment_income_last_year"].isna().all()
    assert (
        person.loc[cps]
        .groupby("person_source_id")["previous_year_income_available"]
        .nunique()
        .eq(1)
        .all()
    )
    assert first.stage_receipts["clone"]["source_preparation"][
        "transient_outputs_carried_through_clone"
    ] == {"person": ["employment_income_last_year"]}
    assert first.stage_receipts["impute"]["suboperators"][0][
        "formula_owned_outputs_removed"
    ] == {"person": ["employment_income_last_year"]}


def test_real_preclone_prefix_runs_before_physical_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(housing_inputs_module, "QRF", _RowSensitiveRentQRF)
    assembled = assemble_spines(
        {
            "asec": _real_pre_clone_source_frame(),
            "acs": _source_frame(offset=100.0),
        },
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )

    prepared = prepare_multispine_source_inputs_for_clone(
        assembled,
        acs_rent_donor=_rent_donor(),
    )

    assert prepared.receipt["operator_order"] == list(
        _EXPECTED_PRE_CLONE_SOURCE_OPERATOR_ORDER
    )
    suboperators = prepared.receipt["suboperators"]
    assert [receipt["operator"] for receipt in suboperators] == list(
        _EXPECTED_PRE_CLONE_SOURCE_OPERATOR_ORDER
    )
    assert [receipt["family"] for receipt in suboperators] == [
        "cps_carried",
        "hours_worked",
        "prior_year_income",
        "relationship_inputs",
        "housing_inputs",
        "eligibility_inputs",
    ]
    assert [receipt["phase"] for receipt in suboperators] == ["pre_clone"] * 6
    assert [receipt["order_index"] for receipt in suboperators] == list(range(6))
    assert prepared.receipt["transient_outputs_carried_through_clone"] == {
        "person": ["employment_income_last_year"]
    }
    hours_gate = suboperators[1]["kernel_receipt"]["hours_worked_signal_gate"]
    assert hours_gate["name"] == "hours_worked_signal"
    assert hours_gate["passed"] is True
    assert hours_gate["failures"] == []

    prepared_person = prepared.frame.table("person")
    prepared_cps = prepared_person["PERIDNUM"].notna()
    assert prepared_person.loc[prepared_cps, "age"].tolist() == [40, 10, 41, 11]
    assert prepared_person.loc[
        prepared_cps, "previous_year_income_available"
    ].tolist() == [False, False, True, True]
    assert prepared_person.loc[prepared_cps, "hours_worked_last_week"].tolist() == [
        42.0,
        0.0,
        30.0,
        0.0,
    ]
    assert prepared_person["hours_worked_last_week"].dtype == np.dtype("float64")
    assert prepared_person.loc[~prepared_cps, "hours_worked_last_week"].isna().all()

    cloned = clone_us_frame_for_puf_support(prepared.frame)
    person = cloned.table("person")
    cps = person["PERIDNUM"].notna()
    heads = cps & person["is_household_head"].eq(True)
    source_id = support_source_id_column("person")
    rent_variants = person.loc[heads].groupby(source_id)["pre_subsidy_rent"].nunique()
    assert rent_variants.eq(1).all()
    assert set(person.loc[heads, "pre_subsidy_rent"]) == {1_000.0, 1_001.0}

    parents = cps & person["A_LINENO"].eq(1)
    assert set(person.loc[parents, support_clone_index_column("person")]) == {0, 1}
    assert person.loc[parents, "own_children_in_household"].eq(1.0).all()


def test_preclone_hours_signal_gate_rejects_implausible_producer_surface() -> None:
    asec = _real_pre_clone_source_frame()
    person = asec.table("person")
    person["weekly_hours_worked_before_lsr"] = [60.0, 65.0, 70.0, 75.0]
    person["hours_worked_last_week"] = [55.0, 60.0, 65.0, 70.0]
    person["weeks_worked"] = [40.0, 44.0, 48.0, 52.0]
    assembled = assemble_spines(
        {"asec": asec, "acs": _source_frame(offset=100.0)},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )

    with pytest.raises(
        ValueError,
        match="Pool pre-clone hours-worked signal gate failed",
    ) as exc_info:
        prepare_multispine_source_inputs_for_clone(
            assembled,
            acs_rent_donor=_rent_donor(),
        )

    assert "worked share 1.000 outside plausibility band" in str(exc_info.value)
    assert "mean weekly hours among workers 67.5 outside" in str(exc_info.value)


def test_row_sensitive_prefix_exposes_clone_first_defects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(housing_inputs_module, "QRF", _RowSensitiveRentQRF)
    donor = _rent_donor()

    old_order = multispine_pool_module.derive_us_cps_carried_inputs(
        _real_pre_clone_source_frame()
    )
    old_order = multispine_pool_module.with_us_prior_year_income_inputs(
        old_order,
        seed=0,
        time_period=2024,
    )
    old_order = multispine_pool_module.with_us_relationship_inputs(
        old_order,
        seed=0,
        time_period=2024,
    )
    old_order = clone_us_frame_for_puf_support(old_order)
    old_order = multispine_pool_module.with_us_housing_inputs(
        old_order,
        seed=0,
        time_period=2024,
        acs_rent_donor=donor,
    )
    old_order = multispine_pool_module.with_us_eligibility_inputs(
        old_order,
        seed=0,
        time_period=2024,
    )

    person = old_order.table("person")
    source_id = support_source_id_column("person")
    heads = person["is_household_head"].eq(True)
    rent_variants = person.loc[heads].groupby(source_id)["pre_subsidy_rent"].nunique()
    assert rent_variants.eq(2).all()
    parents = person["A_LINENO"].eq(1)
    assert person.loc[parents, "own_children_in_household"].eq(2.0).all()


def test_every_source_operator_output_has_a_pool_owner() -> None:
    transferred = {
        column
        for families in pool_transfer_target_families().values()
        for columns in families.values()
        for column in columns
    }
    native = {
        column
        for columns in multispine_pool_module._POOL_NATIVE_COMPLETE_OUTPUTS.values()
        for column in columns
    }
    formula_owned = {
        column
        for columns in multispine_pool_module._FORMULA_OWNED_SOURCE_OUTPUTS.values()
        for column in columns
    }
    seeded = {
        program.variable
        for program in load_take_up_contract().programs
        if program.is_seeded
    }

    unowned: list[str] = []
    for operator_name in POOL_SOURCE_OPERATOR_ORDER:
        family = multispine_pool_module._SOURCE_OPERATOR_FAMILIES[operator_name]
        for columns in PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[family].values():
            for column in columns:
                if column not in transferred | native | formula_owned | seeded:
                    unowned.append(f"{family}.{column}")
    assert not unowned


@pytest.mark.parametrize(
    ("phase", "operator_names", "physically_clone"),
    [
        ("pre_clone", POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER, False),
        ("post_clone", POOL_POST_CLONE_SOURCE_OPERATOR_ORDER, True),
    ],
)
def test_source_operator_chains_are_availability_aware_and_source_blind(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    operator_names: tuple[str, ...],
    physically_clone: bool,
) -> None:
    asec = _source_frame()
    asec_tables = {entity: asec.table(entity).copy() for entity in asec.entities}
    asec_tables["person"]["PERIDNUM"] = ["asec-1", "asec-2"]
    asec = Frame(
        asec_tables,
        asec.schema,
        {"household": asec.weights_for("household")},
        asec.strata,
    )
    acs = _source_frame(offset=100.0)
    acs_tables = {entity: acs.table(entity).copy() for entity in acs.entities}
    first_output = "fixture_source_output_00"
    acs_tables["person"][first_output] = [900.0, 901.0]
    acs = Frame(
        acs_tables,
        acs.schema,
        {"household": acs.weights_for("household")},
        acs.strata,
    )
    assembled = assemble_spines(
        {"asec": asec, "acs": acs},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    boundary_frame = (
        clone_us_frame_for_puf_support(assembled) if physically_clone else assembled
    )

    calls: list[str] = []
    output_families: dict[str, dict[str, frozenset[str]]] = {}
    operators: dict[str, Callable[[Frame], Frame]] = {}
    for index, operator_name in enumerate(operator_names):
        family = multispine_pool_module._SOURCE_OPERATOR_FAMILIES[operator_name]
        output = f"fixture_source_output_{index:02d}"
        output_families[family] = {"person": frozenset({output})}

        def apply(
            available: Frame,
            *,
            name: str = operator_name,
            column: str = output,
            value: float = float(index + 1),
        ) -> Frame:
            calls.append(name)
            assert "us_spine_assembly_manifest" not in available.metadata
            assert not available.mass_log
            person = available.table("person")
            assert ("person_support_channel" in person.columns) is physically_clone
            assert ("person_support_clone_index" in person.columns) is physically_clone
            assert person["PERIDNUM"].notna().all()
            updated = person.copy()
            updated[column] = value
            return _replace_person(available, updated)

        operators[operator_name] = apply

    original_getitem = pd.DataFrame.__getitem__

    def reject_source_channel_read(
        table: pd.DataFrame,
        key: object,
    ) -> object:
        keys = (
            [key]
            if isinstance(key, str)
            else list(key)
            if isinstance(key, (list, tuple))
            else []
        )
        if any(str(column).endswith("_support_channel") for column in keys):
            raise AssertionError("population source channel was read")
        return original_getitem(table, key)

    monkeypatch.setattr(pd.DataFrame, "__getitem__", reject_source_channel_read)
    result = multispine_pool_module._run_source_operator_chain(
        boundary_frame,
        phase=phase,
        operator_names=operator_names,
        operators=operators,
        output_families=output_families,
    )

    pool_rows = 8 if physically_clone else 4
    cps_rows = 4 if physically_clone else 2
    assert calls == list(operator_names)
    assert result.receipt["operator_order"] == list(operator_names)
    assert result.receipt["cps_source_evidence"] == {
        "column": "PERIDNUM",
        "person_rows": cps_rows,
    }
    for index, receipt in enumerate(result.receipt["suboperators"]):
        assert receipt["order_index"] == index
        assert receipt["operator"] == operator_names[index]
        assert receipt["pool_input_rows"]["person"] == pool_rows
        assert receipt["cps_available_rows"]["person"] == cps_rows
        assert receipt["operator_output_rows"]["person"] == cps_rows
        assert receipt["merged_rows"]["person"] == cps_rows
        assert receipt["operator_projection"] == {
            "selection": "PERIDNUM",
            "lineage_state_persisted": False,
            "support_role_metadata_exposed": physically_clone,
        }

    person = result.frame.table("person")
    cps = person["PERIDNUM"].notna()
    assert person.loc[cps, first_output].tolist() == [1.0] * cps_rows
    expected_acs = [900.0, 900.0, 901.0, 901.0] if physically_clone else [900.0, 901.0]
    assert sorted(person.loc[~cps, first_output].tolist()) == expected_acs
    unavailable = "fixture_source_output_01"
    assert person.loc[cps, unavailable].tolist() == [2.0] * cps_rows
    assert person.loc[~cps, unavailable].isna().all()
    assert result.frame.metadata == boundary_frame.metadata
    assert result.frame.mass_log == boundary_frame.mass_log


def test_predictor_prep_fills_cps_rows_without_overwriting_acs_native() -> None:
    asec = _source_frame()
    asec_tables = {entity: asec.table(entity).copy() for entity in asec.entities}
    asec_person = asec_tables["person"].drop(columns=["age"])
    asec_person["PERIDNUM"] = ["asec-1", "asec-2"]
    asec_person["A_AGE"] = [31, 52]
    asec_person["A_SEX"] = [1, 2]
    asec_person["OI_VAL"] = [0.0, 25.0]
    asec_person["OI_OFF"] = [0, 20]
    asec_tables["person"] = asec_person
    asec = Frame(
        asec_tables,
        asec.schema,
        {"household": asec.weights_for("household")},
        asec.strata,
    )

    acs = _source_frame()
    acs_tables = {entity: acs.table(entity).copy() for entity in acs.entities}
    acs_tables["person"]["age"] = [70.0, 80.0]
    acs_tables["person"]["is_female"] = [True, False]
    acs = Frame(
        acs_tables,
        acs.schema,
        {"household": acs.weights_for("household")},
        acs.strata,
    )
    assembled = assemble_spines(
        {"asec": asec, "acs": acs},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )

    result = prepare_multispine_puf_predictors(assembled)

    person = result.frame.table("person")
    cps = person["PERIDNUM"].notna()
    assert sorted(person.loc[cps, "age"].tolist()) == [31.0, 52.0]
    assert sorted(person.loc[~cps, "age"].tolist()) == [70.0, 80.0]
    assert sorted(person.loc[cps, "is_female"].tolist()) == [
        False,
        True,
    ]
    assert sorted(person.loc[~cps, "is_female"].tolist()) == [
        False,
        True,
    ]
    assert result.receipt["operator_order"] == ["derive_us_cps_carried_inputs"]


def test_schedule_d_derivation_preserves_existing_values_and_receipt() -> None:
    assembled = assemble_spines(
        {"asec": _source_frame(), "acs": _source_frame()},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    from populace.build.us_runtime.puf_support import (
        clone_us_frame_for_puf_support,
    )

    frame = clone_us_frame_for_puf_support(assembled)
    person = frame.table("person").copy()
    person["long_term_capital_gains_before_response"] = 100.0
    person["non_sch_d_capital_gains"] = 0.0
    person["schedule_d_capital_gain_distributions"] = np.nan
    person.loc[person.index[0], "schedule_d_capital_gain_distributions"] = 7.0
    frame = _replace_person(frame, person)

    result = _complete_schedule_d_input(frame)
    completed = result.frame
    receipt = result.receipt

    output = completed.table("person")["schedule_d_capital_gain_distributions"]
    assert output.loc[person.index[0]] == 7.0
    assert not output.isna().any()
    assert (output.loc[person.index[1:]] > 0.0).all()
    assert completed.metadata == frame.metadata
    assert receipt["preserved_nonnull_rows"] == 1
    assert receipt["filled_rows"] == len(person) - 1


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
    tables["person"]["takes_up_medicare_if_eligible"] = [True, False]
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


def test_pool_asset_deferrals_are_typed_null_receipted_and_fail_when_stale() -> None:
    frame = _assembled_cloned_with_partial_take_up()

    result = materialize_pool_deferred_transfer_inputs(frame)

    person = result.frame.table("person")
    assert set(result.receipt["inputs"]) == set(POOL_DEFERRED_TRANSFER_INPUTS)
    for column, declaration in POOL_DEFERRED_TRANSFER_INPUTS.items():
        assert person[column].dtype == np.dtype("float64")
        assert person[column].isna().all()
        assert result.receipt["inputs"][column] == {
            **declaration,
            "status": "deferred_pending_source_donor",
            "rows": len(person),
            "null_rows": len(person),
        }

    with pytest.raises(ValueError, match="already exists; retire the stale deferral"):
        materialize_pool_deferred_transfer_inputs(result.frame)


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
    assert medicare["provenance_kind"] == ("transferred_or_preserved_input")
    assert medicare["defaulted_rows"] == 0

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


def test_deferred_asset_defaults_exist_only_on_disposable_simulation_view() -> None:
    frame = materialize_pool_deferred_transfer_inputs(
        seed_multispine_pool_inputs(
            _assembled_cloned_with_partial_take_up(),
            engine=_FakeEngine(),
        ).frame
    ).frame

    class AssetProjectionEngine(_FakeEngine):
        def variables(self) -> list[str]:
            return list(POOL_DEFERRED_TRANSFER_INPUTS)

        def variable_metadata(self, name: str) -> object:
            assert name in POOL_DEFERRED_TRANSFER_INPUTS
            return SimpleNamespace(entity="person")

        def default_values(self, names: list[str]) -> dict[str, object]:
            assert names == list(POOL_DEFERRED_TRANSFER_INPUTS)
            return {name: 0.0 for name in names}

        def materialize(
            self,
            bundle: Frame,
            variables: list[str],
            period: int,
        ) -> dict[str, np.ndarray]:
            person = bundle.table("person")
            assert all(
                person[column].eq(0.0).all() for column in POOL_DEFERRED_TRANSFER_INPUTS
            )
            return super().materialize(bundle, variables, period)

    result = materialize_multispine_agreement_outputs(
        frame,
        engine=AssetProjectionEngine(),
    )

    assert all(
        result.frame.table("person")[column].isna().all()
        for column in POOL_DEFERRED_TRANSFER_INPUTS
    )
    expected_rows = frame.n("person")
    assert result.receipt["simulation_projection_default_fills"] == {
        column: {
            "entity": "person",
            "rows": expected_rows,
            "value": 0.0,
            "persisted_to_pool": False,
        }
        for column in POOL_DEFERRED_TRANSFER_INPUTS
    }
