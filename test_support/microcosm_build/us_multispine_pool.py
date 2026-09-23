# ruff: noqa: F401
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import inspect
import textwrap
from collections import Counter
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.gates import GateResult
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import acs_transfer as acs_transfer_module
from microcosm.build.us_runtime import housing_inputs as housing_inputs_module
from microcosm.build.us_runtime import multispine_pool as multispine_pool_module
from microcosm.build.us_runtime import prior_year_income as prior_year_income_module
from microcosm.build.us_runtime import puf_support as puf_support_module
from microcosm.build.us_runtime import stacked_spine as stacked_spine_module
from microcosm.build.us_runtime.acs_income_universe import (
    ACS_PUMS_EARNINGS_UNIVERSE_PERSON_INPUTS,
)
from microcosm.build.us_runtime.acs_transfer import (
    declared_acs_transfer_target_families,
)
from microcosm.build.us_runtime.multispine_pool import (
    POOL_CHECKPOINT_STAGE_ORDER,
    POOL_DEFERRED_TRANSFER_INPUTS,
    POOL_DERIVE_OPERATOR_ORDER,
    POOL_ENGINE_INPUT_PROJECTION_CONTRACT,
    POOL_OPERATOR_CONTRACTS,
    POOL_OPERATOR_ORDER,
    POOL_POST_CLONE_SOURCE_OPERATOR_ORDER,
    POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER,
    POOL_REMAINING_STAGE_INPUT_MANIFEST_SHA256,
    POOL_SOURCE_OPERATOR_CONTRACTS,
    POOL_SOURCE_OPERATOR_ORDER,
    POOL_SPINE_AGREEMENT_REGISTRY,
    POOL_SSI_DEPENDENCY_CONTRACT,
    MultispinePoolCheckpoint,
    MultispinePoolResult,
    PoolInputSurfaceEntry,
    PoolRemainingStageInput,
    PoolStageOutput,
    _complete_schedule_d_input,
    finalize_multispine_source_inputs,
    materialize_multispine_agreement_outputs,
    materialize_pool_deferred_transfer_inputs,
    pool_engine_input_projection_receipt,
    pool_input_surface,
    pool_post_puf_puf_producer_target_families,
    pool_post_puf_source_producer_target_families,
    pool_post_puf_transfer_target_families,
    pool_pre_clone_gap_fill_target_families,
    pool_remaining_stage_input_manifest,
    pool_remaining_stage_input_manifest_receipt,
    pool_ssi_dependency_closure,
    pool_transfer_target_families,
    prepare_multispine_puf_predictors,
    prepare_multispine_source_inputs_for_clone,
    run_multispine_pool_path,
    run_multispine_post_clone_source_operator,
    seed_multispine_pool_inputs,
)
from microcosm.build.us_runtime.operator_boundary import (
    FORMULA_OWNED_SOURCE_COLUMNS,
    PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
)
from microcosm.build.us_runtime.prior_year_income import (
    with_us_prior_year_income_inputs,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
    clone_us_frame_for_puf_support,
)
from microcosm.build.us_runtime.qbi_inputs import (
    US_QBI_OUTPUT_COLUMNS,
    bind_us_qbi_reconciliation_transition_authority,
    us_qbi_reconciliation_change_receipt,
    with_us_qbi_input_reconciliation,
)
from microcosm.build.us_runtime.spine_agreement import (
    SpineAgreementSpec,
    default_spine_agreement_registry,
    spine_agreement_gate,
    validate_spine_agreement_registry,
)
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.build.us_runtime.support_provenance import (
    support_clone_index_column,
    support_source_id_column,
)
from microcosm.build.us_runtime.take_up_contract import load_take_up_contract
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSEngine,
    PolicyEngineUSVariableMetadataIndex,
)

_FIXTURE_SEED_PERSON_COLUMN = "takes_up_medicaid_if_eligible"

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


def _installed_variable_metadata_index() -> PolicyEngineUSVariableMetadataIndex:
    try:
        return PolicyEngineUSVariableMetadataIndex()
    except ImportError:
        pytest.skip("requires the policyengine-us [us] extra")


def _overlap_person_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3, 4, 5], dtype=np.int64),
            "person_source_id": np.asarray([10, 10, 10, 20, 20], dtype=np.int64),
            "person_support_channel": ["asec"] * 5,
            "person_support_clone_index": np.asarray([0, 1, 2, 0, 1], dtype=np.int64),
            "qualified_tuition_expenses": np.asarray(
                [10.0, 20.0, 20.0, 30.0, 40.0], dtype=np.float64
            ),
            "traditional_ira_contributions_desired": np.asarray(
                [1.0, 101.25, 999.0, 2.0, 202.5], dtype=np.float64
            ),
            "self_employed_pension_contributions_desired": np.asarray(
                [3.0, -0.0, 777.0, 4.0, 404.5], dtype=np.float64
            ),
        }
    )


_EXPECTED_SOURCE_OPERATOR_WRAPPERS = {
    "with_us_hours_worked_inputs": "_with_gated_us_hours_worked_inputs",
    "with_us_qbi_input_reconciliation": "reconcile_qbi_with_receipt",
}


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
            "PEMCPREM": [100.0, 0.0, 25.0, 0.0],
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
            "PAW_VAL": [0.0, 125.0, 0.0, 0.0],
            "PAW_TYP": [0, 3, 0, 0],
            "SPM_SNAPSUB": [0.0, 0.0, 900.0, 900.0],
            "WICYN": [0, 1, 2, 0],
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


def _fixture_pool_operators(
    order: list[str],
) -> dict[str, Callable[[Frame], PoolStageOutput]]:
    def derive(frame: Frame) -> PoolStageOutput:
        order.append("derive")
        person = frame.table("person").copy()
        person["derived"] = person["transferred"] * 2
        person["SEMP"] = 0.0
        person["self_employment_income_before_lsr"] = 0.0
        for column in US_QBI_OUTPUT_COLUMNS:
            person[column] = 0.0
        before = _replace_person(frame, person)
        after = with_us_qbi_input_reconciliation(before)
        receipt = us_qbi_reconciliation_change_receipt(before, after)
        after = bind_us_qbi_reconciliation_transition_authority(after, receipt)
        return PoolStageOutput(
            after,
            {
                "operator": "derive",
                "qbi_input_reconciliation": receipt,
            },
            qbi_transition_authority_sha256=receipt["sha256"],
        )

    def seed(frame: Frame) -> PoolStageOutput:
        order.append("seed")
        person = frame.table("person").copy()
        person[_FIXTURE_SEED_PERSON_COLUMN] = person["age"] >= 40
        return PoolStageOutput(
            _replace_person(frame, person),
            {
                "operator": "seed",
                "programs": {
                    _FIXTURE_SEED_PERSON_COLUMN: {"entity": "person"},
                },
            },
        )

    return {
        "impute": _operator(
            "impute",
            order,
            lambda person: person.__setitem__("transferred", person["age"]),
        ),
        "derive": derive,
        "seed": seed,
        "simulate": _operator(
            "simulate",
            order,
            lambda person: person.__setitem__("ssi", person["derived"]),
        ),
    }


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
            expected_kernel = _EXPECTED_SOURCE_OPERATOR_WRAPPERS.get(
                operator_name,
                operator_name,
            )
            assert value.id == expected_kernel
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


class _ProducerDtypeFittedQRF:
    def __init__(
        self,
        outcomes: tuple[str, ...],
        *,
        calls: Counter[str],
        owner: str,
        observations: list[dict[str, object]],
        weight_kind: str,
    ) -> None:
        self.outcomes = outcomes
        self.calls = calls
        self.owner = owner
        self.observations = observations
        self.weight_kind = weight_kind

    def predict(self, test: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        self.calls[f"{self.owner}.predict"] += 1
        rows = len(test)
        predictions = pd.DataFrame(
            {
                outcome: 1.0 + np.arange(rows, dtype=np.float64)
                for outcome in self.outcomes
            },
            index=test.index,
        )
        self.observations.append(
            {
                "owner": self.owner,
                "phase": "predict",
                "features": test.copy(),
                "outputs": predictions.copy(),
            }
        )
        return predictions


class _ProducerDtypeQRF:
    """Tiny deterministic model beneath the real source-producer wrappers."""

    def __init__(
        self,
        *,
        calls: Counter[str],
        owner: str,
        observations: list[dict[str, object]],
    ) -> None:
        self.calls = calls
        self.owner = owner
        self.observations = observations

    def fit(
        self,
        frame: Frame | pd.DataFrame,
        *args: object,
        **_kwargs: object,
    ) -> _ProducerDtypeFittedQRF:
        self.calls[f"{self.owner}.fit"] += 1
        predictors = _kwargs.get("predictors")
        if predictors is None and args:
            predictors = args[0]
        selected = _kwargs.get("targets")
        if selected is None and len(args) >= 2:
            selected = args[1]
        if predictors is None or selected is None:
            raise AssertionError(
                "Could not resolve QRF predictors/outputs from "
                f"args={args!r}, kwargs={_kwargs!r}."
            )
        predictor_names = tuple(str(value) for value in predictors)
        outcomes = tuple(str(value) for value in selected)
        if isinstance(frame, Frame):
            entity = frame.column_entity(outcomes[0])
            table = frame.table(entity)
            weight_kind = frame.resolve_weights(entity).kind.value
        else:
            table = frame
            weight_kind = "design"
        self.observations.append(
            {
                "owner": self.owner,
                "phase": "fit",
                "predictors": predictor_names,
                "features": table.loc[:, list(predictor_names)].copy(),
            }
        )
        return _ProducerDtypeFittedQRF(
            outcomes,
            calls=self.calls,
            owner=self.owner,
            observations=self.observations,
            weight_kind=weight_kind,
        )


def _producer_dtype_qrf_factory(
    calls: Counter[str],
    *,
    owner: str,
    observations: list[dict[str, object]],
) -> Callable[..., _ProducerDtypeQRF]:
    def build(**_kwargs: object) -> _ProducerDtypeQRF:
        return _ProducerDtypeQRF(
            calls=calls,
            owner=owner,
            observations=observations,
        )

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


def _producer_dtype_acs_source_frame() -> Frame:
    """Small ACS fixture with the native half of cross-spine predictors."""

    frame = _source_frame(offset=100.0)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]
    person["is_female"] = [True, False]
    person["is_household_head"] = [True, True]
    person["employment_income_before_lsr"] = [30_000.0, 45_000.0]
    person["self_employment_income_before_lsr"] = [0.0, 5_000.0]
    person["acs_social_security_income"] = [0.0, 12_000.0]
    person["acs_retirement_income"] = [0.0, 8_000.0]
    person["acs_interest_dividend_rental_income"] = [100.0, 2_000.0]
    household = tables["household"]
    household["state_fips"] = [6, 36]
    household["tenure_type"] = ["RENTED", "OWNED_WITH_MORTGAGE"]
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


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
    donor["puf_predictor_tax_unit_person_count"] = np.arange(
        1,
        5,
        dtype=np.int64,
    )
    donor["weight"] = 1.0
    return donor


def _run_pool_transfer_dtype_producers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    calls: Counter[str],
    observations: list[dict[str, object]],
    stages: dict[str, Frame] | None = None,
) -> Frame:
    """Execute the small production producer path used by the dtype guard."""

    for module_name in _PRODUCER_DTYPE_QRF_MODULES:
        module = importlib.import_module(f"microcosm.build.us_runtime.{module_name}")
        monkeypatch.setattr(
            module,
            "QRF",
            _producer_dtype_qrf_factory(
                calls,
                owner=module_name,
                observations=observations,
            ),
        )
    monkeypatch.setattr(
        puf_support_module,
        "QRF",
        _producer_dtype_qrf_factory(
            calls,
            owner="primary_puf_qrf",
            observations=observations,
        ),
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
            "acs": _producer_dtype_acs_source_frame(),
        },
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    if stages is not None:
        stages["assembled"] = assembled
    prepared = prepare_multispine_source_inputs_for_clone(
        assembled,
        acs_rent_donor=_rent_donor(),
    )
    if stages is not None:
        stages["prepared"] = prepared.frame
    cloned = clone_us_frame_for_puf_support(prepared.frame)
    primary_donor = _primary_puf_dtype_donor()
    primary_chain_inputs = puf_support_module.prepare_us_puf_tax_detail_chain_inputs(
        cloned,
        primary_donor,
    )
    observations.append(
        {
            "owner": "primary_puf_chain",
            "phase": "prepared",
            "inputs": primary_chain_inputs,
        }
    )
    primary = puf_support_module.impute_us_puf_tax_detail_support(
        cloned,
        primary_donor,
        n_estimators=1,
        seed=0,
        tail_bound_diagnostics=[],
    )
    produced = multispine_pool_module.complete_multispine_source_inputs(primary).frame
    if stages is not None:
        stages["produced"] = produced
    return produced


def _assert_pool_transfer_produced_encodings(
    frame: Frame,
    *,
    observations: list[dict[str, object]],
) -> tuple[
    set[tuple[str, str]],
    set[tuple[str, str]],
    tuple[tuple[str, tuple[str, ...]], ...],
]:
    plan = pool_transfer_target_families()
    targets = {
        target
        for families in plan.values()
        for columns in families.values()
        for target in columns
    }
    acs_transfer_module.assert_acs_transfer_targets_are_input_leaves(targets)
    donor, role = acs_transfer_module.resolve_acs_donor_channel(
        frame,
        acs_transfer_module.ACS_DONOR_CHANNEL_AUTO,
    )
    assert role == "puf_tax_detail"
    audited_targets: set[tuple[str, str]] = set()
    audited_predictors: set[tuple[str, str]] = set()
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
            audited_targets.update((entity, target) for target in targets)

            surface = acs_transfer_module._transfer_feature_surface(
                donor,
                frame,
                entity=entity,
                targets=targets,
            )
            predictors = (*surface.required, *surface.optional)
            for feature_frame in (surface.donor, surface.recipient):
                encoded = acs_transfer_module._encoded_predictor_frame(
                    feature_frame,
                    predictors=predictors,
                )
                assert all(dtype == np.dtype("float64") for dtype in encoded.dtypes)
                acs_transfer_module._complete_predictor_mask(
                    feature_frame,
                    predictors=predictors,
                )
            audited_predictors.update((entity, name) for name in predictors)

    prepared_primary = [
        observation
        for observation in observations
        if observation["owner"] == "primary_puf_chain"
        and observation["phase"] == "prepared"
    ]
    assert len(prepared_primary) == 1
    chain_inputs = prepared_primary[0]["inputs"]
    assert isinstance(chain_inputs, puf_support_module.PufTaxDetailChainInputs)
    primary_predictors = tuple(chain_inputs.predictors)
    primary_targets = tuple(chain_inputs.target_order)
    assert len(primary_predictors) == 8
    assert len(primary_targets) == 65

    primary_qrf_observations = [
        observation
        for observation in observations
        if observation["owner"] == "primary_puf_qrf"
    ]
    assert {observation["phase"] for observation in primary_qrf_observations} == {
        "fit",
        "predict",
    }
    primary_prediction = next(
        observation
        for observation in primary_qrf_observations
        if observation["phase"] == "predict"
    )
    raw_draws = primary_prediction["outputs"]
    assert isinstance(raw_draws, pd.DataFrame)
    assert tuple(raw_draws.columns) == primary_targets
    assert raw_draws.index.equals(chain_inputs.recipient_features.index)
    assert all(dtype == np.dtype("float64") for dtype in raw_draws.dtypes)
    assert np.isfinite(raw_draws.to_numpy()).all()

    donor_base = chain_inputs.donor.loc[:, list(primary_predictors)]
    recipient_base = chain_inputs.recipient_features.loc[:, list(primary_predictors)]
    assert donor_base["puf_predictor_tax_unit_person_count"].dtype == np.dtype("int64")
    assert all(
        donor_base[column].dtype == np.dtype("float64")
        for column in primary_predictors
        if column != "puf_predictor_tax_unit_person_count"
    )
    assert all(dtype == np.dtype("float64") for dtype in recipient_base.dtypes)

    primary_predictor_sets: list[tuple[str, tuple[str, ...]]] = []
    for position, target in enumerate(primary_targets):
        predictors = (*primary_predictors, *primary_targets[:position])
        donor_features = chain_inputs.donor.loc[:, list(predictors)]
        recipient_features = recipient_base.copy()
        for prior in primary_targets[:position]:
            recipient_features[prior] = raw_draws[prior].to_numpy(
                dtype=np.float64,
                copy=False,
            )
        for features in (donor_features, recipient_features):
            assert all(
                pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes
            )
            qrf_matrix = features.to_numpy(dtype=np.float64)
            assert qrf_matrix.dtype == np.dtype("float64")
            assert np.isfinite(qrf_matrix).all()
        assert all(
            donor_features[prior].dtype == np.dtype("float64")
            for prior in primary_targets[:position]
        )
        primary_predictor_sets.append((target, predictors))
    return audited_targets, audited_predictors, tuple(primary_predictor_sets)


def _single_post_clone_source_receipt(operator: str) -> dict[str, object]:
    return {
        "phase": "post_clone",
        "operator_order": [operator],
        "cps_source_evidence": {"column": "PERIDNUM", "person_rows": 4},
        "transient_outputs_carried_through_clone": {},
        "suboperators": [
            {
                "operator": operator,
                "order_index": 0,
                "phase": "post_clone",
            }
        ],
    }


def _qbi_ready_derive_frame() -> Frame:
    assembled = assemble_spines(
        {"asec": _source_frame(), "acs": _source_frame()},
        household_mass_shares={"asec": 0.5, "acs": 0.5},
    )
    frame = clone_us_frame_for_puf_support(assembled)
    person = frame.table("person").copy()
    person["long_term_capital_gains_before_response"] = 100.0
    person["non_sch_d_capital_gains"] = 0.0
    for column in US_QBI_OUTPUT_COLUMNS:
        person[column] = 0.0
    person["self_employment_income_before_lsr"] = 10.0
    person["SEMP"] = 10.0
    person["sstb_self_employment_income_before_lsr"] = 5.0
    return _replace_person(frame, person)


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
    tables["person"]["takes_up_wic_if_eligible"] = [True, False]
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
    tables["person"]["takes_up_wic_if_eligible"] = [False, True]
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
    from microcosm.build.us_runtime.puf_support import (
        clone_us_frame_for_puf_support,
    )

    return clone_us_frame_for_puf_support(assembled)


__all__ = [name for name in globals() if not name.startswith("__")]
