"""Historical QBI stream isolation for the passive sibling stage."""

from __future__ import annotations

import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.qbi_inputs import US_QBI_OUTPUT_COLUMNS
from microcosm.build.us_runtime.qbi_passive_passthrough import (
    US_QBI_PASSIVE_PASSTHROUGH_OUTPUT_COLUMN,
    with_us_qbi_passive_passthrough_assignment,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_HISTORY_COMMIT = "cfbf6330493e9b2c1bf2b1e79fd4e0b0b181b184"
_HISTORY_FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures/qbi_history"
_HISTORY_FIXTURE_SHA256 = {
    "qbi_simulation.py.fixture": (
        "4bbc2fd6f2271b22d32120c7662647c759befe8bb5456995190053e68bcf1316"
    ),
    "qbi_inputs.py.fixture": (
        "fae18a126f4b84ff8e81ecf533a47795f5e3a1b7497e487045eb1ea430189c94"
    ),
    "qbi_assumptions_v1.json.fixture": (
        "2df48450ca4bb28f7ddad5f3b517b2052d1f75e988c8685e9d42fadc6e59d7c0"
    ),
    "qbi_assumptions_v2.json.fixture": (
        "2fabf4f4a0547418663546bc71106880f3dba0a7a23be81681659790b603ca1f"
    ),
    "qbi_assumptions_v3.json.fixture": (
        "22a190ab57015415c054aca77fd150af6fc4b6d535b846179b9f099a0a7e9bf2"
    ),
}

# The historical runtime explicitly returns these 15 leaves for every exposed
# version. Repeating the literals per version makes additions or contractions
# to any one version a deliberate test edit rather than an inherited constant.
_COMPLETE_QBI_OUTPUTS_BY_VERSION = {
    1: frozenset(
        {
            "estate_income_would_be_qualified",
            "farm_operations_income_would_be_qualified",
            "farm_rent_income_would_be_qualified",
            "partnership_s_corp_income_would_be_qualified",
            "rental_income_would_be_qualified",
            "self_employment_income_would_be_qualified",
            "sstb_self_employment_income_would_be_qualified",
            "business_is_sstb",
            "qualified_bdc_income",
            "qualified_reit_and_ptp_income",
            "sstb_self_employment_income_before_lsr",
            "sstb_unadjusted_basis_qualified_property",
            "sstb_w2_wages_from_qualified_business",
            "unadjusted_basis_qualified_property",
            "w2_wages_from_qualified_business",
        }
    ),
    2: frozenset(
        {
            "estate_income_would_be_qualified",
            "farm_operations_income_would_be_qualified",
            "farm_rent_income_would_be_qualified",
            "partnership_s_corp_income_would_be_qualified",
            "rental_income_would_be_qualified",
            "self_employment_income_would_be_qualified",
            "sstb_self_employment_income_would_be_qualified",
            "business_is_sstb",
            "qualified_bdc_income",
            "qualified_reit_and_ptp_income",
            "sstb_self_employment_income_before_lsr",
            "sstb_unadjusted_basis_qualified_property",
            "sstb_w2_wages_from_qualified_business",
            "unadjusted_basis_qualified_property",
            "w2_wages_from_qualified_business",
        }
    ),
    3: frozenset(
        {
            "estate_income_would_be_qualified",
            "farm_operations_income_would_be_qualified",
            "farm_rent_income_would_be_qualified",
            "partnership_s_corp_income_would_be_qualified",
            "rental_income_would_be_qualified",
            "self_employment_income_would_be_qualified",
            "sstb_self_employment_income_would_be_qualified",
            "business_is_sstb",
            "qualified_bdc_income",
            "qualified_reit_and_ptp_income",
            "sstb_self_employment_income_before_lsr",
            "sstb_unadjusted_basis_qualified_property",
            "sstb_w2_wages_from_qualified_business",
            "unadjusted_basis_qualified_property",
            "w2_wages_from_qualified_business",
        }
    ),
}

_DECLARED_QBI_RNG_SEEDS_BY_VERSION = {
    1: frozenset({41, 42, 43, 64}),
    2: frozenset({2041, 2042, 2043, 2044, 2064}),
    3: frozenset({2041, 2043, 2064, 3041, 3042, 3043, 3044, 3045}),
}
_QBI_RNG_SEED_ORDER_BY_VERSION = {
    1: (41, 42, 64, 43),
    2: (2041, 2042, 2044, 2043, 2064),
    3: (2041, 3041, 3042, 3044, 3043, 3045, 2043, 2064),
}
_ROUTED_QBI_OUTPUTS = frozenset(
    {*_COMPLETE_QBI_OUTPUTS_BY_VERSION[1], "self_employment_income_before_lsr"}
)


def _fixture_bytes(name: str) -> bytes:
    payload = (_HISTORY_FIXTURE_DIRECTORY / name).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == _HISTORY_FIXTURE_SHA256[name]
    return payload


def _compose_historical_module(fixture: str, module_name: str) -> ModuleType:
    source = _fixture_bytes(fixture).decode()
    source = source.replace("from populace.", "from microcosm.")
    historical = ModuleType(module_name)
    historical.__file__ = str(_HISTORY_FIXTURE_DIRECTORY / fixture)
    historical.__dict__["__history_commit__"] = _HISTORY_COMMIT
    sys.modules[module_name] = historical
    exec(compile(source, historical.__file__, "exec"), historical.__dict__)
    return historical


@pytest.fixture(scope="module")
def historical_qbi_modules() -> tuple[ModuleType, ModuleType]:
    """Compose the frozen pre-rename pipeline against current shared types."""

    simulation = _compose_historical_module(
        "qbi_simulation.py.fixture",
        "_microcosm_historical_qbi_simulation_cfbf6330",
    )
    reconciliation = _compose_historical_module(
        "qbi_inputs.py.fixture",
        "_microcosm_historical_qbi_inputs_cfbf6330",
    )
    return simulation, reconciliation


def _validate_v3_fixture_root(payload: object) -> None:
    """Stand in only for the removed builder's outer JSON schema validator."""

    if not isinstance(payload, dict):
        raise ValueError("Historical QBI v3 fixture must be an object.")
    expected = {
        "schema_version",
        "qbi_simulation_version",
        "engine",
        "rng",
        "source_order",
        "qualification_derivations",
        "sstb_classification",
        "evidence",
        "record_form",
        "industry_mixture",
        "employer_presence",
        "profit_margin",
        "w2",
        "ubia",
        "investment",
    }
    if set(payload) != expected:
        raise ValueError("Historical QBI v3 fixture root keys changed.")
    if payload["schema_version"] != 3 or payload["qbi_simulation_version"] != 3:
        raise ValueError("Historical QBI v3 fixture version changed.")
    seeds = payload["rng"]["seeds"]
    if set(seeds.values()) != _DECLARED_QBI_RNG_SEEDS_BY_VERSION[3]:
        raise ValueError("Historical QBI v3 fixture seed families changed.")


@contextmanager
def _v3_builder_validator_shim():
    """Supply the one deleted builder dependency needed by the frozen parser."""

    name = "microcosm.build.us_runtime.qbi_v3_assumptions"
    prior = sys.modules.get(name)
    shim = ModuleType(name)
    shim.validate_qbi_v3_assumptions_payload = _validate_v3_fixture_root
    sys.modules[name] = shim
    try:
        yield
    finally:
        if prior is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior


@contextmanager
def _historical_simulator_import(historical: ModuleType):
    """Resolve the post-QRF module's deleted runtime import to the fixture."""

    name = "microcosm.build.us_runtime.qbi_simulation"
    prior = sys.modules.get(name)
    sys.modules[name] = historical
    try:
        yield
    finally:
        if prior is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prior


def _historical_assumptions(historical: ModuleType, version: int) -> Any:
    payload = json.loads(_fixture_bytes(f"qbi_assumptions_v{version}.json.fixture"))
    if version == 3:
        with _v3_builder_validator_shim():
            return historical.parse_qbi_simulation_assumptions(
                payload,
                qbi_simulation_version=version,
            )
    return historical.parse_qbi_simulation_assumptions(
        payload,
        qbi_simulation_version=version,
    )


def _synthetic_sstb_crosswalk() -> dict[str, object]:
    return {
        "schema_version": 1,
        "crosswalk_version": "passive-isolation-test-v1",
        "status": "live",
        "meta": {
            "industry_vintage": "synthetic Census industry",
            "occupation_vintage": "synthetic Census occupation",
            "legal_basis": "synthetic Section 199A test basis",
            "wiring_notes": ["synthetic test wiring"],
            "sstb_category_values": ["law"],
        },
        "industry_2017": [
            {
                "census_code": "4040",
                "census_title": "Synthetic non-SSTB industry",
                "naics": "00",
                "sstb_category": ["law"],
                "classification": "non_sstb",
                "probability": 0.0,
                "rationale": "Synthetic deterministic non-SSTB entry",
            }
        ],
        "industry_explicit_nonsstb_neighbors": [
            {
                "census_code": "0000",
                "census_title": "Synthetic documented industry",
                "why": "Synthetic documentation row",
                "probability": 0.0,
            }
        ],
        "occupation_2018": [
            {
                "census_code": "1010",
                "census_title": "Synthetic clear occupation",
                "soc": "00-0001",
                "sstb_category": ["law"],
                "classification": "clear_sstb",
                "probability": 1.0,
                "rationale": "Synthetic deterministic SSTB entry",
            },
            {
                "census_code": "2020",
                "census_title": "Synthetic non-SSTB occupation",
                "soc": "00-0002",
                "sstb_category": ["law"],
                "classification": "non_sstb",
                "probability": 0.0,
                "rationale": "Synthetic deterministic non-SSTB entry",
            },
            {
                "census_code": "3030",
                "census_title": "Synthetic ambiguous occupation",
                "soc": "00-0003",
                "sstb_category": ["law"],
                "classification": "ambiguous",
                "probability": 0.3,
                "rationale": "Synthetic ambiguous entry",
                "provisional": True,
                "basis": "Synthetic provisional basis",
            },
        ],
        "occupation_explicit_nonsstb_notes": [
            {
                "census_code": "0000",
                "census_title": "Synthetic documented occupation",
                "why": "Synthetic documentation row",
                "probability": 0.0,
            }
        ],
    }


def _synthetic_qbi_frame() -> Frame:
    index = np.arange(64, dtype=np.float64)
    partnership_s_corp_income = (index + 6) * 1_500
    columns = {
        "self_employment_income": np.where(
            index % 3 == 0,
            (index + 1) * 1_000,
            np.where(index % 7 == 0, -(index + 1) * 400, 0),
        ),
        "farm_operations_income": np.where(
            index % 5 == 0,
            (index + 2) * 700,
            np.where(index % 11 == 0, -(index + 2) * 250, 0),
        ),
        "farm_rent_income": np.where(index % 8 == 0, (index + 3) * 350, 0),
        "rental_income": np.where(
            index % 4 == 0,
            (index + 4) * 900,
            np.where(index % 13 == 0, -(index + 4) * 300, 0),
        ),
        "estate_income": np.where(index % 9 == 0, (index + 5) * 1_100, 0),
        "partnership_s_corp_income": partnership_s_corp_income,
        "partnership_income": partnership_s_corp_income,
        "s_corp_income": np.zeros(len(index), dtype=np.float64),
        "AGI": (index - 8) * 20_000,
        "PEIOOCC": np.resize(np.array([1010, 2020, 3030]), len(index)),
        "non_qualified_dividend_income": np.where(
            index % 2 == 0,
            (index + 1) * 120,
            0,
        ),
    }
    person = pd.DataFrame(columns)
    ids = np.arange(1, len(person) + 1, dtype=np.int64)
    person.insert(0, "person_id", ids)
    for entity in ("household", "tax_unit", "spm_unit", "family", "marital_unit"):
        person[f"person_{entity}_id"] = ids
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame({"household_id": ids}),
            "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
            "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
            "family": pd.DataFrame({"family_id": ids}),
            "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
        },
        US_SCHEMA,
        {"household": Weights(np.ones(len(person)), WeightKind.DESIGN)},
    )


def _with_person(frame: Frame, person: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables.update({link: frame.link(link).copy() for link in frame.links})
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


class _RecordingGenerator:
    def __init__(
        self,
        generator: np.random.Generator,
        seed: int,
        records: list[tuple[object, ...]],
    ) -> None:
        self._generator = generator
        self._seed = seed
        self._records = records

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._generator, name)
        if not callable(attribute):
            return attribute

        def recorded(*args: object, **kwargs: object) -> Any:
            value = attribute(*args, **kwargs)
            array = np.asarray(value)
            self._records.append(
                (self._seed, name, array.dtype.str, array.shape, array.tobytes())
            )
            return value

        return recorded


def _run_historical_pipeline(
    historical: ModuleType,
    reconciliation: ModuleType,
    assumptions: Any,
    *,
    version: int,
    include_passive_stage: bool,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray | None,
    list[tuple[object, ...]],
]:
    frame = _synthetic_qbi_frame()
    person = frame.table("person")
    inputs = historical.QbiSimulationInputs.from_puf_arrays(
        {column: person[column].to_numpy(copy=True) for column in person}
    )
    records: list[tuple[object, ...]] = []
    real_default_rng = np.random.default_rng

    def recording_default_rng(seed: int) -> _RecordingGenerator:
        records.append(("generator", seed, "PCG64"))
        return _RecordingGenerator(real_default_rng(seed), seed, records)

    np.random.default_rng = recording_default_rng
    try:
        simulation_outputs = historical.simulate_qbi_inputs(
            inputs,
            assumptions=assumptions,
            qbi_simulation_version=version,
            sstb_crosswalk=_synthetic_sstb_crosswalk(),
        )
        assert set(simulation_outputs) == _COMPLETE_QBI_OUTPUTS_BY_VERSION[version]
        person = person.copy(deep=True)
        for column, values in simulation_outputs.items():
            person[column] = values
        person["self_employment_income_before_lsr"] = np.where(
            simulation_outputs["business_is_sstb"],
            0.0,
            inputs.self_employment_income,
        )
        frame = _with_person(frame, person)
        if include_passive_stage:
            frame = with_us_qbi_passive_passthrough_assignment(frame, seed=13)

        if version == 1:
            frame = reconciliation.with_us_qbi_input_reconciliation(frame)
        else:
            with _historical_simulator_import(historical):
                frame = reconciliation.with_host_sstb_classification(
                    frame,
                    qbi_simulation_version=version,
                    assumptions=assumptions,
                    sstb_crosswalk=_synthetic_sstb_crosswalk(),
                )
    finally:
        np.random.default_rng = real_default_rng

    routed = frame.table("person")
    passive = (
        routed[US_QBI_PASSIVE_PASSTHROUGH_OUTPUT_COLUMN].to_numpy(copy=True)
        if include_passive_stage
        else None
    )
    routed_outputs = {
        column: routed[column].to_numpy(copy=True) for column in _ROUTED_QBI_OUTPUTS
    }
    return simulation_outputs, routed_outputs, passive, records


def _legacy_rng_state_identity() -> tuple[object, ...]:
    name, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return (
        name,
        keys.dtype.str,
        keys.shape,
        keys.tobytes(),
        position,
        has_gauss,
        cached_gaussian,
    )


@pytest.mark.parametrize("qbi_simulation_version", (1, 2, 3), ids=("v1", "v2", "v3"))
def test_historical_qbi_versions_are_byte_isolated_from_passive_stage(
    historical_qbi_modules: tuple[ModuleType, ModuleType],
    qbi_simulation_version: int,
) -> None:
    historical, reconciliation = historical_qbi_modules
    assert historical.__history_commit__ == _HISTORY_COMMIT
    assert reconciliation.__history_commit__ == _HISTORY_COMMIT
    assert tuple(historical.QBI_SIMULATION_SUPPORTED_VERSIONS) == (1, 2, 3)
    assert set(_COMPLETE_QBI_OUTPUTS_BY_VERSION) == {1, 2, 3}
    assert set(US_QBI_OUTPUT_COLUMNS) == _COMPLETE_QBI_OUTPUTS_BY_VERSION[1]
    assumptions = _historical_assumptions(historical, qbi_simulation_version)

    declared_seeds = {
        value for name, value in vars(assumptions).items() if name.endswith("_seed")
    }
    assert declared_seeds == _DECLARED_QBI_RNG_SEEDS_BY_VERSION[qbi_simulation_version]

    original_global_state = np.random.get_state()
    try:
        np.random.seed(722)
        untouched_global_state = _legacy_rng_state_identity()
        (
            baseline_simulation,
            baseline,
            baseline_passive,
            baseline_draws,
        ) = _run_historical_pipeline(
            historical,
            reconciliation,
            assumptions,
            version=qbi_simulation_version,
            include_passive_stage=False,
        )
        assert _legacy_rng_state_identity() == untouched_global_state
        staged_simulation, staged, passive, staged_draws = _run_historical_pipeline(
            historical,
            reconciliation,
            assumptions,
            version=qbi_simulation_version,
            include_passive_stage=True,
        )
        assert _legacy_rng_state_identity() == untouched_global_state
    finally:
        np.random.set_state(original_global_state)

    assert baseline_passive is None
    assert passive is not None
    assert np.count_nonzero(passive > 0.0) > 0
    weights = np.linspace(1.0, 2.0, len(passive), dtype=np.float64)
    assert float(np.dot(weights, passive)) > 0.0

    complete_simulator_surface = _COMPLETE_QBI_OUTPUTS_BY_VERSION[
        qbi_simulation_version
    ]
    assert set(baseline_simulation) == complete_simulator_surface
    assert set(staged_simulation) == complete_simulator_surface
    assert set(baseline) == _ROUTED_QBI_OUTPUTS
    assert set(staged) == _ROUTED_QBI_OUTPUTS
    for column in sorted(_ROUTED_QBI_OUTPUTS):
        baseline_values = np.asarray(baseline[column])
        staged_values = np.asarray(staged[column])
        assert staged_values.dtype.str == baseline_values.dtype.str
        assert staged_values.tobytes() == baseline_values.tobytes()

    assert baseline_draws == staged_draws
    assert baseline_draws
    drawn_seeds = tuple(
        record[1] for record in baseline_draws if record[0] == "generator"
    )
    assert drawn_seeds == _QBI_RNG_SEED_ORDER_BY_VERSION[qbi_simulation_version]
