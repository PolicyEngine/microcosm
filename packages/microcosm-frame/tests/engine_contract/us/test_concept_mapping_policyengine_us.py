"""The policyengine-us concept mapping against the installed engine's metadata.

Every mapped input must be a real input on the stated entity with a dtype its
transform can produce; every recode must land in the engine's enum; the
take-up bindings must be exactly the engine's data-seeded take-up flags; the
formula-ownership claims in the mapping's notes must hold; and the committed
coverage report must match the engine. The round trip through a real
Microsimulation is the engine_scenario test of the same name.
"""

import json
from importlib.metadata import version

import pytest

from microcosm.frame.adapters.policyengine_us import (
    POLICYENGINE_US_CONCEPT_MAPPING,
    PolicyEngineUSEngine,
    PolicyEngineUSVariableMetadataIndex,
    _enum_domain,
)
from microcosm.frame.concept_mapping import (
    Recode,
    TakeUpThreshold,
    coverage_report,
    rules_engine_input_refs,
)
from test_support.microcosm_frame.concept_engine_frames import expected_input_dtypes
from test_support.microcosm_frame.concept_mappings import coverage_golden

MAPPING = POLICYENGINE_US_CONCEPT_MAPPING


@pytest.fixture(scope="module")
def index() -> PolicyEngineUSVariableMetadataIndex:
    return PolicyEngineUSVariableMetadataIndex()


@pytest.fixture(scope="module")
def engine() -> PolicyEngineUSEngine:
    return PolicyEngineUSEngine(spm={"geography_kind": "national"})


def test_mapping_was_reviewed_against_the_installed_engine() -> None:
    assert MAPPING.engine_version == version("policyengine-us"), (
        "policyengine-us moved: review the concept mapping against the new "
        "engine, then update engine_version and the coverage report."
    )


def test_every_mapped_input_is_an_engine_input_on_its_entity(index) -> None:
    inputs = set(index.variables())
    for binding in MAPPING.bindings:
        assert binding.engine_input in inputs, binding.engine_input
        metadata = index.variable_metadata(binding.engine_input)
        assert metadata.entity == binding.engine_entity, binding.engine_input


def test_mapped_input_dtypes_fit_their_transforms(index) -> None:
    for binding in MAPPING.bindings:
        dtype = index.variable_metadata(binding.engine_input).dtype
        assert dtype in expected_input_dtypes(binding), (binding.engine_input, dtype)


def test_recodes_land_in_the_engine_enums(engine) -> None:
    for binding in MAPPING.bindings:
        if isinstance(binding.transform, Recode) and binding.engine_input != (
            "is_female"
        ):
            members = set(_enum_domain(engine._variable(binding.engine_input)))
            images = {value for _, value in binding.transform.pairs}
            assert images <= members, (binding.engine_input, images - members)


def test_take_up_bindings_are_the_engines_data_seeded_flags(engine) -> None:
    seeded = {
        name
        for name, facts in engine.take_up_contract().items()
        if facts["engine_class"] == "data_seeded"
    }
    bound = {
        binding.engine_input
        for binding in MAPPING.bindings
        if isinstance(binding.transform, TakeUpThreshold)
        and binding.engine_input.startswith("takes_up")
    }
    assert bound == seeded


def test_structural_inputs_are_engine_inputs(index) -> None:
    assert set(MAPPING.structural_inputs) <= set(index.variables())


def test_the_mapping_notes_formula_ownership_claims_hold(index) -> None:
    # Named in notes as formula-owned aggregates, never inputs.
    claimed = {
        "weeks_worked",
        "employment_income",
        "interest_income",
        "dividend_income",
        "ordinary_dividend_income",
        "long_term_capital_gains",
        "social_security",
        "rent",
        "is_male",
        "is_tax_unit_head",
    }
    assert not claimed & set(index.variables())
    assert index.formula_owned_outputs(claimed) == claimed


def test_the_liquid_asset_reason_names_cpi_uprated_person_inputs(index, engine) -> None:
    inputs = set(index.variables())
    for name in ("bank_account_assets", "stock_assets", "bond_assets"):
        assert name in inputs, name
        assert index.variable_metadata(name).entity == "person", name
        assert index.variable_metadata(name).dtype == "float", name
        assert engine._variable(name).uprating == "gov.bls.cpi.cpi_u", name


def test_committed_coverage_report_matches_the_engine(index) -> None:
    report = coverage_report(MAPPING, rules_engine_input_refs(index))
    assert report.unknown_inputs == ()
    committed = json.loads(coverage_golden("policyengine-us").read_text("utf-8"))
    assert committed == json.loads(json.dumps(report.to_dict())), (
        "Regenerate with: uv run --no-sync python tools/refresh_concept_coverage.py "
        "--engine policyengine-us"
    )
