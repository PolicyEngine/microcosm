"""New Zealand's declared donor money normalization (G8-w w2a).

The declaration is spec data; nothing activates it until a New Zealand
transport graph reads it (w3). These checks pin its routes against the concept
schema, its single-source exchange rate, and the generated donor factors.
"""

from __future__ import annotations

import ast
import json
from fractions import Fraction

import pytest

from microcosm.build.money_normalization import (
    check_money_factor,
    money_column_rosters,
    money_concept_columns,
    normalized_column_families,
    resolve_money_rate,
    selected_money_factors,
    validate_money_normalization,
)
from microcosm.build.reference_factors import validate_factor_context
from microcosm.build.transport import compose as compose_module
from microcosm.frame.concepts import TransportRule
from test_support.paths import paths_for

_SPEC = paths_for("microcosm-build").package / "src/microcosm/build/nz"
_DONOR = {
    "donor_cy2024_to_ty2027_low": 1077090,
    "donor_cy2024_to_ty2027_central": 1079073,
    "donor_cy2024_to_ty2027_high": 1081052,
}


def _load(name):
    return json.loads((_SPEC / name).read_text())


def _declaration():
    return validate_money_normalization(_load("money_normalization.json"))


def _factor(factor_id):
    return next(
        entry
        for entry in _load("reference_aging.json")["factors"]
        if entry["declaration"]["id"] == factor_id
    )


def test_normalized_routes_follow_the_design_table_and_the_concept_schema():
    declaration = _declaration()
    assert (declaration["from_currency"], declaration["to_currency"]) == ("USD", "NZD")
    assert declaration["quote"] == "to_per_from"
    assert declaration["rounding"] == {"digits": 2, "mode": "half_even"}
    period = declaration["donor_money_period"]
    assert (period["type"], period["value"], period["basis"]) == (
        "calendar_year",
        2024,
        "declared_convention",
    )
    assert declaration["cents_suffix"] == "_cents"
    assert normalized_column_families(declaration) == {
        ("person", "nonfarm_self_employment_income"): "mixed_income",
        ("person", "farm_self_employment_income"): "mixed_income",
        ("person", "interest_income"): "capital_income",
        ("person", "dividend_income"): "capital_income",
        ("person", "rental_income"): "capital_income",
        ("person", "realized_capital_gains"): "capital_income",
        ("person", "private_pension_income"): "pensions",
        ("household", "mortgage_interest"): "owner_housing_costs",
        ("household", "mortgage_principal"): "owner_housing_costs",
        ("household", "property_tax"): "owner_housing_costs",
    }
    rosters = money_column_rosters(declaration)
    assert {key for key, (route, _) in rosters.items() if route != "normalized"} == {
        ("person", "employment_income"),
        ("household", "rent"),
        ("person", "liquid_financial_assets"),
    }
    assert {route for route, _ in rosters.values()} == {"normalized", "quantile_mapped"}
    assert declaration["pending"] == {}
    assert rosters["person", "employment_income"][1][0].startswith("P1:")
    assert rosters["household", "rent"][1][0].startswith("C1:")
    assert [item[:3] for item in rosters["person", "liquid_financial_assets"][1]] == [
        "P3:",
        "C2:",
    ]


def test_every_money_concept_surviving_create_is_routed_and_drop_is_not():
    rosters = money_column_rosters(_declaration())
    surviving = {
        key
        for key, item in money_concept_columns().items()
        if item.transport is not TransportRule.DROP
    }
    assert set(rosters) == surviving
    assert len(surviving) == 13
    assert ("person", "public_pension_income") not in rosters
    signed = {
        key for key, item in money_concept_columns().items() if item.monetary.signed
    }
    assert signed == {
        ("person", "nonfarm_self_employment_income"),
        ("person", "farm_self_employment_income"),
        ("person", "rental_income"),
        ("person", "realized_capital_gains"),
    }


def test_every_family_uses_the_donor_factor_for_its_profile_with_a_disclosure():
    declaration = _declaration()
    assert declaration["factor_profile"] == {
        "dimension": "aging_index",
        "profiles": ["low", "central", "high"],
        "central_profile": "central",
    }
    for family, entry in declaration["index_family_factors"].items():
        assert entry["factors"] == {
            profile: f"donor_cy2024_to_ty2027_{profile}"
            for profile in ("low", "central", "high")
        }, family
        assert "AWE" in entry["disclosure"], family
        assert (
            "provisional pending Max's ruling on non-wage ageing (H3)"
            in entry["disclosure"]
        ), family


def test_donor_money_basis_cites_model_lineage_as_a_declared_convention():
    declaration = _declaration()
    evidence = " ".join(declaration["donor_money_period"]["evidence"])
    assert "tools/build_us_fiscal_refresh_release.py" in evidence
    assert "PERIOD = 2024" in evidence and "_dataset_from_frame" in evidence
    assert "time_period to USSingleYearDataset" in evidence
    assert (
        "microcosm.frame.adapters.policyengine_us.PolicyEngineUSEngine"
        "._write_and_verify" in evidence
    )
    assert "_time_period" in evidence
    assert "not authentication" in evidence
    disclosure = declaration["notes"]
    assert "declared convention covering flows and asset stocks" in disclosure
    assert "consistency check, not authentication" in disclosure
    assert "No donor file is opened" in disclosure


def test_donor_money_basis_evidence_matches_the_code_it_cites():
    # The evidence names symbols, not lines (test_nz_spec_package refuses local
    # line citations), so each cited fact is checked against the code itself.
    repository = paths_for("microcosm-build").repository
    tool = ast.parse(
        (repository / "tools/build_us_fiscal_refresh_release.py").read_text()
    )
    periods = [
        node.value.value
        for node in tool.body
        if isinstance(node, ast.Assign)
        and [getattr(target, "id", None) for target in node.targets] == ["PERIOD"]
    ]
    assert periods == [_declaration()["donor_money_period"]["value"]]
    (builder,) = (
        node
        for node in tool.body
        if isinstance(node, ast.FunctionDef) and node.name == "_dataset_from_frame"
    )
    passed = [
        ast.unparse(keyword.value)
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "USSingleYearDataset"
        for keyword in node.keywords
        if keyword.arg == "time_period"
    ]
    assert passed == ["PERIOD"]
    adapter = ast.parse(
        (
            paths_for("microcosm-frame").package
            / "src/microcosm/frame/adapters/policyengine_us.py"
        ).read_text()
    )
    (engine,) = (
        node
        for node in adapter.body
        if isinstance(node, ast.ClassDef) and node.name == "PolicyEngineUSEngine"
    )
    (writer,) = (
        node
        for node in engine.body
        if isinstance(node, ast.FunctionDef) and node.name == "_write_and_verify"
    )
    stored = [
        ast.unparse(node.args[1])
        for node in ast.walk(writer)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "put"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "_time_period"
    ]
    assert stored == ["pd.Series([int(period)])"]


def test_scenario_knobs_select_their_own_donor_factor():
    declaration = _declaration()
    scenarios = _load("scenarios.json")
    assert scenarios["central_knobs"]["aging_index"] == "central"
    for scenario in scenarios["scenarios"]:
        profile, factors = selected_money_factors(declaration, scenario["knobs"])
        expected = {"V1": "low", "V2": "high"}.get(scenario["id"], "central")
        assert profile == expected, scenario["id"]
        assert set(factors.values()) == {f"donor_cy2024_to_ty2027_{expected}"}


def test_the_scenarios_default_aging_index_is_the_money_central_profile():
    # A scenario that leaves the knob unset selects the declaration's central
    # profile, so the scenarios' default must name the same profile.
    profile = _declaration()["factor_profile"]
    scenarios = _load("scenarios.json")
    dimension = profile["dimension"]
    assert scenarios["central_knobs"][dimension] == profile["central_profile"]
    assert set(scenarios["knob_domains"][dimension]) == set(profile["profiles"])


def test_money_and_the_reference_profiles_age_to_one_target_period():
    declaration = _declaration()
    factors = [
        entry["declaration"] for entry in _load("reference_aging.json")["factors"]
    ]
    donor = {row["id"]: row["target_period"] for row in factors if row["id"] in _DONOR}
    assert donor == dict.fromkeys(_DONOR, declaration["target_period"])
    assert {
        row["target_period"] for row in factors if row["purpose"] == "profile_input"
    } == {declaration["target_period"]}


@pytest.mark.parametrize("profile", ["low", "central", "high"])
def test_a_missing_declared_profile_factor_refuses(profile):
    declaration = _load("money_normalization.json")
    del declaration["index_family_factors"]["capital_income"]["factors"][profile]
    with pytest.raises(ValueError, match="one factor for every declared profile"):
        validate_money_normalization(declaration)


def test_the_rate_has_one_source_and_the_bridge_points_to_it():
    declaration = _declaration()
    rate, evidence = resolve_money_rate(declaration)
    assert rate == Fraction(827, 500)
    assert evidence["citation"]["url"].startswith("https://www.irs.gov/")
    assert evidence["citation"]["column"] == "2024"
    occurrences = [
        path.name
        for path in sorted(_SPEC.rglob("*.json"))
        if "1.654" in path.read_text()
    ]
    assert occurrences == ["money_normalization.json"]
    assert _load("money_normalization.json")["rate"]["text"] == "1.654"
    bridge = _load("currency_bridge.json")
    assert bridge["rate"] == {"resource": "money_normalization.json", "path": ["rate"]}
    assert "awaiting D1" not in bridge["status"]
    assert "d1043" in bridge["status"]
    assert "method-card-D1-adopted.md" in bridge["status"]


def test_the_three_donor_factors_are_exact_and_only_money_may_apply_them():
    declaration = _declaration()
    context = _load("reference_aging.json")
    assert set(validate_factor_context(context)) >= set(_DONOR)
    for factor_id, numerator in _DONOR.items():
        entry = _factor(factor_id)
        assert entry["operational_factor"] == {
            "numerator": numerator,
            "denominator": 1000000,
        }
        assert check_money_factor(entry, declaration) == (numerator, 6)
        receipt = entry["receipt"]
        assert Fraction(**receipt["denominator_mean"]) == Fraction("1591.925")
        assert [slot["period"]["value"] for slot in receipt["source_schedule"]] == [
            "2024-Q1",
            "2024-Q2",
            "2024-Q3",
            "2024-Q4",
        ]
        assert [slot["decimal_operand"] for slot in receipt["source_schedule"]] == [
            "1563.54",
            "1586.32",
            "1595.11",
            "1622.73",
        ]
        # The numerator is the TY2024 factors' N_s for the same profile:
        # N_s = 1700.68 * (1 + q + q**2 + q**3) / 4 with q = (1 + g)**(1/4).
        profile = factor_id.rsplit("_", 1)[1]
        ird = _factor(f"ird_ty2024_to_ty2027_{profile}")["receipt"]
        assert receipt["growth"] == ird["growth"]
        assert receipt["root_bounds"]["radicand"] == ird["root_bounds"]["radicand"]
        assert receipt["target_schedule"] == ird["target_schedule"]
        for bound in ("lower", "upper"):
            root = Fraction(**receipt["root_bounds"][bound])
            assert Fraction(**receipt["numerator_mean_bounds"][bound]) == (
                Fraction("1700.68") * sum(root**index for index in range(4)) / 4
            )
        assert entry["declaration"]["purpose"] == "money_normalization_input"


@pytest.mark.parametrize("profile", ["low", "central", "high"])
def test_a_tax_year_factor_is_refused_on_donor_money(profile):
    entry = _factor(f"ird_ty2024_to_ty2027_{profile}")
    with pytest.raises(ValueError, match="applies only money_normalization_input"):
        check_money_factor(entry, _declaration())
    # Even relabelled, its April-to-March schedule is not calendar 2024.
    relabelled = json.loads(json.dumps(entry))
    relabelled["declaration"]["purpose"] = "money_normalization_input"
    with pytest.raises(ValueError, match="not the declared donor money period"):
        check_money_factor(relabelled, _declaration())


def test_no_graph_reads_it_yet_so_preflight_names_no_money_gap():
    spec = compose_module.load_transport_spec("nz")
    assert compose_module._money_producer_gaps(spec) == []
