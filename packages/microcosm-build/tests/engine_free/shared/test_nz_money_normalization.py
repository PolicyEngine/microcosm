"""New Zealand's declared donor money normalization (G8-w w2a).

The declaration is spec data; nothing activates it until a New Zealand
transport graph reads it (w3). These checks pin its routes against the concept
schema, its single-source exchange rate, and the generated donor factors.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from datetime import date
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from fractions import Fraction

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.money_normalization import (
    check_money_factor,
    money_column_rosters,
    money_concept_columns,
    normalize_money_tables,
    normalized_column_families,
    resolve_money_rate,
    selected_money_factors,
    validate_money_normalization,
)
from microcosm.build.reference_factors import validate_factor_context
from microcosm.build.transport import cli as cli_module
from microcosm.build.transport import compose as compose_module
from microcosm.build.transport.money_kernels import MONEY_NORMALIZE_REF
from microcosm.frame.concepts import TransportRule
from microcosm.graph.canonical import canonical_json
from test_support.paths import paths_for

_SPEC = paths_for("microcosm-build").package / "src/microcosm/build/nz"
_DONOR = {
    "donor_cy2024_to_ty2027_low": 1079413,
    "donor_cy2024_to_ty2027_central": 1080408,
    "donor_cy2024_to_ty2027_high": 1081401,
}


def _load(name):
    return json.loads((_SPEC / name).read_text())


def _declaration():
    return validate_money_normalization(_load("money_normalization.json"))


def test_run_entry_checks_all_scenario_donor_factors(monkeypatch):
    packaged = compose_module.load_transport_spec("nz")
    declaration = compose_module.transport_resource(packaged, "money_normalization")
    spec = {
        **packaged,
        "resources": {
            **packaged["resources"],
            compose_module.GRAPH_RESOURCE: {
                "nodes": [
                    {
                        "id": f"money.{profile}",
                        "kernel": MONEY_NORMALIZE_REF,
                        "params": {
                            "declaration": {
                                "resource": "money_normalization",
                                "encoding": "json",
                            },
                            "declaration_sha256": {
                                "resource": "money_normalization",
                                "encoding": "sha256",
                            },
                            "profile_selection": canonical_json(
                                {"aging_index": profile}
                            ).decode("utf-8"),
                            "factor_context": {
                                "resource": "reference_aging",
                                "encoding": "json",
                            },
                            "factor_context_sha256": {
                                "resource": "reference_aging",
                                "encoding": "sha256",
                            },
                        },
                    }
                    for profile in declaration["factor_profile"]["profiles"]
                ]
            },
        },
    }
    checked = {}
    monkeypatch.setattr(cli_module, "_today", lambda: date(2026, 11, 1))
    monkeypatch.setattr(
        cli_module,
        "require_current_factors",
        lambda selected, *, today: checked.update(selected),
    )
    cli_module._require_current_run_factors(spec, resume="auto")
    assert set(checked) & _DONOR.keys() == _DONOR.keys()


# A whole numeric token: an optional sign, digits and points, then any
# exponent, with no letter, digit or underscore right after it. A sign that
# follows a digit separates two numbers, as in the range 1.650-1.654 or the sum
# 0+1.654, so the number after it stands unsigned. Otherwise the token,
# including its sign, may not follow a letter, digit, underscore, point or
# sign: -1.654 is negative, and v-1.654, v1.654 and a1654e-3b hold no number.
# A run with two points such as 2.1.654 matches, but Decimal rejects it and
# _text_numbers skips it. _text_numbers also drops one trailing point, which
# ends a sentence.
_NUMBER_TOKEN = re.compile(
    r"(?:(?<![\w.+-])[-+]?|(?<=\d[-+]))(?:\d[\d.]*|\.\d+)(?:[eE][-+]?\d+)?(?!\w)"
)


def _text_numbers(text):
    for token in _NUMBER_TOKEN.findall(text):
        try:
            yield Decimal(token.removesuffix("."))
        except InvalidOperation:
            continue


def _rate_locations(value, rate, path=()):
    """Yield each JSON location that states ``rate`` exactly, in any form.

    A location matches when it holds the rate as a JSON number, as a
    numerator/denominator object, or as a whole numeric token inside text,
    including signs and exponents. Longer numbers such as 1.6541 or 11.654 do
    not match. Decimal and Fraction compare exactly.
    """
    if isinstance(value, Mapping):
        numerator, denominator = value.get("numerator"), value.get("denominator")
        if (
            type(numerator) is int
            and type(denominator) is int
            and denominator
            and Fraction(numerator, denominator) == rate
        ):
            yield path
        for key, item in value.items():
            yield from _rate_locations(item, rate, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _rate_locations(item, rate, (*path, index))
    elif isinstance(value, Decimal):
        if value == rate:
            yield path
    elif isinstance(value, str):
        if any(number == rate for number in _text_numbers(value)):
            yield path


def _spec_rate_locations(rate):
    return [
        (path.name, location)
        for path in sorted(_SPEC.rglob("*.json"))
        for location in _rate_locations(
            json.loads(path.read_text(), parse_float=Decimal), rate
        )
    ]


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
        ("person", "liquid_financial_assets"): "consumer_prices",
        ("household", "mortgage_interest"): "owner_housing_costs",
        ("household", "mortgage_principal"): "owner_housing_costs",
        ("household", "property_tax"): "owner_housing_costs",
    }
    rosters = money_column_rosters(declaration)
    assert {key for key, (route, _) in rosters.items() if route != "normalized"} == {
        ("person", "employment_income"),
        ("household", "rent"),
    }
    assert {route for route, _ in rosters.values()} == {"normalized", "quantile_mapped"}
    assert declaration["pending"] == {}
    assert rosters["person", "employment_income"][1][0].startswith("P1:")
    assert rosters["household", "rent"][1][0].startswith("C1:")
    assert rosters["person", "liquid_financial_assets"] == ("normalized", ())


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
    # consumer_prices is d1982's identity family; it carries no donor factor.
    aged = {
        family: entry
        for family, entry in declaration["index_family_factors"].items()
        if family != "consumer_prices"
    }
    assert len(aged) == 4
    for family, entry in aged.items():
        assert entry["factors"] == {
            profile: f"donor_cy2024_to_ty2027_{profile}"
            for profile in ("low", "central", "high")
        }, family
        assert "AWE" in entry["disclosure"], family
        disclosure = entry["disclosure"]
        assert (
            "Max's ruling d1981 (2026-10-26) adopts this as an interim convention"
            in disclosure
        ), family
        assert (
            "pre-declares a hold-fixed sensitivity (factor 1), which no profile "
            "here carries yet (G8-w3)" in disclosure
        ), family
        assert "provisional" not in disclosure, family


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
    # line citations), so the test checks the cited constant and call
    # expressions against the code itself.
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
        assert factors.pop("consumer_prices") is None, scenario["id"]
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
    # The VHF sibling declaration copies the central rate; only its four
    # index families differ (tools/build_nz_transport_graph.py).
    assert _spec_rate_locations(rate) == [
        ("money_normalization.json", ("rate", "text")),
        ("money_normalization_hold_fixed.json", ("rate", "text")),
    ]
    assert (
        _load("money_normalization_hold_fixed.json")["rate"]
        == _load("money_normalization.json")["rate"]
    )
    assert _load("money_normalization.json")["rate"]["text"] == "1.654"
    bridge = _load("currency_bridge.json")
    assert bridge["rate"] == {"resource": "money_normalization.json", "path": ["rate"]}
    assert "awaiting D1" not in bridge["status"]
    assert "d1043" in bridge["status"]
    assert "method-card-D1-adopted.md" in bridge["status"]


@pytest.mark.parametrize(
    ("value", "found"),
    [
        (Decimal("1.654"), True),
        ({"numerator": 827, "denominator": 500}, True),
        ({"numerator": 1654, "denominator": 1000}, True),
        ("converted at 1.654 NZD per USD", True),
        ("The rate is 1.654.", True),
        ("1654e-3", True),
        ("16.54E-1", True),
        ("+1.654", True),
        ("1.654e0", True),
        ("1.654e1", False),
        ("1.654e-3", False),
        ("-1.654", False),
        ("1,654", False),
        ("Dependency version v1.654.", False),
        ("a1654e-3b", False),
        ("1.654e+", False),
        ("rate_1.654", False),
        ("Dependency version v-1.654.", False),
        ("a-1.654", False),
        ("v-1654e-3", False),
        ("(-1.654)", False),
        ("(+1.654)", True),
        ("+-1.654", False),
        ("The exchange-rate range is 1.650-1.654.", True),
        ("The exchange-rate range is 1.700-1.654.", True),
        ("The exchange-rate formula is 0+1.654.", True),
        ("Valid 2026-04-01 to 2026-11-03.", False),
        ("(1.654)", True),
        ("1.6540", True),
        ("1.6541", False),
        ("11.654", False),
        ("1.654.2", False),
        ("2.1.654", False),
        ("v1654", False),
        (Decimal("1.6541"), False),
        ({"numerator": True, "denominator": 1}, False),
        ({"numerator": 827, "denominator": 0}, False),
        (1654, False),
    ],
)
def test_the_rate_scan_matches_exact_values_only(value, found):
    document = {"outer": [{"inner": value}]}
    expected = [("outer", 0, "inner")] if found else []
    assert list(_rate_locations(document, Fraction(827, 500))) == expected


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
        # The numerator is the TY2024 factors' N_s for the same profile. June and
        # September 2026 are observed, so
        # N_s = (1700.68 + 1714.86 * (1 + q + q**2)) / 4 with q = (1 + g)**(1/4).
        profile = factor_id.rsplit("_", 1)[1]
        ird = _factor(f"ird_ty2024_to_ty2027_{profile}")["receipt"]
        assert receipt["growth"] == ird["growth"]
        assert receipt["root_bounds"]["radicand"] == ird["root_bounds"]["radicand"]
        assert receipt["target_schedule"] == ird["target_schedule"]
        for bound in ("lower", "upper"):
            root = Fraction(**receipt["root_bounds"][bound])
            assert (
                Fraction(**receipt["numerator_mean_bounds"][bound])
                == (
                    Fraction("1700.68")
                    + Fraction("1714.86") * sum(root**index for index in range(3))
                )
                / 4
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


def test_every_money_column_has_a_producer():
    """G8-x0 maps rent and converts liquid assets, so no money column waits."""
    spec = compose_module.load_transport_spec("nz")
    assert compose_module._money_producer_gaps(spec) == []


_LIQUID = ("person", "liquid_financial_assets")
_LIQUID_CENTS = ("person", "liquid_financial_assets_cents")
#: Stored donor stocks: nonnegative, and small enough for int64 minor units.
_STOCK = st.floats(min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False)
_STOCK_PROPERTY = settings(max_examples=200, deadline=None, database=None)


def _converted_stock(values, declaration):
    """``money.normalize``'s arithmetic on the liquid-asset column alone."""
    rate, evidence = resolve_money_rate(declaration)
    entry = declaration["index_family_factors"]["consumer_prices"]
    assert entry["identity"] is True
    return normalize_money_tables(
        {"person": pd.DataFrame({"liquid_financial_assets": values}, dtype="float64")},
        {**declaration, "columns": {"person": ["liquid_financial_assets"]}},
        rate=rate,
        rate_text=evidence["text"],
        factors={"consumer_prices": {"identity": True, "reason": entry["reason"]}},
    )


@_STOCK_PROPERTY
@given(values=st.lists(_STOCK, min_size=1, max_size=20))
def test_property_the_d1982_stock_converts_once_and_rounds_half_even_to_cents(values):
    arrays, (receipt,) = _converted_stock(values, _declaration())
    expected = [
        int(
            (Decimal(repr(value)) * Decimal("1.654") * 100).to_integral_value(
                rounding=ROUND_HALF_EVEN
            )
        )
        for value in values
    ]
    assert arrays[_LIQUID_CENTS].tolist() == expected
    assert arrays[_LIQUID].tolist() == [
        float(Fraction(cents, 100)) for cents in expected
    ]
    assert min(expected) >= 0
    assert receipt["factor"]["identity"] is True
    # The identity factor adds no second rounding after the conversion.
    assert receipt["sum_after"] == receipt["sum_after_fx"]
    assert receipt["factor_rounding_residual_minor_units"] == {
        "numerator": 0,
        "denominator": 1,
    }


def test_a_zero_stock_stays_exactly_zero():
    arrays, _ = _converted_stock([0.0], _declaration())
    assert arrays[_LIQUID_CENTS].tolist() == [0]
    assert arrays[_LIQUID].tolist() == [0.0]


@pytest.mark.parametrize("profile", ["low", "central", "high"])
def test_the_d1982_stock_carries_no_factor_under_any_aging_profile(profile):
    _, factors = selected_money_factors(_declaration(), {"aging_index": profile})
    assert factors["consumer_prices"] is None


@_STOCK_PROPERTY
@given(values=st.lists(_STOCK, min_size=1, max_size=20))
def test_property_the_hold_fixed_sibling_converts_the_stock_identically(values):
    central = _declaration()
    hold = validate_money_normalization(_load("money_normalization_hold_fixed.json"))
    assert (
        hold["index_family_factors"]["consumer_prices"]
        == central["index_family_factors"]["consumer_prices"]
    )
    first, _ = _converted_stock(values, central)
    second, _ = _converted_stock(values, hold)
    assert first[_LIQUID_CENTS].tolist() == second[_LIQUID_CENTS].tolist()


def test_the_d1982_disclosure_names_the_convention_and_its_limits():
    entry = _declaration()["index_family_factors"]["consumer_prices"]
    assert set(entry) == {"identity", "reason"}
    for phrase in (
        "d1982 option A (2026-10-26)",
        "bank, stock and bond assets",
        "this declaration's rate",
        "factor one and no ageing",
        "1 April 2026 snapshot as a convention",
        "claims no valuation date or authenticated donor lineage",
        "S1 turns the test off and runs beside it",
        "claims no New Zealand calibration",
        "unpublishable diagnostic under the D2 publication block",
    ):
        assert phrase in entry["reason"], phrase


def test_the_d1982_stock_keeps_the_release_candidate_off():
    """A release candidate reading the interim stock needs a new ruling first.

    Every ``gates.battery@1`` node counts: the packaged graph's own and any
    an extension branch adds when the graph composes. A battery that omits
    the flag fails too, since only an explicit ``False`` keeps it off.
    """
    import importlib.util

    reason = _declaration()["index_family_factors"]["consumer_prices"]["reason"]
    assert "d1982" in reason
    packaged = {
        node["id"]: node["params"].get("release_candidate")
        for node in _load("transport_graph.json")["nodes"]
        if node["kernel"] == "gates.battery@1"
    }
    assert packaged == {"nz.gates.terminal": False}
    path = (
        paths_for("microcosm-build").tests
        / "engine_free/shared/test_nz_transport_declarations.py"
    )
    module_spec = importlib.util.spec_from_file_location("_nz_decl_d1982", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec, config = module._nz_composition_spec_and_config()
    graph = compose_module.compose_transport_graph(spec, config)
    composed = {
        node.id: node.params.get("release_candidate")
        for node in graph.nodes
        if node.kernel == "gates.battery@1"
    }
    assert packaged.keys() <= composed.keys()
    assert composed == dict.fromkeys(composed, False)


def test_one_run_carries_the_d1982_diagnostic_and_s1_beside_it():
    scenarios = _load("scenarios.json")
    assert scenarios["central_knobs"]["asset_test"] == "on"
    (s1,) = [row for row in scenarios["scenarios"] if row["id"] == "S1"]
    assert s1["knobs"] == {"asset_test": "off"}


def test_every_run_records_what_the_stock_and_the_bands_leave_open():
    pending = _load("transport_graph.json")["pending_outputs"]
    assert any(
        item.startswith("An all-household New Zealand liquid-asset distribution (C2)")
        and "unpublishable diagnostic beside S1" in item
        for item in pending
    )
    assert any(item.startswith("The C1 rent ageing ruling") for item in pending)
