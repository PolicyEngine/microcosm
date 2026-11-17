"""Belgium's proposed donor money normalization (B4b, proposed pending d1902).

The declaration is generated spec data (tools/generate_be_target_references.py);
no Belgian graph reads it and Belgium stays inactive. These checks pin its routes
against d1902 items 18 to 20, the concept schema and the columns the shared US
mapping can decode from the donor; tie the feed-selected rate to the donor
period, its unit, the reviewed fact and the pinned feed (the shared validator
checks none of these); keep every family an interim hold-fixed proposal rather
than a ruled rule; and keep every applied factor fit for the declared target
period. The feed audit needs the Chronicle ``be`` consumer artifact: set
``BE_CHRONICLE_ARTIFACT``.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.ledger_artifact import (
    chronicle_feed_identity,
    load_ledger_consumer_artifact,
)
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
from microcosm.build.transport.compose import (
    _money_producer_gaps,
    load_transport_spec,
    validate_transport_activation,
)
from microcosm.frame.adapters.policyengine_us_concepts import (
    POLICYENGINE_US_CONCEPT_MAPPING,
)
from microcosm.frame.concepts import TransportRule
from test_support.paths import paths_for
from tools import generate_be_target_references as generator

_SPEC = paths_for("microcosm-build").package / "src/microcosm/build/be"
_NZ = paths_for("microcosm-build").package / "src/microcosm/build/nz"
_NAME = "money_normalization.json"
_RATE = Fraction(1353, 1250)
# Every identity reason opens with this. Hold-fixed is this declaration's interim
# proposal, for new d1902 items; d1902 item 20 states no hold-fixed rule.
_HOLD_FIXED = (
    "proposed_pending_d1902, interim hold-fixed rule proposed as a new d1902 item: "
)
# "item 20's explicit hold-fixed rule" or "d1902 item 20: ... hold-fixed rule".
_ITEM_20_RULE = re.compile(r"item 20(?:'s|:)[^.]*hold-fixed")
_NORMALIZED = {
    ("person", "nonfarm_self_employment_income"): "mixed_income",
    ("person", "farm_self_employment_income"): "mixed_income",
    ("person", "interest_income"): "capital_income",
    ("person", "dividend_income"): "capital_income",
    ("person", "rental_income"): "capital_income",
    ("person", "realized_capital_gains"): "capital_income",
    ("person", "private_pension_income"): "pensions",
    ("person", "liquid_financial_assets"): "consumer_prices",
    ("household", "rent"): "rents",
    ("household", "mortgage_interest"): "owner_housing_costs",
    ("household", "property_tax"): "owner_housing_costs",
}
_QUANTILE_MAPPED = {("person", "employment_income")}
# CREATE never defines it for this donor (see the decoder test below).
_PENDING = {("household", "mortgage_principal")}
# read_populace_us_donor stages it from bank, stock and bond assets; no decoder.
_STAGED = {("person", "liquid_financial_assets")}
_DROP = object()


def _load(name, root=None):
    return json.loads(((root or _SPEC) / name).read_text())


def _declaration():
    return validate_money_normalization(_load(_NAME))


def _year(year):
    return {"start_date": f"{year}-01-01", "end_date": f"{year}-12-31"}


def _same(left, right):
    """Equal with JSON types: 2024, 2024.0 and "2024" differ."""

    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _set(document, path, value):
    node = document
    for key in path[:-1]:
        node = node[key]
    if value is _DROP:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return document


def _rate_violations(document, feed_pin):
    """What validate_money_normalization leaves unchecked about a feed rate."""

    declaration = validate_money_normalization(copy.deepcopy(document))
    rate, period = declaration["rate"], declaration["donor_money_period"]["value"]
    if rate["form"] != "feed_selector":
        return ["form"]
    selector, validation = rate["selector"], rate["validation"]
    pair = (declaration["from_currency"].lower(), declaration["to_currency"].lower())
    unit = {
        "from_per_to": f"{pair[0]}_per_{pair[1]}",
        "to_per_from": f"{pair[1]}_per_{pair[0]}",
    }[declaration["quote"]]
    checks = {
        "rate_period": _same(
            (selector.get("period_type"), selector.get("period_value")),
            ("calendar_year", period),
        ),
        "rate_coverage": validation.get("period_coverage") == _year(period),
        "rate_fact": (selector.get("record_set_id"), selector.get("aggregate_fact_key"))
        == (
            generator.MONEY_RATE["record_set_id"],
            generator.MONEY_RATE["aggregate_fact_key"],
        ),
        # The whole reviewed identity, so CI binds the rate without the artifact.
        "rate_selector": _same(selector, generator.money_rate_selector()),
        "rate_validation": _same(validation, generator.money_rate_validation()),
        "quote_direction": validation.get("unit") == unit,
        "assertion": rate["assertion_policy"] == "observed_only"
        and selector.get("assertion") == validation.get("assertion") == "observation",
        "feed_identity": rate["feed_identity"] == chronicle_feed_identity(feed_pin),
        "expected_decimal": rate["expected_decimal"] == generator.MONEY_RATE["token"],
    }
    return sorted(name for name, ok in checks.items() if not ok)


def _selected_ids(declaration):
    profile = declaration["factor_profile"]
    selections = (
        [{}]
        if profile is None
        else [{profile["dimension"]: name} for name in profile["profiles"]]
    )
    return sorted(
        {
            factor_id
            for selection in selections
            for factor_id in selected_money_factors(declaration, selection)[1].values()
            if factor_id is not None
        }
    )


def _factor_violations(document, context):
    """Every selectable factor exists and ages the donor year to the target year.

    check_money_factor compares only the target label; this also checks the
    target schedule's own coverage.
    """

    declaration = validate_money_normalization(copy.deepcopy(document))
    try:
        entries = validate_factor_context(context, _selected_ids(declaration))
    except ValueError:
        return [("missing", None)]
    return [
        (kind, factor_id)
        for factor_id, entry in entries.items()
        for kind in _entry_violations(entry, declaration)
    ]


def _entry_violations(entry, declaration):
    try:
        check_money_factor(entry, declaration)
    except ValueError:
        return ["unfit"]
    target = entry["declaration"]["target_schedule"]
    covered = {
        "start_date": target[0]["validation"]["period_coverage"]["start_date"],
        "end_date": target[-1]["validation"]["period_coverage"]["end_date"],
    }
    return [] if covered == _year(declaration["target_period"]) else ["target_coverage"]


def test_routes_follow_d1902_items_18_to_20_the_schema_and_the_donor():
    declaration = _declaration()
    assert (declaration["from_currency"], declaration["to_currency"]) == ("USD", "EUR")
    assert declaration["quote"] == "from_per_to"
    assert declaration["rounding"] == {"digits": 2, "mode": "half_even"}
    assert declaration["cents_suffix"] == "_cents"
    # BE-MC3 converts every money column; item 18 converts every unmapped one.
    # Self-employment converts under an interim identity (mixed_income below).
    assert normalized_column_families(declaration) == _NORMALIZED
    rosters = money_column_rosters(declaration)
    # Item 18: wages are mapped, never converted.
    assert {k for k, (route, _) in rosters.items() if route == "quantile_mapped"} == (
        _QUANTILE_MAPPED
    )
    assert {k for k, (route, _) in rosters.items() if route == "pending"} == _PENDING
    assert (len(_NORMALIZED), len(_QUANTILE_MAPPED), len(_PENDING)) == (11, 1, 1)
    assert "earnings" not in declaration["index_family_factors"]


def test_every_money_concept_not_dropped_at_create_is_routed_and_drop_is_not():
    # The concepts CREATE does not drop, not the ones it defines: mortgage
    # principal is among the 13, and CREATE never defines it for this donor
    # (see the decoder test below).
    rosters = money_column_rosters(_declaration())
    not_dropped = {
        key
        for key, item in money_concept_columns().items()
        if item.transport is not TransportRule.DROP
    }
    assert set(rosters) == not_dropped
    assert len(not_dropped) == 13
    assert ("person", "public_pension_income") not in rosters


def test_every_normalized_column_is_one_create_can_define():
    # A money node must rewrite every normalized column, and a rewrite of a
    # column no node defines is refused when the graph is built. CREATE defines
    # only the concepts the shared US mapping decodes (or the reader stages).
    undecodable = {
        key
        for key, item in money_concept_columns().items()
        if item.transport is not TransportRule.DROP
        and POLICYENGINE_US_CONCEPT_MAPPING._decoder(item) is None
    }
    assert undecodable - _STAGED == _PENDING
    assert not set(normalized_column_families(_declaration())) & (undecodable - _STAGED)


def test_every_family_is_an_interim_hold_fixed_proposal_not_a_ruled_rule():
    declaration = _declaration()
    assert declaration["factor_profile"] is None
    families = declaration["index_family_factors"]
    assert set(families) == set(_NORMALIZED.values())
    for family, entry in families.items():
        reason = entry["reason"]
        assert entry["identity"] is True, family
        assert reason.startswith(_HOLD_FIXED), family
        assert "interim" in reason, family
        assert "2024 levels in the 2026 policy year" in reason, family
        # Item 20 names no hold-fixed rule; no reason may attribute one to it.
        assert not _ITEM_20_RULE.search(reason), family
    # Self-employment's identity is forced by the shared gap; item 20's
    # factor replaces it.
    mixed = families["mixed_income"]["reason"]
    assert "d1902 item 20 proposes" in mixed
    assert "replaces it" in mixed


def test_every_disclosure_states_proposed_pending_d1902_and_its_item():
    document = _load(_NAME)
    texts = [document["description"], document["status"], document["notes"]]
    texts += [
        row["reason"]
        for roster in ("quantile_mapped", "pending")
        for rows in document[roster].values()
        for row in rows.values()
    ]
    texts += [
        entry.get("reason", entry.get("disclosure"))
        for entry in document["index_family_factors"].values()
    ]
    for text in texts:
        assert "proposed_pending_d1902" in text, text
        assert "new d1902 item" in text or any(
            f"d1902 item {item}" in text or f"items {item}" in text
            for item in (18, 19, 20)
        ), text
    assert "quantile_surfaces.json" not in (_SPEC / _NAME).read_text()


def test_the_rate_is_the_reviewed_eurostat_2024_fact_tied_to_the_donor_and_feed():
    document = _load(_NAME)
    assert _rate_violations(document, _load("chronicle_feed.json")) == []
    assert (
        document["rate"]["feed_identity"]
        == _load("reference_aging.json")["feed_identity"]
    )


@pytest.mark.parametrize(
    ("path", "value", "kind"),
    [
        (("quote",), "to_per_from", "quote_direction"),
        (("rate", "validation", "unit"), "eur_per_usd", "quote_direction"),
        (("rate", "selector", "period_value"), 2025, "rate_period"),
        (("rate", "selector", "period_value"), "2024", "rate_period"),
        (("rate", "validation", "period_coverage"), _year(2025), "rate_coverage"),
        (("rate", "validation"), {}, "rate_validation"),
        (("rate", "validation", "source_vintage"), "x", "rate_validation"),
        (("rate", "validation", "source_sha256"), "0" * 64, "rate_validation"),
        (("rate", "selector", "dimension_values", "statinfo"), "EOP", "rate_selector"),
        (
            ("rate", "selector", "geography_id"),
            "eurostat:ert_bil_eur_a:GBP",
            "rate_selector",
        ),
        (("rate", "selector", "assertion"), _DROP, "assertion"),
        (("rate", "assertion_policy"), "allow_source_projection", "assertion"),
        (("rate", "feed_identity", "fact_row_count"), 87111, "feed_identity"),
        (("rate", "expected_decimal"), "1.0825", "expected_decimal"),
        (("rate", "expected_decimal"), "1.08240", "expected_decimal"),
        (
            ("rate", "selector", "aggregate_fact_key"),
            "ledger.aggregate_fact.v2:000000000000000000000000",
            "rate_fact",
        ),
    ],
)
def test_rate_mutants_the_shared_validator_accepts_are_caught(path, value, kind):
    document = _set(_load(_NAME), path, value)
    validate_money_normalization(copy.deepcopy(document))
    assert kind in _rate_violations(document, _load("chronicle_feed.json"))


# The rate's token with any trailing zeros; its inverse as a decimal or a fraction.
_RATE_TOKEN = re.compile(r"(?<![\d.])1\.08240*(?!\d)")
_INVERSE = re.compile(r"0\.9238|1250\s*/\s*1353")


def test_the_rate_has_one_location_in_the_package():
    def walk(value, where=()):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from walk(child, (*where, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from walk(child, (*where, index))
        elif (isinstance(value, Decimal) and value == Decimal("1.0824")) or (
            isinstance(value, str) and "1.0824" in value
        ):
            yield where

    found = [
        (path.name, where)
        for path in sorted(_SPEC.rglob("*.json"))
        for where in walk(json.loads(path.read_text(), parse_float=Decimal))
    ]
    assert found == [(_NAME, ("rate", "expected_decimal"))]
    # Every file, YAML and text included: the rate once, its inverse nowhere.
    text = {
        str(path.relative_to(_SPEC)): path.read_text(errors="replace")
        for path in sorted(_SPEC.rglob("*"))
        if path.is_file()
    }
    assert {
        name: len(_RATE_TOKEN.findall(body))
        for name, body in text.items()
        if _RATE_TOKEN.search(body)
    } == {_NAME: 1}
    assert not [name for name, body in text.items() if _INVERSE.search(body)]


# Item 18 and this declaration replace the currency-bridge method, and with it
# the claim that B4 converts every donor input a rules row reads (wages are
# mapped, never converted). Any run of spaces, line breaks, underscores or
# hyphens may sit between "currency" and "bridge", and any whitespace between
# the words of the old RULES_REASON wording, because YAML and generated text wrap.
_RETIRED = (
    re.compile(r"currency[\s_-]+bridge", re.IGNORECASE),
    re.compile(r"euro\s+conversion\s+of\s+the\s+donor\s+inputs", re.IGNORECASE),
)


def test_no_package_text_names_the_retired_currency_bridge():
    text = {
        str(path.relative_to(_SPEC)): path.read_text(errors="replace")
        for path in sorted(_SPEC.rglob("*"))
        if path.is_file()
    }
    assert not [
        (name, pattern.pattern)
        for name, body in text.items()
        for pattern in _RETIRED
        if pattern.search(body)
    ]


def test_money_ages_to_the_bundle_policy_year():
    bundle = yaml.safe_load((_SPEC / "spec/bundle.yaml").read_text())
    assert (
        _declaration()["target_period"]
        == bundle["dataset_run"]["target_period"]
        == generator.POLICY_YEAR
        == 2026
    )


def test_the_donor_money_period_is_declared_for_the_pinned_donor():
    declaration, nz = _declaration(), validate_money_normalization(_load(_NAME, _NZ))
    period, nz_period = declaration["donor_money_period"], nz["donor_money_period"]
    assert period["value"] == generator.DONOR_MONEY_YEAR
    # Belgium pins exactly one donor microdata artifact, in any stage. A repin
    # of it, alone or jointly with NZ, fails here until the money period is
    # re-declared.
    assert _microdata_pins(_SPEC) == [generator.DONOR_MONEY_ARTIFACT]
    donor = generator.DONOR_MONEY_ARTIFACT
    # Belgium's own evidence names that pin, so re-declaring the pin means
    # rewriting the evidence too, whatever NZ pins.
    assert donor["sha256"] in period["evidence"][2]
    assert f"{donor['size_bytes']:,} bytes" in period["evidence"][2]
    # While NZ pins the same donor, in whichever stage or position, Belgium
    # declares NZ's period, and NZ's test checks these two evidence strings
    # against the code they cite. NZ pinning it twice is ambiguous and fails.
    # A repin of NZ alone does not tie Belgium to it.
    matches = [pin for pin in _microdata_pins(_NZ) if pin == donor]
    assert len(matches) <= 1, f"NZ pins Belgium's donor {len(matches)} times"
    if matches:
        assert {k: period[k] for k in ("type", "value", "basis")} == {
            k: nz_period[k] for k in ("type", "value", "basis")
        }
        assert period["evidence"][:2] == nz_period["evidence"]


def _microdata_pins(root):
    return [
        {k: artifact[k] for k in ("sha256", "size_bytes")}
        for stage in _load("source_stages.json", root)["stages"]
        for artifact in stage.get("artifacts", ())
        if artifact["kind"] == "public_microdata"
    ]


def test_no_factor_is_selected_until_a_shared_operation_admits_one():
    # Every family is an identity today, so the factor checks below have
    # nothing to check on the committed document. This fails loudly when a
    # family binds a factor, so that change also carries its own review.
    assert _selected_ids(_declaration()) == []
    assert _factor_violations(_load(_NAME), _load("reference_aging.json")) == []


def _bound(factor_id):
    """A fixed fixture: only self-employment normalized, bound to factor_id.

    Built from the committed periods and rate, not its routes, so a route edit
    in the committed document cannot break these factor checks.
    """

    pending = {}
    for (entity, column), item in sorted(money_concept_columns().items()):
        if (
            item.transport is not TransportRule.DROP
            and item.monetary.index_family.value != "mixed_income"
        ):
            pending.setdefault(entity, {})[column] = {
                "awaits": ["fixture"],
                "reason": "fixture",
            }
    return {
        **_load(_NAME),
        "columns": {
            "person": [
                "nonfarm_self_employment_income",
                "farm_self_employment_income",
            ]
        },
        "factor_profile": None,
        "index_family_factors": {
            "mixed_income": {"factor_id": factor_id, "disclosure": "fixture"}
        },
        "quantile_mapped": {},
        "pending": pending,
    }


def _factor_entry(factor_id):
    return next(
        entry
        for entry in _load("reference_aging.json")["factors"]
        if entry["declaration"]["id"] == factor_id
    )


_RSZ_2024 = "rsz_ordinary_remuneration_cy2024_to_cy2025_observed"


@pytest.mark.parametrize(
    ("factor_id", "kind"),
    [("no_such_factor", "missing"), (_RSZ_2024, "unfit")],
)
def test_factor_mutants_are_caught(factor_id, kind):
    document = _bound(factor_id)
    validate_money_normalization(copy.deepcopy(document))
    found = _factor_violations(document, _load("reference_aging.json"))
    assert kind in [name for name, _ in found]


def test_a_factor_labelled_2026_but_measuring_2025_is_caught():
    # resolve_reference_factor never compares the label with the schedules, so
    # an authored declaration can say 2026 over a 2025 target slot; the shared
    # check_money_factor reads only the label and accepts it.
    entry = copy.deepcopy(_factor_entry(_RSZ_2024))
    entry["declaration"]["purpose"] = "money_normalization_input"
    entry["declaration"]["target_period"] = 2026
    declaration = validate_money_normalization(_bound(_RSZ_2024))
    assert check_money_factor(entry, declaration) == (1032192, 6)
    assert _entry_violations(entry, declaration) == ["target_coverage"]


def test_wages_await_the_rsz_surface_rows_that_exist():
    (awaits,) = [
        row["awaits"] for row in _declaration()["quantile_mapped"]["person"].values()
    ]
    wage_rows = {
        row["name"]
        for row in _load("precal_references.json")["target_references"]
        if row["name"].startswith(f"{generator.WAGE_CONTRACT}__")
    }
    assert len(wage_rows) == 5
    assert all(f"{generator.WAGE_CONTRACT}__*" in text for text in awaits)


def test_the_declaration_adds_no_gap_and_belgium_stays_inactive():
    spec = load_transport_spec(_SPEC)
    assert "money_normalization" in spec["resources"]
    assert _money_producer_gaps(spec) == []
    with pytest.raises(ValueError, match="not activated") as error:
        validate_transport_activation(spec)
    assert "money" not in str(error.value)


def test_an_invalid_declaration_becomes_a_named_activation_gap(tmp_path):
    package = tmp_path / "be"
    shutil.copytree(_SPEC, package)
    document = _load(_NAME)
    # Invalid whatever the routes, so no route edit makes this test pass or fail.
    document["schema_version"] = 2
    (package / _NAME).write_text(json.dumps(document))
    with pytest.raises(ValueError, match="not activated") as error:
        validate_transport_activation(load_transport_spec(package))
    assert f"{_NAME}: " in str(error.value)
    assert "requires schema_version 1" in str(error.value)


_AMOUNT = st.floats(
    min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False
)


@settings(max_examples=200, deadline=None, database=None)
@given(rents=st.lists(_AMOUNT.map(abs), min_size=1, max_size=20), data=st.data())
def test_property_held_columns_convert_once_and_round_half_even(rents, data):
    rental = data.draw(st.lists(_AMOUNT, min_size=len(rents), max_size=len(rents)))
    profit = data.draw(st.lists(_AMOUNT, min_size=len(rents), max_size=len(rents)))
    declaration = _declaration()
    rate = Fraction(declaration["rate"]["expected_decimal"])
    arrays, receipts = normalize_money_tables(
        {
            "person": pd.DataFrame(
                {"rental_income": rental, "nonfarm_self_employment_income": profit},
                dtype="float64",
            ),
            "household": pd.DataFrame({"rent": rents}, dtype="float64"),
        },
        {
            **declaration,
            "columns": {
                "person": ["rental_income", "nonfarm_self_employment_income"],
                "household": ["rent"],
            },
        },
        rate=rate,
        rate_text=declaration["rate"]["expected_decimal"],
        # Fixed identities, so a family edit in the committed document cannot
        # break this arithmetic check; the family test above covers the entries.
        factors={
            family: {"identity": True, "reason": "fixture"}
            for family in ("capital_income", "mixed_income", "rents")
        },
    )
    for (entity, column), values in (
        (("person", "rental_income"), rental),
        (("person", "nonfarm_self_employment_income"), profit),
        (("household", "rent"), rents),
    ):
        expected = []
        for value in values:
            exact = Fraction(repr(value)) * 100 * Fraction(1250, 1353)
            floor = exact.numerator // exact.denominator
            rest = exact - floor
            expected.append(
                floor
                + (rest > Fraction(1, 2) or (rest == Fraction(1, 2) and floor % 2))
            )
        assert arrays[entity, f"{column}_cents"].tolist() == expected
    assert {receipt["factor"]["identity"] for receipt in receipts} == {True}


@pytest.fixture(scope="module")
def artifact():
    path = os.environ.get("BE_CHRONICLE_ARTIFACT")
    if not path:
        pytest.skip("BE_CHRONICLE_ARTIFACT is unset; Belgian feed audit unavailable")
    return load_ledger_consumer_artifact(
        Path(path), feed_pin=_load("chronicle_feed.json"), lexical_floats=True
    )


def test_the_rate_replays_exactly_from_the_pinned_feed(artifact):
    rate, evidence = resolve_money_rate(
        _declaration(), artifact.facts, _load("chronicle_feed.json")
    )
    assert rate == _RATE
    assert Fraction(1) / rate == Fraction(1250, 1353)
    assert (
        evidence["operand"]["aggregate_fact_key"]
        == generator.MONEY_RATE["aggregate_fact_key"]
    )


@pytest.mark.parametrize("field", sorted(generator.MONEY_RATE))
def test_a_changed_reviewed_rate_refuses_generation(artifact, field):
    rate = {**generator.MONEY_RATE, field: "changed"}
    with (
        mock.patch.object(generator, "MONEY_RATE", rate),
        pytest.raises(ValueError, match="Reviewed money rate fact changed"),
    ):
        generator.build_reference_documents(artifact)


def test_a_changed_donor_year_refuses_generation(artifact):
    # The reviewed selector's period and the validation's coverage both come
    # from DONOR_MONEY_YEAR, so a re-declared donor year needs a re-reviewed rate.
    with (
        mock.patch.object(generator, "DONOR_MONEY_YEAR", 2025),
        pytest.raises(ValueError, match="Reviewed money rate fact changed"),
    ):
        generator.build_reference_documents(artifact)
