"""NZ IRD interval declarations: generated cuts, companions and the declared convention.

The row formulas below are NZ data, checked here against the packaged
resource; the shared interval module contains none of them.
"""

from __future__ import annotations

import json
import os
from fractions import Fraction
from pathlib import Path

import pytest

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.reference_intervals import (
    classify_records,
    interval_table,
    validate_interval_document,
)
from microcosm.build.reference_profiles import resolve_reference_profiles
from test_support.paths import paths_for

_SPEC = paths_for("microcosm-build").package / "src/microcosm/build/nz"
_REPOSITORY = paths_for("microcosm-build").repository
NATIONAL = "ird_taxable_income_national_ty2024"
AGE = "ird_taxable_income_age_ty2024"
IRD_REASON = (
    "B4: reviewed IRD-definition taxable-income measurement, annual component "
    "closure and record universe remain required. Ageing resolution does not "
    "establish that measurement."
)


def _document(name):
    return json.loads((_SPEC / name).read_text())


def _national_upper(row):
    if row == 9:
        return 100
    if row == 10:
        return 1_000
    if row <= 209:
        return 1_000 * (row - 9)
    if row <= 229:
        return 200_000 + 5_000 * (row - 209)
    if row <= 243:
        return 300_000 + 50_000 * (row - 229)
    return None


def _age_upper(row):
    if row == 7:
        return 100
    if row == 8:
        return 1_000
    if row <= 187:
        return 1_000 * (row - 7)
    return None


def _tables():
    return {
        table["id"]: table for table in _document("reference_intervals.json")["tables"]
    }


def _upper_dollars(band):
    return None if band["upper_cent"] is None else band["upper_cent"] / 100


def test_document_validates_under_the_shared_contract_with_declared_semantics():
    parsed = validate_interval_document(_document("reference_intervals.json"))
    assert list(parsed) == [NATIONAL, AGE]
    for table in _tables().values():
        assert {
            key: table[key]
            for key in (
                "entity",
                "currency",
                "input_unit",
                "income_column",
                "universe_column",
                "source_period",
                "target_period",
                "positive_band_closure",
                "nil_band",
                "negative_income_policy",
                "first_lower_cut_cents",
                "diagnostics",
            )
        } == {
            "entity": "person",
            "currency": "NZD",
            "input_unit": "integer_cent",
            "income_column": "ird_taxable_income_proxy_cents",
            "universe_column": "ird_taxable_income_proxy_universe_status",
            "source_period": 2024,
            "target_period": 2027,
            "positive_band_closure": "right",
            "nil_band": "present",
            "negative_income_policy": "floor_zero",
            "first_lower_cut_cents": 0,
            "diagnostics": {
                "boundary_window_cents": 100,
                "alternative_positive_band_closure": "left",
            },
        }
        assert "scaling" not in table


def test_national_table_has_237_count_money_interval_triples_and_no_total_row():
    table = _tables()[NATIONAL]
    assert table["source_locator"] == "Income tables 2001-24!BE8:BE244"
    bands = table["bands"]
    assert [band["id"] for band in bands] == [f"row_{row}" for row in range(8, 245)]
    assert not any("246" in json.dumps(band) for band in bands)
    assert bands[0] == {
        "id": "row_8",
        "kind": "nil",
        "indicator": "taxable_income_band_row_8_indicator",
        "source_label": "nil",
        "count_reference": "ird_taxable_income_people_by_band__row_8",
        "amount_reference": "ird_taxable_income_total_by_band__row_8",
    }
    for band in bands[1:]:
        row = int(band["id"].removeprefix("row_"))
        assert band["kind"] == "positive"
        assert _upper_dollars(band) == _national_upper(row)
    assert bands[-1]["source_label"] == "Over $1,000,000"
    references = {
        row["name"]: row
        for row in _document("target_references.json")["target_references"]
    }
    profiles = _document("reference_profiles.json")
    for profile in ("low", "central", "high"):
        (transformation,) = profiles["dimensions"]["aging_index"]["profiles"][profile][
            "transformations"
        ]
        assert transformation["interval_table_ids"] == [NATIONAL]
        assert transformation["count_reference_names"] == [
            band["count_reference"] for band in bands
        ]
        assert transformation["reference_names"] == [
            band["amount_reference"] for band in bands
        ]
    for band in bands:
        for name in (band["count_reference"], band["amount_reference"]):
            assert references[name]["filter"] == band["indicator"]
            assert references[name]["entity"] == "person"


def test_age_table_has_183_bands_and_no_companions_or_fit_rows():
    table = _tables()[AGE]
    assert table["source_locator"] == "A6:A188"
    bands = table["bands"]
    assert [band["id"] for band in bands] == [f"row_{row}" for row in range(6, 189)]
    assert bands[0]["kind"] == "nil"
    for band in bands[1:]:
        assert _upper_dollars(band) == _age_upper(int(band["id"].removeprefix("row_")))
    assert bands[-1]["source_label"] == "Over $180,000"
    assert all(
        "count_reference" not in band and "amount_reference" not in band
        for band in bands
    )
    indicators = {band["indicator"] for band in bands}
    for name in ("target_references.json", "precal_references.json"):
        assert not any(
            row.get("filter") in indicators
            for row in _document(name)["target_references"]
        )
    profiles = json.dumps(_document("reference_profiles.json"))
    assert AGE not in profiles


@pytest.mark.parametrize(
    ("table", "rows", "endpoint"),
    [
        (NATIONAL, (29, 30), 20_000),
        (NATIONAL, (59, 60), 50_000),
        (NATIONAL, (89, 90), 80_000),
        (NATIONAL, (229, 230), 300_000),
        (AGE, (27, 28), 20_000),
        (AGE, (57, 58), 50_000),
        (AGE, (87, 88), 80_000),
    ],
)
def test_shared_endpoint_labels_resolve_to_the_ordered_cuts(table, rows, endpoint):
    declaration = _tables()[table]
    by_id = {band["id"]: band for band in declaration["bands"]}
    lower, upper = (by_id[f"row_{row}"] for row in rows)
    label = f"${endpoint:,}"
    # Both published labels name the endpoint; the ordered cut gives it to the
    # lower band under the declared right-closed convention.
    assert lower["source_label"].replace(" ", "").endswith(label)
    assert upper["source_label"].replace(" ", "").startswith(label)
    assert lower["upper_cent"] == endpoint * 100
    parsed = interval_table(declaration)
    positions = classify_records(
        parsed,
        [endpoint * 100 - 1, endpoint * 100, endpoint * 100 + 1],
        ["included"] * 3,
    )
    assert [parsed.band_ids[index] for index in positions] == [
        lower["id"],
        lower["id"],
        upper["id"],
    ]


def test_central_ageing_keeps_fractional_cent_cuts_exactly():
    _, intervals, receipt = resolve_reference_profiles(
        _document("target_references.json"),
        _document("reference_profiles.json"),
        {},
        _document("reference_intervals.json"),
        factor_context=_document("reference_aging.json"),
    )
    tables = {table["id"]: table for table in intervals["tables"]}
    national = tables[NATIONAL]
    by_id = {band["id"]: band for band in national["bands"]}
    cuts = {
        row: Fraction(
            by_id[f"row_{row}"]["upper_cent"]["numerator"],
            by_id[f"row_{row}"]["upper_cent"]["denominator"],
        )
        for row in (9, 10, 49)
    }
    # $100, $1,000 and $40,000 at the central factor 1.117109.
    assert cuts == {
        9: Fraction("11171.09"),
        10: Fraction("111710.9"),
        49: Fraction(4_468_436),
    }
    parsed = interval_table(national)
    positions = classify_records(
        parsed, [11_171, 11_172, 111_710, 111_711], ["included"] * 4
    )
    assert [parsed.band_ids[index] for index in positions] == [
        "row_9",
        "row_10",
        "row_10",
        "row_11",
    ]
    assert tables[AGE] == _tables()[AGE]
    assert receipt["interval_tables"][NATIONAL]["factor_id"] == (
        "ird_ty2024_to_ty2027_central"
    )


def test_resolution_is_not_activation_and_no_active_row_requests_the_proxy():
    intervals = _document("reference_intervals.json")
    proxy = {
        "ird_taxable_income_proxy_cents",
        "ird_taxable_income_proxy_universe_status",
    }
    indicators = {
        band["indicator"] for table in intervals["tables"] for band in table["bands"]
    }
    for profile in ("low", "central", "high"):
        effective, _, _ = resolve_reference_profiles(
            _document("target_references.json"),
            _document("reference_profiles.json"),
            {"aging_index": profile},
            intervals,
            factor_context=_document("reference_aging.json"),
        )
        ird = [
            row
            for row in effective["target_references"]
            if row["name"].startswith("ird_taxable_income_")
        ]
        assert len(ird) == 474
        assert all(
            row["metadata"]["activation_status"] == "requires_prerequisites"
            and row["metadata"]["parking_reason"] == IRD_REASON
            for row in ird
        )
        for row in effective["target_references"]:
            if row["metadata"].get("activation_status", "active") == "active":
                assert row.get("filter") not in indicators
                assert row.get("measure") not in proxy
    bindings = json.dumps(_document("calibration_bindings.json"))
    assert not any(name in bindings for name in proxy | indicators)
    precal = _document("precal_references.json")["target_references"]
    wages = [row for row in precal if row["name"].startswith("ird_wage_salary_")]
    assert len(wages) == 235
    assert all(
        row["metadata"]["activation_status"] == "requires_g8r2c_wage_quantile_producer"
        for row in wages
    )


def test_declared_convention_is_disclosed_in_the_resource_and_the_docs():
    disclosure = _document("reference_intervals.json")["disclosure"]
    for phrase in (
        "IRD publishes no band closure or cents semantics",
        "Microcosm's declared",
        "convention",
        "publisher reclassification is unavailable",
    ):
        assert phrase in disclosure
    docs = (_REPOSITORY / "docs/nz-calibration-targets.md").read_text()
    assert "IRD publishes no band closure or cents semantics" in docs
    assert "declared convention" in docs


@pytest.fixture(scope="module")
def artifact():
    path = os.environ.get("NZ_CHRONICLE_ARTIFACT")
    if not path:
        pytest.skip("NZ_CHRONICLE_ARTIFACT is unset; public NZ feed audit unavailable")
    return load_ledger_consumer_artifact(
        Path(path), feed_pin=_document("chronicle_feed.json")
    )


def test_every_band_label_is_the_feed_label_for_its_row(artifact):
    record_sets = {
        NATIONAL: "ird_taxable_income_distribution.ty2024.income_bands",
        AGE: "ird_taxable_income_distribution.ty2024.age_by_income_band",
    }
    for table_id, record_set in record_sets.items():
        labels = {}
        for fact in artifact.facts:
            layout = fact.get("layout", {})
            if (
                layout.get("record_set_id") == record_set
                and layout.get("table_record_kind") == "detail"
            ):
                labels.setdefault(layout["source_row_id"], set()).add(
                    fact["dimensions"]["taxable_income_band_source_label"]
                )
        bands = _tables()[table_id]["bands"]
        assert set(labels) == {band["id"] for band in bands}
        for band in bands:
            assert labels[band["id"]] == {band["source_label"]}
