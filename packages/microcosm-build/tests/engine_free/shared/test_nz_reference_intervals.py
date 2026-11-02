"""NZ IRD interval declarations: generated cuts, companions and the declared convention.

The row formulas below are NZ data, checked here against the packaged
resource; the shared interval module contains none of them.
"""

from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
COARSE = "ird_taxable_income_standin_ty2024"
AGE = "ird_taxable_income_age_ty2024"
IRD_STANDIN_REASON = (
    "G8-r2c: these 24 coarse rows need composition's proof (R2C-1, R2C-15) that "
    "exactly one node upstream of each calibration node, declaring measurement_id "
    "ird_taxable_income_ty2027_v1, writes the stand-in income and universe columns "
    "on person. Until that proof holds they stay parked and compilation refuses; "
    "nothing exempts them."
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


def _all_tables():
    return {
        table["id"]: table for table in _document("reference_intervals.json")["tables"]
    }


def _tables():
    """The two IRD taxable-income tables; the wage table has its own tests."""
    return {
        table["id"]: table
        for table in _document("reference_intervals.json")["tables"]
        if table["id"] in {NATIONAL, AGE}
    }


def _upper_dollars(band):
    return None if band["upper_cent"] is None else band["upper_cent"] / 100


def test_document_validates_under_the_shared_contract_with_declared_semantics():
    parsed = validate_interval_document(_document("reference_intervals.json"))
    # G8-r2c E appends the declared-only TY2026 wage table.
    # G8-r2c E appends the declared-only wage table, F the coarse stand-in table.
    assert list(parsed) == [NATIONAL, AGE, "ird_wage_salary_national_ty2026", COARSE]
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


def test_national_table_has_237_bands_and_no_companions_after_r2c4():
    """R2C-4: the national table stays declared and validated only, like the age table."""
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
    }
    for band in bands[1:]:
        row = int(band["id"].removeprefix("row_"))
        assert band["kind"] == "positive"
        assert _upper_dollars(band) == _national_upper(row)
        assert "count_reference" not in band and "amount_reference" not in band
    assert bands[-1]["source_label"] == "Over $1,000,000"
    profiles = _document("reference_profiles.json")
    assert NATIONAL not in json.dumps(profiles)


def test_coarse_table_has_12_bands_each_with_companions_and_support():
    table = _all_tables()[COARSE]
    assert (
        table["entity"],
        table["income_column"],
        table["universe_column"],
        table["positive_band_closure"],
        table["nil_band"],
        table["negative_income_policy"],
        table["first_lower_cut_cents"],
        table["diagnostics"]["support_household_entity"],
    ) == (
        "person",
        "ird_taxable_income_proxy_cents",
        "ird_taxable_income_proxy_universe_status",
        "right",
        "absent",
        "floor_zero",
        4_000_000,
        "household",
    )
    bands = table["bands"]
    assert len(bands) == 12
    assert [band["upper_cent"] for band in bands] == [
        dollars * 100 if dollars is not None else None
        for dollars in (
            50_000,
            60_000,
            70_000,
            80_000,
            100_000,
            125_000,
            150_000,
            200_000,
            300_000,
            500_000,
            1_000_000,
            None,
        )
    ]
    references = {
        row["name"]: row
        for row in _document("target_references.json")["target_references"]
    }
    profiles = _document("reference_profiles.json")
    for profile in ("low", "central", "high"):
        (transformation,) = profiles["dimensions"]["aging_index"]["profiles"][profile][
            "transformations"
        ]
        assert transformation["interval_table_ids"] == [COARSE]
        assert transformation["count_reference_names"] == [
            band["count_reference"] for band in bands
        ]
        assert transformation["reference_names"] == [
            band["amount_reference"] for band in bands
        ]
    for band in bands:
        assert band["kind"] == "positive"
        for name in (band["count_reference"], band["amount_reference"]):
            assert references[name]["filter"] == band["indicator"]
            assert references[name]["entity"] == "person"


@settings(max_examples=80, deadline=None)
@given(
    numerator=st.integers(min_value=1, max_value=10**7),
    denominator=st.integers(min_value=1, max_value=10**7),
)
def test_property_the_scaled_coarse_table_tiles_the_positive_range(
    numerator, denominator
):
    """For any positive factor the scaled coarse table tiles (f·$40k, ∞).

    Every whole cent at or below the scaled first cut is below-first-cut; each
    cent above it lands in exactly the band whose scaled (lower, upper] holds
    it, so the 12 bands leave no gap and no overlap.
    """
    from microcosm.build.reference_intervals import (
        BELOW_FIRST_CUT,
        scale_interval_table,
    )

    factor = Fraction(numerator, denominator)
    scaled = interval_table(
        scale_interval_table(
            _all_tables()[COARSE],
            {"numerator": factor.numerator, "denominator": factor.denominator},
            factor_id="hypothesis",
            factor_declaration_sha256="0" * 64,
            factor_receipt_sha256="0" * 64,
        )
    )
    cuts = scaled.cuts
    assert len(cuts) == 12 and all(b > a for a, b in zip(cuts, cuts[1:], strict=False))
    probes = sorted(
        {
            max(0, value)
            for cut in cuts
            for value in (math.floor(cut) - 1, math.floor(cut), math.floor(cut) + 1)
        }
    )
    positions = classify_records(scaled, probes, ["included"] * len(probes))
    for z, position in zip(probes, positions, strict=True):
        above = sum(cut < z for cut in cuts)
        if above == 0:
            assert position == BELOW_FIRST_CUT
        else:
            assert scaled.band_ids[position] == f"b{above:02d}"


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
    """The coarse table scales exactly: the central first cut is 4,468,436 cents."""
    _, intervals, receipt = resolve_reference_profiles(
        _document("target_references.json"),
        _document("reference_profiles.json"),
        {},
        _document("reference_intervals.json"),
        factor_context=_document("reference_aging.json"),
    )
    tables = {table["id"]: table for table in intervals["tables"]}
    coarse = tables[COARSE]
    first = coarse["first_lower_cut_cents"]
    assert Fraction(first["numerator"], first["denominator"]) == 4_468_436
    by_id = {band["id"]: band for band in coarse["bands"]}
    cuts = {
        band: Fraction(
            by_id[band]["upper_cent"]["numerator"],
            by_id[band]["upper_cent"]["denominator"],
        )
        for band in ("b01", "b06")
    }
    # $50,000 and $125,000 at the central factor 1.117109 (design 80, section 3).
    assert cuts == {"b01": Fraction(5_585_545), "b06": Fraction("13963862.5")}
    parsed = interval_table(coarse)
    positions = classify_records(
        parsed, [4_468_437, 13_963_862, 13_963_863], ["included"] * 3
    )
    assert [parsed.band_ids[index] for index in positions] == ["b01", "b06", "b07"]
    # The national and age tables stay unscaled.
    assert tables[NATIONAL] == _tables()[NATIONAL]
    assert tables[AGE] == _tables()[AGE]
    assert receipt["interval_tables"][COARSE]["factor_id"] == (
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
        assert len(ird) == 24
        assert all(
            row["metadata"]["activation_status"] == "requires_prerequisites"
            and row["metadata"]["parking_reason"] == IRD_STANDIN_REASON
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
    assert all(row["metadata"]["activation_status"] == "active" for row in wages)


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
