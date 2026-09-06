"""The developmental CPI-U price-restatement baseline, on invented bytes only.

No PUF is read, no artifact store is touched and no country model is imported.
The delivery, the raw definition and the projection document are all built
from literals here and travel the explicit ``test_fixture`` route. The CPI
levels are the real published annual averages, because the point of the
developmental authority is that a public resource may say so about itself; the
resource digest and URL below are invented placeholders for the pin a genuine
run supplies.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.us_runtime import puf_price_baseline as baseline
from microcosm.build.us_runtime.puf_growth import (
    PROVENANCE_RECID_COLUMN,
    PROVENANCE_SOURCE_AGI_COLUMN,
    FactorAuthority,
    GrowthFactorTable,
    GrowthKind,
    PufGrowthRefusalError,
    SignBranch,
    apply_puf_design_weight_growth,
    apply_puf_growth,
    compile_puf_growth,
    puf_growth_monetary_metadata,
)
from microcosm.build.us_runtime.puf_monetary_agi_projection import (
    AGI_FIELD,
    fixture_agi_projection_document,
    packaged_agi_projection,
)
from microcosm.build.us_runtime.puf_monetary_source import (
    AGGREGATE_SENTINEL,
    PufMonetaryRefusalError,
    PufMonetarySourceProjection,
    decode_puf_monetary_source,
    projection_document_json,
)
from microcosm.build.us_runtime.puf_price_baseline import (
    DESIGN_WEIGHT_IDENTITY_SERIES,
    PAIRWISE_OBSERVATIONS,
    PRICE_BASELINE_LEVELS,
    PRICE_BASELINE_RATIO,
    PRICE_BASELINE_SERIES_NAME,
    PRICE_BASELINE_SOURCE_YEAR,
    PRICE_BASELINE_TARGET_YEAR,
    adapt_projection_to_money_view,
    compile_price_baseline,
    price_baseline_factor_document,
    price_baseline_invariants,
    price_baseline_recipe,
)
from microcosm.build.us_runtime.puf_raw_source import (
    PUF_AGGREGATE_RECIDS,
    decode_puf_raw_source,
    definition_document_json,
    fixture_definition,
    packaged_definition,
)

PACKAGED_DEFINITION = packaged_definition()
PACKAGED_THIRTEEN = packaged_agi_projection()
MAIN_HEADER = list(PACKAGED_DEFINITION.main.delivered_header)
DEMOGRAPHIC_HEADER = list(PACKAGED_DEFINITION.demographic.delivered_header)

#: Invented stand-ins for the pin a genuine run supplies. The URL is a
#: reserved non-resolvable domain; nothing here retrieved anything.
FIXTURE_RESOURCE = {
    "source_url": "https://example.invalid/pub/time.series/cu/cu.data.1.AllItems",
    "resource_sha256": "a" * 64,
    "retrieved_utc": "2026-09-06T00:00:00+00:00",
}


def header_digest(header) -> str:
    payload = json.dumps(
        list(header),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    ).hexdigest()


def csv_bytes(header, records) -> bytes:
    lines = [",".join(header)]
    lines.extend(",".join(record) for record in records)
    return ("\n".join(lines) + "\n").encode("utf-8")


def main_record(*, recid, mars="1", s006="12345", flpdyr="2015", amounts=None):
    values = {
        "RECID": recid,
        "MARS": mars,
        "DSI": "0",
        "S006": s006,
        "FLPDYR": flpdyr,
        "FLPDMO": "12",
        **(amounts or {}),
    }
    return [values.get(name, "0") for name in MAIN_HEADER]


def demographic_record(*, recid):
    values = {
        "RECID": recid,
        "AGEDP1": "0",
        "AGEDP2": "0",
        "AGEDP3": "0",
        "AGERANGE": "2",
        "EARNSPLIT": "0",
        "GENDER": "1",
    }
    return [values[name] for name in DEMOGRAPHIC_HEADER]


#: Four invented returns and one publisher disclosure aggregate. Row 14 has a
#: qualified dividend above its ordinary dividend, so the pairwise observation
#: has a nonzero violation count to preserve rather than a vacuous zero.
DEFAULT_MAIN = [
    main_record(
        recid="11",
        s006="12345",
        flpdyr="2015",
        amounts={
            "E00100": "71000",
            "E00200": "50000",
            "E00300": "125",
            "E00400": "40",
            "E00600": "900",
            "E00650": "700",
            "E00900": "12000",
            "E01000": "3400",
            "E01500": "8000",
            "E01700": "6400",
            "E02100": "150",
            "P22250": "1200",
            "P23250": "2200",
        },
    ),
    main_record(
        recid="12",
        mars="2",
        s006="9900",
        flpdyr="2013",
        amounts={
            "E00100": "-9300",
            "E00200": "31000",
            "E00900": "-2500",
            "E01000": "-3000",
            "E02100": "-780",
            "P22250": "-3000",
            "P23250": "0",
        },
    ),
    main_record(recid="13", mars="4", s006="700", flpdyr="2014"),
    main_record(
        recid="14",
        s006="42",
        flpdyr="2012",
        amounts={"E00100": "79", "E00200": "77", "E00650": "2"},
    ),
    main_record(
        recid="15",
        s006="1000",
        flpdyr="2015",
        amounts={
            "E00100": "600",
            "E00200": "600",
            "E00600": "500",
            "E00650": "100",
        },
    ),
    main_record(
        recid=str(PUF_AGGREGATE_RECIDS[0]),
        mars="0",
        s006="0",
        amounts={
            "E00100": "-999999999.50",
            "E00200": "999999999.12",
            "E01000": "-99999999.75",
        },
    ),
]
DEFAULT_DEMOGRAPHIC = [demographic_record(recid="11"), demographic_record(recid="13")]


def build_fixture(main_records=None):
    mains = DEFAULT_MAIN if main_records is None else main_records
    main = csv_bytes(MAIN_HEADER, mains)
    demographic = csv_bytes(DEMOGRAPHIC_HEADER, DEFAULT_DEMOGRAPHIC)
    document = definition_document_json(PACKAGED_DEFINITION)
    document["route"] = "test_fixture"
    document["authority"] = "invented_fixture_nonauthority"
    for name, payload, header, records in (
        ("main", main, MAIN_HEADER, len(mains)),
        ("demographic", demographic, DEMOGRAPHIC_HEADER, len(DEFAULT_DEMOGRAPHIC)),
    ):
        entry = document["sources"][name]
        entry["bytes"] = len(payload)
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
        entry["git_blob_sha1"] = git_blob_sha1(payload)
        entry["delivered_header"] = list(header)
        entry["delivered_header_width"] = len(header)
        entry["header_record_canonical_sha256"] = header_digest(header)
        entry["data_records"] = records
    definition = fixture_definition(document)
    projection_json = projection_document_json(PACKAGED_THIRTEEN)
    projection_json["route"] = "test_fixture"
    projection_json["authority"] = "invented_fixture_nonauthority"
    projection_json["raw_source_definition"]["sha256"] = definition.sha256
    projection_json["raw_source_definition"]["bytes"] = len(definition.canonical)
    for name, pin in (
        ("main", definition.main),
        ("demographic", definition.demographic),
    ):
        entry = projection_json["sources"][name]
        entry["sha256"] = pin.sha256
        entry["git_blob_sha1"] = pin.git_blob_sha1
        entry["bytes"] = pin.bytes
        entry["header_record_canonical_sha256"] = pin.header_record_canonical_sha256
        entry["delivered_header_width"] = len(pin.delivered_header)
        entry["data_records"] = pin.data_records
    projection = fixture_agi_projection_document(projection_json, definition)
    status = decode_puf_raw_source(main, demographic, definition)
    decoded = decode_puf_monetary_source(main, status, projection, definition)
    return decoded, status, projection, definition


@pytest.fixture
def fixture():
    return build_fixture()


@pytest.fixture
def contract():
    return compile_price_baseline(PACKAGED_THIRTEEN.fields, **FIXTURE_RESOURCE)


# --------------------------------------------------------------------------
# The resource, the factor table and the recipe
# --------------------------------------------------------------------------


def test_the_pinned_levels_are_the_published_annual_averages():
    assert PRICE_BASELINE_SOURCE_YEAR == 2015
    assert PRICE_BASELINE_TARGET_YEAR == 2024
    assert dict(PRICE_BASELINE_LEVELS) == {2015: "237.017", 2024: "313.689"}
    assert PRICE_BASELINE_RATIO == (313689, 237017)
    assert baseline.CPI_U_SERIES_ID == "CUUR0000SA0"
    assert baseline.CPI_U_ANNUAL_AVERAGE_PERIOD == "M13"


def test_the_declared_factor_is_the_exact_rational_in_float64(contract):
    """The level-text division and the exact rational agree bit for bit here."""
    numerator, denominator = PRICE_BASELINE_RATIO
    exact = numerator / denominator
    from_levels = float(PRICE_BASELINE_LEVELS[2024]) / float(
        PRICE_BASELINE_LEVELS[2015]
    )
    assert exact == from_levels
    bits = struct.pack("<d", exact).hex()
    for field in contract.money_fields:
        resolved = contract.factor_for(field, SignBranch.ANY)
        assert resolved.ratio_bits == bits
        assert resolved.value == exact
        assert resolved.source_level == "237.017"
        assert resolved.target_level == "313.689"


def test_the_factor_table_is_a_developmental_public_resource(contract):
    table = contract.factors
    assert table.authority is FactorAuthority.DEVELOPMENTAL_PUBLIC_RESOURCE
    assert contract.release_eligible is False
    assert table.provenance["population_divisor"] == "none"
    assert table.provenance["index_or_total"] == "index"
    assert table.provenance["resource_sha256"] == FIXTURE_RESOURCE["resource_sha256"]
    assert table.provenance["retrieved_utc"] == FIXTURE_RESOURCE["retrieved_utc"]
    module = Path(baseline.__file__).resolve()
    assert (
        table.provenance["generator_code_sha256"]
        == hashlib.sha256(module.read_bytes()).hexdigest()
    )
    assert GrowthFactorTable.from_bytes(table.document) == table


def test_one_price_series_covers_every_money_field_and_no_other(contract):
    recipe = contract.recipe
    money = [rule for rule in recipe.rules if str(rule.role) == "money"]
    assert len(money) == 13
    assert {rule.factor_series for rule in money} == {PRICE_BASELINE_SERIES_NAME}
    assert {rule.growth_kind for rule in money} == {
        GrowthKind.NOMINAL_PRICE_RESTATEMENT
    }
    assert {rule.sign_branch for rule in money} == {SignBranch.ANY}
    assert {rule.source_reference_year for rule in money} == {2015}
    assert {rule.target_year for rule in money} == {2024}
    weight = contract.design_weight_rule
    assert weight is not None
    assert weight.field == "S006"
    assert weight.factor_series == DESIGN_WEIGHT_IDENTITY_SERIES
    assert contract.factor_for("S006", SignBranch.ANY).value == 1.0


def test_every_grown_field_declares_the_price_restated_basis(contract):
    metadata = puf_growth_monetary_metadata(contract)
    assert set(metadata) == set(contract.money_fields)
    for basis in metadata.values():
        assert basis.valuation == "nominal_price_restated_from_2015"
        assert basis.period == "2024"
        assert basis.currency == "USD"


def test_a_recipe_without_agi_or_with_a_non_money_field_is_refused():
    without = [f for f in PACKAGED_THIRTEEN.fields if f != AGI_FIELD]
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_recipe(without)
    assert error.value.reason == "PRICE_BASELINE_AGI_FIELD"
    for reserved in ("RECID", "S006"):
        with pytest.raises(PufGrowthRefusalError) as error:
            price_baseline_recipe([*PACKAGED_THIRTEEN.fields, reserved])
        assert error.value.reason == "PRICE_BASELINE_NON_MONEY_FIELD"
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_recipe([])
    assert error.value.reason == "PRICE_BASELINE_NO_MONEY_FIELD"


def test_a_developmental_table_cannot_be_relabelled_reviewed():
    """Relabelling the same bytes hits the deliberately empty allowlist."""
    document = json.loads(price_baseline_factor_document(**FIXTURE_RESOURCE))
    document["authority"] = "reviewed_resource"
    forged = GrowthFactorTable.from_bytes(
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        compile_puf_growth(price_baseline_recipe(PACKAGED_THIRTEEN.fields), forged)
    assert error.value.reason == "UNREVIEWED_FACTOR_TABLE"


# --------------------------------------------------------------------------
# The artifact-to-money-view adapter
# --------------------------------------------------------------------------


def test_the_money_view_keeps_every_amount_known_return(fixture, contract):
    decoded, status, _, _ = fixture
    adaptation = adapt_projection_to_money_view(decoded, status, contract)
    view = adaptation.money_view
    assert len(view) == 5
    assert decoded.rows == 6
    assert set(view.columns) == set(contract.money_view_fields)
    assert "S006" not in view.columns
    assert view["RECID"].tolist() == [11, 12, 13, 14, 15]
    assert view[AGI_FIELD].tolist() == [71000, -9300, 0, 79, 600]
    assert all(str(view[name].dtype) == "int64" for name in view.columns)
    assert adaptation.receipt["money_view_rows"] == 5
    assert adaptation.receipt["excluded_rows"] == 1
    assert adaptation.receipt["release_eligible"] is False
    assert adaptation.receipt["design_weight_in_money_view"] is False


def test_only_the_aggregate_rows_are_removed_and_their_identity_is_kept(
    fixture, contract
):
    decoded, status, _, _ = fixture
    adaptation = adapt_projection_to_money_view(decoded, status, contract)
    excluded = adaptation.excluded
    assert excluded["rows"] == 1
    assert excluded["recid"] == [PUF_AGGREGATE_RECIDS[0]]
    assert excluded["aggregate_typed_sentinel"] == AGGREGATE_SENTINEL
    assert excluded["interpretation"].startswith("refused")
    assert len(excluded["delivered_token_sha256"]) == 64
    # The audit identity names token classes, never an aggregate amount.
    classes = excluded["token_class_counts"][AGI_FIELD]
    assert classes == {
        "rows": 1,
        "integer_tokens": 0,
        "fractional_tokens": 1,
        "negative_tokens": 1,
        "max_token_width": 13,
    }
    assert not any(
        key in classes for key in ("sum", "minimum", "maximum", "zero", "negative")
    )


def test_flpdyr_status_and_the_unaltered_weight_travel_separately(fixture, contract):
    decoded, status, _, _ = fixture
    adaptation = adapt_projection_to_money_view(decoded, status, contract)
    assert adaptation.flpdyr.tolist() == [2015, 2013, 2014, 2012, 2015]
    assert adaptation.demographic_status.tolist() == [1, 0, 1, 0, 0]
    assert adaptation.design_weight.tolist() == [12345, 9900, 700, 42, 1000]
    assert np.array_equal(
        adaptation.design_weight,
        np.asarray(status.typed["S006"], dtype="<i8")[
            np.asarray(decoded.amount_known, dtype=bool)
        ],
    )
    assert adaptation.receipt["design_weight_units"] == (
        "delivered_integer_hundredths_unaltered"
    )
    assert adaptation.receipt["return_mass_transition"] == (
        "none_declared_identity_series"
    )
    assert adaptation.receipt["flpdyr_treatment"] == (
        "carried_as_observation_metadata_never_restated"
    )


def test_no_sentinel_ever_reaches_a_numeric_input(fixture, contract):
    decoded, status, _, _ = fixture
    adaptation = adapt_projection_to_money_view(decoded, status, contract)
    for field in contract.money_fields:
        assert not (adaptation.money_view[field] == AGGREGATE_SENTINEL).any()
        assert int(decoded.typed[field][5]) == AGGREGATE_SENTINEL


def test_a_sentinel_on_an_ordinary_row_refuses_the_adaptation(fixture, contract):
    decoded, status, _, _ = fixture
    typed = {name: np.array(array) for name, array in decoded.typed.items()}
    typed[AGI_FIELD][0] = AGGREGATE_SENTINEL
    forged = PufMonetarySourceProjection(
        typed=typed,
        lexical=decoded.lexical,
        facts=decoded.facts,
        column_metadata=decoded.column_metadata,
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        adapt_projection_to_money_view(forged, status, contract)
    assert error.value.reason == "SENTINEL_IN_ORDINARY_ROW"


def test_an_aggregate_row_without_the_sentinel_refuses_the_adaptation(
    fixture, contract
):
    decoded, status, _, _ = fixture
    typed = {name: np.array(array) for name, array in decoded.typed.items()}
    typed[AGI_FIELD][5] = 0
    forged = PufMonetarySourceProjection(
        typed=typed,
        lexical=decoded.lexical,
        facts=decoded.facts,
        column_metadata=decoded.column_metadata,
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        adapt_projection_to_money_view(forged, status, contract)
    assert error.value.reason == "AGGREGATE_SLOT_NOT_SENTINEL"


def test_a_status_artifact_that_is_not_this_projection_is_refused(fixture, contract):
    decoded, status, _, _ = fixture

    class _Status:
        def __init__(self):
            self.typed = {name: np.array(array) for name, array in status.typed.items()}

    shifted = _Status()
    shifted.typed["RECID"] = shifted.typed["RECID"][::-1].copy()
    with pytest.raises(PufGrowthRefusalError) as error:
        adapt_projection_to_money_view(decoded, shifted, contract)
    assert error.value.reason == "STATUS_RECID_JOIN"

    flags = _Status()
    flags.typed["disclosure_aggregate"] = np.zeros_like(
        flags.typed["disclosure_aggregate"]
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        adapt_projection_to_money_view(decoded, flags, contract)
    assert error.value.reason == "STATUS_FLAG_JOIN"

    class _Missing:
        typed = {"RECID": np.array([1], dtype="<i8")}

    with pytest.raises(PufGrowthRefusalError) as error:
        adapt_projection_to_money_view(decoded, _Missing(), contract)
    assert error.value.reason == "STATUS_ARTIFACT_COLUMN"


# --------------------------------------------------------------------------
# Growth, through the existing transform
# --------------------------------------------------------------------------


def grown_for(fixture, contract):
    decoded, status, _, _ = fixture
    adaptation = adapt_projection_to_money_view(decoded, status, contract)
    return adaptation, apply_puf_growth(adaptation.money_view, contract)


def test_the_baseline_grows_through_the_existing_transform(fixture, contract):
    adaptation, grown = grown_for(fixture, contract)
    table = grown.table
    factor = contract.factor_for(AGI_FIELD, SignBranch.ANY).value
    assert table[AGI_FIELD].tolist() == [
        71000 * factor,
        -9300 * factor,
        0.0,
        79 * factor,
        600 * factor,
    ]
    assert str(table[AGI_FIELD].dtype) == "float64"
    assert table[PROVENANCE_RECID_COLUMN].tolist() == [11, 12, 13, 14, 15]
    assert table[PROVENANCE_SOURCE_AGI_COLUMN].tolist() == [
        71000.0,
        -9300.0,
        0.0,
        79.0,
        600.0,
    ]
    receipt = grown.receipt
    assert receipt["release_eligible"] is False
    assert receipt["factor_authority"] == "developmental_public_resource"
    assert receipt["source_reference_year"] == 2015
    assert receipt["target_year"] == 2024
    assert receipt["design_weight_in_money_view"] is False
    assert sorted(receipt["grown_money_fields"]) == sorted(contract.money_fields)


def test_signs_and_zeros_survive_and_the_output_is_unrounded(fixture, contract):
    adaptation, grown = grown_for(fixture, contract)
    for field in contract.money_fields:
        original = adaptation.money_view[field].to_numpy()
        result = grown.table[field].to_numpy()
        assert np.array_equal(np.sign(result), np.sign(original.astype(float)))
        assert np.array_equal(result == 0.0, original == 0)
        assert not np.signbit(result[result == 0.0]).any()
        # Unrounded: at least one value is not an integer number of dollars.
        if (original != 0).any():
            assert not np.array_equal(result, np.round(result))


def test_the_design_weight_is_untouched_by_the_declared_identity(fixture, contract):
    adaptation, _ = grown_for(fixture, contract)
    grown_weights = apply_puf_design_weight_growth(
        adaptation.design_weight.astype("float64"), contract
    )
    assert grown_weights.tolist() == adaptation.design_weight.astype(float).tolist()
    assert contract.factor_for("S006", SignBranch.ANY).value == 1.0


def test_the_invariants_hold_and_are_measured(fixture, contract):
    adaptation, grown = grown_for(fixture, contract)
    invariants = price_baseline_invariants(adaptation, grown, contract)
    assert invariants["rows"] == 5
    assert invariants["output_rounding"] == "none_unrounded_float64"
    assert invariants["release_eligible"] is False
    assert invariants["exact_rational"] == {
        "numerator": 313689,
        "denominator": 237017,
    }
    for field, entry in invariants["columns"].items():
        assert entry["relative_difference"] <= 1e-12, field
        assert entry["ratio_bits"] == invariants["factor_ratio_bits"]
    dividends = invariants["pairwise_observations"][
        "qualified_within_ordinary_dividends"
    ]
    # Row 14 has a qualified dividend above its ordinary dividend in the
    # invented delivery, and one positive factor cannot change that.
    assert dividends["violations_before"] == 1
    assert dividends["violations_after"] == 1
    assert dividends["status"] == "observed_not_enforced"
    pensions = invariants["pairwise_observations"]["taxable_within_total_pensions"]
    assert pensions["violations_before"] == pensions["violations_after"] == 0
    assert {name for name, _, _ in PAIRWISE_OBSERVATIONS} == set(
        invariants["pairwise_observations"]
    )


def test_the_scaled_sum_check_catches_a_wrong_grown_column(fixture, contract):
    adaptation, grown = grown_for(fixture, contract)
    grown.table.loc[0, "E00200"] = grown.table.loc[0, "E00200"] + 1.0
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(adaptation, grown, contract)
    assert error.value.reason == "SCALED_SUM_DISAGREEMENT"


def test_a_moved_pairwise_violation_would_be_caught(fixture, contract):
    """Permuting one column keeps its exact sum, its signs and its zeros.

    All it changes is which row pairs with which, which is precisely what the
    pairwise observation exists to notice.
    """
    adaptation, grown = grown_for(fixture, contract)
    values = grown.table["E00650"].to_numpy(copy=True)
    values[0], values[4] = values[4], values[0]
    grown.table["E00650"] = values
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(adaptation, grown, contract)
    assert error.value.reason == "PAIRWISE_VIOLATIONS_MOVED"


def test_nothing_in_this_baseline_can_become_release_eligible(contract):
    assert contract.release_eligible is False
    _, grown = grown_for(build_fixture(), contract)
    assert grown.receipt["release_eligible"] is False
    assert contract.factors.authority is FactorAuthority.DEVELOPMENTAL_PUBLIC_RESOURCE


def test_a_money_view_carrying_the_design_weight_is_refused(fixture, contract):
    decoded, status, _, _ = fixture
    adaptation = adapt_projection_to_money_view(decoded, status, contract)
    view = adaptation.money_view.copy()
    view["S006"] = adaptation.design_weight
    with pytest.raises(PufGrowthRefusalError) as error:
        apply_puf_growth(view, contract)
    assert error.value.reason == "DESIGN_WEIGHT_NOT_IN_MONEY_VIEW"


def test_the_projection_and_the_contract_must_be_the_same_field_set(fixture):
    """A twelve-field contract cannot be adapted from a thirteen-field artifact."""
    decoded, status, _, _ = fixture
    twelve = compile_price_baseline(
        [f for f in PACKAGED_THIRTEEN.fields if f != "P23250"], **FIXTURE_RESOURCE
    )
    adaptation = adapt_projection_to_money_view(decoded, status, twelve)
    assert "P23250" not in adaptation.money_view.columns
    assert set(adaptation.money_view.columns) == set(twelve.money_view_fields)


def test_an_artifact_that_is_not_a_projection_is_refused(contract):
    with pytest.raises(PufGrowthRefusalError) as error:
        adapt_projection_to_money_view(object(), object(), contract)
    assert error.value.reason == "PROJECTION_ARTIFACT_TYPE"
    assert PufMonetaryRefusalError is not PufGrowthRefusalError
