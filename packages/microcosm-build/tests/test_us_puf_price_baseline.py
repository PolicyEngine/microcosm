"""The developmental CPI-U price-restatement baseline, on invented bytes only.

No PUF is read, no artifact store is touched and no country model is imported.
The delivery, the raw definition and the projection document are all built
from literals here and travel the explicit ``test_fixture`` route. The CPI
levels are the real published annual averages, because the point of the
developmental authority is that a public resource may say so about itself; the
resource body and URL below are fabricated fixtures, with the digest computed
from those exact invented bytes. A genuine run must supply its retained body.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.us_runtime import puf_price_baseline as baseline
from microcosm.build.us_runtime.puf_growth import (
    PROVENANCE_RECID_COLUMN,
    PROVENANCE_SOURCE_AGI_COLUMN,
    FactorAuthority,
    FieldRole,
    GrownPufTable,
    GrowthFactorTable,
    GrowthKind,
    GrowthRecipe,
    GrowthRule,
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
    require_price_baseline_contract,
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

#: A fabricated stand-in for the BLS flat file, in the publisher's own
#: tab-delimited, space-padded shape (see ``us-puf-price-baseline-cpi-pin.py``
#: in the architecture-review lane, which extracted this exact row shape from
#: the genuine download). This is **not** the genuine ~2.7 MB distribution —
#: it is three invented lines carrying the two real published levels, and its
#: digest below is computed from these exact bytes, not asserted.
FIXTURE_CPI_RESOURCE_BYTES = (
    b"series_id        \tyear\tperiod\t       value\tfootnote_codes\r\n"
    b"CUUR0000SA0      \t2015\tM13\t     237.017\t\r\n"
    b"CUUR0000SA0      \t2024\tM13\t     313.689\t\r\n"
)

#: A second fabricated body, distinguished from the one above only by an
#: extra row for a series this baseline does not read. Same two verified
#: records, a different whole-file digest — the shape
#: ``test_the_two_contracts_differ_only_in_identity`` needs, without simply
#: asserting a ``resource_sha256`` string no bytes back up.
FIXTURE_CPI_RESOURCE_BYTES_ALT = FIXTURE_CPI_RESOURCE_BYTES + (
    b"CUUS0000SA0      \t2024\tM13\t     312.332\t\r\n"
)

#: Invented stand-ins for the pin a genuine run supplies. The URL is a
#: reserved non-resolvable domain; nothing here retrieved anything.
FIXTURE_RESOURCE = {
    "source_url": "https://example.invalid/pub/time.series/cu/cu.data.1.AllItems",
    "resource_sha256": hashlib.sha256(FIXTURE_CPI_RESOURCE_BYTES).hexdigest(),
    "resource_bytes": FIXTURE_CPI_RESOURCE_BYTES,
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
# The resource bytes: verified, not merely asserted
# --------------------------------------------------------------------------


def test_verify_cpi_resource_bytes_matches_the_fixture():
    receipt = baseline.verify_cpi_resource_bytes(
        FIXTURE_CPI_RESOURCE_BYTES, resource_sha256=FIXTURE_RESOURCE["resource_sha256"]
    )
    assert receipt["series_id"] == "CUUR0000SA0"
    assert receipt["period"] == "M13"
    assert receipt["resource_bytes"] == len(FIXTURE_CPI_RESOURCE_BYTES)
    assert receipt["resource_sha256"] == FIXTURE_RESOURCE["resource_sha256"]
    assert receipt["records"]["2015"]["level_text"] == "237.017"
    assert receipt["records"]["2024"]["level_text"] == "313.689"
    assert receipt["records"]["2015"]["level_float64"] == float("237.017")
    assert receipt["records"]["2024"]["level_float64"] == float("313.689")


def test_bytes_that_are_not_bytes_are_refused():
    with pytest.raises(PufGrowthRefusalError) as error:
        baseline.verify_cpi_resource_bytes("not bytes", resource_sha256="0" * 64)
    assert error.value.reason == "PRICE_BASELINE_RESOURCE_TYPE"


def test_empty_resource_bytes_are_refused():
    with pytest.raises(PufGrowthRefusalError) as error:
        baseline.verify_cpi_resource_bytes(
            b"", resource_sha256=hashlib.sha256(b"").hexdigest()
        )
    assert error.value.reason == "PRICE_BASELINE_RESOURCE_EMPTY"


def test_oversize_resource_bytes_are_refused():
    body = b"x" * (baseline._CPI_RESOURCE_MAX_BYTES + 1)
    with pytest.raises(PufGrowthRefusalError) as error:
        baseline.verify_cpi_resource_bytes(
            body, resource_sha256=hashlib.sha256(body).hexdigest()
        )
    assert error.value.reason == "PRICE_BASELINE_RESOURCE_OVERSIZE"


def test_a_wrong_digest_is_refused():
    with pytest.raises(PufGrowthRefusalError) as error:
        baseline.verify_cpi_resource_bytes(
            FIXTURE_CPI_RESOURCE_BYTES, resource_sha256="0" * 64
        )
    assert error.value.reason == "PRICE_BASELINE_RESOURCE_DIGEST"


def test_a_missing_record_is_refused():
    body = (
        b"series_id        \tyear\tperiod\t       value\tfootnote_codes\r\n"
        b"CUUR0000SA0      \t2015\tM13\t     237.017\t\r\n"
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        baseline.verify_cpi_resource_bytes(
            body, resource_sha256=hashlib.sha256(body).hexdigest()
        )
    assert error.value.reason == "PRICE_BASELINE_RESOURCE_RECORD_MISSING"
    assert error.value.field == "2024"


def test_a_duplicate_record_is_refused():
    body = (
        FIXTURE_CPI_RESOURCE_BYTES + b"CUUR0000SA0      \t2015\tM13\t     237.017\t\r\n"
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        baseline.verify_cpi_resource_bytes(
            body, resource_sha256=hashlib.sha256(body).hexdigest()
        )
    assert error.value.reason == "PRICE_BASELINE_RESOURCE_DUPLICATE_RECORD"
    assert error.value.field == "2015"


def test_a_wrong_value_is_refused():
    body = (
        b"series_id        \tyear\tperiod\t       value\tfootnote_codes\r\n"
        b"CUUR0000SA0      \t2015\tM13\t     237.017\t\r\n"
        b"CUUR0000SA0      \t2024\tM13\t     999.999\t\r\n"
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        baseline.verify_cpi_resource_bytes(
            body, resource_sha256=hashlib.sha256(body).hexdigest()
        )
    assert error.value.reason == "PRICE_BASELINE_RESOURCE_RECORD_VALUE"
    assert error.value.field == "2024"


def test_a_malformed_matching_line_is_refused():
    """A truncated row for this series is refused, not silently skipped."""
    body = (
        b"series_id        \tyear\tperiod\t       value\tfootnote_codes\r\n"
        b"CUUR0000SA0      \t2015\tM13\r\n"
        b"CUUR0000SA0      \t2024\tM13\t     313.689\t\r\n"
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        baseline.verify_cpi_resource_bytes(
            body, resource_sha256=hashlib.sha256(body).hexdigest()
        )
    assert error.value.reason == "PRICE_BASELINE_RESOURCE_LINE_SHAPE"


def test_a_different_series_period_or_year_row_does_not_interfere():
    """Only this series, this period and the two wanted years are read."""
    body = FIXTURE_CPI_RESOURCE_BYTES + (
        b"CUUR0000SA0      \t2016\tM13\t     240.007\t\r\n"
        b"CUUR0000SA0      \t2024\tM01\t     308.417\t\r\n"
        b"CUUS0000SA0      \t2024\tM13\t     312.332\t\r\n"
    )
    receipt = baseline.verify_cpi_resource_bytes(
        body, resource_sha256=hashlib.sha256(body).hexdigest()
    )
    assert set(receipt["records"]) == {"2015", "2024"}


def test_compile_price_baseline_requires_resource_bytes():
    """The keyword is required, not optional-with-a-trusting-default."""
    without_bytes = {k: v for k, v in FIXTURE_RESOURCE.items() if k != "resource_bytes"}
    with pytest.raises(TypeError):
        compile_price_baseline(PACKAGED_THIRTEEN.fields, **without_bytes)


def test_compile_price_baseline_refuses_a_digest_mismatch():
    with pytest.raises(PufGrowthRefusalError) as error:
        compile_price_baseline(
            PACKAGED_THIRTEEN.fields,
            **{**FIXTURE_RESOURCE, "resource_sha256": "0" * 64},
        )
    assert error.value.reason == "PRICE_BASELINE_RESOURCE_DIGEST"


def test_compile_price_baseline_refuses_bytes_missing_a_record():
    body = (
        b"series_id        \tyear\tperiod\t       value\tfootnote_codes\r\n"
        b"CUUR0000SA0      \t2015\tM13\t     237.017\t\r\n"
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        compile_price_baseline(
            PACKAGED_THIRTEEN.fields,
            **{
                **FIXTURE_RESOURCE,
                "resource_sha256": hashlib.sha256(body).hexdigest(),
                "resource_bytes": body,
            },
        )
    assert error.value.reason == "PRICE_BASELINE_RESOURCE_RECORD_MISSING"


# --------------------------------------------------------------------------
# The artifact-to-money-view adapter
# --------------------------------------------------------------------------


def test_the_money_view_keeps_every_amount_known_return(fixture, contract):
    decoded, status, _, _ = fixture
    adaptation = adapt_projection_to_money_view(
        decoded, status, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
    )
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
    adaptation = adapt_projection_to_money_view(
        decoded, status, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
    )
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
    adaptation = adapt_projection_to_money_view(
        decoded, status, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
    )
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
    adaptation = adapt_projection_to_money_view(
        decoded, status, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
    )
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
        adapt_projection_to_money_view(
            forged, status, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
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
        adapt_projection_to_money_view(
            forged, status, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "AGGREGATE_SLOT_NOT_SENTINEL"


def test_a_status_artifact_that_is_not_this_projection_is_refused(fixture, contract):
    decoded, status, _, _ = fixture

    class _Status:
        def __init__(self):
            self.typed = {name: np.array(array) for name, array in status.typed.items()}

    shifted = _Status()
    shifted.typed["RECID"] = shifted.typed["RECID"][::-1].copy()
    with pytest.raises(PufGrowthRefusalError) as error:
        adapt_projection_to_money_view(
            decoded, shifted, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "STATUS_RECID_JOIN"

    flags = _Status()
    flags.typed["disclosure_aggregate"] = np.zeros_like(
        flags.typed["disclosure_aggregate"]
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        adapt_projection_to_money_view(
            decoded, flags, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "STATUS_FLAG_JOIN"

    class _Missing:
        typed = {"RECID": np.array([1], dtype="<i8")}

    with pytest.raises(PufGrowthRefusalError) as error:
        adapt_projection_to_money_view(
            decoded, _Missing(), contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "STATUS_ARTIFACT_COLUMN"

    # A side column of the wrong length would misalign FLPDYR or S006 against
    # the money view without touching either join key.
    short = _Status()
    short.typed["FLPDYR"] = short.typed["FLPDYR"][:-1].copy()
    with pytest.raises(PufGrowthRefusalError) as error:
        adapt_projection_to_money_view(
            decoded, short, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "STATUS_COLUMN_ROWS"
    assert error.value.field == "FLPDYR"

    with pytest.raises(PufGrowthRefusalError) as error:
        adapt_projection_to_money_view(
            decoded, object(), contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "STATUS_ARTIFACT_TYPE"


# --------------------------------------------------------------------------
# Growth, through the existing transform
# --------------------------------------------------------------------------


def grown_for(fixture, contract):
    decoded, status, _, _ = fixture
    adaptation = adapt_projection_to_money_view(
        decoded, status, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
    )
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
    invariants = price_baseline_invariants(
        adaptation, grown, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
    )
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
        price_baseline_invariants(
            adaptation, grown, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
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
        price_baseline_invariants(
            adaptation, grown, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "PAIRWISE_VIOLATIONS_MOVED"


def test_nothing_in_this_baseline_can_become_release_eligible(contract):
    assert contract.release_eligible is False
    _, grown = grown_for(build_fixture(), contract)
    assert grown.receipt["release_eligible"] is False
    assert contract.factors.authority is FactorAuthority.DEVELOPMENTAL_PUBLIC_RESOURCE


def test_a_money_view_carrying_the_design_weight_is_refused(fixture, contract):
    decoded, status, _, _ = fixture
    adaptation = adapt_projection_to_money_view(
        decoded, status, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
    )
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
    adaptation = adapt_projection_to_money_view(
        decoded, status, twelve, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
    )
    assert "P23250" not in adaptation.money_view.columns
    assert set(adaptation.money_view.columns) == set(twelve.money_view_fields)


def test_an_artifact_that_is_not_a_projection_is_refused(contract):
    with pytest.raises(PufGrowthRefusalError) as error:
        adapt_projection_to_money_view(
            object(), object(), contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "PROJECTION_ARTIFACT_TYPE"
    assert PufMonetaryRefusalError is not PufGrowthRefusalError


# --------------------------------------------------------------------------
# The entry contract: what this module will and will not describe
# --------------------------------------------------------------------------


def factor_document(series: dict, *, authority: str = "developmental_public_resource"):
    """A factor table like the baseline's, with the series a case needs."""
    document = json.loads(price_baseline_factor_document(**FIXTURE_RESOURCE))
    document["authority"] = authority
    document["series"] = series
    document["years"] = sorted(
        {int(year) for entry in series.values() for year in entry["levels"]}
    )
    return GrowthFactorTable.from_bytes(
        json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    )


def price_series(source_year: int, target_year: int, target_level: str = "313.689"):
    return {
        "kind": "nominal_price_restatement",
        "levels": {str(source_year): "237.017", str(target_year): target_level},
    }


def identity_weight_series(source_year: int = 2015, target_year: int = 2024, level="1"):
    return {
        "kind": "design_weight_growth",
        "levels": {str(source_year): level, str(target_year): level},
    }


def baseline_rules(*, money_fields=None, source_year=2015, target_year=2024, **kwargs):
    """The adopted rule shape, with one dimension moved per case."""
    fields = PACKAGED_THIRTEEN.fields if money_fields is None else money_fields
    kind = kwargs.get("growth_kind", GrowthKind.NOMINAL_PRICE_RESTATEMENT)
    series = kwargs.get("factor_series", PRICE_BASELINE_SERIES_NAME)
    rules = [
        GrowthRule(
            field,
            FieldRole.MONEY,
            source_year,
            target_year,
            kind,
            SignBranch.ANY,
            series,
            "case",
        )
        for field in fields
    ]
    rules.append(
        GrowthRule(
            kwargs.get("identifier_field", "RECID"),
            FieldRole.IDENTIFIER,
            kwargs.get("identifier_year", source_year),
            kwargs.get("identifier_year", source_year),
            GrowthKind.NONE,
        )
    )
    if kwargs.get("design_weight", True):
        rules.append(
            GrowthRule(
                "S006",
                FieldRole.DESIGN_WEIGHT,
                source_year,
                target_year,
                GrowthKind.DESIGN_WEIGHT_GROWTH,
                SignBranch.ANY,
                kwargs.get("weight_series", DESIGN_WEIGHT_IDENTITY_SERIES),
                "case",
            )
        )
    rules.extend(kwargs.get("extra_rules", ()))
    return GrowthRecipe("case_recipe_v1", tuple(rules))


def compiled_case(recipe, series, *, authority="developmental_public_resource"):
    return compile_puf_growth(recipe, factor_document(series, authority=authority))


def refusal_from_both_entries(fixture, compiled):
    """Both entry points must decline the same contract with the same reason."""
    decoded, status, _, _ = fixture
    with pytest.raises(PufGrowthRefusalError) as adapter_error:
        adapt_projection_to_money_view(
            decoded, status, compiled, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    with pytest.raises(PufGrowthRefusalError) as gate_error:
        require_price_baseline_contract(
            compiled, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert adapter_error.value.reason == gate_error.value.reason
    return adapter_error.value


def test_a_same_factor_different_source_year_contract_is_declined(fixture):
    """Root's counterexample: the receipt used to say 2015 regardless.

    This contract is internally valid and resolves to the identical factor —
    the same bit pattern, the same published level texts — but its money rules
    name 2016 as the source year. The adaptation receipt used to write
    ``source_reference_year: 2015`` from a module constant, so the run would
    have carried a statement about a year the contract never declared.
    """
    series = {
        PRICE_BASELINE_SERIES_NAME: price_series(2016, 2024),
        DESIGN_WEIGHT_IDENTITY_SERIES: identity_weight_series(2016, 2024),
    }
    compiled = compiled_case(baseline_rules(source_year=2016), series)
    # The contract is a valid contract, and its factor is *this* factor.
    assert (
        compiled.factor_for(AGI_FIELD, SignBranch.ANY).ratio_bits
        == struct.pack("<d", PRICE_BASELINE_RATIO[0] / PRICE_BASELINE_RATIO[1]).hex()
    )
    assert compiled.recipe.rules_for(AGI_FIELD)[0].source_reference_year == 2016
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_SOURCE_YEAR"


def test_a_same_factor_different_target_year_contract_is_declined(fixture):
    """2015 to 2023, with 2023 carrying the 2024 level: the same factor again."""
    series = {
        PRICE_BASELINE_SERIES_NAME: price_series(2015, 2023),
        DESIGN_WEIGHT_IDENTITY_SERIES: identity_weight_series(2015, 2023),
    }
    compiled = compiled_case(baseline_rules(target_year=2023), series)
    assert compiled.target_year == 2023
    assert compiled.factor_for(AGI_FIELD, SignBranch.ANY).value == 313689 / 237017
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_TARGET_YEAR"


def test_a_different_factor_is_declined_rather_than_reported_as_this_one(fixture):
    """The invariants used to write 313689/237017 whatever the contract said."""
    series = {
        PRICE_BASELINE_SERIES_NAME: price_series(2015, 2024, target_level="300.000"),
        DESIGN_WEIGHT_IDENTITY_SERIES: identity_weight_series(),
    }
    compiled = compiled_case(baseline_rules(), series)
    assert compiled.factor_for(AGI_FIELD, SignBranch.ANY).value != 313689 / 237017
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_FACTOR"
    assert error.field in PACKAGED_THIRTEEN.fields


def test_other_level_texts_that_divide_to_the_same_double_are_declined(fixture):
    """The factor is right and the provenance is not, which is still not this.

    More than one pair of decimal texts lands on the one float64 this baseline
    uses. Checking only the bits would admit a contract whose recorded
    ``source_level`` and ``target_level`` are not the two published annual
    averages, and those two strings are what the receipt cites.
    """
    weights = identity_weight_series()
    ratio_as_one_level = {
        "kind": "nominal_price_restatement",
        "levels": {"2015": "1", "2024": "1.3234873447896143"},
    }
    compiled = compiled_case(
        baseline_rules(),
        {
            PRICE_BASELINE_SERIES_NAME: ratio_as_one_level,
            **{DESIGN_WEIGHT_IDENTITY_SERIES: weights},
        },
    )
    assert (
        compiled.factor_for(AGI_FIELD, SignBranch.ANY).ratio_bits
        == baseline.PRICE_BASELINE_FACTOR_BITS
    )
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_SOURCE_LEVEL"

    longer_target = {
        "kind": "nominal_price_restatement",
        "levels": {"2015": "237.017", "2024": "313.68900000000001"},
    }
    compiled = compiled_case(
        baseline_rules(),
        {
            PRICE_BASELINE_SERIES_NAME: longer_target,
            DESIGN_WEIGHT_IDENTITY_SERIES: weights,
        },
    )
    assert (
        compiled.factor_for(AGI_FIELD, SignBranch.ANY).ratio_bits
        == baseline.PRICE_BASELINE_FACTOR_BITS
    )
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_TARGET_LEVEL"


def test_a_real_per_return_aging_contract_is_declined(fixture):
    """Aging is a different claim about the same numbers, and is not this one."""
    series = {
        PRICE_BASELINE_SERIES_NAME: {
            "kind": "real_per_return_aging",
            "levels": {"2015": "237.017", "2024": "313.689"},
        },
        DESIGN_WEIGHT_IDENTITY_SERIES: identity_weight_series(),
    }
    compiled = compiled_case(
        baseline_rules(growth_kind=GrowthKind.REAL_PER_RETURN_AGING), series
    )
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_GROWTH_KIND"


def test_a_sign_branched_contract_is_declined(fixture):
    """Two branches would mean two factors could be resolved for one field."""
    fields = list(PACKAGED_THIRTEEN.fields)
    rules = [
        GrowthRule(
            field,
            FieldRole.MONEY,
            2015,
            2024,
            GrowthKind.NOMINAL_PRICE_RESTATEMENT,
            branch,
            PRICE_BASELINE_SERIES_NAME,
            "case",
        )
        for field in fields
        for branch in (
            (SignBranch.POSITIVE, SignBranch.NEGATIVE)
            if field == "E00900"
            else (SignBranch.ANY,)
        )
    ]
    rules.append(GrowthRule("RECID", FieldRole.IDENTIFIER, 2015, 2015, GrowthKind.NONE))
    rules.append(
        GrowthRule(
            "S006",
            FieldRole.DESIGN_WEIGHT,
            2015,
            2024,
            GrowthKind.DESIGN_WEIGHT_GROWTH,
            SignBranch.ANY,
            DESIGN_WEIGHT_IDENTITY_SERIES,
            "case",
        )
    )
    series = {
        PRICE_BASELINE_SERIES_NAME: price_series(2015, 2024),
        DESIGN_WEIGHT_IDENTITY_SERIES: identity_weight_series(),
    }
    compiled = compiled_case(GrowthRecipe("case_recipe_v1", tuple(rules)), series)
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_SIGN_BRANCH"
    assert error.field == "E00900"


def test_a_second_factor_series_on_one_field_is_declined(fixture):
    """One named series covers every money field, or this is not the baseline."""
    series = {
        "some.other.series": price_series(2015, 2024),
        DESIGN_WEIGHT_IDENTITY_SERIES: identity_weight_series(),
    }
    compiled = compiled_case(baseline_rules(factor_series="some.other.series"), series)
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_SERIES"


def test_a_growing_design_weight_is_declined(fixture):
    """The one refusal that stops this baseline transitioning return mass."""
    series = {
        PRICE_BASELINE_SERIES_NAME: price_series(2015, 2024),
        DESIGN_WEIGHT_IDENTITY_SERIES: {
            "kind": "design_weight_growth",
            "levels": {"2015": "1", "2024": "2"},
        },
    }
    compiled = compiled_case(baseline_rules(), series)
    assert compiled.factor_for("S006", SignBranch.ANY).value == 2.0
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_DESIGN_WEIGHT_FACTOR"
    assert error.field == "S006"


def test_two_weight_levels_that_merely_round_to_one_are_declined(fixture):
    """``1e-323`` over ``1e-323`` is 1.0 in float64 and is not an identity."""
    series = {
        PRICE_BASELINE_SERIES_NAME: price_series(2015, 2024),
        DESIGN_WEIGHT_IDENTITY_SERIES: {
            "kind": "design_weight_growth",
            "levels": {"2015": "1", "2024": "1.0"},
        },
    }
    compiled = compiled_case(baseline_rules(), series)
    assert compiled.factor_for("S006", SignBranch.ANY).value == 1.0
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_DESIGN_WEIGHT_LEVEL"


def test_a_contract_with_no_design_weight_is_declined(fixture):
    """No weight rule at all is not the same statement as a declared identity."""
    series = {PRICE_BASELINE_SERIES_NAME: price_series(2015, 2024)}
    compiled = compiled_case(baseline_rules(design_weight=False), series)
    assert compiled.design_weight_rule is None
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_DESIGN_WEIGHT"


def test_an_identifier_that_is_not_recid_is_declined(fixture):
    series = {
        PRICE_BASELINE_SERIES_NAME: price_series(2015, 2024),
        DESIGN_WEIGHT_IDENTITY_SERIES: identity_weight_series(),
    }
    compiled = compiled_case(baseline_rules(identifier_field="MARS"), series)
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_IDENTIFIER"

    compiled = compiled_case(baseline_rules(identifier_year=2014), series)
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_IDENTIFIER_YEAR"


def test_a_rule_in_any_other_role_is_declined(fixture):
    """A code or count column riding along is not part of this baseline."""
    series = {
        PRICE_BASELINE_SERIES_NAME: price_series(2015, 2024),
        DESIGN_WEIGHT_IDENTITY_SERIES: identity_weight_series(),
    }
    extra = GrowthRule("MARS", FieldRole.CODE, 2015, 2015, GrowthKind.NONE)
    compiled = compiled_case(baseline_rules(extra_rules=(extra,)), series)
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_ROLE"
    assert error.field == "MARS"


def test_an_invented_fixture_table_with_the_right_factor_is_declined(fixture):
    """The authority is part of the baseline, not decoration on a receipt."""
    series = {
        PRICE_BASELINE_SERIES_NAME: price_series(2015, 2024),
        DESIGN_WEIGHT_IDENTITY_SERIES: identity_weight_series(),
    }
    compiled = compiled_case(
        baseline_rules(), series, authority="invented_fixture_nonauthority"
    )
    assert compiled.factors.authority is FactorAuthority.INVENTED_FIXTURE
    assert compiled.factor_for(AGI_FIELD, SignBranch.ANY).value == 313689 / 237017
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_AUTHORITY"


def test_the_gate_accepts_the_contract_the_lane_actually_compiles(contract):
    """The gate is closed, not shut: the adopted contract passes it."""
    assert (
        require_price_baseline_contract(
            contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
        is None
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        require_price_baseline_contract(
            object(), resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "COMPILED_TYPE"


def test_the_declared_factor_bits_constant_is_the_published_level_division():
    from_levels = float(PRICE_BASELINE_LEVELS[2024]) / float(
        PRICE_BASELINE_LEVELS[2015]
    )
    assert baseline.PRICE_BASELINE_FACTOR_BITS == struct.pack("<d", from_levels).hex()
    assert (
        baseline.PRICE_BASELINE_FACTOR_BITS
        == struct.pack("<d", PRICE_BASELINE_RATIO[0] / PRICE_BASELINE_RATIO[1]).hex()
    )


# --------------------------------------------------------------------------
# One contract, named the same way by all three objects
# --------------------------------------------------------------------------


@pytest.fixture
def other_contract():
    """The same baseline pinned to different resource bytes: a second identity."""
    return compile_price_baseline(
        PACKAGED_THIRTEEN.fields,
        **{
            **FIXTURE_RESOURCE,
            "resource_sha256": hashlib.sha256(
                FIXTURE_CPI_RESOURCE_BYTES_ALT
            ).hexdigest(),
            "resource_bytes": FIXTURE_CPI_RESOURCE_BYTES_ALT,
        },
    )


def test_the_two_contracts_differ_only_in_identity(contract, other_contract):
    assert other_contract.sha256 != contract.sha256
    assert (
        require_price_baseline_contract(
            other_contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES_ALT
        )
        is None
    )
    assert (
        other_contract.factor_for(AGI_FIELD, SignBranch.ANY).ratio_bits
        == contract.factor_for(AGI_FIELD, SignBranch.ANY).ratio_bits
    )


def test_measuring_a_table_grown_under_another_contract_is_refused(
    fixture, contract, other_contract
):
    """Every number in the receipt is attributed to the contract it names."""
    adaptation, grown = grown_for(fixture, contract)
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(
            adaptation,
            grown,
            other_contract,
            resource_bytes=FIXTURE_CPI_RESOURCE_BYTES_ALT,
        )
    assert error.value.reason == "GROWN_CONTRACT_MISMATCH"


def test_an_adaptation_from_another_contract_is_refused(
    fixture, contract, other_contract
):
    decoded, status, _, _ = fixture
    foreign = adapt_projection_to_money_view(
        decoded, status, other_contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES_ALT
    )
    grown = apply_puf_growth(foreign.money_view, contract)
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(
            foreign, grown, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "ADAPTATION_CONTRACT_MISMATCH"


def test_a_grown_result_whose_receipt_names_another_contract_is_refused(
    fixture, contract, other_contract
):
    """The digest and the receipt are two claims, and both are checked."""
    from microcosm.build.us_runtime.puf_growth import _RESULT_TOKEN

    adaptation, grown = grown_for(fixture, contract)
    receipt = json.loads(grown.receipt_bytes.decode("ascii"))
    receipt["contract_sha256"] = other_contract.sha256
    forged = GrownPufTable(
        grown.table,
        contract.sha256,
        json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii"),
        _token=_RESULT_TOKEN,
    )
    assert forged.contract_sha256 == contract.sha256
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(
            adaptation, forged, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "GROWN_RECEIPT_CONTRACT_MISMATCH"


def test_the_invariants_refuse_something_that_is_not_an_adaptation(fixture, contract):
    _, grown = grown_for(fixture, contract)
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(
            object(), grown, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "ADAPTATION_TYPE"


# --------------------------------------------------------------------------
# The exact row comparison: what the summaries cannot see
# --------------------------------------------------------------------------


def test_a_permutation_that_preserves_every_summary_is_still_refused(fixture, contract):
    """Root's counterexample, on a column no pairwise observation covers.

    Swapping two rows of ``E00200`` preserves the exact sum, every sign, every
    zero and both pairwise counts. The test asserts each of those explicitly
    before showing that the row comparison refuses anyway, so what is being
    demonstrated is that the summaries were satisfied — not merely that some
    check fired.
    """
    adaptation, grown = grown_for(fixture, contract)
    before = grown.table["E00200"].to_numpy(copy=True)
    after = before.copy()
    after[0], after[1] = before[1], before[0]
    assert not np.array_equal(after, before)
    assert math.fsum(after.tolist()) == math.fsum(before.tolist())
    assert np.array_equal(np.sign(after), np.sign(before))
    assert np.array_equal(after == 0.0, before == 0.0)
    assert "E00200" not in {
        column
        for _, greater, lesser in PAIRWISE_OBSERVATIONS
        for column in (greater, lesser)
    }
    grown.table["E00200"] = after
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(
            adaptation, grown, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "ROW_GROWTH_DISAGREEMENT"
    assert error.value.field == "E00200"


def test_a_differently_associated_rederivation_is_refused(fixture, contract):
    """``x * n / d`` is the same mathematics and not the same float64.

    A caller who recomputed the column from the exact rational instead of the
    contract's resolved double would land within the scaled-sum tolerance on
    every column, with identical signs and zeros. That is precisely the kind
    of near-miss the sums are too coarse to see.
    """
    field = "E01000"
    adaptation, grown = grown_for(fixture, contract)
    numerator, denominator = PRICE_BASELINE_RATIO
    original = adaptation.money_view[field].to_numpy().astype("float64")
    rederived = np.where(original != 0.0, original * numerator / denominator, original)
    exact = grown.table[field].to_numpy(copy=True)
    # One row of five lands on a different double, and the whole-column sum
    # moves by less than a millionth of the scaled-sum tolerance.
    assert int(np.count_nonzero(rederived.view("<u8") != exact.view("<u8"))) == 1
    difference = abs(math.fsum(rederived.tolist()) - math.fsum(exact.tolist()))
    gross = (
        float(np.abs(original).sum()) * contract.factor_for(field, SignBranch.ANY).value
    )
    assert 0.0 < difference <= 1e-12 * gross
    assert np.array_equal(np.sign(rederived), np.sign(exact))
    assert np.array_equal(rederived == 0.0, exact == 0.0)
    grown.table[field] = rederived
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(
            adaptation, grown, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "ROW_GROWTH_DISAGREEMENT"
    assert error.value.field == field


def test_a_negative_zero_written_into_a_grown_column_is_refused(fixture, contract):
    adaptation, grown = grown_for(fixture, contract)
    values = grown.table[AGI_FIELD].to_numpy(copy=True)
    zeros = np.flatnonzero(values == 0.0)
    assert zeros.size
    values[zeros[0]] = -0.0
    grown.table[AGI_FIELD] = values
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(
            adaptation, grown, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "NEGATIVE_ZERO"


def test_a_restated_provenance_column_is_refused(fixture, contract):
    """The two columns the transform copies are compared to their sources."""
    adaptation, grown = grown_for(fixture, contract)
    factor = contract.factor_for(AGI_FIELD, SignBranch.ANY).value
    grown.table[PROVENANCE_SOURCE_AGI_COLUMN] = (
        grown.table[PROVENANCE_SOURCE_AGI_COLUMN].to_numpy() * factor
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(
            adaptation, grown, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "PROVENANCE_DISAGREEMENT"
    assert error.value.field == PROVENANCE_SOURCE_AGI_COLUMN


def test_the_row_comparison_is_recorded_as_what_it_measured(fixture, contract):
    adaptation, grown = grown_for(fixture, contract)
    invariants = price_baseline_invariants(
        adaptation, grown, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
    )
    row = invariants["row_exact_growth"]
    assert row["rows"] == len(adaptation.money_view)
    assert row["disagreeing_rows"] == 0
    assert sorted(row["fields"]) == sorted(contract.money_fields)
    assert row["provenance_columns_compared"] == [
        PROVENANCE_RECID_COLUMN,
        PROVENANCE_SOURCE_AGI_COLUMN,
    ]
    named = invariants["contract_named_by"]
    assert set(named.values()) == {contract.sha256}
    assert invariants["contract_sha256"] == contract.sha256


# --------------------------------------------------------------------------
# The design weight: declared, and separately observed
# --------------------------------------------------------------------------


def test_without_a_weight_array_the_receipt_declares_and_does_not_measure(
    fixture, contract
):
    """The one claim that used to be a literal is now honestly absent."""
    adaptation, grown = grown_for(fixture, contract)
    invariants = price_baseline_invariants(
        adaptation, grown, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
    )
    assert invariants["design_weight_unchanged"] is None
    weight = invariants["design_weight"]
    assert weight["observed"] is False
    assert weight["basis"] == "declared_only_no_post_growth_weight_supplied"
    assert weight["field"] == "S006"
    assert weight["declared_factor_float64"] == 1.0
    assert weight["declared_series"] == DESIGN_WEIGHT_IDENTITY_SERIES
    assert weight["declared_return_mass_transition"] == "none"
    assert "after_sha256" not in weight
    assert "rows_changed" not in weight
    # Nothing is dressed up as a measurement. The gate has already required
    # the declared factor to be exactly 1.0, so multiplying the source weights
    # by it and reporting that nothing moved would be a tautology, not
    # evidence about this run, and the receipt does not carry one.
    assert not any("identity" in key for key in weight)
    assert (
        weight["before_sha256"]
        == hashlib.sha256(
            np.ascontiguousarray(adaptation.design_weight, dtype="<i8").tobytes()
        ).hexdigest()
    )


def test_with_the_weights_the_caller_carried_forward_it_is_measured(fixture, contract):
    adaptation, grown = grown_for(fixture, contract)
    carried = apply_puf_design_weight_growth(
        adaptation.design_weight.astype("float64"), contract
    )
    invariants = price_baseline_invariants(
        adaptation,
        grown,
        contract,
        design_weight_after=carried,
        resource_bytes=FIXTURE_CPI_RESOURCE_BYTES,
    )
    assert invariants["design_weight_unchanged"] is True
    weight = invariants["design_weight"]
    assert weight["observed"] is True
    assert weight["rows_changed"] == 0
    assert weight["basis"].startswith("bitwise float64 equality")
    assert (
        weight["after_sha256"]
        == hashlib.sha256(
            np.ascontiguousarray(carried, dtype="float64").tobytes()
        ).hexdigest()
    )


def test_a_weight_that_actually_moved_is_refused_not_reported_unchanged(
    fixture, contract
):
    adaptation, grown = grown_for(fixture, contract)
    carried = adaptation.design_weight.astype("float64").copy()
    carried[2] += 1.0
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(
            adaptation,
            grown,
            contract,
            design_weight_after=carried,
            resource_bytes=FIXTURE_CPI_RESOURCE_BYTES,
        )
    assert error.value.reason == "DESIGN_WEIGHT_CHANGED"
    assert error.value.field == "S006"


def test_a_weight_array_of_the_wrong_shape_is_refused(fixture, contract):
    adaptation, grown = grown_for(fixture, contract)
    weights = adaptation.design_weight.astype("float64")
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(
            adaptation,
            grown,
            contract,
            design_weight_after=weights[:-1],
            resource_bytes=FIXTURE_CPI_RESOURCE_BYTES,
        )
    assert error.value.reason == "DESIGN_WEIGHT_AFTER_ROWS"
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(
            adaptation,
            grown,
            contract,
            design_weight_after=weights.reshape(-1, 1),
            resource_bytes=FIXTURE_CPI_RESOURCE_BYTES,
        )
    assert error.value.reason == "DESIGN_WEIGHT_AFTER_SHAPE"
    infinite = weights.copy()
    infinite[0] = np.inf
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(
            adaptation,
            grown,
            contract,
            design_weight_after=infinite,
            resource_bytes=FIXTURE_CPI_RESOURCE_BYTES,
        )
    assert error.value.reason == "DESIGN_WEIGHT_AFTER_NONFINITE"


# --------------------------------------------------------------------------
# A subset contract says so, rather than observing nothing quietly
# --------------------------------------------------------------------------


def test_a_subset_contract_names_what_it_left_behind(fixture):
    decoded, status, _, _ = fixture
    dropped = "E00650"
    subset = compile_price_baseline(
        [f for f in PACKAGED_THIRTEEN.fields if f != dropped], **FIXTURE_RESOURCE
    )
    adaptation = adapt_projection_to_money_view(
        decoded, status, subset, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
    )
    assert adaptation.receipt["projected_fields_not_in_contract"] == [dropped]
    assert adaptation.receipt["projected_fields"] == sorted(PACKAGED_THIRTEEN.fields)
    grown = apply_puf_growth(adaptation.money_view, subset)
    invariants = price_baseline_invariants(
        adaptation, grown, subset, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
    )
    dividends = invariants["pairwise_observations"][
        "qualified_within_ordinary_dividends"
    ]
    assert dividends["status"] == "not_observable_column_absent"
    assert dividends["absent_columns"] == [dropped]
    assert dividends["violations_before"] is None
    # The relation whose columns are both present is still observed.
    pensions = invariants["pairwise_observations"]["taxable_within_total_pensions"]
    assert pensions["status"] == "observed_not_enforced"
    assert pensions["absent_columns"] == []


def test_the_sum_tolerance_is_scaled_by_the_gross_not_the_net(fixture, contract):
    """A signed column's gross is what bounds the error, and is recorded."""
    adaptation, grown = grown_for(fixture, contract)
    invariants = price_baseline_invariants(
        adaptation, grown, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
    )
    losses = invariants["columns"]["E00900"]
    assert losses["scale_basis"] == "gross_absolute_source_sum_times_factor"
    original = adaptation.money_view["E00900"].to_numpy()
    assert losses["exact_source_sum"] == int(original.sum())
    assert losses["gross_source_sum_absolute"] == int(np.abs(original).sum())
    # This fixture column cancels: the gross is strictly the larger scale.
    assert losses["gross_source_sum_absolute"] > abs(losses["exact_source_sum"])
    assert losses["relative_difference"] <= losses["relative_difference_net_scaled"]


def test_a_design_weight_on_another_series_or_another_year_is_declined(fixture):
    """The weight's own series and source year are part of the declaration."""
    series = {
        PRICE_BASELINE_SERIES_NAME: price_series(2015, 2024),
        DESIGN_WEIGHT_IDENTITY_SERIES: identity_weight_series(),
        "some.other.identity": identity_weight_series(),
    }
    compiled = compiled_case(
        baseline_rules(weight_series="some.other.identity"), series
    )
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_DESIGN_WEIGHT_SERIES"

    # ``GrowthRecipe`` already pins the weight's target year to the money
    # target year, so the open dimension is its source year.
    rules = [
        rule for rule in baseline_rules().rules if str(rule.role) != "design_weight"
    ]
    rules.append(
        GrowthRule(
            "S006",
            FieldRole.DESIGN_WEIGHT,
            2014,
            2024,
            GrowthKind.DESIGN_WEIGHT_GROWTH,
            SignBranch.ANY,
            DESIGN_WEIGHT_IDENTITY_SERIES,
            "case",
        )
    )
    # Every series must span the table's whole year set, so the price series
    # carries a 2014 column too. No money rule names it.
    compiled = compiled_case(
        GrowthRecipe("case_recipe_v1", tuple(rules)),
        {
            PRICE_BASELINE_SERIES_NAME: {
                "kind": "nominal_price_restatement",
                "levels": {"2014": "236.736", "2015": "237.017", "2024": "313.689"},
            },
            DESIGN_WEIGHT_IDENTITY_SERIES: {
                "kind": "design_weight_growth",
                "levels": {"2014": "1", "2015": "1", "2024": "1"},
            },
        },
    )
    error = refusal_from_both_entries(fixture, compiled)
    assert error.reason == "PRICE_BASELINE_DESIGN_WEIGHT_YEAR"


def test_a_grown_table_missing_a_provenance_column_is_refused(fixture, contract):
    adaptation, grown = grown_for(fixture, contract)
    del grown.table[PROVENANCE_RECID_COLUMN]
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(
            adaptation, grown, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "PROVENANCE_COLUMN_MISSING"
    assert error.value.field == PROVENANCE_RECID_COLUMN


def test_a_weight_side_that_is_not_one_dimensional_is_refused(fixture, contract):
    """The before side is checked too, not only the array a caller passes."""
    adaptation, grown = grown_for(fixture, contract)
    forged = dataclasses.replace(
        adaptation, design_weight=adaptation.design_weight.reshape(-1, 1)
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(
            forged, grown, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "DESIGN_WEIGHT_SHAPE"


def test_a_pairwise_column_present_on_only_one_side_is_refused(fixture):
    """Absent on both sides is an observation not made; absent on one is a bug."""
    decoded, status, _, _ = fixture
    subset = compile_price_baseline(
        [f for f in PACKAGED_THIRTEEN.fields if f != "E00650"], **FIXTURE_RESOURCE
    )
    adaptation = adapt_projection_to_money_view(
        decoded, status, subset, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
    )
    grown = apply_puf_growth(adaptation.money_view, subset)
    assert "E00650" not in adaptation.money_view.columns
    grown.table["E00650"] = np.zeros(len(grown.table), dtype="float64")
    with pytest.raises(PufGrowthRefusalError) as error:
        price_baseline_invariants(
            adaptation, grown, subset, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES
        )
    assert error.value.reason == "PAIRWISE_COLUMN_ASYMMETRY"
    assert error.value.field == "qualified_within_ordinary_dividends"


@pytest.mark.parametrize(
    "text",
    [
        b"313.6890",
        b"313.6890000001",
        b"3.13689e2",
        b"+313.689",
        b"NaN",
        b"inf",
        b"",
        b"\xff",
    ],
)
def test_source_level_text_must_match_exactly(text):
    body = FIXTURE_CPI_RESOURCE_BYTES.replace(b"313.689", text)
    with pytest.raises(PufGrowthRefusalError) as error:
        baseline.verify_cpi_resource_bytes(
            body, resource_sha256=hashlib.sha256(body).hexdigest()
        )
    assert error.value.reason == "PRICE_BASELINE_RESOURCE_RECORD_VALUE"
    assert error.value.field == "2024"


@pytest.mark.parametrize(
    ("old", "new"),
    [(b"CUUR0000SA0", b"CUUS0000SA0"), (b"2024", b"2023"), (b"M13", b"M12")],
)
def test_wrong_series_year_or_period_cannot_supply_a_required_record(old, new):
    body = FIXTURE_CPI_RESOURCE_BYTES.replace(old, new)
    with pytest.raises(PufGrowthRefusalError) as error:
        baseline.verify_cpi_resource_bytes(
            body, resource_sha256=hashlib.sha256(body).hexdigest()
        )
    assert error.value.reason == "PRICE_BASELINE_RESOURCE_RECORD_MISSING"


@pytest.mark.parametrize("year,level", [(b"2015", b"237.017"), (b"2024", b"313.689")])
def test_conflicting_or_repeated_required_records_are_refused(year, level):
    body = (
        FIXTURE_CPI_RESOURCE_BYTES + b"CUUR0000SA0\t" + year + b"\tM13\t999.999\t\r\n"
    )
    with pytest.raises(PufGrowthRefusalError) as error:
        baseline.verify_cpi_resource_bytes(
            body, resource_sha256=hashlib.sha256(body).hexdigest()
        )
    assert error.value.reason == "PRICE_BASELINE_RESOURCE_DUPLICATE_RECORD"
    assert error.value.field == year.decode("ascii")


def test_bls_field_padding_and_lf_or_crlf_do_not_change_the_record_text():
    body = FIXTURE_CPI_RESOURCE_BYTES.replace(b"\r\n", b"\n").replace(b"     ", b" ")
    receipt = baseline.verify_cpi_resource_bytes(
        bytearray(body), resource_sha256=hashlib.sha256(body).hexdigest()
    )
    assert receipt["records"]["2015"]["level_text"] == "237.017"
    assert receipt["records"]["2024"]["level_text"] == "313.689"


@pytest.mark.parametrize(
    "entry", ["factor_document", "compile", "gate", "adapt", "invariants"]
)
def test_every_baseline_entry_requires_resource_bytes(entry, fixture, contract):
    decoded, status, _, _ = fixture
    arguments = {k: v for k, v in FIXTURE_RESOURCE.items() if k != "resource_bytes"}
    with pytest.raises(TypeError, match="resource_bytes"):
        if entry == "factor_document":
            price_baseline_factor_document(**arguments)
        elif entry == "compile":
            compile_price_baseline(PACKAGED_THIRTEEN.fields, **arguments)
        elif entry == "gate":
            require_price_baseline_contract(contract)
        elif entry == "adapt":
            adapt_projection_to_money_view(decoded, status, contract)
        else:
            price_baseline_invariants(object(), object(), contract)


@pytest.mark.parametrize(
    "entry", ["factor_document", "compile", "gate", "adapt", "invariants"]
)
def test_every_baseline_entry_authenticates_the_named_digest(entry, fixture, contract):
    decoded, status, _, _ = fixture
    arguments = {**FIXTURE_RESOURCE, "resource_bytes": FIXTURE_CPI_RESOURCE_BYTES_ALT}
    with pytest.raises(PufGrowthRefusalError) as error:
        if entry == "factor_document":
            price_baseline_factor_document(**arguments)
        elif entry == "compile":
            compile_price_baseline(PACKAGED_THIRTEEN.fields, **arguments)
        elif entry == "gate":
            require_price_baseline_contract(
                contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES_ALT
            )
        elif entry == "adapt":
            adapt_projection_to_money_view(
                decoded, status, contract, resource_bytes=FIXTURE_CPI_RESOURCE_BYTES_ALT
            )
        else:
            price_baseline_invariants(
                object(),
                object(),
                contract,
                resource_bytes=FIXTURE_CPI_RESOURCE_BYTES_ALT,
            )
    assert error.value.reason == "PRICE_BASELINE_RESOURCE_DIGEST"


@pytest.mark.parametrize("entry", ["gate", "adapt", "invariants"])
def test_generic_growth_compiler_cannot_bypass_record_verification(entry, fixture):
    body = FIXTURE_CPI_RESOURCE_BYTES.replace(b"313.689", b"999.999")
    document = json.loads(price_baseline_factor_document(**FIXTURE_RESOURCE))
    # The generic format accepts a declared resource digest. Baseline entry
    # points must authenticate it even when their own factory was bypassed.
    document["provenance"]["resource_sha256"] = hashlib.sha256(body).hexdigest()
    compiled = compile_puf_growth(
        price_baseline_recipe(PACKAGED_THIRTEEN.fields),
        GrowthFactorTable.from_bytes(
            json.dumps(
                document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("ascii")
        ),
    )
    decoded, status, _, _ = fixture
    with pytest.raises(PufGrowthRefusalError) as error:
        if entry == "gate":
            require_price_baseline_contract(compiled, resource_bytes=body)
        elif entry == "adapt":
            adapt_projection_to_money_view(
                decoded, status, compiled, resource_bytes=body
            )
        else:
            price_baseline_invariants(object(), object(), compiled, resource_bytes=body)
    assert error.value.reason == "PRICE_BASELINE_RESOURCE_RECORD_VALUE"
