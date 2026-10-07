"""Authenticated synthetic donor, exact roster, seed and currency contracts."""

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.frame import SignedScale
from microcosm.frame.adapters.policyengine_us_concepts import (
    POLICYENGINE_US_CONCEPT_MAPPING,
)
from microcosm.frame.concept_mapping import ConceptMapping
from microcosm.frame.concepts import ContentBasis, validate_concept_tables
from microcosm.frame.transport import (
    currency_bridge,
    derive_transport_seed,
    quantile_map,
    read_populace_us_donor,
)
from test_support.paths import paths_for

FIXTURE = paths_for("microcosm-frame").tests / "fixtures/transport/synthetic_donor.json"
PROPERTY = settings(max_examples=75, deadline=None, database=None)
READER_PROPERTY = settings(
    max_examples=40,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _tables() -> dict[str, pd.DataFrame]:
    return {entity: pd.DataFrame(rows) for entity, rows in _fixture()["tables"].items()}


def _pin(path: Path) -> dict:
    return {
        "size": path.stat().st_size,
        "sha256": sha256(path.read_bytes()).hexdigest(),
    }


def _write_h5(path: Path, tables: dict[str, pd.DataFrame]) -> dict:
    with pd.HDFStore(path, mode="w") as store:
        for entity, table in tables.items():
            store.put(entity, table, format="table")
    return _pin(path)


def _fake_donor(monkeypatch, tmp_path, tables):
    """Exercise public authentication/reader without repeated fixture H5 writes."""
    path = tmp_path / "synthetic-roster.h5"
    path.write_bytes(b"synthetic donor bytes for mocked HDF decoding")

    class SyntheticStore:
        def __init__(self, path, *, mode):
            assert mode == "r"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def __getitem__(self, key):
            return tables[key].copy(deep=True)

        def get_storer(self, _):
            return SimpleNamespace(attrs=SimpleNamespace())

    monkeypatch.setattr(pd, "HDFStore", SyntheticStore)
    return read_populace_us_donor(path, **_pin(path))


def test_synthetic_donor_matches_decode_and_exact_fixture(tmp_path, monkeypatch):
    raw = _tables()
    path = tmp_path / "synthetic.h5"
    pin = _write_h5(path, raw)
    original_decode = ConceptMapping.decode
    seen = []

    def spy_decode(mapping, tables):
        seen.append({entity: table.copy(deep=True) for entity, table in tables.items()})
        return original_decode(mapping, tables)

    monkeypatch.setattr(ConceptMapping, "decode", spy_decode)
    donor = read_populace_us_donor(path, **pin)
    assert len(seen) == 1
    expected_decode = original_decode(POLICYENGINE_US_CONCEPT_MAPPING, seen[0])
    for entity, table in expected_decode.items():
        # Roster pointers use nullable Int64 even when the decoded reference
        # pointer contains no nulls and pandas chooses a NumPy integer dtype.
        pd.testing.assert_frame_equal(
            donor.tables[entity][table.columns], table, check_dtype=False
        )
    expected = _fixture()["expected"]
    person = donor.tables["person"]
    for column in ("partner_person_id", "parent_1_person_id", "parent_2_person_id"):
        pd.testing.assert_series_equal(
            person[column],
            pd.Series(pd.array(expected[column], dtype="Int64"), name=column),
        )
    np.testing.assert_array_equal(
        donor.tables["household"]["reference_person_id"],
        expected["reference_person_id"],
    )
    np.testing.assert_array_equal(
        person["liquid_financial_assets"], expected["liquid_financial_assets"]
    )
    np.testing.assert_array_equal(
        person["employment_income"], expected["employment_income"]
    )
    np.testing.assert_array_equal(donor.weights, raw["household"]["household_weight"])
    np.testing.assert_array_equal(
        donor.source_person_ids, raw["person"]["source_person_id"]
    )
    assert donor.weights.dtype == np.float64
    assert donor.support_strata.tolist() == ["us:asec", "us:puf_tax_detail"]
    assert donor.content_basis is ContentBasis.TRANSPORT
    assert donor.donor_country == "us"
    assert donor.currency == "USD"
    assert validate_concept_tables(donor.tables) == ()
    assert "A_LINENO" not in person and "bank_account_assets" not in person
    assert "household_weight" not in donor.tables["household"]


def test_synthetic_donor_transport_and_mapping_round_trip(tmp_path):
    document = _fixture()
    raw = _tables()
    path = tmp_path / "roundtrip.h5"
    donor = read_populace_us_donor(path, **_write_h5(path, raw))
    person = donor.tables["person"]
    weights = (
        person["person_household_id"]
        .map(pd.Series(donor.weights, index=donor.tables["household"]["household_id"]))
        .to_numpy()
    )
    before = {entity: table.copy(deep=True) for entity, table in donor.tables.items()}
    original_weights = donor.weights.copy()
    bands = document["target_bands"]
    employment = quantile_map(
        person["employment_income"], weights, bands, interpolation="uniform"
    )
    assets = quantile_map(
        person["liquid_financial_assets"],
        weights,
        bands,
        interpolation="uniform",
        group={"entity_ids": person["person_household_id"]},
    )
    np.testing.assert_allclose(
        employment, document["expected"]["mapped_employment_income"]
    )
    np.testing.assert_allclose(
        assets, document["expected"]["mapped_liquid_financial_assets"]
    )
    np.testing.assert_allclose(
        quantile_map(employment, weights, bands, interpolation="uniform"), employment
    )
    np.testing.assert_allclose(
        quantile_map(
            assets,
            weights,
            bands,
            interpolation="uniform",
            group={"entity_ids": person["person_household_id"]},
        ),
        assets,
    )
    transported = {
        entity: table.copy(deep=True) for entity, table in donor.tables.items()
    }
    transported["person"]["employment_income"] = employment
    encoded = POLICYENGINE_US_CONCEPT_MAPPING.encode(
        transported, shares={"us.taxable_interest_share": 0.6}
    )
    decoded = POLICYENGINE_US_CONCEPT_MAPPING.decode(encoded.tables)
    for entity, table in decoded.items():
        pd.testing.assert_frame_equal(
            table, transported[entity][table.columns], check_dtype=False
        )
    for entity, table in before.items():
        pd.testing.assert_frame_equal(donor.tables[entity], table)
    np.testing.assert_array_equal(donor.weights, original_weights)


@pytest.mark.parametrize("mismatch", ["size", "sha256"])
def test_reader_refuses_pin_mismatch_before_hdf_parse(tmp_path, monkeypatch, mismatch):
    path = tmp_path / "not-an-hdf.h5"
    path.write_bytes(b"not an HDF file")
    pin = _pin(path)
    pin[mismatch] = pin[mismatch] + 1 if mismatch == "size" else "0" * 64
    parse = Mock(side_effect=AssertionError("HDF parsing preceded authentication"))
    monkeypatch.setattr(pd, "HDFStore", parse)
    with pytest.raises(ValueError, match="mismatch"):
        read_populace_us_donor(path, **pin)
    parse.assert_not_called()


@READER_PROPERTY
@given(
    payload=st.binary(min_size=1, max_size=128),
    excess_size=st.integers(min_value=1, max_value=1_000_000),
)
def test_reader_pin_mismatch_property_precedes_hdf_parse(
    tmp_path, monkeypatch, payload, excess_size
):
    path = tmp_path / "generated-donor.h5"
    path.write_bytes(payload)
    actual = _pin(path)
    altered_digest = ("0" if actual["sha256"][0] != "0" else "1") + actual["sha256"][1:]
    parse = Mock(side_effect=AssertionError("HDF parsing preceded authentication"))
    monkeypatch.setattr(pd, "HDFStore", parse)
    for pin in (
        {**actual, "size": actual["size"] + excess_size},
        {**actual, "sha256": altered_digest},
    ):
        with pytest.raises(ValueError, match="mismatch"):
            read_populace_us_donor(path, **pin)
    parse.assert_not_called()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("dangling_partner", "unknown household line"),
        ("dangling_parent", "unknown household line"),
        ("dangling_cohabitant", "unknown household line"),
        ("noninteger_pointer", "integer line numbers"),
        ("infinite_pointer", "integer line numbers"),
        ("asymmetric_partner", "Partners must point at each other"),
        ("asymmetric_cohabitant", "Partners must point at each other"),
        ("spouse_cohabitant_conflict", "name different partners"),
        ("duplicate_line", "unique within a household"),
        ("multiple_reference", "multiple reference persons"),
        ("missing_reference", "no reference person"),
        ("multiple_heads", "multiple is_household_head persons"),
        ("head_disagrees", "reference person and is_household_head disagree"),
        ("head_absent", "reference person and is_household_head disagree"),
        ("channel_mismatch", "support channels disagree"),
    ],
)
def test_reader_refuses_invalid_rosters(tmp_path, monkeypatch, change, message):
    tables = _tables()
    person = tables["person"]
    mutations = {
        "dangling_partner": [(0, "A_SPOUSE", 99)],
        "dangling_parent": [(2, "PEPAR1", 99)],
        "dangling_cohabitant": [(2, "PECOHAB", 99)],
        # Only whole nonpositive or missing pointers are absent; fractional
        # or infinite ones still fail.
        "noninteger_pointer": [(2, "PEPAR1", -1.5)],
        "infinite_pointer": [(2, "PEPAR1", -np.inf)],
        "asymmetric_partner": [(1, "A_SPOUSE", 0)],
        "asymmetric_cohabitant": [
            (3, "A_SPOUSE", 0),
            (4, "A_SPOUSE", 0),
            (3, "PECOHAB", 2),
        ],
        "spouse_cohabitant_conflict": [(0, "PECOHAB", 3)],
        "duplicate_line": [(1, "A_LINENO", 1)],
        "multiple_reference": [(1, "A_EXPRRP", 1)],
        # Neither an A_EXPRRP reference person nor a household head.
        "missing_reference": [(0, "A_EXPRRP", 5), (0, "is_household_head", False)],
        "multiple_heads": [(4, "is_household_head", True)],
        "head_disagrees": [
            (0, "is_household_head", False),
            (1, "is_household_head", True),
        ],
        "head_absent": [(0, "is_household_head", False)],
        "channel_mismatch": [(0, "person_support_channel", "another-channel")],
    }
    for row, column, value in mutations[change]:
        if isinstance(value, float) and person[column].dtype.kind == "i":
            person[column] = person[column].astype(np.float64)
        person.loc[row, column] = value
    with pytest.raises(ValueError, match=message):
        _fake_donor(monkeypatch, tmp_path, tables)


@pytest.mark.parametrize(
    "flags",
    [
        pd.Series([1, 0, 0, 1, 0, 0]),
        pd.Series([True, False, None] * 2, dtype="boolean"),
    ],
    ids=["integer", "nullable"],
)
def test_reader_requires_a_boolean_household_head_flag(tmp_path, monkeypatch, flags):
    tables = _tables()
    tables["person"]["is_household_head"] = flags
    with pytest.raises(ValueError, match="is_household_head must be boolean"):
        _fake_donor(monkeypatch, tmp_path, tables)


@pytest.mark.parametrize("absent", [-1, 0, np.nan])
def test_reader_reads_nonpositive_or_missing_pointers_as_absent(
    tmp_path, monkeypatch, absent
):
    tables = _tables()
    person = tables["person"]
    for column in ("A_SPOUSE", "PEPAR1", "PEPAR2", "PECOHAB"):
        values = person[column].astype(np.float64)
        person[column] = values.where(values > 0, absent)
    donor = _fake_donor(monkeypatch, tmp_path, tables)
    expected = _fixture()["expected"]
    for column in ("partner_person_id", "parent_1_person_id", "parent_2_person_id"):
        pd.testing.assert_series_equal(
            donor.tables["person"][column],
            pd.Series(pd.array(expected[column], dtype="Int64"), name=column),
        )


def test_reader_takes_household_head_when_no_a_exprrp_reference_person(
    tmp_path, monkeypatch
):
    tables = _tables()
    person = tables["person"]
    # Household 202 has no A_EXPRRP reference person, and its head flag marks
    # line 2 rather than line 1, so only the fallback can name that person.
    person.loc[3, "A_EXPRRP"] = 3
    person.loc[3, "is_household_head"] = False
    person.loc[4, "is_household_head"] = True
    donor = _fake_donor(monkeypatch, tmp_path, tables)
    assert donor.tables["household"]["reference_person_id"].tolist() == [
        person.loc[0, "person_id"],
        person.loc[4, "person_id"],
    ]
    assert validate_concept_tables(donor.tables) == ()


@READER_PROPERTY
@given(
    offset=st.integers(min_value=2**53 + 1, max_value=2**63 - 7),
    order=st.permutations(tuple(range(6))),
    reverse_households=st.booleans(),
)
def test_pointer_recode_is_exact_bijective_and_partner_symmetric(
    tmp_path, monkeypatch, offset, order, reverse_households
):
    tables = _tables()
    tables["person"]["person_id"] = np.arange(offset, offset + 6, dtype=np.int64)
    tables["person"] = tables["person"].iloc[list(order)].reset_index(drop=True)
    if reverse_households:
        tables["household"] = tables["household"].iloc[::-1].reset_index(drop=True)
    donor = _fake_donor(monkeypatch, tmp_path, tables)
    raw = tables["person"]
    lookup = {
        (row.person_household_id, row.A_LINENO): row.person_id
        for row in raw.itertuples()
    }
    assert len(lookup) == len(raw)
    concept = donor.tables["person"]
    partners = concept.set_index("person_id")["partner_person_id"]
    for row, recovered in zip(raw.itertuples(), concept.itertuples(), strict=True):
        assert recovered.person_id == lookup[(row.person_household_id, row.A_LINENO)]
        if row.A_SPOUSE:
            partner = lookup[(row.person_household_id, row.A_SPOUSE)]
            assert recovered.partner_person_id == partner
            assert partners.loc[partner] == row.person_id
        else:
            assert pd.isna(recovered.partner_person_id)
        parents = [
            lookup[(row.person_household_id, line)]
            for line in (row.PEPAR1, row.PEPAR2)
            if line > 0
        ]
        for position, column in enumerate(("parent_1_person_id", "parent_2_person_id")):
            observed = getattr(recovered, column)
            if position < len(parents):
                assert observed == parents[position]
            else:
                assert pd.isna(observed)
    for row in donor.tables["household"].itertuples():
        assert row.reference_person_id == lookup[(row.household_id, 1)]
    np.testing.assert_array_equal(donor.source_person_ids, raw["source_person_id"])
    np.testing.assert_array_equal(
        donor.weights, tables["household"]["household_weight"]
    )


def test_reader_preserves_undifferentiated_us_support_and_exact_id_fallback(
    tmp_path, monkeypatch
):
    tables = _tables()
    tables["person"] = tables["person"].drop(
        columns=[
            "person_source_id",
            "source_person_id",
            "source_household_id",
            "source_year",
            "person_support_channel",
        ]
    )
    tables["household"] = tables["household"].drop(columns="household_support_channel")
    donor = _fake_donor(monkeypatch, tmp_path, tables)
    assert donor.support_strata.tolist() == ["us", "us"]
    np.testing.assert_array_equal(
        donor.source_person_ids, tables["person"]["person_id"]
    )


def test_reader_keeps_repeated_exact_source_ids_on_support_clones(
    tmp_path, monkeypatch
):
    tables = _tables()
    tables["person"].loc[3, "person_source_id"] = tables["person"].loc[
        0, "person_source_id"
    ]
    donor = _fake_donor(monkeypatch, tmp_path, tables)
    assert donor.source_person_ids[0] == donor.source_person_ids[3]
    seeds = derive_transport_seed(donor.source_person_ids, "synthetic-clone")
    assert seeds[0] == seeds[3]


@pytest.mark.parametrize("scope", ["household", "year"])
def test_reader_scopes_repeated_person_id_to_exact_source_lineage(
    tmp_path, monkeypatch, scope
):
    tables = _tables()
    person = tables["person"].drop(columns="person_source_id")
    tables["person"] = person
    tables["household"]["household_support_channel"] = "asec"
    person["person_support_channel"] = "asec"
    exact_leaf = person.loc[0, "source_person_id"]
    person.loc[3, "source_person_id"] = exact_leaf
    if scope == "year":
        person.loc[3, "source_household_id"] = person.loc[0, "source_household_id"]
        person.loc[3, "source_year"] = person.loc[0, "source_year"] - 1
    donor = _fake_donor(monkeypatch, tmp_path, tables)
    assert donor.source_person_ids[0] != donor.source_person_ids[3]
    for index in (0, 3):
        composite = json.loads(donor.source_person_ids[index])
        assert composite[0] == "us:asec"
        assert composite[1] == ["int", str(person.loc[index, "source_year"])]
        assert composite[2] == ["int", str(person.loc[index, "source_household_id"])]
        assert composite[3] == ["str", exact_leaf]
    seeds = derive_transport_seed(donor.source_person_ids, "synthetic-lineage")
    assert seeds[0] != seeds[3]


@pytest.mark.parametrize(
    "missing", ["source_year", "source_household_id", "source_person_id"]
)
def test_reader_refuses_partial_source_lineage(tmp_path, monkeypatch, missing):
    tables = _tables()
    tables["person"] = tables["person"].drop(columns=["person_source_id", missing])
    with pytest.raises(ValueError, match="source lineage requires"):
        _fake_donor(monkeypatch, tmp_path, tables)


@pytest.mark.parametrize("housemates", [1, 2])
def test_reader_never_reads_relationship_code_13_as_a_partner(
    tmp_path, monkeypatch, housemates
):
    tables = _tables()
    person = tables["person"]
    # Code 13 also covers roommates and housemates. Household 202's reference
    # person shares the home with one or two code-13 housemates and no spouse
    # or cohabiting-partner pointer, so nobody in it has a partner.
    person.loc[[3, 4], "A_SPOUSE"] = 0
    person.loc[4, "A_EXPRRP"] = 13
    if housemates == 2:
        person.loc[5, ["A_EXPRRP", "PEPAR2"]] = [13, -1]
    donor = _fake_donor(monkeypatch, tmp_path, tables)
    partners = donor.tables["person"]["partner_person_id"]
    assert partners.iloc[3:].isna().all()
    assert partners.iloc[:2].tolist() == [
        person.loc[1, "person_id"],
        person.loc[0, "person_id"],
    ]
    assert validate_concept_tables(donor.tables) == ()


def test_reader_recodes_cohabiting_partner_pointers_symmetrically(
    tmp_path, monkeypatch
):
    tables = _tables()
    person = tables["person"]
    # Household 202's couple cohabits: no spouse pointer, a PECOHAB pair.
    person.loc[[3, 4], "A_SPOUSE"] = 0
    person.loc[[3, 4], "PECOHAB"] = [2, 1]
    donor = _fake_donor(monkeypatch, tmp_path, tables)
    partners = donor.tables["person"]["partner_person_id"]
    assert partners.iloc[3] == person.loc[4, "person_id"]
    assert partners.iloc[4] == person.loc[3, "person_id"]
    assert validate_concept_tables(donor.tables) == ()


@pytest.mark.parametrize("cohabitation", ["agrees_with_spouse", "column_absent"])
def test_reader_spouse_pointers_hold_with_or_without_pecohab(
    tmp_path, monkeypatch, cohabitation
):
    tables = _tables()
    if cohabitation == "agrees_with_spouse":
        # PECOHAB may repeat a spouse pointer; only different partners fail.
        tables["person"].loc[[0, 1], "PECOHAB"] = [2, 1]
    else:
        tables["person"] = tables["person"].drop(columns="PECOHAB")
    donor = _fake_donor(monkeypatch, tmp_path, tables)
    pd.testing.assert_series_equal(
        donor.tables["person"]["partner_person_id"],
        pd.Series(
            pd.array(_fixture()["expected"]["partner_person_id"], dtype="Int64"),
            name="partner_person_id",
        ),
    )


def test_cached_real_donor_size_only_when_present():
    pin = _fixture()["real_donor_pin"]
    path = Path.home() / pin["cache_relative_path"]
    if not path.is_file():
        pytest.skip("Pinned real donor is not locally cached")
    assert path.stat().st_size == pin["size"]


def test_pinned_real_donor_reaches_the_reference_person_ruling_when_cached():
    # About 10 s: hashes the 463 MB pin, reads both tables and stages every
    # line pointer, which must all resolve, before the reference-person
    # check. #1130 is waiting on a ruling: in 113 pinned households (all
    # 2022/2023 rows, whose A_EXPRRP the build derived from line 1) the
    # A_EXPRRP reference person is not the is_household_head (P_SEQ == 1)
    # person, so the reader refuses the file. Once that is ruled, assert the
    # bank instead; one source household's two copies also hold a
    # parent-pointer cycle that concept validation refuses.
    pin = _fixture()["real_donor_pin"]
    path = Path.home() / pin["cache_relative_path"]
    if not path.is_file():
        pytest.skip("Pinned real donor is not locally cached")
    with pytest.raises(
        ValueError,
        match=r"^Donor A_EXPRRP reference person and is_household_head disagree\.$",
    ):
        read_populace_us_donor(path, sha256=pin["sha256"], size=pin["size"])


@PROPERTY
@given(
    ids=st.lists(
        st.one_of(st.integers(min_value=0, max_value=2**100), st.text(min_size=1)),
        max_size=30,
    ),
    stream=st.text(min_size=1),
)
def test_transport_seed_range_determinism_order_and_clone_invariance(ids, stream):
    first = derive_transport_seed(ids, stream)
    assert first.dtype == np.float64
    assert np.all((first >= 0) & (first < 1))
    np.testing.assert_array_equal(first, derive_transport_seed(ids, stream))
    np.testing.assert_array_equal(first[::-1], derive_transport_seed(ids[::-1], stream))
    np.testing.assert_array_equal(
        np.repeat(first, 2),
        derive_transport_seed(np.repeat(np.asarray(ids, dtype=object), 2), stream),
    )


def test_transport_seed_separates_streams_and_exact_typed_ids():
    ids = [2**53 + 1, 2**53 + 2, "0000000000000000000001", "1", 1]
    first = derive_transport_seed(ids, "synthetic-a")
    second = derive_transport_seed(ids, "synthetic-b")
    assert len(np.unique(first)) == len(ids)
    assert np.all(first != second)


@pytest.mark.parametrize("ids", [[None], [True], [1.0], [np.nan], [""], [[1]]])
def test_transport_seed_refuses_inexact_or_missing_ids(ids):
    with pytest.raises(ValueError, match="Source person IDs"):
        derive_transport_seed(ids, "synthetic")


@pytest.mark.parametrize("stream", ["", None, 1])
def test_transport_seed_requires_explicit_nonempty_stream(stream):
    with pytest.raises(ValueError, match="stream"):
        derive_transport_seed([1], stream)


@PROPERTY
@given(
    values=st.lists(
        st.floats(min_value=-1e8, max_value=1e8, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=30,
    ),
    rate=st.floats(
        min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False
    ),
)
def test_currency_bridge_matches_exactly_one_multiplication_and_preserves_inputs(
    values, rate
):
    tables = {
        "person": pd.DataFrame(
            {
                "rental_income": values,
                "age": np.arange(len(values)),
                "employment_income": np.ones(len(values)),
            }
        ),
        "household": pd.DataFrame(
            {"household_id": [1], "rent": [10.0], "household_weight": [7.0]}
        ),
    }
    before = {entity: table.copy(deep=True) for entity, table in tables.items()}
    result = currency_bridge(
        tables, {"person": ["rental_income"], "household": ["rent"]}, rate
    )
    np.testing.assert_array_equal(
        result["person"]["rental_income"], np.round(np.asarray(values) * rate, 2)
    )
    np.testing.assert_array_equal(
        result["household"]["rent"], np.round(np.array([10.0]) * rate, 2)
    )
    pd.testing.assert_frame_equal(
        result["person"].drop(columns="rental_income"),
        before["person"].drop(columns="rental_income"),
    )
    pd.testing.assert_frame_equal(
        result["household"].drop(columns="rent"),
        before["household"].drop(columns="rent"),
    )
    for entity, table in before.items():
        pd.testing.assert_frame_equal(tables[entity], table)


@pytest.mark.parametrize(
    "columns",
    [
        {"person": ["age"]},
        {"person": ["unknown"]},
        {"person": ["rental_income", "rental_income"]},
        {"person": "rental_income"},
        {"missing": ["rent"]},
    ],
)
def test_currency_bridge_refuses_nonamount_missing_duplicate_or_unstructured_columns(
    columns,
):
    tables = {
        "person": pd.DataFrame({"rental_income": [1.0], "age": [30]}),
        "household": pd.DataFrame({"rent": [1.0]}),
    }
    with pytest.raises(ValueError):
        currency_bridge(tables, columns, 1.25)


@pytest.mark.parametrize("rate", [0.0, -1.0, np.inf, np.nan])
def test_currency_bridge_refuses_invalid_declared_rate(rate):
    with pytest.raises(ValueError, match="finite and positive"):
        currency_bridge(
            {"person": pd.DataFrame({"rental_income": [1.0]})},
            {"person": ["rental_income"]},
            rate,
        )


@pytest.mark.parametrize("rate", [True, False, SignedScale(1.0, 2.0)])
def test_currency_bridge_requires_one_numeric_scalar_rate(rate):
    with pytest.raises(ValueError):
        currency_bridge(
            {"person": pd.DataFrame({"rental_income": [1.0]})},
            {"person": ["rental_income"]},
            rate,
        )


def test_currency_bridge_refuses_nonfinite_rounding_result():
    with pytest.raises(ValueError, match="finite"):
        currency_bridge(
            {"person": pd.DataFrame({"rental_income": [np.finfo(np.float64).max]})},
            {"person": ["rental_income"]},
            1.0,
        )
