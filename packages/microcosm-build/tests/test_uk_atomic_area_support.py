"""Invented array/identity controls only; no publisher files or country engine."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build import atomic_geography as geo
from microcosm.build.uk_runtime.atomic_area_support import (
    IDENTITY_COLUMN,
    SOURCES,
    SYSTEMS,
    assemble_uk_atomic_area_support,
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.rowwise_geography import FRS_REGION_TO_REGION_CODE


def parts(system):
    index = SYSTEMS.index(system)
    regions = (
        [
            r
            for r in FRS_REGION_TO_REGION_CODE
            if r not in {"SCOTLAND", "NORTHERN_IRELAND"}
        ]
        if index == 0
        else ["SCOTLAND"]
        if index == 1
        else ["NORTHERN_IRELAND"]
    )
    rows = []
    region_order = list(FRS_REGION_TO_REGION_CODE)
    for region in regions:
        prefix = FRS_REGION_TO_REGION_CODE[region][0]
        for offset in range(3):
            serial = region_order.index(region) * 3 + offset + 1
            area = prefix + ("20" if index == 2 else "00") + f"{serial:06d}"
            rows.append(
                {
                    "oa_code": area,
                    "population": [2.0, 6.0, 1000.0][offset],
                    "households": [0.0, 30.0, 10.0][offset],
                    "constituency_code": prefix
                    + "14"
                    + f"{serial - offset + (offset == 2):06d}",
                    "region_code": FRS_REGION_TO_REGION_CODE[region],
                    "lsoa_code": area
                    if index == 2
                    else prefix + "01" + f"{serial:06d}",
                    "msoa_code": prefix + "02" + f"{serial:06d}",
                    "local_authority_code": prefix + "06" + f"{serial:06d}",
                    "ward_code": prefix + "05" + f"{serial:06d}",
                    "itl3_code": "TL"
                    + "CDEFGHIJKLMN"[region_order.index(region)]
                    + "01",
                }
            )
    arrays = {column: np.asarray([r[column] for r in rows]) for column in rows[0]}
    descriptions = {}
    for column in arrays:
        if column in {"population", "households"}:
            descriptions[column] = {
                "kind": "weight",
                "source": "invented-" + system + "-" + column,
                "basis": "invented persons"
                if column == "population"
                else "invented occupied households",
            }
        else:
            descriptions[column] = {
                "kind": "code",
                "source": "invented-" + system + "-" + column,
                "vintage": "2022_census" if index == 1 else "2021_census",
                "relation": "exact"
                if column in {"oa_code", "lsoa_code", "msoa_code", "region_code"}
                else "best_fit",
            }
    descriptions["constituency_code"]["vintage"] = "2024_pcon"
    if index == 2:
        descriptions["constituency_code"]["relation"] = "official_tabulation"
    return arrays, descriptions


def payloads():
    return {
        system: assemble_uk_atomic_area_support(
            system=system, arrays=arrays, column_metadata=metadata
        )
        for system in SYSTEMS
        for arrays, metadata in [parts(system)]
    }


def key(source_id=101, path=()):
    return household_draw_key(
        source="invented-frs",
        source_vintage="invented-2024-25",
        source_household_id=source_id,
        clone_path=path,
    )


def complete(households, payload):
    spec = uk_atomic_assignment_definition(payload, seed=43)
    supports = {
        system: geo.decode_atomic_support(data) for system, data in payload.items()
    }
    result = pd.concat(
        [households, geo.assign_atomic(households, spec, supports)], axis=1
    )
    result = pd.concat([result, geo.derive_geography(result, spec, supports)], axis=1)
    return result, spec, supports


def test_normalization_preserves_all_rows_counts_mappings_and_source_labels():
    for system in SYSTEMS:
        arrays, metadata = parts(system)
        original = {k: v.copy() for k, v in arrays.items()}
        data = assemble_uk_atomic_area_support(
            system=system, arrays=arrays, column_metadata=metadata
        )
        support = geo.decode_atomic_support(data)
        order = np.argsort(arrays["oa_code"], kind="stable")
        for column in arrays:
            normalized = "area" if column == "oa_code" else column
            np.testing.assert_array_equal(
                support.arrays[normalized], arrays[column][order]
            )
            assert dict(support.metadata["columns"][normalized]) == metadata[column]
            np.testing.assert_array_equal(arrays[column], original[column])
        assert support.arrays["population"].dtype == np.dtype("int64")
        assert support.arrays["households"].dtype == np.dtype("int64")
        assert support.arrays["area"].flags.writeable is False
        assert support.sha256 == geo.decode_atomic_support(data).sha256


def test_normalized_bytes_are_stable_under_input_row_order_and_integer_count_storage():
    arrays, metadata = parts(SYSTEMS[0])
    expected = assemble_uk_atomic_area_support(
        system=SYSTEMS[0], arrays=arrays, column_metadata=metadata
    )
    reversed_arrays = {k: v[::-1].copy() for k, v in arrays.items()}
    for column in ("population", "households"):
        reversed_arrays[column] = reversed_arrays[column].astype(np.int64)
    assert (
        assemble_uk_atomic_area_support(
            system=SYSTEMS[0], arrays=reversed_arrays, column_metadata=metadata
        )
        == expected
    )


@pytest.mark.parametrize(
    "defect",
    [
        "nonintegral",
        "float32",
        "bool",
        "negative",
        "nan",
        "overflow_total",
        "zero_population",
        "zero_region_households",
        "duplicate_area",
        "length",
        "missing_column",
        "object_code",
        "foreign_nation",
        "cross_ew_nation",
        "blank_code",
        "missing_region",
        "bad_atomic_vintage",
        "missing_source",
    ],
)
def test_converter_refuses_bad_counts_codes_coverage_and_metadata(defect):
    arrays, metadata = parts(SYSTEMS[0])
    if defect == "nonintegral":
        arrays["households"][0] = 0.5
    elif defect == "float32":
        arrays["households"] = arrays["households"].astype(np.float32)
    elif defect == "bool":
        arrays["households"] = arrays["households"] > 0
    elif defect == "negative":
        arrays["households"][0] = -1
    elif defect == "nan":
        arrays["population"][0] = np.nan
    elif defect == "overflow_total":
        arrays["population"][0] = float(2**53)
    elif defect == "zero_population":
        arrays["population"][0] = 0
    elif defect == "zero_region_households":
        arrays["households"][:3] = 0
    elif defect == "duplicate_area":
        arrays["oa_code"][1] = arrays["oa_code"][0]
    elif defect == "length":
        arrays["msoa_code"] = arrays["msoa_code"][:-1]
    elif defect == "missing_column":
        del arrays["population"]
    elif defect == "object_code":
        arrays["oa_code"] = arrays["oa_code"].astype(object)
    elif defect == "foreign_nation":
        arrays["ward_code"][0] = "S05000001"
    elif defect == "cross_ew_nation":
        arrays["ward_code"][0] = "W05000001"
    elif defect == "blank_code":
        arrays["ward_code"][0] = ""
    elif defect == "missing_region":
        arrays = {k: v[3:].copy() for k, v in arrays.items()}
    elif defect == "bad_atomic_vintage":
        metadata["oa_code"]["vintage"] = "2011_census"
    elif defect == "missing_source":
        metadata["population"]["source"] = ""
    with pytest.raises(ValueError, match="UK atomic support:"):
        assemble_uk_atomic_area_support(
            system=SYSTEMS[0], arrays=arrays, column_metadata=metadata
        )


@pytest.mark.parametrize("defect", ["inferred_ni_constituency", "ni_lsoa_not_dz"])
def test_ni_source_convention_refusals(defect):
    arrays, metadata = parts(SYSTEMS[2])
    if defect == "inferred_ni_constituency":
        metadata["constituency_code"]["relation"] = "inferred_modal"
    else:
        arrays["lsoa_code"][0] = "N01000001"
    with pytest.raises(ValueError, match="UK atomic support: NI"):
        assemble_uk_atomic_area_support(
            system=SYSTEMS[2], arrays=arrays, column_metadata=metadata
        )


def test_three_system_assignment_keeps_observed_region_and_native_alias_roles():
    households = pd.DataFrame(
        {
            "household_id": np.arange(1, 5, dtype=np.int64),
            "region": pd.array(
                ["LONDON", "WALES", "SCOTLAND", "NORTHERN_IRELAND"], dtype="string"
            ),
            IDENTITY_COLUMN: pd.array(
                [key(i) for i in (101, 102, 103, 104)], dtype="string"
            ),
        }
    )
    result, spec, supports = complete(households, payloads())
    pd.testing.assert_series_equal(result["region"], households["region"])
    assert result["atomic_area_system"].tolist() == [
        SYSTEMS[0],
        SYSTEMS[0],
        SYSTEMS[1],
        SYSTEMS[2],
    ]
    assert result["atomic_area_basis"].tolist() == ["assigned"] * 4
    assert result.loc[:2, "output_area_code"].equals(result.loc[:2, "atomic_area_code"])
    assert pd.isna(result.loc[3, "output_area_code"])
    assert result.loc[2, "data_zone_code"] == result.loc[2, "lsoa_code"]
    assert result.loc[2, "intermediate_zone_code"] == result.loc[2, "msoa_code"]
    assert result.loc[3, "data_zone_code"] == result.loc[3, "atomic_area_code"]
    assert result.loc[3, "super_data_zone_code"] == result.loc[3, "msoa_code"]
    assert result.loc[3, "district_electoral_area_code"] == result.loc[3, "ward_code"]
    assert pd.isna(result.loc[0, "super_data_zone_code"])
    assert geo.validate_geography(result, spec, supports)["outcome"] == "pass"
    assert {s["source"] for s in spec["systems"]} == set(SOURCES.values())


def test_source_metadata_changes_are_identified_and_declaration_refuses_forged_derived_field():
    payload = payloads()
    original = payload[SYSTEMS[0]]
    arrays, metadata = parts(SYSTEMS[0])
    metadata["population"]["source"] += "-revision-2"
    changed = assemble_uk_atomic_area_support(
        system=SYSTEMS[0], arrays=arrays, column_metadata=metadata
    )
    assert changed != original
    support = geo.decode_atomic_support(original)
    forged_arrays = {k: v.copy() for k, v in support.arrays.items()}
    forged_arrays["itl1_code"][0] = "TLN"
    forged = geo.encode_atomic_support(
        {k: dict(v) if k == "columns" else v for k, v in support.metadata.items()},
        forged_arrays,
    )
    payload[SYSTEMS[0]] = forged
    with pytest.raises(ValueError, match="canonical UK adapter output"):
        uk_atomic_assignment_definition(payload, seed=43)


@pytest.mark.parametrize("defect", ["missing_system", "wrong_system", "bad_seed"])
def test_declaration_refuses_mismatched_inputs(defect):
    payload = payloads()
    options = {"seed": 43}
    if defect == "missing_system":
        del payload[SYSTEMS[2]]
    elif defect == "wrong_system":
        payload[SYSTEMS[2]] = payload[SYSTEMS[1]]
    elif defect == "bad_seed":
        options["seed"] = True
    with pytest.raises(ValueError):
        uk_atomic_assignment_definition(payload, **options)


def test_declaration_cannot_substitute_an_income_or_counter_identity_column():
    with pytest.raises(TypeError, match="identity_column"):
        uk_atomic_assignment_definition(
            payloads(), seed=43, identity_column="household_weight"
        )


def test_distinct_post_clone_keys_assign_then_remain_stable_under_order_subset_and_growth():
    paths = [
        (("spi_support_channel", 0), ("cgt_incidence_clone", 0)),
        (("spi_support_channel", 1), ("cgt_incidence_clone", 0)),
        (("spi_support_channel", 1), ("cgt_incidence_clone", 1)),
        (
            ("spi_support_channel", 1),
            ("cgt_incidence_clone", 1),
            ("cgt_band_donors", 1),
        ),
    ]
    original = pd.DataFrame(
        {
            "household_id": np.asarray([11, 22, 33, 44], dtype=np.int64),
            "region": pd.array(["LONDON"] * 4, dtype="string"),
            IDENTITY_COLUMN: pd.array(
                [key(101, path) for path in paths], dtype="string"
            ),
        }
    )
    assert original[IDENTITY_COLUMN].nunique() == 4
    payload = payloads()
    expected, spec, supports = complete(original, payload)
    changed = original.iloc[[3, 1, 0]].copy()
    changed["household_weight"] = [0.0, 2.0, 8.0]
    extra = pd.DataFrame(
        {
            "household_id": [55],
            "region": pd.array(["LONDON"], dtype="string"),
            IDENTITY_COLUMN: pd.array(
                [key(202, (("geographic_support", 1),))], dtype="string"
            ),
            "household_weight": [1.0],
        },
        index=[8],
    )
    actual, _, _ = complete(pd.concat([changed, extra]), payload)
    for column in expected:
        if column != "region":
            pd.testing.assert_series_equal(
                actual.loc[changed.index, column], expected.loc[changed.index, column]
            )
    assert geo.validate_geography(actual, spec, supports)["outcome"] == "pass"
    assigned_columns = ["atomic_area_code", "oa_code", "constituency_code"]
    pruned = expected.iloc[[3, 1]].copy()
    assert geo.validate_geography(pruned, spec, supports)["outcome"] == "pass"
    pd.testing.assert_frame_equal(
        pruned[assigned_columns], expected.loc[pruned.index, assigned_columns]
    )
    with pytest.raises(ValueError, match="assignment output already exists"):
        geo.assign_atomic(expected, spec, supports)


def test_country_declaration_retains_the_two_stage_sampling_law():
    households = pd.DataFrame(
        {
            "household_id": np.arange(1, 2001, dtype=np.int64),
            "region": pd.array(["LONDON"] * 2000, dtype="string"),
            IDENTITY_COLUMN: pd.array([key(i) for i in range(1, 2001)], dtype="string"),
        }
    )
    result, spec, _ = complete(households, payloads())
    assert all(
        s["stages"]
        == [
            {"level": "constituency_code", "weight": "households"},
            {"level": "area", "weight": "population"},
        ]
        for s in spec["systems"]
    )
    arrays, _ = parts(SYSTEMS[0])
    london = arrays["region_code"] == FRS_REGION_TO_REGION_CODE["LONDON"]
    areas = arrays["oa_code"][london]
    shares = result["atomic_area_code"].value_counts(normalize=True)
    # Constituent household mass 30:10, then within-first-area population 2:6;
    # the third area's population 1000 must not dominate the first-stage draw.
    for area, expected in zip(areas, (0.1875, 0.5625, 0.25), strict=True):
        assert abs(shares.get(area, 0) - expected) < 0.04


def test_exact_source_ids_and_ordered_branches_are_preserved():
    first = key(2**53 + 1, (("spi_support_channel", 1), ("cgt_incidence_clone", 0)))
    second = key(2**53 + 2, (("spi_support_channel", 1), ("cgt_incidence_clone", 0)))
    assert first != second
    assert str(2**53 + 1) in first
    assert first == key(
        np.int64(2**53 + 1),
        (("spi_support_channel", np.int64(1)), ("cgt_incidence_clone", 0)),
    )
    grown = [key(101, (("geographic_support", i),)) for i in range(6)]
    assert grown[:3] == [key(101, (("geographic_support", i),)) for i in range(3)]


@pytest.mark.parametrize(
    "defect",
    [
        "float_id",
        "bool_id",
        "zero_id",
        "oversize_id",
        "no_vintage",
        "mutable_path",
        "counter_branch",
        "income_branch",
        "max_id_branch",
        "out_of_order",
        "duplicate_branch",
        "float_ordinal",
        "negative_ordinal",
        "oversize_ordinal",
    ],
)
def test_identity_refuses_inferred_or_inexact_conventions(defect):
    args = {
        "source": "invented-frs",
        "source_vintage": "invented-2024-25",
        "source_household_id": 101,
        "clone_path": (),
    }
    if defect == "float_id":
        args["source_household_id"] = 101.0
    elif defect == "bool_id":
        args["source_household_id"] = True
    elif defect == "zero_id":
        args["source_household_id"] = 0
    elif defect == "oversize_id":
        args["source_household_id"] = 2**63
    elif defect == "no_vintage":
        args["source_vintage"] = ""
    elif defect == "mutable_path":
        args["clone_path"] = []
    elif defect in {"counter_branch", "income_branch", "max_id_branch"}:
        args["clone_path"] = (
            (
                {
                    "counter_branch": "row_counter",
                    "income_branch": "income",
                    "max_id_branch": "sample_max_id",
                }[defect],
                1,
            ),
        )
    elif defect == "out_of_order":
        args["clone_path"] = (("cgt_incidence_clone", 1), ("spi_support_channel", 1))
    elif defect == "duplicate_branch":
        args["clone_path"] = (("spi_support_channel", 0), ("spi_support_channel", 1))
    else:
        ordinal = {
            "float_ordinal": 1.5,
            "negative_ordinal": -1,
            "oversize_ordinal": 2**32,
        }[defect]
        args["clone_path"] = (("spi_support_channel", ordinal),)
    with pytest.raises(ValueError, match="UK geography identity:"):
        household_draw_key(**args)


def test_duplicate_or_unknown_household_identity_is_refused_by_shared_assignment():
    payload = payloads()
    spec = uk_atomic_assignment_definition(payload, seed=43)
    supports = {s: geo.decode_atomic_support(p) for s, p in payload.items()}
    households = pd.DataFrame(
        {
            "household_id": [11, 22],
            "region": ["LONDON", "LONDON"],
            IDENTITY_COLUMN: [key(101), key(101)],
        }
    )
    with pytest.raises(ValueError, match="duplicate draw identity"):
        geo.assign_atomic(households, spec, supports)
    households.loc[1, IDENTITY_COLUMN] = key(102)
    households.loc[1, "region"] = "UNKNOWN"
    with pytest.raises(ValueError, match="exactly one"):
        geo.assign_atomic(households, spec, supports)
