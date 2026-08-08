"""Scotland/NI OA-ladder source loaders and full-UK concat (#495 increment 3).

The full-UK ladder's four adjudicated sources (microcosm#495 register): the
NRS Census 2022 index zip supplies both the OA22 -> Electoral Ward 2022
lookup (``OA_TO_HIGHER_AREAS.csv``, PiP-centroid, zero blanks on the real
file) and per-OA census occupied household counts (``Postcode_To_OA.csv``
summed by OA — spec-defined census occupied households, cell-key perturbed);
the NISRA table-builder HOUSEHOLD dataset supplies DZ21 household counts;
and the already-pinned DZ2021 GeoJSON's ``DEA2014_cd`` supplies the NI ward
analogue. ``join_uk_oa_ladder_layers`` is country-agnostic, so the ladder
extension is these loaders plus a disjointness-checked concat.
"""

from __future__ import annotations

import io
import json
import zipfile

import pandas as pd
import pytest

import microcosm.build.uk_runtime.geography_sources as geography_sources
from microcosm.build.uk_runtime import (
    concat_uk_ladder_frames,
    join_uk_oa_ladder_layers,
    load_ni_dz_households,
    load_ni_dz_ward_lookup,
    load_scotland_oa_households,
    load_scotland_oa_ward_lookup,
)


def zipped_files_bytes(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return buffer.getvalue()


def _scotland_index_zip() -> bytes:
    return zipped_files_bytes(
        {
            "Census_2022_Index/OA_TO_HIGHER_AREAS.csv": (
                "OA2022,CA2019,EW2022,DZ2011\n"
                "S0001,S12000033,S13002835,S01006755\n"
                "S0002,S12000005,S13002604,S01007000\n"
            ),
            "Census_2022_Index/Postcode_To_OA.csv": (
                "Postcode,OutputArea2022Code,HouseholdCount,PopulationCount\n"
                "AB1 1AA,S0001,10,25\n"
                "AB1 1AB,S0001,5,12\n"
                "G1 1AA,S0002,20,44\n"
            ),
        }
    )


def _ni_geojson_zip() -> bytes:
    features = [
        {
            "type": "Feature",
            "properties": {
                "DZ2021_cd": "N20000001",
                "SDZ2021_cd": "N21000001",
                "LGD2014_cd": "N09000001",
                "DEA2014_cd": "N10000104",
                "DEA2014_nm": "Dunsilly",
            },
        },
        {
            "type": "Feature",
            "properties": {
                "DZ2021_cd": "N20000002",
                "SDZ2021_cd": "N21000001",
                "LGD2014_cd": "N09000001",
                "DEA2014_cd": "N10000105",
                "DEA2014_nm": "Antrim",
            },
        },
    ]
    payload = json.dumps({"type": "FeatureCollection", "features": features})
    return zipped_files_bytes({"DZ2021.geojson": payload})


def test_load_scotland_oa_ward_lookup(monkeypatch) -> None:
    monkeypatch.setattr(geography_sources, "SCOTLAND_OA2022_COUNT", 2)
    monkeypatch.setattr(
        geography_sources,
        "_read_url_bytes",
        lambda url: _scotland_index_zip(),
    )
    lookup = load_scotland_oa_ward_lookup("memory://census-index.zip")
    assert lookup.to_dict("records") == [
        {"oa_code": "S0001", "ward_code": "S13002835"},
        {"oa_code": "S0002", "ward_code": "S13002604"},
    ]


def test_load_scotland_oa_ward_lookup_rejects_blank_wards(monkeypatch) -> None:
    monkeypatch.setattr(geography_sources, "SCOTLAND_OA2022_COUNT", 2)
    monkeypatch.setattr(
        geography_sources,
        "_read_url_bytes",
        lambda url: zipped_files_bytes(
            {"OA_TO_HIGHER_AREAS.csv": ("OA2022,EW2022\nS0001,S13002835\nS0002,\n")}
        ),
    )
    with pytest.raises(ValueError, match="blank"):
        load_scotland_oa_ward_lookup("memory://census-index.zip")


def test_load_scotland_oa_households_sums_postcode_counts(monkeypatch) -> None:
    monkeypatch.setattr(geography_sources, "SCOTLAND_OA2022_COUNT", 2)
    monkeypatch.setattr(
        geography_sources,
        "_read_url_bytes",
        lambda url: _scotland_index_zip(),
    )
    households = load_scotland_oa_households("memory://census-index.zip")
    assert households.to_dict("records") == [
        {"oa_code": "S0001", "households": 15},
        {"oa_code": "S0002", "households": 20},
    ]


def test_load_scotland_oa_households_rejects_bad_counts(monkeypatch) -> None:
    monkeypatch.setattr(geography_sources, "SCOTLAND_OA2022_COUNT", 1)
    monkeypatch.setattr(
        geography_sources,
        "_read_url_bytes",
        lambda url: zipped_files_bytes(
            {
                "Postcode_To_OA.csv": (
                    "Postcode,OutputArea2022Code,HouseholdCount,PopulationCount\n"
                    "AB1 1AA,S0001,not_a_number,25\n"
                )
            }
        ),
    )
    with pytest.raises(ValueError):
        load_scotland_oa_households("memory://census-index.zip")


def test_load_ni_dz_ward_lookup(monkeypatch) -> None:
    monkeypatch.setattr(geography_sources, "NI_DZ2021_COUNT", 2)
    monkeypatch.setattr(
        geography_sources,
        "_read_url_bytes",
        lambda url: _ni_geojson_zip(),
    )
    lookup = load_ni_dz_ward_lookup("memory://ni-dz.zip")
    assert lookup.to_dict("records") == [
        {"oa_code": "N20000001", "ward_code": "N10000104"},
        {"oa_code": "N20000002", "ward_code": "N10000105"},
    ]


def test_load_ni_dz_ward_lookup_rejects_missing_dea(monkeypatch) -> None:
    monkeypatch.setattr(geography_sources, "NI_DZ2021_COUNT", 1)
    payload = json.dumps(
        {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {"DZ2021_cd": "N20000001"}}],
        }
    )
    monkeypatch.setattr(
        geography_sources,
        "_read_url_bytes",
        lambda url: zipped_files_bytes({"DZ2021.geojson": payload}),
    )
    with pytest.raises(ValueError, match="blank|missing"):
        load_ni_dz_ward_lookup("memory://ni-dz.zip")


def test_load_ni_dz_households(monkeypatch) -> None:
    monkeypatch.setattr(geography_sources, "NI_DZ2021_COUNT", 2)
    monkeypatch.setattr(
        geography_sources,
        "_read_url_bytes",
        lambda url: (
            b"Census 2021 Data Zone Code,Census 2021 Data Zone Label,Count\n"
            b"N20000001,Dunsilly_A1,249\n"
            b"N20000002,Dunsilly_B1,127\n"
        ),
    )
    households = load_ni_dz_households("memory://ni-households.csv")
    assert households.to_dict("records") == [
        {"oa_code": "N20000001", "households": 249},
        {"oa_code": "N20000002", "households": 127},
    ]


def _ladder_frame(prefix: str, la: str, ward: str, itl: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "oa_code": f"{prefix}0001",
                "population": 100.0,
                "households": 40.0,
                "constituency_code": f"{prefix}14000001",
                "region_code": f"{prefix}99999999",
                "lsoa_code": f"{prefix}0101",
                "msoa_code": f"{prefix}0201",
                "local_authority_code": la,
                "ward_code": ward,
                "itl3_code": itl,
            }
        ]
    )


def test_concat_uk_ladder_frames_requires_disjoint_codes() -> None:
    scotland = _ladder_frame("S", "S12000033", "S13002835", "TLM50")
    ni = _ladder_frame("N", "N09000001", "N10000104", "TLN0A")
    combined = concat_uk_ladder_frames(scotland, ni)
    assert combined["oa_code"].tolist() == ["S0001", "N0001"]

    with pytest.raises(ValueError, match="disjoint|duplicate"):
        concat_uk_ladder_frames(scotland, scotland)
    with pytest.raises(ValueError, match="column"):
        concat_uk_ladder_frames(scotland, ni.drop(columns=["ward_code"]))
    with pytest.raises(ValueError, match="empty"):
        concat_uk_ladder_frames(scotland, ni.iloc[0:0])


def test_join_uk_oa_ladder_layers_is_country_agnostic() -> None:
    base = pd.DataFrame(
        [
            {
                "oa_code": "N20000001",
                "population": 249.0,
                "constituency_code": "N05000001",
                "region_code": "N99999999",
                "lsoa_code": "N20000001",
                "msoa_code": "N21000001",
                "la_code": "N09000001",
            }
        ]
    )
    joined = join_uk_oa_ladder_layers(
        base,
        oa_households=pd.DataFrame([{"oa_code": "N20000001", "households": 100}]),
        oa_ward=pd.DataFrame([{"oa_code": "N20000001", "ward_code": "N10000104"}]),
        lad_itl=pd.DataFrame(
            [{"local_authority_code": "N09000001", "itl3_code": "TLN0A"}]
        ),
    )
    assert joined["ward_code"].tolist() == ["N10000104"]
    assert joined["itl3_code"].tolist() == ["TLN0A"]


def test_itl_pattern_accepts_ni_alphanumeric_suffixes() -> None:
    import numpy as np

    from microcosm.build.uk_runtime.geography_ladder import _itl_code_array

    accepted = _itl_code_array(
        np.array(["TLC11", "TLM50", "TLN0A", "TLN0G"]),
        label="itl3",
    )
    assert accepted.tolist() == ["TLC11", "TLM50", "TLN0A", "TLN0G"]
    with pytest.raises(ValueError, match="ITL"):
        _itl_code_array(np.array(["tln0a"]), label="itl3")


def test_ward_pattern_accepts_split_ward_part_codes() -> None:
    import numpy as np

    from microcosm.build.uk_runtime.geography_ladder import (
        _WARD_CODE_PATTERN,
        _gss_code_array,
    )

    accepted = _gss_code_array(
        np.array(["E05014284", "E05R14284", "E05S14284", "S13002835", "N10000104"]),
        label="ward_code",
        pattern=_WARD_CODE_PATTERN,
    )
    assert accepted.tolist() == [
        "E05014284",
        "E05R14284",
        "E05S14284",
        "S13002835",
        "N10000104",
    ]
    # Every non-ward layer stays strict: part codes are refused by default.
    with pytest.raises(ValueError, match="GSS"):
        _gss_code_array(np.array(["E05R14284"]), label="constituency_code")


def _uk_gate_household() -> pd.DataFrame:
    rows = [
        ("E00000001", "LONDON", "E12000007", "E05014284", "E14000001", "TLI31"),
        ("E00000002", "LONDON", "E12000007", "E05R14284", "E14000001", "TLI31"),
        ("W00000001", "WALES", "W99999999", "W05001517", "W07000041", "TLL11"),
        ("S00000001", "SCOTLAND", "S99999999", "S13002835", "S14000001", "TLM50"),
        (
            "N20000001",
            "NORTHERN_IRELAND",
            "N99999999",
            "N10000104",
            "N05000001",
            "TLN0A",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "oa_code": oa,
                "lsoa_code": oa,
                "msoa_code": oa,
                "local_authority_code": "E06000001"
                if oa.startswith("E")
                else oa[:1] + "09000001",
                "ward_code": ward,
                "constituency_code": constituency,
                "region_code": region_code,
                "itl3_code": itl3,
                "itl2_code": itl3[:4],
                "itl1_code": itl3[:3],
                "region": region,
            }
            for oa, region, region_code, ward, constituency, itl3 in rows
        ]
    )


def test_gate_accepts_full_uk_assignment_with_part_codes() -> None:
    import numpy as np

    from microcosm.build.uk_runtime import uk_geography_ladder_gate

    household = _uk_gate_household()
    weights = np.array([3.0, 3.0, 10.0, 10.0, 10.0])
    result = uk_geography_ladder_gate(
        household,
        weights,
        london_share_bounds=(0.08, 0.20),
    )
    assert result.passed, result.failures


def test_gate_rejects_region_code_inconsistency() -> None:
    import numpy as np

    from microcosm.build.uk_runtime import uk_geography_ladder_gate

    household = _uk_gate_household()
    # A Scottish household mislabelled with a valid English region code must
    # fail the rowwise consistency fence, not pass on structural validity.
    household.loc[household["region"] == "SCOTLAND", "region_code"] = "E12000007"
    weights = np.array([3.0, 3.0, 10.0, 10.0, 10.0])
    result = uk_geography_ladder_gate(household, weights)
    assert not result.passed
    assert any(
        "disagree with the household's declared region" in f for f in result.failures
    )


def test_tightened_patterns_reject_invented_families() -> None:
    import numpy as np

    from microcosm.build.uk_runtime.geography_ladder import (
        _ITL_CODE_PATTERN,
        _WARD_CODE_PATTERN,
        _itl_code_array,
    )

    assert _ITL_CODE_PATTERN.match("TLZZ") is None
    assert _ITL_CODE_PATTERN.match("TLN0A") is not None
    with pytest.raises(ValueError, match="ITL"):
        _itl_code_array(np.array(["TLZZ"]), label="itl3")
    assert _WARD_CODE_PATTERN.match("E05A14284") is None
    assert _WARD_CODE_PATTERN.match("S13Z00000") is None
    assert _WARD_CODE_PATTERN.match("E05R14284") is not None
    assert _WARD_CODE_PATTERN.match("E05S14284") is not None


def test_concat_refuses_cross_country_code_mixing() -> None:
    scotland = _ladder_frame("S", "S12000033", "S13002835", "TLM50")
    mixed = scotland.copy()
    mixed["constituency_code"] = "E14000001"
    ni = _ladder_frame("N", "N09000001", "N10000104", "TLN0A")
    with pytest.raises(ValueError, match="mixes countries"):
        concat_uk_ladder_frames(mixed, ni)


def test_scotland_households_rejects_missing_counts(monkeypatch) -> None:
    monkeypatch.setattr(geography_sources, "SCOTLAND_OA2022_COUNT", 1)
    monkeypatch.setattr(
        geography_sources,
        "_read_url_bytes",
        lambda url: zipped_files_bytes(
            {
                "Postcode_To_OA.csv": (
                    "Postcode,OutputArea2022Code,HouseholdCount,PopulationCount\n"
                    "AB1 1AA,S0001,,25\n"
                )
            }
        ),
    )
    with pytest.raises(ValueError, match="missing HouseholdCount"):
        load_scotland_oa_households("memory://census-index.zip")


def test_ni_postcode_inference_rejects_duplicate_keys() -> None:
    from microcosm.build.uk_runtime import infer_ni_dz_constituencies_from_postcodes

    postcode_oa = pd.DataFrame(
        {
            "pcds": ["BT1 1AA", "BT1 1AA", "BT1 1AB"],
            "doterm": ["", "", ""],
            "oa21cd": ["N20000001", "N20000001", "N20000001"],
        }
    )
    postcode_pcon = pd.DataFrame(
        {
            "pcd": ["BT1 1AA", "BT1 1AB"],
            "pconcd": ["N05000001", "N05000001"],
        }
    )
    with pytest.raises(ValueError, match="duplicate normalized postcode"):
        infer_ni_dz_constituencies_from_postcodes(postcode_oa, postcode_pcon)


def test_full_uk_assemble_load_round_trip(tmp_path) -> None:
    import numpy as np

    from microcosm.build.uk_runtime import assemble_uk_oa_ladder, load_uk_oa_ladder

    frames = concat_uk_ladder_frames(
        _ladder_frame("E", "E06000001", "E05014284", "TLI31"),
        _ladder_frame("S", "S12000033", "S13002835", "TLM50"),
        _ladder_frame("N", "N09000001", "N10000104", "TLN0A"),
    )
    frames["oa_code"] = ["E00000001", "S00000001", "N20000001"]
    frames["lsoa_code"] = frames["oa_code"]
    frames["msoa_code"] = frames["oa_code"]
    frames["constituency_code"] = ["E14000001", "S14000001", "N05000001"]

    def layer(vintage: str) -> dict[str, object]:
        return {
            "vintage": vintage,
            "source": "synthetic test source",
            "countries": {
                "england_and_wales": {"vintage": vintage, "source": "synthetic"},
                "scotland": {"vintage": vintage, "source": "synthetic"},
                "northern_ireland": {"vintage": vintage, "source": "synthetic"},
            },
        }

    metadata = {
        "schema_version": 1,
        "kind": "uk_oa_ladder",
        "coverage": "uk",
        "oa_vintage": "ew:2021;scotland:2022;ni:2021",
        "constituency_sampling_basis": "synthetic household counts",
        "oa_sampling_basis": "synthetic population",
        "layers": {
            "constituency": layer("2024_pcon"),
            "lsoa": layer("composite"),
            "msoa": layer("composite"),
            "local_authority": layer("composite"),
            "ward": layer("composite"),
            "itl": {"vintage": "2021_itl", "source": "synthetic"},
            "region": layer("composite"),
        },
    }
    payload = assemble_uk_oa_ladder(frames, metadata)
    path = tmp_path / "uk_ladder.npz"
    np.savez_compressed(path, **payload)
    ladder = load_uk_oa_ladder(path)
    assert len(ladder) == 3
    assert ladder.layer_vintages["constituency"] == "2024_pcon"

    # A countries submap missing its vintage must refuse to load.
    bad_metadata = json.loads(json.dumps(metadata))
    bad_metadata["layers"]["ward"]["countries"]["scotland"] = {"vintage": ""}
    bad_payload = assemble_uk_oa_ladder(frames, bad_metadata)
    bad_path = tmp_path / "bad_ladder.npz"
    np.savez_compressed(bad_path, **bad_payload)
    with pytest.raises(ValueError, match="scotland"):
        load_uk_oa_ladder(bad_path)
