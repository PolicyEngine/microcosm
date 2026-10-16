"""The committed area-to-BRMA layer and the lookup the BRMA node runs (microcosm#1262)."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from microcosm.build.atomic_geography import decode_atomic_support
from microcosm.build.uk_runtime import brma_area_layer as bal
from microcosm.build.uk_runtime.atomic_area_support import SOURCES, SYSTEMS
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")
_UK = _TEST_PATHS.package / "src/microcosm/build/uk"
_REPOSITORY = _TEST_PATHS.repository
EW, SCOT, NI = SYSTEMS


def _payload() -> bytes:
    return (_UK / "brma_area_layer.json").read_bytes()


def _data() -> dict:
    return json.loads(_payload())


def _provenance() -> dict:
    return json.loads((_UK / "brma_area_layer.provenance.json").read_text("utf-8"))


def _count_table() -> dict:
    return json.loads((_UK / "brma_private_rented_households.json").read_text("utf-8"))


def _toy_layer_payload() -> dict:
    """A two-grain layer small enough to reason about."""

    ew = {
        "LONDON": {"INNER_EAST_LONDON": ["E01000002"], "CENTRAL_LONDON": ["E01000001"]},
        "WALES": {"CARDIFF": ["W01001001"]},
    }
    scotland = {
        "SCOTLAND": {"LOTHIAN": ["S00100001", "S00100002"], "FIFE": ["S00100003"]}
    }
    return {
        "schema_version": 1,
        "kind": bal.KIND,
        "method": bal.METHOD_VERSION,
        "grains": {
            "ew_lsoa21": {
                "area_system": EW,
                "key": "lsoa_code",
                "areas": 3,
                "roster_sha256": bal.roster_sha256(
                    ["E01000001", "E01000002", "W01001001"]
                ),
            },
            "scotland_oa22": {
                "area_system": SCOT,
                "key": "oa_code",
                "areas": 3,
                "roster_sha256": bal.roster_sha256(
                    ["S00100001", "S00100002", "S00100003"]
                ),
            },
        },
        "scotland_brma_codes": {"FIFE": "S33000007", "LOTHIAN": "S33000011"},
        "brmas": {"ew_lsoa21": ew, "scotland_oa22": scotland},
    }


def _toy_layer() -> bal.BRMAAreaLayer:
    return bal.load_brma_area_layer(json.dumps(_toy_layer_payload()).encode())


# ----------------------------------------------------------------------------
# The committed resource


def test_committed_layer_loads_and_its_provenance_hashes_it() -> None:
    layer = bal.load_brma_area_layer(_payload())
    assert len(layer.brma_by_area["ew_lsoa21"]) == 35_672
    assert len(layer.brma_by_area["scotland_oa22"]) == 46_363
    provenance = _provenance()
    assert provenance["resource_sha256"] == hashlib.sha256(_payload()).hexdigest()
    assert provenance["receipts"]["gb_brmas"] == 192
    assert _data()["licence"]["licence"] == "Open Government Licence v3.0"


def test_layer_rosters_are_the_committed_area_rosters() -> None:
    layer = bal.load_brma_area_layer(_payload())
    band_mix = json.loads(
        (_UK / "voa_council_tax_band_mix_lsoa.json").read_text("utf-8")
    )
    assert set(layer.brma_by_area["ew_lsoa21"]) == set(band_mix["lsoa"])
    supports = json.loads((_UK / "uk_atomic_area_supports.provenance.json").read_text())
    assert (
        len(layer.brma_by_area["scotland_oa22"]) == supports["supports"][SCOT]["areas"]
    )
    assert (
        _provenance()["supports"][SCOT]["sha256"]
        == supports["supports"][SCOT]["sha256"]
    )
    assert _provenance()["supports"][EW]["sha256"] == supports["supports"][EW]["sha256"]


def test_layer_names_are_the_engine_names_outside_northern_ireland() -> None:
    layer = bal.load_brma_area_layer(_payload())
    table = _count_table()["cells"]
    northern_ireland = {
        brma for cells in table["NORTHERN_IRELAND"].values() for brma in cells
    }
    every = {
        brma for region in table.values() for cells in region.values() for brma in cells
    }
    layer_names = set(layer.brma_by_area["ew_lsoa21"].values()) | set(
        layer.brma_by_area["scotland_oa22"].values()
    )
    assert layer_names == every - northern_ireland
    assert len(layer_names) == 192


def test_each_region_holds_every_brma_the_count_table_rents_in() -> None:
    """One method: the layer places areas where uk-data#516 counted renters."""

    layer = bal.load_brma_area_layer(_payload())
    for region, categories in _count_table()["cells"].items():
        if region == "NORTHERN_IRELAND":
            continue
        grain = "scotland_oa22" if region == "SCOTLAND" else "ew_lsoa21"
        in_layer = {
            brma
            for code, brma in layer.brma_by_area[grain].items()
            if layer.region_by_area[grain][code] == region
        }
        renting = {
            brma
            for cells in categories.values()
            for brma, count in cells.items()
            if float(count) > 0
        }
        assert renting == in_layer, region


def test_scotland_brmas_carry_their_price_index_codes() -> None:
    layer = bal.load_brma_area_layer(_payload())
    codes = layer.scotland_codes
    assert sorted(codes.values()) == [f"S330000{n:02d}" for n in range(1, 19)]
    assert codes["LOTHIAN"] == "S33000011" and codes["AYRSHIRES"] == "S33000003"
    assert codes["RENFREWSHIRE_AND_INVERCLYDE"] == "S33000014"


def test_scottish_pairs_are_microcosm_1092_pairs() -> None:
    """The 63 BRMA x council pairs microcosm#1092 measured, area for area."""

    pairs = _provenance()["agreement"]["scotland_brma_council_pairs"]
    assert len(pairs) == 63
    assert len({pair["local_authority"] for pair in pairs}) == 32
    assert sum(pair["output_areas"] for pair in pairs) == 46_363


# ----------------------------------------------------------------------------
# The loader


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(kind="other"), "schema-1"),
        (lambda d: d.update(method="uk-brma-area-layer-v0"), "method"),
        (lambda d: d["grains"].pop("scotland_oa22"), "grains"),
        (
            lambda d: d["brmas"]["ew_lsoa21"]["LONDON"]["CENTRAL_LONDON"].append(
                "E01000002"
            ),
            "two BRMAs",
        ),
        (lambda d: d["grains"]["ew_lsoa21"].update(areas=4), "declares 4 areas"),
        (lambda d: d["grains"]["ew_lsoa21"].update(roster_sha256="0" * 64), "digest"),
        (
            lambda d: d["brmas"]["scotland_oa22"]["SCOTLAND"]["FIFE"].append(
                "E01000009"
            ),
            "not a scotland_oa22",
        ),
        (lambda d: d["scotland_brma_codes"].pop("FIFE"), "exactly one ONS code"),
        (lambda d: d["scotland_brma_codes"].update(FIFE="S33000011"), "distinct S33"),
        (
            lambda d: d["grains"]["ew_lsoa21"].update(key="oa_code"),
            "mislabelled",
        ),
    ],
)
def test_loader_refuses_malformed_layers(mutate, message) -> None:
    payload = copy.deepcopy(_toy_layer_payload())
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        bal.load_brma_area_layer(json.dumps(payload).encode())


# ----------------------------------------------------------------------------
# Names


def test_name_rules_reproduce_the_engine_spellings() -> None:
    assert bal.english_welsh_brma_key("Inner East London") == "INNER_EAST_LONDON"
    assert bal.english_welsh_brma_key("Brighton and Hove") == "BRIGHTON_AND_HOVE"
    assert (
        bal.scottish_brma_key("Renfrewshire / Inverclyde")
        == "RENFREWSHIRE_AND_INVERCLYDE"
    )
    assert bal.scottish_brma_key("Ayrshire") == "AYRSHIRES"
    assert bal.list_brma_key("Highlands and Islands") == "HIGHLAND_AND_ISLANDS"
    assert bal.list_brma_key("Renfrewshire/Inverclyde") == "RENFREWSHIRE_AND_INVERCLYDE"
    assert bal.list_brma_key("Central London") == "CENTRAL_LONDON"


def test_roster_digest_ignores_order() -> None:
    assert bal.roster_sha256(["b", "a"]) == bal.roster_sha256(["a", "b"])
    assert bal.roster_sha256(["a"]) != bal.roster_sha256(["a", "b"])


# ----------------------------------------------------------------------------
# The lookup


def _rows():
    return {
        "area_system": np.array([EW, EW, SCOT, NI, EW, SCOT]),
        "lsoa_code": np.array(
            [
                "E01000001",
                "W01001001",
                "S01000001",
                "N00000001",
                "E01000002",
                "S01000002",
            ],
            dtype=object,
        ),
        "oa_code": np.array(
            [
                "E00000001",
                "W00000001",
                "S00100001",
                "N00000001",
                "E00000002",
                "S00100003",
            ],
            dtype=object,
        ),
        "region": np.array(
            ["LONDON", "WALES", "SCOTLAND", "NORTHERN_IRELAND", "LONDON", "SCOTLAND"]
        ),
        "brma": np.array(
            ["MAIDSTONE", "MAIDSTONE", "MAIDSTONE", "BELFAST", "MAIDSTONE", "LOTHIAN"],
            dtype=object,
        ),
    }


def test_area_brmas_reads_the_layer_and_keeps_northern_ireland() -> None:
    out, counts = bal.area_brmas(layer=_toy_layer(), **_rows())
    assert out.tolist() == [
        "CENTRAL_LONDON",
        "CARDIFF",
        "LOTHIAN",
        "BELFAST",
        "INNER_EAST_LONDON",
        "FIFE",
    ]
    assert counts == {
        "northern_ireland": 1,
        "ew_lsoa21": 3,
        "scotland_oa22": 2,
        "changed": 5,
    }


def test_area_brmas_does_not_depend_on_row_order() -> None:
    rows = _rows()
    order = np.array([5, 3, 1, 0, 4, 2])
    permuted = {key: value[order] for key, value in rows.items()}
    out, _ = bal.area_brmas(layer=_toy_layer(), **rows)
    out_permuted, _ = bal.area_brmas(layer=_toy_layer(), **permuted)
    assert out_permuted.tolist() == out[order].tolist()


@pytest.mark.parametrize(
    ("column", "index", "value", "message"),
    [
        ("lsoa_code", 0, "E01999999", "no BRMA in the layer"),
        ("oa_code", 2, "S00199999", "no BRMA in the layer"),
        ("area_system", 0, "uk_ew_output_area_2011", "No BRMA rule"),
        ("region", 0, "SOUTH_EAST", "another FRS region"),
    ],
)
def test_area_brmas_fails_closed(column, index, value, message) -> None:
    rows = _rows()
    rows[column] = rows[column].astype(object)
    rows[column][index] = value
    with pytest.raises(ValueError, match=message):
        bal.area_brmas(layer=_toy_layer(), **rows)


# ----------------------------------------------------------------------------
# The mounted supports


def _mounted(system: str) -> bytes:
    path = _REPOSITORY / "build/uk/supports" / f"{SOURCES[system]}.npz"
    if not path.is_file():
        pytest.skip(f"{path.name} is not mounted under build/uk/supports")
    return path.read_bytes()


@pytest.mark.parametrize("system", [EW, SCOT])
def test_every_mounted_area_has_a_brma_in_its_own_region(system) -> None:
    support = decode_atomic_support(_mounted(system))
    layer = bal.load_brma_area_layer(_payload())
    area_system = np.full(len(support.arrays["area"]), system)
    out, counts = bal.area_brmas(
        layer=layer,
        area_system=area_system,
        lsoa_code=support.arrays["lsoa_code"],
        oa_code=support.arrays["area"],
        region=support.arrays["frs_region"],
        brma=np.full(len(area_system), "MAIDSTONE", dtype=object),
    )
    grain = "ew_lsoa21" if system == EW else "scotland_oa22"
    assert counts[grain] == len(area_system)
    assert set(out) <= set(layer.brma_by_area[grain].values())
