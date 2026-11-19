"""The UK conditioned atomic-area supports (microcosm#931).

Each extends the registered support of its system with the area's IMD
decile and, where council tax is banded, its households by band; only the
geography-first move binds them. The committed provenance, ``sources.yaml``
and ``vintages.yaml`` must agree; when the support bytes are mounted under
``build/uk/supports``, the last tests check them against all three.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.atomic_geography import (
    decode_atomic_support,
    encode_atomic_support,
)
from microcosm.build.atomic_support_conditioning import (
    apportion_weights,
    zero_mass_cells,
)
from microcosm.build.spec_engine import load_bundle
from microcosm.build.spec_engine.compiler_ir import compile_spec
from microcosm.build.uk_runtime.atomic_area_support import (
    BAND_WEIGHT_COLUMNS,
    BAND_WEIGHT_SCALE,
    CONDITIONED_PROVENANCE_RESOURCE,
    CONDITIONED_SOURCES,
    IMD_DECILE_COLUMN,
    IMD_DECILES,
    SOURCES,
    SYSTEMS,
    assemble_uk_conditioned_area_support,
    uk_conditioned_support_base,
)
from microcosm.build.uk_runtime.chronicle_feed import load_uk_chronicle_feed
from microcosm.build.uk_runtime.country_adapter import uk_atomic_support_register
from test_support.microcosm_build.uk_atomic_support_fixtures import toy_support_payloads
from test_support.paths import paths_for

EW, SCOT, NI = SYSTEMS
_TEST_PATHS = paths_for("microcosm-build")
_UK = _TEST_PATHS.package / "src/microcosm/build/uk"
_REPOSITORY = _TEST_PATHS.repository
_SHA = re.compile(r"[0-9a-f]{64}")
#: The vintage authorities each conditioned support adds to its base's.
CONDITIONING_VINTAGES = {
    EW: {"uk_iod_wimd_2019": "2019_iod_wimd", "uk_ctsop_2025": "2025_ctsop"},
    SCOT: {
        "uk_simd_2020v2": "2020v2_simd",
        "uk_nrs_dz_dwellings_2024": "2024_nrs_dz_dwellings",
    },
    NI: {"uk_nimdm_2017": "2017_nimdm_soa"},
}
#: The authority behind each system's decile column.
DECILE_VINTAGE = {EW: "uk_iod_wimd_2019", SCOT: "uk_simd_2020v2", NI: "uk_nimdm_2017"}


def _metadata(system: str) -> dict:
    metadata = {
        IMD_DECILE_COLUMN: {
            "kind": "code",
            "source": "invented@sha256:" + "0" * 64,
            "vintage": "invented_2019",
            "relation": "inferred_modal",
        }
    }
    for column in BAND_WEIGHT_COLUMNS[system]:
        metadata[column] = {"kind": "weight", "source": "invented", "basis": "invented"}
    return metadata


def _inputs(system: str, payload: bytes) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Invented deciles, and each area's households spread evenly over its bands."""
    support = decode_atomic_support(payload)
    areas = np.asarray(support.arrays["area"])
    deciles = np.asarray([IMD_DECILES[i % 10] for i in range(len(areas))], dtype="U")
    bands = BAND_WEIGHT_COLUMNS[system]
    if not bands:
        return deciles, {}
    numerators = {column: np.ones(len(areas), dtype=np.int64) for column in bands}
    if system == EW:
        english = np.char.startswith(areas, "E")
        numerators["households_band_i"] = np.where(english, 0, 1).astype(np.int64)
    weights, _ = apportion_weights(
        np.asarray(support.arrays["households"], dtype=np.int64),
        numerators,
        sum(numerators.values()),
        scale=BAND_WEIGHT_SCALE,
    )
    return deciles, weights


@pytest.fixture(scope="module")
def toys() -> dict[str, bytes]:
    return toy_support_payloads()


@pytest.mark.parametrize("system", SYSTEMS)
def test_conditioned_support_extends_its_canonical_base(toys, system) -> None:
    deciles, weights = _inputs(system, toys[system])
    payload = assemble_uk_conditioned_area_support(
        base=toys[system],
        imd_decile=deciles,
        band_weights=weights,
        column_metadata=_metadata(system),
    )
    assert uk_conditioned_support_base(payload) == toys[system]
    support, base = decode_atomic_support(payload), decode_atomic_support(toys[system])
    assert set(support.arrays) - set(base.arrays) == {
        IMD_DECILE_COLUMN,
        *BAND_WEIGHT_COLUMNS[system],
    }
    for column, array in base.arrays.items():
        assert np.array_equal(support.arrays[column], array)
    assert support.arrays[IMD_DECILE_COLUMN].tolist() == deciles.tolist()
    # Northern Ireland has domestic rates: its support carries no band weights.
    assert bool(BAND_WEIGHT_COLUMNS[system]) is (system != NI)


def _defect(system: str, defect: str, toys: dict[str, bytes]):
    deciles, weights = _inputs(system, toys[system])
    metadata = _metadata(system)
    base = toys[system]
    if defect == "conditioned_base":
        base = assemble_uk_conditioned_area_support(
            base=base,
            imd_decile=deciles,
            band_weights=weights,
            column_metadata=metadata,
        )
    elif defect == "decile_code":
        deciles = deciles.copy()
        deciles[0] = "11"
    elif defect == "missing_band":
        weights.pop("households_band_i")
    elif defect == "english_band_i":
        weights["households_band_i"] = weights["households_band_i"].copy()
        english = int(
            np.flatnonzero(
                np.char.startswith(decode_atomic_support(base).arrays["area"], "E")
            )[0]
        )
        weights["households_band_i"][english] = 1
        weights["households_band_b"][english] -= 1
    elif defect == "unsplit_households":
        weights["households_band_b"] = weights["households_band_b"] + 10
    elif defect == "decile_metadata":
        metadata[IMD_DECILE_COLUMN] = {**metadata[IMD_DECILE_COLUMN], "kind": "weight"}
    elif defect == "float_weights":
        weights["households_band_a"] = weights["households_band_a"].astype(np.float64)
    return base, deciles, weights, metadata


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("conditioned_base", "already conditioned"),
        ("decile_code", "decile code"),
        ("missing_band", "exact band weights"),
        ("english_band_i", "band I exists only in Wales"),
        ("unsplit_households", "split the area's households"),
        ("decile_metadata", "conditioning column description"),
        ("float_weights", "nonnegative integer band weights"),
    ],
)
def test_conditioned_assembler_refuses_defects(toys, defect, message) -> None:
    base, deciles, weights, metadata = _defect(EW, defect, toys)
    with pytest.raises(ValueError, match=message):
        assemble_uk_conditioned_area_support(
            base=base,
            imd_decile=deciles,
            band_weights=weights,
            column_metadata=metadata,
        )


def test_conditioned_base_refuses_a_forged_or_foreign_column(toys) -> None:
    deciles, weights = _inputs(SCOT, toys[SCOT])
    payload = assemble_uk_conditioned_area_support(
        base=toys[SCOT],
        imd_decile=deciles,
        band_weights=weights,
        column_metadata=_metadata(SCOT),
    )
    support = decode_atomic_support(payload)
    columns = {name: dict(c) for name, c in support.metadata["columns"].items()}
    arrays = {name: np.asarray(a) for name, a in support.arrays.items()}
    columns["households_band_i"] = dict(columns["households_band_a"])
    arrays["households_band_i"] = arrays["households_band_a"].copy()
    forged = encode_atomic_support(
        {**dict(support.metadata), "columns": columns}, arrays
    )
    with pytest.raises(ValueError, match="conditioned support columns differ"):
        uk_conditioned_support_base(forged)
    with pytest.raises(ValueError, match="conditioned support columns differ"):
        uk_conditioned_support_base(toys[SCOT])


@pytest.fixture(scope="module")
def provenance() -> dict:
    return json.loads(
        (_UK / CONDITIONED_PROVENANCE_RESOURCE).read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def bundle():
    return compile_spec(load_bundle("uk"))


@pytest.fixture(scope="module")
def tool():
    path = _REPOSITORY / "tools/build_uk_conditioned_area_supports.py"
    spec = importlib.util.spec_from_file_location("uk_conditioned_tool_pins", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_provenance_pins_chronicle_registrations_and_resources(
    provenance, tool
) -> None:
    assert provenance["kind"] == "uk_atomic_area_conditioned_supports_provenance"
    assert provenance["publisher_files"] == tool.SOURCES
    assert provenance["chronicle"] == tool.CHRONICLE
    assert (
        provenance["issues"]["chronicle_registration"] == "PolicyEngine/chronicle#412"
    )
    assert provenance["band_weight_scale"] == BAND_WEIGHT_SCALE
    assert set(provenance["ledger_feed"]) == {
        "source_commit",
        "facts_sha256",
        "manifest_sha256",
        "fact_row_count",
    }
    facts = provenance["feed_facts"]["nrs_dwellings_by_band"]
    assert facts["record_set_id"] == tool.NRS_RECORD_SET
    # Each zone's total and bands A to H.
    assert facts["data_zones"] == tool.NRS_DATA_ZONES
    assert facts["rows"] == 9 * tool.NRS_DATA_ZONES
    assert _SHA.fullmatch(facts["fingerprint_sha256"])
    for name, resource in provenance["resources"].items():
        # A regenerated band mix must regenerate the conditioned supports.
        assert (
            resource["sha256"] == hashlib.sha256((_UK / name).read_bytes()).hexdigest()
        )
        assert _SHA.fullmatch(resource["publisher_sha256"])


@pytest.mark.parametrize("system", SYSTEMS)
def test_column_metadata_names_every_input_and_its_vintage(provenance, system) -> None:
    entry = provenance["supports"][system]
    files = provenance["publisher_files"]
    decile = entry["column_metadata"][IMD_DECILE_COLUMN]
    assert decile["kind"] == "code" and decile["relation"] == "inferred_modal"
    tags = decile["source"].split("#")[0].split("+")
    sha_by_package = {
        row["chronicle_package_id"]: row["sha256"] for row in files.values()
    }
    assert tags[0].startswith("ons-nspl-may-2026@sha256:")
    for tag in tags:
        package, _, digest = tag.partition("@sha256:")
        assert sha_by_package[package] == digest
    assert decile["vintage"] in CONDITIONING_VINTAGES[system].values()
    assert set(entry["column_metadata"]) == {
        IMD_DECILE_COLUMN,
        *BAND_WEIGHT_COLUMNS[system],
    }
    resources = provenance["resources"]
    for column in BAND_WEIGHT_COLUMNS[system]:
        item = entry["column_metadata"][column]
        assert item["kind"] == "weight" and item["basis"].endswith("_x1e6")
        if system == SCOT:
            # Scotland reads NRS's band facts from the pinned feed.
            facts = provenance["feed_facts"]["nrs_dwellings_by_band"]
            assert item["source"] == (
                f"chronicle:{facts['record_set_id']}"
                f"@sha256:{facts['fingerprint_sha256']}"
            )
            continue
        publisher, _, resource = item["source"].partition("#")
        package, _, publisher_digest = publisher.partition("@sha256:")
        name, _, digest = resource.partition("@sha256:")
        assert resources[name]["sha256"] == digest
        assert (
            resources[name]["publisher_package"],
            resources[name]["publisher_sha256"],
        ) == (
            package,
            publisher_digest,
        )


def test_the_pinned_feed_carries_the_band_facts_the_supports_were_built_from(
    provenance, tool
) -> None:
    """The re-pin procedure's check of Scotland's band facts.

    A re-pin whose feed carries NRS's December 2024 band facts unchanged
    leaves the supports as built, and their provenance keeps naming the feed
    they were built from; one that moves them needs the supports rebuilt
    (docs/uk-chronicle-feed-repin.md).
    """

    configured = os.environ.get("CHRONICLE_UK_FACTS")
    feed = (
        Path(configured)
        if configured
        else _REPOSITORY / ".codex-work/consumer_facts_uk.jsonl"
    )
    if feed.is_dir():
        feed = feed / "consumer_facts.jsonl"
    if not feed.is_file():
        pytest.skip("pinned UK Chronicle consumer feed is not present")
    pin = load_uk_chronicle_feed()
    facts = tool._read_feed(
        feed,
        markers=(tool.NRS_RECORD_SET.encode(),),
        identity={
            "facts_sha256": pin.facts_sha256,
            "fact_row_count": pin.fact_row_count,
        },
    )
    _, receipt = tool._nrs_band_counts(facts)
    assert receipt == provenance["feed_facts"]["nrs_dwellings_by_band"], (
        "NRS's band facts differ on the pinned feed from those the conditioned "
        "supports were built from: rebuild them with "
        "tools/build_uk_conditioned_area_supports.py (docs/uk-chronicle-feed-repin.md)."
    )


def test_spec_declares_the_conditioned_bytes_and_their_vintages(
    provenance, bundle
) -> None:
    sources = {row["id"]: row for row in bundle.resource("sources")["sources"]}
    records = {row["id"]: row for row in bundle.resource("vintages")["records"]}
    register = uk_atomic_support_register(conditioned=True)
    for system in SYSTEMS:
        row, base = sources[CONDITIONED_SOURCES[system]], sources[SOURCES[system]]
        entry = provenance["supports"][system]
        assert row["role"] == "uk_atomic_area_conditioned_support"
        assert (row["sha256"], row["byte_size"]) == (
            entry["sha256"],
            entry["size_bytes"],
        )
        assert register[system] == {
            "sha256": entry["sha256"],
            "size_bytes": entry["size_bytes"],
        }
        assert entry["base"] == {"source": SOURCES[system], "sha256": base["sha256"]}
        added = [
            v.removeprefix("vintage:")
            for v in row["vintages"]
            if v not in base["vintages"]
        ]
        assert row["vintages"][: len(base["vintages"])] == base["vintages"]
        assert {
            a["id"]: a["value"] for a in row["vintage_authorities"]
        } == CONDITIONING_VINTAGES[system]
        assert added == list(CONDITIONING_VINTAGES[system])
        for vintage_id in added:
            record = records[vintage_id]
            assert record["kind"] == "geography_vintage_ref"
            assert record["authority_ref"] == {
                "kind": "source_record",
                "source": f"source:{CONDITIONED_SOURCES[system]}",
                "authority": vintage_id,
            }
        assert (
            entry["column_metadata"][IMD_DECILE_COLUMN]["vintage"]
            == CONDITIONING_VINTAGES[system][DECILE_VINTAGE[system]]
        )


def test_receipts_count_areas_and_households_only(provenance) -> None:
    base = json.loads((_UK / "uk_atomic_area_supports.provenance.json").read_text())
    for system in SYSTEMS:
        entry = provenance["supports"][system]
        receipt = entry["imd_decile"]
        assert receipt["areas"] == base["supports"][system]["areas"]
        assert sum(receipt["areas_by_rule"].values()) == receipt["areas"]
        assert set(entry["households_by_imd_decile"]) == set(IMD_DECILES)
        assert (
            sum(entry["households_by_imd_decile"].values())
            == base["supports"][system]["households_total"]
        )
        levels = [
            (tuple(e["constraints"]), e["weight"]) for e in entry["zero_mass_cells"]
        ]
        assert levels == [
            (level, weight)
            for level in (("frs_region", IMD_DECILE_COLUMN), ("frs_region",))
            for weight in ("households", *BAND_WEIGHT_COLUMNS[system])
        ]
        bands = entry["band_weights"]
        if system == NI:
            assert bands is None
            continue
        assert set(bands["column_totals"]) == set(BAND_WEIGHT_COLUMNS[system])
        # Half a unit per band at most, of households × 10^6.
        assert 2 * bands["largest_area_deviation"] <= len(BAND_WEIGHT_COLUMNS[system])
        if system == EW:
            assert bands["areas_with_rounding"] == 0


def _mounted(system: str) -> bytes:
    path = _REPOSITORY / "build/uk/supports" / f"{CONDITIONED_SOURCES[system]}.npz"
    if not path.is_file():
        pytest.skip(f"{path.name} is not mounted under build/uk/supports")
    return path.read_bytes()


@pytest.mark.parametrize("system", SYSTEMS)
def test_mounted_conditioned_support_extends_the_registered_support(
    provenance, system
) -> None:
    payload = _mounted(system)
    entry = provenance["supports"][system]
    assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
    assert len(payload) == entry["size_bytes"]
    base = uk_conditioned_support_base(payload)
    assert (
        hashlib.sha256(base).hexdigest()
        == uk_atomic_support_register()[system]["sha256"]
    )
    support = decode_atomic_support(payload)
    assert {
        c: dict(support.metadata["columns"][c]) for c in entry["column_metadata"]
    } == entry["column_metadata"]
    households = pd.Series(np.asarray(support.arrays["households"], dtype=np.int64))
    by_decile = households.groupby(np.asarray(support.arrays[IMD_DECILE_COLUMN])).sum()
    assert {k: int(by_decile.get(k, 0)) for k in IMD_DECILES} == entry[
        "households_by_imd_decile"
    ]
    assert (
        zero_mass_cells(
            support,
            domains={
                "frs_region": sorted(set(support.arrays["frs_region"].tolist())),
                IMD_DECILE_COLUMN: IMD_DECILES,
            },
            levels=(("frs_region", IMD_DECILE_COLUMN), ("frs_region",)),
            weights=("households", *BAND_WEIGHT_COLUMNS[system]),
        )
        == entry["zero_mass_cells"]
    )


def test_mounted_scottish_band_weights_follow_each_data_zone(provenance) -> None:
    support = decode_atomic_support(_mounted(SCOT))
    frame = pd.DataFrame(
        {
            "zone": np.asarray(support.arrays["lsoa_code"]),
            "households": np.asarray(support.arrays["households"], dtype=np.int64),
            **{
                c: np.asarray(support.arrays[c], dtype=np.int64)
                for c in BAND_WEIGHT_COLUMNS[SCOT]
            },
        }
    )
    totals = frame.groupby("zone").sum()
    for column in BAND_WEIGHT_COLUMNS[SCOT]:
        # Within a data zone every area holds the zone's band share of its
        # households, so its weight is that share times households, rounded.
        share = frame["zone"].map(totals[column] / totals["households"])
        assert np.all(np.abs(frame[column] - frame["households"] * share) <= 1.0)
