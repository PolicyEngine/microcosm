"""The committed stock cell table and the land under homes (microcosm#1524).

Every test reads the committed resources alone; none needs a raw input. The
last one also reads the pinned Chronicle feed when it is present, and fails
once the feed carries inputs the resources were not built from: ONS's
dwellings rows, for one, when chronicle#438 lands.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from importlib import resources as importlib_resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import stock_cell_declarations as declared
from microcosm.build.uk_runtime import stock_cells
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
UK = importlib_resources.files("microcosm.build.uk")
_PINNED_FEED_NAME = ".codex-work/consumer_facts_uk.jsonl"
# Census 2021 TS054, England: households renting from the council or other
# social landlords over all households, and owner occupiers over private
# households (the microcosm#1597 review's figures).
_TS054_ENGLAND_SOCIAL_SHARE = 4_005_663 / 23_436_085
_TS054_ENGLAND_OWNER_SHARE_OF_PRIVATE = 14_369_065 / 19_430_422
# VOA CTSOP 1.1, England and Wales, all properties, 31 March 2026.
_VOA_ENGLAND_AND_WALES_DWELLINGS = 27_438_780


@pytest.fixture(scope="module")
def committed():
    value = stock_cells.load_uk_stock_cells_value()
    structure = stock_cells.load_uk_stock_cells_structure(value=value)
    authorities = stock_cells.load_uk_la_land_shares()
    seats = stock_cells.load_uk_constituency_land_totals()
    provenance = stock_cells.load_uk_stock_cells_provenance()
    return value, structure, authorities, seats, provenance


def test_every_authority_in_england_and_wales_appears_once(committed) -> None:
    _, _, authorities, _, _ = committed
    crosswalk = json.loads(UK.joinpath("local_area_crosswalk.json").read_text("utf-8"))
    roster = [
        code
        for code in crosswalk["levels"]["local_authority"]["area_ids"]
        if code[0] in "EW"
    ]
    english = [code for code in authorities.codes if code.startswith("E")]
    assert len(english) == len(set(english)) == 296
    assert sorted(authorities.codes) == sorted(roster)


def test_cells_sum_to_authorities_and_authorities_to_the_total(committed) -> None:
    value, structure, authorities, seats, _ = committed
    land = pd.Series(value.private_count * structure.land_per_home)
    by_authority = land.groupby(value.authority_of_rows()).sum()
    for i, code in enumerate(authorities.codes):
        count, _ = value.social[code]
        expected = by_authority[code] + count * structure.social[code][3]
        assert expected == pytest.approx(
            authorities.land["central"][i], rel=1e-6, abs=2
        ), code
    for name, land_of_authorities in authorities.land.items():
        assert seats.land[name].sum() == pytest.approx(
            land_of_authorities.sum(), rel=1e-6
        ), name
        assert seats.value[name].sum() == pytest.approx(
            authorities.value[name].sum(), rel=1e-6
        ), name


def test_land_shares_lie_in_the_unit_interval_and_fall_with_kappa(committed) -> None:
    _, _, authorities, _, _ = committed
    for name, share in authorities.land_share.items():
        assert np.all(share >= 0) and np.all(share <= 1), name
    low, one, high = (
        authorities.land[n] for n in ("kappa_0_75", "kappa_1", "kappa_1_25")
    )
    assert np.all(low >= one - 1) and np.all(one >= high - 1)


def test_the_tenure_split_matches_the_census(committed) -> None:
    """The structure tests passed when the social stock was half the
    census's, so these pin the quantities themselves (microcosm#1597
    review). VOA's counts and the census's households differ a little."""
    value, *_ = committed
    english = np.char.startswith(value.authority_of_rows().astype(str), "E")
    private = value.private_count[english].sum()
    social = value.social_count[english].sum()
    assert social / (social + private) == pytest.approx(
        _TS054_ENGLAND_SOCIAL_SHARE, abs=0.005
    )
    owners = (value.private_count * value.owner_share)[english].sum()
    assert owners / private == pytest.approx(
        _TS054_ENGLAND_OWNER_SHARE_OF_PRIVATE, abs=0.005
    )


def test_the_cells_hold_voas_dwellings(committed) -> None:
    value, *_, provenance = committed
    split = provenance["level"]["uk_split"]
    assert split["england_and_wales"] == _VOA_ENGLAND_AND_WALES_DWELLINGS
    # VOA rounds small areas' counts and the cells read `[c]` as 2.5, so
    # the cells' sum is close to the national count, not equal to it.
    assert value.voa_count.sum() == pytest.approx(
        _VOA_ENGLAND_AND_WALES_DWELLINGS, rel=2e-3
    )
    assert (value.private_count + value.social_count).sum() == pytest.approx(
        value.voa_count.sum(), rel=1e-9
    )


def test_three_in_ten_sold_homes_take_a_pooled_floor_area(committed) -> None:
    """UCL's linked file ends in October 2023. A home with no linked floor
    area takes its cell's mean, and seldom one from further away."""
    *_, provenance = committed
    imputation = provenance["receipts"]["floor_area_imputation"]
    imputed = imputation["imputed"]
    assert imputation["imputed_share"] == pytest.approx(0.304, abs=0.03)
    assert (
        provenance["level"]["by_variant"]["central"]["homes_floor_area_imputed"]
        == imputed
    )
    for breakdown in ("by_region", "by_sale_year"):
        assert sum(row["imputed"] for row in imputation[breakdown].values()) == imputed
    by_level = imputation["imputed_by_pool_level"]
    assert sum(by_level.values()) == imputed
    # An authority's, region's or nation's mean stands in for under 5%.
    assert (imputed - by_level["msoa_band"]) / imputed < 0.05


def test_the_digests_chain(committed) -> None:
    value, structure, authorities, seats, provenance = committed
    digests = provenance["digests"]
    assert structure.value_block_sha256 == value.digest == digests["value_block_sha256"]
    assert structure.digest == digests["structure_block_sha256"]
    for name in ("authorities", "constituencies"):
        payload = json.loads(
            UK.joinpath(stock_cells.STOCK_CELL_RESOURCES[name]).read_text("utf-8")
        )
        assert stock_cells.resource_digest(payload) == digests[f"{name}_sha256"]
    assert authorities.digests == seats.digests
    assert value.inputs_sha256 == provenance["inputs_sha256"]


def test_every_check_is_the_declared_one_and_carries_a_status(committed) -> None:
    *_, provenance = committed
    assert [entry["id"] for entry in provenance["decision_rule"]] == [
        rule["id"] for rule in declared.DECISION_RULE
    ]
    assert [entry["id"] for entry in provenance["measurements"]] == [
        entry["id"] for entry in declared.MEASUREMENTS
    ]
    central = provenance["level"]["central"]
    assert central["status"] in ("feed", "parameter") or central["status"].startswith(
        "absent"
    )
    for name, entry in provenance["level"]["inputs"].items():
        assert entry["status"] in ("feed", "parameter", "absent"), name


def _load_tool():
    name = "build_uk_stock_cells"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, PATHS.repository / "tools" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_the_pinned_feed_carries_the_inputs_the_cells_were_built_from(
    committed,
) -> None:
    """The re-pin check: a pinned feed whose inputs differ from those the
    resources record, ONS's level inputs among them, needs the resources
    rebuilt with tools/build_uk_stock_cells.py."""
    configured = os.environ.get("CHRONICLE_UK_FACTS")
    feed = Path(configured) if configured else PATHS.repository / _PINNED_FEED_NAME
    if feed.is_dir():
        feed = feed / "consumer_facts.jsonl"
    if not feed.is_file():
        pytest.skip("the pinned UK Chronicle consumer feed is not present")
    *_, provenance = committed
    tool = _load_tool()
    facts = tool.ref.read_feed_facts(feed)
    fingerprint = tool.ref.feed_inputs_fingerprint(tool.feed_inputs(facts))
    assert fingerprint == provenance["inputs"]["feed_inputs_sha256"], (
        "The pinned feed carries other inputs than the stock cells were built from: "
        "rebuild them with tools/build_uk_stock_cells.py."
    )
