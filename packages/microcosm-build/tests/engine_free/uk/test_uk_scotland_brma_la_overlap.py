"""Scotland's BRMA x council overlap and the private-rent decision it backs.

Invariants of the resource (hold for the committed table):

1. Coverage: the cells cover all 18 BRMAs and exactly the crosswalk's 32
   Scottish local authorities, once each per intersection.
2. Conservation: each weight's cells sum to its census output-area total, and
   per-council and per-BRMA shares each sum to one.
3. Bounds: every cell holds households, and no cell or council has more
   private renters than households.
4. Differential: the private-rented and all-household weights place every
   council in the same main BRMA and join the same blocks; the private
   weight occupies a subset of the all-household cells.
5. One geography: the table's BRMAs, codes and output areas are the
   committed area-to-BRMA layer's.

Invariants of the sum in ``tools/build_uk_scotland_brma_la_overlap.py`` (hold
for every input, property-tested): each output area lands in one cell, so
each weight is conserved by council, by BRMA and in total.

Properties of the rejected rule (hold for every set of BRMA rents): the
household-weighted mean of BRMA rents conserves the total rent bill, and
gives each council wholly inside one BRMA exactly that BRMA's rent, so it
cannot express any rent difference between councils that share a BRMA.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from importlib import resources as importlib_resources
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.uk_runtime import brma_area_layer as bal
from microcosm.build.uk_runtime import scotland_brma_la_overlap as overlap_module
from microcosm.build.uk_runtime.scotland_brma_la_overlap import (
    COINCIDENT_MIN_SHARE,
    OVERLAP_WEIGHTS,
    load_uk_scotland_brma_la_overlap,
    scotland_brma_council_blocks,
    scotland_brma_council_overlaps,
    scotland_coincident_councils,
)
from test_support.paths import paths_for
from tools import build_uk_scotland_brma_la_overlap as tool

_REPOSITORY = paths_for("microcosm-build").repository
_CELLS_SHA256 = "1860f04edd3319f8c786de236ad9371a5a46a873e7dc8bcb348ecae6644f226d"
_AYRSHIRE_COUNCILS = frozenset({"S12000008", "S12000021", "S12000028"})
_COINCIDENT = {
    "S12000006": "DUMFRIES_AND_GALLOWAY",
    "S12000026": "SCOTTISH_BORDERS",
    "S12000040": "WEST_LOTHIAN",
    "S12000047": "FIFE",
}
_STABLE_UK_FACT_FEED_NAME = ".codex-work/consumer_facts_uk.jsonl"


def _resource() -> dict:
    return dict(load_uk_scotland_brma_la_overlap())


def _packaged(name: str) -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk").joinpath(name).read_text()
    )


def _layer() -> bal.BRMAAreaLayer:
    return bal.load_brma_area_layer(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("brma_area_layer.json")
        .read_bytes()
    )


def _scottish_roster() -> set[str]:
    crosswalk = _packaged("local_area_crosswalk.json")
    return {
        area_id
        for area_id in crosswalk["levels"]["local_authority"]["area_ids"]
        if area_id.startswith("S")
    }


def _weighted_mean_rule(brma_rents: dict[str, float]) -> dict[str, float]:
    """The candidate rule: a council's BRMA rents, weighted by its renters."""

    values: dict[str, float] = {}
    for item in scotland_brma_council_overlaps():
        values[item.local_authority] = (
            values.get(item.local_authority, 0.0)
            + item.share_of_council * brma_rents[item.brma]
        )
    return values


# --- the resource ---------------------------------------------------------


def test_cells_are_pinned_and_cover_both_rosters() -> None:
    resource = _resource()
    cells = resource["cells"]
    canonical = json.dumps(cells, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == _CELLS_SHA256
    assert len(cells) == resource["totals"]["cells"] == 63
    assert {cell["brma"] for cell in cells} == set(resource["brmas"])
    assert len(resource["brmas"]) == 18
    assert {cell["local_authority"] for cell in cells} == _scottish_roster()
    assert len(_scottish_roster()) == 32
    keys = [(cell["brma"], cell["local_authority"]) for cell in cells]
    assert len(keys) == len(set(keys))
    assert keys == sorted(keys, key=lambda key: (key[1], key[0]))


def test_cells_conserve_the_census_totals() -> None:
    resource = _resource()
    cells, totals = resource["cells"], resource["totals"]
    assert totals["private_rented_households"] == 322_903
    assert totals["all_households"] == 2_508_542
    assert totals["output_areas"] == 46_363
    assert totals["output_areas_without_private_renters"] == 4_910
    for field in ("output_areas", *OVERLAP_WEIGHTS):
        assert sum(cell[field] for cell in cells) == totals[field]


def test_private_renters_never_exceed_households() -> None:
    cells = _resource()["cells"]
    by_council: Counter[str] = Counter()
    households: Counter[str] = Counter()
    for cell in cells:
        assert cell["all_households"] > 0 and cell["output_areas"] > 0
        assert 0 <= cell["private_rented_households"] <= cell["all_households"]
        by_council[cell["local_authority"]] += cell["private_rented_households"]
        households[cell["local_authority"]] += cell["all_households"]
    assert all(0 < by_council[code] < households[code] for code in households)


@pytest.mark.parametrize("weight", OVERLAP_WEIGHTS)
def test_shares_sum_to_one_on_each_side(weight: str) -> None:
    by_council: dict[str, float] = {}
    by_brma: dict[str, float] = {}
    for item in scotland_brma_council_overlaps(weight):
        assert 0 < item.share_of_council <= 1
        assert 0 < item.share_of_brma <= 1
        by_council[item.local_authority] = (
            by_council.get(item.local_authority, 0.0) + item.share_of_council
        )
        by_brma[item.brma] = by_brma.get(item.brma, 0.0) + item.share_of_brma
    assert len(by_council) == 32 and len(by_brma) == 18
    assert all(abs(total - 1.0) <= 1e-12 for total in by_council.values())
    assert all(abs(total - 1.0) <= 1e-12 for total in by_brma.values())


def test_both_weights_describe_the_same_geography() -> None:
    def main_brma(weight: str) -> dict[str, str]:
        best: dict[str, tuple[float, str]] = {}
        for item in scotland_brma_council_overlaps(weight):
            if item.share_of_council > best.get(item.local_authority, (0.0, ""))[0]:
                best[item.local_authority] = (item.share_of_council, item.brma)
        return {council: brma for council, (_, brma) in best.items()}

    private, everyone = (
        {
            (item.brma, item.local_authority)
            for item in scotland_brma_council_overlaps(w)
        }
        for w in OVERLAP_WEIGHTS
    )
    # Output-area counts are exact, so a sliver can hold no private renters.
    assert private <= everyone
    assert len(everyone - private) == 1
    assert main_brma("private_rented_households") == main_brma("all_households")
    assert scotland_brma_council_blocks(
        "private_rented_households"
    ) == scotland_brma_council_blocks("all_households")


def test_resource_is_built_on_the_committed_layer_and_support() -> None:
    resource = _resource()
    method = resource["method"]
    layer_bytes = (tool.LAYER).read_bytes()
    assert method["layer_sha256"] == hashlib.sha256(layer_bytes).hexdigest()
    supports = _packaged("uk_atomic_area_supports.provenance.json")["supports"]
    assert method["support_sha256"] == supports[tool.SCOTLAND_SUPPORT]["sha256"]
    assert method["uv404"] == tool.UV404
    layer = _layer()
    assert resource["brmas"] == {
        brma: {"pipr_area_code": code, "pipr_area_name": tool.PIPR_NAMES[brma]}
        for brma, code in sorted(layer.scotland_codes.items())
    }
    assert sorted(entry["pipr_area_code"] for entry in resource["brmas"].values()) == [
        f"S330000{index:02d}" for index in range(1, 19)
    ]
    assert resource["totals"]["output_areas"] == len(
        layer.brma_by_area["scotland_oa22"]
    )
    by_brma = Counter(layer.brma_by_area["scotland_oa22"].values())
    for brma, count in by_brma.items():
        assert count == sum(
            cell["output_areas"] for cell in resource["cells"] if cell["brma"] == brma
        )


def _mutated(change) -> dict:
    payload = json.loads(json.dumps(_resource()))
    change(payload)
    return payload


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda p: p["cells"][0].update(local_authority="E06000001"),
            "not a Scottish local authority",
        ),
        (lambda p: p["cells"].append(dict(p["cells"][0])), "duplicate cell"),
        (lambda p: p["cells"][0].update(brma="CENTRAL_LONDON"), "unknown BRMA"),
        (
            lambda p: p["cells"][0].update(private_rented_households=-1),
            "invalid private_rented_households",
        ),
        (
            lambda p: p["cells"][0].update(private_rented_households=1.5),
            "invalid private_rented_households",
        ),
        (
            lambda p: p["cells"][0].update(private_rented_households=True),
            "invalid private_rented_households",
        ),
        (lambda p: p["cells"][0].update(all_households=0), "holds no households"),
        (lambda p: p["cells"][0].update(output_areas=0), "holds no households"),
        (
            lambda p: p["cells"][0].update(
                private_rented_households=p["cells"][0]["all_households"] + 1
            ),
            "more private renters than households",
        ),
        (
            lambda p: p["totals"].update(private_rented_households=322_000),
            "not the declared total",
        ),
        (lambda p: p.update(schema_version=1), "schema_version"),
        (
            lambda p: p.update(
                cells=[c for c in p["cells"] if c["local_authority"] != "S12000027"]
            ),
            "has no cell",
        ),
    ],
)
def test_validation_refuses_a_broken_table(change, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        overlap_module.validate_uk_scotland_brma_la_overlap(_mutated(change))


def _pinned_feed() -> Path:
    configured = os.environ.get("CHRONICLE_UK_FACTS")
    feed = Path(configured) if configured else _REPOSITORY / _STABLE_UK_FACT_FEED_NAME
    if feed.is_dir():
        feed = feed / "consumer_facts.jsonl"
    if not feed.is_file():
        pytest.skip("pinned UK Chronicle national consumer feed is not present")
    return feed


def test_rebuild_from_the_feed_and_the_supports_matches(capsys) -> None:
    feed = _pinned_feed()
    supports = _REPOSITORY / "build/uk/supports"
    declared = _packaged("uk_atomic_area_supports.provenance.json")["supports"]
    if not (supports / declared[tool.SCOTLAND_SUPPORT]["filename"]).is_file():
        pytest.skip("the Scotland support is not mounted under build/uk/supports")
    code = tool.main(
        ["--ledger-facts", str(feed), "--supports", str(supports), "--check"]
    )
    assert code == 0, capsys.readouterr().err


# --- the sum in the tool --------------------------------------------------


@st.composite
def _output_areas(draw):
    areas = draw(st.integers(min_value=1, max_value=40))
    brma_by_area, council_by_area = {}, {}
    counts: dict[str, dict[str, int]] = {
        "all_households": {},
        "private_rented_households": {},
    }
    for index in range(areas):
        area = f"S00{index:06d}"
        brma_by_area[area] = draw(st.sampled_from(["FIFE", "LOTHIAN", "AYRSHIRES"]))
        council_by_area[area] = f"S12{draw(st.integers(min_value=0, max_value=3)):06d}"
        households = draw(st.integers(min_value=1, max_value=400))
        counts["all_households"][area] = households
        if draw(st.booleans()):
            counts["private_rented_households"][area] = draw(
                st.integers(min_value=0, max_value=households)
            )
    return brma_by_area, council_by_area, counts


@settings(max_examples=200, deadline=None)
@given(_output_areas())
def test_cells_conserve_each_weight_by_council_brma_and_total(case) -> None:
    brma_by_area, council_by_area, counts = case
    cells = tool.overlap_cells(brma_by_area, council_by_area, counts)
    keys = [(cell["local_authority"], cell["brma"]) for cell in cells]
    assert keys == sorted(set(keys))
    assert sum(cell["output_areas"] for cell in cells) == len(brma_by_area)
    for weight in OVERLAP_WEIGHTS:
        values = counts[weight]
        assert sum(cell[weight] for cell in cells) == sum(values.values())
        for side, lookup in (
            ("local_authority", council_by_area),
            ("brma", brma_by_area),
        ):
            for key in set(lookup.values()):
                assert sum(c[weight] for c in cells if c[side] == key) == sum(
                    values.get(area, 0)
                    for area, value in lookup.items()
                    if value == key
                )
    assert all(
        0 <= cell["private_rented_households"] <= cell["all_households"]
        for cell in cells
    )


# --- the structure the decision rests on ----------------------------------


def test_brmas_and_councils_do_not_nest() -> None:
    overlaps = scotland_brma_council_overlaps()
    councils_per_brma = Counter(item.brma for item in overlaps)
    brmas_per_council = Counter(item.local_authority for item in overlaps)
    assert sum(count > 1 for count in brmas_per_council.values()) == 18
    assert sum(count > 1 for count in councils_per_brma.values()) == 16
    blocks = scotland_brma_council_blocks()
    assert sorted((len(brmas), len(councils)) for brmas, councils in blocks) == [
        (1, 3),
        (17, 29),
    ]
    ayrshire = next(block for block in blocks if len(block[1]) == 3)
    assert ayrshire == (frozenset({"AYRSHIRES"}), _AYRSHIRE_COUNCILS)


def test_four_councils_coincide_with_one_brma() -> None:
    assert COINCIDENT_MIN_SHARE == 0.99
    assert dict(scotland_coincident_councils()) == _COINCIDENT
    # Loosening the bar adds councils; it never removes one.
    looser = scotland_coincident_councils(min_share=0.95)
    assert set(_COINCIDENT) < set(looser)
    with pytest.raises(ValueError, match="min_share"):
        scotland_coincident_councils(min_share=0.5)


@settings(max_examples=100, deadline=None)
@given(
    st.floats(min_value=0.0, max_value=1.0),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_blocks_partition_and_refine_as_slivers_are_dropped(a: float, b: float) -> None:
    low, high = sorted((a, b))
    coarse = scotland_brma_council_blocks(min_share=low)
    fine = scotland_brma_council_blocks(min_share=high)
    for blocks in (coarse, fine):
        brmas = [brma for block in blocks for brma in block[0]]
        councils = [council for block in blocks for council in block[1]]
        assert len(brmas) == len(set(brmas)) == 18
        assert len(councils) == len(set(councils)) == 32
        assert all(block[0] or block[1] for block in blocks)
    assert len(fine) >= len(coarse)
    for brmas, councils in fine:
        assert any(brmas <= big[0] and councils <= big[1] for big in coarse)


# --- the rejected rule ----------------------------------------------------

_BRMAS = sorted(tool.PIPR_NAMES)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.floats(min_value=100.0, max_value=5_000.0),
        min_size=len(_BRMAS),
        max_size=len(_BRMAS),
    )
)
def test_weighted_mean_rule_gives_councils_their_brma_average(rents) -> None:
    brma_rents = dict(zip(_BRMAS, rents, strict=True))
    values = _weighted_mean_rule(brma_rents)
    overlaps = scotland_brma_council_overlaps()

    # A council's value is a blend of BRMA averages and nothing else.
    assert all(min(rents) - 1e-9 <= v <= max(rents) + 1e-9 for v in values.values())
    # It conserves the rent bill: households x rent agrees on both sides.
    council_households: dict[str, float] = {}
    bill = 0.0
    for item in overlaps:
        council_households[item.local_authority] = (
            council_households.get(item.local_authority, 0.0) + item.households
        )
        bill += item.households * brma_rents[item.brma]
    total = sum(council_households[code] * values[code] for code in values)
    assert abs(total - bill) <= 1e-9 * bill
    # A council wholly inside one BRMA gets that BRMA's rent, unchanged, so
    # councils sharing a BRMA cannot differ.
    whole = {
        item.local_authority: item.brma
        for item in overlaps
        if item.share_of_council == 1.0
    }
    assert len(whole) == 14
    for council, brma in whole.items():
        assert values[council] == brma_rents[brma]
    shared = Counter(whole.values())
    assert sum(count for count in shared.values() if count > 1) == 11


def test_rent_dispersion_receipt_backs_the_decision() -> None:
    receipt = json.loads(
        (
            _REPOSITORY / "experiments/1090-scotland-brma-la-rent-dispersion.json"
        ).read_text(encoding="utf-8")
    )
    summary = receipt["summary"]
    gaps = sorted(
        abs(row["rule_over_records_mean_pct"])
        for row in receipt["councils"]
        if "rule_over_records_mean_pct" in row
    )
    assert summary == {
        "councils_compared": len(gaps),
        "median_abs_gap_pct": gaps[len(gaps) // 2],
        "max_abs_gap_pct": gaps[-1],
        "councils_over_5_pct": sum(gap > 5 for gap in gaps),
        "councils_over_10_pct": sum(gap > 10 for gap in gaps),
    }
    assert (summary["councils_compared"], summary["max_abs_gap_pct"]) == (31, 30.9)
    assert summary["councils_over_10_pct"] == 9
    by_code = {row["local_authority"]: row for row in receipt["councils"]}
    # The coincident councils are the ones the rule gets right.
    for council in _COINCIDENT:
        assert abs(by_code[council]["rule_over_records_mean_pct"]) <= 0.1
    assert set(by_code) == _scottish_roster()
    # The receipt was built from the committed overlap and layer.
    assert receipt["overlap_resource"] == str(tool.RESOURCE.relative_to(tool.REPO_ROOT))
    assert (
        receipt["layer_sha256"] == hashlib.sha256(tool.LAYER.read_bytes()).hexdigest()
    )
    main = {}
    for item in scotland_brma_council_overlaps():
        if item.share_of_council > main.get(item.local_authority, (0.0, ""))[0]:
            main[item.local_authority] = (item.share_of_council, item.brma)
    for council, (share, brma) in main.items():
        assert by_code[council]["main_brma"] == brma
        assert by_code[council]["share_of_council_in_main_brma"] == round(share, 4)
