"""Scotland's BRMA x council overlap and the private-rent decision it backs.

Invariants of the resource (hold for the committed table):

1. Coverage: the cells cover all 18 BRMAs and exactly the crosswalk's 32
   Scottish local authorities, once each per intersection.
2. Conservation: private-rented cells sum to the census ward total and
   all-household cells to the output-area total; per-council and per-BRMA
   shares each sum to one.
3. Bounds: every cell holds households, and no cell or council has more
   private renters than households.
4. Differential: the private-rented and all-household weights place every
   council in the same main BRMA and occupy the same cells.

Invariants of the allocation in ``tools/build_uk_scotland_brma_la_overlap.py``
(hold for every input, property-tested): splitting ward counts across BRMAs
conserves each ward's total, each council's total and the grand total.

Properties of the rejected rule (hold for every set of BRMA rents): the
household-weighted mean of BRMA rents conserves the total rent bill, and
gives each council wholly inside one BRMA exactly that BRMA's rent, so it
cannot express any rent difference between councils that share a BRMA.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from importlib import resources as importlib_resources
from pathlib import Path

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.uk_runtime import data_target_parity
from microcosm.build.uk_runtime import scotland_brma_la_overlap as overlap_module
from microcosm.build.uk_runtime.ledger_targets import (
    UK_CROSS_GRAIN_BRIDGES,
    UK_CROSS_GRAIN_GRAIN_PRECEDENCE,
)
from microcosm.build.uk_runtime.local_targets import (
    AREA_TYPES,
    load_uk_population_contract,
)
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
from tools.generate_uk_local_target_references import _area_signed_deferrals

_REPOSITORY = paths_for("microcosm-build").repository
_CELLS_SHA256 = "f9c6c79c7eba5630fdfca97dd12bc326711584a0236f4d9017521e637bfbeab5"
_AYRSHIRE_COUNCILS = frozenset({"S12000008", "S12000021", "S12000028"})
_COINCIDENT = {
    "S12000006": "DUMFRIES_AND_GALLOWAY",
    "S12000026": "SCOTTISH_BORDERS",
    "S12000040": "WEST_LOTHIAN",
    "S12000047": "FIFE",
}


def _resource() -> dict:
    return dict(load_uk_scotland_brma_la_overlap())


def _packaged(name: str) -> dict:
    return json.loads(
        importlib_resources.files("microcosm.build.uk").joinpath(name).read_text()
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
    assert totals["private_rented_households"] == 322_995
    assert totals["all_households"] == 2_508_542
    assert totals["output_areas"] == 46_363
    assert totals["wards"] == 355
    assert totals["wards_split_across_brmas"] == 36
    private = math.fsum(cell["private_rented_households"] for cell in cells)
    # Each cell is rounded to four decimals, so 63 cells stay within 63 * 5e-5.
    assert abs(private - totals["private_rented_households"]) < 63 * 5e-5
    assert sum(cell["all_households"] for cell in cells) == totals["all_households"]


def test_private_renters_never_exceed_households() -> None:
    cells = _resource()["cells"]
    by_council: Counter[str] = Counter()
    households: Counter[str] = Counter()
    for cell in cells:
        assert cell["all_households"] > 0
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
    assert all(math.isclose(total, 1.0, abs_tol=1e-12) for total in by_council.values())
    assert all(math.isclose(total, 1.0, abs_tol=1e-12) for total in by_brma.values())


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
    assert private == everyone
    assert main_brma("private_rented_households") == main_brma("all_households")
    assert scotland_brma_council_blocks(
        "private_rented_households"
    ) == scotland_brma_council_blocks("all_households")


def test_resource_sources_are_the_tool_pins() -> None:
    resource = _resource()
    assert resource["sources"] == [dict(source) for source in tool.SOURCES]
    assert len(resource["sources"]) == 5
    for source in resource["sources"]:
        assert len(source["sha256"]) == 64
        assert int(source["sha256"], 16) >= 0
        assert source["url"].startswith("https://")
    assert resource["brmas"] == {
        brma: {"pipr_area_code": code, "pipr_area_name": name}
        for brma, (code, name) in sorted(tool.PIPR_BRMAS.items())
    }
    assert sorted(entry["pipr_area_code"] for entry in resource["brmas"].values()) == [
        f"S330000{index:02d}" for index in range(1, 19)
    ]
    assert resource["method"]["overrides"] == tool.OUTPUT_AREA_OVERRIDES


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
            lambda p: p["cells"][0].update(private_rented_households=-1.0),
            "invalid private_rented_households",
        ),
        (lambda p: p["cells"][0].update(all_households=0), "holds no households"),
        (
            lambda p: p["cells"][0].update(private_rented_households=float("nan")),
            "invalid private_rented_households",
        ),
        (
            lambda p: p["totals"].update(private_rented_households=322_000),
            "not the declared total",
        ),
        (lambda p: p.update(schema_version=2), "schema_version"),
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


def test_rebuild_from_the_pinned_originals_matches() -> None:
    cache = os.environ.get("UK_SCOTLAND_BRMA_LA_CACHE")
    uvx = shutil.which("uvx")
    if not cache or uvx is None:
        pytest.skip("UK_SCOTLAND_BRMA_LA_CACHE names no staged originals")
    # geopandas is not a workspace dependency, so the tool runs under uvx.
    extras = ("geopandas", "pandas", "openpyxl", "pyogrio")
    command = [uvx, *(arg for extra in extras for arg in ("--with", extra))]
    command += ["python", str(_REPOSITORY / "tools" / Path(tool.__file__).name)]
    result = subprocess.run(
        [*command, "--cache", cache, "--check"],
        cwd=_REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


# --- the allocation in the tool -------------------------------------------


@st.composite
def _geographies(draw):
    wards = draw(st.integers(min_value=1, max_value=6))
    rows = []
    counts = {}
    for index in range(wards):
        ward = f"S13{index:06d}"
        council = f"S12{draw(st.integers(min_value=0, max_value=2)):06d}"
        counts[ward] = draw(st.integers(min_value=0, max_value=5_000))
        for area in range(draw(st.integers(min_value=1, max_value=5))):
            rows.append(
                {
                    "output_area": f"S00{index:03d}{area:03d}",
                    "households": draw(st.integers(min_value=1, max_value=400)),
                    "ward": ward,
                    "council": council,
                    "brma": draw(st.sampled_from(["FIFE", "LOTHIAN", "AYRSHIRES"])),
                }
            )
    return pd.DataFrame(rows), counts


@settings(max_examples=200, deadline=None)
@given(_geographies())
def test_allocation_conserves_ward_council_and_total(case) -> None:
    output_areas, wards = case
    cells, totals = tool.overlap_cells(
        output_areas, wards, expected_councils=output_areas["council"].nunique()
    )

    def private(**match: str) -> float:
        return sum(
            cell["private_rented_households"]
            for cell in cells
            if all(cell[key] == value for key, value in match.items())
        )

    tolerance = len(cells) * 10**-tool.PRIVATE_RENTED_DECIMALS
    assert totals["private_rented_households"] == sum(wards.values())
    assert abs(private() - sum(wards.values())) <= tolerance
    assert sum(cell["all_households"] for cell in cells) == int(
        output_areas["households"].sum()
    )
    assert all(cell["private_rented_households"] >= 0 for cell in cells)
    ward_council = output_areas.groupby("ward")["council"].first()
    for council in set(output_areas["council"]):
        expected = sum(
            count for ward, count in wards.items() if ward_council[ward] == council
        )
        assert abs(private(local_authority=council) - expected) <= tolerance
    # When no ward crosses a BRMA boundary, each BRMA gets its wards' renters
    # and no within-ward assumption is in play.
    if output_areas.groupby("ward")["brma"].nunique().eq(1).all():
        ward_brma = output_areas.groupby("ward")["brma"].first()
        for brma in set(output_areas["brma"]):
            expected = sum(
                count for ward, count in wards.items() if ward_brma[ward] == brma
            )
            assert abs(private(brma=brma) - expected) <= tolerance


def test_allocation_refuses_a_ward_in_two_councils() -> None:
    output_areas = pd.DataFrame(
        [
            {
                "output_area": "a",
                "households": 5,
                "ward": "w",
                "council": "x",
                "brma": "FIFE",
            },
            {
                "output_area": "b",
                "households": 5,
                "ward": "w",
                "council": "y",
                "brma": "FIFE",
            },
        ]
    )
    with pytest.raises(SystemExit, match="spans two council areas"):
        tool.overlap_cells(output_areas, {"w": 3})


def _ward_export(rows: list[list[str]]) -> str:
    header = ["Number of bedrooms", *tool.BEDROOM_BANDS, "Total", ""]
    lines = [["SuperWEB2(tm)"], ["Counting: Households"]]
    for tenure in (" Owned: Owned outright", *tool.PRIVATE_RENTED_TENURES):
        lines += [[tenure], header, ["Electoral Ward 2022", ""]]
        lines += (
            rows
            if tenure in tool.PRIVATE_RENTED_TENURES
            else [[row[0], *["1000"] * 6, ""] for row in rows]
        )
    return "\n".join(",".join(f'"{cell}"' for cell in line) for line in lines) + "\n"


def test_ward_parser_sums_only_the_private_rented_wafers(tmp_path) -> None:
    rows = [
        [f"S13{index:06d}", "1", "2", "3", "4", "5", "999", ""]
        for index in range(tool.EXPECTED_WARDS)
    ]
    (tmp_path / "census_2022_ward2022_tenure_bedrooms.csv").write_text(
        _ward_export(rows), encoding="utf-8"
    )
    wards = tool.ward_private_rented_households(tmp_path)
    # Two private-rented wafers of 1+2+3+4+5; the owner wafer and the
    # separately perturbed Total column stay out.
    assert set(wards.values()) == {30}
    assert len(wards) == tool.EXPECTED_WARDS


def test_ward_parser_refuses_a_suppressed_export(tmp_path) -> None:
    (tmp_path / "census_2022_ward2022_tenure_bedrooms.csv").write_text(
        '"Counting: Households"\n"ERROR","table too sparse"\n', encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="suppression"):
        tool.ward_private_rented_households(tmp_path)


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

_BRMAS = sorted(tool.PIPR_BRMAS)


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
    assert math.isclose(
        sum(council_households[code] * values[code] for code in values),
        bill,
        rel_tol=1e-9,
    )
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


def test_rent_dispersion_receipt_backs_the_parity_evidence() -> None:
    receipt = json.loads(
        (
            _REPOSITORY / "experiments/1090-scotland-brma-la-rent-dispersion.json"
        ).read_text(encoding="utf-8")
    )
    summary = receipt["summary"]
    gaps = [
        abs(row["rule_over_records_mean_pct"])
        for row in receipt["councils"]
        if "rule_over_records_mean_pct" in row
    ]
    assert summary == {
        "councils_compared": len(gaps),
        "median_abs_gap_pct": sorted(gaps)[len(gaps) // 2],
        "max_abs_gap_pct": max(gaps),
        "councils_over_5_pct": sum(gap > 5 for gap in gaps),
        "councils_over_10_pct": sum(gap > 10 for gap in gaps),
    }
    assert (summary["councils_compared"], summary["max_abs_gap_pct"]) == (31, 30.9)
    assert summary["councils_over_10_pct"] == 9
    by_code = {row["local_authority"]: row for row in receipt["councils"]}
    # The coincident councils are the ones the rule gets right.
    for council in _COINCIDENT:
        assert abs(by_code[council]["rule_over_records_mean_pct"]) <= 0.1
    assert len(receipt["councils"]) == 32
    assert receipt["overlap_resource"] == str(tool.RESOURCE)


# --- the concern, the deferral and the fence ------------------------------


def test_parity_concern_states_the_measured_overlap() -> None:
    rows = {
        row["concern_id"]: row
        for row in data_target_parity.build_uk_data_target_parity()["concerns"]
    }
    concern = rows["cross_grain_private_rent_scotland_brma"]
    assert concern["status"] == "routed"
    assert "microcosm#1090" in concern["reason"]
    totals = _resource()["totals"]
    overlaps = scotland_brma_council_overlaps()
    big = max(scotland_brma_council_blocks(), key=lambda block: len(block[1]))
    for claim in (
        f"{totals['private_rented_households']:,} private-rented households in "
        f"{totals['cells']} BRMA x council cells",
        f"{sum(c > 1 for c in Counter(i.local_authority for i in overlaps).values())} "
        "councils span more than one BRMA",
        f"{sum(c > 1 for c in Counter(i.brma for i in overlaps).values())} "
        "BRMAs span more than one council",
        f"{len(big[1])} councils and {len(big[0])} BRMAs form one connected block",
        f"{len(scotland_coincident_councils())} councils coincide with one BRMA at 99%",
        "up to 30.9%",
        "9 of 31 councils",
    ):
        assert claim in concern["evidence"], claim
    assert (
        "Keep all 32 Scottish LA cells signed deferred"
        in (concern["fence"]["verdict_basis"])
    )


def test_committed_deferral_is_the_generator_output() -> None:
    membership = _packaged("local_target_reference_membership.json")
    generated = {
        deferral.reason_id: deferral
        for deferral in _area_signed_deferrals(
            load_uk_population_contract(), _packaged("local_area_crosswalk.json")
        )
        if deferral.target_id == "ons.rent.private_rent"
    }
    deferral = generated["private_rent_pipr_scotland_brma_grain"]
    committed = next(
        row
        for row in membership["signed_deferrals"]
        if row["reason_id"] == deferral.reason_id
    )
    assert committed["rationale"] == deferral.rationale
    assert (
        committed["area_ids"] == list(deferral.area_ids) == sorted(_scottish_roster())
    )
    candidates = membership["targets"]["ons.rent.private_rent"]["geography_levels"][
        "local_authority"
    ]["candidates"]
    scottish = [row for row in candidates if row["geography_id"].startswith("S")]
    assert len(scottish) == 32
    assert {row["status"] for row in scottish} == {"no_fact_for_area"}
    assert {row["signed_rationale"] for row in scottish} == {deferral.rationale}
    overlaps = scotland_brma_council_overlaps()
    split = sum(c > 1 for c in Counter(i.local_authority for i in overlaps).values())
    assert f"{len(overlaps)} intersections, {split} councils in more than one BRMA" in (
        deferral.rationale
    )


def test_the_surface_still_has_no_brma_grain() -> None:
    """Tripwire: the concern says BRMA rents have no grain to bind at.

    If a BRMA grain, area type or bridge is ever added, this fails and the
    concern's reason and fence need re-adjudicating rather than silently
    going stale.
    """

    assert UK_CROSS_GRAIN_GRAIN_PRECEDENCE == (
        "country",
        "region",
        "constituency",
        "la",
    )
    assert AREA_TYPES == ("constituency", "la")
    assert not any(
        "ons.rent." in side
        for bridge in UK_CROSS_GRAIN_BRIDGES
        for side in (*bridge.higher_target_ids, bridge.lower_side)
    )
    target = next(
        row
        for row in load_uk_population_contract()["targets"]
        if row["target_id"] == "ons.rent.private_rent"
    )
    assert target["geography_levels"] == ["local_authority"]
