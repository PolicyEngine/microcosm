"""Tests for building the 117th->119th CD vintage crosswalk from Census sources."""

import pytest

from populace.build.us_runtime import (
    load_congressional_district_vintage_crosswalk,
    load_default_congressional_district_vintage_crosswalk,
    translate_congressional_district_facts_to_current_vintage,
)
from populace.build.us_runtime.congressional_district_vintage import (
    _BOTH_VINTAGE_AT_LARGE_STATE_FIPS,
    default_congressional_district_vintage_crosswalk_path,
    validate_packaged_congressional_district_vintage_crosswalk,
)
from populace.build.us_runtime.congressional_district_vintage_crosswalk import (
    CROSSWALK_BASIS_BLOCK_POPULATION,
    build_cd_vintage_crosswalk_rows,
    normalize_district_code,
    parse_baf_cd_layer,
    parse_national_cd_bef_districts,
)


def test_normalize_district_code_folds_delegate_and_at_large_to_00() -> None:
    assert normalize_district_code("00") == "00"
    assert normalize_district_code("07") == "07"
    assert normalize_district_code("7") == "07"
    assert normalize_district_code("52") == "52"
    assert normalize_district_code("98") == "00"  # DC non-voting delegate
    assert normalize_district_code("ZZ") is None  # water
    assert normalize_district_code("ZZZ") is None
    assert normalize_district_code("") is None


def test_normalize_district_code_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="expected a two-digit code"):
        normalize_district_code("XY")


def test_parse_baf_cd_layer_reads_blockid_district_and_drops_water() -> None:
    result = parse_baf_cd_layer(
        [
            "BLOCKID|DISTRICT\n",
            "080010078011000|06\n",
            "080010078011001|07\n",
            "080010078011002|ZZ\n",  # water: dropped
            "300010001001308|00\n",  # at-large
            "110010001011000|98\n",  # DC delegate -> 00
        ],
        label="BlockAssign_ST08_CO_CD.txt",
    )
    assert result == {
        80010078011000: "06",
        80010078011001: "07",
        300010001001308: "00",
        110010001011000: "00",
    }


def test_parse_baf_cd_layer_rejects_bad_header() -> None:
    with pytest.raises(ValueError, match="header must be 'BLOCKID\\|DISTRICT'"):
        parse_baf_cd_layer(["GEOID,CDFP\n", "080010078011000|06\n"], label="x")


def test_parse_baf_cd_layer_rejects_duplicate_block() -> None:
    with pytest.raises(ValueError, match="more than once"):
        parse_baf_cd_layer(
            [
                "BLOCKID|DISTRICT\n",
                "080010078011000|06\n",
                "080010078011000|07\n",
            ],
            label="x",
        )


def test_parse_national_cd_bef_districts_keeps_two_char_code() -> None:
    result = parse_national_cd_bef_districts(
        [
            "GEOID,CDFP\n",
            "080010078011000,08\n",
            "300010001001308,01\n",
            "110010001011000,98\n",  # delegate -> 00
            "360610001001001,ZZ\n",  # water dropped
        ]
    )
    assert result == {
        80010078011000: "08",
        300010001001308: "01",
        110010001011000: "00",
    }


def test_build_crosswalk_redistributes_by_population_and_conserves() -> None:
    # Two 2020 blocks whose OLD district is CO-07; one stays in new CO-07, the
    # other moves to the newly-carved CO-08. Weights are the block populations,
    # so old CO-07's population is redistributed, never invented.
    old_cd_by_block = {
        80010000000001: "07",
        80010000000002: "07",
        80010000000003: "01",
    }
    current_cd_by_block = {
        80010000000001: "07",
        80010000000002: "08",
        80010000000003: "01",
    }
    block_population = {
        80010000000001: 600,
        80010000000002: 400,
        80010000000003: 1000,
    }

    rows, diagnostics = build_cd_vintage_crosswalk_rows(
        old_cd_by_block=old_cd_by_block,
        current_cd_by_block=current_cd_by_block,
        block_population=block_population,
    )

    by_pair = {
        (row["source_geography_id"], row["target_geography_id"]): row for row in rows
    }
    assert {pair: row["pair_population"] for pair, row in by_pair.items()} == {
        ("5001700US0807", "5001900US0807"): 600,
        ("5001700US0807", "5001900US0808"): 400,
        ("5001700US0801", "5001900US0801"): 1000,
    }
    # Weights are each pair's share of its old district's population.
    assert by_pair[("5001700US0807", "5001900US0807")]["weight"] == 0.6
    assert by_pair[("5001700US0807", "5001900US0808")]["weight"] == 0.4
    assert by_pair[("5001700US0801", "5001900US0801")]["weight"] == 1.0
    # Conservation: every populated block is assigned, so the old-district
    # population masses sum back to the total population.
    assert sum(row["pair_population"] for row in rows) == 2000
    assert diagnostics["basis"] == CROSSWALK_BASIS_BLOCK_POPULATION
    assert diagnostics["state_conservation"]["08"] == {
        "state_population": 2000,
        "assigned_population": 2000,
        "unmatched_population": 0,
    }


def test_build_crosswalk_reports_uncovered_population_without_inventing() -> None:
    # A populated block with no current-district assignment is reported as
    # unmatched, not mapped anywhere.
    rows, diagnostics = build_cd_vintage_crosswalk_rows(
        old_cd_by_block={300010000000001: "00", 300010000000002: "00"},
        current_cd_by_block={300010000000001: "01"},  # block 2 uncovered
        block_population={300010000000001: 500, 300010000000002: 250},
    )
    assert [row["pair_population"] for row in rows] == [500]
    assert [row["weight"] for row in rows] == [1.0]
    assert diagnostics["unmatched_population_no_current_district"] == 250
    assert diagnostics["state_conservation"]["30"]["unmatched_population"] == 250


def test_build_crosswalk_splits_at_large_state_by_population() -> None:
    # Montana was at-large (117th district 00) and split into two 119th
    # districts. The split weight is the population share, and (because
    # districts are equal-population) it comes out near 50/50.
    old_cd_by_block = {300010000000001: "00", 300010000000002: "00"}
    current_cd_by_block = {300010000000001: "01", 300010000000002: "02"}
    block_population = {300010000000001: 542112, 300010000000002: 542113}

    rows, _ = build_cd_vintage_crosswalk_rows(
        old_cd_by_block=old_cd_by_block,
        current_cd_by_block=current_cd_by_block,
        block_population=block_population,
    )
    by_target = {row["target_geography_id"]: row for row in rows}
    assert by_target["5001900US3001"]["pair_population"] == 542112
    assert by_target["5001900US3002"]["pair_population"] == 542113
    assert by_target["5001900US3001"]["weight"] == pytest.approx(0.5, abs=1e-6)
    assert by_target["5001900US3002"]["weight"] == pytest.approx(0.5, abs=1e-6)
    assert sum(row["weight"] for row in rows) == pytest.approx(1.0, abs=1e-12)
    assert all(row["source_geography_id"] == "5001700US3000" for row in rows)


def test_built_crosswalk_feeds_the_vintage_translator_and_conserves(tmp_path) -> None:
    # End-to-end: the rows this builder emits load through the crosswalk loader
    # and drive the fiscal translator, conserving the source fact's value.
    rows, _ = build_cd_vintage_crosswalk_rows(
        old_cd_by_block={
            80010000000001: "07",
            80010000000002: "07",
        },
        current_cd_by_block={
            80010000000001: "07",
            80010000000002: "08",
        },
        block_population={80010000000001: 600, 80010000000002: 400},
    )
    path = tmp_path / "cd_xwalk.csv"
    import csv as _csv

    with path.open("w", newline="") as stream:
        writer = _csv.DictWriter(
            stream,
            fieldnames=[
                "source_geography_id",
                "target_geography_id",
                "pair_population",
                "weight",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    crosswalk = load_congressional_district_vintage_crosswalk(path)
    assert crosswalk["share"].tolist() == pytest.approx([0.6, 0.4])

    fact = {
        "lineage": {"source_record_id": "irs_soi.ty2023.co_07.adjusted_gross_income"},
        "value": 1000.0,
        "period": {"type": "tax_year", "value": 2023},
        "geography": {
            "level": "congressional_district",
            "id": "5001700US0807",
            "vintage": "117th_congress",
        },
        "entity": {"name": "tax_unit"},
        "aggregation": {"method": "sum"},
        "dimensions": {"income_range": "all", "filing_status": "all"},
        "layout": {
            "record_set_id": "irs_soi.ty2023.congressional_district_2022.all_returns",
            "groupby_dimension": "irs_soi.congressional_district",
            "groupby_value_id": "co_07",
            "measure_id": "adjusted_gross_income",
            "source_row_id": "co_07",
        },
        "observed_measure": {
            "source_name": "irs_soi",
            "source_measure_id": "adjusted_gross_income",
            "source_concept": "adjusted_gross_income",
            "unit": "usd",
        },
        "source": {"source_name": "irs_soi", "vintage": "tax_year_2023", "url": "x"},
    }
    translated = translate_congressional_district_facts_to_current_vintage(
        [fact], crosswalk
    )
    by_geo = {
        row["geography"]["id"]: row["value"]
        for row in translated
        if row["geography"]["level"] == "congressional_district"
    }
    assert by_geo["5001900US0807"] == pytest.approx(600.0)
    assert by_geo["5001900US0808"] == pytest.approx(400.0)
    # Redistribution conserves the source fact's value.
    assert sum(by_geo.values()) == pytest.approx(1000.0)


# --- Integration: the packaged Census-built default crosswalk ---------------

# The issue's shrinking-state districts (removed by 2020 apportionment) must
# appear only as sources, and the growing-state districts only as targets.
_OLD_ONLY_SOURCE_GEOIDS = (
    "5001700US0653",  # CA-53
    "5001700US1718",  # IL-18
    "5001700US2614",  # MI-14
    "5001700US3627",  # NY-27
    "5001700US3916",  # OH-16
    "5001700US4218",  # PA-18
    "5001700US5403",  # WV-03
)
_NEW_ONLY_TARGET_GEOIDS = (
    "5001900US0808",  # CO-08
    "5001900US1228",  # FL-28
    "5001900US3002",  # MT-02
    "5001900US3714",  # NC-14
    "5001900US4106",  # OR-06
    "5001900US4837",  # TX-37
    "5001900US4838",  # TX-38
)


def test_default_crosswalk_is_packaged_and_covers_all_436_current_districts() -> None:
    assert default_congressional_district_vintage_crosswalk_path().exists()
    crosswalk = load_default_congressional_district_vintage_crosswalk()
    assert crosswalk["target_geography_id"].nunique() == 436
    assert crosswalk["source_geography_id"].nunique() == 436
    # Shares of every source district sum to 1 (the loader normalizes weights).
    per_source = crosswalk.groupby("source_geography_id")["share"].sum()
    assert per_source.min() == pytest.approx(1.0)
    assert per_source.max() == pytest.approx(1.0)


def test_default_crosswalk_moves_apportionment_changed_districts_correctly() -> None:
    crosswalk = load_default_congressional_district_vintage_crosswalk()
    sources = set(crosswalk["source_geography_id"])
    targets = set(crosswalk["target_geography_id"])
    # Removed 117th districts are sources, never current targets.
    for geoid in _OLD_ONLY_SOURCE_GEOIDS:
        assert geoid in sources, geoid
        assert geoid.replace("5001700US", "5001900US") not in targets, geoid
    # New 119th districts are populated targets.
    for geoid in _NEW_ONLY_TARGET_GEOIDS:
        assert geoid in targets, geoid


def test_default_crosswalk_splits_montana_at_large_near_evenly() -> None:
    # Equal-population redistricting: at-large MT (117th) splits ~50/50.
    crosswalk = load_default_congressional_district_vintage_crosswalk()
    mt = crosswalk[crosswalk["source_geography_id"] == "5001700US3000"]
    shares = dict(zip(mt["target_geography_id"], mt["share"], strict=True))
    assert set(shares) == {"5001900US3001", "5001900US3002"}
    assert shares["5001900US3001"] == pytest.approx(0.5, abs=0.02)
    assert shares["5001900US3002"] == pytest.approx(0.5, abs=0.02)


def test_default_crosswalk_translates_a_realistic_soi_surface_and_conserves() -> None:
    # One real district (CA-53 -> current CA), one shrinking district split,
    # plus an at-large state fact routed through the state-total proxy. The
    # translated current-CD totals must equal the source totals exactly.
    facts = [
        _soi_cd_fact("adjusted_gross_income", 300.0, "5001700US0601", "ca_01"),
        _soi_cd_fact("adjusted_gross_income", 120.0, "5001700US0653", "ca_53"),
        _soi_state_fact("adjusted_gross_income", 90.0, "0400000US30", "mt_total"),
    ]
    crosswalk = load_default_congressional_district_vintage_crosswalk()

    translated = translate_congressional_district_facts_to_current_vintage(
        facts, crosswalk
    )
    current = [
        row
        for row in translated
        if row["geography"]["level"] == "congressional_district"
        and str(row["geography"]["id"]).startswith("5001900US")
    ]
    # No stale old-vintage CD geoid survives.
    assert all(row["geography"]["id"].startswith("5001900US") for row in current)
    assert "5001900US0653" not in {row["geography"]["id"] for row in current}
    # Montana's state total is redistributed to its two current districts.
    mt = {
        row["geography"]["id"]: row["value"]
        for row in current
        if str(row["geography"]["id"]).startswith("5001900US30")
    }
    assert set(mt) == {"5001900US3001", "5001900US3002"}
    assert sum(mt.values()) == pytest.approx(90.0)
    # California AGI (both source districts) is conserved across current CA.
    ca_total = sum(
        row["value"]
        for row in current
        if str(row["geography"]["id"]).startswith("5001900US06")
    )
    assert ca_total == pytest.approx(420.0)


def _soi_cd_fact(measure_id, value, geography_id, source_row_id):
    record_set_id = "irs_soi.ty2023.congressional_district_2022.all_returns"
    return {
        "lineage": {
            "source_record_id": f"{record_set_id}.{source_row_id}.{measure_id}"
        },
        "value": value,
        "period": {"type": "tax_year", "value": 2023},
        "geography": {
            "level": "congressional_district",
            "id": geography_id,
            "vintage": "117th_congress",
        },
        "entity": {"name": "tax_unit"},
        "aggregation": {"method": "sum"},
        "dimensions": {"income_range": "all", "filing_status": "all"},
        "layout": {
            "record_set_id": record_set_id,
            "groupby_dimension": "irs_soi.congressional_district",
            "groupby_value_id": source_row_id,
            "measure_id": measure_id,
            "source_row_id": source_row_id,
        },
        "observed_measure": {
            "source_name": "irs_soi",
            "source_measure_id": measure_id,
            "source_concept": measure_id,
            "unit": "usd",
        },
        "source": {"source_name": "irs_soi", "vintage": "tax_year_2023", "url": "x"},
    }


def _soi_state_fact(measure_id, value, geography_id, source_row_id):
    return {
        "lineage": {
            "source_record_id": f"irs_soi.ty2023.state_agi.{source_row_id}.{measure_id}"
        },
        "value": value,
        "period": {"type": "tax_year", "value": 2023},
        "geography": {"level": "state", "id": geography_id, "vintage": "2020_census"},
        "entity": {"name": "tax_unit"},
        "aggregation": {"method": "sum"},
        "dimensions": {"income_range": "all", "filing_status": "all"},
        "layout": {
            "record_set_id": "irs_soi.ty2023.congressional_district_2022.all_returns",
            "groupby_dimension": "irs_soi.state",
            "groupby_value_id": source_row_id,
            "measure_id": measure_id,
            "source_row_id": source_row_id,
        },
        "observed_measure": {
            "source_name": "irs_soi",
            "source_measure_id": measure_id,
            "source_concept": measure_id,
            "unit": "usd",
        },
        "source": {"source_name": "irs_soi", "vintage": "tax_year_2023", "url": "x"},
    }


def test_packaged_crosswalk_weights_are_normalized_shares_of_population() -> None:
    # The committed artifact's contract (populace#288 review): `weight` is the
    # pair's share of its source district's 2020 population — summing to 1 per
    # source in the RAW CSV, not merely after loader normalization — with the
    # population masses preserved alongside as `pair_population`.
    crosswalk = load_default_congressional_district_vintage_crosswalk()
    assert "pair_population" in crosswalk.columns
    assert (crosswalk["pair_population"] > 0).all()
    assert (crosswalk["weight"] > 0).all()
    assert (crosswalk["weight"] <= 1.0).all()
    per_source = crosswalk.groupby("source_geography_id")["weight"].sum()
    assert (per_source - 1.0).abs().max() < 1e-9
    population_share = crosswalk["pair_population"] / crosswalk.groupby(
        "source_geography_id"
    )["pair_population"].transform("sum")
    assert (crosswalk["weight"] - population_share).abs().max() < 1e-12


def test_packaged_crosswalk_keeps_both_vintage_at_large_identity_rows() -> None:
    crosswalk = load_default_congressional_district_vintage_crosswalk()
    for state_fips in sorted(_BOTH_VINTAGE_AT_LARGE_STATE_FIPS):
        rows = crosswalk[crosswalk["source_geography_id"] == f"5001700US{state_fips}00"]
        assert len(rows) == 1, state_fips
        assert rows.iloc[0]["target_geography_id"] == f"5001900US{state_fips}00"
        assert rows.iloc[0]["weight"] == 1.0


def test_loader_rejects_duplicate_source_target_pairs(tmp_path) -> None:
    path = tmp_path / "crosswalk.csv"
    path.write_text(
        "source_geography_id,target_geography_id,weight\n"
        "5001700US3000,5001900US3001,0.5\n"
        "5001700US3000,5001900US3001,0.5\n"
    )
    with pytest.raises(ValueError, match="duplicate source/target"):
        load_congressional_district_vintage_crosswalk(path)


def test_loader_rejects_weight_inconsistent_with_pair_population(tmp_path) -> None:
    path = tmp_path / "crosswalk.csv"
    path.write_text(
        "source_geography_id,target_geography_id,pair_population,weight\n"
        "5001700US3000,5001900US3001,600,0.5\n"
        "5001700US3000,5001900US3002,400,0.5\n"
    )
    with pytest.raises(ValueError, match="inconsistent with"):
        load_congressional_district_vintage_crosswalk(path)


def test_packaged_validator_requires_population_provenance_and_full_coverage() -> None:
    import pandas as pd

    without_population = pd.DataFrame(
        {
            "source_geography_id": ["5001700US3000"],
            "target_geography_id": ["5001900US3001"],
            "weight": [1.0],
            "share": [1.0],
        }
    )
    with pytest.raises(ValueError, match="pair_population"):
        validate_packaged_congressional_district_vintage_crosswalk(without_population)
    undersized = without_population.assign(pair_population=[600.0])
    with pytest.raises(ValueError, match="exactly 436 source districts"):
        validate_packaged_congressional_district_vintage_crosswalk(undersized)
