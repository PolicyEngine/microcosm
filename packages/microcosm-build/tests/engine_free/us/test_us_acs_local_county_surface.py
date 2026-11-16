"""Engine-free county targets, held-out scoring and reliability evidence."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import math
import shlex
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from microcosm.build.us_runtime.chronicle_feed import (
    load_us_chronicle_county_facts_pin,
)
from microcosm.calibrate import TargetSpec
from test_support.paths import paths_for


@lru_cache
def _release():
    path = (
        paths_for("microcosm-build").repository
        / "tools"
        / "build_us_acs_local_release.py"
    )
    spec = importlib.util.spec_from_file_location("acs_local_county_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _county():
    return _release().county_surface


@lru_cache
def _production():
    """The production county tests, for their ACS fact builders."""
    path = (
        paths_for("microcosm-build").tests
        / "engine_free"
        / "us"
        / "test_us_acs_local_county_production.py"
    )
    spec = importlib.util.spec_from_file_location("county_production_rows", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _state_spec(measure="return_count", value=100.0, *, state="01", **metadata):
    variable = "count" if measure == "return_count" else measure
    name = f"state.{state}.{measure}"
    return TargetSpec(
        name=name,
        entity="household",
        measure=name,
        value=value,
        period=2024,
        source="fixture state SOI",
        family="irs_soi",
        signed=value < 0,
        metadata={
            "ledger_geography_level": "state",
            "ledger_layout_record_set_spec_id": "irs_soi.historic_table_2.state_broad_totals.v1",
            "state_fips": state,
            "source_measure_id": measure,
            "source_period": "2022",
            "ledger_measure_unit": "count" if measure == "return_count" else "usd",
            "variable": variable,
            "source_variable": variable,
            "soi_variable": variable,
            "measure_mode": "sum",
            "agi_lower_bound": "-inf",
            "agi_upper_bound": "inf",
            "filing_status": "All",
            **metadata,
        },
    )


def _fact(county, measure, value):
    return {
        "geography": {
            "id": f"0500000US{county}",
            "level": "county",
            "name": f"County {county}",
            "vintage": "2022",
        },
        "period": {"type": "tax_year", "value": 2022},
        "observed_measure": {
            "source_measure_id": measure,
            "source_concept": (
                "irs_soi.individual_income_tax_returns"
                if measure == "return_count"
                else "irs_soi.adjusted_gross_income"
            ),
            "source_name": "irs_soi",
            "unit": "count" if measure == "return_count" else "usd",
        },
        "layout": {
            "record_set_id": "irs_soi.ty2022.county_totals",
            "record_set_spec_id": "irs_soi.county_totals.v1",
            "measure_id": measure,
        },
        "dimensions": {"filing_status": "all", "income_range": "all"},
        "value": value,
    }


def test_ladder_county_population_is_scaled_to_each_state() -> None:
    county = _county()
    ladder = SimpleNamespace(
        puma=np.asarray([100001, 100002, 600001]),
        puma_population=np.asarray([60.0, 40.0, 80.0]),
        county_overlap_county=np.asarray([1001, 1001, 1003, 6001]),
        county_overlap_population=np.asarray([15.0, 15.0, 20.0, 40.0]),
    )
    source = {"state": {1: 120.0, 6: 80.0}}
    populations = county.with_county_populations(ladder, source, ["state", "county"])
    assert populations["county"] == {1001: 72.0, 1003: 48.0, 6001: 80.0}
    assert populations["state"] == source["state"]
    assert "county" not in source
    # County-only runs still use PUMA state totals for their normalization.
    only_counties = county.with_county_populations(ladder, {}, ["county"])
    assert sum(v for k, v in only_counties["county"].items() if k // 1000 == 1) == 100.0


def test_population_rows_partition_households_and_accept_an_alternate_county_column() -> (
    None
):
    households = pd.DataFrame(
        {
            "county_fips": [99999, 99999, 99999],
            "county_fips_2023": ["01001", "01003", "01001"],
            "state_fips": [1, 1, 1],
        }
    )
    records, matrix, dropped = _county().population_rows(
        households,
        np.asarray([2.0, 3.0, 1.0]),
        {"county": {1001: 80.0, 1003: 20.0, 1005: 10.0}},
        ["county"],
        county_column="county_fips_2023",
    )
    assert [r["name"] for r in records] == ["pop_county_01001", "pop_county_01003"]
    assert dropped == ["pop_county_01005"]
    np.testing.assert_array_equal(matrix.toarray(), [[2, 0, 1], [0, 3, 0]])
    assert matrix.dtype == np.float32
    assert records[0]["county_fips"] == "01001"
    assert records[0]["state_fips"] == "01"
    with pytest.raises((ValueError, SystemExit), match="01005"):
        _county().require_support(dropped)
    _county().require_support(dropped, capped_smoke=True)


@pytest.mark.parametrize(
    ("allow_partial", "sample_fraction", "refused"),
    [(False, 0.01, True), (True, 1.0, True), (True, 0.01, False)],
)
def test_support_check_writes_every_county_and_requires_a_capped_smoke(
    tmp_path, allow_partial, sample_fraction, refused
) -> None:
    households = pd.DataFrame({"county_fips": [1001], "state_fips": [1]})
    frame = SimpleNamespace(table=lambda _entity: households)
    args = SimpleNamespace(
        checkpoint_dir=tmp_path,
        county_column="county_fips",
        allow_partial_geography=allow_partial,
        sample_fraction=sample_fraction,
    )
    if refused:
        with pytest.raises(SystemExit, match="01003"):
            _county().check_support(frame, {"county": {1001: 10.0, 1003: 20.0}}, args)
    else:
        _county().check_support(frame, {"county": {1001: 10.0, 1003: 20.0}}, args)
    report = json.loads((tmp_path / "county_support.json").read_text())
    assert report["counties"] == {
        "01001": {"supporting_households": 1},
        "01003": {"supporting_households": 0},
    }
    assert report["unsupported_counties"] == ["01003"]


def test_county_population_spec_carries_county_hierarchy() -> None:
    spec = _county().population_spec("pop_county_01001", 123.0, period=2026)
    assert spec.value == 123.0
    assert spec.period == 2026
    assert spec.hierarchy.geography.id == "0500000US01001"
    assert spec.hierarchy.geography.level == "county"
    assert spec.metadata["county_fips"] == "01001"
    record = _county().target_record(spec)
    assert record["geography_level"] == "county"
    assert record["county_fips"] == "01001"


def test_county_soi_reconciles_only_existing_broad_state_concepts() -> None:
    parents = [
        _state_spec(),
        _state_spec("adjusted_gross_income", 1200.0),
        _state_spec("wages_salaries_amount", 900.0),
        dataclasses.replace(
            _state_spec(),
            name="state.01.returns_high_agi",
            metadata={**_state_spec().metadata, "agi_lower_bound": "100000"},
        ),
    ]
    facts = [
        _fact(code, measure, value)
        for measure, values in (
            ("return_count", [1.0, 3.0]),
            ("adjusted_gross_income", [3.0, 2.0]),
        )
        for code, value in zip(["01001", "01003"], values, strict=True)
    ]
    surface = _county().state_county_soi_surface(
        parents, facts, {1001: 50.0, 1003: 50.0}
    )
    assert len(surface.specs) == 4
    by_parent = {}
    for spec in surface.specs:
        assert spec.metadata["ledger_geography_level"] == "county"
        assert spec.period == 2024
        by_parent.setdefault(
            spec.metadata["state_county_parent_target_name"], []
        ).append(spec)
        record = _county().target_record(spec)
        assert record["county_fips"] == spec.metadata["county_fips"]
        assert (
            record["state_county_parent_target_name"]
            == spec.metadata["state_county_parent_target_name"]
        )
    assert set(by_parent) == {parent.name for parent in parents[:2]}
    assert [s.value for s in by_parent[parents[0].name]] == pytest.approx([25.0, 75.0])
    assert [s.value for s in by_parent[parents[1].name]] == pytest.approx(
        [720.0, 480.0]
    )


@pytest.mark.parametrize("bad_facts", ["missing", "duplicate", "zero"])
def test_an_incomplete_or_unusable_county_fact_block_is_refused(bad_facts) -> None:
    facts = [_fact("01001", "return_count", 2.0), _fact("01003", "return_count", 3.0)]
    if bad_facts == "missing":
        facts.pop()
    elif bad_facts == "duplicate":
        facts.append(dict(facts[0]))
    else:
        facts = [{**fact, "value": 0.0} for fact in facts]
    with pytest.raises(ValueError):
        _county().state_county_soi_surface(
            [_state_spec(), _state_spec("adjusted_gross_income", 1000.0)],
            facts
            + [
                _fact("01001", "adjusted_gross_income", 2.0),
                _fact("01003", "adjusted_gross_income", 3.0),
            ],
            {1001: 50.0, 1003: 50.0},
        )


@pytest.mark.parametrize("measure", ["return_count", "adjusted_gross_income"])
@pytest.mark.parametrize("omission", ["county_concept", "state_parent"])
@pytest.mark.parametrize("omitted_state", [None, "06"])
def test_state_county_requires_both_concepts_and_parents_in_every_state(
    measure, omission, omitted_state
) -> None:
    parents = [
        _state_spec(concept, state=state)
        for state in ("01", "06")
        for concept in ("return_count", "adjusted_gross_income")
        if not (
            omission == "state_parent"
            and concept == measure
            and omitted_state in (None, state)
        )
    ]
    facts = [
        _fact(code, concept, 1.0)
        for code in ("01001", "01003", "06001")
        for concept in ("return_count", "adjusted_gross_income")
        if not (
            omission == "county_concept"
            and concept == measure
            and omitted_state in (None, code[:2])
        )
    ]
    with pytest.raises(ValueError, match=f"concept={measure}"):
        _county().state_county_soi_surface(
            parents, facts, {1001: 50.0, 1003: 50.0, 6001: 100.0}
        )


def test_kalawao_without_its_declared_host_still_refuses_with_the_missing_code() -> (
    None
):
    # Kalawao's SOI cell is declared combined with Maui (15009); with Maui
    # absent too there is no published cell to carry it.
    with pytest.raises(ValueError, match="15005"):
        _county().state_county_soi_surface(
            [
                _state_spec(measure, state="15")
                for measure in ("return_count", "adjusted_gross_income")
            ],
            [
                _fact("15001", measure, 1.0)
                for measure in ("return_count", "adjusted_gross_income")
            ],
            {15001: 50.0, 15005: 50.0},
        )


@pytest.mark.parametrize(
    "slice_metadata",
    [
        {"filing_status": "Single"},
        {"itemized_only": "true"},
        {"taxable_only": "true"},
        {"soi_return_universe": "taxable_returns"},
        {"ledger_filter_income_range": "100000_plus"},
    ],
)
def test_all_return_county_facts_do_not_become_sliced_state_targets(
    slice_metadata,
) -> None:
    parent = _state_spec()
    sliced = dataclasses.replace(
        parent,
        name="state.sliced.returns",
        metadata={**parent.metadata, **slice_metadata},
    )
    surface = _county().state_county_soi_surface(
        [parent, sliced, _state_spec("adjusted_gross_income", 1000.0)],
        [
            _fact(code, measure, value)
            for measure in ("return_count", "adjusted_gross_income")
            for code, value in (("01001", 1.0), ("01003", 3.0))
        ],
        {1001: 50.0, 1003: 50.0},
    )
    assert len(surface.specs) == 4
    assert {
        spec.metadata["state_county_parent_target_name"]
        for spec in surface.specs
        if spec.metadata["source_measure_id"] == "return_count"
    } == {parent.name}


def test_county_soi_uses_exact_parent_household_values_including_signed_values() -> (
    None
):
    parent = _state_spec("adjusted_gross_income", 600.0)
    surface = _county().state_county_soi_surface(
        [parent, _state_spec()],
        [
            _fact("01001", "adjusted_gross_income", 1.0),
            _fact("01003", "adjusted_gross_income", 2.0),
            _fact("01001", "return_count", 1.0),
            _fact("01003", "return_count", 2.0),
        ],
        {1001: 50.0, 1003: 50.0},
    )
    households = pd.DataFrame(
        {
            "county_fips": [0, 0, 0],
            "county_fips_2023": [1001, 1003, 1001],
            "state_fips": [1, 1, 1],
        }
    )
    state_matrix = sparse.csr_array(np.asarray([[-10.0, 20.0, 30.0]], dtype=np.float32))
    matrix = _county().split_state_rows(
        [parent],
        state_matrix,
        [
            spec
            for spec in surface.specs
            if spec.metadata["state_county_parent_target_name"] == parent.name
        ],
        households,
        county_column="county_fips_2023",
    )
    np.testing.assert_array_equal(
        matrix.toarray(), [[-10.0, 0.0, 30.0], [0.0, 20.0, 0.0]]
    )
    np.testing.assert_array_equal(
        np.asarray(matrix.sum(axis=0)), state_matrix.toarray()[0]
    )
    assert matrix.nnz == state_matrix.nnz
    assert matrix.dtype == state_matrix.dtype


def test_county_holdout_preserves_existing_roles_and_uses_a_population_baseline() -> (
    None
):
    records = [
        {"name": "state_agi", "value": 100.0, "family": "irs_soi", "role": "train"},
        {
            "name": "cd_held",
            "value": 10.0,
            "geography_level": "congressional_district",
            "role": "holdout",
            "holdout_unit": "existing",
        },
        *(
            {
                "name": f"county_{code}",
                "value": value,
                "family": "irs_soi",
                "geography_level": "county",
                "state_fips": "01",
                "county_fips": code,
                "source_measure_id": "adjusted_gross_income",
                "state_county_parent_target_name": "state_agi",
                "county_population": population,
                "state_population": 100.0,
            }
            for code, value, population in (
                ("01001", 70.0, 50.0),
                ("01003", 30.0, 50.0),
            )
        ),
    ]
    _county().assign_target_roles(records, fraction=0.5, salt="fixture")
    assert records[1]["role"] == "holdout"
    assert records[1]["holdout_unit"] == "existing"
    assert records[2]["role"] == records[3]["role"]
    for record in records[2:]:
        record["role"] = "holdout"
    matrix = sparse.csr_array(
        np.asarray([[1, 1], [0, 0], [1, 0], [0, 1]], dtype=np.float32)
    )
    report = _county().score_county_holdout(
        records,
        matrix,
        design_weights=np.asarray([50.0, 50.0]),
        final_weights=np.asarray([70.0, 30.0]),
        cap=1.0,
    )
    assert report["n_targets"] == 2
    assert report["calibrated"]["mean_abs_rel_error"] == pytest.approx(0.0)
    assert report["pro_rata_baseline"]["mean_abs_rel_error"] == pytest.approx(
        (20 / 70 + 20 / 30) / 2
    )
    assert [target["pro_rata"] for target in report["targets"]] == [50.0, 50.0]
    assert report["calibrated_beats_pro_rata_share"] == 1.0


@pytest.mark.parametrize(
    ("ess", "error", "grade"),
    [
        (400.0, 0.10, "A"),
        (399.0, 0.10, "B"),
        (100.0, 0.20, "B"),
        (99.0, 0.20, "C"),
        (30.0, 0.50, "C"),
        (29.0, 0.01, "D"),
        (500.0, 0.51, "D"),
        (500.0, None, "U"),
    ],
)
def test_reliability_grade_thresholds(ess, error, grade) -> None:
    assert _county().reliability_grade(ess, error) == grade


def test_reliability_report_counts_source_households_and_absent_counties() -> None:
    households = pd.DataFrame(
        {
            "county_fips": [1001, 1001, 1001, 1003],
            "state_fips": [1, 1, 1, 1],
            "household_spine": ["asec_puf", "asec_puf", "acs_2024_1yr", "acs_2024_1yr"],
            "household_source_id": [7, 7, 7, 9],
        }
    )
    records = [
        {
            "name": "train_01001",
            "value": 4.0,
            "role": "train",
            "geography_level": "county",
            "county_fips": "01001",
        },
        {
            "name": "held_01001",
            "value": 8.0,
            "role": "holdout",
            "geography_level": "county",
            "county_fips": "01001",
        },
    ]
    matrix = sparse.csr_array(
        np.asarray([[1, 1, 1, 0], [2, 2, 2, 0]], dtype=np.float32)
    )
    report = _county().reliability_report(
        households,
        records,
        matrix,
        final_weights=np.asarray([1.0, 1.0, 2.0, 4.0]),
        county_populations={1001: 80.0, 1003: 20.0, 1005: 10.0},
    )
    cells = report["counties"]
    assert set(cells) == {"01001", "01003", "01005"}
    assert cells["01001"]["supporting_households"] == 3
    assert cells["01001"]["distinct_source_households"] == 2
    assert cells["01001"]["kish_ess"] == pytest.approx(2.0)
    assert cells["01001"]["kish_ess_rows"] == pytest.approx(16 / 6)
    assert cells["01001"]["training"]["mean_abs_rel_error"] == pytest.approx(0.0)
    assert cells["01001"]["holdout"]["mean_abs_rel_error"] == pytest.approx(0.0)
    assert cells["01001"]["grade"] == "D"
    assert cells["01003"]["grade"] == "U"
    assert cells["01005"]["supporting_households"] == 0
    assert cells["01005"]["distinct_source_households"] == 0
    assert cells["01005"]["kish_ess"] == 0.0
    assert cells["01005"]["kish_ess_rows"] == 0.0
    assert report["schema_version"] == 3
    assert report["grade_rule"]["provisional"] is True
    assert cells["01001"]["status"] == "supported"
    assert cells["01005"]["status"] == "no_support"
    assert "household_source_id" in report["ess_basis"]["kish_ess"]


def test_two_clones_of_200_donors_do_not_raise_the_source_ess_or_grade() -> None:
    households = pd.DataFrame(
        {
            "county_fips": [1001] * 400,
            "state_fips": [1] * 400,
            "household_spine": ["asec_puf"] * 400,
            "household_source_id": np.repeat(np.arange(200), 2),
        }
    )
    report = _county().reliability_report(
        households,
        [
            {
                "name": "held_01001",
                "value": 400.0,
                "role": "holdout",
                "geography_level": "county",
                "county_fips": "01001",
            }
        ],
        sparse.csr_array(np.ones((1, 400))),
        final_weights=np.ones(400),
    )
    cell = report["counties"]["01001"]
    assert cell["distinct_source_households"] == 200
    assert cell["kish_ess"] == 200.0
    assert cell["kish_ess_rows"] == 400.0
    assert cell["grade"] == "B"


@pytest.mark.parametrize(
    "missing_identity", ["household_spine", "household_source_id", "null"]
)
def test_missing_source_identity_cannot_produce_a_row_based_reliability_grade(
    missing_identity,
) -> None:
    households = pd.DataFrame(
        {
            "county_fips": [1001],
            "state_fips": [1],
            "household_spine": ["asec_puf"],
            "household_source_id": [7],
        }
    )
    if missing_identity == "null":
        households["household_source_id"] = None
    else:
        households = households.drop(columns=missing_identity)
    report = _county().reliability_report(
        households,
        [
            {
                "name": "held_01001",
                "value": 1.0,
                "role": "holdout",
                "geography_level": "county",
                "county_fips": "01001",
            }
        ],
        sparse.csr_array([[1.0]]),
        final_weights=np.ones(1),
    )
    cell = report["counties"]["01001"]
    assert cell["distinct_source_households"] is None
    assert cell["kish_ess"] is None
    assert cell["kish_ess_rows"] == 1.0
    assert cell["grade"] == "U"


_RELIABILITY_KEYS = ("supporting_households", "kish_ess", "grade", "status")


@pytest.mark.skipif(
    importlib.util.find_spec("tables") is None, reason="requires pytables"
)
@pytest.mark.parametrize("real_households", [40, 0])
def test_group_quarters_stay_out_of_reliability_after_a_checkpoint_reload(
    tmp_path, real_households
) -> None:
    """A county of 40 households and 360 group-quarters records grades C on
    the full frame. Its lean checkpoint once dropped TYPEHUGQ, so the reload
    counted all 400 records and graded A; a county of group quarters alone
    went from U to A (review of #1630). A county run now keeps TYPEHUGQ and
    grades the same after the reload; a run without county targets writes the
    columns it always did, and a checkpoint without TYPEHUGQ is refused."""
    module, county = _release(), _county()
    from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

    n = 400
    ids = np.arange(1, n + 1)
    households = pd.DataFrame(
        {
            "household_id": ids,
            "county_fips": np.full(n, 1001),
            "state_fips": np.ones(n, dtype=int),
            "household_spine": [county.ACS_SPINE] * n,
            "household_source_id": ids,
            "TYPEHUGQ": [1.0] * real_households + [3.0] * (n - real_households),
        }
    )
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": ids,
                "person_household_id": ids,
                **{f"person_{group}_id": ids for group in US_SCHEMA.group_entities},
            }
        ),
        **{
            group: pd.DataFrame({f"{group}_id": ids})
            for group in US_SCHEMA.group_entities
        },
        # After the group tables, which include an ids-only household table.
        "household": households,
    }
    frame = Frame(
        tables, US_SCHEMA, {"household": Weights(np.ones(n), WeightKind.DESIGN)}
    )
    assert list(frame.table("household").columns) == list(households.columns)
    matrix = sparse.csr_array(np.ones((1, n), dtype=np.float32))
    specs = [county.population_spec("pop_county_01001", 400.0)]
    records = [
        {
            "name": "pop_county_01001",
            "value": 400.0,
            "role": "train",
            "geography_level": "county",
            "county_fips": "01001",
        }
    ]
    validation = {
        "counties": {
            "01001": {
                name: {"abs_rel_error": 0.0}
                for name in county.GRADED_VALIDATION_FAMILIES
            }
        }
    }

    def graded(table):
        cell = county.reliability_report(
            table,
            records,
            matrix,
            final_weights=np.ones(n),
            validation=validation,
            household_universe_mask=county.reliability_universe(table),
        )["counties"]["01001"]
        return {key: cell[key] for key in _RELIABILITY_KEYS}

    full = graded(households)
    assert full["supporting_households"] == real_households
    assert full["grade"] == ("C" if real_households else "U")

    plain = module.extract_struct_tables(frame)
    assert list(plain["household_struct"].columns) == [
        "household_id",
        "state_fips",
        "county_fips",
        "household_spine",
        "household_source_id",
    ]
    struct = module.extract_struct_tables(frame, county_run=True)
    assert "TYPEHUGQ" in struct["household_struct"].columns
    module.write_lean_checkpoint(struct, matrix, specs, records, tmp_path / "county")
    lean, weights = module.load_lean_frame(tmp_path / "county" / "target_frame_lean.h5")
    assert graded(lean.table("household")) == full

    module.write_lean_checkpoint(plain, matrix, specs, records, tmp_path / "old")
    old, _ = module.load_lean_frame(tmp_path / "old" / "target_frame_lean.h5")
    with pytest.raises(ValueError, match="TYPEHUGQ"):
        county.reliability_universe(old.table("household"))
    with pytest.raises(SystemExit, match="materialize again"):
        county.calibration_evidence(old, records, matrix, weights, np.ones(n), 1.0)


@pytest.mark.skipif(
    importlib.util.find_spec("tables") is None, reason="requires pytables"
)
def _materialize_county_fixture(
    monkeypatch,
    tmp_path,
    *,
    acs_values=None,
    households=None,
    county_column="planning_region",
    overlaps=None,
):
    """Materialize a state_county fixture, with a fake engine.

    By default three households and no county support move. ``households``
    replaces the household table (one person each, design weight 10), and
    ``overlaps`` stands in for the ladder's county-PUMA overlaps that minimum
    support reads.
    """
    from microcosm.build import ledger_artifact
    from microcosm.build.us_runtime import county_facts as county_facts_module
    from microcosm.calibrate import TargetRegistry
    from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

    module = _release()
    county = _county()
    if households is None:
        ids = np.arange(1, 4)
        incomes = {
            "employment_income_before_lsr": [30_000.0, 50_000.0, 0.0],
            "self_employment_income_before_lsr": [0.0, 5_000.0, 0.0],
        }
        household_table = pd.DataFrame(
            {
                "household_id": ids,
                "state_fips": [1, 1, 1],
                "county_fips": [99999, 99999, 99999],
                "planning_region": [1001, 1003, 1001],
                "household_spine": ["asec_puf"] * 3,
                "household_source_id": [7, 8, 7],
            }
        )
        design = np.asarray([10.0, 20.0, 30.0])
    else:
        ids = households["household_id"].to_numpy()
        incomes = {
            "employment_income_before_lsr": 10_000.0 * (np.arange(len(ids)) + 1),
            "self_employment_income_before_lsr": np.zeros(len(ids)),
        }
        household_table = households
        design = np.full(len(ids), 10.0)
    zeros = np.zeros(len(ids))
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            **{f"person_{group}_id": ids for group in US_SCHEMA.group_entities},
            # The held-out ACS validation families read household earnings.
            **incomes,
            "sstb_self_employment_income_before_lsr": zeros,
            "farm_operations_income": zeros,
        }
    )
    tables = {"person": person}
    for group in US_SCHEMA.group_entities:
        tables[group] = pd.DataFrame({f"{group}_id": ids})
    tables["household"] = household_table
    frame = Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(design, WeightKind.DESIGN)},
    )
    if overlaps is not None:
        monkeypatch.setattr(county, "county_puma_overlaps", lambda _path: overlaps)
    staging, feed, ladder, county_facts_file = (
        tmp_path / name
        for name in ("staging.h5", "feed.json", "ladder.npz", "county_facts.jsonl")
    )
    for path in (staging, feed, ladder, county_facts_file):
        path.write_bytes(b"fixture")
    staging.with_suffix(".summary.json").write_text("{}")
    county_pin = load_us_chronicle_county_facts_pin()
    args = module._parse_args(
        [
            "--stage",
            "materialize",
            "--staging-h5",
            str(staging),
            "--feed",
            str(feed),
            "--feed-sha256",
            hashlib.sha256(b"fixture").hexdigest(),
            "--county-facts",
            str(county_facts_file),
            "--county-facts-sha256",
            county_pin.facts_sha256,
            "--ladder",
            str(ladder),
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--families",
            "soi",
            "--soi-mode",
            "state_county",
            "--geographies",
            "state,county",
            "--county-column",
            county_column,
            "--county-min-support",
            "3",
            "--sample-fraction",
            "0.01",
            "--county-holdout-fraction",
            "0.5",
        ]
    )
    parents = [_state_spec(), _state_spec("adjusted_gross_income", 1000.0)]
    surface = SimpleNamespace(
        registry=TargetRegistry(parents, country="us"),
        ri_substitutions=[],
        soi_receipt={},
    )
    populations = {
        "state": {1: 200.0},
        "county": {1001: 80.0, 1003: 120.0},
        "cd": {101: 200.0},
    }
    monkeypatch.setattr(module, "state_admin_surface", lambda *_a, **_k: surface)
    monkeypatch.setattr(module, "ladder_population", lambda *_a, **_k: populations)
    monkeypatch.setattr(module, "_load_staging_frame", lambda *_a: frame)
    monkeypatch.setattr(module, "_require_local_hours", lambda *_a: None)
    monkeypatch.setattr(
        module, "_require_acs_self_employment_identity", lambda *_a: None
    )
    # microcosm#1019: the ACS take-up stage needs SPM-unit origin tags this
    # geography fixture does not carry; this test checks county hooks only.
    monkeypatch.setattr(
        module,
        "_with_local_take_up",
        lambda frame, *, seed: (frame, {"seed": seed, "assigned_sha256": "stub"}),
    )
    # microcosm#1020/#1021/#1022: this fixture's staging summary records no
    # immigration, work/disability, income-transfer, SSI disability-criteria or
    # SNAP receipt-anchor stage; this test checks county hooks only.
    monkeypatch.setattr(module, "_require_local_immigration", lambda *_a: None)
    monkeypatch.setattr(module, "_require_local_work_disability", lambda *_a: None)
    monkeypatch.setattr(module, "_require_local_income_transfer", lambda *_a: None)
    monkeypatch.setattr(module, "_require_local_ssi_disability", lambda *_a: None)
    monkeypatch.setattr(module, "_require_local_receipt_anchors", lambda *_a: None)
    # microcosm#1022: the ACS SSI/Medicaid take-up stage compiles SSA and CMS
    # counts from the ledger feed and runs an engine pre-pass this fixture
    # cannot support; this test checks county hooks only.
    monkeypatch.setattr(
        module, "acs_local_ssi_medicaid_take_up_targets", lambda *_a: {}
    )
    monkeypatch.setattr(
        module,
        "_with_local_ssi_medicaid_take_up",
        lambda frame, *, seed, **_k: (frame, {"seed": seed, "assigned_sha256": "stub"}),
    )
    monkeypatch.setattr(module, "_write_ssi_medicaid_take_up", lambda *_a: "stub")
    monkeypatch.setattr(
        module.cd_surface,
        "sample_staging_frame",
        lambda current, **_k: (
            current,
            {"sampled": True, "rung": "f001", "sample_fraction": 0.01},
        ),
    )
    materialized = SimpleNamespace(
        compiled_specs=parents,
        matrix=sparse.csr_array(
            np.asarray([[1] * len(ids), [10] * len(ids)], np.float32)
        ),
        chunk_stats=[],
        carrier_check={},
    )
    seen = []
    materialized_frames = []

    def fake_materialize(current, specs, **_kwargs):
        if households is None:
            assert current is frame
        materialized_frames.append(current)
        seen.extend(specs)
        return materialized

    monkeypatch.setattr(module, "materialize_chunked", fake_materialize)
    county_reads = []

    # The SOI rows plus every ACS county table, in the pinned row shape;
    # B01003 spans its two record sets and 01003 has no B19013 median.
    acs_rows = _production()._acs_rows(
        ["01001", "01003"],
        values=acs_values,
        without_median=["01003"],
        b01003_group_1=["01003"],
    )

    def fake_county_facts(path, **_kwargs):
        county_reads.append(path)
        rows = (
            *(
                _fact(code, measure, value)
                for measure in ("return_count", "adjusted_gross_income")
                for code, value in (("01001", 1.0), ("01003", 3.0))
            ),
            *acs_rows,
        )
        return SimpleNamespace(
            rows=rows,
            receipt=lambda: {
                "path": str(path),
                "facts_sha256": county_pin.facts_sha256,
                "row_count": len(rows),
            },
        )

    def feed_is_not_county_facts(*_args, **_kwargs):
        raise AssertionError("county facts must not be read from --feed")

    monkeypatch.setattr(county_facts_module, "load_us_county_facts", fake_county_facts)
    monkeypatch.setattr(
        ledger_artifact, "load_ledger_consumer_artifact", feed_is_not_county_facts
    )
    module.do_materialize(args)
    return SimpleNamespace(
        module=module,
        county=county,
        args=args,
        frame=frame,
        materialized_frames=materialized_frames,
        parents=parents,
        seen=seen,
        county_reads=county_reads,
        county_facts_file=county_facts_file,
        county_pin=county_pin,
    )


def test_county_hooks_materialize_a_fixture_and_preserve_configured_geography(
    monkeypatch, tmp_path
) -> None:
    """A three-household smoke exercises real sparse checkpoints, without an engine."""
    run = _materialize_county_fixture(monkeypatch, tmp_path)
    module, county, args = run.module, run.county, run.args
    parents, seen, county_reads = run.parents, run.seen, run.county_reads
    county_facts_file, county_pin = run.county_facts_file, run.county_pin
    assert seen == parents
    # Read exactly once, from the resolved --county-facts path; attach_surface
    # reuses what prepare_run loaded.
    assert county_reads == [county_facts_file.resolve()]
    identity = json.loads((args.checkpoint_dir / "run_identity.json").read_text())
    assert identity["county_facts"]["facts_sha256"] == county_pin.facts_sha256
    assert identity["county_facts"]["path"] == str(county_facts_file.resolve())
    loaded, design, registry, records, matrix = module.load_checkpoint_surface(
        args.checkpoint_dir, identity
    )
    assert identity["county_column"] == "planning_region"
    np.testing.assert_array_equal(
        loaded.table("household")["planning_region"], [1001, 1003, 1001]
    )
    # 2 SOI parents, 4 SOI county rows, 1 ACS households state parent and
    # its 2 county rows, and 3 population rows; validation adds no row.
    assert matrix.shape == (12, 3)
    assert len(registry) == 12
    households_rows = [
        index
        for index, record in enumerate(records)
        if record.get("family") == "acs_households"
    ]
    assert len(households_rows) == 3
    np.testing.assert_array_equal(
        matrix[households_rows].toarray().sum(axis=0), [2.0, 2.0, 2.0]
    )
    assert identity["county_acs_households"]["new_state_targets"] == 1
    assert (
        identity["county_validation"]["design_sha256"]
        == hashlib.sha256(
            (args.checkpoint_dir / "county_validation.json").read_bytes()
        ).hexdigest()
    )
    assert identity["county_support"]["moves"] == 0
    assert {
        record["county_column"]
        for record in records
        if record["geography_level"] == "county"
    } == {"planning_region"}
    assert (
        json.loads((args.checkpoint_dir / "county_surface.json").read_text())[
            "new_engine_columns"
        ]
        == 0
    )
    for record in records:
        if record.get("state_county_parent_target_name"):
            record["role"] = "holdout"
    evidence = module.calibration_evidence(
        frame=loaded,
        roles=records,
        matrix=matrix,
        design_weights=design,
        weights=np.asarray([10.0, 75.0, 15.0]),
        target_loss_cap=1.0,
        checkpoint_dir=args.checkpoint_dir,
        identity=identity,
    )
    # Four SOI and two ACS households county rows.
    assert evidence["county_holdout"]["n_targets"] == 6
    assert evidence["county_reliability"]["county_column"] == "planning_region"
    assert set(evidence["county_reliability"]["counties"]) == {"01001", "01003"}
    assert evidence["county_validation"]["never_in_loss"] is True
    for cell in evidence["county_reliability"]["counties"].values():
        assert cell["validation"]["n_families"] == 4
        assert cell["grade"] in {"A", "B", "C", "D"}
    county.write_reports(args.checkpoint_dir, evidence, identity=identity)
    written = json.loads((args.checkpoint_dir / "county_reliability.json").read_text())
    assert written["county_column"] == "planning_region"
    assert written["counties"]["01001"]["distinct_source_households"] == 1
    assert (args.checkpoint_dir / "county_reliability.csv").read_text() == (
        county.reliability_csv(written)
    )


def _export_county_fixture(module, monkeypatch, run, out_h5, weights):
    """Run the consumer export on a materialized fixture, its heavy steps stubbed.

    Staging reloads as the fixture's original frame, so the export sees the
    counties staging stores, not the ones materialize moved.
    """
    monkeypatch.setattr(
        module, "project_input_only", lambda base, period=None: (base, {})
    )
    monkeypatch.setattr(module, "fill_reviewed_nulls", lambda *a, **k: ([], None))
    # The fixture's take-up stub records the digest "stub" for every seed.
    monkeypatch.setattr(
        module,
        "_recorded_take_up",
        lambda identity: {"seed": 0, "assigned_sha256": "stub"},
    )
    monkeypatch.setattr(
        module,
        "_recorded_ssi_medicaid_take_up",
        lambda *a, **k: ({"assigned_sha256": "", "assignment_file": ""}, None),
    )
    monkeypatch.setattr(
        module, "_resample_like_materialize", lambda frame, identity: frame
    )
    monkeypatch.setattr(
        module, "_with_recorded_ssi_medicaid_take_up", lambda frame, *a: frame
    )
    monkeypatch.setattr(
        module,
        "_with_live_stored_inputs",
        lambda frame: SimpleNamespace(frame=frame, provenance={}),
    )
    identity = json.loads((run.args.checkpoint_dir / "run_identity.json").read_text())
    run.args.out_h5 = out_h5
    module._write_calibrated_artifact(
        run.args,
        weights,
        identity,
        {},
        code=module._repo_code_identity(allow_dirty=True),
    )
    return pd.read_hdf(out_h5, "household")


@pytest.mark.skipif(
    importlib.util.find_spec("tables") is None, reason="requires pytables"
)
def test_support_moves_reach_materialize_the_checkpoint_and_the_export(
    monkeypatch, tmp_path
) -> None:
    """Minimum support through the driver (microcosm#922 review).

    Eight eligible households and one group-quarters record sit in 01001 and
    none in 01003, which overlaps the same PUMA. With K=3, exactly three
    households move. The engine pass must see the moved frame, the lean
    checkpoint must carry the moved counties, reliability after the reload
    must count them in 01003, and the export, which reloads unmoved staging,
    must publish them there too.
    """
    county = _county()
    ids = np.arange(1, 10)
    acs = ids % 2 == 1
    households = pd.DataFrame(
        {
            "household_id": ids,
            "state_fips": 1,
            "county_fips": 1001,
            "county_fips_2023": 1001,
            "puma": 100100,
            "household_spine": np.where(acs, "acs_2024_1yr", "asec_puf"),
            "household_source_id": ids + 100,
            # Household 9 is an ACS group-quarters record: it supports no
            # county and never moves.
            "TYPEHUGQ": np.where(acs, np.where(ids == 9, 3.0, 1.0), np.nan),
        }
    )
    staged = households.copy()
    populations = {1001: 80.0, 1003: 120.0}
    overlaps = county.puma_overlaps([100100, 100100], [1001, 1003], [80.0, 120.0])
    moves, receipt = county.plan_minimum_support(
        staged, populations, overlaps, county_column="county_fips", per_county=3
    )
    moved_ids = sorted(move["household_id"] for move in moves)
    assert len(moved_ids) == 3 and 9 not in moved_ids
    expected = np.where(np.isin(ids, moved_ids), 1003, 1001)

    run = _materialize_county_fixture(
        monkeypatch,
        tmp_path,
        households=households,
        county_column="county_fips",
        overlaps=overlaps,
    )
    module, args = run.module, run.args
    # Staging is untouched; the engine pass saw the moved frame.
    pd.testing.assert_frame_equal(run.frame.table("household"), staged)
    (seen_frame,) = run.materialized_frames
    for column in ("county_fips", "county_fips_2023"):
        np.testing.assert_array_equal(
            seen_frame.table("household")[column].to_numpy(), expected
        )
    np.testing.assert_array_equal(seen_frame.table("household")["puma"], 100100)
    np.testing.assert_array_equal(seen_frame.table("household")["state_fips"], 1)

    support = json.loads((args.checkpoint_dir / county.SUPPORT_FILENAME).read_text())
    assert support["unsupported_counties"] == []
    assert support["counties"]["01003"]["supporting_households"] == 3
    assert support["counties"]["01001"]["supporting_households"] == 5
    assert support["minimum_support"]["moves"] == receipt["moves"]
    identity = json.loads((args.checkpoint_dir / "run_identity.json").read_text())
    assert identity["county_support"]["moves"] == 3
    assert identity["county_support"]["moves_sha256"] == receipt["moves_sha256"]
    assert identity["county_support"]["moved_counties"] == ["01003"]

    loaded, design, _registry, records, matrix = module.load_checkpoint_surface(
        args.checkpoint_dir, identity
    )
    lean = loaded.table("household")
    np.testing.assert_array_equal(lean["household_id"], ids)
    np.testing.assert_array_equal(lean["county_fips"].to_numpy(), expected)
    for record in records:
        if record.get("state_county_parent_target_name"):
            record["role"] = "holdout"
    weights = np.full(len(ids), 10.0)
    evidence = module.calibration_evidence(
        frame=loaded,
        roles=records,
        matrix=matrix,
        design_weights=design,
        weights=weights,
        target_loss_cap=1.0,
        checkpoint_dir=args.checkpoint_dir,
        identity=identity,
    )
    graded = evidence["county_reliability"]["counties"]
    assert graded["01003"]["supporting_households"] == 3
    assert graded["01001"]["supporting_households"] == 5

    exported = _export_county_fixture(
        module, monkeypatch, run, tmp_path / "calibrated.h5", weights
    )
    np.testing.assert_array_equal(exported["household_id"], ids)
    for column in ("county_fips", "county_fips_2023"):
        np.testing.assert_array_equal(exported[column].to_numpy(), expected)
    np.testing.assert_array_equal(exported["puma"], 100100)


def test_validation_values_change_no_loss_input_and_no_weight(
    monkeypatch, tmp_path
) -> None:
    """The held-out ACS families never reach the registry, roles, matrix or solve.

    Two materializations differ only in the four validation tables' public
    values (earnings, households with earnings, household income and its
    brackets). Every loss input is byte-identical and the solved weights are
    equal; only the validation design file differs.
    """
    production = _production()
    runs = []
    for scale in (1, 7):
        values = {}
        for code in ("01001", "01003"):
            data = production._county_values(code)
            values[code] = {
                **data,
                "earning_households": data["earning_households"] // scale,
                "earnings": data["earnings"] * scale,
                "income": data["income"] * scale + scale,
                "brackets": data["brackets"][::-1] if scale > 1 else data["brackets"],
            }
        scale_dir = tmp_path / f"scale_{scale}"
        scale_dir.mkdir()
        run = _materialize_county_fixture(monkeypatch, scale_dir, acs_values=values)
        identity = json.loads(
            (run.args.checkpoint_dir / "run_identity.json").read_text()
        )
        runs.append((run, identity))
    (first, first_identity), (second, second_identity) = runs
    module = first.module
    for name in ("target_registry.json", module.TARGET_ROLES_FILENAME):
        assert (first.args.checkpoint_dir / name).read_bytes() == (
            second.args.checkpoint_dir / name
        ).read_bytes()
    for key in ("target_registry_sha256", "target_roles_sha256", "target_matrix"):
        assert first_identity[key] == second_identity[key]
    assert (
        first_identity["county_validation"]["design_sha256"]
        != second_identity["county_validation"]["design_sha256"]
    )
    weights = []
    for run, identity in runs:
        frame, design, registry, roles, matrix = module.load_checkpoint_surface(
            run.args.checkpoint_dir, identity
        )
        targets = module.cd_surface.calibration_target_set(
            roles, matrix, frame.n("household"), specs=registry.specs
        )
        result, _done = module.calibrate_surface(
            frame,
            targets,
            epochs=20,
            epoch_batch=10,
            max_weight_ratio=5.0,
            target_loss_cap=10.0,
            l2_lambda=0.0,
            seed=0,
        )
        weights.append(np.asarray(result.weights))
    np.testing.assert_array_equal(weights[0], weights[1])


def test_county_loss_weights_share_a_state_concept_budget_without_changing_other_rows() -> (
    None
):
    from microcosm.build.us_runtime import target_loss_weights as weighting

    county = _county()
    parents = [_state_spec(), _state_spec("adjusted_gross_income", 1000.0)]
    facts = [
        _fact(code, measure, value)
        for measure in ("return_count", "adjusted_gross_income")
        for code, value in (("01001", 1.0), ("01003", 3.0))
    ]
    surface = county.state_county_soi_surface(parents, facts, {1001: 50.0, 1003: 50.0})
    assert (
        county.loss_row_mapping(parents)
        is weighting.US_ACS_LOCAL_TARGET_LOSS_ROW_MAPPING
    )
    np.testing.assert_array_equal(
        county.target_loss_weights(parents, {}),
        weighting.us_acs_local_target_loss_weights(parents),
    )
    specs = [
        *parents,
        *surface.specs,
        county.population_spec("pop_county_01001", 50.0),
        county.population_spec("pop_county_01003", 50.0),
    ]
    mapping = county.loss_row_mapping(specs)
    weights = county.target_loss_weights(specs, {})
    assert mapping.mapping_id == "us_acs_local_county.v2"
    assert np.isfinite(weights).all()
    assert np.all(weights > 0)
    for parent in parents:
        indices = [
            index
            for index, spec in enumerate(specs)
            if spec.metadata.get("state_county_parent_target_name") == parent.name
        ]
        assert len({mapping.concept_budget_key(specs[index]) for index in indices}) == 1
        assert weights[indices[0]] / weights[indices[1]] == pytest.approx(
            math.sqrt(specs[indices[0]].value / specs[indices[1]].value)
        )
    assert mapping.value_basis(specs[-1]) == weighting.COUNT_BASIS


def test_packaged_reliability_is_bound_to_current_identity_weights_and_evidence(
    tmp_path,
) -> None:
    county = _county()
    identity = {
        "geographies": "state,county",
        "county_column": "planning_region",
        "population_cells_dropped": ["pop_county_01003"],
    }
    stamp = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    summary = {
        "county_reliability": {
            "schema_version": 1,
            "county_column": "planning_region",
            "counties": {"01001": {"grade": "B"}},
        },
        "run_identity_sha256": stamp,
        "weights_sha256": "current-weights",
    }
    county.write_reports(tmp_path, summary, identity=identity)
    (tmp_path / "calibration_summary.json").write_text(json.dumps(summary))
    files = {}
    county.package_reports(files, tmp_path, identity)
    assert (
        files["county_reliability.json"]["counties"]["01003"]["status"] == "no_support"
    )
    # Exporting the report must not mutate the calibration evidence in memory.
    assert "01003" not in summary["county_reliability"]["counties"]
    for stale in (
        {**summary, "weights_sha256": "previous-weights"},
        {
            **summary,
            "county_reliability": {
                **summary["county_reliability"],
                "counties": {"01001": {"grade": "A"}},
            },
        },
    ):
        (tmp_path / "calibration_summary.json").write_text(json.dumps(stale))
        with pytest.raises(SystemExit, match="does not match"):
            county.package_reports({}, tmp_path, identity)
    (tmp_path / "calibration_summary.json").write_text(json.dumps(summary))
    with pytest.raises(SystemExit, match="does not match"):
        county.package_reports(
            {}, tmp_path, {**identity, "county_column": "county_fips"}
        )


def test_full_run_requires_the_complete_county_inventory_before_materialization(
    tmp_path,
) -> None:
    args = SimpleNamespace(
        soi_mode="state",
        families="soi",
        sample_fraction=1.0,
        checkpoint_dir=tmp_path,
    )
    with pytest.raises(SystemExit, match="3,?144"):
        _county().prepare_run(args, ["county"], {"county": {1001: 10.0, 1003: 20.0}})
    report = json.loads((tmp_path / "county_surface.json").read_text())
    assert report["status"] == "refused"
    assert "3144" in report["error"].replace(",", "")
    # A capped fixture may contain fewer counties; it still has to pass
    # the explicit support checks before any target is dropped.
    _county().prepare_run(
        SimpleNamespace(**{**vars(args), "sample_fraction": 0.01}),
        ["county"],
        {"county": {1001: 10.0, 1003: 20.0}},
    )


def test_county_refresh_recipe_round_trips_custom_geography_and_salt() -> None:
    module = _release()
    metadata = {
        "geographies": " state, county ",
        "county_column": "planning_region",
        "county_holdout": {"fraction": 0.0, "salt": "LWI's county | $(literal)"},
        "county_facts": {
            "path": "/data/LWI county facts.jsonl",
            "facts_sha256": load_us_chronicle_county_facts_pin().facts_sha256,
        },
    }
    recipe = _county().refresh_recipe(
        module.release_refresh_recipe("state_county"), metadata
    )
    parsed = module._parse_args(shlex.split(recipe)[3:])
    assert {item.strip() for item in parsed.geographies.split(",")} == {
        "state",
        "county",
    }
    assert parsed.county_column == "planning_region"
    assert parsed.county_holdout_fraction == 0.0
    assert parsed.county_holdout_salt == metadata["county_holdout"]["salt"]
    # Like --feed, the recipe names a placeholder file and the pinned digest,
    # never the build machine's path.
    assert metadata["county_facts"]["path"] not in recipe
    assert str(parsed.county_facts) == "<county-facts.jsonl>"
    assert parsed.county_facts_sha256 == metadata["county_facts"]["facts_sha256"]
    # No support receipt: the run moved no household, and its replay moves none.
    assert parsed.county_min_support == 0
    supported = _county().refresh_recipe(
        module.release_refresh_recipe("state_county"),
        {**metadata, "county_support": {"households_per_county": 5}},
    )
    assert module._parse_args(shlex.split(supported)[3:]).county_min_support == 5


def test_county_run_metadata_adds_only_the_county_facts_receipt() -> None:
    """A run that read no county facts keeps its pre-county metadata keys."""

    args = SimpleNamespace(geographies="state, cd", county_column="county_fips")
    plain = _county().run_metadata(args, {"fraction": 0.1})
    assert set(plain) == {
        "geographies",
        "county_column",
        "county_holdout",
        "county_targets_dropped",
    }
    receipt = {"path": "/x", "facts_sha256": _pinned_sha(), "row_count": 1}
    args.county_facts_receipt = receipt
    assert _county().run_metadata(args, {"fraction": 0.1}) == {
        **plain,
        "county_facts": receipt,
    }


# --- the county facts flags (microcosm#922) ----------------------------------


def _county_argv(tmp_path, *extra, soi_mode="state_county", feed_sha=True):
    for name in ("staging.h5", "feed.jsonl", "ladder.npz", "county.jsonl"):
        (tmp_path / name).write_bytes(b"fixture")
    argv = [
        "--stage",
        "materialize",
        "--staging-h5",
        str(tmp_path / "staging.h5"),
        "--feed",
        str(tmp_path / "feed.jsonl"),
        "--ladder",
        str(tmp_path / "ladder.npz"),
        "--checkpoint-dir",
        str(tmp_path / "checkpoint"),
        "--families",
        "soi",
        "--soi-mode",
        soi_mode,
        "--geographies",
        "state,county",
    ]
    if feed_sha:
        argv += ["--feed-sha256", hashlib.sha256(b"fixture").hexdigest()]
    return argv + list(extra)


def _pinned_sha():
    return load_us_chronicle_county_facts_pin().facts_sha256


@pytest.mark.parametrize(
    ("extra", "options", "message"),
    [
        ((), {}, "needs --county-facts and --county-facts-sha256"),
        (("--county-facts", "{county}"), {}, "needs --county-facts-sha256"),
        (
            ("--county-facts", "{county}", "--county-facts-sha256", "0" * 64),
            {},
            "is not the county pin's facts_sha256",
        ),
        (
            ("--county-facts", "{county}", "--county-facts-sha256", "{pinned}"),
            {"feed_sha": False},
            "needs --feed-sha256",
        ),
        (
            ("--county-facts", "{feed}", "--county-facts-sha256", "{pinned}"),
            {},
            "must not be the --feed file",
        ),
        (
            ("--county-facts", "{county}", "--county-facts-sha256", "{pinned}"),
            {"soi_mode": "state"},
            "needs --soi-mode state_county",
        ),
        (
            ("--county-facts-sha256", "{pinned}"),
            {"soi_mode": "state"},
            "--county-facts-sha256 needs --county-facts",
        ),
        (
            (
                "--county-facts",
                "{county}",
                "--county-facts-sha256",
                "{pinned}",
                "--geographies",
                "state,cd",
            ),
            {},
            "county in --geographies",
        ),
    ],
    ids=[
        "state-county-without-facts",
        "facts-without-sha",
        "sha-not-the-pin",
        "no-feed-sha",
        "facts-are-the-feed",
        "facts-outside-state-county",
        "sha-without-facts",
        "facts-without-county-geography",
    ],
)
def test_the_county_facts_flags_refuse_an_unpinned_or_ambiguous_run(
    tmp_path, capsys, extra, options, message
) -> None:
    fill = {
        "county": str(tmp_path / "county.jsonl"),
        "feed": str(tmp_path / "feed.jsonl"),
        "pinned": _pinned_sha(),
    }
    argv = _county_argv(tmp_path, *(item.format(**fill) for item in extra), **options)
    with pytest.raises(SystemExit):
        _release()._parse_args(argv)
    assert message in capsys.readouterr().err


def test_a_pinned_county_facts_run_parses(tmp_path) -> None:
    args = _release()._parse_args(
        _county_argv(
            tmp_path,
            "--county-facts",
            str(tmp_path / "county.jsonl"),
            "--county-facts-sha256",
            _pinned_sha(),
        )
    )
    assert args.county_facts == tmp_path / "county.jsonl"
    assert args.county_facts_sha256 == _pinned_sha()


def _v5_soi_row(county, measure, value):
    """A county SOI row in the pinned v5 shape, with fields the loader drops."""

    return {
        **_fact(county, measure, value),
        "aggregate_fact_key": f"key-{county}-{measure}",
        "schema_version": "chronicle.consumer_fact.v5",
        "dimension_value_labels": {"irs_soi.county": {county: f"County {county}"}},
        "universe_constraints": {"domain": "tax_returns", "constraints": []},
        "lineage": {"source_record_id": f"irs_soi.county.{county}"},
        "source": {"source_file": "22incyallnoagi.csv"},
    }


def _soi_rows():
    return [
        _v5_soi_row(code, measure, value)
        for measure in ("return_count", "adjusted_gross_income")
        for code, value in (("01001", 1.0), ("01003", 3.0))
    ]


def _pin_for(payload, rows):
    return dataclasses.replace(
        load_us_chronicle_county_facts_pin(),
        facts_sha256=hashlib.sha256(payload).hexdigest(),
        fact_row_count=rows,
    )


@pytest.mark.parametrize("case", ["off-pin digest", "missing file", "no county facts"])
def test_a_county_facts_file_off_the_pin_is_refused_before_any_target(
    tmp_path, case
) -> None:
    facts = tmp_path / "county.jsonl"
    if case == "off-pin digest":
        # Well-formed pinned-contract rows, but not the pinned bytes.
        facts.write_bytes(
            b"".join(json.dumps(row).encode() + b"\n" for row in _soi_rows())
        )
    args = SimpleNamespace(
        soi_mode="state_county",
        families="soi",
        sample_fraction=0.01,
        checkpoint_dir=tmp_path / "checkpoint",
        county_facts=None if case == "no county facts" else facts,
    )
    surface = SimpleNamespace(registry=SimpleNamespace(specs=()))
    with pytest.raises(SystemExit):
        _county().prepare_run(
            args, ["state", "county"], {"county": {1001: 10.0}}, surface=surface
        )
    report = json.loads((args.checkpoint_dir / "county_surface.json").read_text())
    assert report["status"] == "refused"
    expected = {
        "off-pin digest": "the county pin names",
        "missing file": "county.jsonl",
        "no county facts": "needs --county-facts",
    }[case]
    assert expected in report["error"]
    assert not hasattr(surface, "county_surface")


@pytest.mark.parametrize(
    "swapped_row",
    [
        {"period": "2022"},
        {"schema_version": ["chronicle.consumer_fact.v5"]},
        {"schema_version": "chronicle.consumer_fact.v6"},
        {"layout": None, "period": 7},
    ],
    ids=["period-text", "contract-list", "contract-v6", "period-number"],
)
def test_a_county_facts_file_swapped_between_passes_is_recorded_as_changed(
    monkeypatch, tmp_path, swapped_row
) -> None:
    """microcosm#922 review P2-1: a file that changes after the digest pass is
    recorded as changed, whatever its rows hold, never an unrecorded crash."""

    from microcosm.build.us_runtime import county_facts as county_facts_module

    pin = load_us_chronicle_county_facts_pin()
    # The digest pass sees the pinned bytes; the parsing pass reads this file.
    monkeypatch.setattr(
        county_facts_module,
        "_digest",
        lambda _path: (pin.facts_sha256, pin.fact_row_count),
    )
    rows = _soi_rows()
    rows[0] = {**rows[0], **swapped_row}
    facts = tmp_path / "county.jsonl"
    facts.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
    args = SimpleNamespace(
        soi_mode="state_county",
        families="soi",
        sample_fraction=0.01,
        checkpoint_dir=tmp_path / "checkpoint",
        county_facts=facts,
    )
    surface = SimpleNamespace(registry=SimpleNamespace(specs=()))
    with pytest.raises(SystemExit):
        _county().prepare_run(
            args, ["state", "county"], {"county": {1001: 10.0}}, surface=surface
        )
    report = json.loads((args.checkpoint_dir / "county_surface.json").read_text())
    assert report["status"] == "refused"
    assert "changed while it was being read" in report["error"]


def test_projected_county_rows_give_the_same_soi_surface_as_full_rows(
    tmp_path,
) -> None:
    """The loader keeps every field the SOI county surface reads."""

    from microcosm.build.us_runtime.county_facts import load_us_county_facts

    rows = _soi_rows()
    path = tmp_path / "county.jsonl"
    payload = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    path.write_bytes(payload)
    loaded = load_us_county_facts(path, pin=_pin_for(payload, len(rows)))
    assert all("lineage" not in row for row in loaded.rows)
    parents = [_state_spec(), _state_spec("adjusted_gross_income", 1000.0)]
    populations = {1001: 80.0, 1003: 120.0}
    full = _county().state_county_soi_surface(parents, rows, populations)
    projected = _county().state_county_soi_surface(parents, loaded.rows, populations)
    assert projected.receipt == full.receipt
    assert [(s.name, s.value, s.metadata) for s in projected.specs] == [
        (s.name, s.value, s.metadata) for s in full.specs
    ]
