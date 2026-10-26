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


def test_kalawao_gap_still_refuses_the_county_surface_with_the_missing_code() -> None:
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
    assert report["schema_version"] == 2
    assert report["grade_rule"]["provisional"] is True
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


@pytest.mark.skipif(
    importlib.util.find_spec("tables") is None, reason="requires pytables"
)
def test_county_hooks_materialize_a_fixture_and_preserve_configured_geography(
    monkeypatch, tmp_path
) -> None:
    """A three-household smoke exercises real sparse checkpoints, without an engine."""
    from microcosm.build import ledger_artifact
    from microcosm.calibrate import TargetRegistry
    from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

    module = _release()
    county = _county()
    ids = np.arange(1, 4)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            **{f"person_{group}_id": ids for group in US_SCHEMA.group_entities},
        }
    )
    tables = {"person": person}
    for group in US_SCHEMA.group_entities:
        tables[group] = pd.DataFrame({f"{group}_id": ids})
    tables["household"] = pd.DataFrame(
        {
            "household_id": ids,
            "state_fips": [1, 1, 1],
            "county_fips": [99999, 99999, 99999],
            "planning_region": [1001, 1003, 1001],
            "household_spine": ["asec_puf"] * 3,
            "household_source_id": [7, 8, 7],
        }
    )
    frame = Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.asarray([10.0, 20.0, 30.0]), WeightKind.DESIGN)},
    )
    staging, feed, ladder = (
        tmp_path / name for name in ("staging.h5", "feed.json", "ladder.npz")
    )
    for path in (staging, feed, ladder):
        path.write_bytes(b"fixture")
    staging.with_suffix(".summary.json").write_text("{}")
    args = module._parse_args(
        [
            "--stage",
            "materialize",
            "--staging-h5",
            str(staging),
            "--feed",
            str(feed),
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
            "planning_region",
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
        matrix=sparse.csr_array(np.asarray([[1, 1, 1], [10, 10, 10]], np.float32)),
        chunk_stats=[],
        carrier_check={},
    )
    seen = []

    def fake_materialize(current, specs, **_kwargs):
        assert current is frame
        seen.extend(specs)
        return materialized

    monkeypatch.setattr(module, "materialize_chunked", fake_materialize)
    monkeypatch.setattr(
        ledger_artifact,
        "load_ledger_consumer_artifact",
        lambda *_a: SimpleNamespace(
            facts=[
                _fact(code, measure, value)
                for measure in ("return_count", "adjusted_gross_income")
                for code, value in (("01001", 1.0), ("01003", 3.0))
            ]
        ),
    )
    module.do_materialize(args)
    assert seen == parents
    identity = json.loads((args.checkpoint_dir / "run_identity.json").read_text())
    loaded, design, registry, records, matrix = module.load_checkpoint_surface(
        args.checkpoint_dir, identity
    )
    assert identity["county_column"] == "planning_region"
    np.testing.assert_array_equal(
        loaded.table("household")["planning_region"], [1001, 1003, 1001]
    )
    assert matrix.shape == (9, 3)
    assert len(registry) == 9
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
    )
    assert evidence["county_holdout"]["n_targets"] == 4
    assert evidence["county_reliability"]["county_column"] == "planning_region"
    assert set(evidence["county_reliability"]["counties"]) == {"01001", "01003"}
    county.write_reports(args.checkpoint_dir, evidence, identity=identity)
    written = json.loads((args.checkpoint_dir / "county_reliability.json").read_text())
    assert written["county_column"] == "planning_region"
    assert written["counties"]["01001"]["distinct_source_households"] == 1


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
    assert mapping.mapping_id == "us_acs_local_county.v1"
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
