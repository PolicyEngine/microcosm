"""The pre-county target surface stays exact when no county flag is supplied.

The oracle below records the synthetic surface of the builder at dec45a439,
before the county hooks. It declares expected rows and roles independently;
neither the old builder nor git is needed to run the regression.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.microcosm_build.us_acs_local_reviewed_nulls import (
    load_tool_module,
)

_HT2 = "irs_soi.historic_table_2.state_broad_totals.v1"
_CD = "irs_soi.congressional_district_2022.all_returns.v1"

# Deliberately put district 0602 before 0601: surface order is observable.
_SOI_INPUTS = (
    ("state_returns", "return_count", 300.0, None, _HT2),
    ("state_agi", "adjusted_gross_income", 1000.0, None, _HT2),
    ("state_ptc", "premium_tax_credit_amount", 20.0, None, _HT2),
    ("cd_file_returns", "return_count", 150.0, None, _CD),
    ("cd_file_agi", "adjusted_gross_income", 500.0, None, _CD),
    ("cd_0602_returns", "return_count", 90.0, "0602", _CD),
    ("cd_0601_returns", "return_count", 60.0, "0601", _CD),
    ("cd_0602_agi", "adjusted_gross_income", 300.0, "0602", _CD),
    ("cd_0601_agi", "adjusted_gross_income", 200.0, "0601", _CD),
)

_EXPECTED_ADMIN_NAMES = {
    "state": ("snap", "medicaid", "state_returns", "state_agi", "state_ptc"),
    "totals": ("snap", "medicaid", "state_ptc"),
    "full": ("snap", "medicaid", *(row[0] for row in _SOI_INPUTS)),
    "state_cd": (
        "snap",
        "medicaid",
        "state_returns",
        "state_agi",
        "state_ptc",
        "cd_0602_returns",
        "cd_0601_returns",
        "cd_0602_agi",
        "cd_0601_agi",
    ),
}

# Household order is [0602, 0601, 0101, 0602], sizes [2, 1, 3, 1].
# The signed AGI row catches accidental clipping or county masking.
_MEASURES = {
    "snap": [0, 0, 1, 0],
    "medicaid": [1, 0, 0, 1],
    "return_count": [1, 1, 0, 1],
    "adjusted_gross_income": [-50, 150, 0, 25],
    "premium_tax_credit_amount": [0, 20, 0, 0],
    "pop_state_01": [0, 0, 3, 0],
    "pop_state_06": [2, 1, 0, 1],
    "pop_cd_0101": [0, 0, 3, 0],
    "pop_cd_0601": [0, 1, 0, 0],
    "pop_cd_0602": [2, 0, 0, 1],
}
_POPULATIONS = {
    "state": {1: 30.0, 6: 100.0},
    "cd": {101: 30.0, 601: 40.0, 602: 60.0},
}
_EXPECTED_ADMIN_ROWS = {
    "snap": [0, 0, 1, 0],
    "medicaid": [1, 0, 0, 1],
    "state_returns": [1, 1, 0, 1],
    "state_agi": [-50, 150, 0, 25],
    "state_ptc": [0, 20, 0, 0],
    "cd_file_returns": [1, 1, 0, 1],
    "cd_file_agi": [-50, 150, 0, 25],
    "cd_0602_returns": [1, 0, 0, 1],
    "cd_0601_returns": [0, 1, 0, 0],
    "cd_0602_agi": [-50, 0, 0, 25],
    "cd_0601_agi": [0, 150, 0, 0],
}


def _input_specs():
    def spec(name, family, value, **metadata):
        return TargetSpec(
            name=name,
            entity="household",
            measure=name,
            value=value,
            period=2024,
            source="synthetic ledger",
            family=family,
            metadata=metadata,
        )

    rows = [
        spec(
            "snap", "usda_snap", 30.0, state_fips="01", ledger_geography_level="state"
        ),
        spec(
            "medicaid",
            "cms_medicaid",
            40.0,
            state_fips="06",
            ledger_geography_level="state",
            target_role="medicaid_enrollment",
        ),
        spec(
            "medicaid_spending",
            "cms_medicaid",
            400.0,
            state_fips="06",
            ledger_geography_level="state",
            target_role="medicaid_spending",
        ),
        spec("national_snap", "usda_snap", 500.0, ledger_geography_level="national"),
    ]
    for name, measure, value, district, record_set in _SOI_INPUTS:
        variable = "count" if measure == "return_count" else measure
        metadata = {
            "state_fips": "06",
            "ledger_geography_level": "congressional_district" if district else "state",
            "ledger_layout_record_set_spec_id": record_set,
            "source_measure_id": measure,
            "source_variable": variable,
            "variable": variable,
            "agi_lower_bound": "-inf",
            "agi_upper_bound": "inf",
            "filing_status": "All",
            "measure_mode": "sum",
            "target_role": "aca_spending"
            if name == "state_ptc"
            else "soi_fiscal_distribution",
        }
        if district:
            metadata["congressional_district_geoid"] = district
        rows.append(spec(name, "irs_soi", value, **metadata))
    return rows


def _frame():
    ids = np.asarray([10, 20, 30, 40])
    person_households = np.repeat(ids, [2, 1, 3, 1])
    person_ids = np.arange(1, 8)
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": person_ids,
                **{
                    f"person_{group}_id": person_ids
                    for group in US_SCHEMA.group_entities
                },
                "person_household_id": person_households,
            }
        ),
        **{
            group: pd.DataFrame({f"{group}_id": person_ids})
            for group in US_SCHEMA.group_entities
        },
    }
    tables["household"] = pd.DataFrame(
        {
            "household_id": ids,
            "state_fips": [6, 6, 1, 6],
            "congressional_district_geoid": [602, 601, 101, 602],
        }
    )
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(np.asarray([3.0, 4.0, 5.0, 6.0]), WeightKind.DESIGN),
        },
    )


def _expected_admin_record(name, mode):
    if name in {"snap", "medicaid"}:
        family, state, measure, value, district = (
            ("usda_snap", "01", None, 30.0, None)
            if name == "snap"
            else ("cms_medicaid", "06", None, 40.0, None)
        )
    else:
        _, measure, value, district, _ = next(
            row for row in _SOI_INPUTS if row[0] == name
        )
        family, state = "irs_soi", "06"
    record = {
        "name": name,
        "entity": "household",
        "period": 2024,
        "value": value,
        "source": "synthetic ledger",
        "family": family,
        "geography_level": "congressional_district" if district else "state",
        "state_fips": state,
        "congressional_district_geoid": district,
        "source_measure_id": measure,
        "sigma": None,
        "sigma_basis": "not_provided_by_feed",
        "role": "train",
        "holdout_unit": None,
    }
    if mode == "state_cd" and district:
        parent = "state_returns" if measure == "return_count" else "state_agi"
        record.update(
            {
                "value": value * 2,
                "state_cd_parent_target_name": parent,
                "state_cd_parent_basis": "historic_table_2",
                "state_cd_rebase_factor": "2.0",
                "state_cd_cd_file_value": repr(value),
                "cd_population": 40.0 if district == "0601" else 60.0,
                "state_population": 100.0,
                "holdout_unit": "06|return"
                if measure == "return_count"
                else "06|adjusted_gross_income",
            }
        )
        # Historical default 10% CD holdout: the AGI block hashes below .1,
        # and the returns block above it, under the committed CD salt.
        if measure == "adjusted_gross_income":
            record["role"] = "holdout"
    return record


@pytest.mark.parametrize("soi_mode", ["state", "totals", "full", "state_cd"])
@pytest.mark.parametrize("geographies", ["state", "cd", "state,cd", "cd,state"])
def test_no_county_materialization_matches_pre_county_surface(
    monkeypatch, tmp_path, soi_mode, geographies
):
    """Exercise the actual materialize assembly with all county hooks enabled."""
    from microcosm.build import ledger_artifact
    from microcosm.build.us_runtime import fiscal_targets, medicaid_take_up, puma_ladder

    module = load_tool_module()
    frame = _frame()
    inputs = _input_specs()
    crosswalk = pd.DataFrame(
        {
            "source_geography_id": ["5001700US0601", "5001700US0602"],
            "target_geography_id": ["5001900US0601", "5001900US0602"],
        }
    )
    monkeypatch.setattr(
        ledger_artifact,
        "load_ledger_consumer_artifact",
        lambda *_: SimpleNamespace(facts=[]),
    )
    monkeypatch.setattr(
        fiscal_targets,
        "compile_us_fiscal_target_registry",
        lambda *_a, **_kw: TargetRegistry(inputs, country="us"),
    )
    monkeypatch.setattr(
        medicaid_take_up,
        "apply_us_medicaid_enrollment_substitutions",
        lambda registry: (registry, []),
    )
    # Keep state_admin_surface's selection/reconciliation real, with a tiny plan.
    import microcosm.build.us_runtime as runtime

    monkeypatch.setattr(
        runtime, "load_congressional_district_vintage_crosswalk", lambda *_: crosswalk
    )
    monkeypatch.setattr(
        puma_ladder,
        "load_us_puma_ladder",
        lambda *_: SimpleNamespace(
            puma=np.asarray([100001, 600001]),
            puma_population=np.asarray([30.0, 100.0]),
            cd_overlap_cd=np.asarray([101, 601, 602]),
            cd_overlap_population=np.asarray([30.0, 40.0, 60.0]),
        ),
    )
    monkeypatch.setattr(module, "_load_staging_frame", lambda *_: frame)
    monkeypatch.setattr(module, "_require_local_hours", lambda *_: None)
    monkeypatch.setattr(module, "log", lambda *_: None)

    def materialize(_frame, specs, **_kwargs):
        values = []
        for spec in specs:
            measure = spec.metadata.get("source_measure_id", spec.name)
            row = np.asarray(_MEASURES[measure], dtype=np.float32)
            district = spec.metadata.get("congressional_district_geoid")
            if district:
                row = row * (np.asarray([602, 601, 101, 602]) == int(district))
            values.append(row)
        return SimpleNamespace(
            compiled_specs=specs,
            matrix=sparse.csr_array(np.asarray(values)),
            chunk_stats=[],
            carrier_check={},
        )

    monkeypatch.setattr(module, "materialize_chunked", materialize)
    captured = {}

    def checkpoint(_struct, matrix, specs, roles, directory):
        captured.update(matrix=matrix.copy(), specs=specs, roles=roles)
        digests = {
            name: "fixture"
            for name in (
                "target_registry_sha256",
                "target_roles_sha256",
                "lean_h5_sha256",
                "target_matrix_sha256",
            )
        }
        return directory / "unused.h5", None, digests

    monkeypatch.setattr(module, "write_lean_checkpoint", checkpoint)
    for filename in ("staging.h5", "feed.json", "ladder.npz"):
        (tmp_path / filename).write_bytes(b"fixture")
    (tmp_path / "staging.summary.json").write_text("{}")
    args = module._parse_args(
        [
            "--stage",
            "materialize",
            "--staging-h5",
            str(tmp_path / "staging.h5"),
            "--feed",
            str(tmp_path / "feed.json"),
            "--ladder",
            str(tmp_path / "ladder.npz"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--soi-mode",
            soi_mode,
            "--geographies",
            geographies,
        ]
    )
    module.do_materialize(args)

    records = [
        _expected_admin_record(name, soi_mode)
        for name in _EXPECTED_ADMIN_NAMES[soi_mode]
    ]
    expected_rows = [_EXPECTED_ADMIN_ROWS[record["name"]] for record in records]
    for geography in geographies.split(","):
        for code, value in _POPULATIONS[geography].items():
            name = f"pop_{geography}_{code:0{2 if geography == 'state' else 4}d}"
            records.append(
                {
                    "name": name,
                    "value": value,
                    "source": "us_puma_ladder_2020",
                    "family": "census_population_ladder",
                    "geography_level": "state"
                    if geography == "state"
                    else "congressional_district",
                    "state_fips": f"{code if geography == 'state' else code // 100:02d}",
                    "congressional_district_geoid": None
                    if geography == "state"
                    else f"{code:04d}",
                    "role": "train",
                    "holdout_unit": None,
                }
            )
            expected_rows.append(_MEASURES[name])

    assert captured["roles"] == records
    assert [(spec.name, spec.value) for spec in captured["specs"]] == [
        (record["name"], record["value"]) for record in records
    ]
    expected_matrix = sparse.csr_array(np.asarray(expected_rows, dtype=np.float32))
    actual_matrix = captured["matrix"]
    assert actual_matrix.shape == expected_matrix.shape
    assert actual_matrix.dtype == np.float32
    for attribute in ("data", "indices", "indptr"):
        np.testing.assert_array_equal(
            getattr(actual_matrix, attribute), getattr(expected_matrix, attribute)
        )
    identity = json.loads((args.checkpoint_dir / "run_identity.json").read_text())
    assert identity["population_cells_dropped"] == []
    assert identity["n_targets"] == len(records)
    assert module._solver_settings(args, {"fixture": "loss stamp"}) == {
        "method": "adam",
        "learning_rate": 0.02,
        "mass": "conserve",
        "max_weight_ratio": 5.0,
        "target_loss_cap": 1.0,
        "l2_lambda": 0.0,
        "l2_basis": "record",
        "mass_parametrization": "projection",
        "seed": 0,
        "epoch_batch": 400,
        "target_loss": {"fixture": "loss stamp"},
    }
