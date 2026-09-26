"""End-to-end test of tools/build_us_sld_local_layer.py (populace#625)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_tool_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_sld_local_layer.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_sld_local_layer",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_ladder(tmp_path: Path) -> Path:
    """Two Utah districts per chamber over two tracts, one split."""
    path = tmp_path / "ladder.npz"
    np.savez(
        path,
        tract_geoid=np.array(
            [49049000100, 49049000200, 49049000200],
            dtype=np.int64,
        ),
        tract_sldu=np.array(["001", "001", "002"]),
        tract_sldl=np.array(["010", "010", "011"]),
        tract_population=np.array([1000, 600, 400], dtype=np.int64),
        cell_puma=np.array([4904901, 4904901, 4904901], dtype=np.int64),
        cell_cd=np.array([4903, 4903, 4903], dtype=np.int64),
        cell_county=np.array([49049, 49049, 49049], dtype=np.int64),
        cell_sldu=np.array(["001", "001", "002"]),
        cell_sldl=np.array(["010", "010", "011"]),
        cell_population=np.array([1000, 600, 400], dtype=np.int64),
        meta_boundary_vintage=np.array("2024_state_legislative_districts"),
        meta_source_kind=np.array("census_2024_sld_bef"),
    )
    return path


def _frames(n_per_tract: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    tracts = np.repeat(
        [49049000100, 49049000200],
        n_per_tract,
    )
    n = len(tracts)
    households = pd.DataFrame(
        {
            "household_id": np.arange(1, n + 1),
            "state_fips": 49,
            "puma": 4901,  # 5-digit ACS convention; the tool normalizes.
            "congressional_district_geoid": 4903,
            "county_fips": 49049,
            "tract_geoid": tracts,
            "household_weight": rng.uniform(50.0, 150.0, n),
        }
    )
    person_rows = []
    for household_id in households["household_id"]:
        size = int(rng.integers(1, 4))
        for _ in range(size):
            person_rows.append(
                {
                    "person_household_id": household_id,
                    "age": float(rng.integers(0, 90)),
                    "employment_income_before_lsr": float(
                        rng.choice([0.0, 20_000.0, 80_000.0, 250_000.0])
                    ),
                    "self_employment_income_before_lsr": 0.0,
                    "taxable_interest_income": 0.0,
                    "social_security_retirement": 0.0,
                }
            )
    return households, pd.DataFrame(person_rows)


def _fact_row(geo_id, level, concept, value, constraints, aggregation="sum"):
    entity = "person" if concept == "census_acs.person_count" else "household"
    return {
        "geography": {
            "id": geo_id,
            "level": level,
            "vintage": "2024_state_legislative_districts",
        },
        "entity": {"name": entity, "role": f"resident_{entity}"},
        "aggregation": {"method": aggregation, "denominator": None},
        "period": {"type": "calendar_year", "value": 2024},
        "measure": {"concept": concept},
        "constraints": constraints,
        "value": value,
    }


def _write_facts(tmp_path: Path) -> Path:
    """Achievable district facts: modest scalings of plausible masses."""
    rows = []
    for code, households_target in (("001", 9_000.0), ("002", 4_000.0)):
        geo_id = f"610U900US49{code}"
        level = "state_legislative_district_upper"
        rows.append(
            _fact_row(
                geo_id,
                level,
                "census_acs.household_count",
                households_target,
                [],
            )
        )
        rows.append(
            _fact_row(
                geo_id,
                level,
                "census_acs.household_count",
                households_target * 0.4,
                [
                    {
                        "variable": "household_income",
                        "operator": "<",
                        "value": 60_000,
                    }
                ],
            )
        )
        rows.append(
            _fact_row(
                geo_id,
                level,
                "census_acs.person_count",
                households_target * 0.5,
                [
                    {"variable": "age", "operator": ">=", "value": 0},
                    {"variable": "age", "operator": "<", "value": 18},
                ],
            )
        )
        rows.append(
            _fact_row(
                geo_id,
                level,
                "census_acs.median_household_income",
                65_000.0,
                [],
                aggregation="median",
            )
        )
    path = tmp_path / "sld_facts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_layer_runs_end_to_end_and_writes_the_sidecar(tmp_path):
    module = _load_tool_module()
    households, persons = _frames()
    out_dir = tmp_path / "sidecar"
    summary = module.run_us_sld_local_layer(
        households,
        persons,
        facts_path=_write_facts(tmp_path),
        ladder_path=_write_ladder(tmp_path),
        out_dir=out_dir,
        epochs=200,
        seed=0,
    )
    assert summary["scope_states"] == ["49"]
    assert summary["membership_gate"]["n_rows"] == len(households)
    assert summary["doctrine"]["max_weight_ratio"] == 100.0
    (chamber,) = summary["chambers"]
    assert chamber["area_type"] == "sldu"
    assert chamber["n_districts"] == 2
    assert chamber["max_realized_weight_ratio"] <= 100.0
    for filename in (
        "sld_local_weights.csv.gz",
        "sld_achieved_vs_target.csv.gz",
        "sld_local_diagnostics.json",
        "sld_local_boundaries.json",
        "sld_local_boundaries.md",
        "sld_layer_summary.json",
    ):
        assert (out_dir / filename).exists(), filename
    diagnostics = json.loads((out_dir / "sld_local_diagnostics.json").read_text())
    assert diagnostics["median_income_validation"], "median check must run"
    weights = pd.read_csv(out_dir / "sld_local_weights.csv.gz")
    assert set(weights["area_code"]) == {"610U900US49001", "610U900US49002"}
    # Every scoped household lands in exactly one upper district.
    assert len(weights) == len(households)
    # The solve moved district masses toward their targets.
    achieved = pd.read_csv(out_dir / "sld_achieved_vs_target.csv.gz")
    households_rows = achieved[achieved["metric"] == "households"]
    assert (
        (households_rows["final_estimate"] - households_rows["target"]).abs()
        / households_rows["target"]
    ).max() < 0.05


def test_layer_refuses_when_no_fact_state_overlaps(tmp_path):
    module = _load_tool_module()
    households, persons = _frames(n_per_tract=5)
    households["state_fips"] = 6  # California rows only
    with pytest.raises(SystemExit, match="no households in fact state"):
        module.run_us_sld_local_layer(
            households,
            persons,
            facts_path=_write_facts(tmp_path),
            ladder_path=_write_ladder(tmp_path),
            out_dir=tmp_path / "out",
            epochs=8,
        )


def test_normalize_puma_promotes_five_digit_codes():
    module = _load_tool_module()
    frame = pd.DataFrame(
        {
            "state_fips": [49, 49, 6],
            "puma": [4901, 4904901, 101],
        }
    )
    normalized = module.normalize_puma(frame)
    assert normalized["puma"].tolist() == [4904901, 4904901, 600101]


def test_h5_loader_reads_year_keyed_layout(tmp_path):
    h5py = pytest.importorskip("h5py")
    module = _load_tool_module()
    path = tmp_path / "artifact.h5"
    with h5py.File(path, "w") as h5file:
        h5file.create_dataset("household_id/2024", data=np.array([1, 2]))
        h5file.create_dataset("state_fips/2024", data=np.array([49, 49]))
        h5file.create_dataset("household_weight/2024", data=np.array([10.0, 20.0]))
        h5file.create_dataset("person_household_id/2024", data=np.array([1, 2]))
        h5file.create_dataset("age/2024", data=np.array([30.0, 40.0]))
    households, persons = module.load_layer_frames(path)
    assert households["household_id"].tolist() == [1, 2]
    assert persons["age"].tolist() == [30.0, 40.0]


def test_h5_loader_reads_entity_table_layout(tmp_path):
    pytest.importorskip("tables")
    module = _load_tool_module()
    path = tmp_path / "artifact_tables.h5"
    households = pd.DataFrame(
        {
            "household_id": [1, 2],
            "state_fips": [49, 49],
            "household_weight": [10.0, 20.0],
            "tract_geoid": ["49049000100", "49049000200"],
            "unrelated_column": ["x", "y"],
        }
    )
    persons = pd.DataFrame(
        {
            "person_household_id": [1, 2, 2],
            "age": [30.0, 40.0, 8.0],
            "employment_income_before_lsr": [1.0, 2.0, 0.0],
            "another_unrelated": [0, 0, 0],
        }
    )
    households.to_hdf(path, key="household", format="table")
    persons.to_hdf(path, key="person", format="table")
    loaded_households, loaded_persons = module.load_layer_frames(
        path,
        extra_person_columns=("employment_income_before_lsr",),
    )
    assert "unrelated_column" not in loaded_households.columns
    assert loaded_households["tract_geoid"].tolist() == [
        "49049000100",
        "49049000200",
    ]
    assert loaded_persons["employment_income_before_lsr"].tolist() == [
        1.0,
        2.0,
        0.0,
    ]
    assert "another_unrelated" not in loaded_persons.columns
