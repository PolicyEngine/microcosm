"""Census household-count targets from the sha-pinned ladder (#495 inc 6c).

The first bound local target family. Census occupied-household counts by
constituency (and local authority) come straight from the full-UK ladder
artifact, whose three household-count sources are already sha-pinned per
layer — no new external pinning. The family is universe-compatible with the
FRS instrument (census occupied households vs the survey's own household
frame), unlike person-grain families that inherit the
population_universe_private_households adjudication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.uk_runtime import (
    assemble_uk_oa_ladder,
    compute_household_metrics,
    constituency_household_targets,
    load_uk_oa_ladder,
    local_authority_household_targets,
    metric_names,
)


def _ladder(tmp_path):
    def layer(vintage: str) -> dict[str, object]:
        return {"vintage": vintage, "source": "synthetic test source"}

    metadata = {
        "schema_version": 1,
        "kind": "uk_oa_ladder",
        "coverage": "uk",
        "oa_vintage": "synthetic",
        "constituency_sampling_basis": "synthetic household counts",
        "oa_sampling_basis": "synthetic population",
        "layers": {
            "constituency": layer("2024_pcon"),
            "lsoa": layer("synthetic"),
            "msoa": layer("synthetic"),
            "local_authority": layer("synthetic"),
            "ward": layer("synthetic"),
            "itl": layer("2021_itl"),
            "region": layer("synthetic"),
        },
    }
    rows = [
        ("E00000001", "E12000007", "E14000001", "E09000001", 100.0, 40.0),
        ("E00000002", "E12000007", "E14000001", "E09000001", 50.0, 15.0),
        ("E00000003", "E12000007", "E14000002", "E09000002", 80.0, 30.0),
        ("S00000001", "S99999999", "S14000001", "S12000033", 90.0, 35.0),
    ]
    frame = pd.DataFrame(
        [
            {
                "oa_code": oa,
                "population": population,
                "households": households,
                "constituency_code": constituency,
                "region_code": region,
                "lsoa_code": oa,
                "msoa_code": oa,
                "local_authority_code": la,
                "ward_code": "E05014284" if oa.startswith("E") else "S13002835",
                "itl3_code": "TLI31" if oa.startswith("E") else "TLM50",
            }
            for oa, region, constituency, la, population, households in rows
        ]
    )
    payload = assemble_uk_oa_ladder(frame, metadata)
    path = tmp_path / "ladder.npz"
    np.savez_compressed(path, **payload)
    return load_uk_oa_ladder(path)


def test_constituency_household_targets_sum_ladder_counts(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    targets = constituency_household_targets(ladder)
    assert list(targets.columns) == ["code", "households"]
    rows = dict(zip(targets["code"], targets["households"], strict=True))
    assert rows == {
        "E14000001": pytest.approx(55.0),
        "E14000002": pytest.approx(30.0),
        "S14000001": pytest.approx(35.0),
    }
    # Deterministic order for target-surface stability.
    assert targets["code"].tolist() == sorted(targets["code"].tolist())


def test_local_authority_household_targets_sum_ladder_counts(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    targets = local_authority_household_targets(ladder)
    rows = dict(zip(targets["code"], targets["households"], strict=True))
    assert rows == {
        "E09000001": pytest.approx(55.0),
        "E09000002": pytest.approx(30.0),
        "S12000033": pytest.approx(35.0),
    }


def test_households_metric_is_in_the_computed_surface() -> None:
    assert "households" in metric_names("constituency")
    assert "households" in metric_names("la")

    class Result:
        def __init__(self, values):
            self.values = np.asarray(values)

    class FakeUKSimulation:
        person_household = np.asarray([0, 0, 1, 2])
        benunit_household = np.asarray([0, 1, 2])
        data = {
            "household_id": [101, 102, 103],
            "self_employment_income": [0.0, 100.0, 0.0, 50.0],
            "employment_income": [10.0, 20.0, 0.0, 30.0],
            "income_tax": [1.0, 0.0, 2.0, 3.0],
            "age": [5, 35, 72, 12],
            "universal_credit": [0.0, 100.0, 50.0],
            "is_child": [1.0, 0.0, 0.0, 1.0],
        }

        def calculate(self, variable, **_kwargs):
            return Result(self.data[variable])

        def map_result(self, values, from_entity, to_entity):
            values = np.asarray(values, dtype=float)
            out = np.zeros(3, dtype=float)
            mapping = (
                self.person_household
                if from_entity == "person"
                else self.benunit_household
            )
            for index, household_index in enumerate(mapping):
                out[household_index] += values[index]
            return out

    metrics = compute_household_metrics(FakeUKSimulation(), "constituency")
    assert metrics["households"].tolist() == [1.0, 1.0, 1.0]


def test_census_family_and_pinned_sources() -> None:
    from populace.build.uk_runtime import build_uk_local_target_census

    census = build_uk_local_target_census()
    families = {row["family"]: row for row in census["families"]}
    assert "census_households" in families
    family = families["census_households"]
    source_ids = set(family["sources"])
    assert {
        "nomis_ts041_ew_oa_households",
        "nrs_census_2022_index",
        "nisra_dz21_households",
    } <= source_ids

    sources = {row["source_id"]: row for row in census["sources"]}
    for source_id in source_ids:
        assert sources[source_id]["status"] == "pinned_in_ladder"

    metrics = {row["name"]: row for row in census["metrics"]}
    assert metrics["households"]["family"] == "census_households"
    assert set(metrics["households"]["area_types"]) == {"constituency", "la"}
