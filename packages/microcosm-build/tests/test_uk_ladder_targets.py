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

from microcosm.build.uk_runtime import (
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
    from microcosm.build.uk_runtime import build_uk_local_target_census

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


def test_household_targets_refuse_missing_and_normalize_padded_codes(
    tmp_path,
) -> None:
    ladder = _ladder(tmp_path)

    class Doctored:
        households = ladder.households
        constituency_code = np.asarray(
            ["E14000001", None, "E14000002", "S14000001"], dtype=object
        )

    with pytest.raises(ValueError, match="missing"):
        constituency_household_targets(Doctored())

    class Padded:
        households = ladder.households
        constituency_code = np.asarray(
            ["E14000001", " E14000001 ", "E14000002", "S14000001"],
            dtype=object,
        )

    targets = constituency_household_targets(Padded())
    rows = dict(zip(targets["code"], targets["households"], strict=True))
    # Padded variants collapse into one area instead of splitting it.
    assert rows["E14000001"] == pytest.approx(55.0)


def test_household_targets_schema_is_stable_for_both_grains(tmp_path) -> None:
    ladder = _ladder(tmp_path)
    for targets in (
        constituency_household_targets(ladder),
        local_authority_household_targets(ladder),
    ):
        assert list(targets.columns) == ["code", "households"]
        assert targets["code"].tolist() == sorted(targets["code"].tolist())


def test_ladder_target_provenance_names_the_pairing(tmp_path) -> None:
    from microcosm.build.uk_runtime import ladder_target_provenance

    ladder = _ladder(tmp_path)
    provenance = ladder_target_provenance(ladder)
    assert provenance["kind"] == "uk_oa_ladder"
    assert provenance["coverage"] == "uk"
    assert provenance["layer_vintages"]["constituency"] == "2024_pcon"
    assert provenance["output_areas"] == 4
    assert provenance["households_total"] == pytest.approx(120.0)


def test_local_metric_surface_is_append_only() -> None:
    # Positional stability: adding a metric must never renumber the metrics
    # already in the surface, because consumers address them by index
    # (local_rowwise builds target_index as area_index * n_metrics +
    # metric_index). The ladder-derived "households" metric was itself
    # appended under this rule, so it keeps its index rather than staying
    # last: a later family lands after it, never before it.
    prefix_through_households = {
        "constituency": (
            "hmrc/self_employment_income/amount",
            "hmrc/self_employment_income/count",
            "hmrc/employment_income/amount",
            "hmrc/employment_income/count",
            "age/0_10",
            "age/10_20",
            "age/20_30",
            "age/30_40",
            "age/40_50",
            "age/50_60",
            "age/60_70",
            "age/70_80",
            "uc_households",
            "uc_hh_0_children",
            "uc_hh_1_child",
            "uc_hh_2_children",
            "uc_hh_3plus_children",
            "households",
        ),
        "la": (
            "hmrc/self_employment_income/amount",
            "hmrc/self_employment_income/count",
            "hmrc/employment_income/amount",
            "hmrc/employment_income/count",
            "age/0_10",
            "age/10_20",
            "age/20_30",
            "age/30_40",
            "age/40_50",
            "age/50_60",
            "age/60_70",
            "age/70_80",
            "uc_households",
            "ons/equiv_net_income_bhc",
            "ons/equiv_net_income_ahc",
            "ons/equiv_housing_costs",
            "tenure/owned_outright",
            "tenure/owned_mortgage",
            "tenure/private_rent",
            "tenure/social_rent",
            "rent/private_rent",
            "households",
        ),
    }
    for area_type, prefix in prefix_through_households.items():
        names = metric_names(area_type)
        assert names[: len(prefix)] == prefix, (
            f"{area_type} metric indices were renumbered; new metrics must be "
            "appended after the existing surface, never inserted into it."
        )
        assert len(set(names)) == len(names)


def test_census_disclosure_fence_gates_the_household_family() -> None:
    from microcosm.build.uk_runtime import build_uk_local_target_census

    census = build_uk_local_target_census()
    families = {row["family"]: row for row in census["families"]}
    assert (
        "census_disclosure_control_noise"
        in families["census_households"]["adjudications"]
    )
    fences = {row["fence_id"]: row for row in census["binding_fences"]}
    assert "cell-key" in fences["census_disclosure_control_noise"]["rule"]
