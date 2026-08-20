"""The imputation lineage spec is the source of truth; the code must conform.

``specs/us_imputation_lineage.yaml`` declares, per target family, which
predictor set and which model draw every imputed variable, plus the model's
attributes. These tests hold the running code to that declaration so a
predictor set, target list, or model default can never change without the
spec (and therefore the dashboard) changing with it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from microcosm.build.us_runtime.acs_transfer import (
    ACS_GROUP_TRANSFER_PREDICTORS,
    ACS_OPTIONAL_PERSON_TRANSFER_PREDICTORS,
    ACS_PERSON_TRANSFER_PREDICTORS,
    DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
)
from microcosm.build.us_runtime.post_transfer_calibration import (
    post_transfer_calibration_policy_identity,
)
from microcosm.build.us_runtime.puf_support import PUF_TAX_DETAIL_DEFAULT_PREDICTORS
from microcosm.build.us_runtime.stacked_spine import stacked_gap_fill_plan
from microcosm.build.us_runtime.us_late_producer_registry import (
    CANONICAL_US_LATE_PRODUCER_REGISTRY,
    CANONICAL_US_LATE_TRANSFER_GROUPS,
)
from microcosm.fit.qrf import DEFAULT_N_ESTIMATORS, DEFAULT_ZERO_ATOL

SPEC_PATH = Path(__file__).resolve().parents[3] / "specs" / "us_imputation_lineage.yaml"


@pytest.fixture(scope="module")
def spec() -> dict:
    return yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))


def _clean(name: str) -> str:
    return name.replace("__acs_transfer_", "")


def test_spec_declares_every_stage_family_and_target(spec: dict) -> None:
    declared = {f["id"]: f for f in spec["imputed_families"]}

    live: dict[str, list[str]] = {}
    for direction in stacked_gap_fill_plan():
        for entity, families in dict(direction.target_families).items():
            for family, targets in families.items():
                live[f"gap_fill/{direction.name}/{entity}/{family}"] = list(targets)
    for group in CANONICAL_US_LATE_TRANSFER_GROUPS:
        live[f"late_transfer/{group.name}"] = list(group.targets)

    assert set(declared) == set(live), (
        "spec families != live families; regenerate or amend the spec: "
        f"missing={sorted(set(live) - set(declared))} "
        f"stale={sorted(set(declared) - set(live))}"
    )
    for family_id, targets in live.items():
        assert declared[family_id]["targets"] == targets, family_id


def test_every_imputed_family_names_a_declared_predictor_set_and_model(
    spec: dict,
) -> None:
    sets = spec["predictor_sets"]
    models = spec["models"]
    for family in spec["imputed_families"]:
        assert family["predictor_set"] in sets, family["id"]
        assert family["model"] in models, family["id"]


def test_predictor_sets_match_the_code(spec: dict) -> None:
    sets = spec["predictor_sets"]
    assert sets["acs_person_transfer"]["required"] == [
        _clean(c) for c in ACS_PERSON_TRANSFER_PREDICTORS
    ]
    assert sets["acs_person_transfer"]["optional"] == [
        _clean(c) for c in ACS_OPTIONAL_PERSON_TRANSFER_PREDICTORS
    ]
    assert sets["acs_group_transfer"]["required"] == [
        _clean(c) for c in ACS_GROUP_TRANSFER_PREDICTORS
    ]
    assert sets["puf_tax_detail"]["required"] == list(PUF_TAX_DETAIL_DEFAULT_PREDICTORS)


def test_model_attributes_match_the_code(spec: dict) -> None:
    model = spec["models"]["regime_gated_qrf"]
    assert model["n_estimators"] == DEFAULT_N_ESTIMATORS
    assert model["zero_atol"] == DEFAULT_ZERO_ATOL
    assert model["implementation"] == "microcosm.fit.qrf.RegimeGatedQRF"
    for family in spec["imputed_families"]:
        if family["stage"] == "early_gap_fill":
            assert (
                family["max_targets_per_fit"]
                == DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
            ), family["id"]


def test_post_draw_calibration_policy_matches_the_code(spec: dict) -> None:
    declared = spec["models"]["regime_gated_qrf"]["post_draw_calibration"]
    assert declared == post_transfer_calibration_policy_identity()


def test_computed_producers_match_the_registry(spec: dict) -> None:
    declared = {p["producer"]: p for p in spec["computed_producers"]}
    live = {
        name: producer
        for name, producer in CANONICAL_US_LATE_PRODUCER_REGISTRY.items()
        if str(producer.kind) != "late_transfer"
    }
    assert set(declared) == set(live)
    for name, producer in live.items():
        assert declared[name]["outputs"] == [
            getattr(o, "column", None) or getattr(o, "name", None) or str(o)
            for o in producer.outputs
        ], name


def test_every_target_appears_exactly_once(spec: dict) -> None:
    seen: dict[str, str] = {}
    for family in spec["imputed_families"]:
        for target in family["targets"]:
            key = f"{family['entity']}/{target}"
            assert key not in seen, (
                f"{key} declared by both {seen[key]} and {family['id']}"
            )
            seen[key] = family["id"]
    assert len(seen) == 118
