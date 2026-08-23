"""The dashboard and migration gates consume the one US imputation bundle."""

from __future__ import annotations

from collections import Counter

import pytest

from microcosm.build.spec_engine.typed_closure import compile_producer_outputs
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
from tools.emit_lineage_dashboard import emit
from tools.us_bundle_generation.imputation import build_imputation

pytest.importorskip(
    "policyengine_us",
    reason="live-engine oracle: the wheels gate's venv installs no engine",
    exc_type=ModuleNotFoundError,
)


@pytest.fixture(scope="module")
def spec() -> dict[str, object]:
    return build_imputation()


def test_bundle_declares_every_legacy_transfer_family_and_target(
    spec: dict[str, object],
) -> None:
    declared = {family["id"]: family for family in spec["families"]}
    live: dict[str, list[str]] = {}
    for direction in stacked_gap_fill_plan():
        for entity, families in dict(direction.target_families).items():
            for family, targets in families.items():
                live[f"early/{direction.name}/{entity}/{family}"] = list(targets)
    for group in CANONICAL_US_LATE_TRANSFER_GROUPS:
        live[f"late/{group.entity}/{group.family}"] = list(group.targets)

    assert set(live) <= set(declared)
    for family_id, targets in live.items():
        assert [row["name"] for row in declared[family_id]["targets"]] == targets


def test_every_imputed_family_names_declared_predictors_and_model(
    spec: dict[str, object],
) -> None:
    blocks = spec["predictor_blocks"]
    models = spec["models"]
    for family in spec["families"]:
        assert family["model"] in models, family["id"]
        assert family["predictors"], family["id"]
        assert set(family["predictors"]) <= set(blocks), family["id"]


def test_predictor_blocks_match_constants_era_oracle(spec: dict[str, object]) -> None:
    blocks = spec["predictor_blocks"]
    assert blocks["acs_person_required"]["columns"] == list(
        ACS_PERSON_TRANSFER_PREDICTORS
    )
    assert blocks["acs_person_optional_income"]["columns"] + blocks[
        "acs_person_optional_structure"
    ]["columns"] == list(ACS_OPTIONAL_PERSON_TRANSFER_PREDICTORS)
    assert blocks["acs_group_required"]["columns"] == list(
        ACS_GROUP_TRANSFER_PREDICTORS
    )
    assert blocks["puf_tax_detail"]["columns"] == list(
        PUF_TAX_DETAIL_DEFAULT_PREDICTORS
    )


def test_model_and_split_attributes_match_constants_era_oracle(
    spec: dict[str, object],
) -> None:
    params = spec["models"]["regime_gated_qrf"]["params"]
    assert params["n_estimators"] == DEFAULT_N_ESTIMATORS
    assert params["zero_atol"] == DEFAULT_ZERO_ATOL
    for family in spec["families"]:
        if family["stage"] == "gap_fill_stacked_spine":
            assert (
                family["max_targets_per_fit"]
                == DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
            ), family["id"]


def test_post_draw_calibration_policy_matches_the_code(
    spec: dict[str, object],
) -> None:
    declared = spec["models"]["regime_gated_qrf"]["post_draw_calibration"]
    assert declared == post_transfer_calibration_policy_identity()


def test_non_transfer_producers_match_constants_era_registry(
    spec: dict[str, object],
) -> None:
    compiled_outputs = compile_producer_outputs({"imputation": spec})
    declared = {
        node["name"]: node
        for node in spec["producer_graph"]["nodes"]
        if node["kind"] != "late_transfer"
    }
    live = {
        name: producer
        for name, producer in CANONICAL_US_LATE_PRODUCER_REGISTRY.items()
        if str(producer.kind) != "late_transfer"
    }
    assert set(declared) == set(live)
    for name, producer in live.items():
        assert [row["column"] for row in compiled_outputs[name]] == [
            getattr(output, "column", None)
            or getattr(output, "name", None)
            or str(output)
            for output in producer.outputs
        ], name


def test_complete_bundle_family_inventory(spec: dict[str, object]) -> None:
    counts = Counter(family["stage"] for family in spec["families"])
    targets = Counter(
        family["stage"] for family in spec["families"] for _target in family["targets"]
    )
    assert counts == {
        "gap_fill_stacked_spine": 13,
        "primary_puf_qrf": 1,
        "late_producer_dag": 19,
    }
    assert targets == {
        "gap_fill_stacked_spine": 48,
        "primary_puf_qrf": 65,
        "late_producer_dag": 70,
    }
    for family in spec["families"]:
        keys = [(target["entity"], target["name"]) for target in family["targets"]]
        assert len(keys) == len(set(keys)), family["id"]


def test_dashboard_compiles_the_packaged_bundle_without_a_mirror_spec() -> None:
    payload = emit()
    assert payload["spec_binding"]["spec_sha256"] == payload["spec_sha256"]
    assert payload["counts"] == {
        "imputed_variables": 183,
        "families": 33,
        "computed_producers": 38,
        "boolean": 45,
        "amount": 132,
        "categorical": 5,
        "value_kinds": {"amount": 132, "category": 5, "count": 1, "flag": 45},
    }
    assert len(payload["known_gaps"]) == 1
