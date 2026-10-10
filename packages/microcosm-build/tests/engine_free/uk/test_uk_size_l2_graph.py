"""Size-stage L2 penalties as graph configuration and node parameters (microcosm#1124)."""

from __future__ import annotations

import pytest

from microcosm.build.uk_runtime.dataset_size import uk_size_l2
from microcosm.build.uk_runtime.graph_calibration import (
    UKGraphCalibrationConfig,
    uk_calibration_nodes,
)

_COLUMNS = {("household", "marker"): "int64"}
_BASE_PARAMS = {
    "epochs",
    "learning_rate",
    "seed",
    "households",
    "pi_hi",
    "initial_lambda",
}


def _params(config):
    nodes = uk_calibration_nodes(
        base="pool", columns=_COLUMNS, problem_producer="pool", config=config
    ).nodes
    return {node.id.rsplit(".", 1)[-1]: dict(node.params) for node in nodes}


def test_default_size_nodes_keep_their_parameters() -> None:
    params = _params(UKGraphCalibrationConfig(epochs=2, dataset_households=2))
    assert set(params["size_search"]) == _BASE_PARAMS
    assert set(params["size_draw"]) == _BASE_PARAMS
    assert set(params["size_refit"]) == _BASE_PARAMS | {"baseline_pi_floor"}


def test_selection_l2_joins_the_shared_size_params_and_refit_l2_the_refit_only() -> (
    None
):
    params = _params(
        UKGraphCalibrationConfig(
            epochs=2,
            dataset_households=2,
            selection_l2=uk_size_l2("selection", l2_lambda=1e-3),
            refit_l2=uk_size_l2("refit", l2_lambda=0.03, anchor="uniform"),
        )
    )
    selection_keys = {
        "selection_l2_lambda",
        "selection_l2_anchor",
        "selection_l2_basis",
    }
    refit_keys = {"refit_l2_lambda", "refit_l2_anchor", "refit_l2_basis"}
    for node in ("size_search", "size_draw", "size_refit"):
        assert selection_keys <= set(params[node])
    assert not refit_keys & set(params["size_search"])
    assert not refit_keys & set(params["size_draw"])
    assert params["size_refit"]["refit_l2_lambda"] == 0.03
    assert params["size_refit"]["refit_l2_anchor"] == "uniform"


def test_config_refuses_l2_without_a_size_or_on_the_wrong_stage() -> None:
    with pytest.raises(ValueError, match="requires a dataset size"):
        UKGraphCalibrationConfig(refit_l2=uk_size_l2("refit", l2_lambda=0.1))
    with pytest.raises(ValueError, match="selection-stage UKSizeL2"):
        UKGraphCalibrationConfig(
            dataset_households=2, selection_l2=uk_size_l2("refit", l2_lambda=0.1)
        )
    with pytest.raises(ValueError, match="Unknown target_weight_rule"):
        UKGraphCalibrationConfig(target_weight_rule="family_weird")
