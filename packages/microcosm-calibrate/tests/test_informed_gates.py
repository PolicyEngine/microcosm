"""Informed L0 must preserve rare carriers without replacing the search."""

import numpy as np
import pytest
import torch
from scipy import sparse

from microcosm.calibrate.gates import HardConcrete
from microcosm.calibrate.initialization import contribution_initialization


def test_small_weight_large_measure_is_protected():
    init = contribution_initialization(
        sparse.csr_array([[1000.0, 0, 0], [0, 1, 1]]),
        np.array([0.01, 100.0, 100.0]),
        np.array([10.0, 200.0]),
    )
    assert init.protected[0]
    assert init.probabilities[0] > init.probabilities[1]


def test_protected_gates_stay_open_during_training_and_evaluation():
    gates = HardConcrete(
        3,
        initial_probabilities=np.array([0.2, 0.4, 0.8]),
        protected_mask=np.array([True, False, False]),
    )
    assert np.allclose(gates.get_active_prob().detach().numpy(), [1, 0.4, 0.8])
    with torch.no_grad():
        gates.qz_logits.fill_(-100)
    for training in (True, False):
        gates.train(training)
        assert gates()[0].item() == 1
    assert gates.get_active_prob()[0].item() == 1


@pytest.mark.parametrize(
    "probabilities", [[0, 0.5], [0.5, 1], [float("nan"), 0.5], [0.5]]
)
def test_invalid_initial_probabilities_refuse(probabilities):
    with pytest.raises(ValueError):
        HardConcrete(2, initial_probabilities=np.array(probabilities))
