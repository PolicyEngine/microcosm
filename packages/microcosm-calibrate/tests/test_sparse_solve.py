"""The sparse SpMM path: same answers as dense, engaged only when it pays.

The solver must never *require* densifying the constraint matrix — at national
scale (3M+ records) the dense tensor is tens of GB. These tests pin: (1) the
automatic policy (small/dense problems stay dense; large sparse ones go
sparse-CSR), (2) numerical equivalence of the two paths on the same problem,
and (3) the options record builds use for release manifests (including the
max_weight_ratio bound and the realized matrix format).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import sparse

from microcosm.calibrate import Target, TargetSet, calibrate
from microcosm.calibrate.solve import (
    _SPARSE_MIN_CELLS,
    _apply_constraint,
    _torch_constraint_matrix,
)


def _targets(truths: dict[str, float], factor: float) -> TargetSet:
    return TargetSet(
        (
            Target(
                name="population",
                entity="household",
                value=truths["population"] * factor,
                measure="household_count",
            ),
            Target(
                name="income",
                entity="household",
                value=truths["income"] * factor,
                measure="income",
            ),
        )
    )


class TestMatrixPolicy:
    def test_small_problem_stays_dense(self) -> None:
        matrix = sparse.csr_array(np.eye(4))
        out = _torch_constraint_matrix(matrix)
        assert out.layout == torch.strided

    def test_large_sparse_problem_goes_csr(self) -> None:
        n = _SPARSE_MIN_CELLS // 100 + 1  # 100 x n crosses the cell floor
        matrix = sparse.random(100, n, density=0.01, format="csr", random_state=0)
        out = _torch_constraint_matrix(sparse.csr_array(matrix))
        assert out.layout == torch.sparse_csr
        assert out.shape == (100, n)

    def test_large_dense_problem_stays_dense(self) -> None:
        n = _SPARSE_MIN_CELLS // 100 + 1
        rng = np.random.default_rng(0)
        matrix = sparse.csr_array(rng.random((100, n)))  # density ~1
        out = _torch_constraint_matrix(matrix)
        assert out.layout == torch.strided

    def test_apply_constraint_matches_dense_matmul(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.random((6, 9))
        a[a < 0.6] = 0.0
        w = rng.random(9)
        dense = torch.tensor(a, dtype=torch.float32)
        csr = _torch_constraint_matrix(sparse.csr_array(a))
        # force the sparse layout regardless of the size policy
        if csr.layout != torch.sparse_csr:
            csr = dense.to_sparse_csr()
        w_t = torch.tensor(w, dtype=torch.float32)
        np.testing.assert_allclose(
            _apply_constraint(csr, w_t).numpy(),
            _apply_constraint(dense, w_t).numpy(),
            rtol=1e-5,
        )

    def test_sparse_spmm_autograds_to_weights(self) -> None:
        a = torch.tensor(
            [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=torch.float32
        ).to_sparse_csr()
        w = torch.ones(3, requires_grad=True)
        out = _apply_constraint(a, w).sum()
        out.backward()
        np.testing.assert_allclose(w.grad.numpy(), [1.0, 3.0, 2.0])


class TestPathEquivalence:
    def test_sparse_and_dense_paths_agree(self, feasible_frame, monkeypatch) -> None:
        """The same problem solved on both paths lands on the same weights."""
        frame, truths = feasible_frame(n=120)
        targets = _targets(truths, 1.3)

        result_dense = calibrate(frame, targets, epochs=150, seed=0)
        assert result_dense.options["matrix_format"] == "dense"

        # Drop the size floor and density cutoff so the same small problem
        # takes the sparse path (its count/sum rows are fully dense).
        monkeypatch.setattr("microcosm.calibrate.solve._SPARSE_MIN_CELLS", 0)
        monkeypatch.setattr("microcosm.calibrate.solve._SPARSE_DENSITY_CUTOFF", 1.0)
        result_sparse = calibrate(frame, targets, epochs=150, seed=0)
        assert result_sparse.options["matrix_format"] == "sparse_csr"

        np.testing.assert_allclose(
            result_sparse.weights, result_dense.weights, rtol=1e-4
        )
        assert result_sparse.final_loss < result_sparse.initial_loss * 1e-2

    def test_sparse_path_respects_ratio_bound(
        self, landmine_frame, monkeypatch
    ) -> None:
        frame, donor_index, donor_value = landmine_frame()
        targets = TargetSet(
            (
                Target(
                    name="capital_gains",
                    entity="household",
                    value=donor_value * 50.0,
                    measure="capital_gains",
                ),
            )
        )
        monkeypatch.setattr("microcosm.calibrate.solve._SPARSE_MIN_CELLS", 0)
        monkeypatch.setattr("microcosm.calibrate.solve._SPARSE_DENSITY_CUTOFF", 1.0)
        result = calibrate(frame, targets, epochs=200, seed=0, max_weight_ratio=5.0)
        assert result.options["matrix_format"] == "sparse_csr"
        assert result.weights[donor_index] <= 5.0 * 1.0 + 1e-6


class TestOptionsRecord:
    def test_options_capture_manifest_fields(self, feasible_frame) -> None:
        frame, truths = feasible_frame(n=60)
        targets = _targets(truths, 1.1)
        result = calibrate(
            frame,
            targets,
            epochs=50,
            learning_rate=0.2,
            mass="conserve",
            max_weight_ratio=50.0,
            l2_lambda=0.001,
            seed=7,
        )
        opts = result.options
        assert opts["max_weight_ratio"] == 50.0
        assert opts["l2_lambda"] == 0.001
        assert opts["l2_penalty"] == "mean_initial_pre_gate_weight_ratio_squared"
        assert opts["mass"] == "conserve"
        assert opts["epochs"] == 50
        assert opts["learning_rate"] == 0.2
        assert opts["seed"] == 7
        assert opts["matrix_format"] in ("dense", "sparse_csr")
