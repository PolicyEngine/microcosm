"""Hard-concrete L0 gates: the generate-big-then-prune mechanism.

A :class:`HardConcrete` module carries one stochastic gate per record, each a
relaxed Bernoulli (the hard-concrete distribution of Louizos et al., 2018) whose
expected L0 "norm" — the expected number of open gates — is differentiable. The
solver multiplies the record weights by these gates and adds
``l0_lambda * penalty`` to the loss; raising ``l0_lambda`` drives gates shut,
pruning the pool toward fewer non-zero weights. This is the core calibration
path: generate big, then prune to ``target_records``.

The math is the reference one verbatim: a sigmoid-of-logistic-noise stretched to
``[gamma, zeta]`` and clamped to ``[0, 1]`` in training (so gates can hit hard 0
/ 1), the sigmoid of the logits stretched the same way in eval, and the L0
penalty ``sum(sigmoid(logits - temperature*log(-gamma/zeta)))`` — the closed-form
probability a gate is open.
"""

from __future__ import annotations

import math

import torch
from torch import nn

__all__ = ["HardConcrete", "hard_concrete_open_probability_threshold"]

#: Stretch bounds for the hard-concrete distribution (Louizos et al., 2018).
#: ``gamma < 0 < 1 < zeta`` lets the stretched-then-clamped variable reach exact
#: 0 and 1, which is what makes the gate able to prune hard.
_GAMMA = -0.1
_ZETA = 1.1


def hard_concrete_open_probability_threshold(temperature: float) -> float:
    """Return the open-probability cutoff for a positive deterministic gate.

    The evaluation-time hard-concrete gate is positive exactly when its logit
    exceeds ``log(-gamma / zeta)``. Expressed on the per-record open
    probability returned by :meth:`HardConcrete.get_active_prob`, that same
    boundary is

    ``sigmoid((1 - temperature) * log(-gamma / zeta))``.

    This helper describes the gate threshold itself. A calibration result's
    legacy support additionally applies its weight-scale pruning tolerance, so
    callers should not assume equivalence for arbitrarily tiny latent weights.
    """
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            f"temperature must be positive and finite, got {temperature!r}."
        )
    value = (1.0 - float(temperature)) * math.log(-_GAMMA / _ZETA)
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


class HardConcrete(nn.Module):
    """One hard-concrete L0 gate per record, with a differentiable L0 penalty.

    Args:
        n: Number of gates (one per record of the calibrated entity).
        init_mean: Initial expected open-probability of every gate (e.g.
            ``0.999`` to start with essentially all records active and let the
            penalty close them). Must lie in ``(0, 1)``.
        temperature: Concrete-distribution temperature; lower is closer to a
            hard Bernoulli.

    Raises:
        ValueError: If ``init_mean`` is not strictly between 0 and 1, or
            ``temperature`` is not positive.
    """

    def __init__(
        self,
        n: int,
        *,
        init_mean: float = 0.999,
        temperature: float = 0.25,
    ) -> None:
        super().__init__()
        if not (0.0 < init_mean < 1.0):
            raise ValueError(
                f"HardConcrete.init_mean must lie in (0, 1), got {init_mean!r}."
            )
        if temperature <= 0:
            raise ValueError(
                f"HardConcrete.temperature must be positive, got {temperature!r}."
            )
        self.temperature = float(temperature)
        self.gamma = _GAMMA
        self.zeta = _ZETA
        self.qz_logits = nn.Parameter(torch.zeros(n))
        init_val = math.log(init_mean / (1.0 - init_mean))
        with torch.no_grad():
            self.qz_logits.fill_(init_val)

    def forward(self) -> torch.Tensor:
        """Return the current gates: sampled in training, deterministic in eval."""
        if self.training:
            u = torch.zeros_like(self.qz_logits).uniform_(1e-8, 1.0 - 1e-8)
            s = torch.log(u) - torch.log(1.0 - u) + self.qz_logits
            s = torch.sigmoid(s / self.temperature)
        else:
            s = torch.sigmoid(self.qz_logits)
        stretched = s * (self.zeta - self.gamma) + self.gamma
        return torch.clamp(stretched, 0.0, 1.0)

    def get_penalty(self) -> torch.Tensor:
        """Expected number of open gates — the differentiable L0 surrogate.

        Returns:
            ``sum(sigmoid(logits - temperature*log(-gamma/zeta)))``: the
            closed-form expected count of open gates, which the solver scales by
            ``l0_lambda`` and adds to the loss.
        """
        shift = self.temperature * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.qz_logits - shift).sum()

    def get_active_prob(self) -> torch.Tensor:
        """Per-gate open-probability (the per-record version of the penalty)."""
        shift = self.temperature * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.qz_logits - shift)
