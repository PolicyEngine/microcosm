"""Rotated target holdout: every target held out exactly once.

A single train/holdout split of the target surface measures generalization at
one lucky (or unlucky) partition. The rotation fixes that: targets are dealt
into ``n_folds`` deterministic folds; the scorer runs ``n_folds`` evaluations,
each calibrating on all-but-one fold and scoring the held fold, so **every
target is scored exactly once as a holdout**. :func:`summarize_rotations`
aggregates the per-fold losses into the publishable numbers.

The fold structure is pure and deterministic (seeded permutation), so a
scorer on any machine reproduces the same rotation from ``(n_targets,
n_folds, seed)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["rotated_folds", "summarize_rotations", "RotationSummary"]


def rotated_folds(
    n_targets: int, *, n_folds: int = 5, seed: int = 20260529
) -> tuple[np.ndarray, ...]:
    """Deal target indices into ``n_folds`` deterministic folds.

    Args:
        n_targets: Number of targets to rotate.
        n_folds: Number of folds; each is one rotation's holdout set. Fold
            sizes differ by at most one.
        seed: Permutation seed. The default is the stack's published
            holdout-split seed, so single-split results remain reproducible
            as the first rotation of the family.

    Returns:
        ``n_folds`` sorted index arrays. Disjoint, covering
        ``range(n_targets)`` exactly.

    Raises:
        ValueError: If ``n_targets`` is not positive or ``n_folds`` is not
            in ``[2, n_targets]``.
    """
    if n_targets <= 0:
        raise ValueError(f"n_targets must be positive, got {n_targets!r}.")
    if not (2 <= n_folds <= n_targets):
        raise ValueError(
            f"n_folds must be between 2 and n_targets={n_targets}, got {n_folds!r}."
        )
    order = np.random.default_rng(seed).permutation(n_targets)
    return tuple(np.sort(order[fold::n_folds]) for fold in range(n_folds))


@dataclass(frozen=True)
class RotationSummary:
    """Aggregated rotated-holdout result.

    Attributes:
        n_folds: Number of rotations run.
        mean_holdout_loss: Mean of the per-fold holdout losses — the
            headline generalization number.
        worst_holdout_loss: The worst fold (a split-luck detector: a mean
            that hides one catastrophic fold is not generalization).
        fold_losses: The per-fold losses, in fold order.
    """

    n_folds: int
    mean_holdout_loss: float
    worst_holdout_loss: float
    fold_losses: tuple[float, ...]


def summarize_rotations(fold_losses: Iterable[float]) -> RotationSummary:
    """Aggregate per-fold holdout losses into a :class:`RotationSummary`.

    Args:
        fold_losses: One holdout loss per rotation (the loss on the held
            fold's targets, after calibrating on the rest).

    Raises:
        ValueError: On an empty or non-finite input — a rotation that
            produced no loss is a harness bug, not a zero.
    """
    losses: Sequence[float] = tuple(float(loss) for loss in fold_losses)
    if not losses:
        raise ValueError("fold_losses is empty; run at least one rotation.")
    if not np.isfinite(losses).all():
        raise ValueError(f"fold_losses contains non-finite values: {losses}.")
    return RotationSummary(
        n_folds=len(losses),
        mean_holdout_loss=float(np.mean(losses)),
        worst_holdout_loss=float(np.max(losses)),
        fold_losses=tuple(losses),
    )
