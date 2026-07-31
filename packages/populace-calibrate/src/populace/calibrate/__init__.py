"""populace.calibrate: the representation operator of the populace stack.

The only place :class:`~populace.frame.WeightKind.CALIBRATED` weights are
produced. Compiles declared facts as sum targets — including count-like facts
represented by prepared indicator/count columns — into a sparse linear
constraint system over a :class:`~populace.frame.Frame`, then solves for the
weight vector that best reproduces them under capped weighted MAPE,
``weighted_mean(min(abs((A @ w - b) / scale), cap))``. The default
``method="adam"`` path uses torch Adam over log-weights (positivity by
construction); ``method="prox"`` uses proximal gradient over non-negative weight
ratios for L1 selection. Multi-period targets stack as ``(target, period)`` rows
over the *same* weight vector — the charter's "one weight per trajectory".

Load-bearing options beyond the fit include: ``mass`` ("free" or "conserve") to
control the total; ``max_weight_ratio`` as a hard per-record bound (the guard
against tail "landmine" records detonating on reweight); ``target_records`` for
hard-concrete L0 pruning with budget control — the solver searches
``l0_lambda`` so the achieved non-zero count tracks the budget (the
generate-big-then-prune path); ``l1_lambda`` with ``method="prox"`` as a
proximal selection penalty; and experimental ``l2_lambda`` as a soft
concentration penalty. Under L0 gates, ``l2_lambda`` penalizes latent pre-gate
weights so a nearly closed gate cannot hide an exploding underlying weight.
``l0_lambda`` alone prunes at a fixed penalty.

Importing this shard asserts compatibility with the installed
:mod:`populace.frame` kernel — the constellation mechanism from DESIGN.md: a
shard pins ``populace-frame`` in its metadata *and* checks the kernel major at
import, so a resolver that ignores ``[tool.uv.sources]`` cannot silently
assemble an incompatible pair.
"""

from populace.frame import __version__ as _frame_version

#: The populace-frame major this shard is built against. The kernel is
#: pre-1.0, so during the 0.x line compatibility is pinned at the *minor* level
#: (0.x and 0.y may differ incompatibly); from 1.0 on this becomes the major.
#: Kept in lockstep with the ``populace-frame>=...`` floor in ``pyproject.toml``.
_REQUIRED_FRAME_SERIES = (0, 1)


def _assert_frame_compatible(version: str, required: tuple[int, int]) -> None:
    """Raise unless the installed populace-frame is the expected series.

    Args:
        version: The installed ``populace.frame.__version__``.
        required: The ``(major, minor)`` series this shard requires. The minor
            is enforced only while the major is ``0`` (the pre-1.0 convention
            that 0.x minors may break compatibility); from major ``1`` on, only
            the major must match.

    Raises:
        ImportError: If the installed kernel is outside the required series. The
            message names both versions and the fix.
    """
    parts = version.split(".")
    try:
        installed = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):  # pragma: no cover - defensive
        raise ImportError(
            f"populace-calibrate cannot parse populace-frame version "
            f"{version!r}; expected a {required[0]}.{required[1]}.x kernel."
        ) from None

    if required[0] == 0:
        compatible = installed == required
        expected = f"{required[0]}.{required[1]}.x"
    else:
        compatible = installed[0] == required[0]
        expected = f"{required[0]}.x"

    if not compatible:
        raise ImportError(
            f"populace-calibrate requires populace-frame {expected}, but "
            f"{version} is installed. Install the matching constellation "
            "(the workspace releases the shards in lockstep): upgrade or pin "
            f"populace-frame to {expected}."
        )


_assert_frame_compatible(_frame_version, _REQUIRED_FRAME_SERIES)

from populace.calibrate.diagnostics import (  # noqa: E402 - after the compat gate
    CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
    diagnostics_payload,
    past_cap_census,
    write_calibration_diagnostics,
)
from populace.calibrate.exact_k import (  # noqa: E402 - after the compat gate
    assert_exact_k_support,
    select_exact_k,
)
from populace.calibrate.matrix import (  # noqa: E402 - after the compat gate
    CalibrationProblem,
    SkippedTarget,
    build_constraint_matrix,
)
from populace.calibrate.registry import (  # noqa: E402 - after the compat gate
    TargetRegistry,
    TargetSpec,
    specs_from_pe_surface,
)
from populace.calibrate.score import score_targets  # noqa: E402 - after the compat gate
from populace.calibrate.solve import (  # noqa: E402 - after the compat gate
    CONSERVE_MASS,
    FREE_MASS,
    CalibrationResult,
    L0RefitResult,
    TargetDiagnostic,
    calibrate,
    calibrate_l0_refit,
    default_target_loss_scales,
    effective_sample_size,
    refit_l0_selection,
    relative_error_loss,
)
from populace.calibrate.target import (  # noqa: E402 - after the compat gate
    Target,
    TargetSet,
)

__version__ = "0.1.0"

__all__ = [
    "CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION",
    "CONSERVE_MASS",
    "FREE_MASS",
    "CalibrationProblem",
    "CalibrationResult",
    "L0RefitResult",
    "SkippedTarget",
    "Target",
    "TargetDiagnostic",
    "TargetRegistry",
    "TargetSet",
    "TargetSpec",
    "build_constraint_matrix",
    "calibrate",
    "calibrate_l0_refit",
    "assert_exact_k_support",
    "default_target_loss_scales",
    "effective_sample_size",
    "refit_l0_selection",
    "diagnostics_payload",
    "past_cap_census",
    "relative_error_loss",
    "score_targets",
    "select_exact_k",
    "specs_from_pe_surface",
    "write_calibration_diagnostics",
    "__version__",
]
