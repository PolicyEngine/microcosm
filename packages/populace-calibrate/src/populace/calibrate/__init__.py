"""populace.calibrate: the representation operator of the populace stack.

The only place :class:`~populace.frame.WeightKind.CALIBRATED` weights are
produced. Compiles declared facts — population control totals, counts, averages
with standard-error-style tolerances — into a sparse linear constraint system
over a :class:`~populace.frame.Frame`, then solves for the weight vector that
best reproduces them under the eCPS relative-error loss
``mean(((A @ w - b + 1)/(b + 1))**2)``, optimized with torch's Adam over the
log-weights (positivity by construction). Multi-period targets stack as
``(target, period)`` rows over the *same* weight vector — the charter's "one
weight per trajectory".

Three load-bearing options beyond the fit: ``mass`` ("free" or "conserve") to
control the total; ``max_weight_ratio`` as a hard per-record bound (the guard
against tail "landmine" records detonating on reweight); and ``target_records``
for hard-concrete L0 pruning with budget control — the solver searches
``l0_lambda`` so the achieved non-zero count tracks the budget (the
generate-big-then-prune path). ``l0_lambda`` alone prunes at a fixed penalty.

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

from populace.calibrate.matrix import (  # noqa: E402 - after the compat gate
    CalibrationProblem,
    SkippedTarget,
    build_constraint_matrix,
)
from populace.calibrate.solve import (  # noqa: E402 - after the compat gate
    CONSERVE_MASS,
    FREE_MASS,
    CalibrationResult,
    TargetDiagnostic,
    calibrate,
)
from populace.calibrate.target import (  # noqa: E402 - after the compat gate
    AGGREGATIONS,
    Target,
    TargetSet,
)

__version__ = "0.1.0"

__all__ = [
    "AGGREGATIONS",
    "CONSERVE_MASS",
    "FREE_MASS",
    "CalibrationProblem",
    "CalibrationResult",
    "SkippedTarget",
    "Target",
    "TargetDiagnostic",
    "TargetSet",
    "build_constraint_matrix",
    "calibrate",
    "__version__",
]
