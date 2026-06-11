"""populace.build: stage plans and acceptance gates for dataset builds.

A build is a typed :class:`~populace.build.plan.StagePlan` over a
:class:`~populace.frame.Frame` — each stage declares what it consumes, what it
produces, and (for imputations) which donor survey it draws from; the executor
materializes stages in order and fails loudly on any donor it cannot load (no
silent fallbacks, per the charter). The product of a build is only publishable
once it passes the :mod:`populace.build.gates` suite: parity, support,
aggregate-vs-admin, per-family fit — plus the :mod:`populace.build.holdout`
rotation for scoring.

Importing this shard asserts compatibility with the installed
:mod:`populace.frame` kernel — the constellation mechanism from DESIGN.md.
"""

from populace.frame import __version__ as _frame_version

#: The populace-frame series this shard is built against (pre-1.0: minor-level
#: compatibility; see the identical gate in populace-calibrate).
_REQUIRED_FRAME_SERIES = (0, 1)


def _assert_frame_compatible(version: str, required: tuple[int, int]) -> None:
    """Raise unless the installed populace-frame is the expected series."""
    parts = version.split(".")
    try:
        installed = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError):  # pragma: no cover - defensive
        raise ImportError(
            f"populace-build cannot parse populace-frame version "
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
            f"populace-build requires populace-frame {expected}, but "
            f"{version} is installed. Install the matching constellation "
            "(the workspace releases the shards in lockstep): upgrade or pin "
            f"populace-frame to {expected}."
        )


_assert_frame_compatible(_frame_version, _REQUIRED_FRAME_SERIES)

from populace.build.gates import (  # noqa: E402 - after the compat gate
    GateReport,
    GateResult,
    aggregate_admin_gate,
    exported_nonzero_gate,
    parity_gate,
    per_family_fit_gate,
    relative_error_loss,
    support_gate,
)
from populace.build.holdout import (  # noqa: E402 - after the compat gate
    rotated_folds,
    summarize_rotations,
)
from populace.build.plan import (  # noqa: E402 - after the compat gate
    DonorSpec,
    Stage,
    StagePlan,
    StageRecord,
)

__version__ = "0.1.0"

__all__ = [
    "DonorSpec",
    "GateReport",
    "GateResult",
    "Stage",
    "StagePlan",
    "StageRecord",
    "aggregate_admin_gate",
    "exported_nonzero_gate",
    "parity_gate",
    "per_family_fit_gate",
    "relative_error_loss",
    "rotated_folds",
    "summarize_rotations",
    "support_gate",
    "__version__",
]
