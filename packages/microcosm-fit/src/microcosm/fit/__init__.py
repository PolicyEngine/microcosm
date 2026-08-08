"""microcosm.fit: the conditional-models operator of the microcosm stack.

Fits conditional distributions ``P(targets | predictors)`` over a
:class:`~microcosm.frame.Frame` — or a plain :class:`pandas.DataFrame` — and
draws from them. The operator is **weight-aware by construction**: a Frame fit
reads the frame's typed weights, a DataFrame fit requires the weights
explicitly (a weight column name or vector), and on either input the only way
to fit unweighted is to pass ``weights="none"`` and mean it
(:mod:`microcosm.fit.model`). The canonical model is a regime-gated, chained,
weighted-bootstrap quantile-regression-forest imputer (:mod:`microcosm.fit.qrf`).

Importing this shard asserts compatibility with the installed
:mod:`microcosm.frame` kernel — the constellation mechanism from DESIGN.md: a
shard pins ``microcosm-frame`` in its metadata *and* checks the kernel major at
import, so a resolver that ignores ``[tool.uv.sources]`` cannot silently
assemble an incompatible pair.
"""

from microcosm.frame import __version__ as _frame_version

#: The microcosm-frame major this shard is built against. The kernel is
#: pre-1.0, so during the 0.x line compatibility is pinned at the *minor*
#: level (0.x and 0.y may differ incompatibly); from 1.0 on this becomes the
#: major. Kept in lockstep with the ``microcosm-frame>=...`` floor in
#: ``pyproject.toml``.
_REQUIRED_FRAME_SERIES = (0, 1)


def _assert_frame_compatible(version: str, required: tuple[int, int]) -> None:
    """Raise unless the installed microcosm-frame is the expected series.

    Args:
        version: The installed ``microcosm.frame.__version__``.
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
            f"microcosm-fit cannot parse microcosm-frame version {version!r}; "
            f"expected a {required[0]}.{required[1]}.x kernel."
        ) from None

    if required[0] == 0:
        compatible = installed == required
        expected = f"{required[0]}.{required[1]}.x"
    else:
        compatible = installed[0] == required[0]
        expected = f"{required[0]}.x"

    if not compatible:
        raise ImportError(
            f"microcosm-fit requires microcosm-frame {expected}, but "
            f"{version} is installed. Install the matching constellation "
            "(the workspace releases the shards in lockstep): upgrade or pin "
            f"microcosm-frame to {expected}."
        )


_assert_frame_compatible(_frame_version, _REQUIRED_FRAME_SERIES)

from microcosm.fit.model import (  # noqa: E402 - after the compatibility gate
    DESIGN_WEIGHTS,
    EXPLICIT_WEIGHTS,
    NO_WEIGHTS,
    ConditionalModel,
    FittedModel,
    WeightSpec,
    resolved_weight_kind,
)
from microcosm.fit.qrf import (  # noqa: E402 - after the compatibility gate
    DEFAULT_N_ESTIMATORS,
    DEFAULT_ZERO_ATOL,
    FittedRegimeGatedQRF,
    QRFChainState,
    QRFChainStepResult,
    Regime,
    RegimeGatedQRF,
)

__version__ = "0.1.0"

#: The canonical conditional model, under its short public name.
QRF = RegimeGatedQRF


def fit(
    frame_or_df,
    predictors: list[str],
    targets: list[str],
    *,
    weights: WeightSpec = DESIGN_WEIGHTS,
    **model_kwargs,
) -> FittedModel:
    """Fit the canonical conditional model over ``frame_or_df``.

    Convenience constructor: builds a :class:`RegimeGatedQRF` with
    ``model_kwargs`` and fits it. For a different model, instantiate it directly
    and call its :meth:`~microcosm.fit.model.ConditionalModel.fit`.

    Args:
        frame_or_df: The :class:`~microcosm.frame.Frame` or
            :class:`pandas.DataFrame` to fit on.
        predictors: Conditioning variable names (one entity / one table).
        targets: Variable names to learn the conditional of (same entity /
            same table).
        weights: On a Frame: which typed weight vector to weight the fit by;
            defaults to the owning entity's ``"design"`` weights. On a
            DataFrame: required — a weight column name or a weight vector.
            ``"none"`` fits unweighted — the only way to do so.
        **model_kwargs: Forwarded to :class:`RegimeGatedQRF` (e.g.
            ``n_estimators``, ``zero_atol``, ``seed``).

    Returns:
        A :class:`~microcosm.fit.model.FittedModel`.
    """
    return RegimeGatedQRF(**model_kwargs).fit(
        frame_or_df, predictors, targets, weights=weights
    )


__all__ = [
    "ConditionalModel",
    "FittedModel",
    "WeightSpec",
    "DESIGN_WEIGHTS",
    "NO_WEIGHTS",
    "EXPLICIT_WEIGHTS",
    "resolved_weight_kind",
    "QRF",
    "RegimeGatedQRF",
    "FittedRegimeGatedQRF",
    "QRFChainState",
    "QRFChainStepResult",
    "Regime",
    "DEFAULT_N_ESTIMATORS",
    "DEFAULT_ZERO_ATOL",
    "fit",
    "__version__",
]
