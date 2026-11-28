"""Values drawn inside a published band.

Business statistics publish counts by band (turnover, employment), not
values. A firm in band ``[lower, upper)`` gets a value inside its band:
uniform for closed bands, log-uniform for an open top band whose upper edge is
a declared cap. Both keep every value inside its source band.

The caller draws the uniforms, so the random stream stays with the caller:
:func:`draw_in_band` is a pure transform. It works on NumPy arrays and on
torch tensors alike and returns the type and dtype it was given, so a caller
that draws float32 uniforms from torch gets the same float32 values it
computed before this module existed.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["draw_in_band"]


def draw_in_band(
    unit: Any,
    lower: float,
    upper: float,
    *,
    log_uniform: bool = False,
    floor: float = 0.0,
) -> Any:
    """Map uniforms in ``[0, 1)`` to values in ``[max(lower, floor), upper)``.

    Args:
        unit: Uniform draws in ``[0, 1)``, a NumPy array or a torch tensor.
        lower: The band's lower edge.
        upper: The band's upper edge (an open band's declared cap).
        log_uniform: Draw log-uniformly, for an open top band. The low edge
            must then be positive.
        floor: The smallest value drawn, applied to ``lower``; a band whose
            lower edge is zero draws from ``floor`` instead.

    Returns:
        Values the same shape, type and dtype as ``unit``.
    """

    low = max(float(lower), float(floor))
    high = float(upper)
    if high < low:
        raise ValueError(f"band upper edge {high} lies below its low edge {low}")
    if log_uniform:
        if low <= 0:
            raise ValueError(f"a log-uniform band needs a positive low edge, got {low}")
        exponent = math.log(low) + unit * (math.log(high) - math.log(low))
        # torch tensors carry .exp(); NumPy arrays and scalars do not.
        return exponent.exp() if hasattr(exponent, "exp") else np.exp(exponent)
    return low + unit * (high - low)
