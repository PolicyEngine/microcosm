"""The one thread setting of the US build commands, ``MICROCOSM_THREADS``.

An operator who wants to choose the thread count of the US release, base and
ACS staging commands sets ``MICROCOSM_THREADS`` to a positive whole number.
Each command calls :func:`set_build_thread_environment` before numpy loads,
because the libraries read their variables when they load, so this module uses
only the standard library. It copies the count into every variable in
``FILLED_VARIABLES`` that is unset or blank. A variable that already has a
value keeps it.

Without the setting, the base and ACS staging commands set nothing, and each
library chooses its own count as it always has. The release tool passes its
own default instead, ``min(os.cpu_count(), 16)``.

On Linux the OpenBLAS thread count changes output bytes: OpenBLAS splits a
large dot product between its threads, and the order of the additions moves
the last bits of the sum. So a build's output matches one made without the
setting only when the count equals the CPUs OpenBLAS would otherwise use.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping

#: The variable that holds the one thread count.
THREAD_SETTING = "MICROCOSM_THREADS"

#: The variables the thread setting fills, with what reads each one.
FILLED_VARIABLES = (
    # OpenMP: scikit-learn's sign classifier and torch. OpenBLAS and numexpr
    # read it when their own variable is unset. scikit-learn stops limiting
    # itself to physical cores, a CPU quota and LOKY_MAX_CPU_COUNT once it is
    # set.
    "OMP_NUM_THREADS",
    # OpenBLAS, which numpy and scipy bundle on Linux. It reads this before
    # its older name, GOTO_NUM_THREADS.
    "OPENBLAS_NUM_THREADS",
    # Intel MKL, which only torch bundles here. torch also takes its own
    # thread count from it.
    "MKL_NUM_THREADS",
    # BLIS, which nothing installed here uses.
    "BLIS_NUM_THREADS",
    # Apple's Accelerate, which numpy uses on macOS.
    "VECLIB_MAXIMUM_THREADS",
    # numexpr, which pandas uses for large element-wise arithmetic.
    "NUMEXPR_NUM_THREADS",
    # The limit on joblib.cpu_count(), which sizes the QRF's tree-building
    # jobs and prediction workers when POPULACE_FIT_* are unset.
    "LOKY_MAX_CPU_COUNT",
)

__all__ = [
    "FILLED_VARIABLES",
    "THREAD_SETTING",
    "set_build_thread_environment",
    "thread_setting",
]


def thread_setting(environment: Mapping[str, str] = os.environ) -> int | None:
    """The count in ``MICROCOSM_THREADS``, or None if it is unset or blank.

    Only a plain positive whole number is accepted. Anything else is refused
    rather than ignored, because the operator asked for a count.
    """

    value = environment.get(THREAD_SETTING, "")
    if value == "":
        return None
    if value.isascii() and value.isdigit() and int(value) > 0:
        return int(value)
    raise ValueError(
        f"{THREAD_SETTING} must be a positive whole number of threads, got {value!r}"
    )


def set_build_thread_environment(
    environment: MutableMapping[str, str] = os.environ,
    *,
    default_count: int | None = None,
) -> None:
    """Copy the thread setting, or ``default_count``, into each unset variable.

    With neither, nothing changes. A variable in ``FILLED_VARIABLES`` that
    already has a non-blank value keeps it.
    """

    count = thread_setting(environment)
    if count is None:
        count = default_count
    if count is None:
        return
    for name in FILLED_VARIABLES:
        if not environment.get(name):
            environment[name] = str(count)
