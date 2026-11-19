"""The one thread setting of the US build commands, ``MICROCOSM_THREADS``.

An operator who wants to set the thread count of the US release, base, ACS
staging and ACS local-release commands sets ``MICROCOSM_THREADS`` to a positive
whole number. Each command calls :func:`set_build_thread_environment` before
numpy loads, because the libraries read their variables when they load, so this
module uses only the standard library. It copies the count into every variable
in ``FILLED_VARIABLES`` that is unset or blank. A variable that already has a
value keeps it.

Without the setting, the base, ACS staging and ACS local-release commands set
nothing, and each library chooses its own count as it always has, until the
ACS local-release command imports the release tool. The release tool passes
its own default instead, ``min(os.cpu_count(), 16)``, and also applies it when
imported.

On Linux the OpenBLAS thread count can change output bytes: OpenBLAS splits a
large dot product between its threads, and the order of the additions moves
the last bits of the sum. So a build's output can match one made without the
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
    # read it when their own variable is unset. Once it is set, scikit-learn
    # stops limiting itself to physical cores, a CPU quota and
    # LOKY_MAX_CPU_COUNT.
    "OMP_NUM_THREADS",
    # OpenBLAS, which numpy and scipy bundle on Linux. It reads this before
    # its older name, GOTO_NUM_THREADS, and uses no more threads than the CPUs
    # it may run on.
    "OPENBLAS_NUM_THREADS",
    # Intel MKL, which torch's Linux x86-64 builds include. torch also takes
    # its own thread count from it, ahead of OMP_NUM_THREADS.
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

#: numexpr's maximum thread count when ``NUMEXPR_MAX_THREADS`` is unset. It
#: refuses a larger count and runs one thread, so a larger count fills that
#: variable too.
NUMEXPR_DEFAULT_MAXIMUM = 64
NUMEXPR_MAXIMUM_VARIABLE = "NUMEXPR_MAX_THREADS"

__all__ = [
    "FILLED_VARIABLES",
    "NUMEXPR_DEFAULT_MAXIMUM",
    "NUMEXPR_MAXIMUM_VARIABLE",
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
    if value.isascii() and value.isdigit():
        try:
            count = int(value)
        except ValueError:  # longer than int() will parse
            count = 0
        if count > 0:
            return count
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
    already has a value keeps it. The thread setting also replaces a blank
    value: most libraries ignore one, and joblib fails on a blank
    ``LOKY_MAX_CPU_COUNT``. ``default_count`` keeps a blank value, as the release
    tool's default always has.
    """

    count = thread_setting(environment)
    replaces_blank_values = count is not None
    if count is None:
        count = default_count
    if count is None:
        return
    names = list(FILLED_VARIABLES)
    if count > NUMEXPR_DEFAULT_MAXIMUM:
        names.append(NUMEXPR_MAXIMUM_VARIABLE)
    for name in names:
        current = environment.get(name)
        if current is None or (replaces_blank_values and current == ""):
            environment[name] = str(count)
