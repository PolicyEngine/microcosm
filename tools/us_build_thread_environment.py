"""Name the thread counts of the US base and ACS staging build commands.

The commands call :func:`set_build_thread_environment` before numpy loads, so
this module uses only the standard library. Each BLAS and numexpr variable gets
the count its library already uses when the variable is unset:

- On Linux the OpenBLAS thread count changes output bytes. OpenBLAS splits a
  large dot product between its threads, and the order of the additions moves
  the last bits of the sum. Keeping the count keeps the bytes.
- The base command records these variables in its run configuration and compares
  them on resume. With real counts recorded, it refuses a resume on a host with a
  different count, which two unset values could not show.

This limits nothing: the counts have no upper limit, and OpenMP and BLAS follow
CPU affinity but not a CPU quota. Setting ``OMP_NUM_THREADS`` does change two
other counts. scikit-learn then uses that value instead of its physical-core
count limited by a CPU quota and ``LOKY_MAX_CPU_COUNT``, and torch uses it too.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

#: The most threads numexpr starts when nothing configures it.
NUMEXPR_UNCONFIGURED_THREAD_LIMIT = 16

__all__ = ["NUMEXPR_UNCONFIGURED_THREAD_LIMIT", "set_build_thread_environment"]


def _count_or_default(setting: str | None, default: str | None) -> str | None:
    """The count to copy from ``setting``, or ``default`` if it is unset or blank.

    Only a plain positive integer is copied. Libraries read other spellings
    differently (OpenBLAS takes ``4`` from ``4,2`` and numexpr refuses it), so
    for those this returns None and the variable that would copy the value stays
    unset, leaving each library to read ``setting`` as it does today.
    """

    if not setting:
        return default
    if setting.isascii() and setting.isdigit() and int(setting) > 0:
        return str(int(setting))
    return None


def set_build_thread_environment(
    environment: MutableMapping[str, str] = os.environ,
) -> None:
    """Give each unset OpenMP, BLAS and numexpr variable its current count.

    The count is ``OMP_NUM_THREADS`` when that is a plain count, otherwise the
    CPUs this process may use. Two variables have other sources, because their
    libraries do:

    - OpenBLAS reads ``GOTO_NUM_THREADS`` before ``OMP_NUM_THREADS``.
    - numexpr uses ``OMP_NUM_THREADS``, then ``NUMEXPR_MAX_THREADS``, and with
      neither it counts the online CPUs, up to 16. It needs its own value here
      because the ``OMP_NUM_THREADS`` set below would otherwise replace the
      last two.

    A value already present is kept. A variable with no known count stays unset,
    and its library keeps its own default.
    """

    usable_cpus = os.process_cpu_count()
    usable_count = None if usable_cpus is None else str(usable_cpus)
    openmp_setting = environment.get("OMP_NUM_THREADS")
    count = _count_or_default(openmp_setting, usable_count)
    online_cpus = os.cpu_count() or usable_cpus
    numexpr_default = (
        None
        if online_cpus is None
        else str(min(online_cpus, NUMEXPR_UNCONFIGURED_THREAD_LIMIT))
    )
    defaults = {
        "OMP_NUM_THREADS": count,
        "OPENBLAS_NUM_THREADS": _count_or_default(
            environment.get("GOTO_NUM_THREADS"), count
        ),
        "MKL_NUM_THREADS": count,
        "BLIS_NUM_THREADS": count,
        "VECLIB_MAXIMUM_THREADS": count or usable_count,
        "NUMEXPR_NUM_THREADS": _count_or_default(
            openmp_setting,
            _count_or_default(environment.get("NUMEXPR_MAX_THREADS"), numexpr_default),
        ),
    }
    for name, value in defaults.items():
        if value is not None:
            environment.setdefault(name, value)
