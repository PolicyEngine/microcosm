"""Name the thread counts of the US base and local-area build commands.

The commands call :func:`set_build_thread_environment` before numpy loads, so
this module uses only the standard library. Each variable gets the count its
library already uses when the variable is unset, for two reasons:

- On Linux the OpenBLAS thread count changes output bytes. OpenBLAS splits a
  large dot product between its threads, and the order of the additions moves
  the last bits of the sum. Keeping the count keeps the bytes.
- The base command records these variables in its run configuration and refuses
  a resume with different values, which it can only do once they have values.

So this limits nothing. OpenMP and BLAS counts have no upper limit, and they
follow CPU affinity but not a CPU quota. The one thread count that can change
is scikit-learn's: with ``OMP_NUM_THREADS`` unset it uses physical cores, and
with it set it uses that count.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

#: The most threads numexpr starts when nothing configures it.
NUMEXPR_UNCONFIGURED_THREAD_LIMIT = 16

__all__ = ["NUMEXPR_UNCONFIGURED_THREAD_LIMIT", "set_build_thread_environment"]


def _configured_count(value: str | None) -> str | None:
    """The thread count a setting names, or None when it names none.

    OpenMP allows a comma-separated list, one count per nesting level, and
    OpenBLAS reads the leading number. A value without a positive count leaves
    these libraries on their own default.
    """

    if value is None:
        return None
    first = value.split(",", 1)[0].strip()
    if not (first.isascii() and first.isdigit()) or int(first) < 1:
        return None
    return str(int(first))


def set_build_thread_environment(
    environment: MutableMapping[str, str] = os.environ,
) -> None:
    """Give each unset OpenMP, BLAS and numexpr variable its current count.

    The count is ``OMP_NUM_THREADS`` when that names one, otherwise the CPUs
    this process may use. Two variables have other sources, because their
    libraries do:

    - OpenBLAS reads ``GOTO_NUM_THREADS`` before ``OMP_NUM_THREADS``.
    - numexpr uses ``OMP_NUM_THREADS``, then ``NUMEXPR_MAX_THREADS``, and with
      neither it counts the online CPUs, up to 16. It needs its own value here
      because the ``OMP_NUM_THREADS`` set below would otherwise replace the
      last two.

    A value already present is kept. When the usable CPUs are unknown nothing
    is set, and each library keeps its own default.
    """

    openmp_count = _configured_count(environment.get("OMP_NUM_THREADS"))
    usable_cpus = os.process_cpu_count()
    if openmp_count is not None:
        count = openmp_count
    elif usable_cpus is not None:
        count = str(usable_cpus)
    else:
        return
    online_cpus = os.cpu_count() or int(count)
    defaults = {
        "OMP_NUM_THREADS": count,
        "OPENBLAS_NUM_THREADS": _configured_count(environment.get("GOTO_NUM_THREADS"))
        or count,
        "MKL_NUM_THREADS": count,
        "BLIS_NUM_THREADS": count,
        "VECLIB_MAXIMUM_THREADS": count,
        "NUMEXPR_NUM_THREADS": openmp_count
        or _configured_count(environment.get("NUMEXPR_MAX_THREADS"))
        or str(min(online_cpus, NUMEXPR_UNCONFIGURED_THREAD_LIMIT)),
    }
    for name, value in defaults.items():
        environment.setdefault(name, value)
