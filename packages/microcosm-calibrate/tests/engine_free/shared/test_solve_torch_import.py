"""Importing the calibration package does not import torch.

Torch is imported by the solver functions that use it, so a process that only
imports ``microcosm.calibrate`` (every US base step process and QRF worker
does, through ``microcosm.build``) does not pay for importing it.
"""

from __future__ import annotations

import subprocess
import sys

_IMPORT_PROBE = """
import sys

import microcosm.calibrate
import microcosm.calibrate.solve

print(sorted(name for name in ("torch", "microcosm.calibrate.gates") if name in sys.modules))
"""


def test_importing_the_calibration_package_does_not_import_torch() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[-4000:]
    assert completed.stdout.strip() == "[]"


_FIRST_USE_PROBE = """
import sys

import numpy as np
from scipy import sparse

from microcosm.calibrate import solve

assert "torch" not in sys.modules
matrix = solve._torch_constraint_matrix(sparse.csr_matrix(np.eye(2)))
print(type(matrix).__module__, type(matrix).__name__, "torch" in sys.modules)
"""


def test_a_solver_function_imports_torch_on_first_use() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _FIRST_USE_PROBE],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr[-4000:]
    assert completed.stdout.strip() == "torch Tensor True"
