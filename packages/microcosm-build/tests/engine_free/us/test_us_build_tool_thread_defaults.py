"""The one thread setting of the US build commands, ``MICROCOSM_THREADS``.

When it is set, the release, base and ACS staging commands copy it into every
library's thread variable before numpy loads. When it is not, the base and ACS
staging commands set nothing, so each library keeps choosing its own count,
and the release tool keeps its default of ``min(os.cpu_count(), 16)``.

Each process-level case starts a fresh interpreter: the settings depend on what
the process has loaded and must not reach the process that runs the tests.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from types import ModuleType

import pytest

from test_support.paths import paths_for

REPOSITORY = paths_for("microcosm-build").repository
HELPER_PATH = REPOSITORY / "tools" / "us_build_thread_environment.py"
RELEASE_TOOL = "tools/build_us_fiscal_refresh_release.py"
BASE_TOOL = "tools/build_us_puf_support_base.py"
ACS_STAGING_COMMAND = "tools/build_us_acs_multispine_base.py"
ACS_STAGING_IMPLEMENTATION = "tools/_legacy/build_us_acs_multispine_base.py"
BUILD_COMMANDS = (BASE_TOOL, ACS_STAGING_COMMAND, ACS_STAGING_IMPLEMENTATION)
EVERY_COMMAND = (*BUILD_COMMANDS, RELEASE_TOOL)

THREAD_SETTING = "MICROCOSM_THREADS"
OPENMP = "OMP_NUM_THREADS"
OPENBLAS = "OPENBLAS_NUM_THREADS"
MKL = "MKL_NUM_THREADS"
BLIS = "BLIS_NUM_THREADS"
VECLIB = "VECLIB_MAXIMUM_THREADS"
NUMEXPR = "NUMEXPR_NUM_THREADS"
JOBLIB_LIMIT = "LOKY_MAX_CPU_COUNT"
FILLED_VARIABLES = (OPENMP, OPENBLAS, MKL, BLIS, VECLIB, NUMEXPR, JOBLIB_LIMIT)
QRF_VARIABLES = ("POPULACE_FIT_N_JOBS", "POPULACE_FIT_PREDICT_WORKERS")
#: Every variable a process-level case clears before applying its own.
THREAD_VARIABLES = (
    THREAD_SETTING,
    *FILLED_VARIABLES,
    "GOTO_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    *QRF_VARIABLES,
)


def _clean_environment(settings: Mapping[str, str] = {}) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in THREAD_VARIABLES
    }
    environment.update(settings)
    return environment


def _run(command: list[str], environment: dict[str, str]):
    return subprocess.run(
        command,
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _last_json_line(command: list[str], environment: dict[str, str]) -> dict:
    completed = _run(command, environment)
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _every_variable(count: str, **replacements: str) -> dict[str, str]:
    return {**dict.fromkeys(FILLED_VARIABLES, count), **replacements}


# ---------------------------------------------------------------------------
# The values chosen
# ---------------------------------------------------------------------------


@pytest.fixture
def helper() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "us_build_thread_environment_under_test", HELPER_PATH
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _filled(
    helper: ModuleType,
    settings: Mapping[str, str] = {},
    *,
    default_count: int | None = None,
) -> dict[str, str]:
    environment = dict(settings)
    helper.set_build_thread_environment(environment, default_count=default_count)
    return environment


def test_nothing_is_set_without_the_thread_setting(helper: ModuleType) -> None:
    settings = {OPENBLAS: "3", "UNRELATED": "value"}

    assert _filled(helper) == {}
    assert _filled(helper, settings) == settings


def test_a_blank_thread_setting_counts_as_unset(helper: ModuleType) -> None:
    assert _filled(helper, {THREAD_SETTING: ""}) == {THREAD_SETTING: ""}


def test_the_thread_setting_fills_every_thread_variable(helper: ModuleType) -> None:
    environment = _filled(helper, {THREAD_SETTING: "4"})

    assert environment == {THREAD_SETTING: "4", **_every_variable("4")}


def test_the_thread_setting_has_no_upper_limit(helper: ModuleType) -> None:
    environment = _filled(helper, {THREAD_SETTING: "64"}, default_count=16)

    assert environment == {THREAD_SETTING: "64", **_every_variable("64")}


def test_leading_zeros_are_dropped_from_the_copies(helper: ModuleType) -> None:
    environment = _filled(helper, {THREAD_SETTING: "007"})

    assert environment == {THREAD_SETTING: "007", **_every_variable("7")}


def test_variables_already_set_are_kept(helper: ModuleType) -> None:
    settings = {THREAD_SETTING: "4", OPENMP: "2", OPENBLAS: "5"}

    assert _filled(helper, settings) == {
        THREAD_SETTING: "4",
        **_every_variable("4", **{OPENMP: "2", OPENBLAS: "5"}),
    }


def test_a_blank_thread_variable_is_filled(helper: ModuleType) -> None:
    """Libraries read a blank thread variable as unset."""

    environment = _filled(helper, {THREAD_SETTING: "4", OPENMP: ""})

    assert environment == {THREAD_SETTING: "4", **_every_variable("4")}


@pytest.mark.parametrize(
    "configured", ("0", "-1", "+4", "4,2", " 3 ", "4.0", "many", "٣")
)
def test_a_thread_setting_that_is_not_a_positive_count_is_refused(
    helper: ModuleType, configured: str
) -> None:
    environment = {THREAD_SETTING: configured}

    with pytest.raises(ValueError, match=THREAD_SETTING):
        helper.set_build_thread_environment(environment, default_count=16)

    assert environment == {THREAD_SETTING: configured}


def test_the_default_count_applies_without_the_thread_setting(
    helper: ModuleType,
) -> None:
    assert _filled(helper, default_count=16) == _every_variable("16")


def test_the_thread_setting_replaces_the_default_count(helper: ModuleType) -> None:
    environment = _filled(helper, {THREAD_SETTING: "3"}, default_count=16)

    assert environment == {THREAD_SETTING: "3", **_every_variable("3")}


def test_a_second_call_changes_nothing(helper: ModuleType) -> None:
    environment = _filled(helper, {THREAD_SETTING: "4"})
    first_result = dict(environment)

    helper.set_build_thread_environment(environment, default_count=16)

    assert environment == first_result


# ---------------------------------------------------------------------------
# When the commands apply them
# ---------------------------------------------------------------------------

#: Starts one tool under ``runpy`` and stops it at its first import of a
#: numerical library, printing the environment and import path at that moment.
#: The tool's imports never finish, so a case costs no more than the code
#: before them.
_START_PROBE = """
import json
import os
import runpy
import sys
from pathlib import Path


class ReachedNumericalLibrary(BaseException):
    pass


class StopAtNumericalLibrary:
    watched = {"joblib", "numexpr", "numpy", "pandas", "scipy", "sklearn", "torch"}

    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in self.watched:
            raise ReachedNumericalLibrary(name)
        return None


tool = Path(sys.argv[1]).resolve()
run_name = sys.argv[2]
cpu_count = int(sys.argv[3])
os.cpu_count = lambda: cpu_count
os.process_cpu_count = lambda: cpu_count
sys.path.insert(0, str(tool.parent))
path_before = list(sys.path)
sys.argv = [str(tool), "--help"]
sys.meta_path.insert(0, StopAtNumericalLibrary())
library = None
try:
    runpy.run_path(str(tool), run_name=run_name)
except ReachedNumericalLibrary as reached:
    library = str(reached)
print()
print(
    json.dumps(
        {
            "library": library,
            "environment": dict(os.environ),
            "path_unchanged": sys.path == path_before,
        }
    )
)
"""


def _probe_command(tool: str, run_name: str, cpu_count: int) -> list[str]:
    return [sys.executable, "-c", _START_PROBE, tool, run_name, str(cpu_count)]


def _state_at_first_numerical_import(
    tool: str,
    *,
    run_name: str = "__main__",
    cpu_count: int = 3,
    settings: Mapping[str, str] = {},
) -> dict:
    state = _last_json_line(
        _probe_command(tool, run_name, cpu_count), _clean_environment(settings)
    )
    assert state["library"] is not None
    return state


def _thread_variables_in(environment: Mapping[str, str]) -> dict[str, str]:
    return {name: environment[name] for name in THREAD_VARIABLES if name in environment}


@pytest.mark.parametrize("tool", BUILD_COMMANDS)
def test_a_started_command_sets_nothing_without_the_thread_setting(
    tool: str,
) -> None:
    environment = _state_at_first_numerical_import(tool)["environment"]

    assert _thread_variables_in(environment) == {}


@pytest.mark.parametrize("tool", EVERY_COMMAND)
def test_a_started_command_fills_every_variable_before_numerical_imports(
    tool: str,
) -> None:
    environment = _state_at_first_numerical_import(
        tool, cpu_count=32, settings={THREAD_SETTING: "5"}
    )["environment"]

    assert _thread_variables_in(environment) == {
        THREAD_SETTING: "5",
        **_every_variable("5"),
    }


@pytest.mark.parametrize("tool", EVERY_COMMAND)
def test_a_started_command_keeps_the_values_it_was_given(tool: str) -> None:
    settings = {THREAD_SETTING: "5", OPENMP: "2", OPENBLAS: "7"}

    environment = _state_at_first_numerical_import(tool, settings=settings)[
        "environment"
    ]

    assert _thread_variables_in(environment) == {
        THREAD_SETTING: "5",
        **_every_variable("5", **{OPENMP: "2", OPENBLAS: "7"}),
    }


@pytest.mark.parametrize("tool", EVERY_COMMAND)
def test_a_started_command_refuses_an_unusable_thread_setting(tool: str) -> None:
    completed = _run(
        _probe_command(tool, "__main__", 3),
        _clean_environment({THREAD_SETTING: "four"}),
    )

    assert completed.returncode != 0
    assert THREAD_SETTING in completed.stderr
    assert "ReachedNumericalLibrary" not in completed.stderr


@pytest.mark.parametrize("tool", BUILD_COMMANDS)
def test_importing_a_command_changes_no_thread_variable(tool: str) -> None:
    """Tests and other tools import these files, often after numpy has loaded."""

    environment = _state_at_first_numerical_import(
        tool, run_name="imported_tool", settings={THREAD_SETTING: "5"}
    )["environment"]

    assert _thread_variables_in(environment) == {THREAD_SETTING: "5"}


@pytest.mark.parametrize("tool", EVERY_COMMAND)
def test_no_command_sets_the_qrf_variables(tool: str) -> None:
    environment = _state_at_first_numerical_import(
        tool, settings={THREAD_SETTING: "5"}
    )["environment"]

    assert not set(QRF_VARIABLES) & set(environment)


@pytest.mark.parametrize(("cpu_count", "expected"), ((3, "3"), (32, "16")))
def test_release_tool_keeps_its_default_without_the_thread_setting(
    cpu_count: int, expected: str
) -> None:
    """Its existing six variables keep their values; BLIS has no library here."""

    state = _state_at_first_numerical_import(RELEASE_TOOL, cpu_count=cpu_count)

    assert _thread_variables_in(state["environment"]) == _every_variable(expected)
    assert state["path_unchanged"]


def test_importing_the_release_tool_still_applies_its_default() -> None:
    """It always has: its tests import it, and the values only fill unset ones."""

    state = _state_at_first_numerical_import(
        RELEASE_TOOL, run_name="imported_tool", cpu_count=32
    )

    assert _thread_variables_in(state["environment"]) == _every_variable("16")
    assert state["path_unchanged"]


# ---------------------------------------------------------------------------
# What the libraries do with them
# ---------------------------------------------------------------------------

#: Applies the commands' settings, loads every library whose thread count they
#: choose, and prints the counts each library ended with.
_THREAD_COUNT_PROBE = """
import json
import sys

sys.path.insert(0, "tools")
from us_build_thread_environment import set_build_thread_environment

set_build_thread_environment()

import joblib
import numexpr
import numpy
import scipy.linalg
import threadpoolctl
import torch
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

print()
print(
    json.dumps(
        {
            "thread_pools": sorted(
                {pool["num_threads"] for pool in threadpoolctl.threadpool_info()}
            ),
            "numexpr": numexpr.utils.get_num_threads(),
            "scikit_learn": _openmp_effective_n_threads(),
            "joblib_cpu_count": joblib.cpu_count(),
            "torch": torch.get_num_threads(),
        }
    )
)
"""


def test_the_thread_setting_reaches_every_library() -> None:
    counts = _last_json_line(
        [sys.executable, "-c", _THREAD_COUNT_PROBE],
        _clean_environment({THREAD_SETTING: "2"}),
    )

    assert counts == {
        "thread_pools": [2],
        "numexpr": 2,
        "scikit_learn": 2,
        "joblib_cpu_count": 2,
        "torch": 2,
    }


_SIGN_CLASSIFIER_PROBE = """
import hashlib
import json

import numpy as np

from microcosm.fit.qrf import _make_gate

generator = np.random.default_rng(0)
features = generator.lognormal(0.0, 1.5, size=(30_000, 6))
latent = features[:, 0] * generator.lognormal(0.0, 1.0, 30_000) - features[:, 1]
labels = np.sign(np.where(np.abs(latent) < 0.3, 0.0, latent)).astype(int)
weights = generator.lognormal(0.0, 2.0, 30_000)
classifier = _make_gate(1).fit(features, labels, sample_weight=weights)
probabilities = classifier.predict_proba(features[:10_000])
print()
print(json.dumps({"digest": hashlib.sha256(probabilities.tobytes()).hexdigest()}))
"""


def test_sign_classifier_probabilities_do_not_depend_on_the_openmp_count() -> None:
    """The thread setting changes the classifier's OpenMP count."""

    digests = {
        thread_count: _last_json_line(
            [sys.executable, "-c", _SIGN_CLASSIFIER_PROBE],
            _clean_environment({OPENMP: thread_count, OPENBLAS: "1"}),
        )["digest"]
        for thread_count in ("1", "2", "4")
    }

    assert len(set(digests.values())) == 1, digests
