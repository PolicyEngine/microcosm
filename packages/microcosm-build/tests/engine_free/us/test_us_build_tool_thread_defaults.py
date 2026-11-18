"""The thread settings the US base and local-area commands give themselves.

The commands name the thread count each numerical library already uses, so
that it is recorded and is in place before the libraries load. On Linux the
OpenBLAS thread count changes the bytes of large dot products, so these tests
compare real library behaviour with and without the settings.

Each process-level case starts a fresh interpreter: the settings depend on what
the process has loaded and must not reach the process that runs the tests.
"""

from __future__ import annotations

import functools
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
LOCAL_AREA_COMMAND = "tools/build_us_acs_multispine_base.py"
LOCAL_AREA_IMPLEMENTATION = "tools/_legacy/build_us_acs_multispine_base.py"
BUILD_COMMANDS = (BASE_TOOL, LOCAL_AREA_COMMAND, LOCAL_AREA_IMPLEMENTATION)

OPENMP = "OMP_NUM_THREADS"
OPENBLAS = "OPENBLAS_NUM_THREADS"
OPENBLAS_FALLBACK = "GOTO_NUM_THREADS"
MKL = "MKL_NUM_THREADS"
BLIS = "BLIS_NUM_THREADS"
VECLIB = "VECLIB_MAXIMUM_THREADS"
NUMEXPR = "NUMEXPR_NUM_THREADS"
NUMEXPR_MAXIMUM = "NUMEXPR_MAX_THREADS"
JOBLIB_LIMIT = "LOKY_MAX_CPU_COUNT"
QRF_VARIABLES = ("POPULACE_FIT_N_JOBS", "POPULACE_FIT_PREDICT_WORKERS")
SET_BY_COMMANDS = (OPENMP, OPENBLAS, MKL, BLIS, VECLIB, NUMEXPR)
#: Every variable a process-level case clears before applying its own.
THREAD_VARIABLES = (
    *SET_BY_COMMANDS,
    OPENBLAS_FALLBACK,
    NUMEXPR_MAXIMUM,
    JOBLIB_LIMIT,
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


def _last_json_line(command: list[str], environment: dict[str, str]) -> dict:
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


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


def _defaults_for(
    helper: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    settings: Mapping[str, str] = {},
    *,
    usable_cpus: int | None = 6,
    online_cpus: int | None = 6,
) -> dict[str, str]:
    monkeypatch.setattr(helper.os, "process_cpu_count", lambda: usable_cpus)
    monkeypatch.setattr(helper.os, "cpu_count", lambda: online_cpus)
    environment = dict(settings)
    helper.set_build_thread_environment(environment)
    return environment


def _every_variable(count: str, **replacements: str) -> dict[str, str]:
    return {**dict.fromkeys(SET_BY_COMMANDS, count), **replacements}


def test_unset_variables_take_the_usable_cpu_count(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _defaults_for(helper, monkeypatch) == _every_variable("6")


def test_openmp_and_blas_counts_have_no_upper_limit(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """numexpr alone limits itself to 16 threads when nothing configures it."""

    environment = _defaults_for(helper, monkeypatch, usable_cpus=32, online_cpus=32)

    assert environment == _every_variable("32", **{NUMEXPR: "16"})


def test_cpu_affinity_sets_the_count_and_numexpr_counts_online_cpus(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _defaults_for(helper, monkeypatch, usable_cpus=4, online_cpus=10)

    assert environment == _every_variable("4", **{NUMEXPR: "10"})


@pytest.mark.parametrize(
    ("configured", "expected"),
    (("2", "2"), ("4,2", "4"), (" 3 ", "3"), ("007", "7")),
)
def test_an_openmp_count_already_set_is_used_for_the_others(
    helper: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    expected: str,
) -> None:
    environment = _defaults_for(helper, monkeypatch, {OPENMP: configured})

    assert environment == _every_variable(expected, **{OPENMP: configured})


@pytest.mark.parametrize("configured", ("", "0", "-1", "many", "٣"))
def test_an_openmp_value_without_a_count_is_kept_and_not_copied(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch, configured: str
) -> None:
    environment = _defaults_for(helper, monkeypatch, {OPENMP: configured})

    assert environment == _every_variable("6", **{OPENMP: configured})


def test_values_already_set_are_kept(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = {name: str(index + 1) for index, name in enumerate(SET_BY_COMMANDS)}

    assert _defaults_for(helper, monkeypatch, settings) == settings


def test_openblas_follows_its_own_fallback_variable(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _defaults_for(
        helper, monkeypatch, {OPENMP: "2", OPENBLAS_FALLBACK: "3"}
    )

    assert environment == {
        **_every_variable("2", **{OPENBLAS: "3"}),
        OPENBLAS_FALLBACK: "3",
    }


@pytest.mark.parametrize(
    ("settings", "expected"),
    (
        ({NUMEXPR_MAXIMUM: "3"}, "3"),
        ({NUMEXPR_MAXIMUM: "3", OPENMP: "2"}, "2"),
    ),
)
def test_numexpr_follows_openmp_and_then_its_maximum(
    helper: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    settings: dict[str, str],
    expected: str,
) -> None:
    environment = _defaults_for(helper, monkeypatch, settings)

    assert environment[NUMEXPR] == expected


def test_nothing_is_set_when_the_usable_cpus_are_unknown(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _defaults_for(helper, monkeypatch, usable_cpus=None) == {}


def test_numexpr_uses_the_usable_count_when_online_cpus_are_unknown(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _defaults_for(helper, monkeypatch, usable_cpus=20, online_cpus=None)

    assert environment == _every_variable("20", **{NUMEXPR: "16"})


def test_a_second_call_changes_nothing(
    helper: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _defaults_for(helper, monkeypatch, usable_cpus=32, online_cpus=32)
    first_result = dict(environment)

    helper.set_build_thread_environment(environment)

    assert environment == first_result


# ---------------------------------------------------------------------------
# When the commands apply them
# ---------------------------------------------------------------------------

#: Starts one tool under ``runpy`` and stops it at its first import of a
#: numerical library, printing the environment at that moment. The tool's
#: imports never finish, so a case costs no more than the code before them.
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
sys.argv = [str(tool), "--help"]
sys.meta_path.insert(0, StopAtNumericalLibrary())
library = None
try:
    runpy.run_path(str(tool), run_name=run_name)
except ReachedNumericalLibrary as reached:
    library = str(reached)
print()
print(json.dumps({"library": library, "environment": dict(os.environ)}))
"""


def _state_at_first_numerical_import(
    tool: str,
    *,
    run_name: str = "__main__",
    cpu_count: int = 3,
    settings: Mapping[str, str] = {},
) -> dict:
    state = _last_json_line(
        [sys.executable, "-c", _START_PROBE, tool, run_name, str(cpu_count)],
        _clean_environment(settings),
    )
    assert state["library"] is not None
    return state


@pytest.mark.parametrize("tool", BUILD_COMMANDS)
def test_a_started_command_sets_every_variable_before_numerical_imports(
    tool: str,
) -> None:
    environment = _state_at_first_numerical_import(tool)["environment"]

    assert {name: environment.get(name) for name in SET_BY_COMMANDS} == (
        _every_variable("3", **{NUMEXPR: "3"})
    )


@pytest.mark.parametrize("tool", BUILD_COMMANDS)
def test_a_started_command_keeps_the_values_it_was_given(tool: str) -> None:
    settings = {OPENMP: "2", OPENBLAS: "5"}

    environment = _state_at_first_numerical_import(tool, settings=settings)[
        "environment"
    ]

    assert {name: environment.get(name) for name in SET_BY_COMMANDS} == (
        _every_variable("2", **{OPENBLAS: "5"})
    )


@pytest.mark.parametrize("tool", BUILD_COMMANDS)
def test_importing_a_command_changes_no_thread_variable(tool: str) -> None:
    """Tests and other tools import these files, often after numpy has loaded."""

    environment = _state_at_first_numerical_import(tool, run_name="imported_tool")[
        "environment"
    ]

    assert not set(THREAD_VARIABLES) & set(environment)


@pytest.mark.parametrize("tool", (*BUILD_COMMANDS, RELEASE_TOOL))
def test_no_command_sets_the_qrf_or_joblib_limits_of_the_base_build(
    tool: str,
) -> None:
    environment = _state_at_first_numerical_import(tool)["environment"]

    assert not set(QRF_VARIABLES) & set(environment)
    if tool != RELEASE_TOOL:
        assert JOBLIB_LIMIT not in environment


@pytest.mark.parametrize(("cpu_count", "expected"), ((3, "3"), (32, "16")))
def test_release_tool_keeps_its_existing_thread_values(
    cpu_count: int, expected: str
) -> None:
    environment = _state_at_first_numerical_import(RELEASE_TOOL, cpu_count=cpu_count)[
        "environment"
    ]

    for name in (OPENMP, OPENBLAS, MKL, NUMEXPR, VECLIB, JOBLIB_LIMIT):
        assert environment[name] == expected, name
    assert BLIS not in environment


# ---------------------------------------------------------------------------
# What the libraries do with them
# ---------------------------------------------------------------------------

#: Loads the libraries, with or without the commands' settings, and prints the
#: thread counts they chose and results of the BLAS operations whose bytes
#: depend on the OpenBLAS thread count on Linux.
_LIBRARY_PROBE = """
import hashlib
import json
import sys

sys.path.insert(0, "tools")
if sys.argv[1] == "with_settings":
    from us_build_thread_environment import set_build_thread_environment

    set_build_thread_environment()

import numexpr
import numpy as np
import sklearn.utils._openmp_helpers
import threadpoolctl

generator = np.random.default_rng(0)
weights = generator.lognormal(7.0, 1.0, 150_000)
values = generator.lognormal(8.0, 1.5, 150_000)
probabilities = generator.dirichlet(np.ones(3), 150_000)
scaled_column = values * (9.6e11 / float(weights @ values))
print()
print(
    json.dumps(
        {
            "pools": sorted(
                [pool["user_api"], pool["internal_api"], pool["num_threads"]]
                for pool in threadpoolctl.threadpool_info()
            ),
            "numexpr_threads": numexpr.utils.get_num_threads(),
            "dot_product": float(np.dot(weights, values)).hex(),
            "scaled_column": hashlib.sha256(scaled_column.tobytes()).hexdigest(),
            "weighted_probabilities": hashlib.sha256(
                (weights @ probabilities).tobytes()
            ).hexdigest(),
        }
    )
)
"""

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


@functools.cache
def _library_results(
    mode: str, settings: tuple[tuple[str, str], ...]
) -> dict[str, object]:
    return _last_json_line(
        [sys.executable, "-c", _LIBRARY_PROBE, mode],
        _clean_environment(dict(settings)),
    )


@pytest.mark.parametrize(
    "settings",
    (
        (),
        ((OPENMP, "2"),),
        ((OPENBLAS_FALLBACK, "2"),),
        ((NUMEXPR_MAXIMUM, "3"),),
        ((NUMEXPR_MAXIMUM, "3"), (OPENMP, "2")),
    ),
    ids=("nothing", "openmp", "openblas_fallback", "numexpr_maximum", "both"),
)
def test_the_settings_keep_thread_counts_and_result_bytes(
    settings: tuple[tuple[str, str], ...],
) -> None:
    assert _library_results("with_settings", settings) == _library_results(
        "without_settings", settings
    )


def test_sign_classifier_probabilities_do_not_depend_on_the_openmp_count() -> None:
    """scikit-learn may move from physical to logical CPUs under the settings."""

    digests = {
        thread_count: _last_json_line(
            [sys.executable, "-c", _SIGN_CLASSIFIER_PROBE],
            _clean_environment({OPENMP: thread_count, OPENBLAS: "1"}),
        )["digest"]
        for thread_count in ("1", "2", "4")
    }

    assert len(set(digests.values())) == 1, digests
