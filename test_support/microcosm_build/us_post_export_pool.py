"""Fakes a spawned pool worker can import, for pooled post-export scoring tests.

Pooled scoring (``_PooledPostExportScorer`` in
``tools/build_us_fiscal_refresh_release.py``) sends its engine, batch dataset
and reform builder to every worker by name, so each must be a module-level
object a fresh process can import. These give the values of the scoring tests'
fake engine (:mod:`test_support.microcosm_build.us_post_export_scoring`), so the
pooled and serial scorers can be run on the same fake and compared byte for
byte.

Behaviour a test sets in the parent reaches each worker through its
environment, which spawned workers inherit:

- :data:`FAIL_ON`: a variable every engine refuses to compute;
- :data:`COMPUTES`: JSON mapping a variable to the watched formulas computing
  it leaves a value for (the population-aggregate and baseline-branch
  refusals);
- :data:`ENGINE_LOG`: a directory; each process appends one JSON line per
  engine construction and per calculation to ``<pid>.jsonl`` there;
- :data:`MEMORY_KEYS`: the most distinct keys one engine computes before it
  raises ``MemoryError``, as an engine that ran out of memory would;
- :data:`SLOW`: JSON mapping a variable to the seconds each engine sleeps
  before it computes that variable;
- :data:`WAIT_FOR`: JSON mapping a variable to a file each engine waits for
  (two minutes at most) before it computes that variable;
- :data:`WAIT_FOR_OTHER_REFORMS`: JSON ``{"file": path, "except": [labels]}``;
  an engine whose reform's label is not listed waits for the file (two minutes
  at most) before it computes anything, so a test can keep one stage's
  reformed simulations running while another stage's finish;
- :data:`FAIL_ON_HOUSEHOLDS` and :data:`MEMORY_ON_HOUSEHOLDS`: JSON mapping a
  variable to household ids; an engine whose batch holds one of them raises
  ``RuntimeError`` or ``MemoryError`` for that variable.

Each fake reform carries the description it was built from, as the
consumers' own builders mark the reforms they build.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from microcosm.build.us_runtime.post_export_requests import (
    ParameterChangesReform,
    described_reform,
)
from microcosm.build.us_runtime.reform_validation import ReformValidationSpec
from microcosm.build.us_runtime.release_input_coverage import ReformCoverageProbe
from test_support.microcosm_build.us_post_export_scoring import (
    _FakeParameterTree,
    _FakeReform,
    _FakeSeries,
    _household_positions,
    _map_values,
    _native_entity,
    _native_values,
)

FAIL_ON = "MICROCOSM_TEST_POST_EXPORT_FAIL_ON"
COMPUTES = "MICROCOSM_TEST_POST_EXPORT_COMPUTES"
ENGINE_LOG = "MICROCOSM_TEST_POST_EXPORT_ENGINE_LOG"
MEMORY_KEYS = "MICROCOSM_TEST_POST_EXPORT_MEMORY_KEYS"
SLOW = "MICROCOSM_TEST_POST_EXPORT_SLOW"
WAIT_FOR = "MICROCOSM_TEST_POST_EXPORT_WAIT_FOR"
WAIT_FOR_OTHER_REFORMS = "MICROCOSM_TEST_POST_EXPORT_WAIT_FOR_OTHER_REFORMS"
FAIL_ON_HOUSEHOLDS = "MICROCOSM_TEST_POST_EXPORT_FAIL_ON_HOUSEHOLDS"
MEMORY_ON_HOUSEHOLDS = "MICROCOSM_TEST_POST_EXPORT_MEMORY_ON_HOUSEHOLDS"


def _households_for(variable: str, name: str) -> set[int]:
    return set(json.loads(os.environ.get(name) or "{}").get(variable, ()))


#: Parameter paths a fake reform sets from a ``ParameterChangesReform``
#: payload, so a reform can move a behavioral-response parameter.
_RESPONSE_PARAMETERS = "gov.simulation."


def _log(event: dict) -> None:
    directory = os.environ.get(ENGINE_LOG)
    if not directory:
        return
    path = Path(directory) / f"{os.getpid()}.jsonl"
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def build_fake_reform(description):
    """The fake reform of a description, as the scoring tests' ``fake_reforms``
    fixture builds it in the parent: labelled by the probe's or spec's id, or
    by the parameter changes' JSON."""
    if isinstance(description, ReformCoverageProbe | ReformValidationSpec):
        return described_reform(_FakeReform(description.id), description)
    if isinstance(description, ParameterChangesReform):
        changes = dict(description.parameter_changes)
        parameters = {
            path: next(iter(value.values())) if isinstance(value, dict) else value
            for path, value in changes.items()
            if path.startswith(_RESPONSE_PARAMETERS)
        }
        return described_reform(
            _FakeReform(
                json.dumps(changes, sort_keys=True, default=str), parameters=parameters
            ),
            description,
        )
    raise TypeError(f"no fake reform for {type(description).__name__}")


def build_fake_parameter_reform(changes):
    """``reform_validation._build_parameter_reform`` on the fake."""
    return build_fake_reform(ParameterChangesReform(changes))


def identity_dataset(batch_frame):
    """Each batch frame is its own dataset."""
    return batch_frame


class FakeSystem:
    """A reform's rules: its fake reform and its parameter tree."""

    def __init__(self, reform=None) -> None:
        self.reform = reform
        self.parameters = _FakeParameterTree(
            {} if reform is None else reform.parameters
        )


class _FakeHolder:
    def __init__(self, periods) -> None:
        self._periods = periods

    def get_known_periods(self):
        return sorted(self._periods)


def _wait_for_file(path: str) -> None:
    """Wait until ``path`` exists, two minutes at most."""
    deadline = time.monotonic() + 120
    while not Path(path).exists() and time.monotonic() < deadline:
        time.sleep(0.1)


#: This process's engines, to count those not yet released at a construction.
_ENGINES: list = []


class FakeMicrosimulation:
    """The scoring tests' fake engine as a module-level class."""

    default_tax_benefit_system = FakeSystem
    default_tax_benefit_system_instance = SimpleNamespace(
        parameters=_FakeParameterTree({})
    )

    def __init__(self, *, dataset, reform=None, tax_benefit_system=None, spm=None):
        unreleased = sum(engine.dataset is not None for engine in _ENGINES)
        self.dataset = dataset
        self.reform = reform
        self.tax_benefit_system = tax_benefit_system
        self.spm = spm
        self.branches: dict = {}
        self.known: dict[str, set[int]] = {}
        self.number = len(_ENGINES)
        _ENGINES.append(self)
        system_reform = getattr(tax_benefit_system, "reform", None)
        _log(
            {
                "event": "construct",
                "engine": self.number,
                "households": [
                    int(value) for value in dataset.table("household")["household_id"]
                ],
                "reform": None
                if self.policy_reform is None
                else self.policy_reform.label,
                "reform_argument": reform is not None,
                "system_reform": None if system_reform is None else system_reform.label,
                "spm": spm,
                "unreleased": unreleased,
            }
        )

    @property
    def policy_reform(self):
        if self.reform is not None:
            return self.reform
        return getattr(self.tax_benefit_system, "reform", None)

    def get_holder(self, variable):
        return _FakeHolder(self.known.get(variable, ()))

    def calculate(self, variable, period, map_to=None):
        frame = self.dataset
        assert frame is not None, "calculate on a released engine"
        if variable == os.environ.get(FAIL_ON):
            raise RuntimeError(f"fixture engine refuses {variable}")
        period = int(period)
        _log(
            {
                "event": "calculate",
                "engine": self.number,
                "variable": variable,
                "period": period,
            }
        )
        self.asked = getattr(self, "asked", set()) | {(variable, period, map_to)}
        limit = os.environ.get(MEMORY_KEYS)
        if limit and len(self.asked) > int(limit):
            raise MemoryError(f"fixture engine holds more than {limit} keys")
        delay = json.loads(os.environ.get(SLOW) or "{}").get(variable)
        if delay:
            time.sleep(float(delay))
        awaited = json.loads(os.environ.get(WAIT_FOR) or "{}").get(variable)
        if awaited:
            _wait_for_file(awaited)
        held = json.loads(os.environ.get(WAIT_FOR_OTHER_REFORMS) or "{}")
        reform = self.policy_reform
        if held and reform is not None and reform.label not in held["except"]:
            _wait_for_file(held["file"])
        households = set(
            int(value) for value in frame.table("household")["household_id"]
        )
        if households & _households_for(variable, FAIL_ON_HOUSEHOLDS):
            raise RuntimeError(f"fixture engine refuses {variable}")
        if households & _households_for(variable, MEMORY_ON_HOUSEHOLDS):
            raise MemoryError(f"fixture engine holds too much for {variable}")
        computes = json.loads(os.environ.get(COMPUTES) or "{}")
        for computed in (variable, *computes.get(variable, ())):
            self.known.setdefault(computed, set()).add(period)
        native = _native_entity(variable)
        delta = 0.0 if reform is None else reform.delta
        values = _native_values(frame, variable, period, delta)
        target = map_to or native
        values = _map_values(frame, values, native, target)
        weights = frame.weights_for("household").values[
            _household_positions(frame, target)
        ]
        return _FakeSeries(values, weights)


def engine_log(directory: Path) -> list[dict]:
    """Every process's logged events, each with its process id."""
    events = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            events.append({"pid": int(path.stem), **json.loads(line)})
    return events


def float32_weights(series, period: int, uprating) -> _FakeSeries:
    factor = np.float32(uprating.get(int(period), 1.0))
    weights = series.weights.astype(np.float32) * factor
    return _FakeSeries(np.asarray(series), weights.astype(np.float64))


#: The export probe tests' uprating of 2026 weights (see
#: :mod:`test_support.microcosm_build.us_post_export_probe`).
FLOAT32_UPRATING = {2026: 352_304_000 / 346_588_000}


class Float32FakeMicrosimulation(FakeMicrosimulation):
    """:class:`FakeMicrosimulation` with policyengine-core's float32 weights,
    uprated at a later period, as the export probe's tests score."""

    def calculate(self, variable, period, map_to=None):
        series = super().calculate(variable, period, map_to=map_to)
        return float32_weights(series, period, FLOAT32_UPRATING)


__all__ = [name for name in globals() if not name.startswith("__")]
