"""Picklable descriptions of the US release's post-export scoring requests.

The release tool (``tools/build_us_fiscal_refresh_release.py``) learns what
each post-export consumer will ask policyengine-us for by running the consumer
once against engine-free recorders (its "dry run"). The consumers call
``simulate(None)`` for the baseline simulation and ``simulate(reform)`` for a
reformed one, then ``.calculate(variable, period, map_to=...)`` on each. The
release takes only each consumer's baseline plan and reform-pass count from the
dry run; the tool's ``_record_post_export_request_descriptors`` returns every
request of the same dry run as a :class:`PostExportRequest`, which a process
started with the ``spawn`` method can unpickle and run.

A built reform cannot cross a process boundary: ``Reform.from_dict`` returns a
class defined inside a function, which pickle cannot name. So a consumer whose
``simulate`` carries :data:`RECORDS_REQUESTS_ONLY` hands it the description it
builds the reform from instead, and :func:`build_post_export_reform` rebuilds
the reform from that description with the consumer's own builder:

- a :class:`~microcosm.build.us_runtime.release_input_coverage.ReformCoverageProbe`
  (the reform-coverage smoke check), rebuilt by
  ``reform_coverage_smoke._build_reform``;
- a :class:`~microcosm.build.us_runtime.reform_validation.ReformValidationSpec`
  (a reform-validation row simulated on its own), rebuilt by its
  ``build_reform``;
- a :class:`ParameterChangesReform` (one of reform validation's cumulative
  OBBBA states), rebuilt by ``reform_validation._build_parameter_reform``.

These types live here, in an importable package, so that a spawned process
can unpickle them however the release tool was started. The tests and the
guard sweep load the tool by file path, outside ``sys.modules``, where a type
defined in it cannot even be pickled. A release runs the tool as a script,
where a spawned process would unpickle such a type only by running the whole
tool again as its ``__mp_main__`` module (which a process started with
``spawn`` does to the parent's main script in any case).

Neither type can be hashed, because a reform description holds dictionaries;
key a request's results by its ``(consumer, position)``.
"""

from __future__ import annotations

import copy
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from microcosm.build.us_runtime.reform_validation import ReformValidationSpec
from microcosm.build.us_runtime.release_input_coverage import ReformCoverageProbe

__all__ = [
    "RECORDS_REQUESTS_ONLY",
    "ParameterChangesReform",
    "PostExportReform",
    "PostExportRequest",
    "build_post_export_reform",
    "check_post_export_reform",
]

#: Set to ``True`` on a ``simulate`` that only records the requests it is
#: asked for and never applies a reform (the release tool's engine-free dry
#: run). A consumer then hands it the description of each reform (see the
#: module docstring) in place of the built reform, so recording builds no
#: reform, resolves no list edit and needs no engine.
RECORDS_REQUESTS_ONLY = "records_requests_only"


@dataclass(frozen=True)
class ParameterChangesReform:
    """A reform given as a ``Reform.from_dict`` payload for country ``us``.

    Reform validation scores OBBBA's provisions cumulatively from a pre-OBBBA
    state; each state is one such payload. An empty payload is the baseline,
    which a consumer asks for as ``simulate(None)``, so it is refused here.

    It keeps a deep copy of the payload, as a ``dict``: the request records
    the reform as the consumer asked for it, and no later change to the
    consumer's dictionaries (which its reform specs share) reaches it.
    """

    parameter_changes: Mapping[str, Any]

    # Not hashable: the payload is a dictionary.
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.parameter_changes, Mapping):
            raise TypeError(
                "ParameterChangesReform needs its parameter changes as a mapping "
                f"of parameter paths, got {type(self.parameter_changes).__name__}."
            )
        if not self.parameter_changes:
            raise ValueError(
                "ParameterChangesReform needs at least one of its parameter "
                "changes; the baseline is simulate(None)."
            )
        try:
            copied = copy.deepcopy(dict(self.parameter_changes))
        except Exception as error:
            raise TypeError(
                "ParameterChangesReform's parameter changes cannot be copied: "
                f"{type(error).__name__}: {error}"
            ) from error
        object.__setattr__(self, "parameter_changes", copied)


#: What a post-export request records as its reform: the description the
#: consumer builds the reform from.
type PostExportReform = (
    ReformCoverageProbe | ReformValidationSpec | ParameterChangesReform
)


_POST_EXPORT_REFORM_TYPES = (
    ReformCoverageProbe,
    ReformValidationSpec,
    ParameterChangesReform,
)


def check_post_export_reform(reform: object) -> None:
    """Refuse a reform that is not a description a worker can rebuild.

    The description must be exactly one of the accepted types, not a
    subclass, whose module a worker might not have, and must pickle, so a
    failure shows in the release's pre-export plan rather than when a request
    is sent to a worker.

    Raises:
        TypeError: Naming the reform's type and the types accepted, or why
            the description cannot be pickled.
    """
    if type(reform) not in _POST_EXPORT_REFORM_TYPES:
        accepted = ", ".join(kind.__name__ for kind in _POST_EXPORT_REFORM_TYPES)
        raise TypeError(
            "Post-export scoring records each reform as the description a worker "
            f"rebuilds it from ({accepted}); got a {type(reform).__module__}."
            f"{type(reform).__qualname__}. A consumer whose simulate carries "
            "RECORDS_REQUESTS_ONLY must hand it a description of one of those "
            "types exactly, not a built reform or a subclass."
        )
    try:
        pickle.dumps(reform)
    except Exception as error:
        raise TypeError(
            f"Post-export scoring got a {type(reform).__qualname__} reform "
            "description that cannot be pickled for a worker process: "
            f"{type(error).__name__}: {error}"
        ) from error


def build_post_export_reform(reform: PostExportReform) -> Any:
    """Build the reform ``reform`` describes, with its consumer's own builder.

    Each builder is looked up when it is called, so this builds exactly what
    the consumer builds when it scores the same description.
    """
    check_post_export_reform(reform)
    if isinstance(reform, ReformCoverageProbe):
        from microcosm.build.us_runtime import reform_coverage_smoke

        return reform_coverage_smoke._build_reform(reform)
    if isinstance(reform, ParameterChangesReform):
        from microcosm.build.us_runtime import reform_validation

        return reform_validation._build_parameter_reform(reform.parameter_changes)
    return reform.build_reform()


def _require_name(name: str, value: object) -> None:
    if not isinstance(value, str):
        raise TypeError(f"PostExportRequest.{name} must be a string, got {value!r}.")
    if not value:
        raise ValueError(f"PostExportRequest.{name} must not be empty.")


def _require_integer(name: str, value: object, *, minimum: int | None = None) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(
            f"PostExportRequest.{name} must be a Python integer, got {value!r}."
        )
    if minimum is not None and value < minimum:
        raise ValueError(
            f"PostExportRequest.{name} must be {minimum} or more, got {value}."
        )


@dataclass(frozen=True)
class PostExportRequest:
    """One request a post-export consumer makes of policyengine-us.

    Attributes:
        consumer: The consumer's name in the release's post-export plan, such
            as ``"reform_validation"``.
        position: The request's place, from 0, in the order the serial scorer
            runs the consumer's requests: the baseline plan's keys (scored
            together in one pass, in ascending period order and then request
            order), then one reform pass per reform request, in the order the
            consumer first asks each reformed simulation for each key.
        variable: The variable to calculate.
        period: The year to calculate it for.
        map_to: The entity the values are mapped to, or ``None`` for the
            variable's own entity, which the engine resolves when it
            calculates the variable.
        reform: ``None`` for the baseline simulation; otherwise the
            description the reform is rebuilt from (:func:`build_post_export_reform`).
        reformed_simulation: For a reform request, which of the consumer's
            ``simulate(reform)`` calls made the simulation asked, numbered from
            0 in call order (a simulation never asked makes no request but
            keeps its number); ``None`` for the baseline.
    """

    consumer: str
    position: int
    variable: str
    period: int
    map_to: str | None = None
    reform: PostExportReform | None = None
    reformed_simulation: int | None = None

    # Not hashable: most reform descriptions hold dictionaries.
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        _require_name("consumer", self.consumer)
        _require_name("variable", self.variable)
        _require_integer("position", self.position, minimum=0)
        _require_integer("period", self.period)
        if self.map_to is not None:
            _require_name("map_to", self.map_to)
        if self.reform is None:
            if self.reformed_simulation is not None:
                raise ValueError(
                    "A baseline PostExportRequest names no reformed simulation, "
                    f"got {self.reformed_simulation!r}."
                )
            return
        check_post_export_reform(self.reform)
        if self.reformed_simulation is None:
            raise ValueError(
                "A reform PostExportRequest names the reformed simulation that "
                "asked for it."
            )
        _require_integer("reformed_simulation", self.reformed_simulation, minimum=0)

    @property
    def key(self) -> tuple[str, int, str | None]:
        """(variable, period, map_to), the release tool's ``PostExportKey``."""
        return (self.variable, self.period, self.map_to)

    def build_reform(self) -> Any:
        """The reform the worker's engine applies; ``None`` for the baseline."""
        if self.reform is None:
            return None
        return build_post_export_reform(self.reform)
