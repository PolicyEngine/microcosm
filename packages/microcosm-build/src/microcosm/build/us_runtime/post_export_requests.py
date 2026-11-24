"""Picklable descriptions of the US release's post-export scoring requests.

The release tool (``tools/build_us_fiscal_refresh_release.py``) learns what
each post-export consumer will ask policyengine-us for by running the consumer
once against engine-free recorders (its "dry run"). The consumers call
``simulate(None)`` for the baseline simulation and ``simulate(reform)`` for a
reformed one, then ``.calculate(variable, period, map_to=...)`` on each. The
dry run returns every such request as a :class:`PostExportRequest`, which a
process started with the ``spawn`` method can unpickle and run.

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

These types live here, in an importable package, because the release tool is
loaded by file path, outside ``sys.modules``: a spawned process could not
unpickle a type defined in it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from microcosm.build.us_runtime.release_input_coverage import ReformCoverageProbe

if TYPE_CHECKING:
    from microcosm.build.us_runtime.reform_validation import ReformValidationSpec

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
    """

    parameter_changes: Mapping[str, Any]

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


#: What a post-export request records as its reform: the description the
#: consumer builds the reform from.
type PostExportReform = (
    ReformCoverageProbe | ReformValidationSpec | ParameterChangesReform
)


def _post_export_reform_types() -> tuple[type, ...]:
    # Imported here: reform_validation imports this module.
    from microcosm.build.us_runtime.reform_validation import ReformValidationSpec

    return (ReformCoverageProbe, ReformValidationSpec, ParameterChangesReform)


def check_post_export_reform(reform: object) -> None:
    """Refuse a reform that is not a description a worker can rebuild.

    Raises:
        TypeError: Naming the reform's type and the types accepted.
    """
    types = _post_export_reform_types()
    if not isinstance(reform, types):
        accepted = ", ".join(kind.__name__ for kind in types)
        raise TypeError(
            "Post-export scoring records each reform as the description a worker "
            f"rebuilds it from ({accepted}); got a {type(reform).__qualname__}. "
            "A consumer whose simulate carries RECORDS_REQUESTS_ONLY must hand it "
            "that description, not a built reform."
        )


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

    def __post_init__(self) -> None:
        for name in ("consumer", "variable"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"PostExportRequest.{name} must be a non-empty string, "
                    f"got {value!r}."
                )
        for name in ("position", "period"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"PostExportRequest.{name} must be an integer, got {value!r}."
                )
        if self.position < 0:
            raise ValueError(
                f"PostExportRequest.position must be 0 or more, got {self.position}."
            )
        if self.map_to is not None and (
            not isinstance(self.map_to, str) or not self.map_to
        ):
            raise ValueError(
                "PostExportRequest.map_to must be None or an entity name, got "
                f"{self.map_to!r}."
            )
        if self.reform is None:
            if self.reformed_simulation is not None:
                raise ValueError(
                    "A baseline PostExportRequest names no reformed simulation, "
                    f"got {self.reformed_simulation!r}."
                )
            return
        check_post_export_reform(self.reform)
        simulation = self.reformed_simulation
        if (
            not isinstance(simulation, int)
            or isinstance(simulation, bool)
            or simulation < 0
        ):
            raise ValueError(
                "A reform PostExportRequest names its reformed simulation as an "
                f"integer of 0 or more, got {simulation!r}."
            )

    @property
    def key(self) -> tuple[str, int, str | None]:
        """(variable, period, map_to), the release tool's ``PostExportKey``."""
        return (self.variable, self.period, self.map_to)

    def build_reform(self) -> Any:
        """The reform the worker's engine applies; ``None`` for the baseline."""
        if self.reform is None:
            return None
        return build_post_export_reform(self.reform)
