"""How every root input of a bound Axiom module is fed: the input closure.

An Axiom module reads every name a rule references and does not derive. The
adapter (:mod:`microcosm.frame.adapters.axiom`) omits an input the frame does
not carry, and the engine fails a request whose evaluation reaches an absent
input. A concept mapping binds only the inputs a concept determines, so a run
also needs a decision for every other root input. An :class:`InputClosure` is
that decision, as data: for each bound module it puts every root input of the
module's committed input surface in exactly one class.

- ``encoded``: computed per record, by a concept binding of the mapping
  (:class:`~microcosm.frame.concept_mapping.InputBinding`), by a
  :class:`~microcosm.frame.concept_mapping.StateBinding` from model state, or
  by another graph node (``graph_node``, for example an engine-computed
  bridge value).
- ``defaulted``: one value for every record, with the reason and the
  provision it stands in for. Defaults are modelling content, so each says
  whether it is a new default for review.
- ``engine_optional``: the engine itself supplies the input when it is
  absent. The entry carries the evidence that the engine does.

The closure also declares the encoding knobs a country's scenarios turn
(:meth:`InputClosure.group_knobs` translates their values into
:class:`~microcosm.frame.concept_mapping.GroupKnobs`) and how an
undetermined judgment (code ``0``) reaches take-up
(:func:`resolve_judgments`). :meth:`InputClosure.check` compares a closure
with the mapping and the engine-generated surface: the classes partition each
module's surface exactly, and every concept-encoded entry is the mapping's
binding and nothing else.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from microcosm.frame.concept_mapping import (
    Allocation,
    ConceptMapping,
    GroupKnobs,
    InputRef,
    StateBinding,
)
from microcosm.frame.concepts import (
    ContentBasis,
    TransportRule,
    _parsing,
    _record_fields,
    _text_fields,
    concept,
)

if TYPE_CHECKING:
    from microcosm.frame.adapters.axiom_input_surface import AxiomInputSurface

__all__ = [
    "INPUT_CLOSURE_FORMAT",
    "ClosureClass",
    "ClosureEntry",
    "EncodedBy",
    "InputClosure",
    "Knob",
    "KnobKind",
    "UndeterminedAction",
    "UndeterminedPolicy",
    "apply_defaults",
    "resolve_judgments",
]

#: The serialized closure format :meth:`InputClosure.to_dict` writes.
INPUT_CLOSURE_FORMAT = "microcosm.axiom_input_closure.v1"


class ClosureClass(StrEnum):
    """How one root input is fed."""

    ENCODED = "encoded"
    DEFAULTED = "defaulted"
    ENGINE_OPTIONAL = "engine_optional"


class EncodedBy(StrEnum):
    """What computes an encoded input."""

    #: A binding of the concept mapping.
    CONCEPT_BINDING = "concept_binding"
    #: A state binding the closure declares.
    STATE_BINDING = "state_binding"
    #: Another graph node, named by the entry.
    GRAPH_NODE = "graph_node"


class UndeterminedAction(StrEnum):
    """What take-up does with an undetermined (code ``0``) judgment."""

    #: Fail: an undetermined judgment stops the run.
    REFUSE = "refuse"
    #: Treat it as not holding.
    NOT_ELIGIBLE = "not_eligible"
    #: Treat it as holding.
    ELIGIBLE = "eligible"


class KnobKind(StrEnum):
    """What a declared knob changes."""

    #: Where allocated household amounts go (an :class:`Allocation` value).
    ALLOCATION = "allocation"
    #: A test switched on (``true``) or off; off zeroes the targets.
    SWITCH_ZEROES = "switch_zeroes"
    #: Which column a state binding reads (a column on the same entity).
    STATE_COLUMN = "state_column"
    #: A non-negative factor on the targets.
    FACTOR = "factor"


_JSON_SCALAR = bool | int | float | str


@dataclass(frozen=True, kw_only=True)
class Knob:
    """One declared encoding alternative a scenario can turn.

    Attributes:
        name: The knob's name in the country's scenario list.
        kind: What it changes.
        targets: The engine inputs it changes, or for
            :attr:`KnobKind.STATE_COLUMN` the one declared state column
            (``<entity>.<column>``) it replaces.
        central: The value the central run uses.
        values: The values it may take (allocation modes or column names);
            empty for switches and factors.
        method_card_row: The method-card row that rules on it.
        note: What turning it means.
    """

    name: str
    kind: KnobKind
    targets: tuple[str, ...]
    central: _JSON_SCALAR
    values: tuple[_JSON_SCALAR, ...]
    method_card_row: str
    note: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", KnobKind(self.kind))
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "values", tuple(self.values))
        if not self.name or not self.note or not self.method_card_row:
            raise ValueError("A knob needs a name, a note and a method-card row.")
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError(f"Knob {self.name!r} names distinct targets.")
        if self.kind is KnobKind.STATE_COLUMN and len(self.targets) != 1:
            raise ValueError(f"Knob {self.name!r} replaces one state column.")
        if self.kind in (KnobKind.ALLOCATION, KnobKind.STATE_COLUMN):
            if not self.values:
                raise ValueError(f"Knob {self.name!r} lists the values it takes.")
        elif self.values:
            raise ValueError(f"Knob {self.name!r} takes no value list.")
        self.resolve(self.central)

    def resolve(self, value: object) -> _JSON_SCALAR:
        """``value`` checked against the knob's kind and allowed values."""
        kind = self.kind
        if kind is KnobKind.ALLOCATION or kind is KnobKind.STATE_COLUMN:
            if value not in self.values:
                raise ValueError(
                    f"Knob {self.name!r} takes {list(self.values)}, not {value!r}."
                )
            if kind is KnobKind.ALLOCATION:
                Allocation(value)
            return value
        if kind is KnobKind.SWITCH_ZEROES:
            if not isinstance(value, bool):
                raise ValueError(f"Knob {self.name!r} is switched true or false.")
            return value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"Knob {self.name!r} takes a number.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Knob {self.name!r} takes a finite, non-negative factor.")
        return float(value)

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready form."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "targets": list(self.targets),
            "central": self.central,
            "values": list(self.values),
            "method_card_row": self.method_card_row,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Knob:
        """The knob ``data`` describes (see :meth:`to_dict`)."""
        with _parsing("knob"):
            fields = _record_fields(
                data,
                "knob",
                required=(
                    "name",
                    "kind",
                    "targets",
                    "central",
                    "values",
                    "method_card_row",
                    "note",
                ),
            )
            _text_fields(fields, "knob", ("name", "kind", "method_card_row", "note"))
            for name in ("targets", "values"):
                if not isinstance(fields[name], list | tuple):
                    raise ValueError(f"A knob's {name!r} must be a list.")
            if not all(isinstance(target, str) for target in fields["targets"]):
                raise ValueError("A knob's 'targets' must name inputs or a column.")
            return cls(**fields)


@dataclass(frozen=True, kw_only=True)
class UndeterminedPolicy:
    """How an undetermined entitlement judgment reaches take-up.

    Attributes:
        action: What take-up does with code ``0``.
        reason: Why, with the engine evidence.
        alternatives: The other actions a reviewer could choose.
        new_default: Whether this is a new modelling default for review.
    """

    action: UndeterminedAction
    reason: str
    alternatives: tuple[UndeterminedAction, ...]
    new_default: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", UndeterminedAction(self.action))
        object.__setattr__(
            self,
            "alternatives",
            tuple(UndeterminedAction(item) for item in self.alternatives),
        )
        if not self.reason:
            raise ValueError("An undetermined-judgment policy needs a reason.")
        if self.action in self.alternatives:
            raise ValueError("The chosen action is not its own alternative.")
        if not isinstance(self.new_default, bool):
            raise ValueError("new_default is true or false.")

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready form."""
        return {
            "action": self.action.value,
            "reason": self.reason,
            "alternatives": [item.value for item in self.alternatives],
            "new_default": self.new_default,
        }


@dataclass(frozen=True, kw_only=True)
class ClosureEntry:
    """How one root input of one module is fed.

    Attributes:
        module: The RuleSpec module, relative to its rulespec root.
        entity: The engine entity whose program reads the input.
        input: The root input's name.
        closure_class: Its class (serialized as ``class``).
        encoded_by: For an encoded input, what computes it.
        state_binding: For a state-encoded input, the binding.
        node: For a graph-node input, the node id that owns it.
        value: For a defaulted input, the value every record gets.
        reason: Why the input is fed this way. Required except for a concept
            binding, whose evidence is the binding's own note.
        citation: For a defaulted input, the provision the default stands in
            for.
        new_default: For a defaulted input, whether it is a new modelling
            default for review.
        awaiting: For a concept-encoded input whose binding lands in a named
            later change, that change; the mapping may lack the binding until
            then, and a closure whose awaited binding exists is out of date.
    """

    module: str
    entity: str
    input: str
    closure_class: ClosureClass
    encoded_by: EncodedBy | None = None
    state_binding: StateBinding | None = None
    node: str | None = None
    value: bool | int | float | None = None
    reason: str = ""
    citation: str | None = None
    new_default: bool = False
    awaiting: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "closure_class", ClosureClass(self.closure_class))
        if self.encoded_by is not None:
            object.__setattr__(self, "encoded_by", EncodedBy(self.encoded_by))
        label = f"{self.input} ({self.entity}, {self.module})"
        if not self.module or not self.entity or not self.input:
            raise ValueError("A closure entry names its module, entity and input.")
        encoded = self.closure_class is ClosureClass.ENCODED
        if encoded != (self.encoded_by is not None):
            raise ValueError(f"{label}: exactly an encoded input says what encodes it.")
        if (self.encoded_by is EncodedBy.STATE_BINDING) != (
            self.state_binding is not None
        ):
            raise ValueError(f"{label}: a state-encoded input carries its binding.")
        if self.state_binding is not None and self.state_binding.ref != self.ref:
            raise ValueError(f"{label}: its state binding feeds another input.")
        if (self.encoded_by is EncodedBy.GRAPH_NODE) != bool(self.node):
            raise ValueError(f"{label}: a graph-node input names its node.")
        defaulted = self.closure_class is ClosureClass.DEFAULTED
        if defaulted:
            value = self.value
            if value is None or not isinstance(value, bool | int | float):
                raise ValueError(f"{label}: a default is a boolean or a number.")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{label}: a default is finite.")
            if not self.citation:
                raise ValueError(f"{label}: a default cites the provision it fills.")
        elif self.value is not None or self.citation is not None or self.new_default:
            raise ValueError(f"{label}: only a default carries a value or citation.")
        if not isinstance(self.new_default, bool):
            raise ValueError(f"{label}: new_default is true or false.")
        if self.encoded_by is not EncodedBy.CONCEPT_BINDING and not self.reason:
            raise ValueError(f"{label}: needs a reason.")
        if self.awaiting is not None and (
            self.encoded_by is not EncodedBy.CONCEPT_BINDING or not self.awaiting
        ):
            raise ValueError(f"{label}: only a concept binding awaits a change.")

    @property
    def ref(self) -> InputRef:
        """The input this entry closes."""
        return InputRef(self.input, self.entity, self.module)

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready form, omitting absent optional fields."""
        out: dict[str, object] = {
            "rulespec_path": self.module,
            "entity": self.entity,
            "input": self.input,
            "class": self.closure_class.value,
        }
        if self.encoded_by is not None:
            out["encoded_by"] = self.encoded_by.value
        if self.state_binding is not None:
            # The entry names the module once; a country package's spec keys
            # never say "module" (they say rulespec_path).
            state = self.state_binding.to_dict()
            state.pop("module", None)
            out["state_binding"] = state
        if self.node is not None:
            out["node"] = self.node
        if self.value is not None:
            out["value"] = self.value
        if self.reason:
            out["reason"] = self.reason
        if self.citation is not None:
            out["citation"] = self.citation
        if self.new_default:
            out["new_default"] = True
        if self.awaiting is not None:
            out["awaiting"] = self.awaiting
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ClosureEntry:
        """The entry ``data`` describes (see :meth:`to_dict`)."""
        with _parsing("closure entry"):
            fields = _record_fields(
                data,
                "closure entry",
                required=("rulespec_path", "entity", "input", "class"),
                optional=(
                    "encoded_by",
                    "state_binding",
                    "node",
                    "value",
                    "reason",
                    "citation",
                    "new_default",
                    "awaiting",
                ),
            )
            _text_fields(
                fields,
                "closure entry",
                (
                    "rulespec_path",
                    "entity",
                    "input",
                    "class",
                    "encoded_by",
                    "node",
                    "reason",
                    "citation",
                    "awaiting",
                ),
            )
            fields["closure_class"] = fields.pop("class")
            fields["module"] = fields.pop("rulespec_path")
            if "state_binding" in fields:
                state = fields["state_binding"]
                if not isinstance(state, Mapping) or "module" in state:
                    raise ValueError(
                        "A closure entry's state binding is an object that takes "
                        "its module from the entry's rulespec_path."
                    )
                fields["state_binding"] = StateBinding.from_dict(
                    {**state, "module": fields["module"]}
                )
            return cls(**fields)


@dataclass(frozen=True, kw_only=True)
class InputClosure:
    """Every root input of a country's bound Axiom modules, closed.

    Attributes:
        country: The rulespec country code.
        content_basis: The frames the closure feeds. On a ``transport`` frame
            the concepts transport drops (donor program receipts) are absent,
            so a binding that reads one never runs and its input must be
            closed some other way.
        mapping_engine: The concept mapping the closure was reviewed with.
        mapping_engine_version: That mapping's ``engine_version``.
        surface_rulespec_commit: The rulespec commit the input surface was
            generated from.
        surface_engine_repository: The repository of the engine that
            generated it. The engine commit is serialized in a record that
            names this repository, which marks it as the engine's commit, not
            a rulespec commit; :meth:`check` compares the commit with the
            surface, not the repository.
        surface_engine_commit: The engine commit that generated it.
        modules: Module path -> SHA-256 of the module bytes reviewed.
        engine_optional_evidence: What the engine does with an absent input,
            with its locators; the evidence any ``engine_optional`` entry
            rests on, and the reason there may be none.
        undetermined: How undetermined judgments reach take-up.
        knobs: The encoding knobs scenarios turn.
        entries: One entry per root input of every listed module.
    """

    country: str
    content_basis: ContentBasis
    mapping_engine: str
    mapping_engine_version: str
    surface_rulespec_commit: str
    surface_engine_repository: str
    surface_engine_commit: str
    modules: Mapping[str, str]
    engine_optional_evidence: str
    undetermined: UndeterminedPolicy
    knobs: tuple[Knob, ...]
    entries: tuple[ClosureEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_basis", ContentBasis(self.content_basis))
        object.__setattr__(self, "modules", MappingProxyType(dict(self.modules)))
        object.__setattr__(self, "knobs", tuple(self.knobs))
        object.__setattr__(self, "entries", tuple(self.entries))
        if not self.engine_optional_evidence:
            raise ValueError(
                "A closure states what the engine does with absent inputs."
            )
        twice = [
            ref for ref, n in Counter(e.ref for e in self.entries).items() if n > 1
        ]
        if twice:
            raise ValueError(f"Inputs closed twice: {[ref.label() for ref in twice]}.")
        unlisted = sorted({entry.module for entry in self.entries} - set(self.modules))
        if unlisted:
            raise ValueError(
                f"Entries name modules the closure does not list: {unlisted}."
            )
        names = [knob.name for knob in self.knobs]
        if len(set(names)) != len(names):
            raise ValueError("Knob names must be unique.")
        encoded = {
            entry.input
            for entry in self.entries
            if entry.closure_class is ClosureClass.ENCODED
        }
        state_columns = {
            column
            for entry in self.entries
            if entry.state_binding is not None
            for column in entry.state_binding.columns
        }
        for knob in self.knobs:
            if knob.kind is KnobKind.STATE_COLUMN:
                missing = set(knob.targets) - state_columns
            else:
                missing = set(knob.targets) - encoded
            if missing:
                raise ValueError(
                    f"Knob {knob.name!r} targets {sorted(missing)}, which no "
                    "encoded entry provides."
                )

    # --- Queries -----------------------------------------------------------

    def entries_for(self, module: str) -> tuple[ClosureEntry, ...]:
        """The entries of one module."""
        return tuple(entry for entry in self.entries if entry.module == module)

    def defaults(self, module: str, entity: str) -> dict[str, bool | int | float]:
        """Input name -> default, for one module's engine entity."""
        return {
            entry.input: entry.value
            for entry in self.entries_for(module)
            if entry.entity == entity and entry.closure_class is ClosureClass.DEFAULTED
        }

    def state_bindings(
        self, modules: Iterable[str] | None = None
    ) -> tuple[StateBinding, ...]:
        """The declared state bindings, optionally for some modules only."""
        selected = None if modules is None else frozenset(modules)
        return tuple(
            entry.state_binding
            for entry in self.entries
            if entry.state_binding is not None
            and (selected is None or entry.module in selected)
        )

    def graph_node_inputs(self) -> dict[InputRef, str]:
        """Input -> the graph node that supplies it."""
        return {
            entry.ref: entry.node
            for entry in self.entries
            if entry.encoded_by is EncodedBy.GRAPH_NODE
        }

    def new_defaults(self) -> tuple[ClosureEntry, ...]:
        """The defaults marked new: the closure's part of a method review."""
        return tuple(entry for entry in self.entries if entry.new_default)

    def group_knobs(self, values: Mapping[str, object] | None = None) -> GroupKnobs:
        """The encoding a scenario's knob values give.

        Knobs not in ``values`` take their central value.

        Raises:
            ValueError: If ``values`` names an unknown knob or a value the
                knob does not take.
        """
        values = dict(values or {})
        known = {knob.name for knob in self.knobs}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                f"Unknown knobs {unknown}; the closure declares {sorted(known)}."
            )
        allocation: dict[str, Allocation] = {}
        zeroed: set[str] = set()
        factors: dict[str, float] = {}
        state_columns: dict[str, str] = {}
        for knob in self.knobs:
            value = knob.resolve(values.get(knob.name, knob.central))
            # A knob at a value that changes nothing names no target, so the
            # central encoding of a module subset is never refused for it.
            if knob.kind is KnobKind.ALLOCATION:
                if Allocation(value) is not Allocation.REFERENCE_UNIT:
                    allocation.update(dict.fromkeys(knob.targets, Allocation(value)))
            elif knob.kind is KnobKind.SWITCH_ZEROES:
                if not value:
                    zeroed.update(knob.targets)
            elif knob.kind is KnobKind.FACTOR:
                for target in knob.targets:
                    factors[target] = factors.get(target, 1.0) * value
                factors = {name: f for name, f in factors.items() if f != 1.0}
            else:
                (declared,) = knob.targets
                entity = declared.partition(".")[0]
                replacement = f"{entity}.{value}"
                if replacement != declared:
                    state_columns[declared] = replacement
        return GroupKnobs(
            allocation=allocation,
            zeroed=frozenset(zeroed),
            factors=factors,
            state_columns=state_columns,
        )

    # --- Checks ------------------------------------------------------------

    def check(
        self, mapping: ConceptMapping, surface: AxiomInputSurface
    ) -> tuple[str, ...]:
        """Every way the closure disagrees with the mapping or the surface.

        - Each listed module is in the surface, compiled, with the reviewed
          SHA-256, and the mapping names the reviewed engine version.
        - The entries of a module are exactly its surface's root inputs, and
          every input the mapping binds in a listed module is closed.
        - An input is concept-encoded exactly when the mapping binds it with
          a binding that runs on the closure's content basis (on a transport
          frame, one that reads no concept transport drops), or when its
          entry awaits the binding; an awaited binding that exists is
          reported, so the closure is updated when it lands.

        Args:
            mapping: The country's concept mapping.
            surface: The engine-generated input surface.

        Returns:
            One message per disagreement; empty when the closure holds.
        """
        problems: list[str] = []
        if mapping.engine != self.mapping_engine:
            problems.append(
                f"Closure is for {self.mapping_engine}, not {mapping.engine}."
            )
        if mapping.engine_version != self.mapping_engine_version:
            problems.append(
                f"Mapping engine_version {mapping.engine_version} is not the "
                f"reviewed {self.mapping_engine_version}."
            )
        if surface.rulespec_commit != self.surface_rulespec_commit:
            problems.append(
                f"Surface rulespec {surface.rulespec_commit} is not "
                f"{self.surface_rulespec_commit}."
            )
        if surface.engine_commit != self.surface_engine_commit:
            problems.append(
                f"Surface engine {surface.engine_commit} is not {self.surface_engine_commit}."
            )
        bound = {
            binding.ref: binding
            for binding in mapping.bindings
            if binding.module in self.modules
        }
        runs = {ref for ref, binding in bound.items() if self._runs(binding)}
        closed_refs = {entry.ref for entry in self.entries}
        for ref in sorted(set(bound) - closed_refs):
            problems.append(f"{ref.label()} is bound by the mapping but not closed.")
        for path, sha256 in self.modules.items():
            module = surface.modules.get(path)
            if module is None:
                problems.append(f"{path} is not in the input surface.")
                continue
            if module.status != "compiled":
                problems.append(f"{path} did not compile ({module.status}).")
            if module.sha256 != sha256:
                problems.append(f"{path} has SHA-256 {module.sha256}, not {sha256}.")
            expected = set(module.refs())
            closed = {entry.ref for entry in self.entries_for(path)}
            for ref in sorted(expected - closed):
                problems.append(f"{ref.label()} is not closed.")
            for ref in sorted(closed - expected):
                problems.append(f"{ref.label()} is closed but not a root input.")
        for entry in self.entries:
            ref = entry.ref
            by_concept = entry.encoded_by is EncodedBy.CONCEPT_BINDING
            if entry.awaiting is not None:
                if ref in bound:
                    problems.append(
                        f"{ref.label()} is bound now; it no longer awaits "
                        f"{entry.awaiting}."
                    )
                continue
            if by_concept and ref not in bound:
                problems.append(f"{ref.label()} is concept-encoded but not bound.")
            elif by_concept and ref not in runs:
                problems.append(
                    f"{ref.label()} is concept-encoded, but its binding reads a "
                    f"concept {self.content_basis.value} frames drop."
                )
            if not by_concept and ref in runs:
                problems.append(
                    f"{ref.label()} is bound by the mapping but closed as "
                    f"{entry.closure_class.value}"
                    + (f" ({entry.encoded_by.value})" if entry.encoded_by else "")
                    + "."
                )
        return tuple(problems)

    def _runs(self, binding: object) -> bool:
        """Whether a mapping binding runs on this closure's frames."""
        if self.content_basis is not ContentBasis.TRANSPORT:
            return True
        return not any(
            concept(concept_id).transport is TransportRule.DROP
            for concept_id in binding.concepts
        )

    # --- Serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready form that :meth:`from_dict` reads back unchanged."""
        return {
            "format": INPUT_CLOSURE_FORMAT,
            "country": self.country,
            "content_basis": self.content_basis.value,
            "mapping": {
                "engine": self.mapping_engine,
                "engine_version": self.mapping_engine_version,
            },
            "input_surface": {
                "rulespec_commit": self.surface_rulespec_commit,
                "engine": {
                    "repository": self.surface_engine_repository,
                    "commit": self.surface_engine_commit,
                },
            },
            "rulespec_paths": dict(self.modules),
            "engine_optional_evidence": self.engine_optional_evidence,
            "undetermined_judgments": self.undetermined.to_dict(),
            "knobs": [knob.to_dict() for knob in self.knobs],
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> InputClosure:
        """The closure ``data`` describes, validated like any other.

        Unknown top-level fields are allowed (a country pack's resource can
        carry review notes beside the closure); every field the closure
        reads is checked.

        Raises:
            ValueError: If the format is not :data:`INPUT_CLOSURE_FORMAT`, or
                a field is missing or malformed.
        """
        with _parsing("input closure"):
            if not isinstance(data, Mapping):
                raise ValueError("An input closure must be an object.")
            if data.get("format") != INPUT_CLOSURE_FORMAT:
                raise ValueError(
                    f"An input closure must declare format {INPUT_CLOSURE_FORMAT!r}."
                )
            needed = (
                "country",
                "content_basis",
                "mapping",
                "input_surface",
                "rulespec_paths",
                "engine_optional_evidence",
                "undetermined_judgments",
                "knobs",
                "entries",
            )
            missing = [name for name in needed if name not in data]
            if missing:
                raise ValueError(f"An input closure lacks {missing}.")
            mapping = _record_fields(
                data["mapping"],
                "closure mapping",
                required=("engine", "engine_version"),
            )
            surface = _record_fields(
                data["input_surface"],
                "closure input surface",
                required=("rulespec_commit", "engine"),
                optional=("fixture",),
            )
            engine = _record_fields(
                surface["engine"],
                "closure input-surface engine",
                required=("repository", "commit"),
            )
            _text_fields(mapping, "closure mapping", ("engine", "engine_version"))
            _text_fields(surface, "closure input surface", ("rulespec_commit",))
            _text_fields(
                engine, "closure input-surface engine", ("repository", "commit")
            )
            modules = data["rulespec_paths"]
            if not isinstance(modules, Mapping) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in modules.items()
            ):
                raise ValueError(
                    "A closure's 'rulespec_paths' maps module paths to SHA-256 text."
                )
            policy = _record_fields(
                data["undetermined_judgments"],
                "undetermined-judgment policy",
                required=("action", "reason", "alternatives", "new_default"),
            )
            _text_fields(policy, "undetermined-judgment policy", ("action", "reason"))
            if not isinstance(policy["alternatives"], list | tuple):
                raise ValueError("A policy's 'alternatives' must be a list.")
            for name in ("knobs", "entries"):
                if not isinstance(data[name], list | tuple):
                    raise ValueError(f"A closure's {name!r} must be a list.")
            evidence = data["engine_optional_evidence"]
            if not isinstance(evidence, str):
                raise ValueError("A closure's engine evidence must be text.")
            for name in ("country", "content_basis"):
                if not isinstance(data[name], str):
                    raise ValueError(f"A closure's {name!r} must be text.")
            return cls(
                country=data["country"],
                content_basis=ContentBasis(data["content_basis"]),
                mapping_engine=mapping["engine"],
                mapping_engine_version=mapping["engine_version"],
                surface_rulespec_commit=surface["rulespec_commit"],
                surface_engine_repository=engine["repository"],
                surface_engine_commit=engine["commit"],
                modules=modules,
                engine_optional_evidence=evidence,
                undetermined=UndeterminedPolicy(**policy),
                knobs=tuple(Knob.from_dict(knob) for knob in data["knobs"]),
                entries=tuple(
                    ClosureEntry.from_dict(entry) for entry in data["entries"]
                ),
            )


def apply_defaults(
    table: pd.DataFrame, defaults: Mapping[str, bool | int | float]
) -> pd.DataFrame:
    """``table`` with one constant column per default.

    Raises:
        ValueError: If ``table`` already carries a defaulted input: an input
            closed as defaulted is never also encoded.
    """

    clash = sorted(set(defaults) & set(table.columns))
    if clash:
        raise ValueError(f"Defaulted inputs {clash} are already encoded.")
    out = table.copy()
    for name, value in defaults.items():
        dtype = bool if isinstance(value, bool) else type(value)
        out[name] = np.full(len(table), value, dtype=dtype)
    return out


def resolve_judgments(codes: np.ndarray, action: UndeterminedAction) -> np.ndarray:
    """Boolean entitlement from engine judgment codes.

    ``1`` holds and ``-1`` does not; ``0`` (undetermined) follows ``action``.

    Raises:
        ValueError: If a code is not ``-1``, ``0`` or ``1``, or the action
            refuses an undetermined judgment that occurs.
    """

    action = UndeterminedAction(action)
    values = np.asarray(codes)
    if values.size == 0:
        return np.zeros(values.shape, dtype=bool)
    if values.dtype.kind not in "iu" or not np.isin(values, (-1, 0, 1)).all():
        raise ValueError("Judgment codes are -1, 0 or 1.")
    undetermined = values == 0
    if action is UndeterminedAction.REFUSE and undetermined.any():
        raise ValueError(
            f"{int(undetermined.sum())} judgment(s) are undetermined, and the "
            "closure refuses them."
        )
    return np.where(undetermined, action is UndeterminedAction.ELIGIBLE, values == 1)
