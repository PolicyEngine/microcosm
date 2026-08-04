"""Take-up contract inventory: the checked-in classification, engine-asserted.

policyengine-us gates several benefit programs on a ``takes_up_*`` input flag
that defaults to ``True`` (universal take-up) when the dataset provides nothing.
The retired enhanced-CPS pipeline seeded those flags so take-up-gated programs
did not ship at 100% participation; populace produces almost none of them, so
without this work H5 consumers inherit mechanical universal take-up (populace
#170, #70, #312).

This module owns the *inventory*: a checked-in table
(``populace/build/us/take_up_contract.json``) recording, per program, the engine
facts that decide treatment (entity, default, and the derived ``engine_class``)
and the curated populace treatment (seed / count_calibrated / rate_unsourced /
model_simulated / out_of_scope / near_universal) with the administrative
provenance of any rate used. The engine facts are asserted against the installed engine by
:func:`assert_take_up_contract_current`, so the classification tracks the pinned
policyengine-us version instead of a remembered snapshot -- the same
metadata-derivation doctrine as the #301 formula-owned guard.

The seeding stages themselves live in :mod:`populace.build.us_runtime.take_up`;
this module is the contract they and the release diagnostics read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine

__all__ = [
    "TAKE_UP_CONTRACT_ENGINE_FACT_KEYS",
    "TakeUpContract",
    "TakeUpProgram",
    "assert_take_up_contract_current",
    "assert_take_up_treatments_consistent",
    "count_calibrated_take_up_programs",
    "load_take_up_contract",
    "seeded_take_up_programs",
    "take_up_contract_identity",
]

#: Engine-derived fields on each program entry that a test asserts against the
#: installed policyengine-us. These are exactly the keys
#: :meth:`PolicyEngineUSEngine.take_up_contract` reports and must not drift from
#: the pinned engine; the remaining fields are curated treatment/provenance.
TAKE_UP_CONTRACT_ENGINE_FACT_KEYS: tuple[str, ...] = (
    "entity",
    "value_type",
    "default",
    "engine_class",
)

_VALID_TREATMENTS = frozenset(
    {
        "seed",
        "count_calibrated",
        "rate_unsourced",
        "model_simulated",
        "out_of_scope",
        "near_universal",
    }
)


@dataclass(frozen=True)
class TakeUpProgram:
    """One program row of the take-up contract inventory."""

    variable: str
    engine_class: str
    entity: str
    value_type: str
    default: object
    populace_treatment: str
    raw: Mapping[str, Any]

    @property
    def is_seeded(self) -> bool:
        """Whether populace assigns this flag from a sourced rate."""
        return self.populace_treatment == "seed"

    @property
    def rate(self) -> Mapping[str, Any]:
        """The rate block (value(s), source, agency, status) if present."""
        rate = self.raw.get("rate")
        return rate if isinstance(rate, Mapping) else {}

    def engine_facts(self) -> dict[str, object]:
        """The engine-asserted subset of this row, in canonical key order."""
        return {
            "entity": self.entity,
            "value_type": self.value_type,
            "default": self.default,
            "engine_class": self.engine_class,
        }


@dataclass(frozen=True)
class TakeUpContract:
    """The full checked-in take-up contract inventory."""

    version: int
    country: str
    asserted_constraint: str
    inventory_built_against: str
    programs: tuple[TakeUpProgram, ...]

    def program_map(self) -> dict[str, TakeUpProgram]:
        return {program.variable: program for program in self.programs}

    def seeded(self) -> tuple[TakeUpProgram, ...]:
        return tuple(program for program in self.programs if program.is_seeded)


def _contract_path():
    return files("populace.build.us").joinpath("take_up_contract.json")


@lru_cache(maxsize=1)
def load_take_up_contract() -> TakeUpContract:
    """Load and validate the checked-in take-up contract inventory.

    Raises:
        ValueError: If the table is malformed (missing keys, unknown
            treatment, duplicate program).
    """
    raw = json.loads(_contract_path().read_text())
    version = raw.get("version")
    country = raw.get("country")
    if not isinstance(version, int) or version < 1:
        raise ValueError("take-up contract requires a positive integer 'version'.")
    if not isinstance(country, str) or not country:
        raise ValueError("take-up contract requires a non-empty 'country'.")
    asserted = raw.get("asserted_engine", {})
    if not isinstance(asserted, Mapping):
        raise ValueError("take-up contract 'asserted_engine' must be a mapping.")

    programs: list[TakeUpProgram] = []
    seen: set[str] = set()
    for entry in raw.get("programs", ()):
        if not isinstance(entry, Mapping):
            raise ValueError("each take-up contract program must be a mapping.")
        variable = entry.get("variable")
        if not isinstance(variable, str) or not variable:
            raise ValueError("take-up contract program requires a 'variable' name.")
        if variable in seen:
            raise ValueError(f"duplicate take-up contract program: {variable!r}.")
        seen.add(variable)
        treatment = entry.get("populace_treatment")
        if treatment not in _VALID_TREATMENTS:
            raise ValueError(
                f"program {variable!r} has invalid populace_treatment "
                f"{treatment!r}; expected one of {sorted(_VALID_TREATMENTS)}."
            )
        missing = [
            key
            for key in ("engine_class", "entity", "value_type", "default")
            if key not in entry
        ]
        if missing:
            raise ValueError(
                f"program {variable!r} is missing engine fact key(s): {missing}."
            )
        _validate_seed_provenance(variable, treatment, entry)
        _validate_count_calibration(variable, treatment, entry)
        programs.append(
            TakeUpProgram(
                variable=variable,
                engine_class=str(entry["engine_class"]),
                entity=str(entry["entity"]),
                value_type=str(entry["value_type"]),
                default=entry["default"],
                populace_treatment=str(treatment),
                raw=dict(entry),
            )
        )

    return TakeUpContract(
        version=version,
        country=country,
        asserted_constraint=str(asserted.get("constraint", "")),
        inventory_built_against=str(asserted.get("inventory_built_against", "")),
        programs=tuple(programs),
    )


def take_up_contract_identity(
    contract: TakeUpContract | None = None,
) -> dict[str, object]:
    """Return the complete canonical identity surface for the take-up contract.

    Cached build artifacts must call this one constructor instead of selecting
    contract fields independently. Program order and each full checked-in row
    remain identity-bearing alongside every structured contract-level field.
    """

    resolved = contract if contract is not None else load_take_up_contract()
    return {
        "version": resolved.version,
        "country": resolved.country,
        "asserted_constraint": resolved.asserted_constraint,
        "inventory_built_against": resolved.inventory_built_against,
        "programs": [deepcopy(dict(program.raw)) for program in resolved.programs],
    }


def _validate_seed_provenance(
    variable: str, treatment: str, entry: Mapping[str, Any]
) -> None:
    """A seeded program must carry a sourced rate; enforce the provenance rule.

    This is the checked-in half of the hard rule that a rate without a
    traceable administrative source may not drive a seed. It refuses a table
    that marks a program ``seed`` without a rate block that names a source and
    an administrative-grade status, so the fabricated-rate class ledger#77
    removed cannot re-enter through the inventory.
    """
    if treatment != "seed":
        return
    rate = entry.get("rate")
    if not isinstance(rate, Mapping):
        raise ValueError(
            f"seeded program {variable!r} requires a 'rate' block with provenance."
        )
    has_value = "value" in rate or "values_by_num_children" in rate
    if not has_value:
        raise ValueError(
            f"seeded program {variable!r} rate must carry a value or "
            "values_by_num_children."
        )
    if not str(rate.get("source") or ""):
        raise ValueError(
            f"seeded program {variable!r} rate requires a 'source' citation."
        )
    status = str(rate.get("status") or "")
    if not status.startswith("sourced"):
        raise ValueError(
            f"seeded program {variable!r} rate status {status!r} is not "
            "administrative-grade; only 'sourced*' statuses may drive a seed."
        )


def _validate_count_calibration(
    variable: str, treatment: str, entry: Mapping[str, Any]
) -> None:
    """A count-calibrated program must name its anchor and count targets.

    ``count_calibrated`` is the doctrine-compliant path when no administrative
    participation *rate* exists but administrative enrollment *counts* do (the
    native ACA stage precedent): assignment anchors on survey-reported coverage
    and calibrates the fill to Ledger count facts, so no rate is ever cited.
    The entry must therefore carry a ``calibration`` block naming the anchor
    column and the target fact family — and must NOT carry a sourced rate, or
    the treatment is misclassified (a sourced rate is what ``seed`` is for).
    """
    if treatment != "count_calibrated":
        return
    calibration = entry.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError(
            f"count_calibrated program {variable!r} requires a 'calibration' "
            "block naming its anchor and count targets."
        )
    targets = calibration.get("targets")
    if not isinstance(targets, (list, tuple)) or not targets:
        raise ValueError(
            f"count_calibrated program {variable!r} calibration requires a "
            "non-empty 'targets' list of Ledger count-fact families."
        )
    if not str(calibration.get("anchor") or ""):
        raise ValueError(
            f"count_calibrated program {variable!r} calibration requires an "
            "'anchor' reported-coverage column."
        )
    rate = entry.get("rate")
    if isinstance(rate, Mapping) and str(rate.get("status") or "").startswith(
        "sourced"
    ):
        raise ValueError(
            f"count_calibrated program {variable!r} carries a sourced rate; "
            "a program with an administrative rate should be 'seed', not "
            "'count_calibrated'."
        )


def seeded_take_up_programs() -> tuple[TakeUpProgram, ...]:
    """The programs populace seeds, in table order."""
    return load_take_up_contract().seeded()


def count_calibrated_take_up_programs() -> tuple[TakeUpProgram, ...]:
    """The programs populace assigns via anchored count-calibration."""
    return tuple(
        program
        for program in load_take_up_contract().programs
        if program.populace_treatment == "count_calibrated"
    )


def assert_take_up_contract_current(engine: PolicyEngineUSEngine | None = None) -> None:
    """Fail when the checked-in inventory drifts from the installed engine.

    Recomputes the engine facts (entity, value_type, default, engine_class) for
    every take-up variable from :meth:`PolicyEngineUSEngine.take_up_contract`
    and compares them to the checked-in table. Raises if:

    - the engine carries a take-up variable the table omits (a new flag, or a
      migration that renamed one) -- an unlisted default-True flag would ship
      universal participation silently;
    - the table lists a variable the engine no longer has;
    - any asserted engine fact disagrees (e.g. policyengine-us gives a
      previously data-seeded flag a formula, flipping it to model_simulated, so
      populace must stop seeding it).

    The curated treatment/provenance fields are not asserted here -- they are a
    human decision -- but a treatment that contradicts the engine facts (seeding
    a model_simulated flag, or leaving a data_seeded flag with no decision) is
    caught by :func:`assert_take_up_treatments_consistent`.

    Raises:
        AssertionError: On any drift, with a message naming the discrepancies.
        ImportError: If ``policyengine_us`` is not installed.
    """
    engine = engine or PolicyEngineUSEngine()
    engine_contract = engine.take_up_contract()
    table = load_take_up_contract().program_map()

    engine_names = set(engine_contract)
    table_names = set(table)
    problems: list[str] = []

    for missing in sorted(engine_names - table_names):
        info = engine_contract[missing]
        problems.append(
            f"{missing}: present in engine (class={info['engine_class']}, "
            f"default={info['default']!r}) but absent from take_up_contract.json; "
            "add it before release or an unlisted take-up flag ships silently."
        )
    for extra in sorted(table_names - engine_names):
        problems.append(
            f"{extra}: listed in take_up_contract.json but the installed engine "
            "has no such take-up variable; remove or update the entry."
        )

    for name in sorted(engine_names & table_names):
        engine_facts = {
            key: engine_contract[name][key] for key in TAKE_UP_CONTRACT_ENGINE_FACT_KEYS
        }
        table_facts = table[name].engine_facts()
        for key in TAKE_UP_CONTRACT_ENGINE_FACT_KEYS:
            if engine_facts[key] != table_facts[key]:
                problems.append(
                    f"{name}.{key}: engine says {engine_facts[key]!r}, "
                    f"table says {table_facts[key]!r}."
                )

    if problems:
        raise AssertionError(
            "take_up_contract.json is stale against the installed policyengine-us "
            "("
            + str(len(problems))
            + " discrepancy(ies)):\n  - "
            + "\n  - ".join(problems)
        )


def assert_take_up_treatments_consistent() -> None:
    """Fail when a curated treatment contradicts the engine class it carries.

    - A ``model_simulated`` flag must not be marked ``seed`` (populace would
      fight the engine's own draw).
    - A ``data_seeded`` flag must carry a decision other than
      ``model_simulated`` (it is an input leaf, not engine-computed).
    - A ``dead`` flag should not be seeded (seeding changes no output).

    Raises:
        AssertionError: On any inconsistency.
    """
    problems: list[str] = []
    for program in load_take_up_contract().programs:
        cls, treatment, name = (
            program.engine_class,
            program.populace_treatment,
            program.variable,
        )
        if cls == "model_simulated" and treatment == "seed":
            problems.append(
                f"{name}: engine_class model_simulated must not be seeded "
                "(the engine draws take-up itself)."
            )
        if cls == "model_simulated" and treatment != "model_simulated":
            # A model-simulated flag's only populace-native option besides
            # documenting it as model_simulated is out_of_scope (another
            # workstream owns the rate calibration); flag anything else.
            if treatment not in {"model_simulated", "out_of_scope"}:
                problems.append(
                    f"{name}: engine_class model_simulated but treatment "
                    f"{treatment!r}; expected model_simulated or out_of_scope."
                )
        if cls == "data_seeded" and treatment == "model_simulated":
            problems.append(
                f"{name}: engine_class data_seeded but treatment marks it "
                "model_simulated; the engine has no formula for it."
            )
        if cls == "dead" and treatment == "seed":
            problems.append(
                f"{name}: engine_class dead (no consumer) must not be seeded."
            )
    if problems:
        raise AssertionError(
            "take_up_contract.json treatments contradict engine classes:\n  - "
            + "\n  - ".join(problems)
        )
