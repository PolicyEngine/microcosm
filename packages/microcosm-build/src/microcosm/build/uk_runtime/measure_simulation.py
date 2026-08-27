"""UK simulated-measure inputs for the national calibration seam."""

from __future__ import annotations

import json
from datetime import date
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    write_uk_national_frame,
)
from microcosm.build.uk_runtime.weighted_integrity import (
    exclusion_evaluation_date,
)
from microcosm.calibrate import TargetRegistry

_ENTITY_LINK = {"benunit": "person_benunit_id", "household": "person_household_id"}
_ENTITY_ID = {"person": "person_id", "benunit": "benunit_id", "household": "household_id"}


def compute_uk_measure_input(
    frame: Any, simulation: Any, entity: str, variable: str, year: int
) -> tuple[np.ndarray, str]:
    """Compute one policyengine-uk variable at the requested entity grain."""

    definition = simulation.tax_benefit_system.variables.get(variable)
    if definition is None:
        raise KeyError(f"policyengine-uk has no variable {variable!r}")
    native = definition.entity.key
    table = frame.table(entity)
    raw = simulation.calculate(variable, year)

    if native == entity:
        values = _values(raw)
        route = "native"
    else:
        native_values = _values(raw)
        if native_values.dtype.kind in {"O", "U", "S", "b"}:
            if entity == "person" and native in ("benunit", "household"):
                native_table = frame.table(native)
                lookup = pd.Series(
                    native_values, index=native_table[_ENTITY_ID[native]].to_numpy()
                )
                keys = frame.table("person")[_ENTITY_LINK[native]].to_numpy()
                values = lookup.loc[keys].to_numpy()
                route = f"categorical_broadcast_{native}_to_person"
            elif native == "person" and native_values.dtype.kind == "b":
                person = frame.table("person")
                collapsed = (
                    pd.Series(native_values.astype(bool))
                    .groupby(person[_ENTITY_LINK[entity]].to_numpy())
                    .max()
                )
                keys = table[_ENTITY_ID[entity]].to_numpy()
                values = collapsed.loc[keys].to_numpy().astype(float)
                route = f"bool_any_collapse_person_to_{entity}"
            else:
                raise KeyError(
                    f"no categorical mapping from {native} to {entity} "
                    f"for {variable!r}"
                )
        else:
            mapped = simulation.calculate(variable, year, map_to=entity)
            values = _values(mapped)
            route = f"map_to_{entity}"

    if len(values) != len(table):
        raise ValueError(
            f"{variable!r} produced {len(values)} values for {len(table)} "
            f"{entity} rows."
        )
    if values.dtype.kind not in {"O", "U", "S"}:
        values = values.astype(float)
    return values, route


class UKMeasureResolver:
    """B2 measure provider backed by a policyengine-uk Microsimulation."""

    def __init__(
        self,
        *,
        simulation_source: Path | None,
        scratch_dir: Path,
        year: int,
        frame: Any,
        microsimulation_factory: Any | None = None,
    ):
        self.year = int(year)
        policyengine_uk = _policyengine_uk_module()
        factory = (
            microsimulation_factory
            if microsimulation_factory is not None
            else policyengine_uk.Microsimulation
        )
        if simulation_source is None:
            if frame is None:
                raise ValueError("scratch-mode UKMeasureResolver requires a frame.")
            scratch_dir.mkdir(parents=True, exist_ok=True)
            source_path = scratch_dir / "simulation-input.h5"
            write_uk_national_frame(frame, source_path)
            mode = "scratch_frame_export"
        else:
            source_path = Path(simulation_source)
            mode = "direct_h5"
            if frame is None:
                frame, _provenance = load_uk_national_frame(source_path)
        self.frame = frame
        self.simulation = factory(dataset=str(source_path))
        self._receipt = {
            "mode": mode,
            "source_path": str(source_path),
            "policyengine_uk_version": str(
                getattr(policyengine_uk, "__version__", "unknown")
            ),
        }
        self.contract_targets = _uk_contract_targets()

    def knows(self, entity: str, variable: str) -> bool:
        """Whether a route in :func:`compute_uk_measure_input` reaches here.

        The resolution loop turns a False into a named refusal; anything this
        answers True to must actually compute, or the operator gets a wrapped
        provider exception instead of the fence's message.
        """

        definition = self.simulation.tax_benefit_system.variables.get(variable)
        if definition is None or entity not in _ENTITY_ID:
            return False
        native = definition.entity.key
        if native == entity:
            return True
        if "person" in (native, entity):
            # Person-to-group and group-to-person are covered by the
            # broadcast, any-collapse, and map_to routes alike.
            return True
        # Group to group has only the numeric map_to route: a categorical or
        # boolean variable has no mapping between benunit and household and
        # refuses here rather than deep inside the provider.
        return getattr(definition, "value_type", None) in (int, float)

    def entity_for(self, variable: str) -> str | None:
        definition = self.simulation.tax_benefit_system.variables.get(variable)
        if definition is None:
            return None
        return str(definition.entity.key)

    def compute(self, entity: str, variable: str) -> tuple[np.ndarray, str]:
        return compute_uk_measure_input(
            self.frame, self.simulation, entity, variable, self.year
        )

    def receipt(self) -> dict[str, str]:
        return dict(self._receipt)


#: Every field a reviewed measure exclusion must carry (the
#: ``weighted_integrity`` reviewed-exclusion record shape, microcosm#757 /
#: the #743-audit adjudication): an exclusion narrows the calibrated target
#: surface, so it names who approved the narrowing, under which adjudication,
#: and for how long — nothing lapses silently and nothing lives forever.
_UK_MEASURE_EXCLUSION_FIELDS = (
    "name",
    "reason",
    "tracking",
    "approved_by",
    "adjudication",
    "approved_on",
    "expires_on",
)


def load_uk_calibration_measure_exclusions(
    path: Path | None = None,
) -> tuple[dict[str, str], ...]:
    """Load and validate the reviewed UK calibration-measure exclusions."""

    if path is None:
        text = (
            importlib_resources.files("microcosm.build.uk")
            .joinpath("calibration_measure_exclusions.json")
            .read_text(encoding="utf-8")
        )
    else:
        text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    allowed_top = {"schema_version", "exclusions"}
    unknown = sorted(set(payload) - allowed_top)
    if unknown:
        raise ValueError(f"unknown top-level exclusion key(s): {unknown}")
    if payload.get("schema_version") != 2:
        raise ValueError("UK calibration measure exclusions schema_version must be 2.")
    exclusions = payload.get("exclusions")
    if not isinstance(exclusions, list):
        raise ValueError("UK calibration measure exclusions must contain a list.")
    seen: set[str] = set()
    loaded: list[dict[str, str]] = []
    for entry in exclusions:
        if not isinstance(entry, dict):
            raise ValueError("UK calibration measure exclusion entries must be objects.")
        unknown_fields = sorted(set(entry) - set(_UK_MEASURE_EXCLUSION_FIELDS))
        if unknown_fields:
            raise ValueError(
                "unknown UK calibration measure exclusion field(s) "
                f"{unknown_fields} on {entry.get('name')!r}."
            )
        record = {field: str(entry.get(field, "")) for field in _UK_MEASURE_EXCLUSION_FIELDS}
        name = record["name"]
        if name in seen:
            raise ValueError(f"duplicate UK calibration measure exclusion {name!r}.")
        for field in _UK_MEASURE_EXCLUSION_FIELDS:
            if not record[field].strip():
                raise ValueError(
                    f"UK calibration measure exclusion {name!r} has empty {field}; "
                    "a narrowed target surface names why, where it is being "
                    "resolved, who approved it, and for how long."
                )
        for field in ("approved_on", "expires_on"):
            try:
                parsed = date.fromisoformat(record[field])
            except ValueError:
                parsed = None
            if parsed is None or parsed.isoformat() != record[field]:
                raise ValueError(
                    f"UK calibration measure exclusion {name!r} {field} must be "
                    f"canonical ISO (YYYY-MM-DD), got {record[field]!r}."
                )
        if record["expires_on"] <= record["approved_on"]:
            raise ValueError(
                f"UK calibration measure exclusion {name!r} expires_on must be "
                "after approved_on."
            )
        seen.add(name)
        loaded.append(record)
    return tuple(loaded)


def apply_uk_calibration_measure_exclusions(
    registry: TargetRegistry,
    exclusions: tuple[dict[str, str], ...],
    *,
    now: date | None = None,
) -> tuple[TargetRegistry, dict[str, dict[str, str]]]:
    """Remove reviewed excluded references from a UK target registry.

    The receipt carries every field the register declares, tracking included:
    an exclusion narrows the calibrated target surface, so the run evidence
    must say where each narrowing is being resolved, not only why. The window
    is enforced at apply time — outside ``approved_on``..``expires_on`` the
    run fails with a correct-or-renew message rather than the narrowing
    lapsing silently or living forever.
    """

    evaluated_on = exclusion_evaluation_date(now)
    for entry in exclusions:
        approved = date.fromisoformat(entry["approved_on"])
        expires = date.fromisoformat(entry["expires_on"])
        if not approved <= evaluated_on <= expires:
            raise ValueError(
                f"UK calibration measure exclusion {entry['name']!r} is outside "
                f"its reviewed window ({entry['approved_on']}..{entry['expires_on']}, "
                f"evaluated {evaluated_on.isoformat()}): correct the underlying "
                f"gap ({entry['tracking']}) or renew the adjudication with a new "
                "approval and expiry."
            )
    declared = {entry["name"]: entry for entry in exclusions}
    matched = {spec.name for spec in registry.specs if spec.name in declared}
    stale = sorted(set(declared) - matched)
    if stale:
        raise ValueError(
            "UK calibration measure exclusion matched zero registry specs: "
            + ", ".join(stale)
        )
    kept = [spec for spec in registry.specs if spec.name not in declared]
    receipt = {
        name: {
            field: declared[name][field]
            for field in _UK_MEASURE_EXCLUSION_FIELDS
            if field != "name"
        }
        for name in sorted(matched)
    }
    for record in receipt.values():
        record["evaluated_on"] = evaluated_on.isoformat()
    return TargetRegistry(kept, country=registry.country), receipt


def _values(raw: Any) -> np.ndarray:
    return np.asarray(raw.values if hasattr(raw, "values") else raw)


def _policyengine_uk_module() -> Any:
    import policyengine_uk

    return policyengine_uk


def _uk_contract_targets() -> dict[str, Any]:
    payload = (
        importlib_resources.files("microcosm.build.uk")
        .joinpath("uk_national_targets.json")
        .read_text(encoding="utf-8")
    )
    contract = json.loads(payload)
    return {target["target_id"]: target for target in contract["targets"]}
