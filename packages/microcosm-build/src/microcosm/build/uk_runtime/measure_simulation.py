"""UK simulated-measure inputs for the national calibration seam."""

from __future__ import annotations

import json
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    write_uk_national_frame,
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
        definition = self.simulation.tax_benefit_system.variables.get(variable)
        if definition is None:
            return False
        native = definition.entity.key
        return native == entity or entity in _ENTITY_ID

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
    if payload.get("schema_version") != 1:
        raise ValueError("UK calibration measure exclusions schema_version must be 1.")
    exclusions = payload.get("exclusions")
    if not isinstance(exclusions, list):
        raise ValueError("UK calibration measure exclusions must contain a list.")
    seen: set[str] = set()
    loaded: list[dict[str, str]] = []
    for entry in exclusions:
        if not isinstance(entry, dict):
            raise ValueError("UK calibration measure exclusion entries must be objects.")
        name = str(entry.get("name", ""))
        reason = str(entry.get("reason", ""))
        tracking = str(entry.get("tracking", ""))
        if name in seen:
            raise ValueError(f"duplicate UK calibration measure exclusion {name!r}.")
        if not reason.strip():
            raise ValueError(f"UK calibration measure exclusion {name!r} has empty reason.")
        seen.add(name)
        loaded.append({"name": name, "reason": reason, "tracking": tracking})
    return tuple(loaded)


def apply_uk_calibration_measure_exclusions(
    registry: TargetRegistry, exclusions: tuple[dict[str, str], ...]
) -> tuple[TargetRegistry, dict[str, str]]:
    """Remove reviewed excluded references from a UK target registry."""

    reasons = {entry["name"]: entry["reason"] for entry in exclusions}
    matched = {spec.name for spec in registry.specs if spec.name in reasons}
    stale = sorted(set(reasons) - matched)
    if stale:
        raise ValueError(
            "UK calibration measure exclusion matched zero registry specs: "
            + ", ".join(stale)
        )
    kept = [spec for spec in registry.specs if spec.name not in reasons]
    receipt = {name: reasons[name] for name in sorted(matched)}
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
