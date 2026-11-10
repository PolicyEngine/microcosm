"""The column surface of a cut's reference build (microcosm#1095).

The export-surface and release-input-coverage gates compared every UK cut
with the frozen enhanced-FRS parity reference: the columns it exported and
the inputs it populated. A cut now compares with the current build of its
own line, so the reference step writes that build's surface beside it:

- ``export_columns``: the ``entity.column`` names its release frame exports,
  as :func:`~.terminal_gates.uk_export_candidate_columns` reads a candidate;
- the loader inputs it carries, by the input-coverage gate's own rule
  (:func:`~.release_input_coverage.uk_effective_input_signal`: non-default
  signal against the live engine default on at least
  ``minimum_nondefault_mass_share`` of effective population mass), split by
  how many distinct source households carry them on rows of positive weight:
  ``populated_inputs`` at least ``UK_DISCLOSURE_FLOOR``, ``thin_inputs`` one
  or two (a count below the floor reads "<3").

Names and flags only: the file holds no figure beyond the policy floor it was
measured at. Its identity names the build (``label``, ``sha256``), and a gate
refuses a surface of another build than the cut's reference.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "UK_REFERENCE_SURFACE_KIND",
    "UK_REFERENCE_SURFACE_SCHEMA_VERSION",
    "UKReferenceSurface",
    "load_uk_reference_surface",
    "uk_reference_surface_payload",
]

UK_REFERENCE_SURFACE_KIND = "uk_reference_surface"
UK_REFERENCE_SURFACE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class UKReferenceSurface:
    """What the reference build exports and populates, by name."""

    label: str
    sha256: str
    export_columns: frozenset[str]
    populated_inputs: frozenset[str]
    thin_inputs: frozenset[str]
    minimum_nondefault_mass_share: float

    @property
    def carried_inputs(self) -> frozenset[str]:
        """Every input the reference carries with signal, thin ones included."""

        return self.populated_inputs | self.thin_inputs


def uk_reference_surface_payload(
    h5_path: str | Path,
    *,
    label: str,
    sha256: str,
    engine: Any | None = None,
) -> dict[str, Any]:
    """The surface payload of the build at ``h5_path``, names and flags only.

    ``engine`` is the coverage engine whose loader inputs and defaults the
    gate will judge the candidate by (the live PolicyEngine-UK adapter when
    omitted); the floor is the committed coverage manifest's.
    """

    from microcosm.build.uk_runtime.national_frame import (
        _uk_gate_surface,
        uk_published_release_frame,
    )
    from microcosm.build.uk_runtime.receipt_counts import (
        UK_DISCLOSURE_FLOOR,
        distinct_count,
        source_household_ids,
    )
    from microcosm.build.uk_runtime.release_input_coverage import (
        PolicyEngineUKCoverageEngine,
        load_uk_release_input_coverage_manifest,
        uk_effective_input_signal,
    )
    from microcosm.build.uk_runtime.rowwise_dataset import load_uk_rowwise_dataset
    from microcosm.build.uk_runtime.terminal_gates import uk_export_candidate_columns
    from microcosm.build.uk_runtime.weight_year import uk_recorded_solve_year_frame

    if engine is None:
        engine = PolicyEngineUKCoverageEngine()
    floor = load_uk_release_input_coverage_manifest().effective_mass_coverage
    floor = floor.minimum_nondefault_mass_share
    # The row-wise reader opens both lines' files (the dense line's carries
    # one ``clone_index`` name on every table, which the national loader
    # refuses).
    frame, _provenance = load_uk_rowwise_dataset(h5_path)
    export = uk_published_release_frame(uk_recorded_solve_year_frame(frame))
    surface = _uk_gate_surface(export)
    schema = export.schema
    household = surface.household
    household_sources = pd.Series(
        source_household_ids(household, stage="reference surface"),
        index=household[schema.id_column("household")].to_numpy(),
    )
    person = surface.person
    person_sources = household_sources.reindex(
        person[schema.membership_column("household")].to_numpy()
    ).to_numpy()
    first_member = (
        pd.DataFrame(
            {
                "benunit": person[schema.membership_column("benunit")].to_numpy(),
                "source": person_sources,
            }
        )
        .drop_duplicates("benunit")
        .set_index("benunit")["source"]
    )
    sources_by_entity = {
        "household": household_sources.to_numpy(),
        "person": person_sources,
        "benunit": first_member.reindex(
            surface.benunit[schema.id_column("benunit")].to_numpy()
        ).to_numpy(),
    }
    populated: set[str] = set()
    thin: set[str] = set()
    carried = uk_effective_input_signal(
        surface, engine, minimum_nondefault_mass_share=floor
    )
    for name, (entity, signal) in carried.items():
        # ``signal`` already excludes rows of zero weight.
        carriers = distinct_count(sources_by_entity[entity], signal)
        (populated if carriers >= UK_DISCLOSURE_FLOOR else thin).add(str(name))
    return {
        "schema_version": UK_REFERENCE_SURFACE_SCHEMA_VERSION,
        "kind": UK_REFERENCE_SURFACE_KIND,
        "label": label,
        "sha256": sha256,
        "minimum_nondefault_mass_share": floor,
        "export_columns": sorted(uk_export_candidate_columns(export)),
        "populated_inputs": sorted(populated),
        "thin_inputs": sorted(thin),
    }


def load_uk_reference_surface(source: str | Path) -> UKReferenceSurface:
    """Load a reference surface a cut wrote, refusing any other shape."""

    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: a reference surface must be a JSON object.")
    if (
        payload.get("schema_version") != UK_REFERENCE_SURFACE_SCHEMA_VERSION
        or payload.get("kind") != UK_REFERENCE_SURFACE_KIND
    ):
        raise ValueError(
            f"{path}: not a schema-{UK_REFERENCE_SURFACE_SCHEMA_VERSION} "
            f"{UK_REFERENCE_SURFACE_KIND}."
        )
    label = payload.get("label")
    sha256 = payload.get("sha256")
    if not isinstance(label, str) or label.count("@") != 1:
        raise ValueError(f"{path}: a reference surface's label is <release>@<build>.")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError(f"{path}: a reference surface names its build's sha256.")
    floor = payload.get("minimum_nondefault_mass_share")
    if (
        isinstance(floor, bool)
        or not isinstance(floor, int | float)
        or not math.isfinite(float(floor))
        or not 0.0 < float(floor) <= 1.0
    ):
        raise ValueError(
            f"{path}: a reference surface names the effective-mass floor it "
            "was measured at, in (0, 1]."
        )
    lists = {}
    for key in ("export_columns", "populated_inputs", "thin_inputs"):
        value = payload.get(key)
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise ValueError(f"{path}: {key} must be a list of column names.")
        lists[key] = frozenset(value)
    if lists["populated_inputs"] & lists["thin_inputs"]:
        raise ValueError(f"{path}: a column cannot be both populated and thin.")
    if not lists["export_columns"]:
        raise ValueError(f"{path}: a reference surface exports no column.")
    return UKReferenceSurface(
        label=label,
        sha256=sha256,
        minimum_nondefault_mass_share=float(floor),
        **lists,
    )
