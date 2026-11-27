"""A spec country's transport build folder, for the release assembler's tests.

:func:`write_transport_build_folder` writes the files that
``microcosm.build.transport.cli`` writes under ``--out`` and that the release
assembler (``microcosm.build.spec_release_assembly``) reads: the H5 the export
step writes, its export descriptor, the schema 9 calibration diagnostics, the
run manifest and ``build-status.json``. The tables, rows and receipts are
small and synthetic, and no graph runs. The calibration rows are the
country's executable demography references, read from its spec, so the
release files see the target names a build of that spec compiles.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.artifact_files import file_artifact
from microcosm.build.transport.cli import EXPORT_FILENAME
from microcosm.build.transport.compose import load_transport_spec
from microcosm.build.transport.terminal_kernels import (
    _EXPORT_KIND,
    _content,
)
from microcosm.diagnostics import (
    CalibrationDiagnosticsV8,
    catalog_calibration_diagnostics,
)
from microcosm.frame.adapters.axiom import AxiomEntityTableDataset
from microcosm.graph import (
    Capabilities,
    Determinism,
    KernelRole,
    NodeReceipt,
    RunManifest,
    SeedSource,
    StructuralDelta,
)

#: The Microcosm commit a fixture build records that it ran at.
BUILD_COMMIT = "5fa48f07436a806ad75ff76fd22cfb8613bddbe0"
#: When a fixture build's run finished, and the build-ID stamp it gives.
FINISHED_AT = "2026-12-01T10:20:30+00:00"
FINISHED_STAMP = "20261201T102030Z"
#: The Python and package versions a fixture build records.
BUILD_RUNTIME = {
    "python": "3.13.5",
    "microcosm-build": "0.1.0",
    "microcosm-data": "0.1.0",
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "axiom-rules-engine": "0.4.0",
}
#: The code record of a fixture build run from a clean checkout.
CLEAN_CODE = {
    "repository": "PolicyEngine/microcosm",
    "git_commit": BUILD_COMMIT,
    "git_dirty": False,
}
#: The node IDs of the fixture run's gates.
GATE_NODES = ("be.gates.battery", "be.export.readback")
#: A fixture build's household weights, one per household.
HOUSEHOLD_WEIGHTS = (1250.5, 980.25, 2010.0)


@dataclass(frozen=True)
class TransportBuildFolder:
    """A fixture build folder and what a test needs to know about it."""

    path: Path
    country: str
    period: int
    demography_targets: tuple[str, ...]


def _executable(reference: Mapping) -> bool:
    status = reference.get("metadata", {}).get("activation_status", "")
    return status in {"", "active"}


def demography_references(country: str = "be") -> tuple[Mapping, ...]:
    """The country's executable demography references, in spec order."""
    references = load_transport_spec(country)["resources"]["target_references"]
    return tuple(
        reference
        for reference in references["target_references"]
        if reference.get("family") == "demography" and _executable(reference)
    )


def _hierarchy(reference: Mapping, labels: Mapping) -> dict:
    selector = reference["ledger_selector"]
    category = labels["categories"][reference["category_id"]]
    provider_id = category["provider_id"]
    return {
        "provider": {
            "id": provider_id,
            "label": labels["providers"][provider_id]["label"],
        },
        "category": {
            "id": reference["category_id"],
            "label": category["label"],
            "provider_id": provider_id,
        },
        "geography": {
            "id": selector["geography_id"],
            "label": selector["geography_id"],
            "level": selector["geography_level"],
        },
        "dimensions": [
            {
                "id": dimension,
                "label": dimension,
                "value_id": value,
                "value_label": value,
            }
            for dimension, value in selector["dimension_values"].items()
        ],
        "target": {"id": reference["name"], "label": reference["name"]},
    }


def demography_target(index: int) -> float:
    """The benchmark a fixture row gives the demography reference at ``index``."""
    return 100_000.0 + 1_000.0 * index


def demography_estimate(index: int) -> float:
    """The calibrated estimate a fixture row gives the reference at ``index``."""
    return demography_target(index) * 1.01


def calibration_diagnostics(
    country: str = "be",
    *,
    period: int = 2026,
    omit: Sequence[str] = (),
) -> dict:
    """Schema 9 diagnostics with one row per executable demography reference.

    A reference ``omit`` names has no row, as if the build had not
    calibrated it.
    """
    labels = load_transport_spec(country)["resources"]["target_references"]["hierarchy"]
    rows = []
    for index, reference in enumerate(demography_references(country)):
        if reference["name"] in omit:
            continue
        target = demography_target(index)
        rows.append(
            {
                "name": f"{reference['name']}@{period}",
                "target_name": reference["name"],
                "period": period,
                "entity": reference["entity"],
                "measure": {"kind": "column", "name": reference["measure"]},
                "filter": {"kind": "column", "name": reference["filter"]},
                "source": "Fixture Chronicle fact",
                "metadata": {},
                "target": target,
                "compiled_target": target,
                "initial_estimate": target * 0.9,
                "final_estimate": demography_estimate(index),
                "relative_error": 0.01,
                "within_tolerance": True,
                "registry": {
                    "family": reference["family"],
                    "se": None,
                    "signed": False,
                    "notes": "",
                },
                "hierarchy": _hierarchy(reference, labels),
            }
        )
    diagnostics = {
        "schema_version": 8,
        "weight_entity": "household",
        "options": {"epochs": 120},
        "target_surface": {
            "schema_version": 1,
            "weight_entity": "household",
            "n_targets": len(rows),
            "n_records": len(HOUSEHOLD_WEIGHTS),
            "constraint_matrix": {
                "rows": len(rows),
                "columns": len(HOUSEHOLD_WEIGHTS),
                "nnz": len(rows),
            },
            "sha256": "e" * 64,
            "names_sha256": "b" * 64,
            "values_sha256": "f" * 64,
        },
        "target_registry": {
            "country": country,
            "version": "registryabc123",
            "n_specs": len(rows),
        },
        "loss_trajectory": [1.0, 0.5],
        "skipped": [],
        "targets": rows,
        "l0_lambda": 0.0,
        "n_nonzero": len(HOUSEHOLD_WEIGHTS),
        "n_records": len(HOUSEHOLD_WEIGHTS),
        "initial_loss": 1.0,
        "final_loss": 0.5,
        "fraction_within_10pct": 1.0,
        "effective_sample_size": 2.5,
        "realized_max_weight_ratio": 1.5,
        "top_1pct_weight_share": 0.4,
        "past_cap_census": None,
        "diagnostic_warnings": [],
    }
    model = CalibrationDiagnosticsV8.model_validate(diagnostics)
    return json.loads(
        catalog_calibration_diagnostics(model).model_dump_json(exclude_defaults=True)
    )


def _tables() -> dict[str, pd.DataFrame]:
    households = len(HOUSEHOLD_WEIGHTS)
    return {
        "household": pd.DataFrame(
            {
                "household_id": np.arange(1, households + 1, dtype=np.int64),
                "household_weight": np.asarray(HOUSEHOLD_WEIGHTS, dtype=np.float64),
            }
        ),
        "person": pd.DataFrame(
            {
                "person_id": np.arange(1, 2 * households + 1, dtype=np.int64),
                "person_household_id": np.repeat(
                    np.arange(1, households + 1, dtype=np.int64), 2
                ),
                "age": np.asarray([34, 5, 71, 68, 45, 17], dtype=np.int64),
            }
        ),
    }


def export_descriptor(tables: Mapping[str, pd.DataFrame], period: int) -> dict:
    """The descriptor ``export.prepare@1`` gives these tables."""
    entities = tuple(tables)
    return {
        "schema_version": 1,
        "kind": _EXPORT_KIND,
        **_content(tables, entities=entities, time_period=period),
        "weight_entity": "household",
        "weight_kind": "calibrated",
        "bindings": {},
        "graph_only": ["strata", "metadata", "mass_log", "weight_kind"],
    }


def _capabilities(role: KernelRole) -> Capabilities:
    return Capabilities(
        determinism=Determinism.DETERMINISTIC,
        seed_source=SeedSource.NONE,
        structural=StructuralDelta.NONE,
        role=role,
        dependencies=("numpy",),
    )


def run_manifest(country: str, *, gate_outcome: str = "pass") -> RunManifest:
    """A run manifest whose gate nodes record ``gate_outcome``."""
    nodes = {
        node_id: NodeReceipt(
            key=f"{index:x}" * 64,
            hit=False,
            seed=0,
            kernel_ref="gates.battery@1",
            kernel_impl_hash="c" * 64,
            capabilities=_capabilities(KernelRole.GATE),
            receipt={"outcome": gate_outcome},
        )
        for index, node_id in enumerate(GATE_NODES, start=1)
    }
    nodes["be.calibrate"] = NodeReceipt(
        key="d" * 64,
        hit=False,
        seed=0,
        kernel_ref="calibrate.ordered_adam@1",
        kernel_impl_hash="c" * 64,
        capabilities=_capabilities(KernelRole.COMPUTE),
        receipt={},
    )
    return RunManifest(
        country=country,
        nodes=nodes,
        started_at="2026-12-01T09:00:00+00:00",
        finished_at=FINISHED_AT,
        host="fixture-host",
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=1) + "\n")


def write_transport_build_folder(
    root: Path,
    *,
    country: str = "be",
    period: int = 2026,
    code: Mapping[str, object] | None = CLEAN_CODE,
    runtime: Mapping[str, str] | None = BUILD_RUNTIME,
    not_yet_produced: Sequence[str] = (),
    gate_outcome: str = "pass",
    omit_targets: Sequence[str] = (),
    record_spec: bool = True,
    spec_hash_changes: Mapping[str, str] | None = None,
) -> TransportBuildFolder:
    """Write a transport build folder under ``root`` and describe it.

    ``code`` and ``runtime`` are what ``build-status.json`` records the build
    ran on (by default a clean checkout at :data:`BUILD_COMMIT`); None
    leaves a record out, as a build before tracker 5b · 13 did.
    ``not_yet_produced`` lists the outputs the build says it still owes, and
    ``gate_outcome`` is every gate node's outcome. The build records the
    digests of the country's spec resources it ran on, those of the current
    spec with ``spec_hash_changes`` applied, unless ``record_spec`` is false.
    """
    root.mkdir(parents=True, exist_ok=True)
    tables = _tables()
    dataset = root / EXPORT_FILENAME
    AxiomEntityTableDataset(tables=tables, time_period=period).save(dataset)
    _write_json(root / "export-descriptor.json", export_descriptor(tables, period))
    diagnostics_path = root / "calibration-diagnostics.json"
    _write_json(
        diagnostics_path,
        calibration_diagnostics(country, period=period, omit=omit_targets),
    )
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(
        run_manifest(country, gate_outcome=gate_outcome).to_json_bytes()
    )
    status: dict[str, object] = {
        "schema_version": 1,
        "country": country,
        "local_only": True,
        "content_basis": "Transported donor support records reweighted to "
        "destination facts; not observed destination microdata.",
        "produced": {
            "skeleton_dataset": file_artifact(dataset),
            "export_descriptor": file_artifact(root / "export-descriptor.json"),
            "run_manifest": file_artifact(manifest_path),
            "calibration_diagnostics": file_artifact(diagnostics_path),
        },
        "not_yet_produced": list(not_yet_produced),
    }
    if code is not None:
        status["code"] = dict(code)
    if runtime is not None:
        status["runtime"] = dict(runtime)
    if record_spec:
        hashes = dict(load_transport_spec(country)["resource_hashes"])
        hashes.update(spec_hash_changes or {})
        status["spec"] = {"country": country, "resource_hashes": hashes}
    _write_json(root / "build-status.json", status)
    return TransportBuildFolder(
        path=root,
        country=country,
        period=period,
        demography_targets=tuple(
            reference["name"] for reference in demography_references(country)
        ),
    )
