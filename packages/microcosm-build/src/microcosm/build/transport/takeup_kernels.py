"""Weighted entitlement gaps and labelled scenario envelopes.

``takeup.gap@1`` reads published comparators through the same executable
target-reference compiler as calibration. Its references select the measured
columns and filters; each reference has an explicit annualisation factor.
The estimate is the sum of effective weight times annualised measure and the
gap is exactly that estimate minus the selected comparator. Reference
metadata can label any declared table cells, including receipt strata.

``takeup.bands@1`` consumes the gap artifacts from declared scenarios. For
each table cell and each of its estimate, actual and gap values it records
the minimum, central scenario and maximum, naming the scenario at every
bound. Equal-valued ties use the lexicographically first scenario label.
No scenario list, rate or comparator is held in the kernel implementation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

import numpy as np

from microcosm.graph import (
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Numeric,
    SeedSource,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

from . import graph_inputs, target_kernels
from .graph_inputs import (
    canonical_document_param,
    context_frame,
    entities_param,
    require_params,
    sha256_param,
    sha256_text,
    single_source,
    string_param,
)
from .target_kernels import (
    compile_target_surface,
    decode_target_surface,
    parse_reference_document,
)

__all__ = [
    "BANDS_TYPE",
    "GAP_TYPE",
    "TAKEUP_BANDS",
    "TAKEUP_GAP",
    "TakeupBandsKernel",
    "TakeupGapKernel",
    "decode_bands",
    "decode_gap",
    "register_takeup_kernels",
]

GAP_TYPE = ArtifactType("microcosm.takeup.gap", 1)
BANDS_TYPE = ArtifactType("microcosm.takeup.bands", 1)
_GAP_KIND = "transport_entitlement_gap"
_BANDS_KIND = "transport_entitlement_bands"
_METRICS = ("E", "A", "gap")
_ROW_SEMANTICS = ("name", "period", "entity", "measure", "filter", "metadata")
_SCENARIO_SEMANTICS = (
    *_ROW_SEMANTICS,
    "annualisation_factor",
    "reference",
    "ledger_fact_key",
)
_ROW_KEYS = frozenset(
    {*_ROW_SEMANTICS, "annualisation_factor", "reference", "ledger_fact_key", *_METRICS}
)


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be a finite number.")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(
            f"{label} must be finite" + (" and positive." if positive else ".")
        )
    return result


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _document(payload: bytes, kind: str, keys: set[str]) -> dict:
    if type(payload) is not bytes:
        raise TypeError("Transport artifacts are immutable bytes.")
    document = json.loads(payload)
    if (
        not isinstance(document, dict)
        or canonical_json(document) != payload
        or document.get("schema_version") != 1
        or document.get("kind") != kind
        or set(document) != keys | {"schema_version", "kind"}
    ):
        raise ValueError(f"Unsupported or noncanonical {kind} artifact.")
    return document


def _cell(row: Mapping) -> tuple[str, int | str]:
    name = _text(row.get("name"), "gap row.name")
    period = row.get("period")
    if isinstance(period, bool) or not isinstance(period, int | str):
        raise ValueError("gap row.period must be an integer or string.")
    return name, period


def decode_gap(payload: bytes) -> dict:
    """Validate canonical gap bytes, including exact subtraction."""

    document = _document(
        payload,
        _GAP_KIND,
        {
            "country",
            "scenario",
            "weight_entity",
            "weight_kind",
            "facts",
            "references_sha256",
            "gaps",
        },
    )
    for name in ("country", "scenario", "weight_entity", "weight_kind"):
        _text(document[name], f"gap.{name}")
    if not isinstance(document["facts"], dict):
        raise ValueError("gap.facts must carry the comparator source identity.")
    rows = document["gaps"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("A gap artifact needs a non-empty list of table cells.")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != _ROW_KEYS:
            raise ValueError("Unsupported gap row shape.")
        cell = _cell(row)
        if cell in seen:
            raise ValueError("A gap artifact repeats a table cell.")
        seen.add(cell)
        for name in ("entity", "measure", "reference"):
            _text(row[name], f"gap row.{name}")
        if row["reference"] != row["name"]:
            raise ValueError("A gap row must trace to its named reference.")
        if row["filter"] is not None:
            _text(row["filter"], "gap row.filter")
        if not isinstance(row["metadata"], dict):
            raise ValueError("gap row.metadata must be an object.")
        _number(row["annualisation_factor"], "annualisation_factor", positive=True)
        estimate = _number(row["E"], "E")
        actual = _number(row["A"], "A")
        gap = _number(row["gap"], "gap")
        if gap != estimate - actual:
            raise ValueError("The gap must equal E minus A exactly.")
    return document


def decode_bands(payload: bytes) -> dict:
    """Validate a labelled envelope, its ordering and its scenario coverage."""

    document = _document(
        payload,
        _BANDS_KIND,
        {"country", "baseline_scenario", "scenarios", "bands"},
    )
    _text(document["country"], "bands.country")
    baseline = _text(document["baseline_scenario"], "bands.baseline_scenario")
    scenarios = document["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Bands need at least the central scenario.")
    labels = []
    for scenario in scenarios:
        if not isinstance(scenario, dict) or set(scenario) != {
            "scenario",
            "gap_sha256",
        }:
            raise ValueError("Bands scenario rows must name their gap artifact.")
        labels.append(_text(scenario["scenario"], "bands scenario"))
        digest = scenario["gap_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not set(digest) <= set("0123456789abcdef")
        ):
            raise ValueError("Bands scenario gap_sha256 must be a SHA-256 digest.")
    if labels != sorted(set(labels)) or baseline not in labels:
        raise ValueError(
            "Bands scenarios must be distinct, ordered and include the central scenario."
        )
    rows = document["bands"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Bands need a non-empty list of table cells.")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            *_ROW_SEMANTICS,
            "metric",
            "low",
            "baseline",
            "high",
        }:
            raise ValueError("Unsupported bands row shape.")
        cell = (*_cell(row), row["metric"])
        if cell in seen or row["metric"] not in _METRICS:
            raise ValueError("Bands repeat a cell or declare an unknown metric.")
        seen.add(cell)
        values = []
        for bound in ("low", "baseline", "high"):
            value = row[bound]
            if not isinstance(value, dict) or set(value) != {"value", "scenario"}:
                raise ValueError("Every band bound must name its scenario.")
            if value["scenario"] not in labels:
                raise ValueError("A band bound names an undeclared scenario.")
            values.append(_number(value["value"], f"bands.{bound}.value"))
        if row["baseline"]["scenario"] != baseline:
            raise ValueError("The central band must name the central scenario.")
        if not values[0] <= values[1] <= values[2]:
            raise ValueError("Band bounds must be ordered low <= central <= high.")
    return document


def _require_output(
    context: KernelContext, ref: str, name: str, kind: ArtifactType
) -> None:
    declared = {item.name: item.type for item in context.node.artifact_outputs}
    if declared != {name: kind}:
        raise ValueError(
            f"{ref} must declare exactly the {name!r} typed artifact output."
        )


class TakeupGapKernel(KernelBase):
    """``takeup.gap@1``: reference-driven annualised weighted entitlement."""

    ref = "takeup.gap@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas", "scipy"),
    )
    _required = frozenset(
        {
            "country",
            "references",
            "references_sha256",
            "entities",
            "weight_entity",
            "annualisation_factors",
            "scenario",
        }
    )
    _optional = frozenset({"facts_sha256", "annualisation_sha256"})

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            target_kernels,
            *target_kernels._COMPILE_MODULES,
            *target_kernels._MATRIX_MODULES,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context, self.ref, required=self._required, optional=self._optional
        )
        _require_output(context, self.ref, "gap", GAP_TYPE)
        country = string_param(context, self.ref, "country")
        scenario = string_param(context, self.ref, "scenario")
        document = canonical_document_param(context, self.ref, "references")
        if "annualisation_sha256" in context.params:
            sha256_param(context, self.ref, "annualisation_sha256")
        references = parse_reference_document(document, country=country)
        factors = canonical_document_param(context, self.ref, "annualisation_factors")
        names = {reference.name for reference in references}
        if set(factors) != names:
            raise ValueError(
                "Annualisation factors must cover exactly the declared references."
            )
        factors = {
            name: _number(value, f"annualisation factor {name!r}", positive=True)
            for name, value in factors.items()
        }
        weight_entity = string_param(context, self.ref, "weight_entity")
        frame = context_frame(
            context,
            self.ref,
            entities=entities_param(context, self.ref, "entities"),
            weight_entity=weight_entity,
        )
        source = single_source(context, self.ref)
        surface_payload, registry = compile_target_surface(
            context.sources[source],
            document,
            country=country,
            references_sha256=sha256_param(context, self.ref, "references_sha256"),
            expected_facts_sha256=(
                sha256_param(context, self.ref, "facts_sha256")
                if "facts_sha256" in context.params
                else None
            ),
        )
        surface = decode_target_surface(surface_payload)
        problem = target_kernels._compiled_problem(
            frame, registry, weight_entity, self.ref
        )
        weights = frame.weights_for(weight_entity)
        rows = []
        for index, (spec, reference, trace) in enumerate(
            zip(registry.specs, references, surface.trace, strict=True)
        ):
            factor = factors[spec.name]
            with np.errstate(over="raise", invalid="raise"):
                annualised = (
                    problem.matrix[index : index + 1].toarray().ravel() * factor
                )
                estimate = float(np.sum(weights.values * annualised))
            actual = float(spec.value)
            rows.append(
                {
                    "name": spec.name,
                    "period": spec.period,
                    "entity": spec.entity,
                    "measure": spec.measure,
                    "filter": spec.filter,
                    "metadata": dict(reference.metadata),
                    "annualisation_factor": factor,
                    "reference": trace["reference"],
                    "ledger_fact_key": trace["ledger_fact_key"],
                    "E": estimate,
                    "A": actual,
                    "gap": estimate - actual,
                }
            )
        payload = canonical_json(
            {
                "schema_version": 1,
                "kind": _GAP_KIND,
                "country": country,
                "scenario": scenario,
                "weight_entity": weight_entity,
                "weight_kind": weights.kind.value,
                "facts": dict(surface.facts),
                "references_sha256": surface.references_sha256,
                "gaps": rows,
            }
        )
        decode_gap(payload)
        return KernelResult(
            artifacts={"gap": payload},
            receipt={
                "scenario": scenario,
                "n_cells": len(rows),
                "gap_sha256": sha256_text(payload),
            },
        )


class TakeupBandsKernel(KernelBase):
    """``takeup.bands@1``: labelled envelopes of matching scenario tables."""

    ref = "takeup.bands@1"
    # Bounds select decoded Python numbers; no numerical library computes them.
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
    )
    _required = frozenset({"baseline_scenario", "artifact_scenarios"})
    _optional = frozenset({"scenarios_sha256"})

    def implementation_hash(self) -> str:
        return source_hash(
            type(self), graph_inputs, dependencies=self.capabilities.dependencies
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context, self.ref, required=self._required, optional=self._optional
        )
        _require_output(context, self.ref, "bands", BANDS_TYPE)
        baseline = string_param(context, self.ref, "baseline_scenario")
        aliases = canonical_document_param(context, self.ref, "artifact_scenarios")
        if "scenarios_sha256" in context.params:
            sha256_param(context, self.ref, "scenarios_sha256")
        labels = [_text(label, "artifact scenario") for label in aliases.values()]
        if not labels or len(set(labels)) != len(labels) or baseline not in labels:
            raise ValueError(
                "Declared scenarios must be distinct and include the central scenario."
            )
        if set(aliases) != set(context.artifacts):
            raise ValueError(
                "Scenario declarations must cover exactly the consumed artifacts."
            )
        documents = {}
        identities = {}
        for alias, label in aliases.items():
            artifact = context.artifacts[alias]
            if artifact.type != GAP_TYPE:
                raise ValueError(f"{self.ref} needs gap artifacts for every scenario.")
            document = decode_gap(artifact.payload)
            if document["scenario"] != label:
                raise ValueError(
                    "A consumed gap artifact disagrees with its declared scenario."
                )
            documents[label] = document
            identities[label] = sha256_text(artifact.payload)
        central = documents[baseline]
        cells = {
            label: {_cell(row): row for row in document["gaps"]}
            for label, document in documents.items()
        }
        for label, document in documents.items():
            if document["country"] != central["country"] or set(cells[label]) != set(
                cells[baseline]
            ):
                raise ValueError(
                    "Scenario gap artifacts must cover the same country and table cells."
                )
            for cell, row in cells[label].items():
                if any(
                    row[name] != cells[baseline][cell][name]
                    for name in _SCENARIO_SEMANTICS
                ):
                    raise ValueError(
                        "Scenario gap artifacts disagree on a table cell's semantics."
                    )
        rows = []
        ordered_labels = sorted(documents)
        for cell in sorted(cells[baseline], key=lambda value: canonical_json(value)):
            central_row = cells[baseline][cell]
            for metric in _METRICS:
                low = min(ordered_labels, key=lambda label: cells[label][cell][metric])
                high = min(
                    ordered_labels, key=lambda label: -cells[label][cell][metric]
                )
                rows.append(
                    {
                        **{name: central_row[name] for name in _ROW_SEMANTICS},
                        "metric": metric,
                        "low": {"value": cells[low][cell][metric], "scenario": low},
                        "baseline": {
                            "value": central_row[metric],
                            "scenario": baseline,
                        },
                        "high": {"value": cells[high][cell][metric], "scenario": high},
                    }
                )
        payload = canonical_json(
            {
                "schema_version": 1,
                "kind": _BANDS_KIND,
                "country": central["country"],
                "baseline_scenario": baseline,
                "scenarios": [
                    {"scenario": label, "gap_sha256": identities[label]}
                    for label in ordered_labels
                ],
                "bands": rows,
            }
        )
        decode_bands(payload)
        return KernelResult(
            artifacts={"bands": payload},
            receipt={
                "n_scenarios": len(labels),
                "n_cells": len(rows),
                "bands_sha256": sha256_text(payload),
            },
        )


TAKEUP_GAP = TakeupGapKernel()
TAKEUP_BANDS = TakeupBandsKernel()


def register_takeup_kernels(registry: KernelRegistry) -> None:
    """Register the two kernels without modifying a global registry."""

    for kernel in (TAKEUP_GAP, TAKEUP_BANDS):
        registry.register(kernel)
