"""Weighted entitlement gaps and labelled scenario envelopes.

``takeup.gap@1`` reads published comparators through the same executable
target-reference compiler as calibration. Its references select the measured
columns and filters; each reference has an explicit annualisation factor.
The estimate is the sum of effective weight times annualised measure and the
gap is exactly that estimate minus the selected comparator. Reference
metadata can label any declared table cells, including receipt strata.
An optional declared reporting grid adds entitlement breakdowns without
subgroup payment comparators. Its support diagnostics merge clone weights and
dollar contributions by explicit original-source household identity. Screened
cells carry suppression reasons and no reportable numeric values.

``takeup.bands@1`` consumes the gap artifacts from declared scenarios. For
each table cell and each of its estimate, actual and gap values it records
the minimum, central scenario and maximum, naming the scenario at every
bound. Equal-valued ties use the lexicographically first scenario label.
No scenario list, rate or comparator is held in the kernel implementation.
``takeup.cell_diagnostics@1`` combines every declared scenario's diagnostics;
absent scenarios refuse the collection before the release gates read it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from itertools import product

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

from . import graph_inputs, target_kernels, weight_quality
from .artifact_types import CELL_DIAGNOSTICS_TYPE
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
    "CELL_DIAGNOSTICS_TYPE",
    "GAP_TYPE",
    "TAKEUP_BANDS",
    "TAKEUP_GAP",
    "TAKEUP_CELL_DIAGNOSTICS",
    "TakeupBandsKernel",
    "TakeupGapKernel",
    "decode_bands",
    "decode_gap",
    "decode_cell_diagnostics",
    "source_cell_diagnostics",
    "apply_cell_support",
    "reporting_cell_declarations",
    "register_takeup_kernels",
]

GAP_TYPE = ArtifactType("microcosm.takeup.gap", 1)
BANDS_TYPE = ArtifactType("microcosm.takeup.bands", 1)
_GAP_KIND = "transport_entitlement_gap"
_BANDS_KIND = "transport_entitlement_bands"
_CELL_DIAGNOSTICS_KIND = "transport_cell_support_diagnostics"
_CELL_FIELDS = frozenset(
    {
        "source_household_ess",
        "positive_contributors",
        "contribution_cv",
        "maximum_single_donor_dollar_share",
        "dollar_contribution_ess",
        "structural_zero",
    }
)
_SUPPORT_FIELDS = frozenset(
    {
        "minimum_cell_ess",
        "maximum_contribution_cv",
        "maximum_single_donor_dollar_share",
        "target_minimum_cell_ess",
        "target_maximum_single_donor_dollar_share",
    }
)
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


def _kish(values: np.ndarray) -> float:
    maximum = float(np.max(values, initial=0))
    if maximum == 0:
        return float(0)
    scaled = values / maximum
    return float(np.sum(scaled) ** 2 / np.sum(scaled**2))


def source_cell_diagnostics(
    weights, annualised_amounts, source_household_ids, *, support=None
) -> dict:
    """Summarise independent household support after merging clones.

    Inputs align to household rows, with benefit-unit amounts already summed
    within each household. Support includes households in the cell even when
    their amount is zero. The contribution CV includes those zeros. A zero
    weighted total with nonzero amounts is unsupported, not a structural zero.
    """
    from .weight_quality import aggregate_source_weights

    values = np.asarray(weights, dtype=np.float64)
    amounts = np.asarray(annualised_amounts, dtype=np.float64)
    identities = np.asarray(source_household_ids, dtype=object)
    aggregate_source_weights(values, identities)
    if amounts.shape != values.shape or not np.isfinite(amounts).all():
        raise ValueError(
            "Cell amounts must be finite and align with household weights."
        )
    if bool((amounts < 0).any()):
        raise ValueError("Cell entitlement amounts must be nonnegative.")
    selected = (
        np.ones(values.shape, dtype=bool)
        if support is None
        else np.asarray(support, dtype=bool)
    )
    if selected.shape != values.shape:
        raise ValueError("Cell support must align with household weights.")
    values, amounts, identities = (
        vector[selected] for vector in (values, amounts, identities)
    )
    if not len(values):
        return {
            "source_household_ess": float(0),
            "positive_contributors": 0,
            "contribution_cv": float(0),
            "maximum_single_donor_dollar_share": float(0),
            "dollar_contribution_ess": float(0),
            "structural_zero": True,
        }
    # Scaling before multiplication avoids overflow and cancels out in every
    # diagnostic. Clone dollar contributions, rather than clone amounts, add.
    weight_max = float(np.max(values, initial=0))
    amount_max = float(np.max(amounts, initial=0))
    scaled_weights = values / weight_max if weight_max else values
    scaled_amounts = amounts / amount_max if amount_max else amounts
    household_weights = aggregate_source_weights(scaled_weights, identities)
    contributions = aggregate_source_weights(
        scaled_weights * scaled_amounts, identities
    )
    total = float(np.sum(contributions))
    structural_zero = amount_max == 0
    if total:
        mean = total / len(contributions)
        cv = float(np.std(contributions) / mean)
        maximum_share = float(np.max(contributions) / total)
    else:
        cv = None if not structural_zero else float(0)
        maximum_share = None if not structural_zero else float(0)
    return {
        "source_household_ess": _kish(household_weights),
        "positive_contributors": int(np.count_nonzero(contributions > 0)),
        "contribution_cv": cv,
        "maximum_single_donor_dollar_share": maximum_share,
        "dollar_contribution_ess": _kish(contributions),
        "structural_zero": structural_zero,
    }


def _support_thresholds(value: Mapping) -> dict:
    if set(value) != _SUPPORT_FIELDS:
        raise ValueError(
            "Cell support must declare exactly its screen and target thresholds."
        )
    return {name: _number(value[name], name, positive=True) for name in _SUPPORT_FIELDS}


def apply_cell_support(row: Mapping, diagnostics: Mapping, thresholds: Mapping) -> dict:
    """Remove every reportable numeric value from an unsupported cell."""
    thresholds = _support_thresholds(thresholds)
    reasons = []
    if diagnostics["structural_zero"]:
        reasons.append("structural_zero")
    else:
        for field, threshold, lower in (
            ("source_household_ess", "minimum_cell_ess", True),
            ("contribution_cv", "maximum_contribution_cv", False),
            (
                "maximum_single_donor_dollar_share",
                "maximum_single_donor_dollar_share",
                False,
            ),
        ):
            value = diagnostics[field]
            if value is None or (
                value < thresholds[threshold]
                if lower
                else value > thresholds[threshold]
            ):
                reasons.append(field)
    result = {**row, "suppressed": bool(reasons), "suppression_reasons": reasons}
    if reasons:
        result.update({metric: None for metric in _METRICS})
    return result


def _diagnostic_cell(row: Mapping, diagnostics: Mapping, thresholds: Mapping) -> dict:
    screened = apply_cell_support(row, diagnostics, thresholds)
    share = diagnostics["maximum_single_donor_dollar_share"]
    return {
        "name": row["name"],
        "period": row["period"],
        **diagnostics,
        "suppressed": screened["suppressed"],
        "meets_tracked_target": bool(
            not diagnostics["structural_zero"]
            and diagnostics["source_household_ess"]
            >= thresholds["target_minimum_cell_ess"]
            and share is not None
            and share <= thresholds["target_maximum_single_donor_dollar_share"]
        ),
    }


def decode_cell_diagnostics(payload: bytes) -> dict:
    """Validate scenario-complete source-household diagnostics bytes."""
    document = _document(
        payload,
        _CELL_DIAGNOSTICS_KIND,
        {"weight_entity", "weight_kind", "source_household_column", "scenarios"},
    )
    for name in ("weight_entity", "weight_kind", "source_household_column"):
        _text(document[name], f"cell diagnostics.{name}")
    scenarios = document["scenarios"]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Cell diagnostics need a non-empty list of scenarios.")
    labels = []
    expected_cells = None
    for scenario in scenarios:
        if not isinstance(scenario, dict) or set(scenario) != {"scenario", "cells"}:
            raise ValueError(
                "Cell diagnostics scenario rows name exactly scenario and cells."
            )
        labels.append(_text(scenario["scenario"], "cell diagnostics scenario"))
        rows = scenario["cells"]
        if not isinstance(rows, list) or not rows:
            raise ValueError("Every diagnostics scenario needs its declared cells.")
        cells = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "name",
                "period",
                *_CELL_FIELDS,
                "suppressed",
                "meets_tracked_target",
            }:
                raise ValueError("Unsupported cell diagnostics row shape.")
            cell = _cell(row)
            if cell in cells:
                raise ValueError("Cell diagnostics repeat a cell.")
            cells.add(cell)
            for flag in ("structural_zero", "suppressed", "meets_tracked_target"):
                if not isinstance(row[flag], bool):
                    raise ValueError(f"Cell diagnostics {flag} must be boolean.")
            count = row["positive_contributors"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError(
                    "Cell diagnostics contributors must be a nonnegative count."
                )
            for field in _CELL_FIELDS - {"structural_zero", "positive_contributors"}:
                if row[field] is None and field in {
                    "contribution_cv",
                    "maximum_single_donor_dollar_share",
                }:
                    continue
                if _number(row[field], field) < 0:
                    raise ValueError("Cell diagnostics values must be nonnegative.")
        if expected_cells is not None and cells != expected_cells:
            raise ValueError(
                "Cell diagnostics scenarios must cover identical table cells."
            )
        expected_cells = cells
    if labels != sorted(set(labels)):
        raise ValueError("Cell diagnostics scenarios must be distinct and ordered.")
    return document


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
        optional = {"suppressed", "suppression_reasons", "reporting_kind"}
        if not isinstance(row, dict) or set(row) - optional != _ROW_KEYS:
            raise ValueError("Unsupported gap row shape.")
        cell = _cell(row)
        if cell in seen:
            raise ValueError("A gap artifact repeats a table cell.")
        seen.add(cell)
        entitlement = row.get("reporting_kind") == "entitlement"
        if row.get("reporting_kind") not in (None, "entitlement"):
            raise ValueError("Unsupported gap reporting kind.")
        for name in ("entity", "measure"):
            _text(row[name], f"gap row.{name}")
        if not entitlement and row["reference"] != row["name"]:
            raise ValueError("A gap row must trace to its named reference.")
        if entitlement and (
            row["reference"] is not None or row["ledger_fact_key"] is not None
        ):
            raise ValueError(
                "Entitlement breakdowns cannot claim a subgroup comparator."
            )
        if row["filter"] is not None:
            _text(row["filter"], "gap row.filter")
        if not isinstance(row["metadata"], dict):
            raise ValueError("gap row.metadata must be an object.")
        _number(row["annualisation_factor"], "annualisation_factor", positive=True)
        if "suppressed" in row:
            reasons = row.get("suppression_reasons")
            if not isinstance(row["suppressed"], bool) or not isinstance(reasons, list):
                raise ValueError("Gap suppression needs a boolean and a reason list.")
            if bool(reasons) != row["suppressed"]:
                raise ValueError("Gap suppression reasons disagree with its flag.")
            if row["suppressed"]:
                if any(row[metric] is not None for metric in _METRICS):
                    raise ValueError(
                        "A suppressed cell cannot expose reportable values."
                    )
                continue
        estimate = _number(row["E"], "E")
        if entitlement:
            if row["A"] is not None or row["gap"] is not None:
                raise ValueError(
                    "Entitlement breakdowns have no subgroup actual or gap."
                )
            continue
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
    if not isinstance(rows, list):
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


def reporting_cell_declarations(document: Mapping) -> tuple[dict, ...]:
    """Expand a spec-declared table grid without deriving subgroup comparators."""
    if set(document) != {
        "entity",
        "measure",
        "period",
        "annualisation_factor",
        "tables",
    }:
        raise ValueError(
            "Reporting cells must declare their measure, period, factor and tables."
        )
    entity = _text(document["entity"], "reporting entity")
    measure = _text(document["measure"], "reporting measure")
    factor = _number(
        document["annualisation_factor"], "reporting factor", positive=True
    )
    tables = document["tables"]
    if not isinstance(tables, list) or not tables:
        raise ValueError("Reporting cells need a non-empty declared table list.")
    rows = []
    for table in tables:
        if not isinstance(table, dict) or set(table) != {"name", "axes"}:
            raise ValueError("Reporting tables declare exactly name and axes.")
        table_name = _text(table["name"], "reporting table name")
        axes = table["axes"]
        if not isinstance(axes, list) or not axes:
            raise ValueError("Reporting tables need non-empty axes.")
        columns, domains = [], []
        for axis in axes:
            if not isinstance(axis, dict) or set(axis) != {"column", "values"}:
                raise ValueError("Reporting axes declare exactly column and values.")
            columns.append(_text(axis["column"], "reporting axis column"))
            values = axis["values"]
            if not isinstance(values, list) or not values:
                raise ValueError("Reporting axes need non-empty domains.")
            if any(isinstance(value, list | dict) or value is None for value in values):
                raise ValueError("Reporting axis values must be non-null scalars.")
            if len({canonical_json(value) for value in values}) != len(values):
                raise ValueError("Reporting axes cannot repeat values.")
            domains.append(values)
        if len(set(columns)) != len(columns):
            raise ValueError("Reporting axes cannot repeat columns.")
        for values in product(*domains):
            selectors = dict(zip(columns, values, strict=True))
            rows.append(
                {
                    "name": table_name + "/" + "/".join(str(value) for value in values),
                    "period": document["period"],
                    "entity": entity,
                    "measure": measure,
                    "filter": None,
                    "metadata": {"table": table_name, "selectors": selectors},
                    "annualisation_factor": factor,
                    "reference": None,
                    "ledger_fact_key": None,
                    "reporting_kind": "entitlement",
                }
            )
    if len({_cell(row) for row in rows}) != len(rows):
        raise ValueError("Reporting grids cannot repeat a table cell.")
    return tuple(rows)


def _collapse_cell(
    frame, entity: str, values: np.ndarray, weight_entity: str
) -> np.ndarray:
    if entity == weight_entity:
        return values
    positions = target_kernels.matrix_module._nested_positions(
        entity, frame, weight_entity
    )
    collapsed = np.zeros(frame.n(weight_entity), dtype=np.float64)
    np.add.at(collapsed, positions, values)
    return collapsed


def _cell_support_from_reference(frame, target, weight_entity: str) -> np.ndarray:
    mask = target._filter_mask(frame, frame.n(target.entity))
    return _collapse_cell(frame, target.entity, mask, weight_entity) > 0


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
    _optional = frozenset(
        {
            "facts_sha256",
            "chronicle_feed_pin",
            "reference_resolution",
            "reference_selection",
            "annualisation_sha256",
            "source_household_column",
            "cell_support",
            "reporting_cells",
            "expected_weight_kind",
        }
    )

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            target_kernels,
            weight_quality,
            *target_kernels._COMPILE_MODULES,
            *target_kernels._MATRIX_MODULES,
            source_cell_diagnostics,
            apply_cell_support,
            reporting_cell_declarations,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context, self.ref, required=self._required, optional=self._optional
        )
        diagnostics_enabled = "cell_support" in context.params
        declared_outputs = {
            item.name: item.type for item in context.node.artifact_outputs
        }
        expected_outputs = {"gap": GAP_TYPE}
        if diagnostics_enabled:
            expected_outputs["cell_diagnostics"] = CELL_DIAGNOSTICS_TYPE
        if declared_outputs != expected_outputs:
            raise ValueError(
                f"{self.ref} must declare exactly its typed artifact outputs."
            )
        if bool("source_household_column" in context.params) != diagnostics_enabled:
            raise ValueError(
                "Cell support needs its declared source-household lineage column."
            )
        if "reporting_cells" in context.params and not diagnostics_enabled:
            raise ValueError("Reported entitlement cells require support diagnostics.")
        country = string_param(context, self.ref, "country")
        scenario = string_param(context, self.ref, "scenario")
        document = canonical_document_param(context, self.ref, "references")
        if "annualisation_sha256" in context.params:
            sha256_param(context, self.ref, "annualisation_sha256")
        references = parse_reference_document(document, country=country)
        factors = canonical_document_param(context, self.ref, "annualisation_factors")
        reference_selection = (
            canonical_document_param(context, self.ref, "reference_selection")
            if "reference_selection" in context.params
            else None
        )
        if reference_selection is not None:
            target_kernels.validate_exemptions(
                document,
                reference_selection,
                selected={name: (context.node.id,) for name in factors},
            )
            exempt_names = {
                row["name"]
                for row in target_kernels.exemption_rows(reference_selection)
            }
            references = tuple(
                ref for ref in references if ref.name not in exempt_names
            )
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
            feed_pin=(
                canonical_document_param(context, self.ref, "chronicle_feed_pin")
                if "chronicle_feed_pin" in context.params
                else None
            ),
            reference_resolution=(
                canonical_document_param(context, self.ref, "reference_resolution")
                if "reference_resolution" in context.params
                else None
            ),
            reference_selection=reference_selection,
        )
        surface = decode_target_surface(surface_payload)
        problem = target_kernels._compiled_problem(
            frame, registry, weight_entity, self.ref
        )
        weights = frame.weights_for(weight_entity)
        if diagnostics_enabled:
            lineage_column = string_param(context, self.ref, "source_household_column")
            if lineage_column not in frame.table(weight_entity):
                raise ValueError(
                    f"Missing source-household lineage column {lineage_column!r}."
                )
            expected_kind = string_param(context, self.ref, "expected_weight_kind")
            if weights.kind.value != expected_kind:
                raise ValueError(
                    f"Cell diagnostics require {expected_kind!r} household weights."
                )
            identities = frame.table(weight_entity)[lineage_column].to_numpy()
            thresholds = _support_thresholds(
                canonical_document_param(context, self.ref, "cell_support")
            )
        diagnostics_rows = []
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
            row = {
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
            if diagnostics_enabled:
                diagnostics = source_cell_diagnostics(
                    weights.values,
                    annualised,
                    identities,
                    support=_cell_support_from_reference(
                        frame, problem.targets[index], weight_entity
                    ),
                )
                diagnostics_rows.append(_diagnostic_cell(row, diagnostics, thresholds))
                row = apply_cell_support(row, diagnostics, thresholds)
            rows.append(row)
        if "reporting_cells" in context.params:
            reporting = canonical_document_param(context, self.ref, "reporting_cells")
            declarations = reporting_cell_declarations(reporting)
            table = frame.table(reporting["entity"])
            for table_definition in reporting["tables"]:
                for axis in table_definition["axes"]:
                    column = axis["column"]
                    if column not in table:
                        raise ValueError(
                            f"Reported cell columns are missing: {[column]}."
                        )
                    if not table[column].isin(axis["values"]).all():
                        raise ValueError(
                            f"Reported cell axis {column!r} has missing or "
                            "undeclared values."
                        )
            for declaration in declarations:
                table = frame.table(declaration["entity"])
                selectors = declaration["metadata"]["selectors"]
                required_columns = {*selectors, declaration["measure"]}
                missing = sorted(required_columns - set(table.columns))
                if missing:
                    raise ValueError(f"Reported cell columns are missing: {missing}.")
                selected = np.ones(len(table), dtype=bool)
                for column, value in selectors.items():
                    selected &= table[column].to_numpy() == value
                values = table[declaration["measure"]].to_numpy(dtype=np.float64)
                annualised = _collapse_cell(
                    frame,
                    declaration["entity"],
                    values * selected * declaration["annualisation_factor"],
                    weight_entity,
                )
                support = (
                    _collapse_cell(
                        frame,
                        declaration["entity"],
                        selected.astype(np.float64),
                        weight_entity,
                    )
                    > 0
                )
                row = {
                    **declaration,
                    "E": float(np.sum(weights.values * annualised)),
                    "A": None,
                    "gap": None,
                }
                diagnostics = source_cell_diagnostics(
                    weights.values, annualised, identities, support=support
                )
                diagnostics_rows.append(_diagnostic_cell(row, diagnostics, thresholds))
                rows.append(apply_cell_support(row, diagnostics, thresholds))
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
        artifacts = {"gap": payload}
        if diagnostics_enabled:
            diagnostics_payload = canonical_json(
                {
                    "schema_version": 1,
                    "kind": _CELL_DIAGNOSTICS_KIND,
                    "weight_entity": weight_entity,
                    "weight_kind": weights.kind.value,
                    "source_household_column": lineage_column,
                    "scenarios": [{"scenario": scenario, "cells": diagnostics_rows}],
                }
            )
            decode_cell_diagnostics(diagnostics_payload)
            artifacts["cell_diagnostics"] = diagnostics_payload
        return KernelResult(
            artifacts=artifacts,
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
            if any(
                cells[label][cell].get("suppressed", False) for label in ordered_labels
            ):
                continue
            metrics = (
                ("E",)
                if central_row.get("reporting_kind") == "entitlement"
                else _METRICS
            )
            for metric in metrics:
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


class TakeupCellDiagnosticsKernel(KernelBase):
    """Collect all declared scenarios without losing an absent scenario."""

    ref = "takeup.cell_diagnostics@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
    )

    def implementation_hash(self) -> str:
        return source_hash(type(self), decode_cell_diagnostics, graph_inputs)

    def run(self, context: KernelContext) -> KernelResult:
        require_params(context, self.ref, required=frozenset({"artifact_scenarios"}))
        _require_output(context, self.ref, "cell_diagnostics", CELL_DIAGNOSTICS_TYPE)
        aliases = canonical_document_param(context, self.ref, "artifact_scenarios")
        labels = [_text(label, "diagnostics scenario") for label in aliases.values()]
        if not labels or len(set(labels)) != len(labels):
            raise ValueError("Cell diagnostics need distinct declared scenarios.")
        if set(aliases) != set(context.artifacts):
            raise ValueError(
                "Cell diagnostics are missing a declared scenario artifact."
            )
        documents = {}
        for alias, label in aliases.items():
            value = context.artifacts[alias]
            if value.type != CELL_DIAGNOSTICS_TYPE:
                raise ValueError("Every scenario needs typed cell diagnostics.")
            document = decode_cell_diagnostics(value.payload)
            if (
                len(document["scenarios"]) != 1
                or document["scenarios"][0]["scenario"] != label
            ):
                raise ValueError(
                    "Cell diagnostics disagree with their declared scenario."
                )
            documents[label] = document
        first = documents[sorted(documents)[0]]
        identity = ("weight_entity", "weight_kind", "source_household_column")
        if any(
            any(document[name] != first[name] for name in identity)
            for document in documents.values()
        ):
            raise ValueError(
                "Scenario cell diagnostics disagree on household weight lineage."
            )
        payload = canonical_json(
            {
                "schema_version": 1,
                "kind": _CELL_DIAGNOSTICS_KIND,
                **{name: first[name] for name in identity},
                "scenarios": [
                    documents[label]["scenarios"][0] for label in sorted(documents)
                ],
            }
        )
        decode_cell_diagnostics(payload)
        return KernelResult(
            artifacts={"cell_diagnostics": payload},
            receipt={
                "n_scenarios": len(documents),
                "cell_diagnostics_sha256": sha256_text(payload),
            },
        )


TAKEUP_GAP = TakeupGapKernel()
TAKEUP_BANDS = TakeupBandsKernel()
TAKEUP_CELL_DIAGNOSTICS = TakeupCellDiagnosticsKernel()


def register_takeup_kernels(registry: KernelRegistry) -> None:
    """Register the reporting kernels without modifying a global registry."""

    for kernel in (TAKEUP_GAP, TAKEUP_BANDS, TAKEUP_CELL_DIAGNOSTICS):
        registry.register(kernel)
