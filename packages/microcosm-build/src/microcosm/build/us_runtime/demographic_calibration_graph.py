"""National demographic calibration with registry-bound graph diagnostics.

This development seam consumes explicitly prepared household count columns.
It does not acquire references, authorize their use, define their universe, or
turn a calibration into national/CD certification. In particular, a captured
reference bundle with targets_allowed=False is not a TargetRegistry.

The complete declared registry travels in the node key and diagnostics retain
its source, period and family. Standard errors are reported, not used by the
current Adam loss. Reserved B19001 and B25003 observations never enter this
seam. The old calibrate.adam@1 contract and its receipts are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict

import numpy as np
from pandas.api.types import is_complex_dtype, is_numeric_dtype

import microcosm.build.cd_benchmark.protocol as protocol_module
import microcosm.build.us_runtime.national_age_activation as age_module
import microcosm.calibrate._target_loss_attribution as attribution_module
import microcosm.calibrate.diagnostics as diagnostics_module
import microcosm.calibrate.kernels as kernels_module
import microcosm.calibrate.monetary_binding as monetary_binding_module
import microcosm.calibrate.registry as registry_module
import microcosm.calibrate.solve as solve_module
from microcosm.build.cd_benchmark.protocol import RESERVED_FAMILIES
from microcosm.calibrate import calibrate, diagnostics_payload
from microcosm.calibrate.hierarchy import CalibrationHierarchy
from microcosm.calibrate.registry import TargetRegistry, TargetSpec
from microcosm.frame import WeightKind
from microcosm.graph import (
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelResult,
    Node,
    Numeric,
    SeedSource,
    Slice,
    StructuralDelta,
    WeightTransition,
    source_hash,
)

DIAGNOSTICS_TYPE = ArtifactType(
    "microcosm.calibration_diagnostics",
    diagnostics_module.CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
)
_OUTPUTS = (ArtifactOutput("diagnostics", DIAGNOSTICS_TYPE),)
_PARAMS = frozenset(
    {
        "registry",
        "epochs",
        "learning_rate",
        "max_weight_ratio",
        "max_initial_weight_ratio",
        "mass",
        "weight_anchor",
    }
)
_METADATA = frozenset(
    {"table", "reference_sha256", "geography", "universe", "role", "evidence_scope"}
)


def _json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _registry_json(registry: TargetRegistry) -> str:
    return _json(
        {"country": registry.country, "specs": [asdict(s) for s in registry]}
    ).decode()


def _validate_registry(registry: TargetRegistry) -> None:
    if registry.country != "us" or not len(registry):
        raise ValueError("A nonempty US demographic registry is required.")
    scopes = set()
    for spec in registry:
        metadata = spec.metadata
        table = metadata.get("table")
        if table in RESERVED_FAMILIES or spec.family in RESERVED_FAMILIES:
            raise ValueError("Income and tenure tables are reserved holdouts.")
        if table not in {"S0101", "B01001"} or spec.family != f"acs.{table}":
            raise ValueError(
                "Only explicit ACS S0101/B01001 demographic families are supported."
            )
        cell = re.fullmatch(
            r"S0101_C01_(\d{3})" if table == "S0101" else r"B01001_(\d{3})", spec.name
        )
        if cell is None or not 1 <= int(cell[1]) <= (19 if table == "S0101" else 49):
            raise ValueError(
                "A supported count cell must identify each demographic target."
            )
        if (
            set(metadata) != _METADATA
            or metadata["role"] != "calibration"
            or metadata["geography"] != "0100000US"
            or metadata["universe"] != "population"
            or metadata["evidence_scope"] not in {"invented", "source_documented"}
            or re.fullmatch(r"[a-f0-9]{64}", metadata["reference_sha256"]) is None
            or spec.entity != "household"
            or spec.period != "2024"
            or spec.signed
            or spec.value < 0
            or spec.filter is not None
            or spec.tolerance is not None
            or (spec.se is not None and not np.isfinite(spec.se))
        ):
            raise ValueError(
                "Demographic target scope must be explicit national population counts for 2024."
            )
        if spec.hierarchy is None or spec.hierarchy != (
            age_module.expected_demographic_hierarchy(
                table,
                variable=spec.name,
                label=spec.hierarchy.target.label,
                geography=metadata["geography"],
            )
        ):
            raise ValueError(
                "Each demographic target must carry exactly its own national "
                "Census ACS calibration hierarchy."
            )
        scopes.add(metadata["evidence_scope"])
    if len(scopes) != 1:
        raise ValueError(
            "Invented and source-documented evidence scopes cannot be mixed."
        )


def _registry_from_json(text: str) -> TargetRegistry:
    if not isinstance(text, str) or len(text.encode()) > 1_048_576:
        raise ValueError("A bounded canonical registry declaration is required.")
    document = json.loads(text)
    if not isinstance(document, dict) or set(document) != {"country", "specs"}:
        raise ValueError("Unsupported registry declaration shape.")
    specs = []
    for raw in document["specs"]:
        if not isinstance(raw, dict):
            raise ValueError("Unsupported registry declaration shape.")
        hierarchy = raw.get("hierarchy")
        if hierarchy is not None:
            raw = {**raw, "hierarchy": CalibrationHierarchy.from_dict(hierarchy)}
        specs.append(TargetSpec(**raw))
    registry = TargetRegistry(specs, country=document["country"])
    if text != _registry_json(registry):
        raise ValueError(
            "Registry declaration must have its canonical typed representation."
        )
    _validate_registry(registry)
    return registry


def _solver_options(params) -> dict:
    if (
        set(params) != _PARAMS
        or params["mass"] != "free"
        or params["weight_anchor"] != "design"
    ):
        raise ValueError(
            "The development calibration requires free mass and an original design cap."
        )
    if type(params["epochs"]) is not int or params["epochs"] < 1:
        raise ValueError("epochs must be a positive integer.")
    for key, minimum in (
        ("learning_rate", 0),
        ("max_weight_ratio", 1),
        ("max_initial_weight_ratio", 1),
    ):
        value = params[key]
        if (
            type(value) not in {int, float}
            or not np.isfinite(value)
            or value < minimum
            or (key == "learning_rate" and value == 0)
        ):
            raise ValueError(f"Unsupported {key}.")
    return {
        **{key: params[key] for key in ("epochs", "learning_rate", "mass")},
        "max_weight_ratio": params["max_initial_weight_ratio"],
    }


def demographic_calibration_node(
    registry: TargetRegistry,
    *,
    base: str,
    node_id: str = "national.demographic_calibration",
    epochs: int,
    learning_rate: float,
    max_weight_ratio: float,
    max_initial_weight_ratio: float,
) -> Node:
    """Freeze a caller's explicit registry and declare its actual count inputs.

    Source citations/digests are declarations here, not independently verified
    acquisitions or permission to repurpose held-out evidence. Activating a
    genuine registry remains a separate, predeclared build decision.

    max_initial_weight_ratio bounds the solver relative to incoming importance
    weights. max_weight_ratio separately guards the result relative to original
    design weights in the executor. These anchors need not be equal. The kernel
    cannot inspect undeclared design weights; a violated design cap refuses the
    graph result rather than silently changing the solver's answer.
    """
    frozen = _registry_from_json(_registry_json(registry))
    params = dict(
        registry=_registry_json(frozen),
        epochs=epochs,
        learning_rate=learning_rate,
        max_weight_ratio=max_weight_ratio,
        max_initial_weight_ratio=max_initial_weight_ratio,
        mass="free",
        weight_anchor="design",
    )
    _solver_options(params)
    return Node(
        id=node_id,
        kernel=DemographicCalibrationKernel.ref,
        inputs=(Slice("household", tuple(dict.fromkeys(s.measure for s in frozen))),),
        params=params,
        base=base,
        structural=StructuralDelta.REWEIGHT,
        weights=WeightTransition("household", "calibrated", mass="free"),
        mass="free",
        artifact_outputs=_OUTPUTS,
    )


class DemographicCalibrationKernel(KernelBase):
    """Call the real solver and persist its complete v6 diagnostic artifact."""

    ref = "us.demographic_calibration@1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC,
        numeric=Numeric.PLATFORM_BITWISE,
        seed_source=SeedSource.NONE,
        structural=StructuralDelta.REWEIGHT,
        consumes_se=False,
        dependencies=("numpy", "pandas", "scipy", "torch"),
    )

    def implementation_hash(self) -> str:
        return hashlib.sha256(
            _json(
                {
                    "adapter": source_hash(
                        sys.modules[__name__],
                        age_module,
                        registry_module,
                        diagnostics_module,
                        attribution_module,
                        solve_module,
                        kernels_module,
                        monetary_binding_module,
                        protocol_module,
                        dependencies=self.capabilities.dependencies,
                    ),
                    "solver": kernels_module.CALIBRATE_ADAM.implementation_hash(),
                }
            )
        ).hexdigest()

    def run(self, context: KernelContext) -> KernelResult:
        registry = _registry_from_json(context.params["registry"])
        options = _solver_options(context.params)
        expected = demographic_calibration_node(
            registry,
            base=context.node.base,
            node_id=context.node.id,
            **{
                k: context.params[k]
                for k in (
                    "epochs",
                    "learning_rate",
                    "max_weight_ratio",
                    "max_initial_weight_ratio",
                )
            },
        )
        if context.node.normative() != expected.normative():
            raise ValueError("Calibration node differs from its complete declaration.")
        if context.weights["household"].kind is not WeightKind.IMPORTANCE:
            raise ValueError("Calibration requires the declared importance weights.")
        if not np.all(context.weights["household"].values > 0):
            raise ValueError("Calibration requires strictly positive incoming weights.")
        table = context.tables["household"]
        for spec in registry:
            dtype = table[spec.measure].dtype
            if not is_numeric_dtype(dtype) or is_complex_dtype(dtype):
                raise ValueError("Demographic measures require numeric count columns.")
            values = table[spec.measure].to_numpy(dtype=float, na_value=np.nan)
            if not np.all(
                np.isfinite(values) & (values >= 0) & (values == np.floor(values))
            ):
                raise ValueError(
                    "Demographic measures must be nonnegative finite integer counts."
                )
        frame = kernels_module._frame_from_context(context, "household")
        result = calibrate(
            frame,
            registry.to_target_set(),
            weight_entity="household",
            method="adam",
            seed=0,
            **options,
        )
        if result.skipped:
            raise ValueError("Demographic calibration cannot skip declared targets.")
        anchors = {
            "solver_weight_anchor": "incoming_importance",
            "solver_max_weight_ratio": context.params["max_initial_weight_ratio"],
            "executor_weight_anchor": "original_design",
            "executor_max_weight_ratio": context.params["max_weight_ratio"],
        }
        payload = _json(
            diagnostics_payload(result, target_registry=registry, build=anchors)
        )
        return KernelResult(
            weights=result.frame.weights_for("household"),
            artifacts={"diagnostics": payload},
            receipt={
                "scope": "national_demographic_calibration_development",
                "evidence_scope": registry.specs[0].metadata["evidence_scope"],
                "registry_sha256": hashlib.sha256(
                    context.params["registry"].encode()
                ).hexdigest(),
                "registry_version": registry.version,
                "diagnostics_sha256": hashlib.sha256(payload).hexdigest(),
                "reserved_families": sorted(RESERVED_FAMILIES),
                "consumes_standard_errors": False,
                **anchors,
                "release_eligible": False,
                "national_validation": "not_evaluated",
                "congressional_district_validation": "not_evaluated",
            },
        )
