"""Explicit S0101-only activation for the selected ACS/ASEC development path.

A reviewed declaration pins two source responses. The caller's review and
guarded acquisition supply trust; this decoder is not independent attestation.
There is deliberately no default genuine declaration or mixed-inventory reader.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path

from microcosm.calibrate.registry import TargetRegistry, TargetSpec
from microcosm.graph.canonical import canonical_json

from . import demographic_calibration_graph as demographic
from . import national_age_activation as age
from .survey_age_sources import _digest, read_survey_age_response, survey_age_requests

PROFILE = "selected_survey_s0101_development_v1"
_CONVENTION = (
    "ACS 2024 observed interview ages and ASEC 2025 interview ages with income "
    "year 2024; completed years used verbatim without temporal aging; calibration "
    "to the published 2024 all-resident distribution does not establish that "
    "the selected survey population covers that complete universe"
)


def _require(condition: object, reason: str) -> None:
    if not condition:
        raise ValueError("SURVEY_AGE_ACTIVATION_" + reason)


@dataclass(frozen=True, slots=True)
class SurveyAgeActivation:
    """Exact source pins whose approval is a separate caller responsibility."""

    metadata_sha256: str
    data_sha256: str

    def __post_init__(self) -> None:
        _digest(self.metadata_sha256)
        _digest(self.data_sha256)


def activation_binding(declaration: SurveyAgeActivation) -> dict:
    """Return the complete, detached semantic declaration, without source I/O."""
    _require(type(declaration) is SurveyAgeActivation, "DECLARATION")
    declaration.__post_init__()
    return {
        "profile": PROFILE,
        "requests": list(survey_age_requests()),
        "metadata_sha256": declaration.metadata_sha256,
        "data_sha256": declaration.data_sha256,
        "period": "2024",
        "entity": "household",
        "universe": age.NATIONAL_AGE_ACTIVATION.universe,
        "age_convention_id": "observed_interview_age_completed_years",
        "age_convention": _CONVENTION,
        "bands": [
            {
                "variable": b.variable,
                "label": b.label,
                "low": b.low,
                "high": b.high,
                "column": b.column,
            }
            for b in age.NATIONAL_AGE_ACTIVATION.bands
        ],
        "published_total": "S0101_C01_001_consistency_only",
        "consumes_se": False,
        "release_eligible": False,
    }


def declaration_from_binding(binding: object) -> SurveyAgeActivation:
    """Validate value-level profile semantics; grant no source authorization."""
    _require(type(binding) is dict, "DECLARATION")
    _require("metadata_sha256" in binding and "data_sha256" in binding, "DECLARATION")
    declaration = SurveyAgeActivation(
        binding["metadata_sha256"], binding["data_sha256"]
    )
    _require(
        canonical_json(binding) == canonical_json(activation_binding(declaration)),
        "DECLARATION",
    )
    return declaration


def activation_digest(declaration: SurveyAgeActivation) -> str:
    """Bind source pins and all selected-survey conventions in one identifier."""
    return hashlib.sha256(canonical_json(activation_binding(declaration))).hexdigest()


def validate_survey_age_registry(
    registry: TargetRegistry, binding: dict
) -> TargetRegistry:
    """Check numerical profile metadata, not the truth of caller-supplied values."""
    declaration = declaration_from_binding(binding)
    frozen = demographic._registry_from_json(demographic._registry_json(registry))
    bands = age.NATIONAL_AGE_ACTIVATION.bands
    _require(len(frozen) == len(bands), "REGISTRY")
    digest = activation_digest(declaration)
    for spec, band in zip(frozen, bands, strict=True):
        _require(
            spec.name == band.variable
            and spec.measure == band.column
            and spec.metadata["table"] == "S0101"
            and spec.metadata["reference_sha256"] == digest
            and spec.metadata["evidence_scope"] == "source_documented",
            "REGISTRY",
        )
    return frozen


def activate_survey_age_targets(
    source_dir: str | Path, *, declaration: SurveyAgeActivation
) -> TargetRegistry:
    """Derive exactly 18 targets from the two explicitly pinned local responses."""
    binding = activation_binding(declaration)
    _, metadata = read_survey_age_response(
        source_dir, "metadata", declaration.metadata_sha256
    )
    _, data = read_survey_age_response(source_dir, "data", declaration.data_sha256)
    # Reuse the established publisher/count/MOE interpretation without invoking
    # its old inventory loader. This object supplies only pure field semantics.
    semantics = replace(age.NATIONAL_AGE_ACTIVATION, age_convention=_CONVENTION)
    row = age._national_row(data, semantics)
    _require(
        all(
            name in {"GEO_ID", "NAME", "us"} or name.startswith("S0101_")
            for name in row
        ),
        "RESPONSE_SCOPE",
    )
    _require(
        type(metadata) is dict and type(metadata.get("variables")) is dict, "METADATA"
    )
    _require(
        all(
            type(name) is str
            and (name in {"GEO_ID", "NAME", "us"} or name.startswith("S0101_"))
            for name in metadata["variables"]
        ),
        "METADATA_SCOPE",
    )

    def cell(variable):
        _require(
            all(variable + suffix in row for suffix in ("E", "EA", "M", "MA")), "CELL"
        )
        return {
            "estimate": age.classify_acs_value(
                row[variable + "E"], row[variable + "EA"]
            ),
            "moe": age.classify_acs_value(row[variable + "M"], row[variable + "MA"]),
        }

    digest = activation_digest(declaration)
    specs = []
    data_url = survey_age_requests()[1]["url"]
    for band in semantics.bands:
        published = age._published_variable(metadata, band.variable + "E", semantics)
        _require(
            published.get("label") == band.label
            and published.get("predicateType") == "int",
            "PUBLISHED_DEFINITION",
        )
        derived = cell(band.variable)
        specs.append(
            TargetSpec(
                name=band.variable,
                entity="household",
                measure=band.column,
                value=float(age._count_estimate(derived, band.variable)),
                period="2024",
                se=age._standard_error(derived, band.variable, confidence_level=0.9),
                source=f"{data_url} (S0101 {band.variable}E)",
                family="acs.S0101",
                hierarchy=age.demographic_target_hierarchy(
                    "S0101", band, geography="0100000US"
                ),
                notes=f"activation={digest}; label={band.label}; age_convention={_CONVENTION}; published 90% MOE divided by {age._MOE_90_TO_SE}, not consumed by current loss",
                metadata={
                    "table": "S0101",
                    "reference_sha256": digest,
                    "geography": "0100000US",
                    "universe": "population",
                    "role": "calibration",
                    "evidence_scope": "source_documented",
                },
            )
        )
    total = age._published_variable(metadata, "S0101_C01_001E", semantics)
    _require(
        total.get("label") == "Estimate!!Total!!Total population"
        and total.get("predicateType") == "int",
        "PUBLISHED_TOTAL_DEFINITION",
    )
    _require(
        sum(int(s.value) for s in specs)
        == age._count_estimate(cell("S0101_C01_001"), "S0101_C01_001"),
        "PARTITION",
    )
    return validate_survey_age_registry(TargetRegistry(specs, country="us"), binding)


def verify_survey_age_targets(
    source_dir: str | Path,
    *,
    declaration: SurveyAgeActivation,
    registry: TargetRegistry,
) -> None:
    """Rederive the pinned target values; no stored registry substitutes for this."""
    actual = activate_survey_age_targets(source_dir, declaration=declaration)
    _require(
        demographic._registry_json(actual) == demographic._registry_json(registry),
        "REGISTRY",
    )
