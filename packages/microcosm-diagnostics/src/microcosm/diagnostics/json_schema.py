"""A Draft 2020-12 JSON Schema for calibration diagnostics schema 8.

The schema is generated from :class:`CalibrationDiagnosticsV8`, so it cannot
drift from the models in field names, types, bounds, enumerations, required
fields or undeclared-field refusal. The models remain the authority: some of
their rules cannot be written in a schema generated this way, and the schema's
description lists each one (:data:`CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_GAPS`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from microcosm.diagnostics.schema import CalibrationDiagnosticsV8

JSON_SCHEMA_DRAFT_2020_12: Final = "https://json-schema.org/draft/2020-12/schema"
CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_ID: Final = (
    "calibration_diagnostics.v8.schema.json"
)
#: The committed schema, relative to the ``microcosm.diagnostics`` package.
CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_RESOURCE: Final = (
    f"schemas/{CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_ID}"
)
_REGENERATE_COMMAND: Final = (
    "uv run --no-sync python tools/generate_calibration_diagnostics_json_schema.py"
)


@dataclass(frozen=True)
class JsonSchemaGap:
    """A rule the schema-8 models enforce that the generated schema leaves out.

    ``validators`` names the model validator functions that enforce the rule,
    by qualified name; it is empty when the rule comes from model
    configuration or from how validators evaluate patterns.
    """

    key: str
    rule: str
    validators: tuple[str, ...] = ()


CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_GAPS: Final[tuple[JsonSchemaGap, ...]] = (
    JsonSchemaGap(
        key="finite-numbers",
        rule=(
            "Every number is finite, at any depth, including inside options, "
            "targets[].metadata and build. JSON has no NaN or Infinity, but "
            "lenient parsers such as Python's json module accept those tokens, "
            "and a JSON Schema number cannot exclude the values they produce, "
            "so parse submissions with a parser that refuses them."
        ),
        validators=(
            "DiagnosticsModel.reject_non_finite_numbers",
            "CalibrationDiagnosticsV8.finite_optional_scalar",
        ),
    ),
    JsonSchemaGap(
        key="strict-integers",
        rule=(
            "Integer fields hold integers written without a fraction or "
            "exponent. JSON Schema counts 2.0 and 1e3 as integers; the strict "
            "models refuse them, because Python's json module decodes them as "
            "floats."
        ),
    ),
    JsonSchemaGap(
        key="non-blank-text",
        rule=(
            "Strings the models type as non-blank contain at least one "
            "character that is not whitespace. These are most names, "
            "identifiers, labels, sources, entities, reasons, codes and "
            "messages, and the keys of uk_diagnostics.target_observation_basis, "
            "the zero-weight strata and the consumption-drift maps."
        ),
        validators=("_require_non_blank",),
    ),
    JsonSchemaGap(
        key="digest-pattern-anchoring",
        rule=(
            "Digests match ^[0-9a-f]{64}$ with nothing after them. In the "
            "ECMA-262 regular expressions JSON Schema specifies, $ matches "
            "only at the end of the string, but validators that evaluate "
            "pattern with Python's re module also let $ match before a final "
            "newline, so they accept a digest followed by a newline."
        ),
    ),
    JsonSchemaGap(
        key="hierarchy-identity",
        rule=(
            "In each targets[].hierarchy, category.provider_id equals "
            "provider.id and the dimensions have unique ids."
        ),
        validators=("CalibrationHierarchy.category_matches_provider",),
    ),
    JsonSchemaGap(
        key="target-hierarchy",
        rule="Each targets[].hierarchy.target.id equals that row's target_name.",
        validators=("TargetDiagnosticV8.hierarchy_target_matches_row",),
    ),
    JsonSchemaGap(
        key="target-surface-shape",
        rule=(
            "target_surface.constraint_matrix has n_targets rows and "
            "n_records columns, and its nnz is at most rows times columns."
        ),
        validators=("TargetSurface.reconcile_matrix_shape",),
    ),
    JsonSchemaGap(
        key="diagnostics-counts",
        rule=(
            "target_surface.n_targets equals the number of targets, "
            "target_registry.n_specs is at least the number of targets, "
            "target_surface.n_records equals n_records, and n_nonzero is at "
            "most n_records."
        ),
        validators=("CalibrationDiagnosticsV8.reconcile_counts",),
    ),
    JsonSchemaGap(
        key="uk-extension-presence",
        rule=(
            'A UK registry (target_registry.country "uk") requires '
            "uk_diagnostics; any other registry requires it to be null or "
            "absent. In a UK document, uk_diagnostics.weights.n_records, "
            "effective_sample_size and top_1pct_weight_share equal the "
            "top-level n_records, effective_sample_size and "
            "top_1pct_weight_share. The n_targets, n_scored and n_skipped "
            "values in uk_diagnostics.target_pass_rates_by_geography_level sum "
            "to target_registry.n_specs, the number of targets and the number "
            "of skipped targets."
        ),
        validators=("CalibrationDiagnosticsV8.reconcile_counts",),
    ),
    JsonSchemaGap(
        key="uk-weight-summary",
        rule=(
            "In uk_diagnostics.weights, positive_weight_records plus "
            "zero_weight_records equals n_records, effective_sample_size is at "
            "most n_records, and ess_fraction equals effective_sample_size "
            "divided by n_records. median_positive_weight and "
            "max_to_median_positive_weight are null exactly when "
            "positive_weight_records is 0; otherwise "
            "max_to_median_positive_weight equals max_weight divided by "
            "median_positive_weight."
        ),
        validators=("UKWeightSummary.reconcile_weight_summary",),
    ),
    JsonSchemaGap(
        key="uk-zero-weight-strata",
        rule=(
            "In each uk_diagnostics.zero_weight_rows_by_stratum row, "
            "positive_weight_rows plus zero_weight_rows equals rows. Across the "
            "strata, rows, positive_weight_rows and zero_weight_rows sum to "
            "weights.n_records, weights.positive_weight_records and "
            "weights.zero_weight_records."
        ),
        validators=(
            "UKZeroWeightStratum.reconcile_row_counts",
            "UKDiagnosticsV1.reconcile_uk_diagnostics",
        ),
    ),
    JsonSchemaGap(
        key="uk-geography-pass-rates",
        rule=(
            "uk_diagnostics.target_pass_rates_by_geography_level has exactly "
            "one row for each of national, region, country, local_authority "
            "and constituency. In each row, n_scored plus n_skipped equals "
            "n_targets, n_within_10pct is at most n_scored, and pass_rate "
            "equals n_within_10pct divided by n_targets, or is null when "
            "n_targets is 0."
        ),
        validators=(
            "UKGeographyPassRate.reconcile_pass_rate",
            "UKDiagnosticsV1.reconcile_uk_diagnostics",
        ),
    ),
    JsonSchemaGap(
        key="uk-local-fit",
        rule=(
            "uk_diagnostics.weakest_families and weakest_areas_by_fit are both "
            "present or both absent. In weakest_families, "
            "weakest_areas_by_fit.bottom_by_fit and "
            "weakest_areas_by_fit.countries, n_within_10pct is at most "
            "n_targets and pass_rate equals n_within_10pct divided by "
            "n_targets. bottom_by_fit has at most limit rows and at most "
            "n_areas_scored rows."
        ),
        validators=(
            "UKWeakestFamily.reconcile_pass_rate",
            "UKAreaFit.reconcile_pass_rate",
            "UKCountryFit.reconcile_pass_rate",
            "UKWeakestAreasByFit.reconcile_area_counts",
            "UKDiagnosticsV1.reconcile_uk_diagnostics",
        ),
    ),
    JsonSchemaGap(
        key="uk-rotated-holdout",
        rule=(
            "A measured uk_diagnostics.rotated_holdout has unique grains and "
            "n_folds entries in both fold_losses and folds. Its fold ids cover "
            "0 to n_folds - 1, each fold's holdout_loss equals its fold_losses "
            "entry, mean_holdout_loss is the mean of fold_losses and "
            "worst_holdout_loss is their maximum. Each fold has "
            "n_holdout_targets unique holdout_target_indices."
        ),
        validators=(
            "UKRotatedHoldoutFold.reconcile_holdout_indices",
            "UKMeasuredRotatedHoldout.reconcile_folds",
        ),
    ),
    JsonSchemaGap(
        key="uk-consumption-drift",
        rule=(
            "Each uk_diagnostics.consumption_drift value has change equal to "
            "final minus design, and its relative_change is null exactly when "
            "design is 0."
        ),
        validators=("UKDriftValue.reconcile_change",),
    ),
)

CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_LITERAL_NOTE: Final = (
    "Literal fields hold exactly their declared JSON value. The models compare "
    "literals with Python equality, so their parser also accepts 1 for true "
    "and true for 1, for example in rotated_holdout.skipped or "
    "uk_diagnostics.schema_version. The canonical writer always emits the "
    "declared value, which this schema requires."
)

_TITLE: Final = "Microcosm calibration diagnostics, schema 8"
_COMMENT: Final = (
    "Generated from microcosm.diagnostics.CalibrationDiagnosticsV8. Do not "
    f"edit it; regenerate it with `{_REGENERATE_COMMAND}`."
)


def _description() -> str:
    gap_lines = "\n".join(
        f"- {gap.rule}" for gap in CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_GAPS
    )
    return (
        "The calibration_diagnostics.json document, schema 8, that Microcosm "
        "builds write beside a calibrated dataset. The pydantic models it is "
        "generated from remain the authority. A document that satisfies this "
        "schema can still break one of the rules below, which the models "
        "enforce and this schema does not express, so validate with "
        "microcosm.diagnostics.parse_calibration_diagnostics for the complete "
        "contract. Equalities between floating-point values hold within the "
        "models' tolerances.\n\n"
        "Rules the models enforce that this schema does not express:\n"
        f"{gap_lines}\n\n"
        "Where this schema is stricter than the models' parser:\n"
        f"- {CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_LITERAL_NOTE}"
    )


def calibration_diagnostics_v8_json_schema() -> dict[str, JsonValue]:
    """Return the schema-8 JSON Schema generated from the typed models."""

    generated = CalibrationDiagnosticsV8.model_json_schema(mode="validation")
    generated.pop("title", None)
    generated.pop("description", None)
    return {
        "$schema": JSON_SCHEMA_DRAFT_2020_12,
        "$id": CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_ID,
        "$comment": _COMMENT,
        "title": _TITLE,
        "description": _description(),
        **generated,
    }


def render_calibration_diagnostics_v8_json_schema() -> str:
    """Return the schema as the exact text committed to the repository."""

    return (
        json.dumps(
            calibration_diagnostics_v8_json_schema(),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def committed_calibration_diagnostics_v8_json_schema_path() -> Path:
    """Return the committed schema file, shipped inside the package."""

    return Path(
        str(
            files("microcosm.diagnostics").joinpath(
                CALIBRATION_DIAGNOSTICS_V8_JSON_SCHEMA_RESOURCE
            )
        )
    )


def load_calibration_diagnostics_v8_json_schema(
    path: str | Path | None = None,
) -> dict[str, JsonValue]:
    """Read the committed schema, or the schema file at ``path``."""

    source = (
        committed_calibration_diagnostics_v8_json_schema_path()
        if path is None
        else Path(path)
    )
    return json.loads(source.read_bytes())


def write_calibration_diagnostics_v8_json_schema(
    path: str | Path | None = None,
) -> Path:
    """Write freshly generated schema text to the committed file or ``path``."""

    output = (
        committed_calibration_diagnostics_v8_json_schema_path()
        if path is None
        else Path(path)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(render_calibration_diagnostics_v8_json_schema().encode("utf-8"))
    return output


def assert_calibration_diagnostics_v8_json_schema_current(
    path: str | Path | None = None,
) -> None:
    """Raise unless the file is byte-identical to freshly generated output."""

    source = (
        committed_calibration_diagnostics_v8_json_schema_path()
        if path is None
        else Path(path)
    )
    expected = render_calibration_diagnostics_v8_json_schema().encode("utf-8")
    if source.read_bytes() != expected:
        raise ValueError(
            f"{source} differs from the schema the models generate; "
            f"regenerate it with `{_REGENERATE_COMMAND}`."
        )
