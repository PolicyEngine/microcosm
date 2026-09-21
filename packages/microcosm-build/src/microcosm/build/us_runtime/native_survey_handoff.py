"""Development handoff from an issued native owner to maintained Frame consumers.

No assembly, cloning, fitting, default filling or calibration occurs here.
The explicit engine projection only selects cells and stays unqualified. A checkpoint is descriptive storage, never a replacement issuer.
Release admission still belongs to the maintained builder's scientific gates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.frame import Frame
from microcosm.frame.rules import ExportContract
from microcosm.graph.store import _decode_frame_metadata, _encode_frame_metadata

from . import graph_us_survey_enrichment as native
from .input_coverage_profile import (
    HISTORICAL_REQUIRED_INPUTS,
    MANIFEST_SHA256,
    USInputProfile,
)
from .l0_refit_export import (
    US_RELEASE_REQUIRED_HOUSEHOLD_NONCONSTANT_SOURCE_COLUMNS,
    US_RELEASE_REQUIRED_HOUSEHOLD_SOURCE_COLUMNS,
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
    US_RELEASE_REQUIRED_SPM_UNIT_SOURCE_COLUMNS,
    US_RELEASE_REQUIRED_TAX_UNIT_SOURCE_COLUMNS,
)
from .prior_year_income_constants import US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS
from .survey_population_replay import same_replayed_frame

if TYPE_CHECKING:
    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

PROTOCOL = "microcosm.us.native-survey-development-handoff.v1"
# These are outstanding qualifications, not failed numerical measurements.
REQUIRED_RELEASE_EVIDENCE = (
    "native_leaf_source_and_applicability_coverage",
    "source_signal_gates_including_usual_hours",
    "prior_year_income_contract_reconciliation",
    "engine_input_projection_and_formula_ownership",
    "spm_structure_role_and_country_runtime_qualification",
    "chronicle_target_artifact_and_model_identity",
    "common_population_calibration_ancestry",
    "national_and_cd_target_fit",
    "export_readback_target_and_input_mass_parity",
    "reform_coverage_smoke",
    "matched_incumbent_no_worse_and_default_improvement",
)


@dataclass(frozen=True)
class NativeSurveyDevelopmentInput:
    """Usable Frame plus descriptive audit; this is not native-run authority."""

    frame: Frame
    report: dict
    owner_live_verified: bool = False


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_survey_input_inventory_scope() -> dict:
    """Name the descriptive rosters checked here, without live engine authority."""
    return {
        "checked_rosters": ["release_source_columns", "historical_input_profile"],
        "historical_profile": USInputProfile.HISTORICAL.value,
        "historical_source_manifest_sha256": MANIFEST_SHA256,
        "legacy_pool_input_surface_checked": False,
        "live_take_up_abi_inventory_qualified": False,
        "complete_engine_input_inventory_verified": False,
    }


def native_survey_input_inventory(frame: Frame) -> list[dict]:
    """Describe historical/release rosters; pool and live take-up remain unchecked.

    These are name/grain declarations, not a complete engine input contract.
    Calling the legacy pool's roster would instantiate a country engine through
    its agreement and take-up registries, even for storage-only inspection.
    """
    roster = {}
    for entity, columns in (
        ("person", US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS),
        ("tax_unit", US_RELEASE_REQUIRED_TAX_UNIT_SOURCE_COLUMNS),
        ("spm_unit", US_RELEASE_REQUIRED_SPM_UNIT_SOURCE_COLUMNS),
        (
            "household",
            (
                *US_RELEASE_REQUIRED_HOUSEHOLD_SOURCE_COLUMNS,
                *US_RELEASE_REQUIRED_HOUSEHOLD_NONCONSTANT_SOURCE_COLUMNS,
            ),
        ),
    ):
        for name in columns:
            roster.setdefault((entity, name), set()).add("release_source_columns")
    declared_names = {name for _, name in roster}
    for name in HISTORICAL_REQUIRED_INPUTS:
        if name in declared_names:
            continue
        entities = [entity for entity in frame.entities if name in frame.table(entity)]
        # A missing input's grain cannot be inferred from its name.
        entity = entities[0] if len(entities) == 1 else None
        if name == "household_weight":
            entity = "household"
        roster.setdefault((entity, name), set()).add("historical_input_profile")
    rows = []
    for (entity, name), declarations in sorted(
        roster.items(), key=lambda item: (item[0][0] or "", item[0][1])
    ):
        table = frame.table(entity) if entity in frame.entities else None
        present = table is not None and name in table
        typed_weight = (
            name == "household_weight" and "household" in frame.weighted_entities
        )
        unknown = int(table[name].isna().sum()) if present else None
        excluded = name in US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS
        status = (
            "excluded_native_scope"
            if excluded
            else "typed_weight"
            if typed_weight
            else "missing"
            if not present
            else "contains_unknowns"
            if unknown
            else "present"
        )
        rows.append(
            {
                "entity": entity,
                "variable": name,
                "declarations": sorted(declarations),
                "status": status,
                "missing_values": unknown,
                "source_signal_verified": False,
                "applicability_verified": False,
            }
        )
    return rows


def engine_export_inventory(frame: Frame, contract: ExportContract) -> dict:
    """Describe a supplied adapter contract; never drop columns or add defaults."""
    present = {column for entity in frame.entities for column in frame.table(entity)}
    present |= {entity + "_weight" for entity in frame.weighted_entities}
    structural = {frame.schema.entity_id_column(entity) for entity in frame.entities}
    structural |= {
        frame.schema.membership_column(entity) for entity in frame.schema.group_entities
    }
    allowed = set(contract.required) | set(contract.optional) | structural
    return {
        "contract_sha256": hashlib.sha256(
            _json(
                {
                    "required": contract.required,
                    "forbidden": contract.forbidden,
                    "optional": contract.optional,
                    "formula_owned_excluded": contract.formula_owned_excluded,
                    "closed": contract.closed,
                }
            )
        ).hexdigest(),
        "missing_required": sorted(set(contract.required) - present),
        "forbidden_present": sorted(set(contract.forbidden) & present),
        "formula_owned_present": sorted(set(contract.formula_owned_excluded) & present),
        "unexpected_columns": sorted(present - allowed) if contract.closed else [],
        "engine_compatibility_verified": False,
    }


def inspect_native_survey_release_input(
    run, *, export_contract=None
) -> NativeSurveyDevelopmentInput:
    """Borrow the real checked owner; retain its population without another clone."""
    view = native.check_survey_enrichment_run(run)
    owner_receipt = json.loads(view.payload)
    projection_record = run.manifest.node(native.PROJECTION_NODE)
    projection_payload = run.store.load_bytes(
        projection_record.opaque_artifacts["projection"]
    )
    projection = json.loads(projection_payload)
    if projection.get("prior_wages_consumed") is not False:
        raise ValueError("NATIVE_HANDOFF_PRIOR_WAGES_SCOPE")
    frame = view.population.frame
    inventory = native_survey_input_inventory(frame)
    report = {
        "protocol": PROTOCOL,
        "owner_receipt_sha256": view.digest,
        "owner_receipt": owner_receipt,
        "manifest_key": run.manifest.key,
        "population_version": view.population.version,
        "source_identities": dict(sorted(run.manifest.source_identities.items())),
        "amount_projection_sha256": hashlib.sha256(projection_payload).hexdigest(),
        "source_model_flags": {
            key: projection[key]
            for key in (
                "prior_wages_consumed",
                "PUF_donor_values_consumed",
                "held_out_quality_verified",
                "household_or_insurance_unit_correlation_verified",
                "clone_draw_policy",
                "ASEC_clone_policy",
                "donor_weights",
            )
        },
        "input_inventory": inventory,
        "input_inventory_scope": native_survey_input_inventory_scope(),
        "missing_inputs": [
            r for r in inventory if r["status"] in ("missing", "contains_unknowns")
        ],
        "excluded_native_scope": [
            r for r in inventory if r["status"] == "excluded_native_scope"
        ],
        "required_release_evidence": list(REQUIRED_RELEASE_EVIDENCE),
        "engine_contract": None if export_contract is None else asdict(export_contract),
        "engine_export": None
        if export_contract is None
        else engine_export_inventory(frame, export_contract),
        "release_eligible": False,
        "simulation_ready": False,
        "checkpoint_is_native_authority": False,
        "recloned": False,
        "weights_changed": False,
    }
    final = native.check_survey_enrichment_run(run)
    if final.digest != view.digest or final.population is not view.population:
        raise ValueError("NATIVE_HANDOFF_OWNER_CHANGED")
    return NativeSurveyDevelopmentInput(frame, report, True)


def _write_checkpoint(frame, report, directory):
    """Storage primitive, separated so invented tests need not imitate an issuer."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    try:
        metadata = {
            "protocol": PROTOCOL,
            "report_sha256": hashlib.sha256(_json(report)).hexdigest(),
            "frame_metadata": _encode_frame_metadata(frame.metadata),
        }
        path = write_frame_checkpoint(root / "population.h5", frame, metadata=metadata)
        restored = load_frame_checkpoint(path, frame_metadata=frame.metadata)
        same_replayed_frame(frame, restored.frame)
        document = {
            "protocol": PROTOCOL,
            "population_sha256": _sha_file(path),
            "report": report,
        }
        (root / "handoff.json").write_bytes(_json(document))
    except BaseException:
        # Only files created by this call; never erase an existing destination.
        for name in ("population.h5", "handoff.json"):
            (root / name).unlink(missing_ok=True)
        root.rmdir()
        raise


def load_native_survey_development_checkpoint(
    directory, *, run=None
) -> NativeSurveyDevelopmentInput:
    """Read exact Frame storage; only a retained owner supplies live authentication."""
    root = Path(directory)
    document = json.loads((root / "handoff.json").read_bytes())
    if document.get("protocol") != PROTOCOL or _sha_file(
        root / "population.h5"
    ) != document.get("population_sha256"):
        raise ValueError("NATIVE_HANDOFF_CHECKPOINT_IDENTITY")
    path = root / "population.h5"
    # Read external metadata without constructing the full Frame twice.
    from microcosm.build import frame_checkpoint

    with frame_checkpoint._h5py().File(path, "r") as h5:
        metadata = frame_checkpoint._read_metadata(h5[frame_checkpoint._ROOT], path)[
            "external_metadata"
        ]
    report = document["report"]
    if (
        metadata.get("protocol") != PROTOCOL
        or metadata.get("report_sha256") != hashlib.sha256(_json(report)).hexdigest()
    ):
        raise ValueError("NATIVE_HANDOFF_CHECKPOINT_REPORT")
    loaded = load_frame_checkpoint(
        path, frame_metadata=_decode_frame_metadata(metadata["frame_metadata"])
    )
    if _sha_file(path) != document["population_sha256"]:
        raise ValueError("NATIVE_HANDOFF_CHECKPOINT_CHANGED")
    live = False
    if run is not None:
        contract_values = report.get("engine_contract")
        contract = (
            None if contract_values is None else ExportContract(**contract_values)
        )
        borrowed = inspect_native_survey_release_input(run, export_contract=contract)
        final = native.check_survey_enrichment_run(run)
        # The owner check performs I/O. All comparisons of returned objects must
        # follow that last foreign call so it cannot mutate an already-checked
        # readback or descriptive report and escape the handoff checks.
        _validate_readback_after_owner_io(borrowed, loaded.frame, report, final)
        live = True
    return NativeSurveyDevelopmentInput(loaded.frame, report, live)


def _validate_readback_after_owner_io(borrowed, frame, report, final):
    """Pure final comparisons; these supplied values cannot issue run authority."""
    if (
        final.digest != borrowed.report["owner_receipt_sha256"]
        or final.population.frame is not borrowed.frame
        or _json(borrowed.report) != _json(report)
    ):
        raise ValueError("NATIVE_HANDOFF_CHECKPOINT_OWNER")
    same_replayed_frame(borrowed.frame, frame)


def write_native_survey_development_checkpoint(
    run, directory, *, export_contract=None
) -> NativeSurveyDevelopmentInput:
    """Export a real native owner and verify complete readback before returning."""
    borrowed = inspect_native_survey_release_input(run, export_contract=export_contract)
    _write_checkpoint(borrowed.frame, borrowed.report, directory)
    try:
        return load_native_survey_development_checkpoint(directory, run=run)
    except BaseException:
        # The writer owns this newly created directory; failed final owner checks
        # must not leave a completed-looking handoff document.
        (Path(directory) / "handoff.json").unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class NativeSurveyEngineProjectionSpec:
    """Small caller declaration, not a qualified consumer or input contract.

    ``columns`` lists all six entities, each with ordered (name, exact dtype
    string) pairs, including structural columns. Optional enum domains contain
    strings only and apply to known cells; nulls are preserved. An optional
    integer consumer ID dtype requests a losslessness check, never a cast.
    Required/optional names must also permit the Frame's typed weight inputs;
    those weights are preserved separately, never added as source columns.
    Consumer identity and SPM settings are descriptive and remain unqualified.
    """

    period: int | str
    consumer_identity: str
    columns: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    export_contract: ExportContract
    enum_domains: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    consumer_id_dtype: str | None = None
    spm_settings: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class NativeSurveyEngineProjection:
    """Borrowed source view and detached selection; neither issues authority."""

    source_frame: Frame
    frame: Frame
    report: dict


def _projection_require(condition, code):
    if not condition:
        raise ValueError("NATIVE_ENGINE_PROJECTION_" + code)


def _projection_spec_bytes(spec):
    _projection_require(type(spec) is NativeSurveyEngineProjectionSpec, "DECLARATION")
    _projection_require(type(spec.export_contract) is ExportContract, "CONTRACT_TYPE")
    return _json(asdict(spec))


def _projection_context(frame):
    from microcosm.frame import MassChangeRecord, Weights

    _projection_require(
        all(type(row) is MassChangeRecord for row in frame.mass_log)
        and all(
            type(frame.weights_for(entity)) is Weights
            for entity in frame.weighted_entities
        ),
        "CONTEXT_TYPES",
    )
    return _json(
        {
            "schema": asdict(frame.schema),
            "entities": frame.entities,
            "links": frame.links,
            "metadata": _encode_frame_metadata(frame.metadata),
            "mass_log": [asdict(row) for row in frame.mass_log],
            "weights": [
                {
                    "entity": entity,
                    "kind": frame.weights_for(entity).kind.value,
                    "dtype": frame.weights_for(entity).values.dtype.str,
                    "shape": frame.weights_for(entity).values.shape,
                    "sha256": hashlib.sha256(
                        frame.weights_for(entity).values.tobytes()
                    ).hexdigest(),
                }
                for entity in frame.weighted_entities
            ],
            "strata": native.physical._table_stamp(frame.strata.to_frame()),
        }
    )


def _projection_stamp(frame):
    """Small in-process comparison digest; never an owner/source credential."""
    return hashlib.sha256(
        _json(
            {
                "context": hashlib.sha256(_projection_context(frame)).hexdigest(),
                "tables": [
                    (entity, native.physical._table_stamp(frame.table(entity)))
                    for entity in frame.entities
                ],
            }
        )
    ).hexdigest()


def _compare_engine_projection(source, projected, spec):
    """Pure exact selection comparison, including storage under null masks."""
    import pandas as pd

    from microcosm.graph.population import storage_equal

    _projection_require(
        _projection_context(source) == _projection_context(projected), "FRAME_CONTEXT"
    )
    for entity, declarations in spec.columns:
        original = source.table(entity)
        actual = projected.table(entity)
        selected = original.loc[:, [name for name, _ in declarations]]
        _projection_require(
            selected.index.identical(actual.index)
            and selected.columns.identical(actual.columns)
            and selected.flags == actual.flags
            and storage_equal(
                pd.Series(selected.index.array), pd.Series(actual.index.array)
            ),
            "AXES",
        )
        _projection_require(
            all(storage_equal(selected[name], actual[name]) for name in selected),
            "SELECTED_STORAGE",
        )


def _project_native_survey_frame(frame, spec):
    """Pure mechanical projection for small invented tests; grants no authority.

    This does not establish consumer, source signal, applicability or scientific
    domain validity, even when every caller-declared dtype/enum check matches.
    """
    import numpy as np

    from microcosm.build.spm_input_contract import UNIVERSE_INPUT, UNIVERSE_STATUSES
    from microcosm.frame import MassChangeRecord, WeightKind, Weights
    from microcosm.frame.units import US_SCHEMA

    try:
        declaration = _projection_spec_bytes(spec)
        _projection_require(
            isinstance(frame, Frame) and frame.schema == US_SCHEMA and not frame.links,
            "SCHEMA",
        )
        _projection_require(
            type(spec.columns) is tuple
            and tuple(e for e, _ in spec.columns) == US_SCHEMA.entities,
            "ENTITY_ROSTER",
        )
        _projection_require(
            type(spec.period) in (int, str) and bool(str(spec.period)), "PERIOD"
        )
        _projection_require(
            type(spec.consumer_identity) is str and bool(spec.consumer_identity),
            "CONSUMER_DESCRIPTION",
        )
        _projection_require(
            type(spec.spm_settings) is tuple
            and all(
                type(k) is str and type(v) is str and k and v
                for k, v in spec.spm_settings
            )
            and len(dict(spec.spm_settings)) == len(spec.spm_settings),
            "SPM_DESCRIPTION",
        )
        contract = spec.export_contract
        _projection_require(contract.closed is True, "CLOSED_CONTRACT")
        rosters = (
            contract.required,
            contract.optional,
            contract.forbidden,
            contract.formula_owned_excluded,
        )
        _projection_require(
            all(
                type(row) is tuple
                and all(type(v) is str and v for v in row)
                and len(set(row)) == len(row)
                for row in rosters
            ),
            "CONTRACT_ROSTER",
        )
        _projection_require(
            not (set(contract.required) & set(contract.optional)), "CONTRACT_ROSTER"
        )
        excluded = (
            set(contract.forbidden)
            | set(contract.formula_owned_excluded)
            | set(US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS)
        )
        allowed = set(contract.required) | set(contract.optional)
        _projection_require(not (allowed & excluded), "EXCLUDED_INPUT")
        _projection_require(
            frame.weighted_entities == ("household",)
            and frame.weights_for("household").kind is WeightKind.IMPORTANCE,
            "WEIGHT_TOPOLOGY",
        )
        consumer_dtype = (
            None if spec.consumer_id_dtype is None else np.dtype(spec.consumer_id_dtype)
        )
        _projection_require(
            consumer_dtype is None or consumer_dtype.kind in ("i", "u"), "ID_CAST_DTYPE"
        )
        typed_weight_names = {entity + "_weight" for entity in frame.weighted_entities}
        _projection_require(not (typed_weight_names & excluded), "EXCLUDED_INPUT")
        _projection_require(typed_weight_names <= allowed, "UNDECLARED_TYPED_WEIGHT")
        reserved_weight_names = {entity + "_weight" for entity in frame.entities}
        tables, report_columns, excluded_columns = {}, [], []
        projected_names = set()
        for entity, declarations in spec.columns:
            _projection_require(
                type(declarations) is tuple
                and all(
                    type(name) is str and type(dtype) is str and name and dtype
                    for name, dtype in declarations
                ),
                "COLUMN_DECLARATION",
            )
            names = tuple(name for name, _ in declarations)
            _projection_require(len(set(names)) == len(names), "COLUMN_ROSTER")
            structural = {frame.schema.entity_id_column(entity)}
            if entity == frame.schema.person_entity:
                structural |= {
                    frame.schema.membership_column(group)
                    for group in frame.schema.group_entities
                }
            _projection_require(structural <= set(names), "STRUCTURAL_COLUMNS")
            original = frame.table(entity)
            for name, dtype in declarations:
                _projection_require(
                    name not in reserved_weight_names, "RESERVED_WEIGHT_SOURCE_COLUMN"
                )
                owners = [e for e in frame.entities if name in frame.table(e)]
                _projection_require(bool(owners), "MISSING_COLUMN")
                _projection_require(owners == [entity], "COLUMN_ENTITY")
                _projection_require(name not in excluded, "EXCLUDED_INPUT")
                _projection_require(
                    name in allowed or name in structural, "UNDECLARED_INPUT"
                )
                series = original[name]
                _projection_require(str(series.dtype) == dtype, "COLUMN_DTYPE")
                if name in structural:
                    values = series.to_numpy(copy=False)
                    _projection_require(
                        values.dtype.kind in ("i", "u") and not series.isna().any(),
                        "ID_DTYPE",
                    )
                    if consumer_dtype is not None and len(values):
                        bounds = np.iinfo(consumer_dtype)
                        _projection_require(
                            int(values.min()) >= bounds.min
                            and int(values.max()) <= bounds.max,
                            "ID_CAST_RANGE",
                        )
                        _projection_require(
                            np.array_equal(
                                values,
                                values.astype(consumer_dtype).astype(values.dtype),
                            ),
                            "ID_CAST_ROUNDTRIP",
                        )
                report_columns.append(
                    {
                        "entity": entity,
                        "column": name,
                        "dtype": dtype,
                        "rows": len(series),
                        "null_count": int(series.isna().sum()),
                    }
                )
            # Frame construction below owns the detached copies.
            tables[entity] = original.loc[:, list(names)]
            projected_names.update(names)
            excluded_columns.extend(
                {"entity": entity, "column": name, "rows": len(original)}
                for name in original
                if name not in names
            )
        _projection_require(
            set(contract.required) <= projected_names | typed_weight_names,
            "MISSING_REQUIRED",
        )
        _projection_require(type(spec.enum_domains) is tuple, "ENUM_DECLARATION")
        seen_domains = set()
        for entity, column, values in spec.enum_domains:
            _projection_require(
                (entity, column) not in seen_domains
                and entity in tables
                and column in tables[entity]
                and type(values) is tuple
                and values
                and all(type(v) is str for v in values)
                and len(set(values)) == len(values),
                "ENUM_DECLARATION",
            )
            seen_domains.add((entity, column))
            known = tables[entity][column].dropna()
            _projection_require(bool(known.isin(values).all()), "DECLARED_DOMAIN")
        projected = Frame(
            tables,
            frame.schema,
            {
                entity: Weights(
                    frame.weights_for(entity).values, frame.weights_for(entity).kind
                )
                for entity in frame.weighted_entities
            },
            frame.strata.copy(deep=True),
            metadata=frame.metadata,
            mass_log=tuple(MassChangeRecord(**asdict(row)) for row in frame.mass_log),
        )
        _compare_engine_projection(frame, projected, spec)
        status = projected.table("spm_unit").get(UNIVERSE_INPUT)
        report = {
            "protocol": "microcosm.us.native-survey-engine-projection.v1",
            "declaration_kind": "caller_supplied_unqualified",
            "declaration_sha256": hashlib.sha256(declaration).hexdigest(),
            "declared_consumer_identity": spec.consumer_identity,
            "declared_period": spec.period,
            "declared_spm_settings": dict(spec.spm_settings),
            "consumer_id_dtype": spec.consumer_id_dtype,
            "id_cast_performed": False,
            "typed_weight_inputs": [
                {
                    "entity": entity,
                    "name": entity + "_weight",
                    "kind": frame.weights_for(entity).kind.value,
                    "rows": frame.n(entity),
                    "materialized_as_source_column": False,
                }
                for entity in frame.weighted_entities
            ],
            "columns": report_columns,
            "excluded_columns": excluded_columns,
            "projected_frame_sha256": _projection_stamp(projected),
            "source_context_sha256": hashlib.sha256(
                _projection_context(frame)
            ).hexdigest(),
            "headship_unknown_count": None
            if "is_household_head" not in projected.person
            else int(projected.person.is_household_head.isna().sum()),
            "spm_status_counts": None
            if status is None
            else {
                name: int(status.eq(name).sum()) for name in sorted(UNIVERSE_STATUSES)
            },
            "spm_status_null_count": None
            if status is None
            else int(status.isna().sum()),
            "spm_status_other_count": None
            if status is None
            else int((status.notna() & ~status.isin(UNIVERSE_STATUSES)).sum()),
            "consumer_qualified": False,
            "period_qualified": False,
            "formula_ownership_qualified": False,
            "source_signal_qualified": False,
            "domain_qualified": False,
            "source_applicability_qualified": False,
            "simulation_ready": False,
            "release_eligible": False,
            "source_owner_required": True,
            "source_defaults_filled": False,
            "enrichment_repeated": False,
        }
        _json(report)
        return NativeSurveyEngineProjection(frame, projected, report)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if type(error) is ValueError and str(error).startswith(
            "NATIVE_ENGINE_PROJECTION_"
        ):
            raise
        # Frame/schema codecs may include row examples in their errors. Never
        # expose those through this descriptive native-reporting entry point.
        raise ValueError("NATIVE_ENGINE_PROJECTION_STRUCTURE") from None


def _validate_engine_projection_after_owner_io(result, spec, final, *, expected):
    """Pure last-fence comparisons; supplied descriptive values issue nothing."""
    try:
        owner_digest, population, frame_stamp, report_bytes, declaration_bytes = (
            expected
        )
        _projection_require(
            final.digest == owner_digest
            and final.population is population
            and final.population.frame is result.source_frame,
            "OWNER_CHANGED",
        )
        _projection_require(
            _projection_spec_bytes(spec) == declaration_bytes, "DECLARATION_CHANGED"
        )
        _projection_require(
            _projection_stamp(result.frame) == frame_stamp
            and _json(result.report) == report_bytes,
            "RESULT_CHANGED",
        )
        _compare_engine_projection(result.source_frame, result.frame, spec)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        if type(error) is ValueError and str(error).startswith(
            "NATIVE_ENGINE_PROJECTION_"
        ):
            raise
        raise ValueError("NATIVE_ENGINE_PROJECTION_FINAL_STRUCTURE") from None


def _validate_engine_projection_consumer(result, declaration, consumer):
    """Optional representation check; this helper cannot issue native authority."""
    if consumer is None:
        return
    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    _projection_require(isinstance(consumer, PolicyEngineUSEngine), "CONSUMER_TYPE")
    consumer.validate_input_representation(result.frame, period=declaration.period)
    _compare_engine_projection(result.source_frame, result.frame, declaration)
    result.report["consumer_representation_compatible"] = True


def prepare_native_survey_engine_input(
    run: native.SurveyEnrichmentRun,
    *,
    declaration: NativeSurveyEngineProjectionSpec,
    consumer: PolicyEngineUSEngine | None = None,
) -> NativeSurveyEngineProjection:
    """Select cells from a live issued owner without admitting an engine consumer.

    Source and projected views remain separate. Checkpoints, receipts, Frames,
    and forged run dataclasses cannot pass the existing owner's public check.
    Omitting consumer imports no engine. An optional adapter checks representation
    against its live metadata, without runtime or scientific qualification. No
    source variable is cast, filled, or recomputed in either path.
    """
    # Authenticate before even examining the caller's descriptive declaration.
    borrowed = inspect_native_survey_release_input(run)
    population = run.population
    result = _project_native_survey_frame(borrowed.frame, declaration)
    _validate_engine_projection_consumer(result, declaration, consumer)
    result.report["owner_receipt_sha256"] = borrowed.report["owner_receipt_sha256"]
    expected = (
        borrowed.report["owner_receipt_sha256"],
        population,
        _projection_stamp(result.frame),
        _json(result.report),
        _projection_spec_bytes(declaration),
    )
    final = native.check_survey_enrichment_run(run)
    _validate_engine_projection_after_owner_io(
        result, declaration, final, expected=expected
    )
    return result
