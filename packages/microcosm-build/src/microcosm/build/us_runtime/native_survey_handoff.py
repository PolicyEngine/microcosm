"""Development handoff from an issued native owner to maintained Frame consumers.

No assembly, cloning, fitting, default filling, engine projection or calibration
occurs here. A checkpoint is descriptive storage, never a replacement issuer.
Release admission still belongs to the maintained builder's scientific gates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.frame import Frame
from microcosm.frame.rules import ExportContract
from microcosm.graph.store import _decode_frame_metadata, _encode_frame_metadata

from . import graph_us_survey_enrichment as native
from .input_coverage_profile import HISTORICAL_REQUIRED_INPUTS
from .l0_refit_export import (
    US_RELEASE_REQUIRED_HOUSEHOLD_NONCONSTANT_SOURCE_COLUMNS,
    US_RELEASE_REQUIRED_HOUSEHOLD_SOURCE_COLUMNS,
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
    US_RELEASE_REQUIRED_SPM_UNIT_SOURCE_COLUMNS,
    US_RELEASE_REQUIRED_TAX_UNIT_SOURCE_COLUMNS,
)
from .multispine_pool import pool_input_surface
from .prior_year_income import US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS
from .survey_population_replay import same_replayed_frame

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


def native_survey_input_inventory(frame: Frame) -> list[dict]:
    """Compare maintained rosters without inventing sources or applicability."""
    roster = {}
    for entry in pool_input_surface():
        roster.setdefault((entry.entity, entry.variable), set()).add(
            "pool:" + entry.family
        )
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
