"""Score US incumbent and candidate artifacts on one replacement yardstick.

The owner's publish bar for the next US artifact is a head-to-head: the live
artifact versus the candidate on the SAME yardstick, flipped on evidence.
This tool is that yardstick. It has two parts:

* every row of one compiled fiscal target registry, evaluated under each
  artifact's own shipped weights with production's concept-budgeted
  amount/count loss; and
* the terminal by-origin battery: reported from an authenticated pool
  manifest receipt when the artifact is a pool, computed on an ephemeral
  terminal gate view for a finished H5 carrying both origins, and reported as
  observed inapplicability (never synthesized, never faked) when the artifact
  carries no evaluable ASEC-vs-ACS surface.

Artifacts differ only at the loading boundary. Both normalized frames pass
through the same population repair, target materialization, constraint
matrix, scoring, loss attribution, contract checks, and rendering path.
Scoring is sequential and refuses a process peak at or above 20 GiB RSS.

Memory design, from measurements on the live incumbent (57,240 households,
166,321 persons): the unbatched full-frame base microsimulation alone peaks
at ~31 GiB (the federal income-tax DAG holds every intermediate array), and
the materializer additionally emits one full-length float64 household
column per registry spec (32,842 distinct measures post-#741, ~15 GB
whole-registry). Neither fits the 20 GiB scoring budget, so both sides run
the same two-axis decomposition through unmodified canonical calls:

* households are sliced with the production reform-sweep batching machinery
  (`_household_position_batches` + `_select_households_by_position`,
  tools/build_us_fiscal_refresh_release.py:3730-3750), and each slice is
  materialized and scored through the canonical calls before it is freed;
* registry specs are scored in fixed contiguous chunks. A chunk's estimates
  are additive matrix products, so slice estimates are accumulated in fixed
  household order; targets, scales, names, and scored-column contracts must
  be identical on every slice. No chunk-wide dense household-by-measure table
  is assembled. The aggregate is one `relative_error_loss` evaluation over
  the full combined vectors — the single canonical loss definition every
  measurement imports (packages/microcosm-calibrate/src/microcosm/calibrate/
  solve.py:471-537).

The per-measure reform-vector cache is refused under household slicing:
its identity is (context, measure, n_households), so equal-sized slices of
one artifact would collide into one cache entry and poison each other
(tools/build_us_fiscal_refresh_release.py:1717-1739).

No gate, threshold, tolerance, or band is applied to the comparison: the
output is evidence for the owner's flip decision, not a verdict.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import build_us_fiscal_refresh_release as release
import h5py
import numpy as np
import pandas as pd
import score_us_fiscal_targets as fiscal_scorer

from microcosm.build.us_runtime.congressional_district_vintage import (
    CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
    CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
)
from microcosm.build.us_runtime.h5_io import (
    US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
    AuthenticatedPoolH5,
    load_authenticated_us_multispine_pool_for_scoring,
    read_nullable_us_h5_metadata,
)
from microcosm.build.us_runtime.multispine_pool import (
    materialize_multispine_agreement_outputs,
)
from microcosm.build.us_runtime.stacked_spine import (
    ACS_STACKED_SUPPORT_CHANNEL,
    CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY,
    CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY,
    by_origin_battery_artifact_evidence,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    support_channel_column,
    support_clone_index_column,
)
from microcosm.calibrate import TargetRegistry, score_targets
from microcosm.calibrate.solve import relative_error_loss
from microcosm.frame import US_SCHEMA, Frame

SCHEMA_VERSION = 3
MAX_RSS_BYTES = 20 * 1024**3
MARKDOWN_WORST_TARGET_ROWS = 50
# Registry chunk size for streaming materialize-and-score. The target-column
# payload is bounded by the household slice, never total artifact size. Three
# float64-sized copies per cell conservatively cover materialized columns,
# matrix storage, and scoring temporaries: 8,192 x 5,000 x 8 x 3 is under
# 1 GiB, on top of one live household-slice microsimulation. Chunk and slice
# boundaries are fixed and content-independent, so output bytes are
# deterministic.
MATERIALIZE_SCORE_CHUNK_SPECS = 8_192
_STREAMING_TARGET_COLUMN_COPIES = 3

# The package resolver authority for the live US incumbent, read from
# policyengine.py 5.0.3 (PyPI latest, tagged 2026-08-21) this lane session:
# the bundled manifest's US entry certifies populace_us_2024 as the default
# dataset, resolve_managed_dataset_reference(country, dataset=None) returns
# manifest.default_dataset_uri, and dataset overlays are additive-only so
# they can never shadow the default.
# policyengine.py@5.0.3 src/policyengine/data/bundle/manifest.json:113-140,
#   156-160,181-189 (US artifact, default, and model pins)
# policyengine.py@5.0.3 src/policyengine/provenance/manifest.py:180-187
#   (default_dataset_uri), :270-299 (overlay no-shadow), :301-318
#   (get_release_manifest), :540-560
#   (resolve_managed_dataset_reference default path)
_POLICYENGINE_LIVE_US_INCUMBENT = {
    "policyengine_version": "5.0.3",
    "policyengine_source_commit": "cfdd128fc316e07ef54c182f2149fac217e8706f",
    "bundle_id": "us-5.0.3",
    "data_producer": "populace",
    "default_dataset": "populace_us_2024",
    "build_id": "populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z",
    "repo_id": "policyengine/populace-us",
    "repo_type": "dataset",
    "revision": "populace-us-2024-buildp-sparse-rmloss100-cae8640-20260728T011454Z",
    "filename": "populace_us_2024.h5",
    "sha256": "48b9d479fb4fd1c3537f9383ce4697d130b6f618658409d74f6233c43b994c7e",
    "resolved_hf_commit": "26dcad66867687f15735dc4926523e3741920836",
    "certified_for_model_version": "1.764.6",
}

_CODE_CITATIONS = {
    "single_registry_surface": (
        "packages/microcosm-build/src/microcosm/build/us_runtime/"
        "fiscal_targets.py:933-1017"
    ),
    "canonical_scorer_seam": "tools/score_us_fiscal_targets.py:383-528",
    "entity_h5_loader": "tools/build_us_fiscal_refresh_release.py:2454-2471",
    "legacy_flat_loader": "tools/score_us_fiscal_targets.py:240-333",
    "pool_manifest_authentication": (
        "packages/microcosm-build/src/microcosm/build/us_runtime/"
        "h5_io.py:371-588,719-869"
    ),
    "materialization_drop_detection": (
        "tools/build_us_fiscal_refresh_release.py:4323-4350"
    ),
    "matrix_skip_detection": (
        "packages/microcosm-calibrate/src/microcosm/calibrate/matrix.py:286-355"
    ),
    "loss_weighting": (
        "tools/build_us_fiscal_refresh_release.py:344-348,481-516,5781-5814,6214-6290"
    ),
    "loss_aggregate": (
        "packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:"
        "471-537,576-600 "
        "(relative_error_loss, the single canonical loss definition)"
    ),
    "fraction_within_10pct": (
        "packages/microcosm-calibrate/src/microcosm/calibrate/solve.py:249-260"
    ),
    "loss_attribution": (
        "packages/microcosm-calibrate/src/microcosm/calibrate/"
        "_target_loss_attribution.py:143-176 (per-row capped error, weight "
        "share, and contribution formulas, applied with whole-registry "
        "weight normalization)"
    ),
    "chunked_materialize_score": (
        "tools/score_us_release_head_to_head.py:654-852,1399-1477; "
        "tools/build_us_fiscal_refresh_release.py:3730-3750; "
        "packages/microcosm-calibrate/src/microcosm/calibrate/"
        "matrix.py:286-355; score.py:79-142"
    ),
    "relative_error": (
        "packages/microcosm-calibrate/src/microcosm/calibrate/score.py:25-51"
    ),
    "battery_registry": (
        "packages/microcosm-build/src/microcosm/build/us_runtime/"
        "stacked_spine.py:3011-3025"
    ),
    "battery_origin_masks": (
        "packages/microcosm-build/src/microcosm/build/us_runtime/"
        "stacked_spine.py:11644-11709,11824-11832"
    ),
    "battery_receipt_keys": (
        "packages/microcosm-build/src/microcosm/build/us_runtime/"
        "stacked_spine.py:11948-12154"
    ),
    "battery_finished_h5_materialization": (
        "packages/microcosm-build/src/microcosm/build/us_runtime/"
        "multispine_pool.py:3137-3215 (ephemeral SSI gate view); "
        "packages/microcosm-build/src/microcosm/build/us_runtime/"
        "stacked_spine.py:7808-7897,11474-11492,11530-11898 "
        "(finished-artifact evidence seam and canonical evaluator)"
    ),
    "battery_channel_constants": (
        "packages/microcosm-build/src/microcosm/build/us_runtime/"
        "support_provenance.py:31,333-344; "
        "packages/microcosm-build/src/microcosm/build/us_runtime/"
        "stacked_spine.py:263"
    ),
    "pool_battery_persistence": (
        "tools/build_us_multispine_pool.py:3263-3334,3533-3550,3766-3786; "
        "packages/microcosm-build/src/microcosm/build/gates.py:690-700"
    ),
    "cd_provenance_check": (
        "tools/build_us_fiscal_refresh_release.py:2519-2567,2570-2597"
    ),
    "incumbent_package_resolution": (
        "policyengine.py@5.0.3 src/policyengine/data/bundle/manifest.json:"
        "113-140,156-160,181-189; src/policyengine/provenance/manifest.py:"
        "180-187,270-299,301-318,540-560; src/policyengine/provenance/"
        "dataset_sources.py:57-74,77-117; src/policyengine/"
        "tax_benefit_models/us/model.py:423-462"
    ),
}


def _empty_historical_formula_owned_columns_receipt() -> dict[str, object]:
    return {
        "count": 0,
        "columns_by_entity": {},
    }


@dataclass(frozen=True)
class LoadedArtifact:
    """One role-neutral artifact normalized to a US entity frame."""

    frame: Frame
    identity: Mapping[str, object]
    loader: Mapping[str, object]
    h5_path: Path
    terminal_gates: Mapping[str, object] | None = None
    authenticated_pool_h5: AuthenticatedPoolH5 | None = None
    historical_formula_owned_columns: Mapping[str, object] = field(
        default_factory=_empty_historical_formula_owned_columns_receipt
    )


@dataclass(frozen=True)
class FiscalYardstick:
    """The single compiled registry and exact production loss basis."""

    registry: TargetRegistry
    loss_weights: np.ndarray
    loss_basis: Mapping[str, object]
    identity: Mapping[str, object]


@dataclass(frozen=True)
class ScoredChunk:
    """One fixed registry chunk reduced across fixed household slices."""

    estimates: np.ndarray
    targets: np.ndarray
    scales: np.ndarray
    diagnostic_names: tuple[str, ...]
    scored_contract: tuple[tuple[str, str, str], ...]
    compilation: Mapping[str, object]


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score a US incumbent and optional candidate on one fiscal registry "
            "and every artifact-computable terminal battery comparison."
        )
    )
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        help="Candidate entity H5, ready pool manifest, or pool directory.",
    )
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        required=True,
        help="Ledger consumer facts JSONL that freezes the common target registry.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        required=True,
        help="Write <prefix>.json and <prefix>.md.",
    )
    parser.add_argument(
        "--candidate-manifest-sha256",
        help="Optional external SHA-256 pin for a candidate pool manifest.",
    )
    parser.add_argument(
        "--age-targets",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--allow-unaged-dollar-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--congressional-district-vintage-crosswalk",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--maximum-microsim-batch-size",
        "--maximum-microsimulation-batch-size",
        dest="maximum_microsim_batch_size",
        type=int,
        default=release.DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the other supported CI hosts report KiB.
    return peak if sys.platform == "darwin" else peak * 1024


def _assert_rss_below_limit(boundary: str) -> None:
    observed = _peak_rss_bytes()
    # Operational receipt on stderr only; JSON/Markdown outputs stay
    # byte-deterministic.
    print(
        f"[h2h {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] "
        f"{boundary}: peak_rss={observed / 1024**3:.2f} GiB",
        file=sys.stderr,
        flush=True,
    )
    if observed >= MAX_RSS_BYTES:
        raise MemoryError(
            f"{boundary}: peak RSS {observed / 1024**3:.3f} GiB reached the "
            f"{MAX_RSS_BYTES / 1024**3:.0f} GiB scoring limit."
        )


def _pool_manifest_from_directory(path: Path) -> Path:
    candidates = sorted(path.glob("*.manifest.json"))
    candidates.extend(
        candidate
        for candidate in (path / "manifest.json", path / "pool.manifest.json")
        if candidate.is_file() and candidate not in candidates
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Pool directory {path} must contain exactly one direct pool "
            f"manifest; found {[candidate.name for candidate in candidates]}."
        )
    return candidates[0]


def _resolved_artifact_path(path: Path) -> Path:
    # Never resolve() the artifact file path: Hugging Face cache snapshots
    # are symlinks named <file>.h5 pointing at extensionless blob files, so
    # resolving would both strip the .h5 suffix the policyengine-us dataset
    # loader validates and replace the meaningful filename identity with a
    # blob hash.
    expanded = path.expanduser()
    if expanded.is_dir():
        return _pool_manifest_from_directory(expanded)
    if not expanded.is_file():
        raise FileNotFoundError(f"Scoring artifact is not a file: {expanded}")
    return expanded


def _h5_layout(path: Path) -> str:
    if not h5py.is_hdf5(path):
        raise ValueError(f"Scoring input is not an HDF5 file: {path}")
    with h5py.File(path, "r") as root:
        keys = set(root.keys())
    entity_layout = set(US_SCHEMA.entities).issubset(keys)
    legacy_layout = "person_id" in keys and "household_weight" in keys
    if entity_layout == legacy_layout:
        raise ValueError(
            f"H5 {path} has an ambiguous or unsupported US layout; "
            f"entity_layout={entity_layout}, legacy_flat_layout={legacy_layout}."
        )
    return "entity_tables" if entity_layout else "legacy_policyengine_flat"


def _live_incumbent_identity_if_matched(sha256: str) -> dict[str, object] | None:
    if sha256 != _POLICYENGINE_LIVE_US_INCUMBENT["sha256"]:
        return None
    return {
        "policyengine_package_resolved": dict(_POLICYENGINE_LIVE_US_INCUMBENT),
        "resolution_citation": _CODE_CITATIONS["incumbent_package_resolution"],
    }


def _drop_historical_formula_owned_columns(
    frame: Frame,
) -> tuple[Frame, dict[str, object]]:
    """Normalize one loaded artifact to current-engine input leaves.

    Historical H5 artifacts can carry columns that were inputs under the
    PolicyEngine-US version that built them but are formula-owned under the
    scorer's current locked engine. The fresh-release builder must reject such
    columns; this scorer-only loading seam instead proves that every current
    formula input leaf is present, drops the stale derived columns, and lets
    the one comparison engine recompute them.
    """

    metadata_index = release._formula_owned_gate_adapter()
    tables = {entity: frame.table(entity) for entity in frame.entities}
    formula_owned = metadata_index._engine_computed_columns(
        tables,
        period=release.PERIOD,
    )
    if not formula_owned:
        return frame, _empty_historical_formula_owned_columns_receipt()

    missing_leaves_by_column: dict[str, list[str]] = {}
    for column in sorted(formula_owned):
        closure = metadata_index.variable_dependency_closure(column)
        missing_leaves = sorted(
            leaf
            for leaf in closure.input_leaves
            if (
                (entity := metadata_index.variable_metadata(leaf).entity) not in tables
                or leaf not in tables[entity].columns
            )
        )
        if missing_leaves:
            missing_leaves_by_column[column] = missing_leaves
    if missing_leaves_by_column:
        details = "; ".join(
            f"{column}: {missing_leaves}"
            for column, missing_leaves in missing_leaves_by_column.items()
        )
        raise ValueError(
            "Historical scoring artifact has formula-owned PolicyEngine "
            "column(s) that cannot be recomputed under the current engine "
            f"because required input leaves are absent: {details}."
        )

    columns_by_entity = {
        entity: sorted(set(tables[entity].columns) & formula_owned)
        for entity in sorted(frame.entities)
        if set(tables[entity].columns) & formula_owned
    }
    cleaned_tables = {}
    for entity in frame.entities:
        columns = columns_by_entity.get(entity)
        cleaned_tables[entity] = (
            tables[entity].drop(columns=columns)
            if columns
            else tables[entity].copy()
        )
    cleaned_weights = {
        entity: frame.weights_for(entity) for entity in frame.weighted_entities
    }
    cleaned = Frame(
        cleaned_tables,
        frame.schema,
        cleaned_weights,
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    return cleaned, {
        "count": sum(len(columns) for columns in columns_by_entity.values()),
        "columns_by_entity": columns_by_entity,
    }


def _load_pool_manifest(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str | None,
) -> LoadedArtifact:
    frame, manifest, authenticated = load_authenticated_us_multispine_pool_for_scoring(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    frame, formula_owned_receipt = _drop_historical_formula_owned_columns(frame)
    terminal_gates = manifest.get("terminal_gates")
    if not isinstance(terminal_gates, Mapping):
        raise ValueError(
            f"Authenticated pool manifest {manifest_path} has no terminal_gates "
            "receipt; a current stacked publication always seals one."
        )
    identity = {
        "kind": "authenticated_pool",
        "filename": authenticated.path.name,
        "sha256": authenticated.sha256,
        "size_bytes": authenticated.size_bytes,
        "manifest_filename": manifest_path.name,
        "manifest_sha256": authenticated.manifest_sha256,
        "publication_run_id": authenticated.publication_run_id,
        "release_id": manifest.get("release_id"),
    }
    return LoadedArtifact(
        frame=frame,
        identity=identity,
        loader={
            "kind": "authenticated_pool_manifest_for_scoring",
            "weight_kind": frame.weights_for("household").kind.value,
            "publication_status": manifest.get("status"),
            "simulation_ready": manifest.get("simulation_ready"),
        },
        h5_path=authenticated.path,
        terminal_gates=terminal_gates,
        authenticated_pool_h5=authenticated,
        historical_formula_owned_columns=formula_owned_receipt,
    )


def _load_h5(path: Path) -> LoadedArtifact:
    layout = _h5_layout(path)
    if layout == "entity_tables":
        try:
            metadata = read_nullable_us_h5_metadata(path)
        except ValueError:
            metadata = {}
        if metadata.get("artifact_kind") == US_MULTISPINE_POOL_H5_ARTIFACT_KIND:
            raise ValueError(
                f"{path} is a naked multispine pool H5. Pass its companion "
                "manifest so readiness, weights, diagnostics, and terminal "
                "battery receipts are authenticated."
            )
        # The exact loader the canonical read-only fiscal scorer uses
        # (tools/score_us_fiscal_targets.py:436 -> release._load_frame).
        frame = release._load_frame(path)
        loader: dict[str, object] = {
            "kind": "microcosm_entity_h5",
            "weight_kind": frame.weights_for("household").kind.value,
        }
    else:
        frame, layout_receipt = fiscal_scorer._load_legacy_pe_flat_frame(path)
        loader = {
            "kind": "legacy_policyengine_flat_h5",
            "weight_kind": frame.weights_for("household").kind.value,
            "layout_receipt": layout_receipt,
        }
    frame, formula_owned_receipt = _drop_historical_formula_owned_columns(frame)
    sha256 = _sha256(path)
    identity: dict[str, object] = {
        "kind": "h5",
        "filename": path.name,
        "sha256": sha256,
        "size_bytes": path.stat().st_size,
    }
    published = _live_incumbent_identity_if_matched(sha256)
    if published is not None:
        identity.update(published)
    return LoadedArtifact(
        frame=frame,
        identity=identity,
        loader=loader,
        h5_path=path,
        historical_formula_owned_columns=formula_owned_receipt,
    )


def load_artifact(
    path: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> LoadedArtifact:
    """Load a role-neutral H5 or authenticated pool into the common frame API."""

    resolved = _resolved_artifact_path(path)
    if resolved.suffix.lower() == ".json":
        artifact = _load_pool_manifest(
            resolved,
            expected_manifest_sha256=expected_manifest_sha256,
        )
    else:
        if expected_manifest_sha256 is not None:
            raise ValueError(
                "--candidate-manifest-sha256 applies only to a pool manifest."
            )
        artifact = _load_h5(resolved)
    _assert_rss_below_limit(f"after loading {artifact.identity['filename']}")
    return artifact


def compile_yardstick(
    *,
    ledger_facts: Path,
    age_targets: bool,
    allow_unaged_dollar_targets: bool,
    congressional_district_vintage_crosswalk: Path,
) -> FiscalYardstick:
    """Compile the complete registry and loss vector exactly once."""

    ledger_facts = ledger_facts.expanduser().resolve()
    crosswalk = congressional_district_vintage_crosswalk.expanduser().resolve()
    ledger = release.load_ledger_consumer_artifact(ledger_facts)
    registry = release.compile_us_fiscal_target_registry(
        ledger.facts,
        target_period=release.PERIOD,
        age_targets=age_targets,
        allow_unaged_dollar_targets=allow_unaged_dollar_targets,
        congressional_district_vintage_crosswalk=(
            release.load_congressional_district_vintage_crosswalk(crosswalk)
        ),
    )
    loss_weights = release._fiscal_target_loss_weights(registry)
    loss_basis = release._fiscal_target_loss_basis(registry, loss_weights)
    profile_gate = release.target_profile_coverage_gate(
        registry.specs,
        release.US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    identity = {
        "country": registry.country,
        "version": registry.version,
        "target_count": len(registry.specs),
        "ledger_facts": {
            "filename": ledger_facts.name,
            "sha256": _sha256(ledger_facts),
            "size_bytes": ledger_facts.stat().st_size,
        },
        "congressional_district_vintage_crosswalk": {
            "filename": crosswalk.name,
            "sha256": _sha256(crosswalk),
            "size_bytes": crosswalk.stat().st_size,
        },
        "target_period": release.PERIOD,
        "age_targets": age_targets,
        "allow_unaged_dollar_targets": allow_unaged_dollar_targets,
        "target_profile_coverage": {
            "passed": profile_gate.passed,
            "failures": list(profile_gate.failures),
        },
        "environment": {
            "microcosm_commit": release._git_output("rev-parse", "HEAD"),
            "policyengine_us_version": release._package_or_workspace_version(
                "policyengine-us"
            ),
        },
    }
    _assert_rss_below_limit("after compiling the fiscal yardstick")
    return FiscalYardstick(
        registry=registry,
        loss_weights=loss_weights,
        loss_basis=loss_basis,
        identity=identity,
    )


def _crosswalk_metadata(yardstick: FiscalYardstick) -> dict[str, object]:
    value = yardstick.identity["congressional_district_vintage_crosswalk"]
    if not isinstance(value, Mapping):  # pragma: no cover - constructed above
        raise TypeError("Yardstick crosswalk identity must be a mapping.")
    return dict(value)


def _validate_cd_provenance(
    artifact: LoadedArtifact,
    yardstick: FiscalYardstick,
) -> dict[str, object]:
    """Enforce the strict CD crosswalk provenance check, with one recorded
    waiver for entity H5s that predate the provenance attributes entirely."""

    if artifact.loader.get("kind") == "legacy_policyengine_flat_h5":
        return {
            "mode": "legacy_flat_surface_compatibility",
            "strict_h5_attributes": False,
            "reason": (
                "read-only scoring of a legacy flat H5 layout that predates "
                "Microcosm crosswalk provenance attributes and entity tables"
            ),
        }
    provenance = release._read_cd_vintage_support_provenance(artifact.h5_path)
    sha_attr = provenance.get(CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR)
    if sha_attr:
        release._assert_cd_vintage_support_matches(
            artifact.h5_path,
            _crosswalk_metadata(yardstick),
            authenticated_pool_h5=artifact.authenticated_pool_h5,
        )
        return {
            "mode": "strict_h5_provenance",
            "strict_h5_attributes": True,
        }
    if artifact.authenticated_pool_h5 is not None:
        raise ValueError(
            f"Authenticated pool H5 {artifact.h5_path} carries no "
            "congressional-district vintage provenance attributes; candidate "
            "pools are provenance-strict with no legacy waiver."
        )
    lookup = provenance.get("household_congressional_district_geoid")
    if (
        not isinstance(lookup, Mapping)
        or not lookup.get("exists")
        or int(lookup.get("positive_unique_count") or 0) <= 0
    ):
        raise ValueError(
            f"{artifact.h5_path} predates CD vintage provenance attributes AND "
            "has no usable household congressional_district_geoid lookup "
            f"column ({lookup!r}); it cannot stand on the current CD-bearing "
            "target surface."
        )
    return {
        "mode": "legacy_missing_cd_provenance_attrs",
        "strict_h5_attributes": False,
        "reason": (
            "entity H5 predates the CD vintage crosswalk provenance "
            "attributes; the household congressional_district_geoid lookup "
            "surface was verified instead"
        ),
        "observed": {
            CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR: sha_attr,
            CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR: provenance.get(
                CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR
            ),
            "household_congressional_district_geoid": dict(lookup),
        },
    }


def _streaming_target_column_payload_upper_bound_bytes(
    *,
    total_households: int,
    household_slice_size: int | None,
    chunk_spec_count: int,
) -> int:
    """Conservative dense target-column payload bound for one live slice."""

    if total_households < 0 or chunk_spec_count < 0:
        raise ValueError("Streaming memory dimensions must be non-negative.")
    if household_slice_size is None or household_slice_size <= 0:
        live_households = total_households
    else:
        live_households = min(total_households, household_slice_size)
    return (
        live_households
        * chunk_spec_count
        * np.dtype(np.float64).itemsize
        * _STREAMING_TARGET_COLUMN_COPIES
    )


def _score_chunk_household_sliced(
    base_frame: Frame,
    chunk_specs: Sequence,
    *,
    chunk_loss_weights: np.ndarray,
    artifact_name: str,
    chunk_label: str,
    maximum_microsim_batch_size: int | None,
) -> ScoredChunk:
    """Materialize, score, and reduce one chunk without a dense full-pool table.

    Each household slice runs the unmodified canonical materializer and
    scorer. A target estimate is a matrix-row/weight dot product, so the full
    artifact estimate is the fixed-order sum of slice estimates. Every slice
    must reproduce the exact target, scale, diagnostic-name, and scored-column
    contracts before its estimates may enter that sum.
    """

    n_households = base_frame.n("household")
    expected_keys = _spec_keys(chunk_specs)
    slice_batches = tuple(
        release._household_position_batches(
            n_households,
            maximum_microsim_batch_size,
        )
    )
    if not slice_batches:
        raise ValueError(f"{artifact_name} has no households to score.")
    accumulated_estimates: np.ndarray | None = None
    reference_targets: np.ndarray | None = None
    reference_scales: np.ndarray | None = None
    reference_names: tuple[str, ...] | None = None
    reference_contract: tuple[tuple[str, str, str], ...] | None = None
    first_compilation: dict[str, object] | None = None
    compilation_digests: list[str] = []
    slice_sizes: list[int] = []
    for slice_index, positions in enumerate(slice_batches):
        slice_label = f"{chunk_label} slice {slice_index + 1}/{len(slice_batches)}"
        full_slice = len(positions) == n_households
        slice_frame = (
            base_frame
            if full_slice
            else release._select_households_by_position(base_frame, positions)
        )
        slice_target_frame, slice_registry, slice_compilation = (
            release._materialize_target_frame(
                slice_frame,
                chunk_specs,
                maximum_microsim_batch_size=maximum_microsim_batch_size,
                target_materialization_cache_dir=None,
                target_materialization_cache_context=None,
            )
        )
        _assert_nothing_dropped(
            artifact_name=f"{artifact_name} {slice_label}",
            compilation=slice_compilation,
        )
        if first_compilation is None:
            first_compilation = dict(slice_compilation)
        compilation_digests.append(_canonical_sha256(dict(slice_compilation)))
        slice_sizes.append(len(positions))
        if _spec_keys(slice_registry.specs) != expected_keys:
            raise ValueError(
                f"{artifact_name} {slice_label} compiled a different target "
                "contract than the chunk."
            )
        if slice_target_frame.n("household") != len(positions):
            raise RuntimeError(
                f"{artifact_name} {slice_label} returned "
                f"{slice_target_frame.n('household')} households for {len(positions)} "
                "positions."
            )
        slice_contract = scored_column_contract(
            slice_target_frame,
            chunk_specs,
            artifact_name=f"{artifact_name} {slice_label}",
        )
        result = score_targets(
            slice_target_frame,
            slice_registry.to_target_set(),
            target_loss_weights=chunk_loss_weights,
            target_loss_cap=release.US_FISCAL_TARGET_LOSS_CAP,
            options={
                "mass": "existing_weights",
                "target_loss_weighting": release.US_FISCAL_TARGET_LOSS_WEIGHTING,
                "maximum_microsim_batch_size": maximum_microsim_batch_size,
            },
        )
        _assert_full_chunk_surface(
            artifact_name=artifact_name,
            chunk_label=slice_label,
            chunk_specs=chunk_specs,
            materialized_registry=slice_registry,
            result=result,
        )
        expected_slice_weights = np.asarray(
            slice_frame.weights_for("household").values,
            dtype=np.float64,
        )
        if not np.array_equal(
            expected_slice_weights,
            np.asarray(result.weights, dtype=np.float64),
        ):
            raise RuntimeError(
                f"{artifact_name} {slice_label} scorer changed the shipped "
                "household weight vector."
            )
        estimates = np.asarray(
            [row.final_estimate for row in result.diagnostics],
            dtype=np.float64,
        )
        targets = np.asarray(
            [row.target for row in result.diagnostics],
            dtype=np.float64,
        )
        scales = np.asarray(result.target_loss_scales, dtype=np.float64)
        names = tuple(row.name for row in result.diagnostics)
        if accumulated_estimates is None:
            accumulated_estimates = estimates.copy()
            reference_targets = targets.copy()
            reference_scales = scales.copy()
            reference_names = names
            reference_contract = slice_contract
        else:
            if not np.array_equal(reference_targets, targets):
                raise RuntimeError(
                    f"{artifact_name} {slice_label} target vector differs from "
                    "the first household slice."
                )
            if not np.array_equal(reference_scales, scales):
                raise RuntimeError(
                    f"{artifact_name} {slice_label} loss-scale vector differs "
                    "from the first household slice."
                )
            if reference_names != names:
                raise RuntimeError(
                    f"{artifact_name} {slice_label} diagnostic names differ "
                    "from the first household slice."
                )
            if reference_contract != slice_contract:
                raise RuntimeError(
                    f"{artifact_name} {slice_label} scored-column contract "
                    "differs from the first household slice."
                )
            np.add(accumulated_estimates, estimates, out=accumulated_estimates)
        del slice_target_frame, slice_registry, result
        if not full_slice:
            del slice_frame
        gc.collect()
        _assert_rss_below_limit(f"after scoring {artifact_name} {slice_label}")
    if any(
        value is None
        for value in (
            accumulated_estimates,
            reference_targets,
            reference_scales,
            reference_names,
            reference_contract,
            first_compilation,
        )
    ):  # pragma: no cover - non-empty batches are enforced above
        raise RuntimeError(f"{artifact_name} {chunk_label} produced no slice result.")
    compilation = {
        **first_compilation,
        "household_slices": len(slice_batches),
        "household_slice_size": maximum_microsim_batch_size,
        "household_slice_row_counts": slice_sizes,
        "slice_compilation_sha256s": compilation_digests,
    }
    return ScoredChunk(
        estimates=accumulated_estimates,
        targets=reference_targets,
        scales=reference_scales,
        diagnostic_names=reference_names,
        scored_contract=reference_contract,
        compilation=compilation,
    )


def _spec_keys(specs: Sequence) -> tuple[tuple[str, object], ...]:
    return tuple(spec.key for spec in specs)


def scored_column_contract(
    target_frame: Frame,
    specs: Sequence,
    *,
    artifact_name: str,
) -> tuple[tuple[str, str, str], ...]:
    """Return and validate every entity/column/role used by the given specs."""

    contract: set[tuple[str, str, str]] = set()
    missing: list[str] = []
    for spec in specs:
        table = target_frame.table(spec.entity)
        for role, column in (("measure", spec.measure), ("filter", spec.filter)):
            if column is None:
                continue
            key = (spec.entity, str(column), role)
            contract.add(key)
            if column not in table.columns:
                missing.append(
                    f"{spec.name}@{spec.period}: {spec.entity}.{column} ({role})"
                )
    if missing:
        raise ValueError(
            f"{artifact_name} lacks scored column(s): " + "; ".join(missing[:20])
        )
    return tuple(sorted(contract))


def _assert_nothing_dropped(
    *,
    artifact_name: str,
    compilation: Mapping[str, object],
) -> None:
    dropped = tuple(compilation.get("dropped_target_names") or ())
    if dropped:
        raise ValueError(
            f"{artifact_name} did not materialize the full fiscal registry; "
            f"dropped {len(dropped)} target(s): {list(dropped[:20])}."
        )


def _assert_full_chunk_surface(
    *,
    artifact_name: str,
    chunk_label: str,
    chunk_specs: Sequence,
    materialized_registry: TargetRegistry,
    result,
) -> None:
    expected = _spec_keys(chunk_specs)
    materialized = _spec_keys(materialized_registry.specs)
    if materialized != expected:
        raise ValueError(
            f"{artifact_name} {chunk_label} materialized target contract "
            f"differs from the yardstick: expected {len(expected)} ordered "
            f"keys, got {len(materialized)}."
        )
    if result.skipped:
        examples = [
            f"{item.target.row_name}: {item.reason}" for item in result.skipped[:20]
        ]
        raise ValueError(
            f"{artifact_name} {chunk_label} skipped {len(result.skipped)} "
            f"target(s) in the constraint matrix: {examples}."
        )
    problem_keys = tuple(target.key for target in result.problem.targets)
    if problem_keys != expected:
        raise ValueError(
            f"{artifact_name} {chunk_label} constraint-matrix target contract "
            "differs from the yardstick."
        )
    expected_names = tuple(spec.to_target().row_name for spec in chunk_specs)
    diagnostic_names = tuple(row.name for row in result.diagnostics)
    if diagnostic_names != expected_names:
        raise ValueError(
            f"{artifact_name} {chunk_label} diagnostic row contract differs "
            "from the yardstick."
        )


def _fiscal_rows_and_aggregate(
    *,
    yardstick: FiscalYardstick,
    estimates: np.ndarray,
    targets: np.ndarray,
    scales: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Build per-target rows and THE canonical aggregate over the full surface.

    Per-target quantities reproduce the production attribution row formulas
    (_target_loss_attribution.py:143-176) with whole-registry weight
    normalization; the aggregate is one relative_error_loss evaluation — the
    same call a one-shot score_targets would make on identical vectors.
    """

    specs = yardstick.registry.specs
    weights = np.asarray(yardstick.loss_weights, dtype=np.float64)
    cap = float(release.US_FISCAL_TARGET_LOSS_CAP)
    if not (
        len(specs)
        == estimates.shape[0]
        == targets.shape[0]
        == scales.shape[0]
        == weights.shape[0]
    ):
        raise RuntimeError("Combined scoring vectors do not align with the registry.")
    aggregate_loss = relative_error_loss(
        estimates,
        targets,
        target_loss_weights=weights,
        target_loss_scales=scales,
        target_loss_cap=cap,
    )
    total_weight = float(weights.sum())
    within = 0
    rows: list[dict[str, object]] = []
    for index, spec in enumerate(specs):
        target = float(targets[index])
        estimate = float(estimates[index])
        relative_error = (
            (estimate - target) / target if target != 0.0 else estimate - target
        )
        if abs(relative_error) <= 0.10:
            within += 1
        scale = float(scales[index])
        weight = float(weights[index])
        weight_share = weight / total_weight
        capped_error = min(abs(estimate - target) / scale, cap)
        rows.append(
            {
                "name": spec.name,
                "period": spec.period,
                "entity": spec.entity,
                "family": spec.family,
                "value_basis": release._fiscal_target_value_basis(spec),
                "target": target,
                "actual": estimate,
                "relative_error": float(relative_error),
                "absolute_relative_error": abs(float(relative_error)),
                "target_loss_weight": weight,
                "target_loss_weight_share": weight_share,
                "target_loss_scale": scale,
                "capped_scaled_absolute_error": capped_error,
                "weighted_loss_contribution": weight_share * capped_error,
            }
        )
    contribution_sum = math.fsum(
        float(row["weighted_loss_contribution"]) for row in rows
    )
    if not math.isclose(
        contribution_sum,
        float(aggregate_loss),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Fiscal row contributions do not reproduce the aggregate loss."
        )
    return rows, {
        "weighted_loss": float(aggregate_loss),
        "fraction_within_10pct": within / len(specs),
    }


def _battery_label(target: tuple[str, str, str, int]) -> str:
    entity, family, column, clone_index = target
    return f"{entity}/{family}/{column}[clone_{clone_index}]"


def _joint_battery_label(
    target: tuple[str, str, tuple[str, ...], int],
) -> str:
    entity, family, columns, clone_index = target
    return f"{entity}/{family}/joint[{','.join(columns)}][clone_{clone_index}]"


def _metric_legs(metric: str) -> tuple[str, ...]:
    if metric == "boolean_incidence":
        return ("incidence_ratio_acs_over_asec",)
    if metric == "monetary_sign_separated":
        return (
            "positive_incidence_ratio_acs_over_asec",
            "positive_quantile_envelope_distance",
            "negative_incidence_ratio_acs_over_asec",
            "negative_quantile_envelope_distance",
        )
    if metric == "categorical_tvd":
        return ("total_variation_distance",)
    raise ValueError(f"Unknown canonical terminal-battery metric {metric!r}.")


def _canonical_battery_contract() -> dict[str, dict[str, object]]:
    contract: dict[str, dict[str, object]] = {}
    for target, metric in sorted(CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY.items()):
        contract[_battery_label(target)] = {
            "metric": metric,
            "metric_legs": list(_metric_legs(metric)),
            "joint": False,
        }
    for target, metric in sorted(
        CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY.items()
    ):
        contract[_joint_battery_label(target)] = {
            "metric": metric,
            "metric_legs": list(_metric_legs(metric)),
            "joint": True,
        }
    return dict(sorted(contract.items()))


def _battery_entities() -> tuple[str, ...]:
    return tuple(
        sorted(
            {key[0] for key in CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY}
            | {key[0] for key in CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY}
        )
    )


def _observed_origin_receipt(frame: Frame) -> dict[str, object]:
    """Count battery-origin rows in the canonical clone-0/positive-weight scope."""

    receipt: dict[str, object] = {}
    entities_missing_columns: list[str] = []
    total_acs_rows = 0
    total_asec_rows = 0
    for entity in _battery_entities():
        table = frame.table(entity)
        channel_column = support_channel_column(entity)
        clone_column = support_clone_index_column(entity)
        if channel_column not in table.columns or clone_column not in table.columns:
            entities_missing_columns.append(entity)
            receipt[entity] = {"provenance_columns_present": False}
            continue
        raw_counts = {
            str(channel): int(count)
            for channel, count in table[channel_column]
            .astype(str)
            .value_counts()
            .items()
        }
        clone_index = pd.to_numeric(table[clone_column], errors="raise").astype("int64")
        weights = np.asarray(frame.resolve_weights(entity).values, dtype=np.float64)
        scope = clone_index.eq(0).to_numpy() & (weights > 0.0)
        scoped_counts = {
            str(channel): int(count)
            for channel, count in table.loc[scope, channel_column]
            .astype(str)
            .value_counts()
            .items()
        }
        asec_rows = scoped_counts.get(BASE_ASEC_SUPPORT_CHANNEL, 0)
        acs_rows = scoped_counts.get(ACS_STACKED_SUPPORT_CHANNEL, 0)
        receipt[entity] = {
            "provenance_columns_present": True,
            "scope": {"clone_index": 0, "positive_weight_only": True},
            "origin_row_counts": dict(sorted(scoped_counts.items())),
            "raw_origin_row_counts": dict(sorted(raw_counts.items())),
            "asec_rows": asec_rows,
            "acs_rows": acs_rows,
        }
        total_asec_rows += asec_rows
        total_acs_rows += acs_rows
    return {
        "entities": receipt,
        "entities_missing_provenance_columns": entities_missing_columns,
        "total_asec_rows": total_asec_rows,
        "total_acs_rows": total_acs_rows,
    }


def _unavailable_scalar_legs(
    contract_row: Mapping[str, object],
    *,
    status: str,
) -> list[dict[str, object]]:
    return [
        {"name": name, "status": status, "value": None}
        for name in contract_row["metric_legs"]
    ]


def _normalized_scalar_legs(
    *,
    label: str,
    contract_row: Mapping[str, object],
    receipt: Mapping[str, object],
) -> list[dict[str, object]]:
    """Validate and flatten every nominal terminal-battery scalar leg."""

    metric = str(contract_row["metric"])
    status = str(receipt.get("status") or "missing_status")
    if status != "tested":
        return _unavailable_scalar_legs(contract_row, status=status)
    if metric == "boolean_incidence":
        key = "incidence_ratio_acs_over_asec"
        if key not in receipt:
            raise ValueError(
                f"Battery comparison {label!r} omits computed leg {key!r}."
            )
        return [{"name": key, "status": "computed", "value": receipt[key]}]
    if metric == "categorical_tvd":
        key = "total_variation_distance"
        if key not in receipt:
            raise ValueError(
                f"Battery comparison {label!r} omits computed leg {key!r}."
            )
        return [{"name": key, "status": "computed", "value": receipt[key]}]
    if metric != "monetary_sign_separated":  # pragma: no cover - contract guards
        raise ValueError(f"Unknown canonical terminal-battery metric {metric!r}.")
    sign_receipts = receipt.get("legs")
    if not isinstance(sign_receipts, Mapping) or set(sign_receipts) != {
        "positive",
        "negative",
    }:
        raise ValueError(
            f"Battery comparison {label!r} must carry exact positive/negative "
            "monetary leg receipts."
        )
    scalar_legs: list[dict[str, object]] = []
    for sign in ("positive", "negative"):
        sign_receipt = sign_receipts[sign]
        if not isinstance(sign_receipt, Mapping):
            raise ValueError(f"Battery comparison {label!r}/{sign} is not an object.")
        incidence_name = f"{sign}_incidence_ratio_acs_over_asec"
        quantile_name = f"{sign}_quantile_envelope_distance"
        sign_status = str(sign_receipt.get("status") or "computed")
        if sign_status == "absent_on_both_origins":
            scalar_legs.extend(
                (
                    {"name": incidence_name, "status": sign_status, "value": None},
                    {"name": quantile_name, "status": sign_status, "value": None},
                )
            )
            continue
        incidence_key = "incidence_ratio_acs_over_asec"
        if incidence_key not in sign_receipt:
            raise ValueError(
                f"Battery comparison {label!r}/{sign} omits computed incidence leg."
            )
        scalar_legs.append(
            {
                "name": incidence_name,
                "status": "computed",
                "value": sign_receipt[incidence_key],
            }
        )
        if "quantile_envelope_distance" in sign_receipt:
            scalar_legs.append(
                {
                    "name": quantile_name,
                    "status": "computed",
                    "value": sign_receipt["quantile_envelope_distance"],
                }
            )
        elif sign_receipt.get("quantile_envelope") == "leg_insufficient_support":
            scalar_legs.append(
                {
                    "name": quantile_name,
                    "status": "insufficient_support",
                    "value": None,
                }
            )
        else:
            raise ValueError(
                f"Battery comparison {label!r}/{sign} omits its quantile leg "
                "value or explicit insufficient-support status."
            )
    expected = list(contract_row["metric_legs"])
    if [row["name"] for row in scalar_legs] != expected:
        raise RuntimeError(f"Battery comparison {label!r} scalar-leg order drifted.")
    return scalar_legs


def _normalize_battery_comparisons(
    comparisons: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    contract = _canonical_battery_contract()
    missing = sorted(set(contract) - set(comparisons))
    extra = sorted(set(comparisons) - set(contract))
    if missing or extra:
        raise ValueError(
            "Terminal battery does not exactly cover the canonical contract; "
            f"missing={missing[:20]}, extra={extra[:20]}."
        )
    normalized: dict[str, dict[str, object]] = {}
    for label, contract_row in contract.items():
        receipt = comparisons[label]
        if not isinstance(receipt, Mapping):
            raise ValueError(f"Battery comparison {label!r} is not an object.")
        if receipt.get("metric") != contract_row["metric"]:
            raise ValueError(
                f"Battery comparison {label!r} metric {receipt.get('metric')!r} "
                f"does not match canonical {contract_row['metric']!r}."
            )
        normalized[label] = {
            **contract_row,
            **dict(receipt),
            "scalar_legs": _normalized_scalar_legs(
                label=label,
                contract_row=contract_row,
                receipt=receipt,
            ),
        }
    return normalized


def _scalar_leg_status_counts(
    comparisons: Mapping[str, Mapping[str, object]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for comparison in comparisons.values():
        for leg in comparison["scalar_legs"]:
            status = str(leg["status"])
            counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _inapplicable_battery_payload(
    *,
    reason: str,
    observed: Mapping[str, object],
) -> dict[str, object]:
    contract = _canonical_battery_contract()
    comparisons = _normalize_battery_comparisons(
        {
            label: {
                **row,
                "status": "inapplicable",
                "reason": reason,
            }
            for label, row in contract.items()
        }
    )
    return {
        "status": "inapplicable",
        "by_origin_only": True,
        "reason": reason,
        "observed_origins": dict(observed),
        "comparison_count": len(contract),
        "metric_leg_count": sum(len(row["metric_legs"]) for row in contract.values()),
        "scalar_leg_status_counts": _scalar_leg_status_counts(comparisons),
        "comparisons": comparisons,
    }


def _battery_payload_from_observed_origins(frame: Frame) -> dict[str, object]:
    observed = _observed_origin_receipt(frame)
    if observed["entities_missing_provenance_columns"]:
        reason = (
            "artifact has no support-channel/clone-role provenance columns on "
            f"{observed['entities_missing_provenance_columns']}; the by-origin "
            "battery cannot scope its ASEC/ACS masks"
        )
        return _inapplicable_battery_payload(reason=reason, observed=observed)
    if observed["total_asec_rows"] == 0:
        reason = (
            "artifact carries no positive-weight clone-0 ASEC origin rows "
            "(observed channels are listed in observed_origins); the "
            "ASEC-vs-ACS battery has an empty ASEC side and is definitionally "
            "inapplicable"
        )
        return _inapplicable_battery_payload(reason=reason, observed=observed)
    elif observed["total_acs_rows"] == 0:
        reason = (
            "artifact carries no positive-weight clone-0 ACS-stacked origin "
            "rows (observed channels are listed in observed_origins); the "
            "ASEC-vs-ACS battery has an empty ACS side and is definitionally "
            "inapplicable"
        )
        return _inapplicable_battery_payload(reason=reason, observed=observed)

    if "ssi" in frame.table("person"):
        evaluation_frame = frame
        materialization_receipt: Mapping[str, object] = {
            "mode": "artifact_already_carried_ssi"
        }
    else:
        materialized = materialize_multispine_agreement_outputs(frame)
        evaluation_frame = materialized.frame
        materialization_receipt = dict(materialized.receipt)
    gate = by_origin_battery_artifact_evidence(evaluation_frame)
    raw_comparisons = gate.details.get("comparisons")
    if not isinstance(raw_comparisons, Mapping):
        raise ValueError("Computed by-origin battery has no comparisons receipt.")
    comparisons = _normalize_battery_comparisons(raw_comparisons)
    return {
        "status": "computed_finished_h5",
        "by_origin_only": True,
        "production_receipt_authenticated": False,
        "gate_passed": gate.passed,
        "failures": list(gate.failures),
        "observed_origins": observed,
        "ephemeral_gate_view_materialization": dict(materialization_receipt),
        "comparison_count": len(comparisons),
        "metric_leg_count": sum(
            len(row["scalar_legs"]) for row in comparisons.values()
        ),
        "scalar_leg_status_counts": _scalar_leg_status_counts(comparisons),
        "comparisons": comparisons,
    }


def _battery_payload_from_pool_receipt(
    terminal_gates: Mapping[str, object],
) -> dict[str, object]:
    gates = terminal_gates.get("gates")
    if not isinstance(gates, Mapping):
        raise ValueError("Authenticated pool terminal_gates has no gates object.")
    battery = gates.get("us_by_origin_battery")
    if not isinstance(battery, Mapping):
        raise ValueError(
            "Authenticated pool terminal receipt has no us_by_origin_battery gate."
        )
    details = battery.get("details")
    comparisons = details.get("comparisons") if isinstance(details, Mapping) else None
    if not isinstance(comparisons, Mapping):
        raise ValueError(
            "Authenticated pool by-origin battery has no comparisons receipt."
        )
    normalized = _normalize_battery_comparisons(comparisons)
    return {
        "status": "authenticated_pool_receipt",
        "by_origin_only": True,
        "production_receipt_authenticated": True,
        "gate_passed": battery.get("passed"),
        "failures": list(battery.get("failures") or ()),
        "comparison_count": len(normalized),
        "metric_leg_count": sum(len(row["scalar_legs"]) for row in normalized.values()),
        "scalar_leg_status_counts": _scalar_leg_status_counts(normalized),
        "comparisons": normalized,
    }


def _terminal_battery_payload(artifact: LoadedArtifact) -> dict[str, object]:
    if artifact.terminal_gates is not None:
        return _battery_payload_from_pool_receipt(artifact.terminal_gates)
    return _battery_payload_from_observed_origins(artifact.frame)


def score_loaded_artifact(
    *,
    artifact: LoadedArtifact,
    artifact_name: str,
    yardstick: FiscalYardstick,
    maximum_microsim_batch_size: int | None,
) -> tuple[dict[str, object], tuple[tuple[str, str, str], ...]]:
    """Run the common scoring path for one already-normalized artifact."""

    cd_provenance = _validate_cd_provenance(artifact, yardstick)
    terminal_battery = _terminal_battery_payload(artifact)
    base_frame, mass_repair = release._with_base_population_mass_repair(artifact.frame)
    population_gate = release._base_population_scale_gate(
        base_frame,
        mass_repair=mass_repair,
    )
    health_gate = release._health_input_signal_gate(base_frame)
    specs = yardstick.registry.specs
    chunk_size = MATERIALIZE_SCORE_CHUNK_SPECS
    chunk_count = max(1, math.ceil(len(specs) / chunk_size))
    contract_parts: set[tuple[str, str, str]] = set()
    estimate_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    scale_parts: list[np.ndarray] = []
    diagnostic_names: list[str] = []
    chunk_receipts: list[dict[str, object]] = []
    base_weights = np.asarray(
        base_frame.weights_for("household").values,
        dtype=np.float64,
    )
    household_count = int(base_weights.shape[0])
    nonzero_count = int(np.count_nonzero(base_weights))
    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_size
        chunk_specs = specs[start : start + chunk_size]
        chunk_label = f"chunk {chunk_index + 1}/{chunk_count}"
        scored_chunk = _score_chunk_household_sliced(
            base_frame,
            chunk_specs,
            chunk_loss_weights=np.asarray(
                yardstick.loss_weights[start : start + len(chunk_specs)],
                dtype=np.float64,
            ),
            artifact_name=artifact_name,
            chunk_label=chunk_label,
            maximum_microsim_batch_size=maximum_microsim_batch_size,
        )
        _assert_nothing_dropped(
            artifact_name=f"{artifact_name} {chunk_label}",
            compilation=scored_chunk.compilation,
        )
        contract_parts.update(scored_chunk.scored_contract)
        estimate_parts.append(scored_chunk.estimates)
        target_parts.append(scored_chunk.targets)
        scale_parts.append(scored_chunk.scales)
        diagnostic_names.extend(scored_chunk.diagnostic_names)
        chunk_receipts.append(
            {
                "chunk_index": chunk_index,
                "spec_range": [start, start + len(chunk_specs)],
                "target_compilation": dict(scored_chunk.compilation),
            }
        )
        del scored_chunk
        gc.collect()
        _assert_rss_below_limit(f"after scoring {artifact_name} {chunk_label}")
    expected_names = [spec.to_target().row_name for spec in specs]
    if diagnostic_names != expected_names:
        raise RuntimeError(
            f"{artifact_name} combined diagnostic rows do not reproduce the "
            "full ordered registry surface."
        )
    contract = tuple(sorted(contract_parts))
    rows, aggregate = _fiscal_rows_and_aggregate(
        yardstick=yardstick,
        estimates=np.concatenate(estimate_parts),
        targets=np.concatenate(target_parts),
        scales=np.concatenate(scale_parts),
    )
    payload = {
        "identity": dict(artifact.identity),
        "loader": dict(artifact.loader),
        "scored_column_contract": {
            "sha256": _canonical_sha256(contract),
            "column_count": len(contract),
            "columns": [list(item) for item in contract],
        },
        "fiscal": {
            "weighted_loss": aggregate["weighted_loss"],
            "fraction_within_10pct": aggregate["fraction_within_10pct"],
            "household_count": household_count,
            "nonzero_household_weight_count": nonzero_count,
            "target_count": len(rows),
            "targets": rows,
        },
        "terminal_battery": terminal_battery,
        "normalization_receipts": {
            "historical_formula_owned_columns": dict(
                artifact.historical_formula_owned_columns
            ),
            "congressional_district_provenance": cd_provenance,
            "base_population_mass_repair": dict(mass_repair),
            "base_population_scale_gate": {
                "passed": population_gate.passed,
                "failures": list(population_gate.failures),
                "details": dict(population_gate.details),
            },
            "health_input_signal_gate": {
                "passed": health_gate.passed,
                "failures": list(health_gate.failures),
                "details": dict(health_gate.details),
            },
            "materialize_score_chunking": {
                "chunk_size_specs": chunk_size,
                "chunk_count": chunk_count,
                "household_count": household_count,
                "household_slice_size": maximum_microsim_batch_size,
                "target_column_payload_upper_bound_bytes": (
                    _streaming_target_column_payload_upper_bound_bytes(
                        total_households=household_count,
                        household_slice_size=maximum_microsim_batch_size,
                        chunk_spec_count=min(chunk_size, len(specs)),
                    )
                ),
                "reason": (
                    "the full-frame microsimulation and whole-registry target "
                    "columns can exceed the scoring budget; fixed household "
                    "slices are materialized and scored before being freed, "
                    "so no dense full-pool household-by-measure table exists"
                ),
                "recombination": (
                    "fixed-order slice estimates are additive matrix/weight "
                    "products; target, scale, name, and column contracts match "
                    "on every slice; the aggregate is one relative_error_loss "
                    "call over the full combined vectors"
                ),
                "code_citation": _CODE_CITATIONS["chunked_materialize_score"],
                "chunks": chunk_receipts,
            },
        },
    }
    return payload, contract


def _assert_identical_scored_contracts(
    contracts: Mapping[str, tuple[tuple[str, str, str], ...]],
) -> None:
    items = list(contracts.items())
    if len(items) < 2:
        return
    reference_name, reference = items[0]
    for artifact_name, contract in items[1:]:
        if contract != reference:
            missing = sorted(set(reference) - set(contract))
            extra = sorted(set(contract) - set(reference))
            raise ValueError(
                f"{artifact_name} scored-column contract differs from "
                f"{reference_name}; missing={missing[:20]}, extra={extra[:20]}."
            )


def _battery_status_counts(battery: Mapping[str, object]) -> dict[str, int]:
    comparisons = battery.get("comparisons")
    if not isinstance(comparisons, Mapping):
        return {}
    counts: dict[str, int] = {}
    for row in comparisons.values():
        status = (
            str(row.get("status") or "unknown")
            if isinstance(row, Mapping)
            else ("unknown")
        )
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _comparison_payload(
    incumbent: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    incumbent_fiscal = incumbent["fiscal"]
    candidate_fiscal = candidate["fiscal"]
    if not isinstance(incumbent_fiscal, Mapping) or not isinstance(
        candidate_fiscal, Mapping
    ):
        raise TypeError("Artifact fiscal payloads must be mappings.")
    incumbent_rows = incumbent_fiscal["targets"]
    candidate_rows = candidate_fiscal["targets"]
    if not isinstance(incumbent_rows, list) or not isinstance(candidate_rows, list):
        raise TypeError("Artifact target payloads must be lists.")
    if len(incumbent_rows) != len(candidate_rows):
        raise ValueError("Fiscal target payload lengths differ after contract checks.")
    target_comparisons: list[dict[str, object]] = []
    candidate_lower = 0
    equal = 0
    incumbent_lower = 0
    for incumbent_row, candidate_row in zip(
        incumbent_rows,
        candidate_rows,
        strict=True,
    ):
        key = (incumbent_row["name"], incumbent_row["period"])
        candidate_key = (candidate_row["name"], candidate_row["period"])
        if candidate_key != key:
            raise ValueError(
                f"Target comparison contracts differ: {key!r} vs {candidate_key!r}."
            )
        incumbent_error = float(incumbent_row["absolute_relative_error"])
        candidate_error = float(candidate_row["absolute_relative_error"])
        if candidate_error < incumbent_error:
            candidate_lower += 1
            lower = "candidate"
        elif candidate_error > incumbent_error:
            incumbent_lower += 1
            lower = "incumbent"
        else:
            equal += 1
            lower = "equal"
        target_comparisons.append(
            {
                "name": key[0],
                "period": key[1],
                "incumbent_absolute_relative_error": incumbent_error,
                "candidate_absolute_relative_error": candidate_error,
                "candidate_minus_incumbent_absolute_relative_error": (
                    candidate_error - incumbent_error
                ),
                "lower_absolute_relative_error": lower,
            }
        )
    incumbent_loss = float(incumbent_fiscal["weighted_loss"])
    candidate_loss = float(candidate_fiscal["weighted_loss"])
    incumbent_battery = incumbent["terminal_battery"]
    candidate_battery = candidate["terminal_battery"]
    if not isinstance(incumbent_battery, Mapping) or not isinstance(
        candidate_battery, Mapping
    ):
        raise TypeError("Artifact terminal_battery payloads must be mappings.")
    computable_statuses = {
        "authenticated_pool_receipt",
        "computed_finished_h5",
    }
    both_computable = (
        incumbent_battery.get("status") in computable_statuses
        and candidate_battery.get("status") in computable_statuses
    )
    return {
        "decision": "owner_decides_flip",
        "no_threshold_applied": True,
        "fiscal_weighted_loss": {
            "incumbent": incumbent_loss,
            "candidate": candidate_loss,
            "candidate_minus_incumbent": candidate_loss - incumbent_loss,
        },
        "per_target_absolute_relative_error": {
            "candidate_lower_count": candidate_lower,
            "equal_count": equal,
            "incumbent_lower_count": incumbent_lower,
            "targets": target_comparisons,
        },
        "terminal_battery": {
            "head_to_head_comparable": both_computable,
            "incumbent_status": incumbent_battery.get("status"),
            "candidate_status": candidate_battery.get("status"),
            "incumbent_status_counts": _battery_status_counts(incumbent_battery),
            "candidate_status_counts": _battery_status_counts(candidate_battery),
            "note": (
                "each side's battery evidence is reported on its own terms; "
                "no value is synthesized for a side whose battery is "
                "inapplicable"
            ),
        },
    }


def score_head_to_head(
    *,
    incumbent: Path,
    candidate: Path | None,
    ledger_facts: Path,
    age_targets: bool = False,
    allow_unaged_dollar_targets: bool = True,
    congressional_district_vintage_crosswalk: Path | None = None,
    maximum_microsim_batch_size: int | None = (
        release.DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE
    ),
    candidate_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Compile once, then score incumbent and optional candidate sequentially."""

    crosswalk = congressional_district_vintage_crosswalk or (
        release.default_congressional_district_vintage_crosswalk_path()
    )
    yardstick = compile_yardstick(
        ledger_facts=ledger_facts,
        age_targets=age_targets,
        allow_unaged_dollar_targets=allow_unaged_dollar_targets,
        congressional_district_vintage_crosswalk=crosswalk,
    )
    artifact_paths = [("incumbent", incumbent, None)]
    if candidate is not None:
        artifact_paths.append(("candidate", candidate, candidate_manifest_sha256))
    artifacts: dict[str, dict[str, object]] = {}
    contracts: dict[str, tuple[tuple[str, str, str], ...]] = {}
    for name, path, expected_manifest_sha256 in artifact_paths:
        loaded = load_artifact(
            path,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        artifact_payload, contract = score_loaded_artifact(
            artifact=loaded,
            artifact_name=name,
            yardstick=yardstick,
            maximum_microsim_batch_size=maximum_microsim_batch_size,
        )
        artifacts[name] = artifact_payload
        contracts[name] = contract
        del loaded
        gc.collect()
        _assert_rss_below_limit(f"after releasing {name} scoring state")
    _assert_identical_scored_contracts(contracts)
    candidate_payload = artifacts.get("candidate")
    comparison = (
        _comparison_payload(artifacts["incumbent"], candidate_payload)
        if candidate_payload is not None
        else None
    )
    battery_contract = _canonical_battery_contract()
    return {
        "schema_version": SCHEMA_VERSION,
        "yardstick": {
            "fiscal_registry": dict(yardstick.identity),
            "fiscal_aggregate": {
                "name": release.US_FISCAL_TARGET_LOSS_WEIGHTING,
                "target_loss_cap": release.US_FISCAL_TARGET_LOSS_CAP,
                "loss_basis": dict(yardstick.loss_basis),
                "weighting_rule": (
                    "raw weight = sqrt(max(abs(target value), 1)); "
                    "mean-normalized within amount/count value basis; each "
                    "semantic concept group's total weight is scaled to its "
                    "largest row weight; the amount and count bases are then "
                    "scaled to equal total budget; the vector is "
                    "mean-normalized; the aggregate is the weighted mean of "
                    "per-target |actual - target| / max(|target|, 1), each "
                    "capped at 100%"
                ),
                "family_multipliers": None,
                "code_citations": [
                    _CODE_CITATIONS["loss_weighting"],
                    _CODE_CITATIONS["loss_aggregate"],
                    _CODE_CITATIONS["loss_attribution"],
                ],
            },
            "relative_error": {
                "rule": (
                    "(actual - target) / target; when target is zero, the raw "
                    "actual-minus-target delta"
                ),
                "code_citation": _CODE_CITATIONS["relative_error"],
            },
            "terminal_battery": {
                "by_origin_only": True,
                "single_column_comparison_count": len(
                    CANONICAL_ORIGIN_BATTERY_METRIC_REGISTRY
                ),
                "joint_comparison_count": len(
                    CANONICAL_ORIGIN_BATTERY_JOINT_METRIC_REGISTRY
                ),
                "comparison_count": len(battery_contract),
                "metric_leg_count": sum(
                    len(row["metric_legs"]) for row in battery_contract.values()
                ),
                "code_citations": [
                    _CODE_CITATIONS["battery_registry"],
                    _CODE_CITATIONS["battery_origin_masks"],
                    _CODE_CITATIONS["battery_receipt_keys"],
                    _CODE_CITATIONS["battery_channel_constants"],
                    _CODE_CITATIONS["pool_battery_persistence"],
                ],
            },
            "code_citations": dict(_CODE_CITATIONS),
        },
        "artifacts": {
            "incumbent": artifacts["incumbent"],
            "candidate": candidate_payload,
        },
        "comparison": comparison,
    }


def _markdown_number(value: object) -> str:
    if value is None:
        return "—"
    number = float(value)
    if number == 0:
        return "0"
    if abs(number) >= 1_000_000 or abs(number) < 0.0001:
        return f"{number:.8e}"
    return f"{number:.8f}".rstrip("0").rstrip(".")


def _markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _family_basis_rollup(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["family"]), str(row["value_basis"]))
        group = groups.setdefault(
            key,
            {
                "family": key[0],
                "value_basis": key[1],
                "target_count": 0,
                "weight_share": 0.0,
                "loss_contribution": 0.0,
                "worst_target": None,
                "worst_contribution": -1.0,
            },
        )
        group["target_count"] = int(group["target_count"]) + 1
        group["weight_share"] = float(group["weight_share"]) + float(
            row["target_loss_weight_share"]
        )
        contribution = float(row["weighted_loss_contribution"])
        group["loss_contribution"] = float(group["loss_contribution"]) + contribution
        if contribution > float(group["worst_contribution"]):
            group["worst_contribution"] = contribution
            group["worst_target"] = f"{row['name']}@{row['period']}"
    rollup = list(groups.values())
    for group in rollup:
        share = float(group["weight_share"])
        group["mean_capped_scaled_error"] = (
            float(group["loss_contribution"]) / share if share > 0 else 0.0
        )
        del group["worst_contribution"]
    rollup.sort(key=lambda group: -float(group["loss_contribution"]))
    return rollup


def _battery_markdown_section(
    role: str,
    battery: Mapping[str, object],
) -> list[str]:
    lines = [f"### {role}", ""]
    status = str(battery.get("status") or "unknown")
    counts = _battery_status_counts(battery)
    counts_text = ", ".join(f"{name}: {count}" for name, count in counts.items())
    leg_counts = battery.get("scalar_leg_status_counts")
    leg_counts_text = (
        ", ".join(f"{name}: {count}" for name, count in sorted(leg_counts.items()))
        if isinstance(leg_counts, Mapping)
        else "unavailable"
    )
    if status == "inapplicable":
        lines.append(
            f"All {battery['comparison_count']} comparisons inapplicable — "
            f"{battery['reason']}."
        )
        observed = battery.get("observed_origins")
        if isinstance(observed, Mapping):
            entities = observed.get("entities")
            if isinstance(entities, Mapping):
                for entity, receipt in sorted(entities.items()):
                    if not isinstance(receipt, Mapping):
                        continue
                    if not receipt.get("provenance_columns_present"):
                        lines.append(
                            f"- `{entity}`: no support-channel/clone-role columns."
                        )
                        continue
                    origin_counts = receipt.get("origin_row_counts")
                    lines.append(
                        f"- `{entity}` origin rows: "
                        + ", ".join(
                            f"{channel}={count:,}"
                            for channel, count in sorted((origin_counts or {}).items())
                        )
                    )
    elif status == "authenticated_pool_receipt":
        lines.append(
            f"Authenticated pool receipt; gate passed: "
            f"`{battery.get('gate_passed')}`. Comparison statuses: {counts_text}. "
            f"Scalar-leg statuses: {leg_counts_text}."
        )
    elif status == "computed_finished_h5":
        lines.append(
            "Computed from the finished H5's shipped weights on an ephemeral "
            f"terminal gate view; gate passed: `{battery.get('gate_passed')}`. "
            f"Comparison statuses: {counts_text}. Scalar-leg statuses: "
            f"{leg_counts_text}. This is computed evidence, not an authenticated "
            "pool-build receipt."
        )
    else:
        lines.append(f"Unknown battery evidence status `{_markdown_escape(status)}`.")
    if status != "inapplicable":
        failures = battery.get("failures")
        if isinstance(failures, list) and failures:
            lines.append("")
            lines.append(f"{len(failures)} sealed failure line(s):")
            lines.extend(f"- {_markdown_escape(line)}" for line in failures[:40])
            if len(failures) > 40:
                lines.append(f"- … {len(failures) - 40} more in the JSON twin.")
        comparisons = battery.get("comparisons")
        if isinstance(comparisons, Mapping):
            lines.extend(
                [
                    "",
                    "| comparison | scalar leg | status | value |",
                    "|---|---|---|---:|",
                ]
            )
            for label, comparison in sorted(comparisons.items()):
                if not isinstance(comparison, Mapping):
                    continue
                scalar_legs = comparison.get("scalar_legs")
                if not isinstance(scalar_legs, list):
                    continue
                for leg in scalar_legs:
                    if not isinstance(leg, Mapping):
                        continue
                    value = leg.get("value")
                    rendered_value = (
                        _markdown_number(value)
                        if isinstance(value, (int, float, np.integer, np.floating))
                        and not isinstance(value, bool)
                        else _markdown_escape("—" if value is None else value)
                    )
                    lines.append(
                        f"| {_markdown_escape(label)} | "
                        f"{_markdown_escape(leg.get('name'))} | "
                        f"{_markdown_escape(leg.get('status'))} | "
                        f"{rendered_value} |"
                    )
    lines.append("")
    return lines


def render_markdown(payload: Mapping[str, object]) -> str:
    """Render a deterministic head-to-head Markdown scorecard.

    The complete per-target table (every registry row) lives in the JSON twin;
    this scorecard renders the aggregate, family rollups, and the worst rows.
    """

    yardstick = payload["yardstick"]
    artifacts = payload["artifacts"]
    if not isinstance(yardstick, Mapping) or not isinstance(artifacts, Mapping):
        raise TypeError("Scorecard payload has invalid yardstick/artifacts objects.")
    incumbent = artifacts["incumbent"]
    candidate = artifacts.get("candidate")
    if not isinstance(incumbent, Mapping):
        raise TypeError("Scorecard incumbent must be a mapping.")
    roles: list[tuple[str, Mapping[str, object]]] = [("incumbent", incumbent)]
    if isinstance(candidate, Mapping):
        roles.append(("candidate", candidate))
    registry = yardstick["fiscal_registry"]
    aggregate = yardstick["fiscal_aggregate"]
    battery_meta = yardstick["terminal_battery"]
    lines = [
        "# US release replacement scorecard",
        "",
        "This is evidence for the owner's flip decision, not an automatic "
        "publication verdict. No gate, threshold, tolerance, or band is applied "
        "by this scorecard. The complete per-target table is in the JSON twin "
        "of this file.",
        "",
        "## Frozen yardstick",
        "",
        f"- Fiscal registry: `{registry['version']}` with "
        f"{registry['target_count']:,} targets from Ledger facts "
        f"`{registry['ledger_facts']['sha256']}`.",
        f"- Weighted aggregate: `{aggregate['name']}` with cap "
        f"`{aggregate['target_loss_cap']}` and no family multipliers.",
        f"- Weighting rule: {aggregate['weighting_rule']}.",
        f"- Terminal battery: {battery_meta['single_column_comparison_count']} "
        f"single-column plus {battery_meta['joint_comparison_count']} joint "
        "comparison(s); all are by-origin-only (ASEC vs ACS within one "
        "artifact).",
        "",
        "## Artifact summary",
        "",
        "| role | artifact SHA-256 | loader | households | nonzero weights | "
        "weighted loss | within 10% | terminal battery |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for role, artifact in roles:
        fiscal = artifact["fiscal"]
        battery = artifact["terminal_battery"]
        lines.append(
            f"| {role} | `{artifact['identity']['sha256']}` | "
            f"{_markdown_escape(artifact['loader']['kind'])} | "
            f"{fiscal['household_count']:,} | "
            f"{fiscal['nonzero_household_weight_count']:,} | "
            f"{_markdown_number(fiscal['weighted_loss'])} | "
            f"{_markdown_number(fiscal['fraction_within_10pct'])} | "
            f"{_markdown_escape(battery['status'])} |"
        )
    identity = incumbent["identity"]
    if isinstance(identity, Mapping) and "policyengine_package_resolved" in identity:
        resolved = identity["policyengine_package_resolved"]
        lines.extend(
            [
                "",
                "The incumbent SHA-256 matches the artifact policyengine.py "
                f"`{resolved['policyengine_version']}` resolves as the US "
                f"default dataset: `{resolved['repo_id']}` revision "
                f"`{resolved['revision']}`, file `{resolved['filename']}` "
                f"(build `{resolved['build_id']}`).",
            ]
        )
    lines.extend(
        [
            "",
            "## Historical formula-owned column normalization",
            "",
            "Loaded artifact columns owned by formulas in the current locked "
            "PolicyEngine-US are dropped only after their required input "
            "leaves are verified present, then recomputed by the shared "
            "comparison engine.",
        ]
    )
    for role, artifact in roles:
        normalization = artifact.get("normalization_receipts")
        receipt = (
            normalization.get("historical_formula_owned_columns")
            if isinstance(normalization, Mapping)
            else None
        )
        if not isinstance(receipt, Mapping):
            raise TypeError(
                f"Scorecard {role} has no historical formula-owned column receipt."
            )
        columns_by_entity = receipt.get("columns_by_entity")
        if not isinstance(columns_by_entity, Mapping):
            raise TypeError(
                f"Scorecard {role} formula-owned column receipt has no entity map."
            )
        count = int(receipt.get("count", -1))
        lines.extend(["", f"### {role}", "", f"Dropped column count: **{count}**."])
        if columns_by_entity:
            for entity, columns in sorted(columns_by_entity.items()):
                if not isinstance(columns, list):
                    raise TypeError(
                        f"Scorecard {role} receipt columns for {entity} must be a list."
                    )
                rendered = ", ".join(f"`{column}`" for column in columns)
                lines.append(f"- `{entity}`: {rendered}")
        else:
            lines.append("No formula-owned columns were present.")
    comparison = payload.get("comparison")
    if isinstance(comparison, Mapping):
        loss = comparison["fiscal_weighted_loss"]
        counts = comparison["per_target_absolute_relative_error"]
        lines.extend(
            [
                "",
                "## Raw head-to-head comparison",
                "",
                f"Candidate minus incumbent weighted loss: "
                f"**{_markdown_number(loss['candidate_minus_incumbent'])}** "
                f"(incumbent {_markdown_number(loss['incumbent'])}, candidate "
                f"{_markdown_number(loss['candidate'])}). A negative value is "
                "a lower candidate loss; the owner decides whether the full "
                "evidence warrants the flip.",
                "",
                f"Per-target absolute relative error is lower for the candidate "
                f"on {counts['candidate_lower_count']:,} targets, equal on "
                f"{counts['equal_count']:,}, and lower for the incumbent on "
                f"{counts['incumbent_lower_count']:,}. No materiality threshold "
                "is imposed.",
            ]
        )
    for role, artifact in roles:
        fiscal = artifact["fiscal"]
        rows = fiscal["targets"]
        if not isinstance(rows, list):
            raise TypeError("Artifact fiscal targets must be a list.")
        rollup = _family_basis_rollup(rows)
        lines.extend(
            [
                "",
                f"## {role}: loss by family and value basis",
                "",
                "| family | basis | targets | weight share | loss contribution | "
                "weighted mean capped error | worst target |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for group in rollup:
            lines.append(
                f"| {_markdown_escape(group['family'])} | "
                f"{_markdown_escape(group['value_basis'])} | "
                f"{group['target_count']:,} | "
                f"{_markdown_number(group['weight_share'])} | "
                f"{_markdown_number(group['loss_contribution'])} | "
                f"{_markdown_number(group['mean_capped_scaled_error'])} | "
                f"{_markdown_escape(group['worst_target'])} |"
            )
        worst = sorted(
            rows,
            key=lambda row: -float(row["weighted_loss_contribution"]),
        )[:MARKDOWN_WORST_TARGET_ROWS]
        lines.extend(
            [
                "",
                f"## {role}: worst {len(worst)} targets by loss contribution",
                "",
                "| target | period | family | target value | actual | "
                "relative error | loss contribution |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in worst:
            lines.append(
                f"| {_markdown_escape(row['name'])} | "
                f"{_markdown_escape(row['period'])} | "
                f"{_markdown_escape(row['family'])} | "
                f"{_markdown_number(row['target'])} | "
                f"{_markdown_number(row['actual'])} | "
                f"{_markdown_number(row['relative_error'])} | "
                f"{_markdown_number(row['weighted_loss_contribution'])} |"
            )
    lines.extend(
        [
            "",
            "## Terminal by-origin battery",
            "",
        ]
    )
    for role, artifact in roles:
        battery = artifact["terminal_battery"]
        if not isinstance(battery, Mapping):
            raise TypeError("Artifact terminal_battery must be a mapping.")
        lines.extend(_battery_markdown_section(role, battery))
    lines.extend(
        [
            "## Mechanism citations",
            "",
        ]
    )
    for name, citation in sorted(yardstick["code_citations"].items()):
        lines.append(f"- `{name}`: `{citation}`")
    return "\n".join(lines) + "\n"


def write_scorecard(
    payload: Mapping[str, object],
    out_prefix: Path,
) -> tuple[Path, Path]:
    """Write deterministic JSON and Markdown bytes for one payload."""

    prefix = out_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    json_text = (
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    markdown_text = render_markdown(payload)
    json_path.write_text(json_text)
    markdown_path.write_text(markdown_text)
    return json_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = score_head_to_head(
        incumbent=args.incumbent,
        candidate=args.candidate,
        ledger_facts=args.ledger_facts,
        age_targets=args.age_targets,
        allow_unaged_dollar_targets=args.allow_unaged_dollar_targets,
        congressional_district_vintage_crosswalk=(
            args.congressional_district_vintage_crosswalk
        ),
        maximum_microsim_batch_size=args.maximum_microsim_batch_size,
        candidate_manifest_sha256=args.candidate_manifest_sha256,
    )
    json_path, markdown_path = write_scorecard(payload, args.out_prefix)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(markdown_path),
                "incumbent_sha256": payload["artifacts"]["incumbent"]["identity"][
                    "sha256"
                ],
                "candidate_scored": payload["artifacts"]["candidate"] is not None,
                "peak_rss_gib": round(_peak_rss_bytes() / 1024**3, 3),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
