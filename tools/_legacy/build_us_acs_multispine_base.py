"""Build a nullable ASEC-by-PUF plus ACS multispine staging artifact.

This tool starts from the dense, input-complete ASEC-by-PUF H5 produced by
the US input-family pipeline. It acquires the byte-pinned ACS PUMS archives,
loads and maps the ACS spine, transfers model input leaves from the dense
donor, assigns the PUMA-anchored state/CD/county geography ladder, audits every
fit's resolved typed weight kind, and writes the combined base. Calibration is
deliberately downstream.

``--location-rule block_v1`` (microcosm#696; default ``legacy``) replaces the
PUMA-ladder assignment with one 2020 census block per household from
``--block-ladder``: ACS rows draw within their observed PUMA, donor rows keep a
block their base already carries (else draw within their state), and every
geography derives from the block. The block-location gate is then a hard
failure, and the rule, seed and both ladders' sha256 are recorded in the
summary (``household_location``, ``orchestration``, ``geography_ladder``) and
the staging H5's root attributes. The legacy default is unchanged.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build import (
    FitWeightRecord,
    GateResult,
    default_valued_columns_gate,
    weights_audit_gate,
)
from microcosm.build.us_runtime import acs_sources
from microcosm.build.us_runtime.acs_local_hours import (
    ACS_UNDER15_ZERO_POLICY,
    acs_local_hours_signal_gate,
    acs_local_transfer_target_families,
    prepare_acs_local_hours_donor,
)
from microcosm.build.us_runtime.acs_multispine import (
    AcsMultispineResult,
    build_optional_acs_multispine,
)
from microcosm.build.us_runtime.acs_pums import (
    ACS_2024_1YR_SPINE,
    DEFAULT_CHUNKSIZE,
    AcsPumsSource,
)
from microcosm.build.us_runtime.acs_transfer import (
    ACS_DONOR_CHANNEL_AUTO,
    DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    TargetFamilies,
    acs_derived_transfer_expectations,
    acs_transfer_donor_requirements,
    declared_acs_transfer_target_families,
    default_acs_transfer_target_families,
    resolve_acs_donor_channel,
)
from microcosm.build.us_runtime.base_pool import (
    ACS_POOL_LOCATION_CLONES_REFUSAL,
    DONOR_BLOCK_PRESERVED,
    DONOR_BLOCK_STATE_DRAWN,
    spine_column,
)
from microcosm.build.us_runtime.block_location import (
    POPULACE_BLOCK_LADDER_SHA256_ATTR,
    POPULACE_BLOCK_LADDER_VINTAGES_ATTR,
    POPULACE_LOCATION_CLONES_ATTR,
    POPULACE_LOCATION_RULE_ATTR,
    POPULACE_LOCATION_SEED_ATTR,
    US_LOCATION_RULE_BLOCK_V1,
    US_LOCATION_RULE_CHOICES,
    US_LOCATION_RULE_LEGACY,
    UsLocationLadder,
    load_us_location_ladder,
    location_geography_columns,
    us_block_location_gate,
)
from microcosm.build.us_runtime.h5_io import (
    assert_h5_unchanged,
    refuse_denied_frame,
    refuse_denied_pool_h5,
)
from microcosm.build.us_runtime.puma_ladder import (
    PUMA_LADDER_ARTIFACT_SHA256_ATTR,
    PUMA_LADDER_VINTAGES_ATTR,
    UsPumaLadder,
    load_us_puma_ladder,
)
from microcosm.frame import (
    Frame,
    WeightKind,
    Weights,
    put_frame_table,
    read_frame_table,
)
from microcosm.frame.units import US_SCHEMA

PERIOD = 2024
# This implementation moved from tools/ into tools/_legacy/ unchanged.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS_DIR = _REPOSITORY_ROOT / "inputs" / "acs_2024_1yr"
DEFAULT_PUMA_LADDER = _REPOSITORY_ROOT / "build" / "us" / "us_puma_ladder_2020.npz"
DEFAULT_N_ESTIMATORS = 32
DEFAULT_STAGING_EXPORT_PEAK_LIMIT_BYTES = int(
    os.environ.get("POPULACE_STAGING_EXPORT_PEAK_LIMIT_BYTES", 30_000_000_000)
)
_STAGING_EXPORT_FIXED_OVERHEAD_BYTES = 512 * 1024**2
_PACKAGED_MANIFEST_REFERENCE = (
    "package:microcosm.build.us_runtime/acs_2024_1yr_sources.json"
)
#: H5 root attributes that record a build's household location rule. The
#: base-H5 line writes the first three under ``block_v1``, and the stacked pool
#: writes all five; this line writes them (plus the PUMA ladder's
#: ``PUMA_LADDER_ARTIFACT_SHA256_ATTR`` / ``PUMA_LADDER_VINTAGES_ATTR``) on its
#: staging H5 only under ``block_v1`` (a legacy staging H5 is unchanged).
LOCATION_RULE_ATTR = POPULACE_LOCATION_RULE_ATTR
LOCATION_SEED_ATTR = POPULACE_LOCATION_SEED_ATTR
LOCATION_CLONES_ATTR = POPULACE_LOCATION_CLONES_ATTR
LOCATION_BLOCK_LADDER_SHA256_ATTR = POPULACE_BLOCK_LADDER_SHA256_ATTR
LOCATION_BLOCK_LADDER_VINTAGES_ATTR = POPULACE_BLOCK_LADDER_VINTAGES_ATTR


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append the pinned ACS 2024 1-year PUMS spine to an already-built "
            "dense ASEC-by-PUF base. The nullable result has reviewed source-"
            "universe limitations and is simulation-ready except for the "
            "downstream calibration solve."
        )
    )
    parser.add_argument("--base-h5", required=True, type=Path)
    parser.add_argument("--out-h5", required=True, type=Path)
    parser.add_argument(
        "--summary",
        type=Path,
        help=(
            "JSON build summary path. Defaults to OUT-H5 with a .summary.json suffix."
        ),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Strict ACS source manifest override; defaults to the packaged pin.",
    )
    parser.add_argument(
        "--inputs-dir",
        default=DEFAULT_INPUTS_DIR,
        type=Path,
        help="Hash-verified ACS archive cache (default: inputs/acs_2024_1yr).",
    )
    parser.add_argument(
        "--puma-ladder",
        default=DEFAULT_PUMA_LADDER,
        type=Path,
        help=(
            "Validated national PUMA-ladder NPZ (default: "
            "build/us/us_puma_ladder_2020.npz)."
        ),
    )
    parser.add_argument(
        "--max-households",
        type=_positive_int,
        help="Optional deterministic smoke limit applied after ACS household sort.",
    )
    parser.add_argument("--period", default=PERIOD, type=_positive_int)
    parser.add_argument("--chunksize", default=DEFAULT_CHUNKSIZE, type=_positive_int)
    parser.add_argument("--acs-share", default=0.5, type=_open_unit_interval)
    parser.add_argument("--seed", default=0, type=_nonnegative_int)
    parser.add_argument(
        "--hours-under15-policy",
        choices=[ACS_UNDER15_ZERO_POLICY],
        default=None,
        help="Explicit modeled hours completion for unresolved ACS children; disabled by default.",
    )
    parser.add_argument(
        "--geography-seed",
        default=0,
        type=_nonnegative_int,
        help="Deterministic PUMA/CD/county assignment seed (default: 0).",
    )
    parser.add_argument(
        "--location-rule",
        choices=US_LOCATION_RULE_CHOICES,
        default=US_LOCATION_RULE_LEGACY,
        help=(
            "Household location rule (default: legacy, the PUMA-ladder draw "
            "seeded by --geography-seed). block_v1 (microcosm#696) gives every "
            "household one 2020 census block drawn by population within its "
            "finest source geography (ACS: observed PUMA; an ASEC-by-PUF donor "
            "keeps a block its base already carries, else draws within its "
            "state) and derives every other geography from the block. "
            "Requires --block-ladder."
        ),
    )
    parser.add_argument(
        "--location-seed",
        default=0,
        type=_nonnegative_int,
        help="block_v1 location seed (default: 0; legacy refuses non-zero).",
    )
    parser.add_argument(
        "--location-clones",
        default=1,
        type=_positive_int,
        help=(
            "block_v1 location clones per household (default: 1; values above "
            "1 are refused on this line for now, see the error for why)."
        ),
    )
    parser.add_argument(
        "--block-ladder",
        type=Path,
        help=(
            "Block-ladder NPZ carrying the per-block 2020 PUMA (required by, "
            "and only accepted with, --location-rule block_v1). The PUMA "
            "ladder is still loaded for its coverage summary and downstream "
            "population targets."
        ),
    )
    parser.add_argument(
        "--n-estimators",
        default=DEFAULT_N_ESTIMATORS,
        type=_positive_int,
    )
    parser.add_argument(
        "--max-targets-per-fit",
        default=DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
        type=_positive_int,
        help="Maximum chained QRF targets retained at once (default: 8).",
    )
    parser.add_argument(
        "--donor-channel",
        default=ACS_DONOR_CHANNEL_AUTO,
        help=(
            "ASEC-by-PUF support channel used by transfer. The default selects "
            "the appropriate support channel for each input family."
        ),
    )
    parser.add_argument(
        "--donor-release-manifest",
        type=Path,
        help=(
            "release_manifest.json of the published release BASE-H5 came "
            "from. When given, the donor H5's sha256 must match the "
            "manifest's root microdata artifact, and the donor release "
            "identity is pinned into the staging summary."
        ),
    )
    return parser.parse_args(argv)


def _donor_release_identity(
    manifest_path: Path | None,
    base_sha256: str,
) -> dict[str, object] | None:
    """Verify the donor H5 against its release manifest and pin its identity."""

    if manifest_path is None:
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"Cannot read donor release manifest {manifest_path}: {exc}."
        ) from exc
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SystemExit(
            f"Donor release manifest {manifest_path} carries no artifacts map."
        )
    microdata = [
        (name, entry)
        for name, entry in artifacts.items()
        if isinstance(entry, dict) and entry.get("kind") == "microdata"
    ]
    if len(microdata) != 1:
        raise SystemExit(
            "Donor release manifest must carry exactly one microdata "
            f"artifact; found {sorted(name for name, _ in microdata)}."
        )
    name, entry = microdata[0]
    if entry.get("sha256") != base_sha256:
        raise SystemExit(
            "Donor H5 sha256 does not match its release manifest: base-h5 "
            f"is {base_sha256} but {manifest_path} pins "
            f"{entry.get('sha256')!r} for {name!r}. Refusing to transfer "
            "from an unverified donor."
        )
    build = manifest.get("build") if isinstance(manifest.get("build"), dict) else {}
    return {
        "manifest_path": str(manifest_path.resolve()),
        "artifact": name,
        "release_id": build.get("build_id"),
        "revision": entry.get("revision"),
        "repo_id": entry.get("repo_id"),
        "sha256": entry.get("sha256"),
        "dataset_role": manifest.get("dataset_role"),
        "is_default": manifest.get("is_default"),
    }


def _validate_location_arguments(args: argparse.Namespace) -> None:
    """Refuse location flags the selected rule would silently ignore."""

    if args.location_rule == US_LOCATION_RULE_LEGACY:
        if args.location_seed != 0 or args.location_clones != 1:
            raise SystemExit(
                "--location-seed and --location-clones apply only to "
                "--location-rule block_v1; the legacy rule is seeded by "
                f"--geography-seed (got --location-seed {args.location_seed}, "
                f"--location-clones {args.location_clones})."
            )
        if args.block_ladder is not None:
            raise SystemExit(
                "--block-ladder is only used by --location-rule block_v1; the "
                "legacy rule assigns geography from --puma-ladder."
            )
        return
    if args.block_ladder is None:
        raise SystemExit("--location-rule block_v1 requires --block-ladder.")
    if args.location_clones > 1:
        raise SystemExit(ACS_POOL_LOCATION_CLONES_REFUSAL)
    if args.geography_seed != 0:
        raise SystemExit(
            "--geography-seed seeds only the legacy PUMA-ladder draw; under "
            "--location-rule block_v1 use --location-seed (refusing "
            f"--geography-seed {args.geography_seed} rather than ignoring it)."
        )


def _base_location_rule(path: Path) -> dict[str, object]:
    """The donor base's recorded location rule, from its H5 root attributes.

    A base H5 without :data:`LOCATION_RULE_ATTR` predates the attribute, so
    its rule is recorded as ``legacy``.
    """

    with pd.HDFStore(path, mode="r") as store:
        attributes = store.get_node("/")._v_attrs
        names = set(attributes._v_attrnames)
        recorded = {
            key: _attribute_text(attributes[key])
            for key in (LOCATION_RULE_ATTR, LOCATION_SEED_ATTR, LOCATION_CLONES_ATTR)
            if key in names
        }
    return {
        "location_rule": recorded.get(LOCATION_RULE_ATTR, US_LOCATION_RULE_LEGACY),
        "recorded": LOCATION_RULE_ATTR in recorded,
        "attributes": recorded,
    }


def _attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _attribute_text(value.item())
    return str(value)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary_path = args.summary or args.out_h5.with_suffix(".summary.json")
    _validate_artifact_paths(args, summary_path=summary_path)
    _validate_location_arguments(args)
    block_location = args.location_rule != US_LOCATION_RULE_LEGACY
    block_ladder: UsLocationLadder | None = None
    base_location: dict[str, object] | None = None
    if block_location:
        block_ladder = load_us_location_ladder(args.block_ladder)
        if block_ladder.puma is None:
            raise SystemExit(
                f"--block-ladder {args.block_ladder} carries no per-block 'puma' "
                "array; block_v1 ACS rows draw within their observed PUMA."
            )
        base_location = _base_location_rule(args.base_h5)

    manifest = acs_sources.load_acs_source_manifest(args.source_manifest)
    manifest_file = _manifest_file(args.source_manifest)
    manifest_sha256 = _sha256(manifest_file)
    base_sha256 = _sha256(args.base_h5)
    donor_release = _donor_release_identity(args.donor_release_manifest, base_sha256)
    base = _load_base_frame(args.base_h5)
    _require_benefit_participation_inputs(base)
    transfer_plan = declared_acs_transfer_target_families()
    _require_dense_donor_coverage(
        base,
        donor_channel=args.donor_channel,
        target_families=transfer_plan,
    )
    base_rows = _row_counts(base)
    base_mass = float(base.weights_for("household").total)
    puma_ladder_sha256 = _sha256(args.puma_ladder)
    puma_ladder = load_us_puma_ladder(args.puma_ladder)
    ladder_agreement: dict[str, object] | None = None
    if block_location:
        assert block_ladder is not None
        ladder_agreement = _block_puma_ladder_agreement(block_ladder, puma_ladder)

    source = acs_sources.fetch_acs_pums_sources(
        args.inputs_dir,
        manifest=manifest,
    )
    if args.max_households is not None:
        source = replace(source, max_households=args.max_households)

    # The legacy call is unchanged; block_v1 adds its options only when set.
    location_options: dict[str, object] = {}
    if block_location:
        location_options = {
            "location_rule": args.location_rule,
            "block_ladder": block_ladder,
            "location_seed": args.location_seed,
            "location_clones": args.location_clones,
        }
    result = build_optional_acs_multispine(
        base,
        source,
        chunksize=args.chunksize,
        acs_share=args.acs_share,
        target_families=transfer_plan,
        hours_donor_factory=lambda donor: prepare_acs_local_hours_donor(
            donor, seed=args.seed, period=args.period
        ),
        hours_under15_policy=args.hours_under15_policy,
        donor_channel=args.donor_channel,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_targets_per_fit=args.max_targets_per_fit,
        puma_ladder=puma_ladder,
        geography_seed=args.geography_seed,
        **location_options,
    )
    _require_puma_ladder_assignment(result)
    location_gate = None
    if block_location:
        assert block_ladder is not None
        location_gate = _require_block_location(result, block_ladder)
    _require_benefit_participation_transfer(result)
    transfer_coverage = _require_default_transfer_coverage(
        result,
        base,
        target_families=acs_local_transfer_target_families(),
    )
    weights_audit = _audit_fits(result)

    # The pooled frame owns its assembled blocks. Release the dense donor
    # before HDF serialization so export cannot retain both full spines plus
    # writer scratch at once.
    del base
    gc.collect()
    input_null_audit = _engine_input_null_audit(result.frame)
    gc.collect()
    hours_gate = acs_local_hours_signal_gate(
        result.frame, source_null_audit=input_null_audit
    )
    if not hours_gate.passed:
        raise SystemExit(
            "Local staging hours gate failed: " + "; ".join(hours_gate.failures)
        )

    args.out_h5.parent.mkdir(parents=True, exist_ok=True)
    staging_export_peak_bytes = _preflight_staging_export(result.frame)
    if block_location:
        assert block_ladder is not None
        _write_dataset(
            result.frame,
            args.out_h5,
            period=args.period,
            root_attributes=location_root_attributes(
                location_rule=args.location_rule,
                location_seed=args.location_seed,
                location_clones=args.location_clones,
                block_ladder_sha256=str(block_ladder.sha256),
                block_ladder_vintages=block_ladder.layer_vintages,
                puma_ladder_sha256=puma_ladder_sha256,
                puma_ladder_vintages=puma_ladder.layer_vintages,
            ),
        )
    else:
        _write_dataset(result.frame, args.out_h5, period=args.period)

    summary = _build_summary(
        args=args,
        summary_path=summary_path,
        manifest=manifest,
        source=source,
        base_rows=base_rows,
        base_mass=base_mass,
        base_sha256=base_sha256,
        donor_release=donor_release,
        manifest_sha256=manifest_sha256,
        result=result,
        weights_audit=weights_audit,
        transfer_coverage=transfer_coverage,
        input_null_audit=input_null_audit,
        staging_export_peak_bytes=staging_export_peak_bytes,
        puma_ladder=puma_ladder,
        puma_ladder_sha256=puma_ladder_sha256,
    )
    if block_location:
        assert block_ladder is not None and location_gate is not None
        summary["household_location"] = _household_location_summary(
            args,
            result,
            block_ladder=block_ladder,
            puma_ladder_sha256=puma_ladder_sha256,
            base_location=base_location,
            ladder_agreement=ladder_agreement,
            gate=location_gate,
        )
    summary["local_hours_source"] = result.provenance.get("local_hours_source")
    summary["local_hours_gate"] = {
        "name": hours_gate.name,
        "passed": hours_gate.passed,
        "failures": list(hours_gate.failures),
        "details": dict(hours_gate.details),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    summary_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


def _audit_fits(result: AcsMultispineResult) -> dict[str, object]:
    records = result.fit_records
    if not records:
        raise SystemExit(
            "ACS input transfer produced no fit records. The dense donor must "
            "carry transferable tax-detail and benefit-participation inputs."
        )
    non_typed = [
        type(record).__name__
        for record in records
        if not isinstance(record, FitWeightRecord)
    ]
    if non_typed:
        raise TypeError(
            "ACS input transfer returned non-FitWeightRecord audit evidence: "
            f"{non_typed}."
        )
    gate = weights_audit_gate(records)
    if not gate.passed:
        raise SystemExit("Weights audit failed:\n  " + "\n  ".join(gate.failures))
    return {
        "passed": gate.passed,
        "failures": list(gate.failures),
        "details": dict(gate.details),
    }


def _require_benefit_participation_inputs(base: Frame) -> None:
    participation = sorted(
        column
        for entity in base.entities
        for column in base.table(entity).columns
        if column.startswith("takes_up_")
    )
    if not participation:
        raise SystemExit(
            "Dense ASEC-by-PUF donor has no takes_up_* benefit-participation "
            "inputs. This tool must run after the benefit input-family stages."
        )


def _require_dense_donor_coverage(
    base: Frame,
    engine: Any | None = None,
    *,
    donor_channel: str | None = ACS_DONOR_CHANNEL_AUTO,
    target_families: TargetFamilies | None = None,
) -> None:
    """Fail unless every column consumed by the QRF plan has donor signal."""

    if engine is None:
        from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

        engine = PolicyEngineUSEngine()
    selected, resolved_channel = _coverage_donor_channel(base, donor_channel)
    plan = target_families or declared_acs_transfer_target_families()
    try:
        requirements = acs_transfer_donor_requirements(selected, plan)
    except ValueError as exc:
        raise SystemExit(
            "Dense ASEC-by-PUF donor has an invalid ACS transfer feature "
            f"surface; selected channel={resolved_channel!r}: {exc}."
        ) from exc

    failures: list[str] = []
    present_values: dict[str, Any] = {}
    owners: dict[str, str] = {}
    for entity, columns in requirements.items():
        if entity not in selected.entities:
            failures.extend(
                f"{entity}.{column}: transfer-consumed column is absent "
                "because the donor entity is missing."
                for column in columns
            )
            continue
        table = selected.table(entity)
        for column in columns:
            if column not in table.columns:
                actual_owner = _column_owner(selected, column)
                location = (
                    f"; found on entity {actual_owner!r}"
                    if actual_owner is not None
                    else ""
                )
                failures.append(
                    f"{entity}.{column}: transfer-consumed column is absent{location}."
                )
                continue
            values = table[column].to_numpy()
            if not pd.notna(values).any():
                failures.append(
                    f"{entity}.{column}: transfer-consumed column has no "
                    "observed donor values."
                )
                continue
            present_values[column] = values
            owners[column] = entity

    defaults = engine.default_values(sorted(present_values))
    default_gate = default_valued_columns_gate(present_values, defaults)
    default_valued = default_gate.details["default_valued_columns"]
    for column, default in sorted(default_valued.items()):
        failures.append(
            f"{owners[column]}.{column}: every observed donor value equals "
            f"the engine default ({default!r}); QRF transfer requires usable "
            "target/predictor signal."
        )

    if failures:
        raise SystemExit(
            "Dense ASEC-by-PUF donor failed the hard ACS transfer-consumption "
            f"gate; selected channel={resolved_channel!r}:\n  " + "\n  ".join(failures)
        )


def _column_owner(frame: Frame, column: str) -> str | None:
    try:
        return frame.column_entity(column)
    except ValueError:
        return None


def _coverage_donor_channel(
    base: Frame,
    requested: str | None,
) -> tuple[Frame, str | None]:
    try:
        return resolve_acs_donor_channel(base, requested)
    except ValueError as exc:
        raise SystemExit(f"Invalid ACS transfer donor support metadata: {exc}") from exc


def _require_benefit_participation_transfer(result: AcsMultispineResult) -> None:
    imputed = result.provenance.get("imputed_inputs", [])
    transferred = isinstance(imputed, list) and any(
        isinstance(item, dict)
        and isinstance(item.get("family"), str)
        and item["family"].split("__batch_", 1)[0] == "benefit_participation"
        and isinstance(item.get("column"), str)
        and item["column"].startswith("takes_up_")
        for item in imputed
    )
    if not transferred:
        raise SystemExit(
            "ACS default transfer produced no takes_up_* benefit-participation "
            "input. Refusing to report a complete multispine base."
        )


def _require_puma_ladder_assignment(result: AcsMultispineResult) -> None:
    """Fail unless the enabled multispine resolved launch geography inputs."""

    raw = result.provenance.get("geography_ladder")
    if not isinstance(raw, dict) or raw.get("applied") is not True:
        raise SystemExit(
            "ACS multispine did not apply the required PUMA geography ladder."
        )
    household = result.frame.table("household")
    required = ("puma", "congressional_district_geoid", "county_fips")
    missing = [column for column in required if column not in household]
    if missing:
        raise SystemExit(
            f"PUMA geography assignment omitted household column(s): {missing}."
        )
    nulls = {
        column: int(household[column].isna().sum())
        for column in required
        if household[column].isna().any()
    }
    if nulls:
        raise SystemExit(
            f"PUMA geography assignment left null household values: {nulls}."
        )
    deferred = result.provenance.get("deferred_inputs", [])
    still_deferred = sorted(
        {"congressional_district_geoid", "county_fips"}.intersection(deferred)
        if isinstance(deferred, list)
        else {"congressional_district_geoid", "county_fips"}
    )
    if still_deferred:
        raise SystemExit(
            "PUMA geography assignment still reports resolved input(s) as "
            f"deferred: {still_deferred}."
        )


def _require_block_location(
    result: AcsMultispineResult, block_ladder: UsLocationLadder
) -> GateResult:
    """Hard-gate a ``block_v1`` pool: every geography is its block's lookup."""

    household = result.frame.table("household")
    columns = location_geography_columns(block_ladder)
    missing = [column for column in columns if column not in household]
    if missing:
        raise SystemExit(f"Block location omitted household column(s): {missing}.")
    nulls = {
        column: int(household[column].isna().sum())
        for column in columns
        if household[column].isna().any()
    }
    if nulls:
        raise SystemExit(f"Block location left null household values: {nulls}.")
    if not isinstance(result.provenance.get("household_location"), dict):
        raise SystemExit(
            "ACS multispine did not record the block_v1 household location."
        )
    gate = us_block_location_gate(household, block_ladder)
    if not gate.passed:
        raise SystemExit("Block location gate failed: " + "; ".join(gate.failures))
    return gate


def _block_puma_ladder_agreement(
    block_ladder: UsLocationLadder, puma_ladder: UsPumaLadder
) -> dict[str, object]:
    """Refuse a block/PUMA ladder pair that is not the same 2020 geography.

    Under ``block_v1`` each household's PUMA and congressional district come
    from the block ladder, while the local release still takes its state and
    CD population targets from the PUMA ladder
    (``tools/build_us_acs_local_release.py:ladder_population``). The pair
    must therefore carry the same PUMAs, the same districts and the same CD
    vintage; that is checked here, before the transfer runs. The largest
    per-state and per-district population differences are recorded, not
    refused.
    """

    assert block_ladder.puma is not None  # checked when the ladder is loaded
    plan = block_ladder.primary_congressional_district_plan
    puma_cd_vintage = puma_ladder.layer_vintages["congressional_district"]
    if plan != puma_cd_vintage:
        raise SystemExit(
            f"--block-ladder assigns {plan!r} congressional districts but "
            f"--puma-ladder's population targets use {puma_cd_vintage!r}."
        )
    population = np.asarray(block_ladder.population, dtype=np.float64)
    block_state = block_ladder.block_geoid // 10**13
    block_cd = np.asarray(
        block_ladder.congressional_district_plans[plan], dtype=np.int64
    )
    puma_state = np.asarray(puma_ladder.puma, dtype=np.int64) // 100_000
    comparisons = {
        "pumas": (
            pd.Series(population).groupby(block_ladder.puma).sum(),
            pd.Series(np.asarray(puma_ladder.puma_population, dtype=np.float64))
            .groupby(np.asarray(puma_ladder.puma, dtype=np.int64))
            .sum(),
        ),
        "states": (
            pd.Series(population).groupby(block_state).sum(),
            pd.Series(np.asarray(puma_ladder.puma_population, dtype=np.float64))
            .groupby(puma_state)
            .sum(),
        ),
        "congressional_districts": (
            pd.Series(population).groupby(block_cd).sum(),
            pd.Series(np.asarray(puma_ladder.cd_overlap_population, dtype=np.float64))
            .groupby(np.asarray(puma_ladder.cd_overlap_cd, dtype=np.int64))
            .sum(),
        ),
    }
    record: dict[str, object] = {"congressional_district_plan": plan}
    mismatched: list[str] = []
    for name, (block_totals, puma_totals) in comparisons.items():
        only_block = sorted(set(block_totals.index) - set(puma_totals.index))
        only_puma = sorted(set(puma_totals.index) - set(block_totals.index))
        shared = block_totals.index.intersection(puma_totals.index)
        difference = (block_totals[shared] - puma_totals[shared]).abs()
        record[name] = {
            "block_ladder": int(len(block_totals)),
            "puma_ladder": int(len(puma_totals)),
            "only_in_block_ladder": [int(value) for value in only_block[:10]],
            "only_in_puma_ladder": [int(value) for value in only_puma[:10]],
            "max_abs_population_difference": (
                float(difference.max()) if len(difference) else 0.0
            ),
        }
        if only_block or only_puma:
            mismatched.append(
                f"{name}: {len(only_block)} only in --block-ladder "
                f"(e.g. {only_block[:3]}), {len(only_puma)} only in "
                f"--puma-ladder (e.g. {only_puma[:3]})"
            )
    if mismatched:
        raise SystemExit(
            "--block-ladder and --puma-ladder are not the same 2020 geography: "
            + "; ".join(mismatched)
            + "."
        )
    return record


def location_root_attributes(
    *,
    location_rule: str,
    location_seed: int,
    location_clones: int,
    block_ladder_sha256: str,
    block_ladder_vintages: Mapping[str, str],
    puma_ladder_sha256: str,
    puma_ladder_vintages: Mapping[str, str],
) -> dict[str, str]:
    """H5 root attributes recording a ``block_v1`` location step.

    The block ladder located every household; the PUMA ladder is still pinned
    because the local release draws its state and CD population targets from
    it. ``tools/build_us_acs_local_release.py`` writes the same attributes on
    the calibrated artifact from the staging summary.
    """

    return {
        LOCATION_RULE_ATTR: location_rule,
        LOCATION_SEED_ATTR: str(int(location_seed)),
        LOCATION_CLONES_ATTR: str(int(location_clones)),
        LOCATION_BLOCK_LADDER_SHA256_ATTR: str(block_ladder_sha256),
        LOCATION_BLOCK_LADDER_VINTAGES_ATTR: json.dumps(
            dict(block_ladder_vintages), sort_keys=True
        ),
        PUMA_LADDER_ARTIFACT_SHA256_ATTR: str(puma_ladder_sha256),
        PUMA_LADDER_VINTAGES_ATTR: json.dumps(
            dict(puma_ladder_vintages), sort_keys=True
        ),
    }


def _household_location_summary(
    args: argparse.Namespace,
    result: AcsMultispineResult,
    *,
    block_ladder: UsLocationLadder,
    puma_ladder_sha256: str,
    base_location: dict[str, object] | None,
    ladder_agreement: dict[str, object] | None,
    gate: GateResult,
) -> dict[str, object]:
    """The staging summary's ``household_location`` block (``block_v1`` only)."""

    record = result.provenance["household_location"]
    assert isinstance(record, dict)
    return {
        **record,
        "block_ladder_path": str(args.block_ladder.resolve()),
        "puma_ladder": {
            "path": str(args.puma_ladder.resolve()),
            "sha256": puma_ladder_sha256,
            "role": (
                "coverage summary and downstream state/CD population targets; "
                "not used to assign geography under block_v1"
            ),
        },
        "donor_base_location": base_location,
        "ladder_agreement": ladder_agreement,
        "gate": {
            "name": gate.name,
            "passed": gate.passed,
            "failures": list(gate.failures),
            "details": dict(gate.details),
        },
    }


def _require_default_transfer_coverage(
    result: AcsMultispineResult,
    donor: Frame,
    *,
    target_families: TargetFamilies | None = None,
) -> dict[str, object]:
    """Prove every planned target is present and complete on its ACS universe."""

    expected: dict[str, str] = {}
    plan = target_families or default_acs_transfer_target_families(donor)
    for entity, entity_families in plan.items():
        for targets in entity_families.values():
            for target in targets:
                expected[target] = entity
    # Deterministically derived columns are as load-bearing as fitted ones:
    # a plan carrying the CGD parents owes the derived memo leg too.
    expected.update(acs_derived_transfer_expectations(plan))
    raw_imputed = result.provenance.get("imputed_inputs", [])
    if not isinstance(raw_imputed, list):
        raise SystemExit("ACS imputed-input provenance must be a JSON list.")
    entries = {
        item["column"]: item
        for item in raw_imputed
        if isinstance(item, dict) and isinstance(item.get("column"), str)
    }
    if len(entries) != len(
        [item for item in raw_imputed if isinstance(item, dict) and "column" in item]
    ):
        raise SystemExit("ACS imputed-input provenance contains duplicate columns.")
    # WKHP plus an explicitly recorded completion can legitimately need no
    # fit. Keep this exception narrow: other default targets still require
    # transfer receipts; any remaining hours gaps require a fallback receipt.
    native_hours_column = "weekly_hours_worked_before_lsr"
    native_inputs = result.provenance.get("native_inputs", {})
    native_hours = (
        native_inputs.get(native_hours_column, {})
        if isinstance(native_inputs, dict)
        else {}
    )
    modeled = result.provenance.get("hours_modeled_completion")
    modeled_rows = 0
    if modeled is not None:
        if not (
            isinstance(modeled, dict)
            and modeled.get("policy") == ACS_UNDER15_ZERO_POLICY
            and modeled.get("version") == 1
            and modeled.get("provenance") == "modeled_assumption"
            and modeled.get("column") == native_hours_column
            and type(modeled.get("modeled_rows")) is int
            and modeled["modeled_rows"] >= 0
        ):
            raise SystemExit("Invalid ACS modeled hours completion receipt.")
        modeled_rows = modeled["modeled_rows"]
    native_complete: dict[str, dict] = {}
    if (
        native_hours_column in expected
        and isinstance(native_hours, dict)
        and native_hours.get("entity") == "person"
        and native_hours.get("provenance") == "acs_2024_1yr_native"
        and isinstance(native_hours.get("source_columns"), list)
        and "WKHP" in native_hours["source_columns"]
        and type(native_hours.get("missing_rows")) is int
        and native_hours["missing_rows"] == modeled_rows
    ):
        native_complete[native_hours_column] = native_hours
    missing = sorted(set(expected) - set(entries) - set(native_complete))
    if missing:
        raise SystemExit(
            "ACS default transfer omitted donor-observed model input(s): "
            f"{missing}. Refusing to report a complete multispine base."
        )

    structural_pending: list[dict[str, object]] = []
    for column, entity in sorted(expected.items()):
        table = result.frame.table(entity)
        if column not in table:
            raise SystemExit(
                f"ACS transfer registered {column!r} but the combined {entity!r} "
                "table does not contain it."
            )
        tag = spine_column(entity)
        if tag not in table:
            raise SystemExit(
                f"Combined staging frame lacks ACS spine tag {tag!r} on {entity!r}."
            )
        acs_mask = table[tag].eq(ACS_2024_1YR_SPINE)
        if not acs_mask.any():
            raise SystemExit(f"Combined staging frame has no ACS rows on {entity!r}.")
        missing_mask = table[column].isna() & acs_mask
        missing_rows = int(missing_mask.sum())
        if column in entries:
            raw_unmodeled = entries[column].get("unmodeled_recipient_rows", 0)
        else:
            receipt = native_complete[column]
            if type(receipt.get("observed_rows")) is not int or receipt[
                "observed_rows"
            ] + modeled_rows != int(acs_mask.sum()):
                raise SystemExit("ACS native hours receipt has incorrect row coverage.")
            raw_unmodeled = receipt["missing_rows"] - modeled_rows
        if type(raw_unmodeled) is not int or raw_unmodeled < 0:
            raise SystemExit(
                f"ACS imputation provenance for {column!r} has invalid "
                f"unmodeled_recipient_rows={raw_unmodeled!r}."
            )
        if missing_rows != raw_unmodeled:
            raise SystemExit(
                f"ACS transfer provenance for {column!r} reports "
                f"{raw_unmodeled} unmodeled row(s), but the combined ACS spine "
                f"contains {missing_rows} missing row(s)."
            )
        if missing_rows == 0:
            continue
        if column != "pre_subsidy_rent" or entity != "person":
            raise SystemExit(
                f"ACS default transfer left {missing_rows} unmodeled row(s) in "
                f"required input {column!r}."
            )
        gq_mask = _acs_group_quarters_person_mask(result.frame)
        if not missing_mask.equals(gq_mask):
            raise SystemExit(
                "ACS pre_subsidy_rent may remain absent only for native "
                "group-quarters rows."
            )
        structural_pending.append(
            {
                "column": column,
                "entity": entity,
                "rows": missing_rows,
                "reason": "ACS group-quarters rows are outside the housing-tenure universe",
            }
        )

    coverage = {
        "expected_inputs": sorted(expected),
        "registered_inputs": sorted(entries),
        "structural_pending": structural_pending,
    }
    if native_complete:
        coverage["native_registered_inputs"] = sorted(native_complete)
    if modeled is not None:
        coverage["modeled_hours_rows"] = modeled_rows
    return coverage


def _acs_group_quarters_person_mask(frame: Frame) -> pd.Series:
    household = frame.table("household")
    person = frame.table("person")
    required = {"household_id", "TYPEHUGQ", spine_column("household")}
    missing = sorted(required - set(household.columns))
    if missing:
        raise SystemExit(
            "Cannot validate ACS group-quarters transfer gaps; household table "
            f"lacks {missing}."
        )
    kind = pd.to_numeric(household["TYPEHUGQ"], errors="coerce")
    gq_households = household.loc[
        household[spine_column("household")].eq(ACS_2024_1YR_SPINE) & kind.isin([2, 3]),
        "household_id",
    ]
    return person[spine_column("person")].eq(ACS_2024_1YR_SPINE) & person[
        "person_household_id"
    ].isin(gq_households)


def _reviewed_limitations(
    result: AcsMultispineResult,
    *,
    transfer_coverage: dict[str, object],
    input_null_audit: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Document accepted source-universe and geographic precision limits."""

    geography = result.provenance["geography_ladder"]
    if not isinstance(geography, dict):  # guarded by _require_puma_ladder_assignment
        raise TypeError("geography_ladder provenance must be a mapping.")

    acs_nulls = [
        entry
        for entry in input_null_audit
        if isinstance(entry.get("missing_rows_by_spine"), dict)
        and int(entry["missing_rows_by_spine"].get(ACS_2024_1YR_SPINE, 0)) > 0
    ]
    gq_counts = _acs_group_quarters_counts(result.frame)
    gq_entity_by_column = {
        "pre_subsidy_rent": "person",
        "spm_unit_tenure_type": "spm_unit",
        "tenure_type": "household",
    }

    def is_exact_gq_null(entry: dict[str, object]) -> bool:
        column = entry.get("column")
        entity = gq_entity_by_column.get(column)
        missing_by_spine = entry.get("missing_rows_by_spine")
        return (
            entity is not None
            and entry.get("entity") == entity
            and isinstance(missing_by_spine, dict)
            and int(missing_by_spine.get(ACS_2024_1YR_SPINE, 0)) == gq_counts[entity]
        )

    gq_nulls = [entry for entry in acs_nulls if is_exact_gq_null(entry)]
    other_native_nulls = [entry for entry in acs_nulls if not is_exact_gq_null(entry)]
    structural_pending = transfer_coverage.get("structural_pending", [])
    if not isinstance(structural_pending, list):
        raise TypeError("transfer_coverage.structural_pending must be a list.")

    return [
        {
            "id": "acs_group_quarters_housing_universe",
            "status": "reviewed_structural_absence",
            "affected_spine": ACS_2024_1YR_SPINE,
            "affected_rows": gq_counts,
            "affected_columns": {
                "household": [
                    "tenure_type",
                    "acs_monthly_contract_rent",
                    "acs_monthly_gross_rent",
                    "acs_annual_property_tax",
                ],
                "spm_unit": ["spm_unit_tenure_type"],
                "person": ["pre_subsidy_rent", "real_estate_taxes"],
            },
            "reason": (
                "ACS PUMS housing-unit tenure, rent, and property-tax fields "
                "are outside the TYPEHUGQ 2/3 group-quarters universe."
            ),
            "treatment": (
                "Preserve those values as structural nulls; filling them with "
                "zero or donor housing values would synthesize an unobserved "
                "housing unit."
            ),
            "engine_input_nulls": gq_nulls,
            "transfer_evidence": structural_pending,
            "calibration_blocker": False,
        },
        {
            "id": "native_acs_source_universe_blanks",
            "status": "reviewed_source_missingness",
            "affected_spine": ACS_2024_1YR_SPINE,
            "reason": (
                "Native ACS inputs retain official blank universes; the ACS "
                "mapping contract forbids inventing zeros or component splits."
            ),
            "treatment": (
                "Preserve nullable source semantics after transferring every "
                "donor-observed model-required input with an eligible fit."
            ),
            "engine_input_nulls_excluding_group_quarters_housing": (other_native_nulls),
            "calibration_blocker": False,
        },
        (
            _block_v1_sub_puma_limitation(geography)
            if geography.get("location_rule") == US_LOCATION_RULE_BLOCK_V1
            else {
                "id": "sub_puma_geographic_precision",
                "status": "reviewed_probabilistic_assignment",
                "observed_geography": {
                    "acs_2024_1yr": ["state_fips", "puma"],
                    "asec_puf": (
                        [
                            "state_fips",
                            "tract_geoid",
                            "congressional_district_geoid",
                            "county_fips",
                        ]
                        if geography.get("donor_geography") == "preserved_assigned"
                        else ["state_fips"]
                    ),
                },
                "assigned_geography": [
                    "puma",
                    "congressional_district_geoid",
                    "county_fips",
                ],
                "donor_geography": geography.get("donor_geography"),
                "unavailable_exact_geography": list(
                    geography.get(
                        "unresolved_sub_puma_inputs",
                        ["block_geoid", "tract_geoid"],
                    )
                ),
                "unavailable_exact_geography_scope": (
                    "acs_2024_1yr"
                    if geography.get("donor_geography") == "preserved_assigned"
                    else "acs_2024_1yr,asec_puf"
                ),
                "reason": (
                    "ACS PUMS identifies residence only through state and 2020 "
                    "PUMA; exact block, tract, county, and congressional district "
                    "cannot be recovered for a source microrecord."
                ),
                "treatment": (
                    (
                        "Retain each ACS record's observed PUMA and assign its "
                        "119th-CD/county from official population-weighted PUMA "
                        "overlaps using the recorded seed. Donor records keep "
                        "their certified block-ladder district/county, with PUMA "
                        "derived exactly from the assigned 2020 tract. Do not "
                        "synthesize ACS block or tract."
                    )
                    if geography.get("donor_geography") == "preserved_assigned"
                    else (
                        "Retain each ACS record's observed PUMA, draw ASEC PUMA "
                        "within native state, and assign 119th-CD/county from "
                        "official population-weighted PUMA overlaps using the "
                        "recorded seed. Do not synthesize block or tract."
                    )
                ),
                "assignment_seed": geography.get("seed"),
                "layer_vintages": geography.get("layer_vintages", {}),
                "calibration_blocker": False,
            }
        ),
    ]


def _block_v1_sub_puma_limitation(geography: dict[str, object]) -> dict[str, object]:
    """The ``sub_puma_geographic_precision`` limitation under ``block_v1``.

    Every household carries one 2020 block and every geography derived from
    it; what stays unknowable is which block inside the source geography a
    record's respondent actually lives in.
    """

    donor_geography = geography.get("donor_geography")
    columns = list(geography.get("resolved_model_inputs", []))
    if donor_geography == DONOR_BLOCK_PRESERVED:
        donor_observed: object = ["state_fips", "block_geoid"]
        donor_reason = (
            "An ASEC-by-PUF donor's block is the one its base-H5 build assigned "
            "(donor_base_location in the staging summary's household_location "
            "records that build's rule); this stage keeps it rather than "
            "redrawing it."
        )
    elif donor_geography == DONOR_BLOCK_STATE_DRAWN:
        donor_observed = ["state_fips"]
        donor_reason = (
            "The donor base carries no block_geoid, so each ASEC-by-PUF "
            "donor's finest known geography here is its state; this stage "
            "draws its block within that state."
        )
    else:
        donor_observed = {
            "with_base_block": ["state_fips", "block_geoid"],
            "without_base_block": ["state_fips"],
        }
        donor_reason = (
            "ASEC-by-PUF donors that carry a base-H5 block keep it (never "
            "redrawn); donors without one draw a block within their state "
            "(counts in household_location.donor_blocks)."
        )
    return {
        "id": "sub_puma_geographic_precision",
        "status": "reviewed_probabilistic_assignment",
        "location_rule": US_LOCATION_RULE_BLOCK_V1,
        "assignment_rule": geography.get("assignment_rule"),
        "observed_geography": {
            "acs_2024_1yr": ["state_fips", "puma"],
            "asec_puf": donor_observed,
        },
        "assigned_geography": columns,
        "donor_geography": donor_geography,
        "unavailable_exact_geography": [],
        "unavailable_exact_geography_scope": None,
        "reason": (
            "ACS PUMS identifies residence only through state and 2020 PUMA, "
            "so the block an ACS household occupies cannot be recovered for "
            "a source microrecord; it is drawn. " + donor_reason
        ),
        "treatment": (
            "Give each ACS household one 2020 census block drawn with "
            "probability proportional to 2020 block population within its "
            "observed PUMA, keep each donor's already-assigned block (a donor "
            "without one draws within its state), and derive block, tract, "
            "county, place, SLDU/SLDL, CBSA, PUMA and congressional district "
            "from the block through the block ladder. ACS rows therefore now "
            "carry block/tract/place/SLD/CBSA, and no household's geographies "
            "contradict each other."
        ),
        "assignment_seed": geography.get("seed"),
        "layer_vintages": geography.get("block_layer_vintages", {}),
        "calibration_blocker": False,
    }


def _acs_group_quarters_counts(frame: Frame) -> dict[str, int]:
    household = frame.table("household")
    person = frame.table("person")
    household_tag = spine_column("household")
    required_household = {"household_id", "TYPEHUGQ", household_tag}
    required_person = {
        "person_household_id",
        "person_spm_unit_id",
        spine_column("person"),
    }
    missing = sorted(
        (required_household - set(household.columns))
        | (required_person - set(person.columns))
    )
    if missing:
        raise SystemExit(
            "Cannot summarize reviewed ACS group-quarters limitations; "
            f"combined frame lacks {missing}."
        )
    kind = pd.to_numeric(household["TYPEHUGQ"], errors="coerce")
    household_mask = household[household_tag].eq(ACS_2024_1YR_SPINE) & kind.isin([2, 3])
    household_ids = household.loc[household_mask, "household_id"]
    person_mask = person[spine_column("person")].eq(ACS_2024_1YR_SPINE) & person[
        "person_household_id"
    ].isin(household_ids)
    return {
        "household": int(household_mask.sum()),
        "person": int(person_mask.sum()),
        "spm_unit": int(person.loc[person_mask, "person_spm_unit_id"].nunique()),
    }


def _build_summary(
    *,
    args: argparse.Namespace,
    summary_path: Path,
    manifest: acs_sources.AcsSourceManifest,
    source: AcsPumsSource,
    base_rows: dict[str, int],
    base_mass: float,
    base_sha256: str,
    donor_release: dict[str, object] | None,
    manifest_sha256: str,
    result: AcsMultispineResult,
    weights_audit: dict[str, object],
    transfer_coverage: dict[str, object],
    input_null_audit: list[dict[str, object]],
    staging_export_peak_bytes: int,
    puma_ladder: UsPumaLadder,
    puma_ladder_sha256: str,
) -> dict[str, object]:
    output_rows = _row_counts(result.frame)
    output_mass = float(result.frame.weights_for("household").total)
    block_location = args.location_rule != US_LOCATION_RULE_LEGACY
    geography_ladder: dict[str, object] = {
        "path": str(args.puma_ladder.resolve()),
        "sha256": puma_ladder_sha256,
        "pumas": len(puma_ladder),
        "layer_vintages": puma_ladder.layer_vintages,
        "seed": args.geography_seed,
        "assignment": result.provenance["geography_ladder"],
    }
    orchestration: dict[str, object] = {
        "chunksize": args.chunksize,
        "acs_share": args.acs_share,
        "max_households": args.max_households,
        "seed": args.seed,
        "geography_seed": args.geography_seed,
        "n_estimators": args.n_estimators,
        "max_targets_per_fit": args.max_targets_per_fit,
        "donor_channel": args.donor_channel,
        "provenance": result.provenance,
    }
    if block_location:
        # The PUMA ladder no longer assigns geography; its sha stays pinned
        # (population targets) and the block ladder's is recorded beside it.
        household_location = result.provenance["household_location"]
        assert isinstance(household_location, dict)
        geography_ladder.update(
            {
                "seed": args.location_seed,
                "location_rule": args.location_rule,
                "block_ladder": {
                    "path": str(args.block_ladder.resolve()),
                    **dict(household_location["block_ladder"]),
                },
            }
        )
        orchestration.update(
            {
                "location_rule": args.location_rule,
                "location_seed": args.location_seed,
                "location_clones": args.location_clones,
                "block_ladder_sha256": household_location["block_ladder"]["sha256"],
            }
        )
    return {
        "version": 1,
        "stage": "acs_2024_1yr_multispine_base",
        "artifact_kind": "nullable_precalibration_staging_h5",
        "calibration_applied": False,
        "simulation_ready": False,
        "simulation_ready_except_calibration": True,
        "simulation_readiness_blockers": ["calibration_not_applied"],
        "reviewed_limitations": _reviewed_limitations(
            result,
            transfer_coverage=transfer_coverage,
            input_null_audit=input_null_audit,
        ),
        "period": args.period,
        "base": {
            "path": str(args.base_h5.resolve()),
            "sha256": base_sha256,
            "rows": base_rows,
            "household_weight_total": base_mass,
            "donor_release": donor_release,
        },
        "acs_sources": _source_provenance(
            manifest,
            source,
            manifest_path=args.source_manifest,
            manifest_sha256=manifest_sha256,
        ),
        "geography_ladder": geography_ladder,
        "orchestration": orchestration,
        "weights_audit": weights_audit,
        "transfer_coverage": transfer_coverage,
        "reviewed_engine_input_nulls": input_null_audit,
        "staging_export_peak_estimate_bytes": staging_export_peak_bytes,
        "rows": {
            "base": base_rows,
            "combined": output_rows,
        },
        "household_weight_totals": {
            "base": base_mass,
            "combined": output_mass,
        },
        "spine_totals": _spine_totals(result.frame),
        "output": {
            "path": str(args.out_h5.resolve()),
            "sha256": _sha256(args.out_h5),
            "summary_path": str(summary_path.resolve()),
            "rows": output_rows,
            "household_weight_total": output_mass,
        },
    }


def _source_provenance(
    manifest: acs_sources.AcsSourceManifest,
    source: AcsPumsSource,
    *,
    manifest_path: Path | None,
    manifest_sha256: str,
) -> dict[str, object]:
    resolved_manifest_path = _manifest_file(manifest_path)
    local_paths = {
        "household": source.household_zip.resolve(),
        "person": source.person_zip.resolve(),
    }
    return {
        "manifest": (
            str(resolved_manifest_path)
            if manifest_path is not None
            else _PACKAGED_MANIFEST_REFERENCE
        ),
        "manifest_sha256": manifest_sha256,
        "version": manifest.version,
        "spine": manifest.spine,
        "vintage": manifest.vintage,
        "verified_on": manifest.verified_on,
        "source_directory": manifest.source_directory,
        "artifacts": [
            {
                "role": artifact.role,
                "filename": artifact.filename,
                "url": artifact.url,
                "sha256": artifact.sha256,
                "size_bytes": artifact.size_bytes,
                "local_path": str(local_paths[artifact.role]),
            }
            for artifact in manifest.artifacts
        ],
    }


def _manifest_file(override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    return Path(acs_sources.__file__).with_name("acs_2024_1yr_sources.json")


def _validate_artifact_paths(
    args: argparse.Namespace,
    *,
    summary_path: Path,
) -> None:
    base = args.base_h5.resolve()
    output = args.out_h5.resolve()
    summary = summary_path.resolve()
    if args.out_h5.suffix != ".h5":
        raise SystemExit(f"--out-h5 must end with .h5, got {args.out_h5.name!r}.")
    if base == output:
        raise SystemExit("--out-h5 must differ from --base-h5.")
    if summary in {base, output}:
        raise SystemExit("--summary must differ from both --base-h5 and --out-h5.")
    ladder = args.puma_ladder.resolve()
    if ladder == output:
        raise SystemExit("--puma-ladder must differ from --out-h5.")
    block_ladder = getattr(args, "block_ladder", None)
    if block_ladder is not None and block_ladder.resolve() in {output, summary}:
        raise SystemExit("--block-ladder must differ from --out-h5 and --summary.")


def _spine_totals(frame: Frame) -> dict[str, dict[str, Any]]:
    spine_values: set[str] = set()
    for entity in frame.entities:
        column = spine_column(entity)
        table = frame.table(entity)
        if column not in table:
            raise ValueError(f"Combined ACS base lacks required spine tag {column!r}.")
        if table[column].isna().any():
            raise ValueError(
                f"Combined ACS base carries missing values in spine tag {column!r}."
            )
        spine_values.update(map(str, table[column].dropna().unique()))

    household = frame.table("household")
    household_spine = household[spine_column("household")]
    weights = pd.Series(
        frame.weights_for("household").values,
        index=household.index,
    )
    return {
        spine: {
            "rows": {
                entity: int(frame.table(entity)[spine_column(entity)].eq(spine).sum())
                for entity in frame.entities
            },
            "household_weight_total": float(
                weights.loc[household_spine.eq(spine)].sum()
            ),
        }
        for spine in sorted(spine_values)
    }


def _load_base_frame(path: Path) -> Frame:
    """Load the dense donor H5 without importing PolicyEngine-US at tool import."""
    consumer = "legacy ACS multispine base --base-h5 loader (_load_base_frame)"
    sha256 = refuse_denied_pool_h5(path, consumer=consumer)

    with pd.HDFStore(path, mode="r") as store:
        tables = {
            entity: read_frame_table(store, entity) for entity in US_SCHEMA.entities
        }
    tables["household"] = tables["household"].copy()
    household_weights = (
        tables["household"].pop("household_weight").to_numpy(dtype=np.float64)
    )
    assert_h5_unchanged(path, sha256, consumer=consumer)
    frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                household_weights,
                WeightKind.CALIBRATED,
            )
        },
    )
    refuse_denied_frame(frame, consumer=consumer)
    return frame


def _write_dataset(
    frame: Frame,
    path: Path,
    *,
    period: int,
    artifact_kind: str = "nullable_precalibration_staging_h5",
    root_attributes: dict[str, str] | None = None,
) -> None:
    """Write a microcosm US H5 with one-table-at-a-time verification.

    ``root_attributes`` (text values) are set on the HDF5 root node and read
    back; the default writes none, so a legacy file is unchanged.
    """

    output = Path(path)
    output.unlink(missing_ok=True)
    try:
        with pd.HDFStore(output, mode="w") as store:
            for entity in frame.entities:
                table = frame.table(entity)
                if entity == "household":
                    table = table.copy()
                    table["household_weight"] = frame.weights_for("household").values
                if len(table):
                    put_frame_table(
                        store,
                        entity,
                        table,
                        preferred_format="fixed",
                    )
            store.put(
                "_time_period",
                pd.Series([int(period)]),
                format="table",
            )
            store.put(
                "_populace_staging_metadata",
                pd.Series(
                    [
                        json.dumps(
                            {
                                "artifact_kind": artifact_kind,
                                "entity_hdf_format": "fixed_nullable",
                                "household_weight_kind": frame.weights_for(
                                    "household"
                                ).kind.value,
                            },
                            sort_keys=True,
                        )
                    ]
                ),
                format="table",
            )
            if root_attributes:
                node_attributes = store.get_node("/")._v_attrs
                for key, value in root_attributes.items():
                    node_attributes[key] = str(value)

        with pd.HDFStore(output, mode="r") as store:
            if root_attributes:
                stored_attributes = store.get_node("/")._v_attrs
                for key, value in root_attributes.items():
                    if _attribute_text(stored_attributes[key]) != str(value):
                        raise RuntimeError(
                            f"Staging H5 root attribute {key!r} did not round-trip."
                        )
            for entity in frame.entities:
                expected = frame.table(entity)
                if not len(expected):
                    continue
                stored = read_frame_table(store, entity)
                expected_columns = list(expected.columns)
                if entity == "household":
                    expected_columns.append("household_weight")
                if (
                    len(stored) != len(expected)
                    or list(stored.columns) != expected_columns
                ):
                    raise RuntimeError(
                        f"Staging H5 round trip changed {entity!r}: expected "
                        f"{len(expected)} rows/{expected_columns}, got "
                        f"{len(stored)} rows/{list(stored.columns)}."
                    )
                del stored
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def _engine_input_null_audit(
    frame: Frame,
    engine: Any | None = None,
) -> list[dict[str, object]]:
    """Inventory nullable engine inputs for the reviewed-limitations summary."""

    if engine is None:
        from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

        engine = PolicyEngineUSEngine()
    input_names = set(engine.variables())
    entries: list[dict[str, object]] = []
    for entity in frame.entities:
        table = frame.table(entity)
        tag = spine_column(entity)
        for column in sorted(set(table.columns).intersection(input_names)):
            missing = table[column].isna()
            if not missing.any():
                continue
            by_spine: dict[str, int] = {}
            if tag in table:
                by_spine = {
                    str(spine): int((missing & table[tag].eq(spine)).sum())
                    for spine in sorted(map(str, table[tag].dropna().unique()))
                    if int((missing & table[tag].eq(spine)).sum())
                }
            entries.append(
                {
                    "entity": entity,
                    "column": column,
                    "dtype": engine.variable_metadata(column).dtype,
                    "missing_rows": int(missing.sum()),
                    "rows": len(table),
                    "missing_rows_by_spine": by_spine,
                }
            )
    return entries


def _preflight_staging_export(
    frame: Frame,
    *,
    max_peak_bytes: int = DEFAULT_STAGING_EXPORT_PEAK_LIMIT_BYTES,
) -> int:
    if type(max_peak_bytes) is not int or max_peak_bytes <= 0:
        raise ValueError("max_peak_bytes must be a positive integer.")
    table_bytes = {
        entity: int(frame.table(entity).memory_usage(index=True, deep=True).sum())
        for entity in frame.entities
    }
    resident = sum(table_bytes.values())
    resident += sum(
        frame.weights_for(entity).values.nbytes for entity in frame.weighted_entities
    )
    resident += int(frame.strata.memory_usage(index=True, deep=True))
    largest_table = max(table_bytes.values(), default=0)
    household_copy = table_bytes.get("household", 0) + 8 * frame.n("household")
    estimate = int(
        resident
        + 2 * largest_table
        + household_copy
        + _STAGING_EXPORT_FIXED_OVERHEAD_BYTES
    )
    if estimate > max_peak_bytes:
        raise MemoryError(
            "Nullable ACS staging export is estimated to require "
            f"{estimate / 1_000_000_000:.2f} GB, above the "
            f"{max_peak_bytes / 1_000_000_000:.2f} GB limit."
        )
    return estimate


def _row_counts(frame: Frame) -> dict[str, int]:
    return {entity: frame.n(entity) for entity in frame.entities}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _open_unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be strictly between 0 and 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
