"""Materialize, calibrate, finalize, and package the US ACS local-area release.

The committed successor of the Build L runtime chain
(`national_combined_calibrate.py` + `materialize_admin.py` +
`finalize_artifact.py` + `package_release.py`), so a lineage refresh is one
command instead of a hand-carried script directory. Input: the multispine
staging H5 from ``tools/build_us_acs_multispine_base.py`` (donor spine +
ACS 2024 1-year spine, nullable, simulation-ready except calibration).

Stages (``--stage all`` runs materialize -> calibrate -> qa -> finalize ->
package; each is separately resumable):

  materialize : compile the state-level administrative surface from the
                Ledger feed exactly like the production path
                (``compile_us_fiscal_target_registry`` -> RI Medicaid
                substitution -> state {usda_snap, cms_medicaid[enrollment],
                irs_soi}; ``--soi-mode state`` by default -- Build O's
                state-geography SOI contract -- with ``totals`` and ``full``
                as explicit opt-ins), refuse a staging run that records no
                passing ACS local immigration stage (microcosm#1020), no
                passing native ACS work/disability stage (microcosm#1021), no
                passing ACS local income transfer or no passing ACS local SSI
                disability-criteria stage (microcosm#1022), seed
                ACS-row SNAP/TANF take-up (microcosm#1019) and fill
                the ACS rows' discretionary ABAWD exemption (a cap-based
                upper-bound proxy), housing-assistance receipt, Medicare
                take-up, vehicle count (native VEH) and Head Start take-up
                without the engine
                (microcosm#1022; the consumer export re-derives the same
                values), assign the ACS rows' SSI and Medicaid take-up
                after a household-batched engine pre-pass over the ACS
                households (microcosm#1022; the consumer export applies the
                recorded assignment), run the household-chunked engine pass
                under the
                nullable-artifact contract (input-schema projection +
                reviewed-null fill), add PUMA-ladder population marginals
                (state + congressional district), and write a lean float32
                target-frame checkpoint + targets.json. Heavy stage; a crash
                in calibrate never re-runs the microsim.
  calibrate   : epoch-batched warm-start calibrate on the lean checkpoint
                (adam, mass conserved, hard weight-ratio cap; resumable with
                --resume), write diagnostics and the calibrated weights onto
                a copy of the staging H5.
  qa          : chunked engine probe of the calibrated artifact recording
                per-spine SSI incidence and intensity (the microcosm#403
                signature, measured rather than assumed).
  finalize    : release-style gate report (PUMA-ladder gate, calibration
                gate, spine-composition evidence) + the reviewed-limitations
                register for this lineage (inherited #507 SSI aged-band
                collapse, #393 miscellaneous-income defect, CD marginal
                vintage, ESS concentration, sparse-selection donor, mixed
                sub-PUMA coverage), and flip the summary simulation-ready.
  package     : refuse a calibrated H5 that stores a model input the
                installed policyengine-us does not define (microcosm#1026;
                ``microcosm.data.stored_inputs``), refuse ACS persons made
                SSI-criteria-positive while the run records no SSI take-up
                handling for ACS rows (the SSI take-up release block,
                microcosm#1022), then
                assemble releases/<id>/ with the non-default local-area
                manifest shape (dataset_role ``non_default_local_area``,
                namespace ``buildo_acs_local``, donor identity chain, the
                one-command refresh recipe) + sha256sums. Publication stays
                a separate reviewed step (``microcosm-publish-release
                <dir> --no-latest``); this tool never uploads and never
                touches latest.json.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.acs_local_hours import acs_local_hours_signal_gate
from microcosm.build.us_runtime.acs_local_immigration import (
    ACS_LOCAL_IMMIGRATION_COLUMNS,
    ACS_LOCAL_IMMIGRATION_GATE_NAME,
    ACS_LOCAL_IMMIGRATION_ISSUE,
    acs_local_immigration_signal_gate,
)
from microcosm.build.us_runtime.acs_local_income import (
    ACS_LOCAL_INCOME_DONOR_CHANNEL,
    ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS,
    ACS_LOCAL_INCOME_TRANSFER_COLUMNS,
    ACS_LOCAL_INCOME_TRANSFER_GATE_NAME,
    ACS_LOCAL_INCOME_TRANSFER_ISSUE,
    acs_local_income_transfer_signal_gate,
)
from microcosm.build.us_runtime.acs_local_receipt_anchors import (
    ACS_LOCAL_RECEIPT_ANCHOR_GATE_NAME,
    ACS_LOCAL_RECEIPT_ANCHOR_ISSUE,
    ACS_LOCAL_RECEIPT_ANCHOR_METHOD,
    acs_local_receipt_anchor_signal_gate,
)
from microcosm.build.us_runtime.acs_local_spm_units import (
    ACS_LOCAL_SPM_UNIT_GATE_NAME,
    ACS_LOCAL_SPM_UNIT_ISSUE,
    ACS_LOCAL_SPM_UNIT_METHOD,
    acs_local_spm_unit_signal_gate,
)
from microcosm.build.us_runtime.acs_local_ssi_disability import (
    ACS_LOCAL_SSI_DISABILITY_COLUMN,
    ACS_LOCAL_SSI_DISABILITY_GATE_NAME,
    ACS_LOCAL_SSI_DISABILITY_ISSUE,
    ACS_LOCAL_SSI_DISABILITY_METHOD,
    acs_local_ssi_disability_signal_gate,
)
from microcosm.build.us_runtime.acs_local_ssi_medicaid_take_up import (
    ACS_LOCAL_SSI_MEDICAID_TAKE_UP_COLUMNS,
    ACS_LOCAL_SSI_MEDICAID_TAKE_UP_GATE_NAME,
    ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE,
    ACS_LOCAL_SSI_MEDICAID_TAKE_UP_METHOD,
    acs_local_ssi_medicaid_take_up_assignment,
    acs_local_ssi_medicaid_take_up_signal_gate,
    with_acs_local_ssi_medicaid_take_up,
    with_recorded_acs_local_ssi_medicaid_take_up,
)
from microcosm.build.us_runtime.acs_local_take_up import (
    ACS_LOCAL_ENGINE_FREE_FILL_COLUMNS,
    ACS_LOCAL_ENGINE_FREE_FILLS_ISSUE,
    ACS_LOCAL_TAKE_UP_COLUMNS,
    ACS_LOCAL_TAKE_UP_GATE_NAME,
    acs_local_take_up_signal_gate,
    with_acs_local_take_up_inputs,
)
from microcosm.build.us_runtime.acs_local_vehicles_head_start import (
    US_VEHICLES_VALUE_COLUMN,
)
from microcosm.build.us_runtime.acs_local_work_disability import (
    ACS_LOCAL_DISABILITY_COLUMNS,
    ACS_LOCAL_WORK_DISABILITY_GATE_NAME,
    ACS_LOCAL_WORK_DISABILITY_ISSUE,
    ACS_NATIVE_PROVENANCE,
    WEEKS_WORKED_EXPORT_BLOCKER,
    acs_local_work_disability_signal_gate,
)
from microcosm.build.us_runtime.ssi_disability_criteria import (
    SIPP_2023_SSI_DISABILITY_DONOR_SHA256,
)

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TOOLS_DIR.parent
if str(_TOOLS_DIR) not in sys.path:
    # The chunked materializer reuses the production target-frame seam from
    # the sibling release tool rather than forking it.
    sys.path.insert(0, str(_TOOLS_DIR))

PERIOD = 2024
RELEASE_NAMESPACE = "buildo_acs_local"
RELEASE_ID_PREFIX = "populace-us-2024-buildo-acs-local"
ARTIFACT_NAME = "populace_us_2024_acs_local"
ARTIFACT_FILENAME = f"{ARTIFACT_NAME}.h5"
HF_REPO_ID = "policyengine/populace-us"
LEGACY_STAGING_REFRESH_RECIPE = (
    "uv run tools/build_us_acs_multispine_base.py "
    "--base-h5 <next-certified-release>.h5 "
    "--donor-release-manifest <release_manifest.json> "
    "--out-h5 <run>/acs_multispine_staging.h5 "
    "--inputs-dir <acs-archive-cache> "
    "--puma-ladder build/us/us_puma_ladder_2020.npz"
)

#: ``--soi-mode`` values for the state-level ``irs_soi`` surface.
#:
#: ``state`` (the default; Max's ruling of 2026-09-22) is the SOI contract every
#: earlier ACS local-area release was calibrated to, Build O's and Build P's:
#: every ``irs_soi`` spec at state geography whose Ledger record set is not a
#: congressional-district file -- on the pinned feed, the three TY2022 Historic
#: Table 2 state tables (broad totals, AGI bands, EITC by qualifying
#: children). Build O selected it as ``full`` with congressional-district
#: targets switched off (``include_congressional_district_targets=False`` at
#: populace@77e2061); commit b7922b089 removed that switch, so no mode
#: reproduced it until this one.
#:
#: ``totals`` keeps only the specs whose ``target_role`` is not
#: ``soi_fiscal_distribution`` -- on the pinned feed, ACA premium tax credit
#: rows and no state AGI, income-tax or EITC total. ``full`` keeps every
#: state-bearing spec, including the TY2023 congressional-district file, which
#: needs a dense matrix too large for one 128 GB machine. Both are explicit
#: opt-ins. docs/us-acs-local-soi-target-surface.md has the measured surfaces.
SOI_MODE_STATE = "state"
SOI_MODE_TOTALS = "totals"
SOI_MODE_FULL = "full"
SOI_MODES = (SOI_MODE_STATE, SOI_MODE_TOTALS, SOI_MODE_FULL)
DEFAULT_SOI_MODE = SOI_MODE_STATE
#: Ledger record-set specs from a congressional-district file start with this;
#: ``state`` mode excludes them, which is what Build O's switch did.
CONGRESSIONAL_DISTRICT_RECORD_SET_SPEC_PREFIX = "irs_soi.congressional_district_"


def _require_soi_mode(soi_mode: str) -> str:
    if soi_mode not in SOI_MODES:
        raise ValueError(
            f"soi_mode must be one of {SOI_MODES}, got {soi_mode!r}; an "
            "unrecognised mode must never fall through to either surface."
        )
    return soi_mode


def release_refresh_recipe(soi_mode: str) -> str:
    """The one-command release refresh, pinned to the SOI surface it built.

    The recipe names ``--soi-mode`` explicitly so re-running it reproduces
    the recorded surface even if the parser default changes again.
    """

    _require_soi_mode(soi_mode)
    return (
        "uv run tools/build_us_acs_local_release.py --stage all "
        "--staging-h5 <run>/acs_multispine_staging.h5 "
        "--feed <ledger-facts.jsonl> --feed-sha256 <sha> "
        f"--soi-mode {soi_mode} "
        "--ladder build/us/us_puma_ladder_2020.npz "
        "--checkpoint-dir <run>/checkpoints "
        "--out-h5 <run>/populace_us_2024_acs_local.h5 "
        "--out <run>/release"
    )


# ---------------------------------------------------------------------------
# Shared telemetry
# ---------------------------------------------------------------------------


def rss() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9


def log(message: str) -> None:
    print(
        f"[{time.strftime('%H:%M:%S')}] {message} (peakRSS {rss():.2f}GB)",
        flush=True,
    )


class RssSampler(threading.Thread):
    """Sample peak RSS on a fixed cadence for the run's memory ledger."""

    def __init__(self, every: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.every = every
        self._stop = threading.Event()
        self.samples: list[tuple[float, float]] = []

    def run(self) -> None:
        while not self._stop.is_set():
            self.samples.append((round(time.time(), 1), round(rss(), 3)))
            self._stop.wait(self.every)

    def stop(self) -> None:
        self._stop.set()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


# ---------------------------------------------------------------------------
# Target surface: state-level administrative families (production compile)
# ---------------------------------------------------------------------------


def soi_surface_predicate(soi_mode: str):
    """The ``irs_soi`` spec filter for one ``--soi-mode``.

    Every mode keeps only specs carrying ``state_fips`` (state rows, and the
    congressional-district rows that also carry their state).

    - ``state`` then keeps a spec only at ``ledger_geography_level ==
      "state"`` whose ``ledger_layout_record_set_spec_id`` is not a
      congressional-district file, of any ``target_role`` (AGI-band rows
      included). A spec missing either key is not selected: the mode cannot
      tell which contract it belongs to.
    - ``totals`` drops every ``soi_fiscal_distribution`` spec (the
      jetsam-safe Option B of the Build L runbook). That role is not only
      AGI-band slices: every SOI fact without a named role gets it, which at
      state level includes the all-income-range state and district rows.
    - ``full`` keeps them all.
    """

    _require_soi_mode(soi_mode)

    def selected(spec) -> bool:
        metadata = spec.metadata
        if "state_fips" not in metadata:
            return False
        if soi_mode == SOI_MODE_STATE:
            record_set_spec = metadata.get("ledger_layout_record_set_spec_id")
            return (
                metadata.get("ledger_geography_level") == "state"
                and isinstance(record_set_spec, str)
                and bool(record_set_spec)
                and not record_set_spec.startswith(
                    CONGRESSIONAL_DISTRICT_RECORD_SET_SPEC_PREFIX
                )
            )
        return (
            soi_mode == SOI_MODE_FULL
            or metadata.get("target_role") != "soi_fiscal_distribution"
        )

    return selected


def _compiled_fiscal_registry(feed: str | Path):
    """The production compile path and its reviewed substitution records.

    feed -> ``compile_us_fiscal_target_registry(age_targets=True)`` ->
    ``apply_us_medicaid_enrollment_substitutions`` (RI FIPS-44).
    """

    from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
    from microcosm.build.us_runtime import (
        default_congressional_district_vintage_crosswalk_path,
        load_congressional_district_vintage_crosswalk,
    )
    from microcosm.build.us_runtime.fiscal_targets import (
        compile_us_fiscal_target_registry,
    )
    from microcosm.build.us_runtime.medicaid_take_up import (
        apply_us_medicaid_enrollment_substitutions,
    )

    artifact = load_ledger_consumer_artifact(str(feed))
    registry = compile_us_fiscal_target_registry(
        artifact.facts,
        target_period=PERIOD,
        congressional_district_vintage_crosswalk=(
            load_congressional_district_vintage_crosswalk(
                default_congressional_district_vintage_crosswalk_path()
            )
        ),
        age_targets=True,
    )
    return apply_us_medicaid_enrollment_substitutions(registry)


def state_admin_specs(
    feed: str | Path, families: list[str], soi_mode: str = DEFAULT_SOI_MODE
):
    """Select the state-level admin surface from the production compile path.

    feed -> ``compile_us_fiscal_target_registry(age_targets=True)`` ->
    ``apply_us_medicaid_enrollment_substitutions`` (RI FIPS-44) -> state-level
    {usda_snap, cms_medicaid[enrollment], irs_soi}. The SOI slice follows
    :func:`soi_surface_predicate`: ``state`` (the default), ``totals`` or
    ``full``.
    """

    # Refuse an unknown mode before loading the feed and compiling the registry.
    _require_soi_mode(soi_mode)

    from microcosm.calibrate.registry import TargetRegistry

    registry, ri_substitutions = _compiled_fiscal_registry(feed)

    def state_level(spec) -> bool:
        return "state_fips" in spec.metadata

    picked = []
    if "snap" in families:
        picked += registry.select(family="usda_snap", predicate=state_level).specs
    if "medicaid" in families:
        picked += registry.select(
            family="cms_medicaid",
            predicate=lambda spec: (
                state_level(spec)
                and spec.metadata.get("target_role") == "medicaid_enrollment"
            ),
        ).specs
    if "soi" in families:
        picked += registry.select(
            family="irs_soi", predicate=soi_surface_predicate(soi_mode)
        ).specs
    return TargetRegistry(list(picked), country="us"), ri_substitutions


def acs_local_ssi_medicaid_take_up_targets(feed: str | Path) -> dict[str, object]:
    """The counts the ACS SSI/Medicaid take-up stage scales (microcosm#1022).

    Compiled from the same feed and path as :func:`state_admin_specs` (a
    second compile, run before the staging frame is loaded) and read exactly
    as the fiscal lane's take-up stages read them: the SSA age-band SSI
    recipient counts, and the post-substitution CMS state Medicaid enrollment
    table with its reviewed substitution records.
    """

    import build_us_fiscal_refresh_release as release_tool

    registry, substitutions = _compiled_fiscal_registry(feed)
    try:
        ssi = release_tool._ssi_take_up_band_targets_from_registry(registry.specs)
        medicaid = release_tool._medicaid_source_target_table(registry.specs)
    except RuntimeError as exc:
        raise SystemExit(
            "The ledger feed cannot supply the ACS local SSI/Medicaid take-up "
            f"counts ({ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE}): {exc}"
        ) from exc
    return {
        "ssi_band_targets": ssi,
        "medicaid_state_targets": medicaid,
        "medicaid_substitutions": [dict(record) for record in substitutions],
    }


# ---------------------------------------------------------------------------
# Engine pass under the nullable-artifact contract
# ---------------------------------------------------------------------------


class UnregisteredNullError(ValueError):
    """NaN in an engine-input column NOT in reviewed_engine_input_nulls."""


class DeniedDefaultFillError(ValueError):
    """NaN in an engine input whose engine default must never be the fill."""


#: Why each runtime-owned input must never take its engine default, and what
#: fills it instead. A missing cell is a build defect even when the staging
#: register lists the column.
_NEVER_DEFAULT_FILLED_REASONS: dict[tuple[str, str], str] = {
    # microcosm#1019: the take-up default, True, is universal take-up; this
    # tool's ACS take-up stage fills them first.
    **{
        ("spm_unit", column): (
            "the engine default is universal take-up (microcosm#1019); run the "
            "ACS local take-up stage first"
        )
        for column in ACS_LOCAL_TAKE_UP_COLUMNS
    },
    # microcosm#1020: the immigration defaults are a citizen with a valid SSN
    # and 5 years since entry; staging's ACS local immigration stage fills
    # them.
    **{
        ("person", column): (
            "the engine default is a citizen with a valid SSN, 5 years since "
            f"entry ({ACS_LOCAL_IMMIGRATION_ISSUE}); re-run staging with the "
            "current builder"
        )
        for column in ACS_LOCAL_IMMIGRATION_COLUMNS
    },
    # microcosm#1021: the disability default, False, removes every disability
    # exemption; staging maps ACS rows natively and the donor release carries
    # its own, so neither spine should ever reach the fill.
    **{
        ("person", column): (
            "the engine default False removes every disability exemption "
            f"({ACS_LOCAL_WORK_DISABILITY_ISSUE}); re-run staging with the "
            "current builder"
        )
        for column in ACS_LOCAL_DISABILITY_COLUMNS
    },
    # microcosm#1022: the first three defaults bias SNAP on ACS rows and the
    # last two ignore measured or donor evidence; this tool's ACS take-up
    # stage fills them first, without the engine.
    **{
        key: f"{reason} ({ACS_LOCAL_ENGINE_FREE_FILLS_ISSUE}); run the ACS local "
        "take-up stage first"
        for key, reason in zip(
            ACS_LOCAL_ENGINE_FREE_FILL_COLUMNS,
            (
                "the engine default False switches the cap-based discretionary "
                "ABAWD exemption proxy (drawn at the 8% statutory cap) off for "
                "every ACS adult",
                "the engine default False drops every ACS housing-assistance "
                "recipient the transferred take-up flag records",
                "the engine default True enrolls every Medicare-eligible ACS "
                "person regardless of measured HINS3 coverage",
                "the engine default 0 gives every ACS household no vehicle "
                "although ACS VEH counts them",
                "the engine default True enrolls every Head Start-eligible ACS "
                "child aged 3-5",
            ),
            strict=True,
        )
    },
    # microcosm#1022: the income default, 0, drops every ACS recipient's
    # income; staging's separate ASEC-channel pass fills the ACS rows and the
    # donor release carries its own measured or imputed amounts.
    **{
        ("person", column): (
            f"the engine default 0 drops {reason} for every ACS recipient "
            f"({ACS_LOCAL_INCOME_TRANSFER_ISSUE}); re-run staging with the "
            "current builder"
        )
        for column, reason in zip(
            ACS_LOCAL_INCOME_TRANSFER_COLUMNS,
            (
                "child support received, SNAP and SSI unearned income",
                "child support paid, the SNAP child-support deduction",
                "workers' compensation, SNAP and SSI unearned income",
                "non-SSA disability benefits, SNAP and SSI unearned income",
                "401(k) distributions, gross and SNAP unearned income",
                "403(b) distributions, gross and SNAP unearned income",
                "SEP distributions, gross and SNAP unearned income",
                "Keogh distributions, gross and SNAP unearned income",
                "Roth-IRA distributions, SNAP and SSI unearned income",
            ),
            strict=True,
        )
    },
    # microcosm#1022: the SSI disability default, False, fails every ACS
    # person under 65 who is not blind; staging runs the donor's SIPP model on
    # the ACS rows and the donor release carries its own values.
    ("person", ACS_LOCAL_SSI_DISABILITY_COLUMN): (
        "the engine default False fails the SSI disability test for every ACS "
        f"person under 65 who is not blind ({ACS_LOCAL_SSI_DISABILITY_ISSUE}); "
        "re-run staging with the current builder"
    ),
    # microcosm#1022: the SSI and Medicaid take-up default, True, is universal
    # take-up; materialize assigns the ACS rows after an engine pre-pass and
    # the consumer export applies the recorded assignment.
    **{
        ("person", column): (
            f"the engine default is universal {program} take-up "
            f"({ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE}); run the ACS local "
            "SSI/Medicaid take-up stage first"
        )
        for column, program in zip(
            ACS_LOCAL_SSI_MEDICAID_TAKE_UP_COLUMNS, ("SSI", "Medicaid"), strict=True
        )
    },
}
NEVER_DEFAULT_FILLED = frozenset(_NEVER_DEFAULT_FILLED_REASONS)

#: The checkpoint file holding the ACS rows' SSI and Medicaid take-up that
#: materialize assigned (microcosm#1022); the consumer export applies it.
ACS_SSI_MEDICAID_TAKE_UP_FILENAME = "acs_local_ssi_medicaid_take_up.npz"


def _formula_owned_columns(base_frame) -> set[str]:
    """The present non-structural columns ``formula_owned_outputs`` classifies
    as engine outputs: what the engine pass holds back."""

    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    adapter = PolicyEngineUSEngine()
    structural = set(adapter._structural_columns())
    present = {
        column
        for entity in base_frame.entities
        for column in base_frame.table(entity).columns
    }
    return set(adapter.formula_owned_outputs(present)) - structural


def project_input_only(base_frame, period: int = PERIOD):
    """Hold back every non-input variable so FED SET == ENGINE INPUT SET.

    The multispine pool keeps raw/aggregate source columns alongside leaf
    inputs by design, and pe-core refuses NaN inputs, so every variable that
    is not an engine input is held back from the engine pass using one
    classifier — ``PolicyEngineUSEngine.formula_owned_outputs`` — the same
    complement of ``variables()`` the reviewed-null register is built from.
    Structural columns (ids, membership, weights, geography keys) stay.
    The H5 is never touched. Returns (projected_frame, dropped_by_entity).
    """

    from microcosm.frame import Frame

    formula_owned = _formula_owned_columns(base_frame)
    dropped: dict[str, list[str]] = {}
    if not formula_owned:
        return base_frame, dropped
    new_tables = {}
    for entity in base_frame.entities:
        table = base_frame.table(entity)
        held = sorted(set(table.columns) & formula_owned)
        if held:
            dropped[entity] = held
            new_tables[entity] = table.drop(columns=held)
        else:
            new_tables[entity] = table
    weights = {
        entity: base_frame.weights_for(entity)
        for entity in base_frame.weighted_entities
    }
    projected = Frame(
        new_tables,
        base_frame.schema,
        weights,
        base_frame.strata,
        mass_log=base_frame.mass_log,
    )
    return projected, dropped


def _resolve_engine_default(system, column: str):
    """The pe-us variable's own default (never a hardcoded zero)."""

    from enum import Enum

    variable = system.variables.get(column)
    if variable is None:
        raise KeyError(f"{column!r} is not a policyengine-us variable.")
    default = variable.default_value
    if isinstance(default, Enum):
        return default.name, "enum"
    if isinstance(default, (bool, np.bool_)):
        return bool(default), "bool"
    return default, type(default).__name__


def fill_reviewed_nulls(
    frame,
    summary_path: Path,
    manifest_path: Path | None = None,
    period: int = PERIOD,
):
    """Apply the nullable-artifact contract for the engine pass (H5 untouched).

    Every registered (entity, column) has its NaN filled with the pe-us
    variable's own default; NaN in any engine-input column NOT in the
    register is a hard error with a per-spine diagnostic — an artifact
    defect, surfaced not filled. NaN in a :data:`NEVER_DEFAULT_FILLED`
    column is refused before anything is filled, registered or not.
    """

    denied = [
        f"{entity}.{column} ({int(frame.table(entity)[column].isna().sum())} null "
        f"rows): {_NEVER_DEFAULT_FILLED_REASONS[(entity, column)]}"
        for entity, column in sorted(NEVER_DEFAULT_FILLED)
        if column in frame.table(entity) and frame.table(entity)[column].isna().any()
    ]
    if denied:
        raise DeniedDefaultFillError(
            f"Refusing to default-fill runtime-owned input(s): {'; '.join(denied)}."
        )

    from policyengine_us import CountryTaxBenefitSystem

    from microcosm.build.us_runtime.base_pool import spine_column
    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    summary = json.loads(Path(summary_path).read_text())
    register = {
        (entry["entity"], entry["column"]): entry
        for entry in summary.get("reviewed_engine_input_nulls", [])
    }
    if not register:
        raise ValueError(
            f"{summary_path} carries no reviewed_engine_input_nulls register; "
            "cannot apply the nullable-artifact engine-pass contract."
        )
    adapter = PolicyEngineUSEngine()
    input_names = set(adapter.variables())
    system = CountryTaxBenefitSystem()

    fills, drift_warnings, violations = [], [], []
    for entity in frame.entities:
        table = frame.table(entity)
        tag = spine_column(entity)
        for column in sorted(set(table.columns) & input_names):
            missing = table[column].isna()
            n_missing = int(missing.sum())
            if n_missing == 0:
                continue
            by_spine = {}
            if tag in table.columns:
                by_spine = {
                    str(spine): int((missing & table[tag].eq(spine)).sum())
                    for spine in sorted(map(str, table[tag].dropna().unique()))
                    if int((missing & table[tag].eq(spine)).sum())
                }
            if (entity, column) not in register:
                violations.append(
                    {
                        "entity": entity,
                        "column": column,
                        "missing_rows": n_missing,
                        "rows": len(table),
                        "missing_rows_by_spine": by_spine,
                    }
                )
                continue
            fill_value, kind = _resolve_engine_default(system, column)
            filled = table[column].fillna(fill_value)
            value_type = getattr(system.variables[column], "value_type", None)
            if value_type is bool:
                filled = filled.astype(bool)
            elif value_type is int:
                filled = filled.astype(np.int64)
            elif value_type is float:
                filled = filled.astype(np.float64)
            table[column] = filled
            recorded = int(register[(entity, column)]["missing_rows"])
            entry = {
                "entity": entity,
                "column": column,
                "filled_rows": n_missing,
                "rows": len(table),
                "fill_value": repr(fill_value),
                "fill_kind": kind,
                "register_missing_rows": recorded,
                "missing_rows_by_spine": by_spine,
            }
            fills.append(entry)
            if n_missing != recorded:
                drift_warnings.append(
                    {**entry, "warning": "actual null count differs from register"}
                )

    if manifest_path is not None:
        Path(manifest_path).write_text(
            json.dumps(
                {
                    "period": period,
                    "register_entries": len(register),
                    "columns_filled": len(fills),
                    "total_values_filled": sum(f["filled_rows"] for f in fills),
                    "count_mismatch_warnings": drift_warnings,
                    "unregistered_violations": violations,
                    "fills": fills,
                },
                indent=2,
            )
        )
    log(
        f"reviewed-null fill: {len(fills)} registered column(s), "
        f"{sum(f['filled_rows'] for f in fills):,} values (engine pass only)"
    )
    for drift in drift_warnings:
        log(
            f"  WARNING count drift {drift['entity']}.{drift['column']}: "
            f"actual {drift['filled_rows']} vs register "
            f"{drift['register_missing_rows']}"
        )
    if violations:
        detail = "; ".join(
            f"{v['entity']}.{v['column']} ({v['missing_rows']} null rows, "
            f"by spine {v['missing_rows_by_spine']})"
            for v in violations
        )
        raise UnregisteredNullError(
            f"{len(violations)} engine-input column(s) carry NaN but are NOT "
            "in the reviewed_engine_input_nulls register — artifact defect, "
            f"refusing to fill: {detail}"
        )
    return fills, drift_warnings


def materialize_chunked(
    base_frame,
    specs,
    *,
    hh_chunk: int = 40_000,
    batch: int = 5_000,
    period: int = PERIOD,
    dropped_manifest_path: Path | None = None,
    summary_path: Path | None = None,
    fills_manifest_path: Path | None = None,
    matrix_path: Path | None = None,
):
    """Household-chunked production materialization (memory-bounded).

    One full-frame Microsimulation caches the entire dependency closure of
    the SOI components (the 1.6M-row jetsam class), so the unchanged
    production ``_materialize_target_frame`` runs on whole-household
    sub-frames; each chunk's calc cache dies with its simulation and the
    float32 measure matrix accumulates on disk when ``matrix_path`` is set.
    """

    import build_us_fiscal_refresh_release as release_tool

    projected, dropped = project_input_only(base_frame, period=period)
    total_held = sum(len(columns) for columns in dropped.values())
    log(f"input-schema projection held back {total_held} formula-owned column(s)")
    if dropped_manifest_path is not None:
        Path(dropped_manifest_path).write_text(
            json.dumps(
                {
                    "period": period,
                    "held_back_formula_owned_columns": dropped,
                    "total": total_held,
                },
                indent=2,
            )
        )
    if summary_path is not None:
        fill_reviewed_nulls(
            projected,
            summary_path,
            manifest_path=fills_manifest_path,
            period=period,
        )
    else:
        raise ValueError(
            "materialize_chunked requires the staging summary; the "
            "nullable-artifact engine-pass contract needs its "
            "reviewed_engine_input_nulls register."
        )

    n_households = projected.n("household")
    household_ids = projected.table("household")["household_id"].to_numpy()
    position_by_id = pd.Series(np.arange(n_households), index=household_ids)
    person_household = projected.table("person")["person_household_id"].to_numpy()
    person_position = position_by_id.reindex(person_household).to_numpy()
    if np.isnan(person_position).any():
        raise RuntimeError(
            "person rows reference household ids absent from the household "
            "table; frame is invalid."
        )

    measure_names = None
    compiled_specs = None
    matrix = None
    chunk_stats = []
    n_chunks = (n_households + hh_chunk - 1) // hh_chunk
    if n_chunks > 1:
        release_tool._assert_group_entities_nest_in_households(projected)
        release_tool._assert_medicaid_claiming_tax_units_local(projected)
    for chunk_index, low in enumerate(range(0, n_households, hh_chunk)):
        high = min(low + hh_chunk, n_households)
        started = time.time()
        mask = (person_position >= low) & (person_position < high)
        sub_frame = projected.select(mask)
        target_frame, compiled_registry, _ = release_tool._materialize_target_frame(
            sub_frame,
            tuple(specs),
            maximum_microsim_batch_size=batch,
            refuse_population_aggregates=True if n_chunks > 1 else None,
        )
        names = [spec.measure for spec in compiled_registry.specs]
        if measure_names is None:
            measure_names = names
            compiled_specs = list(compiled_registry.specs)
            if matrix_path is not None:
                matrix = np.memmap(
                    matrix_path,
                    dtype=np.float32,
                    mode="w+",
                    shape=(n_households, len(names)),
                )
            else:
                matrix = np.zeros((n_households, len(names)), dtype=np.float32)
        elif names != measure_names:
            raise RuntimeError(
                f"chunk {chunk_index} compiled a different measure set "
                f"({len(names)} vs {len(measure_names)}); refusing to assemble."
            )
        chunk_households = target_frame.table("household")
        got_ids = chunk_households["household_id"].to_numpy()
        if len(got_ids) != high - low or not np.array_equal(
            got_ids, household_ids[low:high]
        ):
            raise RuntimeError(
                f"chunk {chunk_index} household order/id mismatch; refusing "
                "to assemble."
            )
        for j, measure in enumerate(measure_names):
            matrix[low:high, j] = chunk_households[measure].to_numpy(dtype=np.float32)
        if matrix_path is not None:
            matrix.flush()
        del target_frame, compiled_registry, chunk_households, sub_frame, got_ids
        gc.collect()
        stat = {
            "chunk": chunk_index + 1,
            "of": n_chunks,
            "households": high - low,
            "wall_s": round(time.time() - started, 1),
            "peak_rss_gb": round(rss(), 2),
        }
        chunk_stats.append(stat)
        log(f"materialize chunk {chunk_index + 1}/{n_chunks}: {stat['wall_s']}s")
    del projected
    gc.collect()
    return matrix, measure_names, compiled_specs, chunk_stats


# ---------------------------------------------------------------------------
# Population marginals + lean checkpoint
# ---------------------------------------------------------------------------

GROUP_IDS = {
    "tax_unit": "tax_unit_id",
    "spm_unit": "spm_unit_id",
    "family": "family_id",
    "marital_unit": "marital_unit_id",
}
PERSON_STRUCT = [
    "person_id",
    "person_household_id",
    "person_tax_unit_id",
    "person_spm_unit_id",
    "person_family_id",
    "person_marital_unit_id",
]


def ladder_population(ladder_path: Path, geographies: list[str]):
    from microcosm.build.us_runtime.puma_ladder import load_us_puma_ladder

    ladder = load_us_puma_ladder(ladder_path)
    puma_states = (ladder.puma // 100_000).astype(int)
    populations: dict[str, dict[int, float]] = {}
    if "state" in geographies:
        populations["state"] = {
            int(state): float(ladder.puma_population[puma_states == state].sum())
            for state in np.unique(puma_states)
        }
    if "cd" in geographies:
        populations["cd"] = {
            int(cd): float(
                ladder.cd_overlap_population[ladder.cd_overlap_cd == cd].sum()
            )
            for cd in np.unique(ladder.cd_overlap_cd)
        }
    return populations


def population_measure_arrays(frame, ladder_populations, geographies: list[str]):
    """Household-grain population measures as float32 arrays.

    Population in a geography = sum over persons-in-geo of household weight;
    every person shares its household's geography, so the household-grain
    measure is household_size x 1[household geo == value]. No engine
    involved. A ladder cell with no supporting household is DROPPED from the
    surface and returned in the fourth element — the caller decides whether
    a shrunken surface is acceptable (a capped smoke) or a defect (release).
    """

    households = frame.table("household")
    persons = frame.table("person")
    size = persons.groupby("person_household_id").size()
    household_size = households["household_id"].map(size).fillna(0).to_numpy(np.float32)
    names, arrays, values, dropped = [], [], [], []
    column_map = {
        "state": ("state_fips", "state"),
        "cd": ("congressional_district_geoid", "cd"),
    }
    for geography in geographies:
        column, key = column_map[geography]
        geo_values = pd.to_numeric(households[column]).to_numpy()
        for value, population in sorted(ladder_populations[key].items()):
            present = geo_values == value
            width = 2 if geography == "state" else 4
            name = f"pop_{geography}_{value:0{width}d}"
            if not present.any():
                dropped.append(name)
                continue
            names.append(name)
            arrays.append((household_size * present).astype(np.float32))
            values.append(float(population))
    return names, arrays, values, dropped


def population_target_specs(names, values):
    """Declare ladder population targets with the shared diagnostics hierarchy."""

    from microcosm.calibrate import TargetSpec
    from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_POSTAL
    from microcosm.calibrate.hierarchy import (
        CalibrationHierarchy,
        HierarchyCategory,
        HierarchyGeography,
        HierarchyNode,
    )

    provider = HierarchyNode("census_population", "Census population")
    category = HierarchyCategory(
        "census_population.resident_population",
        "Resident population",
        provider.id,
    )
    specs = []
    for name, value in zip(names, values, strict=True):
        if name.startswith("pop_state_"):
            fips = name.removeprefix("pop_state_")
            label = US_STATE_FIPS_TO_POSTAL.get(fips, f"State {fips}")
            geography = HierarchyGeography(f"0400000US{fips}", label, "state")
        elif name.startswith("pop_cd_"):
            geoid = name.removeprefix("pop_cd_")
            state_fips, district = geoid[:2], geoid[2:]
            state = US_STATE_FIPS_TO_POSTAL.get(state_fips, state_fips)
            district_label = (
                "at-large" if district == "00" else f"district {int(district)}"
            )
            label = f"{state} congressional {district_label}"
            geography = HierarchyGeography(
                f"5001800US{geoid}", label, "congressional_district"
            )
        else:  # pragma: no cover - names originate in population_measure_arrays
            raise ValueError(f"Unknown population target name {name!r}.")
        specs.append(
            TargetSpec(
                name=name,
                entity="household",
                measure=name,
                value=value,
                period=PERIOD,
                source="US Census Bureau 2020 PUMA population ladder",
                family="census_population",
                metadata={"geography_level": geography.level},
                hierarchy=CalibrationHierarchy(
                    provider=provider,
                    category=category,
                    geography=geography,
                    dimensions=(),
                    target=HierarchyNode(name, f"{label} resident population"),
                ),
            )
        )
    return tuple(specs)


def extract_struct_tables(frame):
    """Small structural/geography copies so the big frame can be freed early."""

    households = frame.table("household")
    struct_columns = [
        column
        for column in (
            "household_id",
            "state_fips",
            "congressional_district_geoid",
            "county_fips",
        )
        if column in households.columns
    ]
    person_columns = [
        column for column in PERSON_STRUCT if column in frame.table("person").columns
    ]
    return {
        "household_struct": households[struct_columns].copy(),
        "person": frame.table("person")[person_columns].copy(),
        "groups": {
            group: frame.table(group)[[id_column]].copy()
            for group, id_column in GROUP_IDS.items()
        },
        "weights": np.asarray(frame.weights_for("household").values, dtype=np.float64),
    }


def write_lean_checkpoint(
    struct,
    admin_matrix,
    admin_names,
    admin_specs,
    pop_names,
    pop_arrays,
    pop_values,
    checkpoint_dir: Path,
):
    """Assemble the lean target frame and its versioned target registry."""

    from microcosm.calibrate import TargetRegistry
    from microcosm.frame import put_frame_table

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(admin_matrix, np.memmap):
        admin_matrix = np.array(admin_matrix)
    parts = {
        column: struct["household_struct"][column]
        for column in struct["household_struct"].columns
    }
    for j, name in enumerate(admin_names):
        parts[name] = admin_matrix[:, j]
    for name, array in zip(pop_names, pop_arrays, strict=True):
        parts[name] = array
    lean_households = pd.DataFrame(parts)
    del parts
    gc.collect()
    lean_households["household_weight"] = struct["weights"]
    checkpoint_h5 = checkpoint_dir / "target_frame_lean.h5"
    with pd.HDFStore(checkpoint_h5, mode="w") as store:
        put_frame_table(
            store,
            "household",
            lean_households,
            preferred_format="fixed",
        )
        put_frame_table(
            store,
            "person",
            struct["person"],
            preferred_format="fixed",
        )
        for group, table in struct["groups"].items():
            put_frame_table(
                store,
                group,
                table,
                preferred_format="fixed",
            )
        store.put("_time_period", pd.Series([PERIOD]), format="table")
    registry = TargetRegistry(
        (*admin_specs, *population_target_specs(pop_names, pop_values)),
        country="us",
    )
    registry.to_json(checkpoint_dir / "target_registry.json")
    log(
        f"checkpoint: {checkpoint_h5.name} ({len(lean_households)} hh, "
        f"{len(admin_names) + len(pop_names)} measures), target_registry.json "
        f"({len(registry)} targets)"
    )
    return checkpoint_h5, registry


def load_lean_frame(checkpoint_h5: Path):
    from microcosm.frame import Frame, WeightKind, Weights, read_frame_table
    from microcosm.frame.units import US_SCHEMA

    tables = {}
    with pd.HDFStore(checkpoint_h5, mode="r") as store:
        for key in ["household", "person"] + list(GROUP_IDS):
            tables[key] = read_frame_table(store, key)
    design_weights = tables["household"].pop("household_weight").to_numpy(np.float64)
    return (
        Frame(
            tables,
            US_SCHEMA,
            {"household": Weights(design_weights, WeightKind.CALIBRATED)},
        ),
        design_weights,
    )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _staging_summary_path(args) -> Path:
    if args.staging_summary is not None:
        return args.staging_summary
    return Path(str(args.staging_h5)).with_suffix(".summary.json")


def _load_staging_frame(path: Path):
    from build_us_acs_multispine_base import _load_base_frame

    return _load_base_frame(Path(path))


def _with_local_take_up(frame, *, seed: int):
    """Seed ACS-row SNAP/TANF take-up (#1019) and the engine-free fills
    (#1022) before any reviewed-null fill."""

    frame, receipt = with_acs_local_take_up_inputs(frame, seed=seed)
    for column, entry in receipt["programs"].items():
        log(
            f"ACS take-up {column}: filled {entry['filled_rows']:,} rows, "
            f"weighted ACS share {entry['weighted_take_up_share']:.3f}"
        )
    for column, entry in receipt["engine_free_fills"]["columns"].items():
        log(f"ACS engine-free fill {column}: filled {entry['filled_rows']:,} rows")
    audit = receipt["engine_free_fills"]["columns"]["takes_up_medicare_if_eligible"][
        "hins3_audit"
    ]
    for kind in ("blank", "invalid"):
        counts = audit[kind]
        log(
            f"ACS HINS3 {kind} (read as not covered; informational): "
            f"{counts['rows']:,} rows, weight {counts['weight']:,.0f}; at 65+ "
            f"{counts['rows_65_plus']:,} rows, weight {counts['weight_65_plus']:,.0f}"
        )
    return frame, receipt


def _recorded_take_up(identity: dict) -> dict:
    """The ACS take-up assignment materialize calibrated against, or refuse.

    The consumer export re-derives the flags with the recorded seed and must
    reproduce the recorded digest, so both engine passes see the same flags.
    A receipt without the engine-free fills is a pre-#1022 checkpoint, whose
    ACS rows reached the engine pass with three SNAP-relevant defaults; one
    whose fills omit a column (the vehicle count or Head Start take-up)
    predates that fill.
    """

    receipt = identity.get("acs_local_take_up")
    fills = receipt.get("engine_free_fills") if isinstance(receipt, dict) else None
    if (
        not isinstance(receipt, dict)
        or type(receipt.get("seed")) is not int
        or not isinstance(receipt.get("assigned_sha256"), str)
    ):
        raise SystemExit(
            "run_identity.json records no ACS local take-up assignment "
            "(microcosm#1019): the checkpoint was materialized with every ACS "
            "SPM unit default-filled to take up SNAP and TANF. Re-run --stage "
            "materialize."
        )
    if (
        not isinstance(fills, dict)
        or fills.get("issue") != ACS_LOCAL_ENGINE_FREE_FILLS_ISSUE
    ):
        raise SystemExit(
            "run_identity.json records no ACS engine-free fills "
            f"({ACS_LOCAL_ENGINE_FREE_FILLS_ISSUE}): the checkpoint was "
            "materialized with every ACS row's discretionary ABAWD exemption, "
            "housing-assistance receipt and Medicare take-up at the engine "
            "default. Re-run --stage materialize."
        )
    columns = fills.get("columns")
    if not isinstance(columns, dict) or any(
        column not in columns for _, column in ACS_LOCAL_ENGINE_FREE_FILL_COLUMNS
    ):
        raise SystemExit(
            "run_identity.json records ACS engine-free fills without the vehicle "
            f"count and Head Start take-up ({ACS_LOCAL_ENGINE_FREE_FILLS_ISSUE}): "
            "the checkpoint was materialized with no ACS household owning a "
            "vehicle and every Head Start-eligible ACS child taking Head Start "
            "up. Re-run --stage materialize."
        )
    return receipt


def _acs_engine_values(
    view, *, variable: str, summary_path: Path, hh_chunk: int
) -> np.ndarray:
    """One household-batched engine pass over a pre-pass view (microcosm#1022).

    The stage calls this on the donor households first, to measure the donor
    rows' pooled contribution that sets each ACS residual target, then on the
    ACS households. The view gets materialize's own engine-pass contract
    (input-schema projection plus reviewed-null fill), then the fiscal lane's
    batched helper evaluates ``variable`` in chunks of ``hh_chunk``
    households, so the pre-pass holds one chunk's simulation at a time. The
    view is the stage's own copy of those households, so both contract steps
    edit it in place (the classifier of :func:`project_input_only`, without
    its copy): the pre-pass never holds a second copy of the rows, and the
    donor view (about 5% of persons) is freed before the ACS view is built.
    """

    import build_us_fiscal_refresh_release as release_tool

    helpers = {
        "uncapped_ssi": release_tool._ssi_person_uncapped_amount,
        "is_medicaid_eligible": release_tool._medicaid_person_eligibility,
    }
    helper = helpers[variable]
    started = time.time()
    formula_owned = _formula_owned_columns(view)
    for entity in view.entities:
        table = view.table(entity)
        held = sorted(set(table.columns) & formula_owned)
        if held:
            table.drop(columns=held, inplace=True)
    fill_reviewed_nulls(view, summary_path, period=PERIOD)
    if view.n("household") > hh_chunk:
        release_tool._assert_group_entities_nest_in_households(view)
        release_tool._assert_medicaid_claiming_tax_units_local(view)
    values = helper(view, maximum_microsim_batch_size=hh_chunk)
    gc.collect()
    log(
        f"Engine pre-pass {variable}: {len(values):,} persons "
        f"({time.time() - started:.1f}s, peak RSS {rss():.2f}GB)"
    )
    return values


def _with_local_ssi_medicaid_take_up(
    frame, *, seed: int, targets: dict, summary_path: Path, hh_chunk: int
):
    """Assign and gate ACS-row SSI and Medicaid take-up (microcosm#1022).

    Runs before materialize's engine pass, so the CMS enrollment targets
    select enrollees among eligible ACS persons instead of shrinking the
    eligible population; a failed gate stops the stage before the hours of
    materialization. The ACS targets are what the donor rows leave of the SSA
    and CMS counts on the pooled weights, measured with the same two engine
    passes over the donor households (microcosm#1060 review).
    """

    started = time.time()
    try:
        frame, receipt = with_acs_local_ssi_medicaid_take_up(
            frame,
            seed=seed,
            ssi_band_targets=targets["ssi_band_targets"],
            medicaid_state_targets=targets["medicaid_state_targets"],
            medicaid_substitutions=targets["medicaid_substitutions"],
            uncapped_ssi=lambda view: _acs_engine_values(
                view,
                variable="uncapped_ssi",
                summary_path=summary_path,
                hh_chunk=hh_chunk,
            ),
            medicaid_eligibility=lambda view: _acs_engine_values(
                view,
                variable="is_medicaid_eligible",
                summary_path=summary_path,
                hh_chunk=hh_chunk,
            ),
        )
    except ValueError as exc:
        raise SystemExit(
            "ACS local SSI/Medicaid take-up stage failed "
            f"({ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE}): {exc}"
        ) from exc
    receipt["engine_prepass"].update(
        stage_wall_s=round(time.time() - started, 1),
        hh_chunk=hh_chunk,
        peak_rss_gb=round(rss(), 2),
    )
    gate = acs_local_ssi_medicaid_take_up_signal_gate(frame, receipt=receipt)
    receipt["gate"] = {
        "name": gate.name,
        "passed": gate.passed,
        "failures": list(gate.failures),
    }
    if not gate.passed:
        raise SystemExit(
            "ACS local SSI/Medicaid take-up gate failed: " + "; ".join(gate.failures)
        )
    for program in ("ssi", "medicaid"):
        entry = receipt[program]
        log(
            f"ACS take-up {entry['column']}: filled {entry['filled_rows']:,} rows, "
            f"weighted ACS share {entry['weighted_take_up_share']:.3f}"
        )
    return frame, receipt


def _write_ssi_medicaid_take_up(frame, path: Path) -> str:
    """Record the ACS persons' SSI/Medicaid take-up; return the file sha256."""

    assignment = acs_local_ssi_medicaid_take_up_assignment(frame)
    person_ids = assignment["person_id"].to_numpy()
    if person_ids.dtype == object:
        person_ids = person_ids.astype(str)
    with Path(path).open("wb") as handle:
        np.savez(
            handle,
            person_id=person_ids,
            **{
                column: assignment[column].to_numpy(dtype=bool)
                for column in ACS_LOCAL_SSI_MEDICAID_TAKE_UP_COLUMNS
            },
        )
    return _sha256(path)


def _recorded_ssi_medicaid_take_up(identity: dict, checkpoint_dir: Path):
    """The ACS SSI/Medicaid take-up materialize assigned, or refuse.

    Returns the run-identity receipt and the recorded assignment file. A
    checkpoint without a passing receipt, or whose file is missing or changed,
    was materialized with every eligible ACS person taking SSI and Medicaid
    up (the engine default), or cannot prove what it calibrated against.
    """

    receipt = identity.get("acs_local_ssi_medicaid_take_up")
    recorded = receipt.get("assignment_file") if isinstance(receipt, dict) else None
    gate = receipt.get("gate") if isinstance(receipt, dict) else None
    if (
        not isinstance(receipt, dict)
        or receipt.get("issue") != ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE
        or receipt.get("method") != ACS_LOCAL_SSI_MEDICAID_TAKE_UP_METHOD
        or type(receipt.get("seed")) is not int
        or not isinstance(receipt.get("assigned_sha256"), str)
        or not isinstance(gate, dict)
        or gate.get("passed") is not True
        or not isinstance(recorded, dict)
        or recorded.get("name") != ACS_SSI_MEDICAID_TAKE_UP_FILENAME
        or not isinstance(recorded.get("sha256"), str)
    ):
        raise SystemExit(
            "run_identity.json records no passing ACS local SSI/Medicaid take-up "
            f"assignment ({ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE}): the checkpoint "
            "was materialized with every eligible ACS person taking SSI and "
            "Medicaid up (the engine default). Re-run --stage materialize."
        )
    path = Path(checkpoint_dir) / ACS_SSI_MEDICAID_TAKE_UP_FILENAME
    if not path.is_file() or _sha256(path) != recorded["sha256"]:
        raise SystemExit(
            f"The recorded ACS SSI/Medicaid take-up assignment {path} is missing "
            f"or changed since materialize ({ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE})."
            " Re-run --stage materialize."
        )
    return receipt, path


def _with_recorded_ssi_medicaid_take_up(frame, receipt: dict, path: Path):
    """Apply the recorded ACS SSI/Medicaid take-up; refuse a mismatch."""

    columns = ("person_id", *ACS_LOCAL_SSI_MEDICAID_TAKE_UP_COLUMNS)
    try:
        with np.load(path, allow_pickle=False) as recorded:
            table = pd.DataFrame({column: recorded[column] for column in columns})
        return with_recorded_acs_local_ssi_medicaid_take_up(
            frame, table, assigned_sha256=receipt["assigned_sha256"]
        )
    except (KeyError, OSError, ValueError) as exc:
        raise SystemExit(
            "The consumer export cannot reproduce the ACS SSI/Medicaid take-up "
            f"materialize calibrated against ({ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE}):"
            f" {exc} Re-run --stage materialize against this staging file."
        ) from exc


def do_materialize(args) -> None:
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    geographies = [item.strip() for item in args.geographies.split(",") if item.strip()]
    started = time.time()
    if args.feed_sha256:
        actual = _sha256(args.feed)
        if actual != args.feed_sha256:
            raise SystemExit(
                f"Ledger feed sha256 mismatch: {args.feed} is {actual}, "
                f"expected {args.feed_sha256}."
            )
    registry, ri_substitutions = state_admin_specs(
        args.feed, families, soi_mode=args.soi_mode
    )
    log(
        f"admin specs: {len(registry)} ({families}, soi_mode={args.soi_mode}); "
        f"RI substitution records={len(ri_substitutions)}"
    )
    summary_path = _staging_summary_path(args)
    if not summary_path.exists():
        raise SystemExit(
            f"Staging summary not found at {summary_path}; the "
            "nullable-artifact engine-pass contract requires its "
            "reviewed_engine_input_nulls register."
        )
    staging_summary = _load_json(summary_path)
    # microcosm#1020/#1021/#1022/#1023: refuse a pre-change staging run
    # (including one without the native SNAP receipt anchor) before hashing
    # or loading.
    _require_local_immigration(staging_summary)
    _require_local_work_disability(staging_summary)
    _require_local_income_transfer(staging_summary)
    _require_local_ssi_disability(staging_summary)
    _require_local_spm_units(staging_summary)
    _require_local_receipt_anchors(staging_summary)
    # microcosm#1022: the SSI and Medicaid take-up counts, from the same feed
    # and compile path, before the staging frame is loaded.
    ssi_medicaid_targets = acs_local_ssi_medicaid_take_up_targets(args.feed)
    log("hashing staging inputs for the run identity …")
    staging_sha = _sha256(args.staging_h5)
    ladder_sha = _sha256(args.ladder)
    frame = _load_staging_frame(args.staging_h5)
    _require_local_hours(frame, staging_summary)
    frame, take_up = _with_local_take_up(frame, seed=args.seed)
    log(
        f"loaded staging frame households={frame.n('household')} "
        f"({time.time() - started:.1f}s)"
    )
    # microcosm#1022: assign the ACS rows' SSI and Medicaid take-up after a
    # household-batched engine pre-pass over the ACS households, before the
    # engine pass materializes the Medicaid enrollment targets, and record the
    # assignment for the consumer export.
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    frame, ssi_medicaid = _with_local_ssi_medicaid_take_up(
        frame,
        seed=args.seed,
        targets=ssi_medicaid_targets,
        summary_path=summary_path,
        hh_chunk=args.hh_chunk,
    )
    del ssi_medicaid_targets
    ssi_medicaid["assignment_file"] = {
        "name": ACS_SSI_MEDICAID_TAKE_UP_FILENAME,
        "sha256": _write_ssi_medicaid_take_up(
            frame, args.checkpoint_dir / ACS_SSI_MEDICAID_TAKE_UP_FILENAME
        ),
    }
    gc.collect()

    started = time.time()
    matrix_path = args.checkpoint_dir / "measures_f32.mmap"
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    matrix, admin_names, compiled_specs, chunk_stats = materialize_chunked(
        frame,
        registry.specs,
        hh_chunk=args.hh_chunk,
        batch=args.batch,
        period=PERIOD,
        dropped_manifest_path=args.checkpoint_dir / "held_back_columns.json",
        summary_path=summary_path,
        fills_manifest_path=args.checkpoint_dir / "reviewed_null_fills.json",
        matrix_path=matrix_path,
    )
    log(
        f"materialized admin: {len(admin_names)} measures over "
        f"{len(chunk_stats)} chunks ({time.time() - started:.1f}s)"
    )
    if len(admin_names) != len(registry):
        raise SystemExit(
            f"Compiled admin surface has {len(admin_names)} measures but "
            f"{len(registry)} specs were declared; admin targets must never "
            "disappear silently between compile and materialization."
        )
    populations = ladder_population(args.ladder, geographies)
    pop_names, pop_arrays, pop_values, pop_dropped = population_measure_arrays(
        frame, populations, geographies
    )
    if pop_dropped:
        message = (
            f"{len(pop_dropped)} ladder population cell(s) have no "
            f"supporting household (e.g. {pop_dropped[:5]}). The calibrated "
            "surface would silently shrink."
        )
        if not args.allow_partial_geography:
            raise SystemExit(
                message + " Refusing for a release build; pass "
                "--allow-partial-geography only for capped smokes."
            )
        log("WARNING " + message + " Continuing (--allow-partial-geography).")
    log(f"population measures: {len(pop_names)} ({geographies})")
    struct = extract_struct_tables(frame)
    n_households = frame.n("household")
    del frame
    gc.collect()

    write_lean_checkpoint(
        struct,
        matrix,
        admin_names,
        compiled_specs,
        pop_names,
        pop_arrays,
        pop_values,
        args.checkpoint_dir,
    )
    del matrix, struct, pop_arrays
    gc.collect()
    matrix_path.unlink(missing_ok=True)
    registry_digest = _sha256(args.checkpoint_dir / "target_registry.json")
    (args.checkpoint_dir / "run_identity.json").write_text(
        json.dumps(
            {
                "staging_h5": str(Path(args.staging_h5).resolve()),
                "staging_sha256": staging_sha,
                "ladder_sha256": ladder_sha,
                "households": n_households,
                "n_targets": len(admin_names) + len(pop_names),
                "target_registry_sha256": registry_digest,
                "declared_admin_specs": len(registry),
                "compiled_admin_specs": len(admin_names),
                "population_cells_dropped": pop_dropped,
                "acs_local_take_up": take_up,
                "acs_local_ssi_medicaid_take_up": ssi_medicaid,
            },
            indent=2,
        )
    )
    (args.checkpoint_dir / "materialize_rss.json").write_text(
        json.dumps(
            {
                "soi_mode": args.soi_mode,
                "families": families,
                "n_admin": len(admin_names),
                "n_population": len(pop_names),
                "population_cells_dropped": pop_dropped,
                "hh_chunk": args.hh_chunk,
                "chunk_stats": chunk_stats,
                "materialize_peak_rss_gb": round(rss(), 3),
            },
            indent=2,
        )
    )
    log(f"materialize stage complete, peak RSS {rss():.2f}GB")


def _verify_run_identity(args, *, require: bool = True) -> dict:
    """Load and re-verify the materialize-time run identity."""

    identity = _load_json(args.checkpoint_dir / "run_identity.json")
    if not identity:
        if require:
            raise SystemExit(
                f"No run_identity.json under {args.checkpoint_dir}; run "
                "--stage materialize first."
            )
        return {}
    staging_sha = _sha256(args.staging_h5)
    if staging_sha != identity.get("staging_sha256"):
        raise SystemExit(
            "Staging H5 does not match the materialized checkpoint: "
            f"{args.staging_h5} is {staging_sha} but the run identity pins "
            f"{identity.get('staging_sha256')}. Re-run --stage materialize "
            "against this staging file."
        )
    return identity


def do_calibrate(args) -> None:
    from microcosm.calibrate import (
        TargetRegistry,
        calibrate,
        write_calibration_diagnostics,
    )

    identity = _verify_run_identity(args)
    # Refuse a pre-#1019 checkpoint before hours of solving, not at export.
    _recorded_take_up(identity)
    # Likewise a checkpoint materialized from a pre-#1020, pre-#1021 or
    # pre-#1022 (income or SSI disability) staging run: the consumer export at
    # the end of this stage would refuse its summary.
    staging_summary = _load_json(_staging_summary_path(args))
    _require_local_immigration(staging_summary)
    _require_local_work_disability(staging_summary)
    _require_local_income_transfer(staging_summary)
    _require_local_ssi_disability(staging_summary)
    _require_local_spm_units(staging_summary)
    _require_local_receipt_anchors(staging_summary)
    # microcosm#1022: and a checkpoint without the recorded ACS SSI/Medicaid
    # take-up the export must apply.
    _recorded_ssi_medicaid_take_up(identity, args.checkpoint_dir)
    checkpoint_h5 = args.checkpoint_dir / "target_frame_lean.h5"
    registry_path = args.checkpoint_dir / "target_registry.json"
    registry_sha = _sha256(registry_path)
    if registry_sha != identity.get("target_registry_sha256"):
        raise SystemExit(
            "target_registry.json changed since materialize; the checkpoint and "
            "surface no longer agree. Re-run --stage materialize."
        )
    registry = TargetRegistry.from_json(registry_path)
    frame, design_weights = load_lean_frame(checkpoint_h5)
    n_households = frame.n("household")
    if n_households != identity.get("households"):
        raise SystemExit(
            f"Lean checkpoint has {n_households} households but the run "
            f"identity pins {identity.get('households')}."
        )
    target_set = registry.to_target_set()
    log(
        f"calibrate: households={n_households}, targets={len(target_set)}, "
        f"design_total={design_weights.sum():,.0f}"
    )

    resume_npz = args.checkpoint_dir / "weights_latest.npz"
    warm, done = None, 0
    if args.resume and resume_npz.exists():
        saved = np.load(resume_npz)
        saved_identity = (
            str(saved["staging_sha256"]) if "staging_sha256" in saved else None
        )
        if saved_identity is not None and saved_identity != identity.get(
            "staging_sha256"
        ):
            raise SystemExit(
                "weights_latest.npz was produced against a different staging "
                "H5; refusing to warm-start from a foreign checkpoint."
            )
        warm, done = saved["weights"], int(saved["epochs_done"])
        if len(warm) != n_households:
            raise SystemExit(
                f"weights_latest.npz carries {len(warm)} weights but the "
                f"checkpoint has {n_households} households."
            )
        log(f"RESUME from {done} epochs")
    if done >= args.epochs:
        summary_path = args.checkpoint_dir / "calibration_summary.json"
        if summary_path.exists():
            log(
                f"calibration already complete at {done} epochs and "
                "the calibration summary exists; nothing to do (delete "
                "weights_latest.npz to recalibrate)."
            )
            _write_calibrated_artifact(
                args, np.asarray(warm, dtype=np.float64), identity
            )
            return
        raise SystemExit(
            f"weights_latest.npz reports {done} epochs (>= --epochs "
            f"{args.epochs}) but calibration_summary.json is missing. "
            "Delete the checkpoint to recalibrate, or raise --epochs."
        )
    batch = args.epoch_batch if args.epoch_batch > 0 else args.epochs
    result = None
    started = time.time()
    while done < args.epochs:
        this_batch = min(batch, args.epochs - done)
        batch_started = time.time()
        result = calibrate(
            frame,
            target_set,
            weight_entity="household",
            method="adam",
            epochs=this_batch,
            learning_rate=0.02,
            mass="conserve",
            max_weight_ratio=args.max_weight_ratio,
            target_loss_cap=args.target_loss_cap,
            l2_lambda=args.l2_lambda,
            seed=args.seed,
            warm_start_weights=warm,
        )
        done += this_batch
        warm = result.weights.copy()
        np.savez(
            resume_npz,
            weights=warm,
            epochs_done=done,
            initial_weights=design_weights,
            staging_sha256=np.str_(identity["staging_sha256"]),
        )
        log(
            f"batch -> {done}/{args.epochs} ep, "
            f"{time.time() - batch_started:.1f}s, "
            f"loss={result.final_loss:.5f}, "
            f"within10%={result.fraction_within_10pct:.2%}, "
            f"ESS={result.effective_sample_size:,.0f}"
        )

    if result.problem.skipped:
        skipped = [getattr(item, "name", str(item)) for item in result.problem.skipped]
        raise SystemExit(
            f"Calibration compiled {len(skipped)} target(s) away "
            f"(e.g. {skipped[:5]}); the surface silently shrank. Fix the "
            "measures or the targets before shipping."
        )
    summary = {
        "households": n_households,
        "n_targets": result.problem.n_targets,
        "families": args.families,
        "geographies": args.geographies,
        "matrix_format": result.options["matrix_format"],
        "matrix_shape": [int(x) for x in result.problem.matrix.shape],
        "matrix_nnz": int(result.problem.matrix.nnz),
        "epochs": args.epochs,
        "epoch_batch": args.epoch_batch,
        "max_weight_ratio": args.max_weight_ratio,
        "l2_lambda": args.l2_lambda,
        "seed": args.seed,
        "initial_loss": round(result.initial_loss, 6),
        "final_loss": round(result.final_loss, 6),
        "fraction_within_10pct": round(result.fraction_within_10pct, 4),
        "effective_sample_size": round(result.effective_sample_size, 1),
        "ess_fraction": round(result.effective_sample_size / n_households, 4),
        "realized_max_weight_ratio": round(result.realized_max_weight_ratio, 4),
        "mass_conserved_ratio": round(
            float(result.weights.sum()) / float(design_weights.sum()), 6
        ),
        "total_wall_seconds": round(time.time() - started, 1),
        "peak_rss_gb": round(rss(), 3),
    }
    _write_calibrated_artifact(
        args, np.asarray(result.weights, dtype=np.float64), identity
    )

    outcome = write_calibration_diagnostics(
        result,
        args.checkpoint_dir / "calibration_diagnostics.json",
        target_registry=registry,
        build={
            "dataset_role": "non_default_local_area",
            "families": args.families,
            "geographies": args.geographies,
            "epochs": args.epochs,
            "epoch_batch": args.epoch_batch,
            "total_wall_seconds": summary["total_wall_seconds"],
            "peak_rss_gb": summary["peak_rss_gb"],
            "ess_fraction": summary["ess_fraction"],
            "mass_conserved_ratio": summary["mass_conserved_ratio"],
        },
    )
    summary["calibration_diagnostics"] = (
        {
            "status": "available",
            "schema_version": outcome.schema_version,
            "sha256": outcome.sha256,
        }
        if outcome.status == "available"
        else {
            "status": "failed",
            "expected_schema_version": outcome.expected_schema_version,
            "error_code": outcome.error_code,
            "message": outcome.message,
        }
    )
    (args.checkpoint_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    log(
        f"calibrate stage complete: loss={summary['final_loss']}, "
        f"within10%={summary['fraction_within_10pct']:.2%}, "
        f"diagnostics={outcome.status}"
    )


def _write_calibrated_artifact(args, weights: np.ndarray, identity: dict) -> None:
    """Write the consumer-ready calibrated H5 with verified attachment.

    The weights attach by verified household-id vector equality between the
    lean checkpoint and the freshly loaded staging H5 (never positionally to
    an unverified file), and the engine-pass contract is applied to the
    artifact bytes — input-schema projection plus reviewed-null default
    fill — so a plain ``USSingleYearDataset``/``Microsimulation`` consumer
    loads the published file directly. The nullable staging H5 remains the
    archival nullable truth; every fill is recorded in the run's consumer
    manifest.
    """

    if args.out_h5 is None:
        return
    from build_us_acs_multispine_base import _write_dataset

    from microcosm.frame import Frame, WeightKind, Weights

    recorded_take_up = _recorded_take_up(identity)
    staging_summary = _load_json(_staging_summary_path(args))
    _require_local_immigration(staging_summary)
    _require_local_work_disability(staging_summary)
    _require_local_income_transfer(staging_summary)
    _require_local_ssi_disability(staging_summary)
    _require_local_spm_units(staging_summary)
    _require_local_receipt_anchors(staging_summary)
    ssi_medicaid, ssi_medicaid_path = _recorded_ssi_medicaid_take_up(
        identity, args.checkpoint_dir
    )
    frame = _load_staging_frame(args.staging_h5)
    _require_local_hours(frame, staging_summary)
    frame, take_up = _with_local_take_up(frame, seed=recorded_take_up["seed"])
    if take_up["assigned_sha256"] != recorded_take_up["assigned_sha256"]:
        raise SystemExit(
            "The consumer export's ACS take-up assignment differs from the one "
            "materialize calibrated against (microcosm#1019). Re-run --stage "
            "materialize against this staging file."
        )
    # microcosm#1022: apply the ACS SSI/Medicaid take-up materialize recorded;
    # it must reproduce the recorded digest on this staging file.
    frame = _with_recorded_ssi_medicaid_take_up(frame, ssi_medicaid, ssi_medicaid_path)
    staging_ids = frame.table("household")["household_id"].to_numpy()
    with pd.HDFStore(args.checkpoint_dir / "target_frame_lean.h5", mode="r") as store:
        lean_ids = store["household"]["household_id"].to_numpy()
    if len(staging_ids) != len(lean_ids) or not np.array_equal(staging_ids, lean_ids):
        raise SystemExit(
            "Staging household ids do not match the lean checkpoint's; "
            "refusing to attach calibrated weights to a different or "
            "reordered file."
        )
    if len(weights) != len(staging_ids):
        raise SystemExit(
            f"{len(weights)} calibrated weights cannot attach to "
            f"{len(staging_ids)} households."
        )

    # TODO(policyengine-us#9660): the staging ACS rows carry native
    # weeks_worked (microcosm#1021), but the pinned policyengine-us gives it a
    # formula_2025, so this projection holds it back as formula-owned. Once
    # #9660 part A1 deletes that formula and the pin/ABI lock are bumped, it
    # becomes an input and ships; the donor rows then need weeks_worked from
    # WKSWORK too (US_HOURS_WORKED_POOL_EXCLUDED_COLUMNS).
    projected, dropped = project_input_only(frame, period=PERIOD)
    del frame
    gc.collect()
    fills, _ = fill_reviewed_nulls(
        projected,
        _staging_summary_path(args),
        manifest_path=args.checkpoint_dir / "consumer_reviewed_null_fills.json",
        period=PERIOD,
    )
    calibrated = Frame(
        {entity: projected.table(entity) for entity in projected.entities},
        projected.schema,
        {"household": Weights(weights, WeightKind.CALIBRATED)},
        projected.strata,
        mass_log=projected.mass_log,
    )
    _write_dataset(
        calibrated,
        Path(args.out_h5),
        period=PERIOD,
        artifact_kind="calibrated_local_area_artifact",
    )
    (args.checkpoint_dir / "consumer_export.json").write_text(
        json.dumps(
            {
                "out_h5": str(Path(args.out_h5).resolve()),
                "staging_sha256": identity.get("staging_sha256"),
                "held_back_formula_owned": dropped,
                "held_back_total": sum(len(v) for v in dropped.values()),
                "filled_columns": len(fills),
                "total_values_filled": sum(f["filled_rows"] for f in fills),
                "acs_local_take_up": take_up,
                "acs_local_ssi_medicaid_take_up": {
                    "assigned_sha256": ssi_medicaid["assigned_sha256"],
                    "assignment_file": ssi_medicaid["assignment_file"],
                },
                "note": (
                    "The engine-pass contract (input-schema projection + "
                    "reviewed-null default fill) is applied to the published "
                    "artifact bytes, so plain USSingleYearDataset/"
                    "Microsimulation consumers load it directly. The "
                    "nullable staging H5 remains the archival nullable "
                    "truth."
                ),
            },
            indent=2,
        )
    )
    log(f"wrote consumer-ready calibrated artifact {args.out_h5}")


def spine_composition(households: pd.DataFrame, persons: pd.DataFrame, weights):
    """Per-spine composition evidence (the microcosm#403 skew signature)."""

    from microcosm.build.us_runtime.base_pool import spine_column

    tag = spine_column("household")
    weights = np.asarray(weights, dtype=np.float64)
    person_counts = persons.groupby("person_household_id").size()
    household_size = (
        households["household_id"].map(person_counts).fillna(0).to_numpy(np.float64)
    )
    total_weight = float(weights.sum())
    composition = {}
    for spine in sorted(map(str, households[tag].dropna().unique())):
        mask = households[tag].eq(spine).to_numpy()
        spine_weight = float(weights[mask].sum())
        spine_person_weight = float((weights[mask] * household_size[mask]).sum())
        composition[spine] = {
            "households": int(mask.sum()),
            "household_weight": spine_weight,
            "household_weight_share": (
                spine_weight / total_weight if total_weight else 0.0
            ),
            "person_weight": spine_person_weight,
            "implied_persons_per_household": (
                spine_person_weight / spine_weight if spine_weight else 0.0
            ),
        }
    ess = float(weights.sum() ** 2 / (weights**2).sum()) if len(weights) else 0.0
    composition["_all"] = {
        "households": int(len(weights)),
        "household_weight": total_weight,
        "effective_sample_size": ess,
        "ess_fraction": ess / len(weights) if len(weights) else 0.0,
    }
    return composition


def do_qa(args) -> None:
    """Chunked engine probe: per-spine SSI incidence on the calibrated artifact.

    Loads the packaged artifact bytes PLAIN — no private projection or fill —
    so the probe doubles as the proof that ordinary
    ``USSingleYearDataset``/``Microsimulation`` consumers can load the file
    (the engine-pass contract was applied at export).
    """

    from microcosm.build.us_runtime.base_pool import spine_column

    projected = _load_staging_frame(args.out_h5)

    n_households = projected.n("household")
    household_ids = projected.table("household")["household_id"].to_numpy()
    position_by_id = pd.Series(np.arange(n_households), index=household_ids)
    person_household = projected.table("person")["person_household_id"].to_numpy()
    person_position = position_by_id.reindex(person_household).to_numpy()
    household_weights = np.asarray(
        projected.weights_for("household").values, dtype=np.float64
    )
    person_tag = projected.table("person")[spine_column("person")].to_numpy()

    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    adapter = PolicyEngineUSEngine()
    per_spine: dict[str, dict[str, float]] = {}
    n_chunks = (n_households + args.hh_chunk - 1) // args.hh_chunk
    for chunk_index, low in enumerate(range(0, n_households, args.hh_chunk)):
        high = min(low + args.hh_chunk, n_households)
        mask = (person_position >= low) & (person_position < high)
        sub_frame = projected.select(mask)
        ssi = np.asarray(
            adapter.materialize(sub_frame, ["ssi"], PERIOD)["ssi"],
            dtype=np.float64,
        )
        sub_person_household = sub_frame.table("person")[
            "person_household_id"
        ].to_numpy()
        sub_positions = position_by_id.reindex(sub_person_household).to_numpy()
        person_weight = household_weights[sub_positions.astype(np.int64)]
        sub_tags = person_tag[mask]
        for spine in np.unique(sub_tags):
            spine_mask = sub_tags == spine
            entry = per_spine.setdefault(
                str(spine),
                {
                    "person_weight": 0.0,
                    "ssi_recipient_weight": 0.0,
                    "ssi_dollars": 0.0,
                },
            )
            entry["person_weight"] += float(person_weight[spine_mask].sum())
            recipients = spine_mask & (ssi > 0)
            entry["ssi_recipient_weight"] += float(person_weight[recipients].sum())
            entry["ssi_dollars"] += float(
                (ssi[recipients] * person_weight[recipients]).sum()
            )
        del sub_frame
        gc.collect()
        log(f"qa chunk {chunk_index + 1}/{n_chunks}")

    for entry in per_spine.values():
        entry["ssi_incidence"] = (
            entry["ssi_recipient_weight"] / entry["person_weight"]
            if entry["person_weight"]
            else 0.0
        )
    payload = {
        "period": PERIOD,
        "variable": "ssi",
        "artifact": str(Path(args.out_h5).resolve()),
        "artifact_sha256": _sha256(args.out_h5),
        "plain_consumption": True,
        "per_spine": per_spine,
        "note": (
            "microcosm#403 re-measure: per-spine SSI incidence and intensity, "
            "computed by loading the packaged artifact bytes PLAIN (no "
            "private projection or fill) — the probe doubles as the "
            "consumer-loadability proof. Recorded as evidence, not gated."
        ),
    }
    (args.checkpoint_dir / "spine_qa.json").write_text(json.dumps(payload, indent=2))
    log(
        f"qa stage complete: {json.dumps({k: round(v['ssi_incidence'], 4) for k, v in per_spine.items()})}"
    )


def _repo_code_identity(allow_dirty: bool) -> dict[str, object]:
    def _git(*parts: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), *parts], text=True
        ).strip()

    sha = _git("rev-parse", "--short", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    if dirty and not allow_dirty:
        raise SystemExit(
            "Repository tree is dirty; a packaged release id must name a "
            "committed code vintage. Commit first or pass --allow-dirty "
            "(the manifest will record dirty=true)."
        )
    return {"sha": sha, "dirty": dirty, "branch": _git("branch", "--show-current")}


def finalize_reviewed_limitations(
    staging_summary: dict,
    diagnostics: dict,
    spine_qa: dict,
) -> list[dict]:
    """The reviewed-limitations register for the buildo-acs-local lineage.

    Carries the staging summary's reviewed limitations (GQ housing universe,
    native source-universe blanks, sub-PUMA precision) and adds the
    lineage-level entries, id-deduped so re-running finalize never
    duplicates.
    """

    ess = diagnostics.get("effective_sample_size")
    ess_fraction = diagnostics.get("ess_fraction", 0.0)
    households = diagnostics.get("households") or 0
    donor_release = (staging_summary.get("base") or {}).get("donor_release") or {}
    limitations = list(staging_summary.get("reviewed_limitations", []))
    aged_ssi = (spine_qa or {}).get("per_spine", {})
    limitations += [
        {
            "id": "ssi_aged_band_collapse_inherited",
            "status": "reviewed_inherited_defect",
            "reason": (
                "The donor lineage (base-O via the certified Build O sparse "
                "release) carries the microcosm#507 SSI aged-band take-up "
                "collapse: the 65+ one-shot Bernoulli seed at 8.4% collapses "
                "the aged SSI baseline to roughly 0.94M against SSA's 2.42M. "
                "The QRF transfer replicates donor SSI participation onto "
                "the ACS spine, so both spines of this artifact inherit the "
                "defect by construction."
            ),
            "treatment": (
                "microcosm#508 owns the fix; when the corrected base "
                "certifies, this artifact's refresh_recipe re-runs the "
                "chain against the new certified release in one command."
            ),
            "measured_spine_ssi": {
                spine: round(entry.get("ssi_incidence", 0.0), 5)
                for spine, entry in aged_ssi.items()
            },
            "calibration_blocker": False,
        },
        {
            "id": "miscellaneous_income_loss_side_donor_defect",
            "status": "reviewed_inherited_distortion",
            "column": "miscellaneous_income",
            "affected_spines": ["asec_puf", "acs_2024_1yr"],
            "reason": (
                "The donor pool's miscellaneous_income loss side remains "
                "distorted (microcosm#393, open at build time; roughly 4.6x "
                "SOI's loss-return prevalence measured in the Build J dense "
                "remedy experiments). The QRF transfer replicates the donor "
                "distribution onto the ACS spine."
            ),
            "treatment": "Tracked via microcosm#393; propagates on re-transfer.",
            "calibration_blocker": False,
        },
        {
            "id": "tips_return_count_carrier_deficit_inherited",
            "status": "reviewed_inherited_defect",
            "reason": (
                "The certified Build O sparse donor under-carries tip-return "
                "counts (-50.25% against its ledger target, recorded in its "
                "own release evidence). Tip columns transfer from that "
                "donor, so the deficit rides this artifact too."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "acs_take_up_engine_defaults",
            "status": "reviewed_known_gap",
            "affected_spines": ["acs_2024_1yr"],
            "reason": (
                "SNAP and TANF take-up on ACS rows are seeded by the local "
                "runtime and gated by acs_local_take_up_signal "
                "(microcosm#1019; its SNAP reporters are the native household "
                "FS anchor, acs_snap_receipt_anchor), Medicare take-up is "
                "native ACS HINS3 and Head Start take-up a keyed draw at the "
                "donor spine's share among ages 3-5 "
                "(acs_engine_free_default_fills), and SSI and Medicaid "
                "take-up are assigned after an engine pre-pass "
                "(acs_local_ssi_medicaid_take_up). The other runtime-owned "
                "takes_up_* flags (EITC, ACA, Early Head Start and the rest) "
                "are neither transferred nor seeded on ACS rows, so they ship "
                "at the engine default, universal take-up."
            ),
            "treatment": "Tracked via microcosm#1022.",
            "calibration_blocker": False,
        },
        {
            "id": "acs_engine_free_default_fills",
            "status": "reviewed_modeling_decision",
            "affected_spines": ["acs_2024_1yr"],
            "columns": [column for _, column in ACS_LOCAL_ENGINE_FREE_FILL_COLUMNS],
            "reason": (
                "Five inputs are filled on ACS rows without the engine "
                f"({ACS_LOCAL_ENGINE_FREE_FILLS_ISSUE}) instead of taking their "
                "engine defaults; the first three bias SNAP. "
                "is_snap_abawd_discretionary_exempt "
                "mirrors the donor seeding: every person aged 18-64 draws "
                "against the snap_abawd_discretionary_exemption manifest rate "
                "(the statutory cap), keyed on acs_2024_1yr:SERIALNO:SPORDER. "
                "It is a cap-based proxy, not an observed exemption assignment: "
                "seeding all adults rather than covered individuals only, at "
                "the cap rather than actual state usage, makes it an "
                "upper-bound propensity, as on the donor. "
                "receives_housing_assistance copies the transferred "
                "takes_up_housing_assistance_if_eligible (equal on the donor by "
                "construction) and is False in TYPEHUGQ 2/3 group quarters, "
                "whose transferred take-up flag is the open transfer-side fix "
                "microcosm#975. takes_up_medicare_if_eligible is ACS HINS3 == 1 "
                "(coverage at interview), as the donor maps ASEC MCARE == 1; a "
                "blank or invalid HINS3 reads as not covered. household_vehicles_owned "
                "is "
                "ACS VEH, the cars, vans and trucks kept at home for household "
                "use (0-6, 6 meaning six or more), which also counts leased and "
                "employer-provided vehicles the donor's SIPP TVEH_NUM (vehicles "
                "owned) leaves out; it is 0 in group quarters, outside the "
                "item's universe, and household_vehicles_value keeps its "
                "default (acs_household_vehicle_value_default). "
                "takes_up_head_start_if_eligible is True for a person aged 3-5 "
                "whose stable draw, keyed on acs_2024_1yr:SERIALNO:SPORDER, "
                "falls below the donor spine's weighted take-up share at those "
                "ages, the output of the donor's measured SIPP model "
                "(sipp_head_start) at its population rate but without its "
                "household and earnings conditioning; it is False at every "
                "other age, as on the donor, and the engine applies Head Start "
                "eligibility."
            ),
            "treatment": (
                "Filled by the release tool before both reviewed-null fills, "
                "digested in run_identity.json and re-derived at export; gated "
                "by acs_local_take_up_signal (exempt share of ages 18-64 around "
                "the manifest rate, receipt equal to the transferred take-up, "
                "Medicare equal to HINS3 with a high share at 65+, vehicle "
                "counts equal to VEH and 0 in group quarters, Head Start "
                "take-up only at ages 3-5 and within half to one and a half "
                "times the donor spine's share there). Blank and invalid HINS3 "
                "counts (unweighted, weighted and at 65+) are reported in the "
                "receipt and the gate detail, not graded."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "acs_household_vehicle_value_default",
            "status": "reviewed_modeling_decision",
            "affected_spines": ["acs_2024_1yr"],
            "columns": [US_VEHICLES_VALUE_COLUMN],
            "reason": (
                "The ACS asks how many vehicles a household keeps (VEH, the ACS "
                "rows' household_vehicles_owned) but not what they are worth, "
                f"so {US_VEHICLES_VALUE_COLUMN} stays at the reviewed engine "
                f"default, 0, on ACS rows ({ACS_LOCAL_ENGINE_FREE_FILLS_ISSUE}). "
                "In policyengine-us 2.2.1 its only SNAP consumer is Texas's "
                "broad-based categorical eligibility asset test "
                "(meets_tanf_non_cash_asset_test), which counts vehicle value "
                "above $22,500 for the first vehicle and $8,700 for each other "
                "against a $5,000 limit; SNAP's own asset test reads no vehicle. "
                "It also feeds the TANF resource tests of Texas, Washington, "
                "Montana, Tennessee, California and Wisconsin (TANF is SNAP "
                "unearned income), Illinois AABD, California CAPI and "
                "California county general assistance. A zero value counts no "
                "vehicle equity, so those tests pass more often on ACS rows "
                "than on donor rows, whose value is the SIPP model's "
                "(sipp_vehicles)."
            ),
            "treatment": (
                "Reviewed-null fill to 0, recorded in the consumer fill "
                "manifest. If the Texas asset test matters, the follow-up is "
                "the donor stage's SIPP value model run at staging on the ACS "
                "rows with the native VEH count in place of its predicted count."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "acs_immigration_status_method",
            "status": "reviewed_modeling_decision",
            "affected_spines": ["acs_2024_1yr", "asec_puf"],
            "columns": list(ACS_LOCAL_IMMIGRATION_COLUMNS),
            "reason": (
                "ACS rows run the ASEC immigration stage's cited residual "
                "method through a CPS-named view of native ACS fields "
                "(microcosm#1020): CIT for citizenship, the exact YOEP entry "
                "year for the PEINUSYR bins, POBP for nativity, and HINS3-7 / "
                "COW / ESR / MIL / SSP / SSIP / SCHG for the legal-status "
                "indicators. ACS coverage is at interview (CPS: any time last "
                "year), and the federal-pension, Social Security reason and "
                "housing-subsidy indicators have no ACS field, so the ACS "
                "residual (likely undocumented) pool can only be as large or "
                "larger than the ASEC method would find for the same people. "
                "The Pew worker and Higher Ed student controls are scaled by "
                "the ACS rows' share of staging person weight. "
                "years_since_us_entry is years since entry (age for the "
                "US-born), not years in qualified status, which neither "
                "survey measures (policyengine-us#9658)."
            ),
            "treatment": (
                "Gated by acs_local_immigration_signal; the staging summary's "
                "acs_local_immigration receipt records the mapping, the "
                "scaled controls and an assignment digest."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "acs_work_disability_inputs",
            "status": "reviewed_modeling_decision",
            "affected_spines": ["acs_2024_1yr"],
            "columns": list(ACS_LOCAL_DISABILITY_COLUMNS),
            "reason": (
                "ACS rows read is_disabled and is_blind from their own "
                "disability-difficulty items (microcosm#1021), not the ASEC "
                "transfer: is_disabled is any of DEAR/DEYE/DOUT/DPHY/DREM/DDRS "
                "== 1, or SSIP > 0 under age 65 (the ASEC eligibility-inputs "
                "definition), and is_blind is DEYE == 1. An item below its "
                "minimum question age (5 for DPHY/DREM/DDRS, 15 for DOUT) is "
                "blank and reads as no difficulty. Native ACS weeks_worked "
                "(WKWN, under the WKHP universe rules) is carried on the "
                "staging ACS spine only: the pinned policyengine-us gives "
                "weeks_worked a formula_2025, so the export holds it back as "
                f"formula-owned until {WEEKS_WORKED_EXPORT_BLOCKER}. is_veteran "
                "(ACS MIL == 2) is not exported because policyengine-us "
                "computes it from veterans_benefits for every year; "
                "is_incapable_of_self_care stays on the ASEC transfer."
            ),
            "treatment": (
                "Gated by acs_local_work_disability_signal at staging and "
                "finalize; the staging summary's acs_local_work_disability "
                "receipt records the definitions, item universes, SSI "
                "increment, zero transfer-imputed cells and the weeks-worked "
                "counts."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "acs_local_income_transfer",
            "status": "reviewed_modeling_decision",
            "affected_spines": ["acs_2024_1yr"],
            "columns": list(ACS_LOCAL_INCOME_TRANSFER_COLUMNS),
            "reason": (
                "ACS rows get child support received and paid, workers' "
                "compensation, non-SSA disability benefits and 401(k)/403(b)/"
                "SEP/Keogh/Roth-IRA distributions from a separate local-lane "
                f"QRF pass ({ACS_LOCAL_INCOME_TRANSFER_ISSUE}), outside the "
                "shared declared transfer plan and its execution contract. "
                "The ACS asks none of them separately (OIP and RETP are "
                "combined amounts), so the pass fits on the donor's ASEC "
                "observation role, the measured CPS values, never the PUF "
                "clone role's CPS-trained predictions, with the existing "
                "transfer predictors (ACS RETP among them; OIP is not "
                "loaded). The shared plan already transfers "
                + ", ".join(ACS_LOCAL_INCOME_SHARED_RETIREMENT_COMPONENTS)
                + ", so this pass transfers only the other account types and "
                "refuses any overlap with the shared plan."
            ),
            "treatment": (
                "Gated by acs_local_income_transfer_signal at staging and "
                "finalize (complete, non-negative amounts on both spines, "
                "ACS signal where the donor has recipients, a complete "
                "ASEC-channel receipt); ACS/donor recipient-share and "
                "recipient-mean ratios outside the review band are "
                "reported, not failed."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "acs_local_ssi_disability_criteria",
            "status": "reviewed_modeling_decision",
            "affected_spines": ["acs_2024_1yr"],
            "columns": [ACS_LOCAL_SSI_DISABILITY_COLUMN],
            "reason": (
                "ACS rows get meets_ssi_disability_criteria from the donor "
                "release's archived SIPP QRF (the same pinned full SIPP 2023 "
                "file, screens, predictors, weighted sample and fixed forest), "
                "run on a CPS-named view of each ACS person "
                f"({ACS_LOCAL_SSI_DISABILITY_ISSUE}). The six ACS difficulty "
                "items stand in for the ASEC PEDIS* items, and a blank below "
                "an item's minimum question age reads as no difficulty; age, "
                "sex, wages and marriage are native; interest, dividend and "
                "rental income, liquid assets and Social Security disability "
                "come from the shared transfer, and disability benefits from "
                "the local income pass. ACS persons under 65 who report SSI "
                "(SSIP) are anchored, and the draws are keyed on "
                "acs_2024_1yr:SERIALNO:SPORDER and the build seed. SSI take-up "
                "on ACS rows is then assigned against these criteria "
                "(acs_local_ssi_medicaid_take_up)."
            ),
            "treatment": (
                "Gated by acs_local_ssi_disability_signal at staging and "
                "finalize (a complete boolean on both spines, non-constant with "
                "a positive weighted share among ACS persons aged 18-64, and a "
                "receipt pinned to the SIPP file with no unfilled ACS row); the "
                "ACS/donor 18-64 share ratio outside the review band and "
                "under-65 SSI reporters without the criteria are reported, not "
                "failed. Packaging is blocked while any ACS person meets the "
                "criteria and the run records no SSI take-up handling for ACS "
                "rows (the SSI take-up release block); the recorded "
                "acs_local_ssi_medicaid_take_up assignment is that handling."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "acs_local_ssi_medicaid_take_up",
            "status": "reviewed_modeling_decision",
            "affected_spines": ["acs_2024_1yr"],
            "columns": list(ACS_LOCAL_SSI_MEDICAID_TAKE_UP_COLUMNS),
            "reason": (
                "ACS rows get takes_up_ssi_if_eligible and "
                "takes_up_medicaid_if_eligible from the donor stages' own "
                "methods at materialize "
                f"({ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE}), after a "
                "household-batched engine pre-pass over the ACS households "
                "(December 2024 uncapped_ssi > 0 as the SSI candidates, then "
                "is_medicaid_eligible with the SSI flags assigned). Each "
                "target is the residual of the donor stage's count (the SSA "
                "federal-payment recipients by age band; the CMS December "
                "2024 state enrollment) after the donor rows' pooled "
                "recipients: the donor H5 was assigned against the full "
                "count and pool assembly scaled its weights by 1 - acs_share, "
                "so the stage measures the donor's contribution with the same "
                "two engine passes over the donor households (stored flag x "
                "candidacy or eligibility x pre-calibration person weight) "
                "and targets count minus contribution, floored at zero "
                "(microcosm#1060 review). SSI: ssi_reported (SSIP, asked from "
                "age 15) reporters always take up, as a floor even above the "
                "residual, and everyone else draws at the ssi_take_up band "
                "prior; where the ACS candidate capacity is below the "
                "residual every candidate takes up and the shortfall is "
                "recorded. The under-18 band draws but is fenced from "
                "grading, as on the donor. Medicaid: the medicaid_take_up "
                "manifest stage fills and greedily calibrates each state to "
                "its residual among eligible non-anchored persons. ACS HINS4 "
                "== 1 is the anchor, but HINS4 also covers CHIP and "
                "state-funded means-tested plans, so it is not forced "
                "wholesale: where a state's anchored eligible weight exceeds "
                "its residual, each HINS4 record there stays anchored with "
                "probability residual / anchored eligible weight on a keyed "
                "draw, and the state's excess is recorded. The treatment is "
                "state-level because the CMS counts carry no child/adult "
                "split; policyengine-us 2.2.1 keeps CHIP take-up separate "
                "and CHIP eligibility exclusive of Medicaid eligibility. "
                "Draws are keyed on acs_2024_1yr:SERIALNO:SPORDER and the "
                "build seed. The targets bind the pre-calibration frame; the "
                "release solve does not carry the SSA band counts."
            ),
            "treatment": (
                "Gated by acs_local_ssi_medicaid_take_up_signal at "
                "materialize and finalize: both flags complete and "
                "non-constant on both spines, every ACS SSIP reporter taking "
                "up and every HINS4 anchor outside the thinned states taking "
                "up, every band and state target equal to its count minus "
                "the donor contribution, the 18-64 and 65+ SSI bands within "
                "5% of their residuals wherever the prior was count-truthful "
                "(every candidate assigned where saturated), and the donor "
                "medicaid_take_up gate passing on the ACS state diagnostics "
                "at the residuals. The under-18 band, SSI bands the donor "
                "meets or whose anchors exceed the residual, saturated "
                "Medicaid states and every shortfall and excess are "
                "reported, not failed. The consumer export applies the "
                "recorded assignment and must reproduce its digest."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "acs_spm_unit_adult_nonrelatives",
            "status": "reviewed_modeling_decision",
            "affected_spines": ["acs_2024_1yr"],
            "reason": (
                "The ACS has no SPM unit id, so the loader makes each ACS "
                "housing unit one SPM unit. Staging "
                f"({ACS_LOCAL_SPM_UNIT_ISSUE}) gives every roommate or "
                "housemate (RELSHIPP 34) aged 15 or over an SPM unit of their "
                "own, per the Census SPM unit definition (related people, "
                "co-resident unrelated children cared for by the family, and "
                "cohabiting partners and their children and relatives). An "
                "other nonrelative (36) aged 15 or over gets their own unit "
                "only when the household has no unmarried partner (22 or "
                "24): the ACS records relationship to the householder only, "
                "so a partner's child or relative is coded 36, and with a "
                "partner present a 36 stays with the reference person and "
                "the partner (microcosm#1061 review). RELSHIPP relates each "
                "person to the reference person only, so a couple or a "
                "parent and child among the nonrelatives is not "
                "identifiable: each mover is a one-person unit, and a 36 "
                "moved from a household that also holds a roommate, who may "
                "be the roommate's child, is counted in the staging receipt, "
                "with the unit-size sensitivity under the 34-only and "
                "34-and-every-36 rules. Nonrelatives under 15 stay in the "
                "reference person's unit, whose tax unit claims them. "
                "Unmarried partners (22, 24), foster children (35) and every "
                "relative stay with the reference person. New units take the "
                "household's tenure through SPM membership; the household's "
                "measured housing amounts stay on the reference person, and "
                "the transferred pre_subsidy_rent is a person-level fit. "
                "ACS families remain whole households. Related adults who "
                "buy and prepare food apart are not split: SNAP-unit "
                "construction is option 2 of the same issue."
            ),
            "treatment": (
                "Gated by acs_local_spm_unit_signal at staging and finalize: "
                "no ACS SPM unit holds a mover (a roommate, or an other "
                "nonrelative with no partner in the household) together with "
                "the reference person, no other nonrelative in a partner's "
                "household is outside the reference person's unit, the ACS "
                "partition is exactly the rule, ACS tax units nest in SPM "
                "units, and the staging receipt moved and kept as many "
                "people as the packaged ACS rows hold."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "acs_snap_receipt_anchor",
            "status": "reviewed_modeling_decision",
            "affected_spines": ["acs_2024_1yr"],
            "columns": ["receives_snap", "receives_tanf"],
            "reason": (
                "receives_snap on ACS rows is anchored by native household FS "
                "(anyone in the household received SNAP in the past 12 "
                "months), not left to the QRF transfer, whose predictors carry "
                f"no receipt signal ({ACS_LOCAL_RECEIPT_ANCHOR_ISSUE}). FS "
                "names the housing unit, not the SNAP household, so FS == 1 "
                "is a constraint that at least one SPM unit of the housing "
                "unit received SNAP (microcosm#1062 review): the only SPM "
                "unit of a single-unit housing unit is True; in a housing "
                "unit with several SPM units (microcosm#1023's roommate "
                "units) each unit keeps its transferred QRF value, which is "
                "conditioned on the unit's own size, ages, income and tenure, "
                "and the reference person's unit is set True when the QRF "
                "names none. A roommate who buys and prepares food apart is "
                "therefore not forced to report the family's receipt. Census "
                "prorates a household SNAP amount to every SPM unit (the "
                "donor's SPM_SNAPSUB > 0) to measure poverty resources, which "
                "does not establish that every unit enrolled. FS == 2 marks "
                "none; group quarters, outside the FS universe, are False. FS "
                "under-reports administrative SNAP, so the anchor is a floor "
                "the take-up draw (microcosm#1019) fills to the FNS rate. "
                "receives_tanf keeps its QRF transfer: ACS PAP covers TANF and "
                "general assistance together, and PAP > 0 is not TANF receipt "
                "(microcosm#591); PAP is loaded and its overlap with the "
                "transferred receives_tanf is recorded, not applied, and the "
                "SNAP x TANF cross-tab of ACS units before and after the "
                "override is recorded beside the donor spine's."
            ),
            "treatment": (
                "Applied at staging after the transfer, whose declared plan "
                "is unchanged, and before pooling, so donor rows keep their "
                "ASEC receipt. Gated by acs_local_receipt_anchor_signal at "
                "staging and finalize: no unit of an FS == 2 housing unit or "
                "of group quarters reports SNAP, the only SPM unit of an FS "
                "== 1 housing unit does, every FS == 1 housing unit with "
                "several SPM units has at least one reporting unit, every ACS "
                "housing unit carries FS 1 or 2, and the staging receipt "
                "counts the households, units, anchors, non-reference "
                "reporters, resolution paths and PAP recipients the packaged "
                "ACS rows hold. The "
                "weighted FS == 1 share of ACS housing units is reported "
                "against the 2023 ACS figure (12.2%), not graded."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "cd_population_marginal_vintage_2020",
            "status": "reviewed_vintage",
            "reason": (
                "Congressional-district population marginals use "
                "119th-boundary / 2020-apportionment Census populations from "
                "the PUMA ladder; the build period is 2024. Boundaries are "
                "2020-census-drawn (honest v1)."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "low_effective_sample_size_lambda_zero",
            "status": "reviewed_concentration",
            "reason": (
                f"Kish ESS is {ess} = {ess_fraction:.2%} of {households} "
                "households under the hard 5x weight cap with l2_lambda=0 "
                "(kept for consistency with the certified default and the "
                "Build L doctrine; no new calibration knobs per "
                "microcosm#492)."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "donor_sparse_selection_training_set",
            "status": "reviewed_construction",
            "reason": (
                "The QRF transfer donor is the certified Build O sparse "
                "release "
                f"({donor_release.get('release_id', 'unknown donor id')}): "
                "the 57,240-household certified selection rather than a "
                "dense full-row donor (buildl trained on the June dense "
                "artifact). The conditioning set is thinner; per-fit donor "
                "row counts are recorded in the staging provenance."
            ),
            "calibration_blocker": False,
        },
        {
            "id": "mixed_sub_puma_column_coverage",
            "status": "reviewed_construction",
            "columns": [
                "block_geoid",
                "tract_geoid",
                "cbsa_code",
                "place_fips",
                "sldl",
                "sldu",
            ],
            "reason": (
                "Donor rows keep their certified block-ladder geography "
                "columns; ACS rows carry no sub-PUMA geography, so these "
                "columns are donor-spine-only. Consumers filtering on them "
                "must scope to the asec_puf spine."
            ),
            "calibration_blocker": False,
        },
    ]
    deduped: dict[str, dict] = {}
    for limitation in limitations:
        deduped[limitation["id"]] = limitation
    return list(deduped.values())


def _local_hours_gate(frame, staging_summary: dict):
    audit = staging_summary.get("reviewed_engine_input_nulls")
    if not isinstance(audit, list) or not all(isinstance(item, dict) for item in audit):
        raise SystemExit("Local hours gate requires the staging input-null audit.")
    return acs_local_hours_signal_gate(frame, source_null_audit=audit)


def _require_local_hours(frame, staging_summary: dict) -> None:
    gate = _local_hours_gate(frame, staging_summary)
    if not gate.passed:
        raise SystemExit("Local hours coverage failed: " + "; ".join(gate.failures))


def _require_local_immigration(staging_summary: dict) -> dict:
    """The staging immigration receipt (microcosm#1020), or refuse the staging.

    Staging runs the ACS local immigration stage and its gate before writing
    the H5. A summary without both is a pre-#1020 staging run, whose ACS
    persons would all be default-filled citizens with a valid SSN and whose
    entry clocks would all be the engine default of 5 years.
    """

    receipt = staging_summary.get("acs_local_immigration")
    gate = staging_summary.get("acs_local_immigration_gate")
    if (
        not isinstance(receipt, dict)
        or receipt.get("issue") != ACS_LOCAL_IMMIGRATION_ISSUE
        or not isinstance(receipt.get("assigned_sha256"), str)
        or not isinstance(gate, dict)
        or gate.get("passed") is not True
    ):
        raise SystemExit(
            "The staging summary records no passing ACS local immigration stage "
            f"({ACS_LOCAL_IMMIGRATION_ISSUE}): every ACS person would reach the "
            "engine as a citizen with a valid SSN and every years_since_us_entry "
            "as the engine default. Re-run staging "
            "(tools/build_us_acs_multispine_base.py) with the current builder."
        )
    return receipt


def _require_local_work_disability(staging_summary: dict) -> dict:
    """The staging work/disability receipt (microcosm#1021), or refuse it.

    Staging maps ACS ``is_disabled``/``is_blind``/``weeks_worked`` natively
    before the transfer and gates them before writing the H5. A summary
    without a passing receipt and gate is a pre-#1021 staging run, whose ACS
    disability flags were imputed from ASEC donors and whose ACS rows carry
    no weeks worked.
    """

    receipt = staging_summary.get("acs_local_work_disability")
    gate = staging_summary.get("acs_local_work_disability_gate")
    native = isinstance(receipt, dict) and all(
        isinstance(receipt.get(column), dict)
        and receipt[column].get("source") == ACS_NATIVE_PROVENANCE
        and type(receipt[column].get("imputed_rows")) is int
        and receipt[column]["imputed_rows"] == 0
        for column in ACS_LOCAL_DISABILITY_COLUMNS
    )
    if (
        not native
        or receipt.get("issue") != ACS_LOCAL_WORK_DISABILITY_ISSUE
        or not isinstance(gate, dict)
        or gate.get("passed") is not True
    ):
        raise SystemExit(
            "The staging summary records no passing native ACS work/disability "
            f"stage ({ACS_LOCAL_WORK_DISABILITY_ISSUE}): the ACS rows' "
            "is_disabled and is_blind would be ASEC-imputed rather than read "
            "from the ACS difficulty items, and weeks_worked would be absent. "
            "Re-run staging (tools/build_us_acs_multispine_base.py) with the "
            "current builder."
        )
    return receipt


def _require_local_income_transfer(staging_summary: dict) -> dict:
    """The staging income-transfer receipt (microcosm#1022), or refuse it.

    Staging runs a separate ASEC-channel QRF pass for the SNAP-relevant income
    leaves the shared plan does not carry, and gates them before writing the
    H5. A summary without a passing receipt and gate is a pre-#1022 staging
    run, whose ACS rows reach the reviewed-null fill with no child support,
    workers' compensation, disability benefits or account distributions.
    """

    receipt = staging_summary.get("acs_local_income_transfer")
    gate = staging_summary.get("acs_local_income_transfer_gate")
    columns = receipt.get("columns") if isinstance(receipt, dict) else None
    complete = isinstance(columns, dict) and all(
        isinstance(columns.get(column), dict)
        and type(columns[column].get("imputed_rows")) is int
        and columns[column].get("unmodeled_rows") == 0
        for column in ACS_LOCAL_INCOME_TRANSFER_COLUMNS
    )
    if (
        not complete
        or receipt.get("issue") != ACS_LOCAL_INCOME_TRANSFER_ISSUE
        or receipt.get("donor_channel") != ACS_LOCAL_INCOME_DONOR_CHANNEL
        or not isinstance(gate, dict)
        or gate.get("passed") is not True
    ):
        raise SystemExit(
            "The staging summary records no passing ACS local income transfer "
            f"({ACS_LOCAL_INCOME_TRANSFER_ISSUE}): the ACS rows' child support, "
            "workers' compensation, disability benefits and 401(k)/403(b)/SEP/"
            "Keogh/Roth-IRA distributions would be the engine default 0. Re-run "
            "staging (tools/build_us_acs_multispine_base.py) with the current "
            "builder."
        )
    return receipt


def _require_local_ssi_disability(staging_summary: dict) -> dict:
    """The staging SSI disability-criteria receipt (microcosm#1022), or refuse it.

    Staging runs the donor release's archived SIPP model on the ACS rows and
    gates it before writing the H5. A summary without a passing receipt and
    gate, pinned to the SIPP file, is a pre-change staging run, whose ACS rows
    reach the reviewed-null fill with ``meets_ssi_disability_criteria`` False.
    """

    receipt = staging_summary.get("acs_local_ssi_disability")
    gate = staging_summary.get("acs_local_ssi_disability_gate")
    donor = receipt.get("sipp_donor") if isinstance(receipt, dict) else None
    if (
        not isinstance(receipt, dict)
        or receipt.get("issue") != ACS_LOCAL_SSI_DISABILITY_ISSUE
        or receipt.get("column") != ACS_LOCAL_SSI_DISABILITY_COLUMN
        or receipt.get("method") != ACS_LOCAL_SSI_DISABILITY_METHOD
        or type(receipt.get("filled_rows")) is not int
        or receipt.get("unfilled_acs_rows") != 0
        or not isinstance(donor, dict)
        or donor.get("sha256") != SIPP_2023_SSI_DISABILITY_DONOR_SHA256
        or not isinstance(gate, dict)
        or gate.get("passed") is not True
    ):
        raise SystemExit(
            "The staging summary records no passing ACS local SSI "
            f"disability-criteria stage ({ACS_LOCAL_SSI_DISABILITY_ISSUE}): "
            "every ACS person under 65 who is not blind would fail the SSI "
            "disability test (the engine default False). Re-run staging "
            "(tools/build_us_acs_multispine_base.py) with the current builder "
            "and the pinned SIPP donor."
        )
    return receipt


def _require_local_spm_units(staging_summary: dict) -> dict:
    """The staging ACS SPM-unit receipt (microcosm#1023), or refuse it.

    Staging gives every ACS roommate (``RELSHIPP`` 34) aged 15 or over, and
    every other nonrelative (36) aged 15 or over in a household with no
    unmarried partner (22/24), an SPM unit of their own, per the Census SPM
    unit definition, and gates the partition before writing the H5. A summary
    without a passing receipt and gate, or with an earlier method, is a
    pre-change staging run: whole-household ACS SPM units, or a partner's
    child or relative split from the partner.
    """

    receipt = staging_summary.get("acs_local_spm_units")
    gate = staging_summary.get("acs_local_spm_units_gate")
    counted = (
        isinstance(receipt, dict)
        and type(receipt.get("persons_moved")) is int
        and type(receipt.get("units_created")) is int
        and receipt["persons_moved"] == receipt["units_created"]
    )
    if (
        not counted
        or receipt.get("issue") != ACS_LOCAL_SPM_UNIT_ISSUE
        or receipt.get("method") != ACS_LOCAL_SPM_UNIT_METHOD
        or not isinstance(gate, dict)
        or gate.get("passed") is not True
    ):
        raise SystemExit(
            "The staging summary records no passing ACS SPM-unit split "
            f"({ACS_LOCAL_SPM_UNIT_ISSUE}): every ACS household would be one "
            "SPM unit, with adult roommates and other nonrelatives in the "
            "reference person's unit. Re-run staging "
            "(tools/build_us_acs_multispine_base.py) with the current builder."
        )
    return receipt


def _require_local_receipt_anchors(staging_summary: dict) -> dict:
    """The staging ACS receipt-anchor receipt (microcosm#1022), or refuse it.

    Staging constrains the transferred ``receives_snap`` on ACS SPM units by
    native household ``FS`` (at least one unit of an FS == 1 housing unit,
    none of an FS == 2 one) and gates it before writing the H5. A summary
    without a passing receipt and gate is a pre-change staging run, whose ACS
    SNAP reporters are the QRF transfer's; one with an earlier method marked
    every unit of an FS == 1 housing unit, roommates included.
    """

    receipt = staging_summary.get("acs_local_receipt_anchors")
    gate = staging_summary.get("acs_local_receipt_anchors_gate")
    snap = receipt.get("snap") if isinstance(receipt, dict) else None
    counted = isinstance(snap, dict) and all(
        type(snap.get(key)) is int for key in ("fs_yes_households", "units_anchored")
    )
    if (
        not counted
        or receipt.get("issue") != ACS_LOCAL_RECEIPT_ANCHOR_ISSUE
        or receipt.get("method") != ACS_LOCAL_RECEIPT_ANCHOR_METHOD
        or not isinstance(gate, dict)
        or gate.get("passed") is not True
    ):
        raise SystemExit(
            "The staging summary records no passing native SNAP receipt anchor "
            f"({ACS_LOCAL_RECEIPT_ANCHOR_ISSUE}): the ACS rows' receives_snap "
            "would be the QRF transfer, not household FS. Re-run staging "
            "(tools/build_us_acs_multispine_base.py) with the current builder."
        )
    return receipt


def _recorded_ssi_take_up_handling(identity: dict, checkpoint_dir: Path) -> dict | None:
    """The run's recorded SSI take-up handling for ACS rows, or ``None``.

    The ACS SSI/Medicaid take-up stage assigns ``takes_up_ssi_if_eligible`` on
    ACS rows at materialize (microcosm#1022) and records the assignment in
    ``run_identity.json``. That receipt, with its assignment file unchanged,
    is the handling; a run without it has none. A present receipt that is
    not a passing one, or whose file changed, is refused outright. Packaging
    separately requires the ``acs_local_ssi_medicaid_take_up_signal``
    finalize gate bound to the packaged bytes.
    """

    if "acs_local_ssi_medicaid_take_up" not in identity:
        return None
    receipt, _ = _recorded_ssi_medicaid_take_up(identity, checkpoint_dir)
    return {
        "stage": "acs_local_ssi_medicaid_take_up",
        "issue": receipt["issue"],
        "method": receipt["method"],
        "assigned_sha256": receipt["assigned_sha256"],
        "assignment_file": dict(receipt["assignment_file"]),
        "finalize_gate": ACS_LOCAL_SSI_MEDICAID_TAKE_UP_GATE_NAME,
    }


def _require_ssi_take_up_handling(
    staging_summary: dict, identity: dict, checkpoint_dir: Path
) -> dict:
    """Refuse to package SSI criteria that universal SSI take-up would pay.

    The SSI disability-criteria stage makes ACS persons under 65
    criteria-positive. With ``takes_up_ssi_if_eligible`` at the engine default
    ``True``, every one of them who passes the income and resource tests
    receives SSI, which overstates ACS SSI under 65 and moves SNAP through SSI
    income, the elderly-or-disabled definition and categorical eligibility.
    Packaging is therefore blocked while the staged ACS rows carry any
    criteria-positive person (the staging receipt's ``outcome.acs_true_rows``)
    and the run records no SSI take-up handling (microcosm#1022, review of
    PR #1058). A summary that cannot count those persons is refused too.

    Returns the block's evidence for the build manifest.
    """

    receipt = staging_summary.get("acs_local_ssi_disability")
    outcome = receipt.get("outcome") if isinstance(receipt, dict) else None
    positive = outcome.get("acs_true_rows") if isinstance(outcome, dict) else None
    if type(positive) is not int or positive < 0:
        raise SystemExit(
            "The staging summary's acs_local_ssi_disability receipt records no "
            "count of criteria-positive ACS persons (outcome.acs_true_rows), so "
            "packaging cannot show that no ACS person ships SSI disability "
            f"criteria at universal SSI take-up ({ACS_LOCAL_SSI_DISABILITY_ISSUE})."
            " Re-run staging with the current builder."
        )
    block: dict[str, object] = {
        "criteria_positive_acs_rows": positive,
        "filled_true_rows": outcome.get("filled_true_rows"),
        "take_up_handling": None,
    }
    if positive == 0:
        return block
    handling = _recorded_ssi_take_up_handling(identity, checkpoint_dir)
    if handling is None:
        raise SystemExit(
            f"Refusing to package: {positive:,} ACS person(s) meet "
            f"{ACS_LOCAL_SSI_DISABILITY_COLUMN} after the SSI disability-criteria "
            "stage, but the run records no SSI take-up handling for ACS rows. "
            "takes_up_ssi_if_eligible would ship at the engine default True, so "
            "every one of them who passes the income and resource tests would "
            "receive SSI and ACS SSI under 65 would be overstated "
            f"({ACS_LOCAL_SSI_DISABILITY_ISSUE}). Land the ACS SSI/Medicaid "
            "take-up stage (microcosm PR #1060), then re-run --stage "
            "materialize through --stage finalize before packaging."
        )
    return {**block, "take_up_handling": handling}


def do_finalize(args) -> None:
    from microcosm.build.us_runtime.hours_worked import (
        US_HOURS_WORKED_POOL_OUTPUT_COLUMNS,
        us_hours_worked_signal_gate,
    )
    from microcosm.build.us_runtime.puma_ladder import (
        load_us_puma_ladder,
        us_puma_ladder_gate,
    )
    from microcosm.calibrate import TargetRegistry

    staging_summary = _load_json(_staging_summary_path(args))
    diagnostics = _load_json(args.checkpoint_dir / "calibration_summary.json")
    if not diagnostics:
        raise SystemExit(
            f"No calibration summary under {args.checkpoint_dir}; run "
            "--stage calibrate first."
        )
    identity = _verify_run_identity(args)
    ladder_sha = _sha256(args.ladder)
    if ladder_sha != identity.get("ladder_sha256"):
        raise SystemExit(
            f"--ladder {args.ladder} (sha {ladder_sha[:12]}…) is not the "
            "ladder the surface was materialized with "
            f"({str(identity.get('ladder_sha256'))[:12]}…)."
        )
    materialize_rss = _load_json(args.checkpoint_dir / "materialize_rss.json")
    registry_path = args.checkpoint_dir / "target_registry.json"
    if registry_path.is_file():
        registry = TargetRegistry.from_json(registry_path)
        targets = [
            {"name": spec.name, "family": spec.family} for spec in registry.specs
        ]
    else:
        # Read-only compatibility for checkpoints created before the target
        # registry became the materialize-stage artifact. Current materialize
        # runs always write target_registry.json.
        targets = _load_json(args.checkpoint_dir / "targets.json") or []
    spine_qa = _load_json(args.checkpoint_dir / "spine_qa.json")
    consumer_export = _load_json(args.checkpoint_dir / "consumer_export.json")

    # The hours gate certifies specific artifact bytes. Hash the calibrated
    # H5 before loading it, so the binding the package stage checks is the
    # bytes the gate actually evaluated, and refuse if they moved meanwhile.
    if not args.out_h5.exists():
        raise SystemExit(f"Calibrated H5 not found: {args.out_h5}.")
    hours_artifact_sha = _sha256(args.out_h5)
    frame = _load_staging_frame(args.out_h5)
    local_hours_gate = _local_hours_gate(frame, staging_summary)
    households = frame.table("household")
    weights = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    load_us_puma_ladder(args.ladder)
    ladder_gate = us_puma_ladder_gate(households, weights)
    composition = spine_composition(households, frame.table("person"), weights)
    # microcosm#765: refuse to package an artifact whose usual-weekly-hours
    # surface is the engine's constant-40 default (or otherwise out of band),
    # which silently no-ops SNAP's ABAWD and general work-requirement tests.
    # weeks_worked is dropped from the pool/ACS surface, so scope the gate to
    # the two hours columns it carries.
    hours_gate = us_hours_worked_signal_gate(
        frame, required_columns=US_HOURS_WORKED_POOL_OUTPUT_COLUMNS
    )
    # microcosm#1019: likewise refuse SNAP/TANF take-up that is missing or
    # constant (the engine-default universal take-up) on either spine, or
    # unanchored or out of band on the ACS spine this tool seeds.
    take_up_gate = acs_local_take_up_signal_gate(frame)
    # microcosm#1020: refuse immigration labels or entry clocks that are
    # missing, all-CITIZEN or all at the engine default on either spine,
    # labels that contradict measured ACS CIT citizenship, and a file outside
    # the non-citizen and undocumented-anchor bands.
    immigration_gate = acs_local_immigration_signal_gate(frame)
    # microcosm#1021: refuse ACS disability flags that are missing, constant,
    # or not the native ACS difficulty items on the packaged bytes, or a
    # staging receipt that shows them imputed. weeks_worked is graded at
    # staging only: the export holds it back until policyengine-us#9660.
    work_disability_gate = acs_local_work_disability_signal_gate(
        frame,
        receipt=staging_summary.get("acs_local_work_disability"),
        require_weeks_worked=False,
    )
    # microcosm#1022: refuse transferred ACS income leaves that are missing,
    # negative or carry no signal on the packaged bytes, or a staging receipt
    # that is not the complete ASEC-channel pass.
    income_gate = acs_local_income_transfer_signal_gate(
        frame, receipt=staging_summary.get("acs_local_income_transfer")
    )
    # microcosm#1022: refuse ACS SSI disability criteria that are missing or
    # carry no working-age signal on the packaged bytes, or a staging receipt
    # that is not the complete SIPP-pinned pass.
    ssi_disability_gate = acs_local_ssi_disability_signal_gate(
        frame, receipt=staging_summary.get("acs_local_ssi_disability")
    )
    # microcosm#1022: refuse ACS SSI or Medicaid take-up that is missing,
    # constant or unanchored on the packaged bytes, or a materialize receipt
    # whose enforced SSI bands or Medicaid states miss their donor residuals.
    ssi_medicaid_gate = acs_local_ssi_medicaid_take_up_signal_gate(
        frame, receipt=identity.get("acs_local_ssi_medicaid_take_up")
    )
    # microcosm#1023: refuse ACS SPM units that keep an adult roommate or
    # other nonrelative with the reference person on the packaged bytes, or a
    # staging receipt that did not split them.
    spm_unit_gate = acs_local_spm_unit_signal_gate(
        frame, receipt=staging_summary.get("acs_local_spm_units")
    )
    # microcosm#1022: refuse ACS SNAP reporters that are not the native
    # household FS on the packaged bytes, or a staging receipt that counts
    # other households, units or anchors.
    receipt_anchor_gate = acs_local_receipt_anchor_signal_gate(
        frame, receipt=staging_summary.get("acs_local_receipt_anchors")
    )
    del frame
    gc.collect()
    if _sha256(args.out_h5) != hours_artifact_sha:
        raise SystemExit(
            "The calibrated H5 changed during hours_worked_signal validation. "
            "Re-run --stage qa and --stage finalize against the current artifact."
        )

    breakdown: dict[str, int] = {}
    for target in targets:
        name = target["name"]
        if name.startswith("pop_state"):
            key = "population_state"
        elif name.startswith("pop_cd"):
            key = "population_cd"
        elif name.startswith("usda_snap") and "average_monthly_households" in name:
            key = "snap_caseloads"
        elif name.startswith("usda_snap"):
            key = "snap_benefits"
        elif name.startswith("cms_medicaid"):
            key = "medicaid_enrollment"
        elif name.startswith("irs_soi"):
            key = "soi"
        else:
            key = "other"
        breakdown[key] = breakdown.get(key, 0) + 1

    mass = diagnostics.get("mass_conserved_ratio", 0.0)
    gates = {
        "acs_local_hours_signal": {
            "passed": local_hours_gate.passed,
            "failures": list(local_hours_gate.failures),
            "detail": dict(local_hours_gate.details),
            "artifact_sha256": hours_artifact_sha,
        },
        "us_puma_ladder_gate": {
            "passed": bool(ladder_gate.passed),
            "failures": list(ladder_gate.failures),
            "detail": dict(ladder_gate.details),
        },
        "hours_worked_signal": {
            "passed": bool(hours_gate.passed),
            "failures": list(hours_gate.failures),
            "detail": dict(hours_gate.details),
            "artifact_sha256": hours_artifact_sha,
        },
        ACS_LOCAL_TAKE_UP_GATE_NAME: {
            "passed": bool(take_up_gate.passed),
            "failures": list(take_up_gate.failures),
            "detail": dict(take_up_gate.details),
            "artifact_sha256": hours_artifact_sha,
        },
        ACS_LOCAL_IMMIGRATION_GATE_NAME: {
            "passed": bool(immigration_gate.passed),
            "failures": list(immigration_gate.failures),
            "detail": dict(immigration_gate.details),
            "artifact_sha256": hours_artifact_sha,
        },
        ACS_LOCAL_WORK_DISABILITY_GATE_NAME: {
            "passed": bool(work_disability_gate.passed),
            "failures": list(work_disability_gate.failures),
            "detail": dict(work_disability_gate.details),
            "artifact_sha256": hours_artifact_sha,
        },
        ACS_LOCAL_INCOME_TRANSFER_GATE_NAME: {
            "passed": bool(income_gate.passed),
            "failures": list(income_gate.failures),
            "detail": dict(income_gate.details),
            "artifact_sha256": hours_artifact_sha,
        },
        ACS_LOCAL_SSI_DISABILITY_GATE_NAME: {
            "passed": bool(ssi_disability_gate.passed),
            "failures": list(ssi_disability_gate.failures),
            "detail": dict(ssi_disability_gate.details),
            "artifact_sha256": hours_artifact_sha,
        },
        ACS_LOCAL_SSI_MEDICAID_TAKE_UP_GATE_NAME: {
            "passed": bool(ssi_medicaid_gate.passed),
            "failures": list(ssi_medicaid_gate.failures),
            "detail": dict(ssi_medicaid_gate.details),
            "artifact_sha256": hours_artifact_sha,
        },
        ACS_LOCAL_SPM_UNIT_GATE_NAME: {
            "passed": bool(spm_unit_gate.passed),
            "failures": list(spm_unit_gate.failures),
            "detail": dict(spm_unit_gate.details),
            "artifact_sha256": hours_artifact_sha,
        },
        ACS_LOCAL_RECEIPT_ANCHOR_GATE_NAME: {
            "passed": bool(receipt_anchor_gate.passed),
            "failures": list(receipt_anchor_gate.failures),
            "detail": dict(receipt_anchor_gate.details),
            "artifact_sha256": hours_artifact_sha,
        },
        "calibration": {
            # The cap criterion alone is near-tautological (the solver clips
            # per-row losses at the same cap); the solve must also have
            # actually improved on the design weights and conserved mass.
            # Numeric acceptance thresholds beyond that (within-10% floors,
            # per-target error bars) are maintainer-adjudicated surface
            # policy (#398-class), recorded here rather than invented.
            "passed": bool(
                diagnostics.get("final_loss", 1.0) < args.target_loss_cap
                and diagnostics.get("final_loss", 1.0)
                < diagnostics.get("initial_loss", 0.0)
                and abs(mass - 1.0) < 1e-3
            ),
            "initial_loss": diagnostics.get("initial_loss"),
            "final_loss": diagnostics.get("final_loss"),
            "fraction_within_10pct": diagnostics.get("fraction_within_10pct"),
            "effective_sample_size": diagnostics.get("effective_sample_size"),
            "ess_fraction": diagnostics.get("ess_fraction"),
            "realized_max_weight_ratio": diagnostics.get("realized_max_weight_ratio"),
            "mass_conserved_ratio": mass,
            "n_targets": diagnostics.get("n_targets"),
            "calibrated_surface": {
                "soi_mode": materialize_rss.get("soi_mode"),
                "n_targets": len(targets),
                "n_admin": materialize_rss.get("n_admin"),
                "n_population": materialize_rss.get("n_population"),
                "breakdown": breakdown,
            },
        },
        "input_coverage": {
            # Enforced upstream: the staging driver's donor-coverage gate
            # hard-fails before transfer, so reaching finalize means it held.
            "passed": True,
            "binds": "donor_certified_release",
            "note": (
                "Input coverage is enforced on the certified donor release "
                "before transfer; the ACS spine inherits required inputs via "
                "QRF transfer. ACS native GQ-housing / source-universe nulls "
                "are reviewed_limitations (calibration_blocker: false) and "
                "are NOT re-imposed on the ACS spine. Take-up draws are not "
                "transferred: ACS SNAP/TANF take-up is seeded by this tool "
                "and gated by acs_local_take_up_signal (microcosm#1019). The "
                "same stage fills the ACS rows' discretionary ABAWD exemption "
                "(a cap-based proxy: the donor's seeded 18-64 draw at the "
                "statutory-cap manifest rate, not an observed assignment), "
                "housing-assistance receipt (the transferred housing take-up "
                "flag; False in group quarters), Medicare take-up (native "
                "HINS3), vehicle count (native VEH; 0 in group quarters) and "
                "Head Start take-up (a keyed draw at the donor spine's share "
                "among ages 3-5) without the engine, gated by the same gate "
                f"({ACS_LOCAL_ENGINE_FREE_FILLS_ISSUE}; reviewed limitation "
                "acs_engine_free_default_fills); the ACS vehicle value keeps "
                "its default (reviewed limitation "
                "acs_household_vehicle_value_default), and the other ACS "
                "take-up flags are the reviewed limitation "
                "acs_take_up_engine_defaults. "
                "Immigration "
                "labels are not transferred either: staging derives ACS "
                "ssn_card_type/immigration_status_str from native ACS fields "
                "and years_since_us_entry on both spines, gated by "
                "acs_local_immigration_signal (microcosm#1020; reviewed "
                "limitation acs_immigration_status_method). ACS "
                "is_disabled/is_blind are not transferred: staging reads them "
                "from the native ACS difficulty items (and SSIP) on every ACS "
                "row, gated by acs_local_work_disability_signal "
                "(microcosm#1021; reviewed limitation "
                "acs_work_disability_inputs). Native ACS weeks_worked is "
                "staged but held back from the export until "
                f"{WEEKS_WORKED_EXPORT_BLOCKER}. ACS child support "
                "received/paid, workers' compensation, disability benefits "
                "and 401(k)/403(b)/SEP/Keogh/Roth-IRA distributions come from "
                "a separate local QRF pass on the donor's ASEC observation "
                "role, outside the shared transfer plan, gated by "
                "acs_local_income_transfer_signal "
                f"({ACS_LOCAL_INCOME_TRANSFER_ISSUE}; reviewed limitation "
                "acs_local_income_transfer). ACS "
                "meets_ssi_disability_criteria is not transferred: staging "
                "runs the donor release's archived SIPP SSI disability model "
                "on the ACS rows, with the native ACS difficulty items as its "
                "difficulty predictors, gated by "
                "acs_local_ssi_disability_signal "
                f"({ACS_LOCAL_SSI_DISABILITY_ISSUE}; reviewed limitation "
                "acs_local_ssi_disability_criteria). ACS SSI and Medicaid "
                "take-up are not transferred either: materialize assigns them "
                "on the ACS rows after an engine pre-pass, against the SSA "
                "band and CMS state counts scaled to the ACS rows' weight "
                "share, gated by acs_local_ssi_medicaid_take_up_signal "
                f"({ACS_LOCAL_SSI_MEDICAID_TAKE_UP_ISSUE}; reviewed limitation "
                "acs_local_ssi_medicaid_take_up). ACS SPM units are not the "
                "loader's whole households: staging gives each ACS roommate "
                "and other nonrelative aged 15 or over an SPM unit of their "
                "own before the transfer, gated by acs_local_spm_unit_signal "
                f"({ACS_LOCAL_SPM_UNIT_ISSUE}; reviewed limitation "
                "acs_spm_unit_adult_nonrelatives). ACS receives_snap is not "
                "the transfer's either: staging replaces it with native "
                "household FS on every ACS SPM unit before pooling, gated by "
                "acs_local_receipt_anchor_signal "
                f"({ACS_LOCAL_RECEIPT_ANCHOR_ISSUE}; reviewed limitation "
                "acs_snap_receipt_anchor)."
            ),
        },
        "spine_composition": {
            "passed": True,
            "note": (
                "Recorded evidence for the microcosm#403 weight-composition "
                "signature; not a pass/fail bound."
            ),
            "detail": composition,
        },
    }
    if spine_qa:
        gates["spine_ssi_qa"] = {
            "passed": True,
            "note": spine_qa.get("note"),
            "detail": spine_qa.get("per_spine"),
        }
    # Consumer-loadability: the export applied the engine-pass contract to
    # the artifact bytes, and the QA probe loaded those bytes plain.
    gates["consumer_ready"] = {
        "passed": bool(consumer_export)
        and bool(spine_qa.get("plain_consumption"))
        and spine_qa.get("artifact_sha256") is not None,
        "consumer_export": {
            key: consumer_export.get(key)
            for key in (
                "held_back_total",
                "filled_columns",
                "total_values_filled",
                "staging_sha256",
            )
        }
        if consumer_export
        else None,
        "plain_load_proven_by": "spine_ssi_qa"
        if spine_qa.get("plain_consumption")
        else None,
        "artifact_sha256": spine_qa.get("artifact_sha256"),
    }

    limitations = finalize_reviewed_limitations(staging_summary, diagnostics, spine_qa)
    hard_failures = [
        name
        for name in (
            "us_puma_ladder_gate",
            "hours_worked_signal",
            "acs_local_hours_signal",
            ACS_LOCAL_TAKE_UP_GATE_NAME,
            ACS_LOCAL_IMMIGRATION_GATE_NAME,
            ACS_LOCAL_WORK_DISABILITY_GATE_NAME,
            ACS_LOCAL_INCOME_TRANSFER_GATE_NAME,
            ACS_LOCAL_SSI_DISABILITY_GATE_NAME,
            ACS_LOCAL_SSI_MEDICAID_TAKE_UP_GATE_NAME,
            ACS_LOCAL_SPM_UNIT_GATE_NAME,
            ACS_LOCAL_RECEIPT_ANCHOR_GATE_NAME,
            "calibration",
            "consumer_ready",
        )
        if not gates[name]["passed"]
    ]
    updated_summary = dict(staging_summary)
    updated_summary.update(
        {
            "calibration_applied": True,
            "simulation_ready": not hard_failures,
            "simulation_ready_except_calibration": True,
            "simulation_readiness_blockers": hard_failures,
            "reviewed_limitations": limitations,
            "calibration_diagnostics": {
                key: value for key, value in diagnostics.items() if key != "targets"
            },
        }
    )
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(updated_summary, indent=2))
    gate_report = {
        "gates": gates,
        "reviewed_limitations": limitations,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    args.gate_report.write_text(json.dumps(gate_report, indent=2))
    if hard_failures:
        raise SystemExit(
            f"finalize: hard gate failure(s): {hard_failures} — see {args.gate_report}."
        )
    log(f"finalize stage complete: gates green, {len(limitations)} limitations")


#: Staging settings a packaged build manifest records, so a reader can see how
#: the ACS spine was built without opening the staging summary.
_STAGING_ORCHESTRATION_KEYS = (
    "max_households",
    "n_estimators",
    "max_targets_per_fit",
    "acs_share",
    "chunksize",
    "seed",
    "geography_seed",
    "donor_channel",
)


def _require_uncapped_staging(staging_summary: dict) -> dict[str, object]:
    """Refuse to package a staging run that was capped, or cannot show it was not.

    ``tools/build_us_acs_multispine_base.py --max-households`` caps the ACS
    spine for smokes. The donor spine keeps every state and congressional
    district populated, so a capped run drops no ladder population cell and
    passes every finalize gate; without this check it packages into a release
    directory with a publish command and nothing in either manifest says it
    is a smoke. A summary that does not record the cap cannot establish that
    the spine is whole, so it is refused too.
    """

    orchestration = staging_summary.get("orchestration")
    if not isinstance(orchestration, dict) or "max_households" not in orchestration:
        raise SystemExit(
            "The staging summary does not record orchestration.max_households, "
            "so packaging cannot establish that the ACS spine is uncapped. "
            "Re-run staging with the current builder; a smoke's output must "
            "not be packaged."
        )
    cap = orchestration["max_households"]
    if cap is not None:
        raise SystemExit(
            f"The staging run was capped at {cap} ACS household(s) "
            "(--max-households); a smoke's output must not be packaged."
        )
    return {key: orchestration.get(key) for key in _STAGING_ORCHESTRATION_KEYS}


def _require_recorded_soi_mode(materialize_rss: dict) -> str:
    """The SOI surface the checkpoint was materialized with, or refuse.

    Later stages never re-select targets, so the mode that counts is the one
    ``--stage materialize`` recorded, not this invocation's ``--soi-mode``.
    A checkpoint that does not record a known mode cannot say which surface
    was calibrated, and its refresh recipe could not reproduce it.
    """

    soi_mode = materialize_rss.get("soi_mode")
    if soi_mode not in SOI_MODES:
        raise SystemExit(
            f"materialize_rss.json records soi_mode={soi_mode!r}, not one of "
            f"{SOI_MODES}; packaging cannot record which SOI surface was "
            "calibrated. Re-run --stage materialize with the current tool."
        )
    return soi_mode


def _require_stored_inputs(calibrated_h5: Path) -> dict[str, object]:
    """Refuse an artifact that stores a model input the installed engine lacks.

    microcosm#1026: the ACS local-area release of 2026-09-23 records
    policyengine-us 2.2.1 as its built-with engine but stores the WIC take-up
    draw under its retired name, ``would_claim_wic``, which 2.2.1 ignores.
    :mod:`microcosm.data.stored_inputs` owns the rule and the reviewed register
    of deliberately non-variable columns. The engine is the installed
    policyengine-us, the one :func:`do_package` records as
    ``build.built_with_model_package``. Only HDF metadata is read.

    Returns the gate entry the release's gate summary records.
    """

    from microcosm.data import stored_inputs

    try:
        engine = stored_inputs.installed_us_engine()
    except ImportError as error:
        raise SystemExit(
            "Refusing to package: the installed policyengine-us cannot be "
            "imported, so the artifact cannot be checked against the engine "
            f"the release records as built-with: {error}"
        ) from error
    try:
        summary = stored_inputs.require_h5_stored_inputs(calibrated_h5, engine=engine)
    except (
        stored_inputs.StoredInputRefusalError,
        stored_inputs.StoredTableLayoutError,
    ) as error:
        raise SystemExit(f"Refusing to package: {error}") from error
    return {"passed": True, "failures": [], "engine": engine.label, **summary}


def _require_bound_finalize_gate(gates: object, name: str, h5_sha: str) -> None:
    """Refuse packaging unless the finalize report's ``name`` gate passed on
    exactly the H5 bytes being packaged.

    A report finalized before the gate existed, a failed gate, or one bound to
    other bytes (a recalibrate without re-running finalize) cannot vouch for
    the packaged surface.
    """

    gate = gates.get(name) if isinstance(gates, dict) else None
    if (
        not isinstance(gate, dict)
        or gate.get("passed") is not True
        or gate.get("artifact_sha256") != h5_sha
    ):
        raise SystemExit(
            f"Packaging requires a present, passing {name} gate bound to the "
            "packaged H5; an old simulation_ready summary is insufficient. "
            "Re-run --stage finalize against the current artifact."
        )


def do_package(args) -> dict:
    diagnostics = _load_json(args.checkpoint_dir / "calibration_summary.json")
    diagnostics_status = diagnostics.get("calibration_diagnostics")
    if not isinstance(diagnostics_status, dict) or diagnostics_status.get(
        "status"
    ) not in {"available", "failed"}:
        raise SystemExit(
            "calibration_summary.json has no valid calibration_diagnostics status; "
            "run --stage calibrate with the current builder."
        )
    gate_report = _load_json(args.gate_report)
    staging_summary = _load_json(_staging_summary_path(args))
    final_summary = _load_json(args.out_summary)
    held_back = _load_json(args.checkpoint_dir / "held_back_columns.json")
    null_fills = _load_json(args.checkpoint_dir / "reviewed_null_fills.json")
    materialize_rss = _load_json(args.checkpoint_dir / "materialize_rss.json")
    spine_qa = _load_json(args.checkpoint_dir / "spine_qa.json")
    consumer_export = _load_json(args.checkpoint_dir / "consumer_export.json")
    identity = _verify_run_identity(args)
    if not gate_report:
        raise SystemExit("No gate report; run --stage finalize first.")
    if not final_summary.get("simulation_ready"):
        raise SystemExit(
            "Refusing to package: the finalized summary is not simulation_ready."
        )
    # Before any release directory exists: a refused smoke leaves nothing.
    staging_orchestration = _require_uncapped_staging(staging_summary)
    soi_mode = _require_recorded_soi_mode(materialize_rss)
    calibrated_h5 = Path(args.out_h5)
    if not calibrated_h5.exists():
        raise SystemExit(f"Calibrated H5 not found: {calibrated_h5}.")
    # microcosm#1026: the manifest below records the installed policyengine-us
    # as build.built_with_model_package, so the artifact may store no model
    # input that engine does not define. A refused artifact leaves nothing.
    stored_inputs_gate = _require_stored_inputs(calibrated_h5)
    # microcosm#1022 (review of #1058): ACS persons the SSI disability-criteria
    # stage made criteria-positive must not ship at universal SSI take-up.
    ssi_take_up_block = _require_ssi_take_up_handling(
        staging_summary, identity, args.checkpoint_dir
    )

    code = _repo_code_identity(args.allow_dirty)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    release_id = f"{RELEASE_ID_PREFIX}-{code['sha']}-{timestamp}"
    release_dir = args.out / "releases" / release_id
    release_dir.mkdir(parents=True, exist_ok=True)

    log("hashing calibrated H5 …")
    h5_sha = _sha256(calibrated_h5)
    # The gate report certifies specific artifact bytes: the QA probe
    # recorded the sha it loaded plain. Packaging different bytes (a
    # recalibrate without re-running qa+finalize) is refused.
    # Package-time evidence is REQUIRED, not conditional: absent files must
    # never read as vacuously green (a stale finalized summary could
    # otherwise pair with deleted or never-produced evidence).
    if not spine_qa:
        raise SystemExit(
            "spine_qa.json is missing or empty; the packaged bytes have no "
            "plain-consumption certification. Run --stage qa (then "
            "finalize) before packaging."
        )
    if not consumer_export:
        raise SystemExit(
            "consumer_export.json is missing or empty; the artifact's "
            "engine-pass export evidence is required. Run --stage calibrate "
            "before packaging."
        )
    qa_sha = spine_qa.get("artifact_sha256")
    if qa_sha is None:
        raise SystemExit(
            "spine_qa.json carries no artifact_sha256; the gate report "
            "cannot be bound to these bytes. Re-run --stage qa."
        )
    if qa_sha != h5_sha:
        raise SystemExit(
            "The calibrated H5 does not match the bytes the QA probe "
            f"certified ({h5_sha[:12]}… vs {str(qa_sha)[:12]}…). Re-run "
            "--stage qa and --stage finalize against the current artifact."
        )
    gates = gate_report.get("gates")
    hours_gate = gates.get("hours_worked_signal") if isinstance(gates, dict) else None
    if not isinstance(hours_gate, dict) or hours_gate.get("passed") is not True:
        raise SystemExit(
            "Packaging requires a present, passing hours_worked_signal gate; "
            "an old simulation_ready summary is insufficient. Re-run --stage finalize."
        )
    if hours_gate.get("artifact_sha256") != h5_sha:
        raise SystemExit(
            "The hours_worked_signal gate is missing its artifact binding or "
            "certifies different H5 bytes. Re-run --stage finalize against the "
            "current artifact."
        )
    # microcosm#1019: a report finalized before the take-up gate existed (or
    # against other bytes) cannot vouch for the packaged ACS take-up surface.
    _require_bound_finalize_gate(gates, ACS_LOCAL_TAKE_UP_GATE_NAME, h5_sha)
    # microcosm#1020: nor, before the immigration gate existed, for the
    # packaged immigration labels and years_since_us_entry clock.
    _require_bound_finalize_gate(gates, ACS_LOCAL_IMMIGRATION_GATE_NAME, h5_sha)
    # microcosm#1021: nor, before the work/disability gate existed, for the
    # packaged native ACS disability flags.
    _require_bound_finalize_gate(gates, ACS_LOCAL_WORK_DISABILITY_GATE_NAME, h5_sha)
    # microcosm#1022: nor, before the income-transfer gate existed, for the
    # packaged ACS child support, disability and distribution income.
    _require_bound_finalize_gate(gates, ACS_LOCAL_INCOME_TRANSFER_GATE_NAME, h5_sha)
    # microcosm#1022: nor, before the SSI disability gate existed, for the
    # packaged ACS SSI disability criteria.
    _require_bound_finalize_gate(gates, ACS_LOCAL_SSI_DISABILITY_GATE_NAME, h5_sha)
    # microcosm#1022: nor, before the SSI/Medicaid take-up gate existed, for
    # the packaged ACS SSI and Medicaid take-up.
    _require_bound_finalize_gate(
        gates, ACS_LOCAL_SSI_MEDICAID_TAKE_UP_GATE_NAME, h5_sha
    )
    # microcosm#1023: nor, before the SPM-unit gate existed, for the packaged
    # ACS SPM units.
    _require_bound_finalize_gate(gates, ACS_LOCAL_SPM_UNIT_GATE_NAME, h5_sha)
    # microcosm#1022: nor, before the receipt-anchor gate existed, for the
    # packaged ACS SNAP reporters.
    _require_bound_finalize_gate(gates, ACS_LOCAL_RECEIPT_ANCHOR_GATE_NAME, h5_sha)
    # Old summaries can say simulation_ready despite #765. Recheck the
    # actual artifact and the source-null evidence before packaging it, and
    # bind that result to the bytes being packaged: the finalize-time report
    # is copied into the release, so a checkpoint finalized before this gate
    # existed must not ship as if it had passed it.
    finalize_hours = gate_report.get("gates", {}).get("acs_local_hours_signal")
    if not isinstance(finalize_hours, dict) or finalize_hours.get("passed") is not True:
        raise SystemExit(
            "The finalize gate report carries no passing acs_local_hours_signal; "
            "an old simulation_ready summary is insufficient. Re-run --stage "
            "finalize."
        )
    hours_frame = _load_staging_frame(calibrated_h5)
    package_hours_gate = _local_hours_gate(hours_frame, staging_summary)
    del hours_frame
    gc.collect()
    if not package_hours_gate.passed:
        raise SystemExit(
            "Local hours coverage failed: " + "; ".join(package_hours_gate.failures)
        )
    gate_report = {
        **gate_report,
        "gates": {
            **gate_report.get("gates", {}),
            "acs_local_hours_signal": {
                "passed": True,
                "failures": [],
                "detail": dict(package_hours_gate.details),
                "artifact_sha256": h5_sha,
                "checked_at_stage": "package",
            },
            "stored_inputs": {
                **stored_inputs_gate,
                "artifact_sha256": h5_sha,
                "checked_at_stage": "package",
            },
        },
    }
    dropped_cells = identity.get("population_cells_dropped") or []
    if dropped_cells:
        raise SystemExit(
            f"The materialized surface dropped {len(dropped_cells)} ladder "
            "population cell(s); a release never ships a shrunken surface "
            "(--allow-partial-geography is for capped smokes only, and a "
            "smoke's output must not be packaged)."
        )

    def _version(package: str) -> str:
        try:
            from importlib.metadata import version

            return version(package)
        except Exception:
            return "unknown"

    donor_release = (staging_summary.get("base") or {}).get("donor_release")
    refresh_recipe = {
        "note": (
            "When the next certified default publishes (e.g. the microcosm#508 "
            "SSI fix -> O-2), refresh this artifact by re-running the chain "
            "against the new certified release H5; everything else is "
            "unchanged."
        ),
        "staging": LEGACY_STAGING_REFRESH_RECIPE,
        "release": release_refresh_recipe(soi_mode),
        "publish": (
            "tools/publish_release.sh <release_dir> --no-latest "
            f"--artifact-root <run> --repo-id {HF_REPO_ID}"
        ),
    }

    build_manifest = {
        "build_id": release_id,
        "build_sha": code["sha"],
        "build_dirty": code["dirty"],
        "created_at": datetime.now(UTC).isoformat(),
        "code": {"branch": code["branch"], "repo": "microcosm"},
        "runtime": {
            "policyengine_core": _version("policyengine-core"),
            "policyengine_us": _version("policyengine-us"),
            "python": sys.version.split()[0],
        },
        "dataset": {
            "kind": "acs_local_area_multispine",
            "staging_h5_sha256": (staging_summary.get("output", {}) or {}).get(
                "sha256"
            ),
            "households": diagnostics.get("households"),
            "period": PERIOD,
            "spines": sorted((staging_summary.get("spine_totals", {}) or {}).keys())
            or ["acs_2024_1yr", "asec_puf"],
            "donor_release": donor_release,
        },
        "calibration": {
            key: diagnostics.get(key)
            for key in (
                "families",
                "geographies",
                "n_targets",
                "epochs",
                "max_weight_ratio",
                "l2_lambda",
                "seed",
                "final_loss",
                "fraction_within_10pct",
                "effective_sample_size",
                "ess_fraction",
                "realized_max_weight_ratio",
                "mass_conserved_ratio",
            )
        },
        "materialize": {
            "soi_mode": soi_mode,
            "peak_rss_gb": materialize_rss.get("materialize_peak_rss_gb"),
            "hh_chunk": materialize_rss.get("hh_chunk"),
            "engine_pass": (
                "input-schema projection (fed==input) + reviewed-null fill"
            ),
            "held_back_formula_owned": held_back.get("total"),
            "reviewed_null_columns_filled": null_fills.get("columns_filled"),
        },
        "gates": gate_report.get("gates", {}),
        "run_identity": identity,
        "staging_orchestration": staging_orchestration,
        "ssi_take_up_release_block": ssi_take_up_block,
        "refresh_recipe": refresh_recipe,
    }

    source_coverage = {
        "schema_version": 1,
        "note": "ACS local-area artifact coverage summary.",
        "acs_sources": staging_summary.get("acs_sources"),
        "geography_ladder": staging_summary.get("geography_ladder"),
        "transfer_coverage": staging_summary.get("transfer_coverage"),
        "donor_release": donor_release,
        "input_coverage_gate": gate_report.get("gates", {}).get("input_coverage"),
    }

    contract_files = {
        "build_manifest.json": build_manifest,
        "us_source_coverage.json": source_coverage,
        "gate_summary.json": gate_report,
        "held_back_columns.json": held_back,
        "reviewed_null_fills.json": null_fills,
        "run_identity.json": identity,
    }
    contract_files["spine_qa.json"] = spine_qa
    contract_files["consumer_export.json"] = consumer_export
    consumer_fills = _load_json(
        args.checkpoint_dir / "consumer_reviewed_null_fills.json"
    )
    if not consumer_fills:
        raise SystemExit(
            "The per-column consumer fill ledger "
            "(consumer_reviewed_null_fills.json) is missing; the published "
            "fills must ship with their evidence."
        )
    contract_files["consumer_reviewed_null_fills.json"] = consumer_fills
    for name, payload in contract_files.items():
        (release_dir / name).write_text(json.dumps(payload, indent=1))
    if diagnostics_status["status"] == "available":
        source_diagnostics = args.checkpoint_dir / "calibration_diagnostics.json"
        if not source_diagnostics.is_file():
            raise SystemExit(
                "calibration_summary.json declares available diagnostics but "
                "calibration_diagnostics.json is missing."
            )
        if _sha256(source_diagnostics) != diagnostics_status.get("sha256"):
            raise SystemExit(
                "calibration_diagnostics.json no longer matches the validated "
                "digest recorded by the calibration stage."
            )
        shutil.copy2(source_diagnostics, release_dir / "calibration_diagnostics.json")

    def _artifact(path_name: str, kind: str, local: Path) -> dict:
        return {
            "kind": kind,
            "path": path_name,
            "repo_id": HF_REPO_ID,
            "revision": release_id,
            "sha256": _sha256(local),
        }

    artifacts = {
        ARTIFACT_NAME: {
            "kind": "microdata",
            "path": ARTIFACT_FILENAME,
            "repo_id": HF_REPO_ID,
            "revision": release_id,
            "sha256": h5_sha,
        },
        "gate_summary": _artifact(
            "gate_summary.json",
            "diagnostics",
            release_dir / "gate_summary.json",
        ),
        "us_source_coverage": _artifact(
            "us_source_coverage.json",
            "diagnostics",
            release_dir / "us_source_coverage.json",
        ),
    }
    if diagnostics_status["status"] == "available":
        artifacts["calibration_diagnostics"] = _artifact(
            "calibration_diagnostics.json",
            "diagnostics",
            release_dir / "calibration_diagnostics.json",
        )

    release_manifest = {
        "schema_version": 1,
        "data_package": {"name": "microcosm-data", "version": "0.1.0"},
        # NON-DEFAULT: this artifact claims NO default dataset slot.
        # Publishing it must never flip latest.json; it is discoverable by
        # its immutable release id / tag only.
        "default_datasets": {},
        "dataset_role": "non_default_local_area",
        "is_default": False,
        "namespace": RELEASE_NAMESPACE,
        "build": {
            "build_id": release_id,
            "built_at": datetime.now(UTC).isoformat(),
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": _version("policyengine-core"),
            },
            "built_with_model_package": {
                "name": "policyengine-us",
                "version": _version("policyengine-us"),
            },
        },
        "calibration_diagnostics": diagnostics_status,
        "artifacts": artifacts,
        "reviewed_limitations": gate_report.get("reviewed_limitations", []),
        "donor_release": donor_release,
        "refresh_recipe": refresh_recipe,
    }
    (release_dir / "release_manifest.json").write_text(
        json.dumps(release_manifest, indent=1)
    )

    sums = {path.name: _sha256(path) for path in sorted(release_dir.glob("*.json"))}
    sums[ARTIFACT_FILENAME] = h5_sha
    (release_dir / "sha256sums.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(sums.items()))
    )

    artifact_root = args.out
    root_copy = artifact_root / ARTIFACT_FILENAME
    if root_copy.resolve() != calibrated_h5.resolve():
        if not root_copy.exists() or _sha256(root_copy) != h5_sha:
            log(f"copying calibrated H5 to artifact root {root_copy} …")
            shutil.copy2(calibrated_h5, root_copy)
    # Check the actual final artifact, including a reused copy or the no-copy
    # path. A changed source/copy must not inherit the earlier hours verdict.
    if _sha256(root_copy) != h5_sha:
        if root_copy.resolve() != calibrated_h5.resolve():
            # A refused copy must not sit where a good package puts its artifact.
            root_copy.unlink(missing_ok=True)
        raise SystemExit(
            "The packaged H5 no longer matches the hours_worked_signal artifact "
            "binding. Re-run --stage qa and --stage finalize against stable bytes."
        )

    result = {
        "release_id": release_id,
        "release_dir": str(release_dir),
        "artifact_root": str(artifact_root),
        "root_artifact": {
            "name": ARTIFACT_FILENAME,
            "local_path": str(root_copy),
            "sha256": h5_sha,
        },
        "is_default": False,
        "files": sorted(path.name for path in release_dir.iterdir()),
        "publish_command": (
            f"tools/publish_release.sh {release_dir} --no-latest "
            f"--artifact-root {artifact_root} --repo-id {HF_REPO_ID}"
        ),
    }
    (args.out / "package_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["materialize", "calibrate", "qa", "finalize", "package", "all"],
        default="all",
    )
    parser.add_argument("--staging-h5", type=Path, required=True)
    parser.add_argument(
        "--staging-summary",
        type=Path,
        default=None,
        help="Staging summary JSON (default: <staging-h5>.summary.json).",
    )
    parser.add_argument("--feed", type=Path)
    parser.add_argument(
        "--feed-sha256",
        help="Expected ledger-feed sha256; materialize refuses a mismatch.",
    )
    parser.add_argument(
        "--ladder",
        type=Path,
        default=_REPO_ROOT / "build" / "us" / "us_puma_ladder_2020.npz",
    )
    parser.add_argument("--families", default="snap,medicaid,soi")
    parser.add_argument("--geographies", default="state,cd")
    parser.add_argument(
        "--soi-mode",
        choices=SOI_MODES,
        default=DEFAULT_SOI_MODE,
        help=(
            "State SOI target surface for --stage materialize. 'state' "
            "(default) is Build O's contract: every state-geography SOI spec "
            "outside the congressional-district file. 'totals' drops every "
            "soi_fiscal_distribution spec (no state AGI, income-tax or EITC "
            "total); 'full' keeps every state-bearing spec, including the "
            "district file, and needs a much larger dense admin matrix "
            "(contents and sizes in docs/us-acs-local-soi-target-surface.md). "
            "Later stages use the mode the checkpoint recorded."
        ),
    )
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--epoch-batch", type=int, default=400)
    parser.add_argument("--max-weight-ratio", type=float, default=5.0)
    parser.add_argument("--target-loss-cap", type=float, default=1.0)
    parser.add_argument("--l2-lambda", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch", type=int, default=5_000)
    parser.add_argument(
        "--hh-chunk",
        type=int,
        default=40_000,
        help=(
            "Households per engine chunk; bounds the pe-core calc-cache "
            "closure (the 1.6M single-shot jetsam class)."
        ),
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--out-h5",
        type=Path,
        help="Calibrated artifact H5 (staging copy + calibrated weights).",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=None,
        help="Finalized summary path (default: <out-h5>.summary.json).",
    )
    parser.add_argument(
        "--gate-report",
        type=Path,
        default=None,
        help="Gate report path (default: <checkpoint-dir>/gate_summary.json).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Release root for --stage package (releases/<id>/ lands here).",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--allow-partial-geography",
        action="store_true",
        help=(
            "Permit ladder population cells with no supporting household "
            "(a capped smoke shrinks the surface). Never for a release "
            "build: materialize hard-fails on dropped cells without this."
        ),
    )
    args = parser.parse_args(argv)

    stages = (
        ["materialize", "calibrate", "qa", "finalize", "package"]
        if args.stage == "all"
        else [args.stage]
    )
    if "materialize" in stages and args.feed is None:
        parser.error("--feed is required for the materialize stage.")
    if {"calibrate", "qa", "finalize", "package"} & set(stages) and (
        args.out_h5 is None
    ):
        parser.error("--out-h5 is required for calibrate/qa/finalize/package.")
    if "package" in stages and args.out is None:
        parser.error("--out is required for the package stage.")
    if args.out_h5 is not None and args.out_summary is None:
        args.out_summary = args.out_h5.with_suffix(".summary.json")
    if args.gate_report is None:
        args.gate_report = args.checkpoint_dir / "gate_summary.json"
    args.stages = stages
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sampler = RssSampler()
    sampler.start()
    started = time.time()
    try:
        if "materialize" in args.stages:
            do_materialize(args)
        if "calibrate" in args.stages:
            do_calibrate(args)
        if "qa" in args.stages:
            do_qa(args)
        if "finalize" in args.stages:
            do_finalize(args)
        if "package" in args.stages:
            do_package(args)
    finally:
        sampler.stop()
        sampler.join(timeout=5)
        (args.checkpoint_dir / "rss_samples.json").write_text(
            json.dumps(sampler.samples[-2000:])
        )
    log(f"DONE stages={args.stages} ({time.time() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
