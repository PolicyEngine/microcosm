"""US stacked-spine pilot: one origin-labeled spine (populace#578 revision).

The ratified #578 increment-2 revision removes the two-spine / two-pipeline
agreement seam instead of patching it: ASEC and a seeded ACS household sample
are assembled into ONE spine whose rows carry their origin in the
receipt-validated support-channel columns.  Survey-specific fields are then
gap-filled cross-origin with native predictors, a single PUF pass runs after
gap-fill, and by-origin statistics replace spine-vs-spine agreement.

This module is a source-spine provenance OWNER (see the reviewed allowlist in
``test_us_spine_blindness.py``): stacking, gap-fill donor routing, activation
authority, the pre-simulation completeness gate, and the by-origin battery are
exactly the surfaces that must read origin labels.  Population operators stay
spine-blind; this module selects donors and verifies activation authority so
they never have to.

Weight harmonization (the two-arm P-lineage precedent)
------------------------------------------------------
Both origins jointly represent the same population once, exactly like the
ASEC/PUF support channels in :mod:`populace.build.us_runtime.puf_support`
(each channel receives a declared share of the incoming mass so the population
does not double).  :func:`~populace.build.us_runtime.spine_assembly.assemble_spines`
implements the allocation: for a household ``i`` of arm ``s`` with incoming
weight ``w_i``,

    ``w_i' = w_i * share_s * M_anchor / M_s``

where ``M_s`` is arm ``s``'s incoming household mass and ``M_anchor`` is the
mass-anchor arm's incoming mass.  For the seeded ACS sample,
``M_acs_sample ~= fraction * M_acs_full``, so the allocation factor contains
the inverse-sampling upweighting ``1 / fraction`` automatically; the realized
per-arm scale factors are receipted rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from populace.build.gates import FitWeightRecord, GateResult
from populace.build.us_runtime.acs_transfer import (
    DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    AcsTransferResult,
    AcsTransferTargetBank,
    TargetFamilies,
    declared_acs_transfer_target_families,
    transfer_acs_inputs,
)
from populace.build.us_runtime.puf_support import (
    PUF_ABSENT_CELLS_PRESERVE_NULLS,
    clone_us_frame_for_puf_support,
    impute_us_puf_tax_detail_support,
    validate_puf_clone_attachment,
)
from populace.build.us_runtime.spine_assembly import assemble_spines
from populace.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    SPINE_ASSEMBLY_MANIFEST_KEY,
    spine_source_id_column,
    support_channel_column,
    support_clone_index_column,
    validate_assembly_provenance,
)
from populace.frame import US_SCHEMA, Frame

__all__ = [
    "ACS_STACKED_SUPPORT_CHANNEL",
    "DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES",
    "ORIGIN_BATTERY_METRIC_KINDS",
    "STACKED_PILOT_ACS_SAMPLE_FRACTION",
    "STACKED_PILOT_ACS_SAMPLE_SEED",
    "STACKED_SPINE_MANIFEST_KEY",
    "AbsenceProof",
    "GapFillDirection",
    "GapFillResult",
    "OriginBatterySpec",
    "StackedPufPassResult",
    "StackedSpineResult",
    "assemble_stacked_spine",
    "by_origin_battery",
    "gap_fill_stacked_spine",
    "run_stacked_puf_pass",
    "sample_acs_households",
    "stacked_completeness_gate",
    "stacked_gap_fill_plan",
    "validate_stacked_spine_frame",
]

ACS_STACKED_SUPPORT_CHANNEL = "acs"

#: Fixed arm shares for the pilot stack, matching the two-arm P-lineage
#: precedent (the ASEC/PUF support expansion splits incoming mass in half so
#: two arms jointly represent the population once).  Calibration remains
#: downstream.
DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES: Mapping[str, float] = {
    BASE_ASEC_SUPPORT_CHANNEL: 0.5,
    ACS_STACKED_SUPPORT_CHANNEL: 0.5,
}

STACKED_SPINE_MANIFEST_KEY = "us_stacked_spine_manifest"
_STACKED_SPINE_MANIFEST_VERSION = 1
_EXACT_COUNT_RULE = "floor(fraction * eligible)"
_MASS_RTOL = 1e-9

#: The ratified pilot stack configuration (#578 revision): a seeded 10% ACS
#: household sample enters the spine.  Scale-up beyond the pilot changes this
#: declared fraction, never an implicit default.
STACKED_PILOT_ACS_SAMPLE_FRACTION = 0.10
STACKED_PILOT_ACS_SAMPLE_SEED = 578


@dataclass(frozen=True)
class StackedSpineResult:
    """One stacked spine plus its manifest-ready stack receipt."""

    frame: Frame
    receipt: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.frame, Frame):
            raise TypeError(
                "StackedSpineResult.frame must be a Frame, got "
                f"{type(self.frame).__name__}."
            )
        if not isinstance(self.receipt, Mapping):
            raise TypeError("StackedSpineResult.receipt must be a mapping.")


def sample_acs_households(
    acs: Frame,
    *,
    fraction: float,
    seed: int,
) -> tuple[Frame, dict[str, object]]:
    """Draw a seeded, whole-household ACS sample with an exact-count receipt.

    The realized count follows the deterministic exact-count rule
    ``floor(fraction * eligible)``.  Selection operates on the sorted
    household-ID inventory so equal frames produce equal samples regardless of
    incidental row order, and whole lineages (every entity row of a selected
    household) enter the sample together via :meth:`Frame.select`.

    Args:
        acs: The full pre-assembly ACS source frame (US schema, household
            weights, no support provenance).
        fraction: Household sampling fraction in ``(0, 1]``.
        seed: Non-negative integer seed for the selection RNG.

    Returns:
        The sampled frame and a JSON-ready receipt with eligible, requested,
        and realized counts, the selection digest, and mass bookkeeping.

    Raises:
        TypeError: If ``acs`` is not a Frame.
        ValueError: If the schema, configuration, or realized selection
            violates the sampling contract (including a floor of zero
            households, which fails closed).
    """

    if not isinstance(acs, Frame):
        raise TypeError(f"acs must be a Frame, got {type(acs).__name__}.")
    if acs.schema != US_SCHEMA:
        raise ValueError("ACS household sampling requires the US entity schema.")
    _validate_fraction(fraction)
    _validate_seed(seed)
    household_channel = support_channel_column("household")
    if household_channel in acs.table("household").columns:
        raise ValueError(
            "ACS household sampling runs before assembly; the source frame "
            f"already carries support provenance ({household_channel!r})."
        )

    household_ids = acs.table("household")["household_id"].to_numpy()
    eligible = int(len(household_ids))
    incoming_mass = float(acs.weights_for("household").total)
    requested = int(math.floor(fraction * eligible))
    if requested < 1:
        raise ValueError(
            f"ACS sample fraction {fraction!r} floors to zero households "
            f"({_EXACT_COUNT_RULE} with eligible={eligible}); the stacked "
            "spine requires at least one sampled household."
        )

    ordered_ids = np.sort(np.asarray(household_ids, copy=True))
    if requested == eligible:
        selected_ids = ordered_ids
        sampled = acs
    else:
        rng = np.random.default_rng(seed)
        selected_ids = np.sort(rng.choice(ordered_ids, size=requested, replace=False))
        person_mask = (
            acs.table("person")["person_household_id"].isin(selected_ids).to_numpy()
        )
        sampled = acs.select(person_mask)

    realized_ids = np.sort(sampled.table("household")["household_id"].to_numpy())
    if not np.array_equal(realized_ids, selected_ids):
        raise ValueError(
            "ACS household sampling realized a different household set than "
            "it selected; whole-household selection failed."
        )
    receipt: dict[str, object] = {
        "fraction": float(fraction),
        "seed": int(seed),
        "eligible_household_count": eligible,
        "requested_household_count": requested,
        "realized_household_count": int(len(realized_ids)),
        "exact_count_rule": _EXACT_COUNT_RULE,
        "selected_household_ids_sha256": _ids_sha256(selected_ids),
        "incoming_household_mass": incoming_mass,
        "sampled_household_mass": float(sampled.weights_for("household").total),
    }
    return sampled, receipt


def assemble_stacked_spine(
    asec: Frame,
    acs: Frame,
    *,
    acs_sample_fraction: float,
    acs_sample_seed: int,
    household_mass_shares: Mapping[str, float] | None = None,
    mass_anchor_channel: str = BASE_ASEC_SUPPORT_CHANNEL,
) -> StackedSpineResult:
    """Assemble ASEC plus a seeded ACS household sample into one spine.

    The sample is drawn by :func:`sample_acs_households`, the combination
    reuses the reviewed :func:`assemble_spines` seam unchanged, and the
    resulting frame carries a stacked-spine manifest binding the sampling
    configuration (fraction, seed), the realized selection digest, and the
    per-arm weight-harmonization receipts to the live rows.  Origin labels
    survive as the ordinary support-channel columns.

    Returns:
        A validated :class:`StackedSpineResult` whose receipt mirrors the
        frozen manifest as a JSON-ready mapping.
    """

    shares = (
        dict(DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES)
        if household_mass_shares is None
        else dict(household_mass_shares)
    )
    sampled, sample_receipt = sample_acs_households(
        acs,
        fraction=acs_sample_fraction,
        seed=acs_sample_seed,
    )
    asec_incoming_mass = float(asec.weights_for("household").total)
    incoming_masses = {
        BASE_ASEC_SUPPORT_CHANNEL: asec_incoming_mass,
        ACS_STACKED_SUPPORT_CHANNEL: float(sample_receipt["sampled_household_mass"]),
    }
    assembled = assemble_spines(
        {
            BASE_ASEC_SUPPORT_CHANNEL: asec,
            ACS_STACKED_SUPPORT_CHANNEL: sampled,
        },
        household_mass_shares=shares,
        mass_anchor_channel=mass_anchor_channel,
    )

    harmonization = _harmonization_receipt(
        assembled,
        shares=shares,
        anchor_mass=incoming_masses[mass_anchor_channel],
        incoming_masses=incoming_masses,
    )
    manifest: dict[str, object] = {
        "version": _STACKED_SPINE_MANIFEST_VERSION,
        "acs_sample_fraction": float(acs_sample_fraction),
        "acs_sample_seed": int(acs_sample_seed),
        "acs_sample": sample_receipt,
        "household_mass_shares": {
            channel: float(share) for channel, share in shares.items()
        },
        "mass_anchor_channel": mass_anchor_channel,
        "weight_harmonization": harmonization,
    }
    # The assembly metadata is preserved in full and augmented with the stack
    # manifest; the mass history is carried unchanged from the same source.
    stacked_metadata = {**assembled.metadata, STACKED_SPINE_MANIFEST_KEY: manifest}
    stacked_mass_log = assembled.mass_log
    stacked = Frame(
        {entity: assembled.table(entity) for entity in assembled.entities},
        assembled.schema,
        {
            entity: assembled.weights_for(entity)
            for entity in assembled.weighted_entities
        },
        assembled.strata,
        mass_log=stacked_mass_log,
        metadata=stacked_metadata,
    )
    validated = validate_stacked_spine_frame(
        stacked,
        boundary="stacked spine assembly output",
    )
    return StackedSpineResult(frame=stacked, receipt=_json_ready(validated))


def validate_stacked_spine_frame(
    frame: Frame,
    *,
    boundary: str,
) -> Mapping[str, object]:
    """Validate the stacked-spine manifest against the live origin labels.

    Layered on :func:`validate_assembly_provenance` (which already proves the
    live channel counts against the frozen assembly manifest), this validator
    binds the sampling identity: the manifest's fraction and seed must be
    present and typed, the realized count must satisfy the exact-count rule,
    the live native ACS household lineage must hash to the manifest's
    selection digest, and the live per-arm household masses must match the
    declared share allocation.  Any mutation of the sample, the counts, or the
    manifest fails closed with a named error.
    """

    validate_assembly_provenance(frame, boundary=boundary)
    manifest = frame.metadata.get(STACKED_SPINE_MANIFEST_KEY)
    if manifest is None:
        raise ValueError(
            f"{boundary}: stacked spine manifest {STACKED_SPINE_MANIFEST_KEY!r} "
            "is absent."
        )
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{boundary}: stacked spine manifest is malformed.")
    if manifest.get("version") != _STACKED_SPINE_MANIFEST_VERSION:
        raise ValueError(
            f"{boundary}: stacked spine manifest has unsupported version "
            f"{manifest.get('version')!r}."
        )
    assembly = frame.metadata[SPINE_ASSEMBLY_MANIFEST_KEY]
    channels = tuple(assembly["channels"])
    expected_channels = (BASE_ASEC_SUPPORT_CHANNEL, ACS_STACKED_SUPPORT_CHANNEL)
    if set(channels) != set(expected_channels):
        raise ValueError(
            f"{boundary}: stacked spine requires exactly the channels "
            f"{sorted(expected_channels)}; assembly declares {sorted(channels)}."
        )

    fraction = manifest.get("acs_sample_fraction")
    seed = manifest.get("acs_sample_seed")
    if not isinstance(fraction, float) or isinstance(fraction, bool):
        raise ValueError(
            f"{boundary}: stacked spine manifest acs_sample_fraction must be "
            f"a float, got {fraction!r}."
        )
    _validate_fraction(fraction, boundary=boundary)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(
            f"{boundary}: stacked spine manifest acs_sample_seed must be a "
            f"non-negative integer, got {seed!r}."
        )

    sample = manifest.get("acs_sample")
    if not isinstance(sample, Mapping):
        raise ValueError(f"{boundary}: stacked spine sample receipt is absent.")
    required_keys = (
        "eligible_household_count",
        "requested_household_count",
        "realized_household_count",
        "exact_count_rule",
        "selected_household_ids_sha256",
    )
    missing = [key for key in required_keys if key not in sample]
    if missing:
        raise ValueError(
            f"{boundary}: stacked spine sample receipt is missing {missing}."
        )
    if sample["exact_count_rule"] != _EXACT_COUNT_RULE:
        raise ValueError(
            f"{boundary}: stacked spine sample declares exact-count rule "
            f"{sample['exact_count_rule']!r}; expected {_EXACT_COUNT_RULE!r}."
        )
    eligible = int(sample["eligible_household_count"])
    requested = int(sample["requested_household_count"])
    realized = int(sample["realized_household_count"])
    if requested != int(math.floor(fraction * eligible)):
        raise ValueError(
            f"{boundary}: stacked spine requested household count {requested} "
            f"violates {_EXACT_COUNT_RULE} for fraction={fraction!r}, "
            f"eligible={eligible}."
        )
    if realized != requested:
        raise ValueError(
            f"{boundary}: stacked spine realized household count {realized} "
            f"differs from the requested count {requested}."
        )

    household = frame.table("household")
    channel_values = household[support_channel_column("household")].astype(str)
    clone_index = household[support_clone_index_column("household")]
    native_acs = channel_values.eq(ACS_STACKED_SUPPORT_CHANNEL) & clone_index.eq(0)
    live_count = int(native_acs.sum())
    if live_count != realized:
        raise ValueError(
            f"{boundary}: live native ACS household count {live_count} differs "
            f"from the stacked spine manifest's realized count {realized}."
        )
    live_ids = np.sort(
        household.loc[native_acs, spine_source_id_column("household")].to_numpy()
    )
    live_sha = _ids_sha256(live_ids)
    if live_sha != sample["selected_household_ids_sha256"]:
        raise ValueError(
            f"{boundary}: live native ACS household lineage digest {live_sha} "
            "differs from the stacked spine manifest's selection digest "
            f"{sample['selected_household_ids_sha256']}."
        )

    shares = manifest.get("household_mass_shares")
    if not isinstance(shares, Mapping) or set(shares) != set(expected_channels):
        raise ValueError(
            f"{boundary}: stacked spine manifest household_mass_shares must "
            f"exactly cover {sorted(expected_channels)}."
        )
    total_share = float(sum(float(value) for value in shares.values()))
    if not np.isclose(total_share, 1.0, rtol=_MASS_RTOL, atol=_MASS_RTOL):
        raise ValueError(
            f"{boundary}: stacked spine household_mass_shares sum to "
            f"{total_share!r}; expected 1.0."
        )
    harmonization = manifest.get("weight_harmonization")
    if not isinstance(harmonization, Mapping):
        raise ValueError(
            f"{boundary}: stacked spine weight-harmonization receipt is absent."
        )
    mass_anchor_channel = manifest.get("mass_anchor_channel")
    if mass_anchor_channel != channels[0]:
        raise ValueError(
            f"{boundary}: stacked spine mass_anchor_channel "
            f"{mass_anchor_channel!r} differs from the assembly anchor "
            f"channel {channels[0]!r}."
        )
    live_anchor_mass = float(frame.weights_for("household").total)
    anchor_arm = harmonization.get(mass_anchor_channel)
    if not isinstance(anchor_arm, Mapping) or "incoming_mass" not in anchor_arm:
        raise ValueError(
            f"{boundary}: stacked spine weight-harmonization receipt for "
            f"anchor {mass_anchor_channel!r} is malformed."
        )
    anchor_incoming = float(anchor_arm["incoming_mass"])
    if not np.isclose(
        anchor_incoming,
        live_anchor_mass,
        rtol=_MASS_RTOL,
        atol=0.0,
    ):
        raise ValueError(
            f"{boundary}: selected anchor {mass_anchor_channel!r} incoming "
            f"mass {anchor_incoming!r} differs from live anchor mass "
            f"{live_anchor_mass!r}."
        )

    weights = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    for channel in expected_channels:
        arm = harmonization.get(channel)
        if not isinstance(arm, Mapping) or not {
            "allocated_mass",
            "declared_allocation",
        }.issubset(arm):
            raise ValueError(
                f"{boundary}: stacked spine weight-harmonization receipt for "
                f"{channel!r} is malformed."
            )
        live_mass = float(weights[channel_values.eq(channel).to_numpy()].sum())
        allocated = float(arm["allocated_mass"])
        declared_allocation = float(arm["declared_allocation"])
        expected_allocation = float(shares[channel]) * live_anchor_mass
        if not np.isclose(
            declared_allocation,
            expected_allocation,
            rtol=_MASS_RTOL,
            atol=0.0,
        ):
            raise ValueError(
                f"{boundary}: declared {channel!r} allocation "
                f"{declared_allocation!r} differs from share "
                f"{float(shares[channel])!r} times live anchor mass "
                f"{live_anchor_mass!r}."
            )
        if not np.isclose(live_mass, allocated, rtol=_MASS_RTOL, atol=0.0):
            raise ValueError(
                f"{boundary}: live {channel!r} household mass {live_mass!r} "
                f"drifted from the allocated arm mass {allocated!r}."
            )
    return manifest


def _harmonization_receipt(
    assembled: Frame,
    *,
    shares: Mapping[str, float],
    anchor_mass: float,
    incoming_masses: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    household = assembled.table("household")
    channel_values = household[support_channel_column("household")].astype(str)
    weights = np.asarray(assembled.weights_for("household").values, dtype=np.float64)
    receipt: dict[str, dict[str, float]] = {}
    for channel, share in shares.items():
        incoming = float(incoming_masses[channel])
        allocated = float(weights[channel_values.eq(channel).to_numpy()].sum())
        receipt[channel] = {
            "share": float(share),
            "incoming_mass": incoming,
            "allocated_mass": allocated,
            "declared_allocation": float(share) * anchor_mass,
            "scale_factor": allocated / incoming,
        }
    return receipt


def _ids_sha256(ids: np.ndarray) -> str:
    payload = json.dumps(
        [int(value) for value in np.asarray(ids).tolist()],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_fraction(fraction: float, *, boundary: str | None = None) -> None:
    prefix = f"{boundary}: " if boundary else ""
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not np.isfinite(fraction)
        or not 0.0 < float(fraction) <= 1.0
    ):
        raise ValueError(
            f"{prefix}ACS sample fraction must be a finite number in (0, 1]; "
            f"got {fraction!r}."
        )


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(
            f"ACS sample seed must be a non-negative integer; got {seed!r}."
        )


def _json_ready(value: object) -> dict[str, object]:
    def thaw(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): thaw(nested) for key, nested in item.items()}
        if isinstance(item, (list, tuple)):
            return [thaw(nested) for nested in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    if not isinstance(value, Mapping):
        raise TypeError("Stacked spine receipts must be mappings.")
    return {str(key): thaw(item) for key, item in value.items()}


# ---------------------------------------------------------------------------
# Cross-origin gap-fill (charter item 2)
# ---------------------------------------------------------------------------

_GAP_FILL_ASEC_TO_ACS = "asec_survey_to_acs"
_GAP_FILL_ACS_TO_ASEC = "acs_housing_to_asec"
_GAP_FILL_HOUSING_FAMILY = "housing"


@dataclass(frozen=True)
class GapFillDirection:
    """One declared cross-origin fill: recipient origin <- donor origin.

    Activation authority is declared here, not inferred from nullness: the
    named recipient channel's rows are the only rows the direction may fill,
    and the named donor channel's native rows are the only donor evidence.
    The transfer machinery itself stays spine-blind; this owner-level
    declaration is what makes the run-7 silent-skip class impossible — a
    direction either fills its declared families on its declared rows or
    fails by name.
    """

    name: str
    recipient_channel: str
    donor_channel: str
    target_families: TargetFamilies

    def __post_init__(self) -> None:
        for label, value in (
            ("name", self.name),
            ("recipient_channel", self.recipient_channel),
            ("donor_channel", self.donor_channel),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"GapFillDirection.{label} must be a non-empty string."
                )
        if self.recipient_channel == self.donor_channel:
            raise ValueError(
                "GapFillDirection must fill across origins; recipient and "
                f"donor are both {self.donor_channel!r}."
            )
        if not isinstance(self.target_families, Mapping) or not any(
            families for families in self.target_families.values()
        ):
            raise ValueError(
                f"GapFillDirection {self.name!r} declares no target families."
            )


@dataclass(frozen=True)
class GapFillResult:
    """The gap-filled stacked spine plus per-direction receipts."""

    frame: Frame
    receipt: Mapping[str, object]
    transfer_results: Mapping[str, AcsTransferResult] = field(default_factory=dict)


def stacked_gap_fill_plan(
    target_families: TargetFamilies | None = None,
) -> tuple[GapFillDirection, ...]:
    """Return the declared two-direction gap-fill plan for the stacked spine.

    ACS-origin rows receive every declared ASEC-survey transfer family except
    housing; ASEC-origin rows receive the ACS-native housing family.  The
    declaration reuses the reviewed ACS-transfer family registry unchanged so
    the gap-fill, the completeness gate, and the by-origin battery all consume
    one plan.
    """

    families = (
        declared_acs_transfer_target_families()
        if target_families is None
        else target_families
    )
    survey_families: dict[str, dict[str, tuple[str, ...]]] = {}
    housing_families: dict[str, dict[str, tuple[str, ...]]] = {}
    for entity, entity_families in families.items():
        for family, targets in entity_families.items():
            bucket = (
                housing_families
                if family == _GAP_FILL_HOUSING_FAMILY
                else survey_families
            )
            bucket.setdefault(entity, {})[family] = tuple(targets)
    directions: list[GapFillDirection] = []
    if survey_families:
        directions.append(
            GapFillDirection(
                name=_GAP_FILL_ASEC_TO_ACS,
                recipient_channel=ACS_STACKED_SUPPORT_CHANNEL,
                donor_channel=BASE_ASEC_SUPPORT_CHANNEL,
                target_families=survey_families,
            )
        )
    if housing_families:
        directions.append(
            GapFillDirection(
                name=_GAP_FILL_ACS_TO_ASEC,
                recipient_channel=BASE_ASEC_SUPPORT_CHANNEL,
                donor_channel=ACS_STACKED_SUPPORT_CHANNEL,
                target_families=housing_families,
            )
        )
    return tuple(directions)


def gap_fill_stacked_spine(
    frame: Frame,
    *,
    plan: Sequence[GapFillDirection] | None = None,
    seed: int = 0,
    n_estimators: int = 100,
    max_targets_per_fit: int = DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    target_banks: Mapping[str, AcsTransferTargetBank] | None = None,
) -> GapFillResult:
    """Gap-fill survey-specific fields cross-origin on the stacked spine.

    Runs before any clone operator: every row still carries clone index zero,
    so filled values are cloned into the PUF arm afterwards and the single
    PUF pass conditions on observed predictors for every origin.

    Per direction, in order:

    1. **Activation authority** — declared, then verified: every target
       column must exist, the donor origin's rows must observe it completely,
       and every null cell must lie on the declared recipient origin.  A
       null anywhere else is a named terminal failure, so absence can never
       silently reroute or skip a family (populace#578 audit item 2).
    2. **Authoritative donors** — the donor frame passed to the spine-blind
       transfer is this owner's projection of the donor origin's native rows
       (audit item 3); ``donor_channel=None`` marks the deliberate
       whole-donor fit of that projection.
    3. **Banked transfer** — the reviewed #608 target-at-a-time banking
       machinery is reused unchanged via ``target_banks[direction.name]``.
    4. **Post-verification** — donor-origin cells must be byte-identical
       before and after, and no null may remain on authorized rows beyond
       the transfer's receipted unmodeled rows.

    Returns a :class:`GapFillResult` whose receipt records, per direction and
    target, the authorized-null, imputed, unmodeled, and residual-null
    counts alongside the transfer's fit provenance.
    """

    validate_stacked_spine_frame(frame, boundary="stacked gap-fill entry")
    directions = tuple(stacked_gap_fill_plan() if plan is None else plan)
    if not directions:
        raise ValueError("Stacked gap-fill requires at least one direction.")
    names = [direction.name for direction in directions]
    if len(set(names)) != len(names):
        raise ValueError(f"Stacked gap-fill directions repeat names: {names}.")
    if target_banks is not None:
        unknown_banks = sorted(set(target_banks) - set(names))
        if unknown_banks:
            raise ValueError(
                f"target_banks name unknown gap-fill direction(s): {unknown_banks}."
            )

    person_clone = frame.table("person")[support_clone_index_column("person")]
    if not person_clone.eq(0).all():
        raise ValueError(
            "Stacked gap-fill must run before clone operators; found nonzero "
            "person support clone indices."
        )

    current = frame
    receipts: dict[str, object] = {}
    transfer_results: dict[str, AcsTransferResult] = {}
    for direction in directions:
        pre_counts = _verify_gap_fill_activation_authority(
            current,
            direction=direction,
        )
        donor = _origin_projection(current, channel=direction.donor_channel)
        donor_snapshot = {
            entity: _direction_targets_snapshot(
                current,
                entity=entity,
                targets=targets,
                channel=direction.donor_channel,
            )
            for entity, targets in _direction_entity_targets(direction).items()
        }
        result = transfer_acs_inputs(
            current,
            donor,
            target_families=direction.target_families,
            donor_channel=None,
            seed=seed,
            n_estimators=n_estimators,
            max_targets_per_fit=max_targets_per_fit,
            target_bank=(target_banks or {}).get(direction.name),
        )
        transfer_results[direction.name] = result
        current = result.frame
        receipts[direction.name] = _verify_gap_fill_outcome(
            current,
            direction=direction,
            pre_counts=pre_counts,
            donor_snapshot=donor_snapshot,
            result=result,
        )

    validate_stacked_spine_frame(current, boundary="stacked gap-fill output")
    return GapFillResult(
        frame=current,
        receipt={"directions": receipts},
        transfer_results=transfer_results,
    )


def _direction_entity_targets(
    direction: GapFillDirection,
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for entity, families in direction.target_families.items():
        collected: list[str] = []
        for targets in families.values():
            collected.extend(targets)
        result[entity] = tuple(collected)
    return result


def _origin_projection(frame: Frame, *, channel: str) -> Frame:
    """Project one origin's native lineages as a standalone donor frame."""

    person = frame.table("person")
    mask = (
        person[support_channel_column("person")].astype(str).eq(channel)
        & person[support_clone_index_column("person")].eq(0)
    ).to_numpy()
    if not mask.any():
        raise ValueError(
            f"Stacked spine has no native person rows for origin {channel!r}."
        )
    return frame.select(mask)


def _direction_targets_snapshot(
    frame: Frame,
    *,
    entity: str,
    targets: Sequence[str],
    channel: str,
) -> pd.DataFrame:
    table = frame.table(entity)
    mask = table[support_channel_column(entity)].astype(str).eq(channel)
    present = [target for target in targets if target in table.columns]
    return table.loc[mask, present].copy(deep=True)


def _verify_gap_fill_activation_authority(
    frame: Frame,
    *,
    direction: GapFillDirection,
) -> dict[tuple[str, str], dict[str, int]]:
    """Verify declared activation authority before any modeling runs."""

    failures: list[str] = []
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for entity, families in direction.target_families.items():
        table = frame.table(entity)
        channel = table[support_channel_column(entity)].astype(str)
        recipient_rows = channel.eq(direction.recipient_channel)
        donor_rows = channel.eq(direction.donor_channel)
        recipient_count = int(recipient_rows.sum())
        donor_count = int(donor_rows.sum())
        if recipient_count == 0:
            failures.append(
                f"{direction.name}/{entity}: declared recipient channel "
                f"{direction.recipient_channel!r} has no live rows."
            )
        if donor_count == 0:
            failures.append(
                f"{direction.name}/{entity}: declared donor channel "
                f"{direction.donor_channel!r} has no live rows."
            )
        for family, targets in families.items():
            for target in targets:
                label = f"{direction.name}/{entity}/{family}/{target}"
                if target not in table.columns:
                    failures.append(
                        f"{label}: declared gap-fill target column is absent "
                        "from the stacked spine."
                    )
                    continue
                null_mask = table[target].isna()
                donor_nulls = int((null_mask & donor_rows).sum())
                unauthorized_nulls = int(
                    (null_mask & ~recipient_rows & ~donor_rows).sum()
                )
                if donor_nulls:
                    failures.append(
                        f"{label}: donor origin {direction.donor_channel!r} "
                        f"has {donor_nulls} null cell(s); donors must observe "
                        "every declared target."
                    )
                if unauthorized_nulls:
                    failures.append(
                        f"{label}: {unauthorized_nulls} null cell(s) lie "
                        "outside the declared recipient origin "
                        f"{direction.recipient_channel!r}."
                    )
                counts[(entity, target)] = {
                    "authorized_null_rows": int((null_mask & recipient_rows).sum()),
                    "recipient_rows": int(recipient_rows.sum()),
                    "donor_rows": int(donor_rows.sum()),
                }
    if failures:
        raise ValueError(
            "Stacked gap-fill activation authority failed:\n  " + "\n  ".join(failures)
        )
    return counts


def _verify_gap_fill_outcome(
    frame: Frame,
    *,
    direction: GapFillDirection,
    pre_counts: Mapping[tuple[str, str], Mapping[str, int]],
    donor_snapshot: Mapping[str, pd.DataFrame],
    result: AcsTransferResult,
) -> dict[str, object]:
    """Verify donor invariance and residual nulls; build the direction receipt."""

    failures: list[str] = []
    imputed_by_target = {
        (record.entity, record.column): record for record in result.imputed_inputs
    }
    target_receipts: dict[str, dict[str, object]] = {}
    for entity, families in direction.target_families.items():
        table = frame.table(entity)
        channel = table[support_channel_column(entity)].astype(str)
        recipient_rows = channel.eq(direction.recipient_channel)
        donor_after = {
            entity_name: _direction_targets_snapshot(
                frame,
                entity=entity_name,
                targets=targets,
                channel=direction.donor_channel,
            )
            for entity_name, targets in _direction_entity_targets(direction).items()
        }[entity]
        for family, targets in families.items():
            for target in targets:
                label = f"{direction.name}/{entity}/{family}/{target}"
                before = donor_snapshot[entity].get(target)
                after = donor_after.get(target)
                if before is None or after is None or not before.equals(after):
                    failures.append(
                        f"{label}: donor origin cells changed during the "
                        "gap-fill transfer; observed donor data must be "
                        "byte-identical."
                    )
                null_mask = table[target].isna()
                residual_nulls = int((null_mask & recipient_rows).sum())
                outside_nulls = int((null_mask & ~recipient_rows).sum())
                if outside_nulls:
                    failures.append(
                        f"{label}: {outside_nulls} null cell(s) appeared "
                        "outside the declared recipient origin during the "
                        "transfer."
                    )
                record = imputed_by_target.get((entity, target))
                imputed = record.imputed_recipient_rows if record else 0
                unmodeled = record.unmodeled_recipient_rows if record else 0
                pre = pre_counts[(entity, target)]
                authorized = pre["authorized_null_rows"]
                if residual_nulls != unmodeled:
                    failures.append(
                        f"{label}: residual-null equation failed: "
                        f"residual_null_rows={residual_nulls} != "
                        f"unmodeled_rows={unmodeled}."
                    )
                if authorized != imputed + unmodeled:
                    failures.append(
                        f"{label}: activation accounting equation failed: "
                        f"authorized_null_rows={authorized} != "
                        f"imputed_rows={imputed} + unmodeled_rows={unmodeled}."
                    )
                target_receipts[f"{entity}/{family}/{target}"] = {
                    "authorized_null_rows": authorized,
                    "imputed_rows": imputed,
                    "unmodeled_rows": unmodeled,
                    "residual_null_rows": residual_nulls,
                }
    if failures:
        raise ValueError(
            "Stacked gap-fill outcome verification failed:\n  " + "\n  ".join(failures)
        )
    return {
        "recipient_channel": direction.recipient_channel,
        "donor_channel": direction.donor_channel,
        "donor_selection": "owner_projection_of_native_donor_rows",
        "resolved_donor_channel": result.resolved_donor_channel,
        "targets": target_receipts,
        "deferred_inputs": list(result.deferred_inputs),
        "fit_records": [
            {"fit_name": record.fit_name, "weight_kind": record.weight_kind}
            for record in result.fit_records
        ],
    }


# ---------------------------------------------------------------------------
# The single PUF pass over the stacked spine (charter item 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StackedPufPassResult:
    """The post-PUF stacked frame plus attachment and fit receipts."""

    frame: Frame
    receipt: Mapping[str, object]


def run_stacked_puf_pass(
    frame: Frame,
    donor_tax_units: pd.DataFrame,
    *,
    clone_attachment_fraction: float,
    clone_attachment_seed: int,
    predictors: Sequence[str] | None = None,
    person_outputs: Sequence[str] | None = None,
    tax_unit_outputs: Sequence[str] | None = None,
    seed: int = 0,
    n_estimators: int = 100,
    fit_records: list[FitWeightRecord] | None = None,
    tail_bound_diagnostics: list[dict[str, object]] | None = None,
) -> StackedPufPassResult:
    """Run the one PUF pass over the gap-filled stacked spine.

    Order is the charter's: the spine must already be gap-filled (this entry
    validates the stacked manifest and refuses cloned input), the PUF clone
    arm attaches to a seeded whole-household sample of stacked households
    (both origins; reusing the reviewed clone-routing discipline), and the
    primary QRF then runs under both stacked doctrines — recipient predictors
    must be complete (no zero-filled absence) and finalization preserves
    nulls on every cell the pass does not own.
    """

    validate_stacked_spine_frame(frame, boundary="stacked PUF pass entry")
    person_clone = frame.table("person")[support_clone_index_column("person")]
    if not person_clone.eq(0).all():
        raise ValueError(
            "The stacked PUF pass owns clone attachment; found nonzero person "
            "support clone indices on its input."
        )
    cloned = clone_us_frame_for_puf_support(
        frame,
        clone_attachment_fraction=clone_attachment_fraction,
        clone_attachment_seed=clone_attachment_seed,
    )
    attachment = validate_puf_clone_attachment(
        cloned,
        boundary="stacked PUF pass clone attachment",
        expected_fraction=clone_attachment_fraction,
        expected_seed=clone_attachment_seed,
    )

    kwargs: dict[str, object] = {}
    if predictors is not None:
        kwargs["predictors"] = tuple(predictors)
    if person_outputs is not None:
        kwargs["person_outputs"] = tuple(person_outputs)
    if tax_unit_outputs is not None:
        kwargs["tax_unit_outputs"] = tuple(tax_unit_outputs)
    imputed = impute_us_puf_tax_detail_support(
        cloned,
        donor_tax_units,
        seed=seed,
        n_estimators=n_estimators,
        fit_records=fit_records,
        tail_bound_diagnostics=tail_bound_diagnostics,
        require_complete_recipient_predictors=True,
        absent_cells=PUF_ABSENT_CELLS_PRESERVE_NULLS,
        **kwargs,
    )
    validate_stacked_spine_frame(imputed, boundary="stacked PUF pass output")
    validate_puf_clone_attachment(
        imputed,
        boundary="stacked PUF pass output",
        expected_fraction=clone_attachment_fraction,
        expected_seed=clone_attachment_seed,
    )

    person = imputed.table("person")
    channel = person[support_channel_column("person")].astype(str)
    clone_index = person[support_clone_index_column("person")]
    recipients_by_origin = {
        origin: int((channel.eq(origin) & clone_index.eq(1)).sum())
        for origin in sorted(channel.unique())
    }
    return StackedPufPassResult(
        frame=imputed,
        receipt={
            "clone_attachment": _json_ready(attachment),
            "doctrines": {
                "require_complete_recipient_predictors": True,
                "absent_cells": PUF_ABSENT_CELLS_PRESERVE_NULLS,
            },
            "recipient_person_rows_by_origin": recipients_by_origin,
        },
    )


# ---------------------------------------------------------------------------
# Pre-simulation completeness gate (charter item 4)
# ---------------------------------------------------------------------------

_COMPLETENESS_GATE_NAME = "us_stacked_completeness"
_ANY_CHANNEL = "*"


@dataclass(frozen=True)
class AbsenceProof:
    """An explicit source-by-role authority proof for permitted null cells.

    A declared target's cells may be null only where a proof names the exact
    origin channel and clone role, with the reason recorded.  ``"*"`` may
    cover every origin only when no gap-fill direction declares a donor for
    that target.  This is the audit's item-5 contract: a target is banked or
    imputed, or its absence carries an explicit source-by-role proof — silence
    is never authority.
    """

    entity: str
    column: str
    channel: str
    clone_index: int
    reason: str

    def __post_init__(self) -> None:
        for label, value in (
            ("entity", self.entity),
            ("column", self.column),
            ("channel", self.channel),
            ("reason", self.reason),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"AbsenceProof.{label} must be a non-empty string.")
        if (
            isinstance(self.clone_index, bool)
            or not isinstance(self.clone_index, int)
            or self.clone_index < 0
        ):
            raise ValueError("AbsenceProof.clone_index must be a non-negative integer.")


def stacked_completeness_gate(
    frame: Frame,
    *,
    declared_surface: TargetFamilies,
    declared_gap_fill_plan: Sequence[GapFillDirection],
    absence_proofs: Sequence[AbsenceProof] = (),
) -> GateResult:
    """Prove every declared target is filled or carries absence authority.

    For every declared ``entity/family/target``: a missing column is a named
    terminal failure (a whole registered family can never silently vanish
    again — this is the check that would have caught run 7's 58-target skip);
    a null cell is permitted only where an :class:`AbsenceProof` names its
    origin channel and clone role, and every unproven null fails by name with
    its per-origin, per-role counts.  ``declared_gap_fill_plan`` is required
    authority: its donor direction disables wildcard proofs for that target.
    """

    declared_directions: dict[tuple[str, str], GapFillDirection] = {}
    for direction in declared_gap_fill_plan:
        if not isinstance(direction, GapFillDirection):
            raise ValueError(
                "declared_gap_fill_plan must contain GapFillDirection values, got "
                f"{type(direction).__name__}."
            )
        for entity, targets in _direction_entity_targets(direction).items():
            for target in targets:
                key = (entity, target)
                previous = declared_directions.get(key)
                if previous is not None and previous != direction:
                    raise ValueError(
                        "declared_gap_fill_plan assigns conflicting directions "
                        f"to {entity}/{target}: {previous.name!r} and "
                        f"{direction.name!r}."
                    )
                declared_directions[key] = direction

    proof_index: dict[tuple[str, str], dict[tuple[str, int], str]] = {}
    for proof in absence_proofs:
        if not isinstance(proof, AbsenceProof):
            raise ValueError(
                "absence_proofs must contain AbsenceProof values, got "
                f"{type(proof).__name__}."
            )
        proof_index.setdefault((proof.entity, proof.column), {})[
            (proof.channel, proof.clone_index)
        ] = proof.reason

    failures: list[str] = []
    target_receipts: dict[str, dict[str, object]] = {}
    declared_count = 0
    for entity, families in declared_surface.items():
        if entity not in frame.entities:
            failures.append(
                f"{entity}: declared entity is absent from the stacked frame."
            )
            continue
        table = frame.table(entity)
        channel = table[support_channel_column(entity)].astype(str)
        clone_index = pd.to_numeric(
            table[support_clone_index_column(entity)],
            errors="raise",
        ).astype("int64")
        for family, targets in families.items():
            for target in targets:
                declared_count += 1
                label = f"{entity}/{family}/{target}"
                if target not in table.columns:
                    failures.append(
                        f"{label}: declared target column is missing from the "
                        "pre-simulation pool; the registered family must "
                        "never silently vanish from the active bank."
                    )
                    continue
                null_mask = table[target].isna()
                if not null_mask.any():
                    target_receipts[label] = {"status": "complete", "null_rows": 0}
                    continue
                proofs = proof_index.get((entity, target), {})
                declared_direction = declared_directions.get((entity, target))
                null_channels = channel.loc[null_mask]
                null_clones = clone_index.loc[null_mask]
                unproven: dict[str, int] = {}
                proven: dict[str, dict[str, object]] = {}
                grouped = (
                    pd.DataFrame({"channel": null_channels, "clone_index": null_clones})
                    .groupby(["channel", "clone_index"], sort=True)
                    .size()
                )
                for (cell_channel, cell_clone), count in grouped.items():
                    reason = proofs.get((str(cell_channel), int(cell_clone)))
                    authority_form = "origin_exact"
                    if reason is None and declared_direction is None:
                        reason = proofs.get((_ANY_CHANNEL, int(cell_clone)))
                        authority_form = "wildcard_no_declared_donor_plan"
                    key = f"{cell_channel}/clone_{int(cell_clone)}"
                    if reason is None:
                        unproven[key] = int(count)
                        if declared_direction is not None:
                            failures.append(
                                f"{label}: {int(count)} null cell(s) on {key} "
                                "require an origin-exact authority proof because "
                                "declared gap-fill direction "
                                f"{declared_direction.name!r} names donor "
                                f"{declared_direction.donor_channel!r}; wildcard "
                                "authority is forbidden."
                            )
                    else:
                        proof_receipt: dict[str, object] = {
                            "null_rows": int(count),
                            "reason": reason,
                            "authority_form": authority_form,
                        }
                        if declared_direction is not None:
                            proof_receipt.update(
                                {
                                    "declared_direction": declared_direction.name,
                                    "declared_donor_channel": (
                                        declared_direction.donor_channel
                                    ),
                                    "declared_recipient_channel": (
                                        declared_direction.recipient_channel
                                    ),
                                }
                            )
                        proven[key] = proof_receipt
                if unproven:
                    failures.append(
                        f"{label}: {sum(unproven.values())} null cell(s) have "
                        "no source-by-role authority proof "
                        f"(by origin/role: {unproven})."
                    )
                target_receipts[label] = {
                    "status": "proven_absent" if not unproven else "unproven",
                    "null_rows": int(null_mask.sum()),
                    "proven": proven,
                    "unproven": unproven,
                }
    return GateResult(
        name=_COMPLETENESS_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details={
            "declared_targets": declared_count,
            "targets": target_receipts,
        },
    )


# ---------------------------------------------------------------------------
# By-origin battery (charter item 5)
# ---------------------------------------------------------------------------

_BATTERY_GATE_NAME = "us_by_origin_battery"
ORIGIN_BATTERY_METRIC_KINDS = (
    "boolean_incidence",
    "rare_incidence",
    "monetary_sign_separated",
    "categorical_tvd",
)
_BATTERY_INCIDENCE_RATIO_BOUNDS = (0.8, 1.25)
_BATTERY_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
_BATTERY_QUANTILE_ENVELOPE_TOLERANCE = 0.25
_BATTERY_CATEGORICAL_TVD_TOLERANCE = 0.25


@dataclass(frozen=True)
class _BatterySupportProfile:
    profile_id: str
    version: int
    min_effective_support: int


_BATTERY_SUPPORT_PROFILE = _BatterySupportProfile(
    profile_id="us_stacked_origin_battery_support",
    version=1,
    min_effective_support=5,
)
_BATTERY_SUPPORT_PROFILE_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "min_effective_support": (_BATTERY_SUPPORT_PROFILE.min_effective_support),
            "profile_id": _BATTERY_SUPPORT_PROFILE.profile_id,
            "version": _BATTERY_SUPPORT_PROFILE.version,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()

_BATTERY_DECLARED_PLAN_ID = "stacked_gap_fill_plan"
_BATTERY_DECLARED_PLAN_VERSION = 1
_BATTERY_DECLARED_TARGETS = tuple(
    sorted(
        (
            entity,
            family,
            target,
            0,
        )
        for direction in stacked_gap_fill_plan()
        for entity, families in direction.target_families.items()
        for family, targets in families.items()
        for target in targets
    )
)
_BATTERY_DECLARED_TARGET_SET = frozenset(_BATTERY_DECLARED_TARGETS)
_BATTERY_DECLARED_PLAN_SHA256 = hashlib.sha256(
    json.dumps(
        [
            f"{entity}/{family}/{column}[clone_{clone_index}]"
            for entity, family, column, clone_index in _BATTERY_DECLARED_TARGETS
        ],
        separators=(",", ":"),
    ).encode()
).hexdigest()
_BATTERY_AUTHORITATIVE_METRICS = {
    (
        "person",
        "model_required_boolean",
        "has_champva_health_coverage_at_interview",
        0,
    ): "boolean_incidence",
}


@dataclass(frozen=True)
class OriginBatterySpec:
    """Declared per-column metrics for one gap-filled family's comparison.

    The metric is DECLARED per column at registration — never dispatched
    from a physical dtype (the audit's registry defect: semantically equal
    boolean take-up fields received different contracts because one was
    object-backed).  ``clone_index`` scopes the comparison to one clone role:
    0 compares gap-filled native rows across origins, 1 compares the PUF
    arm's imputed rows across origins.
    """

    entity: str
    family: str
    column_metrics: Mapping[str, str]
    clone_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.entity, str) or not self.entity.strip():
            raise ValueError("OriginBatterySpec.entity must be a non-empty string.")
        if not isinstance(self.family, str) or not self.family.strip():
            raise ValueError("OriginBatterySpec.family must be a non-empty string.")
        if not isinstance(self.column_metrics, Mapping) or not self.column_metrics:
            raise ValueError(
                f"OriginBatterySpec {self.entity}/{self.family} declares no "
                "column metrics."
            )
        unknown = sorted(
            {
                metric
                for metric in self.column_metrics.values()
                if metric not in ORIGIN_BATTERY_METRIC_KINDS
            }
        )
        if unknown:
            raise ValueError(
                f"OriginBatterySpec {self.entity}/{self.family} declares "
                f"unknown metric kind(s) {unknown}; expected one of "
                f"{list(ORIGIN_BATTERY_METRIC_KINDS)}."
            )
        if (
            isinstance(self.clone_index, bool)
            or not isinstance(self.clone_index, int)
            or self.clone_index < 0
        ):
            raise ValueError(
                "OriginBatterySpec.clone_index must be a non-negative integer."
            )


def by_origin_battery(
    frame: Frame,
    *,
    registry: Sequence[OriginBatterySpec],
) -> GateResult:
    """Compare declared statistics between origins within the one spine.

    Replaces the retired spine-vs-spine agreement: the same comparisons,
    scoped to gap-filled families, with per-family DECLARED metrics.  The
    tolerances are the chartered ones — incidence ratios within
    ``[0.8, 1.25]``, conditional-quantile envelopes within ``0.25``,
    categorical total-variation within ``0.25`` — deliberately NOT widened.

    Support-awareness is a validity domain, not a tolerance: a comparison
    whose scope rows on either origin fall below the spec's minimum
    effective support is receipted ``insufficient_support`` instead of
    producing a fake confident verdict, and a quantile envelope is evaluated
    only when both origins carry at least that many nonzero rows on the
    compared leg.  A tested rare comparison still fails on any one-sided
    hole — the run-7 ``160,667x`` class fails under every profile because
    both origins carry ample support.
    """

    specs = tuple(registry)
    if not specs:
        raise ValueError("The by-origin battery requires at least one spec.")
    keys = [(spec.entity, spec.family, spec.clone_index) for spec in specs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"By-origin battery repeats family specs: {duplicates}.")
    registered_targets = {
        (spec.entity, spec.family, column, spec.clone_index)
        for spec in specs
        for column in spec.column_metrics
    }
    missing_targets = tuple(sorted(_BATTERY_DECLARED_TARGET_SET - registered_targets))
    registration_failures = [
        f"missing declared battery target {_battery_target_label(target)}."
        for target in missing_targets
    ]
    for target, required_metric in _BATTERY_AUTHORITATIVE_METRICS.items():
        supplied_metric = next(
            (
                spec.column_metrics[target[2]]
                for spec in specs
                if (spec.entity, spec.family, spec.clone_index)
                == (target[0], target[1], target[3])
                and target[2] in spec.column_metrics
            ),
            None,
        )
        if supplied_metric is not None and supplied_metric != required_metric:
            registration_failures.append(
                f"declared battery target {_battery_target_label(target)} must "
                f"use authoritative metric {required_metric!r}, got "
                f"{supplied_metric!r}."
            )
    coverage_details = {
        "declared_target_count": len(_BATTERY_DECLARED_TARGETS),
        "registered_target_count": len(registered_targets),
        "missing_declared_targets": [
            _battery_target_label(target) for target in missing_targets
        ],
        "extra_registered_targets": [
            _battery_target_label(target)
            for target in sorted(registered_targets - _BATTERY_DECLARED_TARGET_SET)
        ],
        "declared_plan": {
            "plan_id": _BATTERY_DECLARED_PLAN_ID,
            "version": _BATTERY_DECLARED_PLAN_VERSION,
            "sha256": _BATTERY_DECLARED_PLAN_SHA256,
        },
        "support_profile": _battery_support_profile_receipt(),
    }
    if registration_failures:
        return GateResult(
            name=_BATTERY_GATE_NAME,
            passed=False,
            failures=tuple(registration_failures),
            details={
                **coverage_details,
                "registered_specs": len(specs),
                "tested_comparisons": 0,
                "untestable_comparisons": [],
                "comparisons": {},
            },
        )
    validate_stacked_spine_frame(frame, boundary="by-origin battery")

    failures: list[str] = []
    comparisons: dict[str, object] = {}
    untestable: list[str] = []
    tested = 0
    for spec in specs:
        table = frame.table(spec.entity)
        channel = table[support_channel_column(spec.entity)].astype(str)
        clone_index = pd.to_numeric(
            table[support_clone_index_column(spec.entity)],
            errors="raise",
        ).astype("int64")
        weights = np.asarray(
            frame.resolve_weights(spec.entity).values,
            dtype=np.float64,
        )
        scope = (clone_index.eq(spec.clone_index)).to_numpy() & (weights > 0.0)
        left_rows = scope & channel.eq(BASE_ASEC_SUPPORT_CHANNEL).to_numpy()
        right_rows = scope & channel.eq(ACS_STACKED_SUPPORT_CHANNEL).to_numpy()
        for column, metric in spec.column_metrics.items():
            label = f"{spec.entity}/{spec.family}/{column}[clone_{spec.clone_index}]"
            if column not in table.columns:
                failures.append(f"{label}: registered column is absent from the frame.")
                continue
            series = table[column]
            scoped_nulls = int(series.isna().to_numpy(dtype=bool)[scope].sum())
            if scoped_nulls:
                failures.append(
                    f"{label}: {scoped_nulls} null value(s) inside the "
                    "comparison scope; the battery runs only on completed "
                    "surfaces."
                )
                continue
            support = {
                "asec": int(left_rows.sum()),
                "acs": int(right_rows.sum()),
            }
            if min(support.values()) < _BATTERY_SUPPORT_PROFILE.min_effective_support:
                comparisons[label] = {
                    "status": "insufficient_support",
                    "metric": metric,
                    "scope_rows": support,
                }
                untestable.append(label)
                continue
            tested += 1
            if metric == "categorical_tvd":
                _battery_categorical_comparison(
                    label=label,
                    series=series,
                    left_rows=left_rows,
                    right_rows=right_rows,
                    weights=weights,
                    failures=failures,
                    comparisons=comparisons,
                )
                continue
            values = _battery_numeric_values(
                label,
                series,
                metric=metric,
                failures=failures,
            )
            if values is None:
                continue
            if metric in {"boolean_incidence", "rare_incidence"}:
                _battery_incidence_comparison(
                    label=label,
                    metric=metric,
                    values=values,
                    left_rows=left_rows,
                    right_rows=right_rows,
                    weights=weights,
                    failures=failures,
                    comparisons=comparisons,
                )
            else:
                _battery_sign_separated_comparison(
                    label=label,
                    values=values,
                    left_rows=left_rows,
                    right_rows=right_rows,
                    weights=weights,
                    failures=failures,
                    comparisons=comparisons,
                )
    return GateResult(
        name=_BATTERY_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details={
            **coverage_details,
            "tolerances": {
                "incidence_ratio_bounds": list(_BATTERY_INCIDENCE_RATIO_BOUNDS),
                "max_quantile_envelope_distance": (
                    _BATTERY_QUANTILE_ENVELOPE_TOLERANCE
                ),
                "max_categorical_total_variation_distance": (
                    _BATTERY_CATEGORICAL_TVD_TOLERANCE
                ),
            },
            "registered_specs": len(specs),
            "tested_comparisons": tested,
            "untestable_comparisons": sorted(untestable),
            "comparisons": comparisons,
        },
    )


def _battery_target_label(target: tuple[str, str, str, int]) -> str:
    entity, family, column, clone_index = target
    return f"{entity}/{family}/{column}[clone_{clone_index}]"


def _battery_support_profile_receipt() -> dict[str, object]:
    return {
        "profile_id": _BATTERY_SUPPORT_PROFILE.profile_id,
        "version": _BATTERY_SUPPORT_PROFILE.version,
        "min_effective_support": _BATTERY_SUPPORT_PROFILE.min_effective_support,
        "sha256": _BATTERY_SUPPORT_PROFILE_SHA256,
    }


def _battery_numeric_values(
    label: str,
    series: pd.Series,
    *,
    metric: str,
    failures: list[str],
) -> np.ndarray | None:
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        failures.append(f"{label}: declared {metric} metric requires numeric values.")
        return None
    if metric == "boolean_incidence":
        invalid = ~np.isin(values, (0.0, 1.0))
        if invalid.any():
            failures.append(
                f"{label}: declared boolean incidence requires values in "
                f"{{0, 1}}; found {int(invalid.sum())} other value(s)."
            )
            return None
    return values


def _battery_incidence_comparison(
    *,
    label: str,
    metric: str,
    values: np.ndarray,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    weights: np.ndarray,
    failures: list[str],
    comparisons: dict[str, object],
) -> None:
    left = _weighted_nonzero_incidence(values[left_rows], weights[left_rows])
    right = _weighted_nonzero_incidence(values[right_rows], weights[right_rows])
    record: dict[str, object] = {
        "status": "tested",
        "metric": metric,
        "asec_incidence": left,
        "acs_incidence": right,
        "nonzero_rows": {
            "asec": int((values[left_rows] != 0.0).sum()),
            "acs": int((values[right_rows] != 0.0).sum()),
        },
    }
    comparisons[label] = record
    if left == 0.0 and right == 0.0:
        failures.append(
            f"{label}: zero weighted incidence on both origins with adequate "
            "support; the registered comparison is dead."
        )
        record["status"] = "dead_comparison"
        return
    ratio = math.inf if left == 0.0 else right / left
    record["incidence_ratio_acs_over_asec"] = ratio if math.isfinite(ratio) else "inf"
    lower, upper = _BATTERY_INCIDENCE_RATIO_BOUNDS
    if not lower <= ratio <= upper:
        failures.append(
            f"{label}: weighted incidence ratio {ratio:.6g} is outside "
            f"[{lower:.6g}, {upper:.6g}] (asec={left:.6g}, acs={right:.6g})."
        )


def _battery_sign_separated_comparison(
    *,
    label: str,
    values: np.ndarray,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    weights: np.ndarray,
    failures: list[str],
    comparisons: dict[str, object],
) -> None:
    record: dict[str, object] = {
        "status": "tested",
        "metric": "monetary_sign_separated",
        "legs": {},
    }
    comparisons[label] = record
    lower, upper = _BATTERY_INCIDENCE_RATIO_BOUNDS
    for leg_name, leg_mask in (
        ("positive", values > 0.0),
        ("negative", values < 0.0),
    ):
        left_leg = leg_mask & left_rows
        right_leg = leg_mask & right_rows
        left_incidence = _weighted_mask_incidence(
            left_leg[left_rows], weights[left_rows]
        )
        right_incidence = _weighted_mask_incidence(
            right_leg[right_rows], weights[right_rows]
        )
        leg_record: dict[str, object] = {
            "asec_incidence": left_incidence,
            "acs_incidence": right_incidence,
            "nonzero_rows": {
                "asec": int(left_leg.sum()),
                "acs": int(right_leg.sum()),
            },
        }
        record["legs"][leg_name] = leg_record
        if left_incidence == 0.0 and right_incidence == 0.0:
            leg_record["status"] = "absent_on_both_origins"
            continue
        ratio = math.inf if left_incidence == 0.0 else right_incidence / left_incidence
        leg_record["incidence_ratio_acs_over_asec"] = (
            ratio if math.isfinite(ratio) else "inf"
        )
        if not lower <= ratio <= upper:
            failures.append(
                f"{label}/{leg_name}: weighted {leg_name}-leg incidence ratio "
                f"{ratio:.6g} is outside [{lower:.6g}, {upper:.6g}] "
                f"(asec={left_incidence:.6g}, acs={right_incidence:.6g})."
            )
        if (
            int(left_leg.sum()) < _BATTERY_SUPPORT_PROFILE.min_effective_support
            or int(right_leg.sum()) < _BATTERY_SUPPORT_PROFILE.min_effective_support
        ):
            leg_record["quantile_envelope"] = "leg_insufficient_support"
            continue
        left_quantiles = _battery_conditional_quantiles(
            np.abs(values[left_leg]), weights[left_leg]
        )
        right_quantiles = _battery_conditional_quantiles(
            np.abs(values[right_leg]), weights[right_leg]
        )
        distance = _battery_quantile_envelope_distance(left_quantiles, right_quantiles)
        leg_record["quantile_envelope_distance"] = distance
        if distance > _BATTERY_QUANTILE_ENVELOPE_TOLERANCE:
            failures.append(
                f"{label}/{leg_name}: conditional-quantile envelope distance "
                f"{distance:.6g} exceeds "
                f"{_BATTERY_QUANTILE_ENVELOPE_TOLERANCE:.6g}."
            )


def _battery_categorical_comparison(
    *,
    label: str,
    series: pd.Series,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    weights: np.ndarray,
    failures: list[str],
    comparisons: dict[str, object],
) -> None:
    values = series.to_numpy(dtype=object)
    distributions: dict[str, dict[str, float]] = {}
    for origin, rows in (("asec", left_rows), ("acs", right_rows)):
        origin_weights = weights[rows]
        total = float(origin_weights.sum())
        shares: dict[str, float] = {}
        for value, weight in zip(values[rows], origin_weights, strict=True):
            shares[str(value)] = shares.get(str(value), 0.0) + float(weight)
        distributions[origin] = {
            category: share / total for category, share in shares.items()
        }
    categories = sorted(set(distributions["asec"]) | set(distributions["acs"]))
    distance = 0.5 * sum(
        abs(
            distributions["asec"].get(category, 0.0)
            - distributions["acs"].get(category, 0.0)
        )
        for category in categories
    )
    comparisons[label] = {
        "status": "tested",
        "metric": "categorical_tvd",
        "total_variation_distance": distance,
        "category_shares": distributions,
    }
    if distance > _BATTERY_CATEGORICAL_TVD_TOLERANCE:
        failures.append(
            f"{label}: categorical total-variation distance {distance:.6g} "
            f"exceeds {_BATTERY_CATEGORICAL_TVD_TOLERANCE:.6g}."
        )


def _weighted_nonzero_incidence(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(weights[values != 0.0].sum() / total)


def _weighted_mask_incidence(mask: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(weights[mask].sum() / total)


def _battery_conditional_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    positions = np.minimum(
        np.searchsorted(cumulative, np.asarray(_BATTERY_QUANTILES), side="left"),
        len(sorted_values) - 1,
    )
    return sorted_values[positions]


def _battery_quantile_envelope_distance(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    denominator = np.abs(left) + np.abs(right)
    distances = np.divide(
        2.0 * np.abs(left - right),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    return float(np.max(distances))
