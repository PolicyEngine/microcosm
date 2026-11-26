"""Employer and direct-purchase coverage of ACS rows (ACS local lane).

``has_esi`` and the two direct-purchase flags
(``has_marketplace_health_coverage_at_interview``,
``has_non_marketplace_direct_purchase_health_coverage_at_interview``) reach
ACS rows as draws of the shared transfer, on age, sex, State and income,
although the ACS asks both items (microcosm#1720). ``has_esi`` bars the
premium tax credit and CHIP and the self-employed health insurance deduction,
so the drawn flag made those pools noisy on ACS rows.

This lane-own step writes the ACS items over the draws, on the ACS-only frame
after every transfer and before pooling, the pattern of the native SNAP
receipt anchor (:mod:`~microcosm.build.us_runtime.acs_local_receipt_anchors`).
Writing them before the transfer would take ``has_esi`` out of its chained
batch and move every other draw of the batch; after it, the shared plan, its
chain and every other draw are unchanged.

- **has_esi** is ACS ``HINS1 == 1``: insurance through a current or former
  employer or union, of the person or another family member, at interview.
  The donor reads ASEC ``NOW_GRP``, current employment-based coverage
  (:mod:`~microcosm.build.us_runtime.cps_carried`).
- **The two direct-purchase flags** follow ACS ``HINS2`` (insurance purchased
  directly from an insurance company), which does not tell Marketplace
  coverage from other direct purchase. ``HINS2 == 2``: both False.
  ``HINS2 == 1``: ACS ``HIMRKS`` (Subsidized Marketplace Coverage, loaded as
  ``ACS_HIMRKS``) code 1, direct purchase with a premium subsidy, which only
  the Marketplace pays, sets the Marketplace flag True and keeps the drawn
  non-Marketplace flag (ruled 25 Nov 2026). Under codes 0 and 2 and a blank
  the transfer's draws stand where one of them is True; where both are False,
  one keyed draw picks the type (Marketplace only, non-Marketplace only, or
  both) at the donor ASEC role's conditional shares among direct purchasers,
  by age band (under 65, 65 and over: at 65+ direct purchase is mostly
  Medigap). The draw is keyed on ``acs_2024_1yr:SERIALNO:SPORDER`` and the
  build seed. Code 2 cannot tell Marketplace coverage at full price from
  other direct purchase, and code 0's universe is not documented, so no rule
  reads either. This choice among direct purchasers is one function,
  :func:`choose_acs_direct_purchase_types`.
- **Blanks.** The 2024 PUMS dictionary lists no blank code for HINS1, HINS2
  or HIMRKS; a blank that arrives anyway keeps the earlier rule and is
  counted.
- **The ACA anchor** (ruled 25 Nov 2026, "Yes, with thinning"). The stage
  writes the ACA manifest stage's own anchor column,
  ``reported_has_subsidized_marketplace_health_coverage_at_interview`` (read
  from the manifest's ``reported_true_anchor``), on the ACS rows: True where
  ``HINS2 == 1`` and HIMRKS is 1, allocated values included; False on every
  other ACS row. Donor rows are not written: the donor reads no subsidized
  item (``NOW_MRKS`` is unread), and the national tool reads the absent
  column as False there. The release's ACA stage thins anchors above a
  state's residual. ``HIMRKS`` also stays on the staged ACS rows as a raw
  source item.

The step must run before the self-employed premium attribution
(:mod:`~microcosm.build.us_runtime.acs_local_se_health_premiums`), which
reads ``has_esi``. It records a receipt, with the agreement of each native
value with the draw it replaces, and grades nothing. It refuses only a frame
without the columns it reads, and malformed source codes.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import load_source_manifest
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.us_runtime.acs_inputs import _nullable_source_codes
from microcosm.build.us_runtime.acs_local_ssi_disability import _acs_draw_keys
from microcosm.build.us_runtime.acs_pums import (
    ACS_2024_1YR_SPINE,
    ACS_SOURCE_COLUMN_RENAMES,
)
from microcosm.build.us_runtime.acs_transfer import resolve_acs_donor_channel
from microcosm.build.us_runtime.support_provenance import BASE_ASEC_SUPPORT_CHANNEL
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_DIRECT_PURCHASE_CHOICE_SOURCE",
    "ACS_DIRECT_PURCHASE_TYPES",
    "ACS_LOCAL_COVERAGE_ANCHOR_COLUMNS",
    "ACS_LOCAL_COVERAGE_ANCHORS_ISSUE",
    "ACS_LOCAL_COVERAGE_ANCHORS_METHOD",
    "ACS_LOCAL_COVERAGE_SOURCE_COLUMNS",
    "acs_local_aca_reported_true_anchor",
    "acs_local_direct_purchase_donor_shares",
    "choose_acs_direct_purchase_types",
    "with_acs_local_coverage_anchors",
]

ACS_LOCAL_COVERAGE_ANCHORS_ISSUE = "microcosm#1720"
ACS_LOCAL_COVERAGE_ANCHORS_METHOD = (
    "native_hins1_hins2_after_every_transfer_direct_purchase_type_draw"
)
_ESI = "has_esi"
_MARKETPLACE = "has_marketplace_health_coverage_at_interview"
_OTHER_DIRECT = "has_non_marketplace_direct_purchase_health_coverage_at_interview"
ACS_LOCAL_COVERAGE_ANCHOR_COLUMNS: tuple[str, ...] = (_ESI, _MARKETPLACE, _OTHER_DIRECT)
_EMPLOYER_ITEM = "HINS1"
_DIRECT_ITEM = "HINS2"
#: HIMRKS and its allocation flag, as the loader renames them.
_SUBSIDY_ITEM = ACS_SOURCE_COLUMN_RENAMES["HIMRKS"]
_SUBSIDY_ALLOCATION = ACS_SOURCE_COLUMN_RENAMES["FHIMRKSP"]
#: HIMRKS 1: direct purchase coverage, with premium subsidy.
_SUBSIDIZED_CODE = 1
#: HIMRKS 1 and 2 are direct purchase codes; a conflict with HINS2 == 2.
_DIRECT_PURCHASE_CODES = (1, 2)
ACS_LOCAL_COVERAGE_SOURCE_COLUMNS: tuple[str, ...] = (
    _EMPLOYER_ITEM,
    _DIRECT_ITEM,
    _SUBSIDY_ITEM,
)
_AGE = "age"
#: The type of direct purchase as the two donor flags record it.
ACS_DIRECT_PURCHASE_TYPES: Mapping[str, tuple[bool, bool]] = {
    "marketplace_only": (True, False),
    "non_marketplace_only": (False, True),
    "both": (True, True),
}
#: Direct purchase at 65+ is mostly Medigap, so the shares are by age band.
_AGE_BANDS: tuple[tuple[str, float, float], ...] = (
    ("under_65", 0.0, 65.0),
    ("65_plus", 65.0, np.inf),
)
_DRAW_KEY_FORMAT = f"{ACS_2024_1YR_SPINE}:SERIALNO:SPORDER"
_DRAW_SALT = "acs_local_direct_purchase_type"
#: Where the Marketplace-versus-other choice comes from (ruled 25 Nov 2026:
#: HIMRKS splits the direct-purchase flags).
ACS_DIRECT_PURCHASE_CHOICE_SOURCE = (
    "himrks_1_marketplace_else_transfer_draws_else_donor_asec_type_shares_by_age_band"
)
_SOURCE_LABELS: Mapping[str, str] = {
    _EMPLOYER_ITEM: "Insurance through a current or former employer or union",
    _DIRECT_ITEM: "Insurance purchased directly from an insurance company",
    _SUBSIDY_ITEM: (
        "HIMRKS, Subsidized Marketplace Coverage: 0 categorically ineligible "
        "for subsidy, 1 direct purchase coverage with premium subsidy, 2 "
        "direct purchase coverage without a premium subsidy"
    ),
    _SUBSIDY_ALLOCATION: "FHIMRKSP, Subsidized Marketplace Coverage allocation flag",
}
_NOT_USED: Mapping[str, str] = {
    "HIMRKS code 0": "its universe is not documented; no rule reads it",
    "HIMRKS code 2": (
        "direct purchase without a subsidy can be on or off the Marketplace; "
        "the earlier rule applies"
    ),
}
#: The US manifest stage whose own anchor column the stage writes.
_ACA_STAGE = "aca_marketplace_inputs"
_ANCHOR_PARAMETER = "reported_true_anchor"


@cache
def acs_local_aca_reported_true_anchor() -> str:
    """The ACA manifest stage's own anchor column, from the US manifest.

    Raises:
        ValueError: If the manifest has no ACA stage or not exactly one
            anchor column.
    """

    stage = (
        load_source_manifest(files("microcosm.build.us").joinpath("source_stages.json"))
        .stage_map()
        .get(_ACA_STAGE)
    )
    if stage is None:
        raise ValueError(
            f"The US source manifest declares no {_ACA_STAGE!r} stage "
            f"({ACS_LOCAL_COVERAGE_ANCHORS_ISSUE})."
        )
    anchors = {
        str(operation.parameters[_ANCHOR_PARAMETER])
        for operation in stage.operations
        if _ANCHOR_PARAMETER in operation.parameters
    }
    if len(anchors) != 1:
        raise ValueError(
            f"The {_ACA_STAGE!r} stage must declare one {_ANCHOR_PARAMETER}, "
            f"got {sorted(anchors)} ({ACS_LOCAL_COVERAGE_ANCHORS_ISSUE})."
        )
    return anchors.pop()


def _flags(values: pd.Series, column: str) -> np.ndarray:
    """A complete boolean column as an array; refuses anything else."""

    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError(
            f"ACS {column} must be a complete boolean after the shared transfer "
            f"({ACS_LOCAL_COVERAGE_ANCHORS_ISSUE})."
        )
    return values.astype(bool).to_numpy(dtype=bool)


def _weighted_share(weights: np.ndarray, flags: np.ndarray) -> float:
    total = float(weights.sum())
    return float(weights[flags].sum()) / total if total > 0 else 0.0


def _band_masks(age: np.ndarray) -> dict[str, np.ndarray]:
    return {name: (age >= low) & (age < high) for name, low, high in _AGE_BANDS}


def _type_shares(
    marketplace: np.ndarray, other: np.ndarray, weights: np.ndarray
) -> dict[str, float] | None:
    """Weighted type shares among direct purchasers, or None without any."""

    purchasers = marketplace | other
    total = float(weights[purchasers].sum())
    if total <= 0:
        return None
    return {
        name: float(weights[(marketplace == m) & (other == o) & purchasers].sum())
        / total
        for name, (m, o) in ACS_DIRECT_PURCHASE_TYPES.items()
    }


def acs_local_direct_purchase_donor_shares(donor: Frame) -> dict[str, Any]:
    """The donor ASEC role's direct-purchase type shares, by age band.

    Measured ``NOW_MRK``/``NOW_NONM`` on the ASEC observation role, weighted
    by person weight; the PUF clones are not read. A band without a direct
    purchaser falls back to the all-ages shares (recorded).

    Raises:
        ValueError: If the donor lacks the ASEC role, an age or a flag.
    """

    try:
        asec, channel = resolve_acs_donor_channel(donor, BASE_ASEC_SUPPORT_CHANNEL)
    except ValueError as exc:
        raise ValueError(
            "The ACS direct-purchase draw needs the donor's ASEC observation "
            f"role ({ACS_LOCAL_COVERAGE_ANCHORS_ISSUE}): {exc}"
        ) from exc
    person = asec.table("person")
    missing = [
        column for column in (_AGE, _MARKETPLACE, _OTHER_DIRECT) if column not in person
    ]
    if missing:
        raise ValueError(
            f"The ASEC donor lacks {missing} ({ACS_LOCAL_COVERAGE_ANCHORS_ISSUE})."
        )
    age = pd.to_numeric(person[_AGE], errors="coerce").to_numpy(dtype=np.float64)
    marketplace = _flags(person[_MARKETPLACE], f"donor {_MARKETPLACE}")
    other = _flags(person[_OTHER_DIRECT], f"donor {_OTHER_DIRECT}")
    weights = np.asarray(asec.resolve_weights("person").values, dtype=np.float64)
    overall = _type_shares(marketplace, other, weights)
    bands: dict[str, Any] = {}
    for name, mask in _band_masks(age).items():
        shares = _type_shares(marketplace[mask], other[mask], weights[mask])
        bands[name] = {
            "shares": shares if shares is not None else overall,
            "fallback_to_all_ages": shares is None,
            "purchaser_rows": int((mask & (marketplace | other)).sum()),
        }
    return {
        "channel": channel,
        "person_rows": int(len(person)),
        "all_ages": overall,
        "bands": bands,
    }


def _draw_types(
    uniforms: np.ndarray, shares: Mapping[str, float]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Each uniform's type by the cumulative shares, in a fixed order."""

    names = list(ACS_DIRECT_PURCHASE_TYPES)
    edges = np.cumsum([shares[name] for name in names])
    index = np.minimum(np.searchsorted(edges, uniforms, side="right"), len(names) - 1)
    chosen = [names[i] for i in index]
    marketplace = np.asarray([ACS_DIRECT_PURCHASE_TYPES[c][0] for c in chosen])
    other = np.asarray([ACS_DIRECT_PURCHASE_TYPES[c][1] for c in chosen])
    return marketplace.astype(bool), other.astype(bool), chosen


def _agreement(native: np.ndarray, drawn: np.ndarray) -> dict[str, int]:
    return {
        "both_true_rows": int((native & drawn).sum()),
        "native_only_rows": int((native & ~drawn).sum()),
        "drawn_only_rows": int((~native & drawn).sum()),
        "both_false_rows": int((~native & ~drawn).sum()),
    }


def _code_rows(codes: pd.Series) -> dict[str, int]:
    counts = {
        str(int(code)): int(count)
        for code, count in codes.value_counts().sort_index().items()
    }
    counts["blank"] = int(codes.isna().sum())
    return counts


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def choose_acs_direct_purchase_types(
    rows: pd.DataFrame,
    household: pd.DataFrame,
    *,
    drawn_marketplace: np.ndarray,
    drawn_other: np.ndarray,
    donor_shares: Mapping[str, Any],
    seed: int,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The Marketplace-versus-other choice for ACS direct purchasers.

    As ruled on 25 Nov 2026, ``HIMRKS`` (``ACS_HIMRKS``) code 1, direct
    purchase coverage with a premium subsidy, sets the Marketplace flag True;
    the drawn non-Marketplace flag stands, since HIMRKS says nothing about
    another direct plan. Under codes 0 and 2 and a blank the earlier rule
    holds: the shared transfer's draws stand where either flag is True; where
    both are False, one draw keyed on ``acs_2024_1yr:SERIALNO:SPORDER`` and
    ``seed`` picks the type at the donor ASEC role's shares in the person's
    age band (``donor_shares``, from
    :func:`acs_local_direct_purchase_donor_shares`); a band without shares
    leaves both False and counts the row as unresolved. Code 0's universe is
    not documented, so no rule reads it. The HINS2 == 2 rule, ``has_esi`` and
    the rest of the stage do not depend on this function.

    Args:
        rows: The HINS2 == 1 person rows, with ``ACS_HIMRKS``, ``age``,
            ``SPORDER`` and ``person_household_id`` (and ``ACS_FHIMRKSP`` if
            the source has it).
        household: The household table, for ``SERIALNO``.
        drawn_marketplace: The transferred Marketplace flag of ``rows``.
        drawn_other: The transferred non-Marketplace direct-purchase flag of
            ``rows``.
        donor_shares: The donor's type shares by age band.
        seed: The build seed.
        weights: The person weights of ``rows``, for the receipt's shares.

    Returns:
        The two flags for ``rows``, in their order, and the receipt of the
        choice (row counts and weighted shares only).

    Raises:
        ValueError: If a HIMRKS or FHIMRKSP code is malformed, or a row needs
            a draw and its key cannot be built.
    """

    subsidy = _nullable_source_codes(
        rows[_SUBSIDY_ITEM].rename("HIMRKS"), minimum=0, maximum=2
    )
    subsidized = subsidy.eq(_SUBSIDIZED_CODE).to_numpy(dtype=bool)
    allocated: np.ndarray | None = None
    if _SUBSIDY_ALLOCATION in rows:
        allocated = (
            _nullable_source_codes(
                rows[_SUBSIDY_ALLOCATION].rename("FHIMRKSP"), minimum=0, maximum=1
            )
            .eq(1)
            .to_numpy(dtype=bool)
        )
    weights = np.asarray(weights, dtype=np.float64)
    drawn_marketplace = np.asarray(drawn_marketplace, dtype=bool)
    drawn_other = np.asarray(drawn_other, dtype=bool)
    marketplace = drawn_marketplace | subsidized
    other = drawn_other.copy()
    to_draw = ~subsidized & ~drawn_marketplace & ~drawn_other
    drawn_types: dict[str, int] = dict.fromkeys(ACS_DIRECT_PURCHASE_TYPES, 0)
    unresolved = np.zeros(len(rows), dtype=bool)
    if to_draw.any():
        positions = np.flatnonzero(to_draw)
        try:
            keys = _acs_draw_keys(household, rows.iloc[positions])
        except ValueError as exc:
            raise ValueError(
                f"ACS direct-purchase type draw: {exc} "
                f"({ACS_LOCAL_COVERAGE_ANCHORS_ISSUE})"
            ) from exc
        uniforms = stable_identity_uniforms(
            keys.tolist(), seed=int(seed), salt=_DRAW_SALT
        )
        age = pd.to_numeric(rows[_AGE], errors="coerce").to_numpy(dtype=np.float64)
        for name, mask in _band_masks(age[positions]).items():
            if not mask.any():
                continue
            chosen_rows = positions[mask]
            shares = donor_shares["bands"][name]["shares"]
            if shares is None:
                unresolved[chosen_rows] = True
                continue
            band_marketplace, band_other, chosen = _draw_types(uniforms[mask], shares)
            marketplace[chosen_rows] = band_marketplace
            other[chosen_rows] = band_other
            for choice in chosen:
                drawn_types[choice] += 1
    codes = subsidy.to_numpy(dtype=np.float64, na_value=np.nan)
    himrks: dict[str, Any] = {
        "codes": _code_rows(subsidy),
        "weighted_shares": {
            **{
                str(code): _weighted_share(weights, codes == code) for code in (0, 1, 2)
            },
            "blank": _weighted_share(weights, np.isnan(codes)),
        },
        "allocation_flag_loaded": allocated is not None,
        "allocated_rows": int(allocated.sum()) if allocated is not None else None,
        "allocated_code_1_rows": (
            int((allocated & subsidized).sum()) if allocated is not None else None
        ),
        "code_1": {
            "rows": int(subsidized.sum()),
            "drawn_marketplace_true_rows": int((subsidized & drawn_marketplace).sum()),
            "drawn_marketplace_false_rows": int(
                (subsidized & ~drawn_marketplace).sum()
            ),
            "drawn_marketplace_weighted_share": _weighted_share(
                weights[subsidized], drawn_marketplace[subsidized]
            ),
            "drawn_non_marketplace_kept_rows": int((subsidized & drawn_other).sum()),
            "drawn_non_marketplace_only_became_both_rows": int(
                (subsidized & drawn_other & ~drawn_marketplace).sum()
            ),
        },
    }
    receipt: dict[str, Any] = {
        "source": ACS_DIRECT_PURCHASE_CHOICE_SOURCE,
        "rows": int(len(rows)),
        "himrks": himrks,
        "set_marketplace_by_code_1_rows": int((subsidized & ~drawn_marketplace).sum()),
        "draw_kept_rows": int((~subsidized & ~to_draw).sum()),
        "type_drawn_rows": int((to_draw & ~unresolved).sum()),
        "type_drawn_by_type": drawn_types,
        "unresolved_rows": int(unresolved.sum()),
        "donor_shares": dict(donor_shares),
        "draw_key": _DRAW_KEY_FORMAT,
        "draw_salt": _DRAW_SALT,
        "seed": int(seed),
    }
    return marketplace, other, receipt


def with_acs_local_coverage_anchors(
    frame: Frame,
    donor: Frame,
    *,
    seed: int,
) -> tuple[Frame, dict[str, Any]]:
    """Write ACS HINS1 and HINS2 over the transferred coverage draws.

    Args:
        frame: The ACS-only frame after every transfer, before pooling. Its
            person table carries HINS1, HINS2, ``age``, ``SPORDER``,
            ``person_household_id`` and the three transferred flags; its
            household table carries ``SERIALNO``.
        donor: The ASEC-by-PUF base, for the direct-purchase type shares.
        seed: The build seed for the keyed type draw.

    Returns:
        The frame with the native values and a JSON-ready receipt.

    Raises:
        ValueError: If the frame is not US-schema, a column is absent, a code
            is malformed, a transferred flag is not a complete boolean, or the
            donor cannot give the type shares.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("ACS coverage anchors require the US schema.")
    person = frame.table("person")
    missing = [
        column
        for column in (
            *ACS_LOCAL_COVERAGE_SOURCE_COLUMNS,
            *ACS_LOCAL_COVERAGE_ANCHOR_COLUMNS,
            _AGE,
        )
        if column not in person
    ]
    if missing:
        raise ValueError(
            f"ACS coverage anchors require person column(s) {missing}: HINS1, "
            "HINS2 and HIMRKS (as ACS_HIMRKS) from the pinned ACS PUMS source, "
            f"the flags from the shared transfer ({ACS_LOCAL_COVERAGE_ANCHORS_ISSUE})."
        )
    employer = _nullable_source_codes(person[_EMPLOYER_ITEM], minimum=1, maximum=2)
    direct = _nullable_source_codes(person[_DIRECT_ITEM], minimum=1, maximum=2)
    subsidy = _nullable_source_codes(
        person[_SUBSIDY_ITEM].rename("HIMRKS"), minimum=0, maximum=2
    )
    allocated: np.ndarray | None = None
    if _SUBSIDY_ALLOCATION in person:
        allocated = (
            _nullable_source_codes(
                person[_SUBSIDY_ALLOCATION].rename("FHIMRKSP"), minimum=0, maximum=1
            )
            .eq(1)
            .to_numpy(dtype=bool)
        )
    drawn_esi = _flags(person[_ESI], _ESI)
    drawn_marketplace = _flags(person[_MARKETPLACE], _MARKETPLACE)
    drawn_other = _flags(person[_OTHER_DIRECT], _OTHER_DIRECT)
    age = pd.to_numeric(person[_AGE], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(age).any():
        raise ValueError("ACS coverage anchors require every age.")

    employer_coded = employer.notna().to_numpy(dtype=bool)
    esi = np.where(employer_coded, employer.eq(1).to_numpy(dtype=bool), drawn_esi)

    no_direct = direct.eq(2).to_numpy(dtype=bool)
    yes_direct = direct.eq(1).to_numpy(dtype=bool)
    marketplace = np.where(no_direct, False, drawn_marketplace)
    other = np.where(no_direct, False, drawn_other)
    # The Marketplace-versus-other choice among direct purchasers is one
    # function; HIMRKS code 1 decides it there (ruled 25 Nov 2026).
    yes_rows = np.flatnonzero(yes_direct)
    chosen_marketplace, chosen_other, choice = choose_acs_direct_purchase_types(
        person.iloc[yes_rows],
        frame.table("household"),
        drawn_marketplace=drawn_marketplace[yes_rows],
        drawn_other=drawn_other[yes_rows],
        donor_shares=acs_local_direct_purchase_donor_shares(donor),
        seed=seed,
        weights=np.asarray(frame.resolve_weights("person").values, dtype=np.float64)[
            yes_rows
        ],
    )
    marketplace[yes_rows] = chosen_marketplace
    other[yes_rows] = chosen_other
    # The ACA stage's fixed points: direct purchase with a premium subsidy.
    anchor_column = acs_local_aca_reported_true_anchor()
    subsidized = subsidy.eq(_SUBSIDIZED_CODE).to_numpy(dtype=bool)
    anchor = yes_direct & subsidized

    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables.update({name: frame.link(name) for name in frame.links})
    updated = person.copy()
    updated[_ESI] = esi.astype(bool)
    updated[_MARKETPLACE] = marketplace.astype(bool)
    updated[_OTHER_DIRECT] = other.astype(bool)
    updated[anchor_column] = anchor.astype(bool)
    tables["person"] = updated
    result = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )

    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    receipt: dict[str, Any] = {
        "issue": ACS_LOCAL_COVERAGE_ANCHORS_ISSUE,
        "method": ACS_LOCAL_COVERAGE_ANCHORS_METHOD,
        "spine": ACS_2024_1YR_SPINE,
        "sources": dict(_SOURCE_LABELS),
        "acs_persons": int(len(person)),
        "hins1_rows": _code_rows(employer),
        "hins2_rows": _code_rows(direct),
        _ESI: {
            "rule": "HINS1 == 1; a blank keeps the transfer's draw",
            "agreement_with_draw": _agreement(
                esi[employer_coded], drawn_esi[employer_coded]
            ),
            "changed_rows": int((esi != drawn_esi).sum()),
            "native_weighted_share": _weighted_share(weights, esi),
            "drawn_weighted_share": _weighted_share(weights, drawn_esi),
        },
        "direct_purchase": {
            "rule": (
                "HINS2 == 2: both flags False; HINS2 == 1: "
                "choose_acs_direct_purchase_types (see choice.source); a "
                "blank keeps the draws"
            ),
            "hins2_no_rows": int(no_direct.sum()),
            "set_false_from_draw": {
                _MARKETPLACE: int((no_direct & drawn_marketplace).sum()),
                _OTHER_DIRECT: int((no_direct & drawn_other).sum()),
            },
            "hins2_yes_rows": int(yes_direct.sum()),
            # HINS2 == 2 wins over a HIMRKS direct-purchase code: both flags
            # stay False. Counted, never resolved the other way.
            "himrks_conflicts": {
                "hins2_no_himrks_direct_rows": {
                    str(code): int((no_direct & subsidy.eq(code).to_numpy()).sum())
                    for code in _DIRECT_PURCHASE_CODES
                },
                "hins2_no_himrks_direct_weighted_share": _weighted_share(
                    weights,
                    no_direct & subsidy.isin(_DIRECT_PURCHASE_CODES).to_numpy(),
                ),
                "hins2_blank_himrks_direct_rows": int(
                    (
                        direct.isna().to_numpy()
                        & subsidy.isin(_DIRECT_PURCHASE_CODES).to_numpy()
                    ).sum()
                ),
            },
            "choice": choice,
            "weighted_shares": {
                column: {
                    "native": _weighted_share(weights, final),
                    "drawn": _weighted_share(weights, drawn),
                }
                for column, final, drawn in (
                    (_MARKETPLACE, marketplace, drawn_marketplace),
                    (_OTHER_DIRECT, other, drawn_other),
                )
            },
        },
        "aca_anchor": {
            "column": anchor_column,
            "manifest_stage": _ACA_STAGE,
            "rule": (
                "HINS2 == 1 and HIMRKS == 1, allocated values included; False "
                "on every other ACS row; donor rows are not written"
            ),
            "replaced_existing_column": anchor_column in person,
            "true_rows": int(anchor.sum()),
            "weighted_share": _weighted_share(weights, anchor),
            # Census-allocated code 1 values are anchors too (not ruled).
            "allocation_flag_loaded": allocated is not None,
            "allocated_true_rows": (
                int((anchor & allocated).sum()) if allocated is not None else None
            ),
            "allocated_weighted_share_of_true": (
                _weighted_share(weights[anchor], allocated[anchor])
                if allocated is not None
                else None
            ),
            # HIMRKS 1 where HINS2 is 2 (or blank): no anchor; counted.
            "conflicts": {
                "himrks_1_hins2_no_rows": int((no_direct & subsidized).sum()),
                "himrks_1_hins2_no_weighted_share": _weighted_share(
                    weights, no_direct & subsidized
                ),
                "himrks_1_hins2_blank_rows": int(
                    (direct.isna().to_numpy() & subsidized).sum()
                ),
            },
        },
        "not_used": dict(_NOT_USED),
    }
    return result, _json_ready(receipt)
