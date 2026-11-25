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
  ``HINS2 == 1``: the transfer's draws stand where one of them is True;
  where both are False, one keyed draw picks the type (Marketplace only,
  non-Marketplace only, or both) at the donor ASEC role's conditional shares
  among direct purchasers, by age band (under 65, 65 and over: at 65+ direct
  purchase is mostly Medigap). The draw is keyed on
  ``acs_2024_1yr:SERIALNO:SPORDER`` and the build seed. This choice among
  direct purchasers is one function, :func:`choose_acs_direct_purchase_types`,
  so a ruling on ``HIMRKS`` changes that function and nothing else in the
  stage.
- **Blanks.** The 2024 PUMS dictionary lists no blank code for HINS1 or
  HINS2; a blank that arrives anyway keeps the transfer's draw and is counted.
- **Not loaded:** ``HIMRKS`` (Subsidized Marketplace Coverage), which could
  split the direct-purchase types, awaits a ruling.

The step must run before the self-employed premium attribution
(:mod:`~microcosm.build.us_runtime.acs_local_se_health_premiums`), which
reads ``has_esi``. It records a receipt, with the agreement of each native
value with the draw it replaces, and grades nothing. It refuses only a frame
without the columns it reads, and malformed source codes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.us_runtime.acs_inputs import _nullable_source_codes
from microcosm.build.us_runtime.acs_local_ssi_disability import _acs_draw_keys
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
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
ACS_LOCAL_COVERAGE_SOURCE_COLUMNS: tuple[str, ...] = (_EMPLOYER_ITEM, _DIRECT_ITEM)
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
#: Where the Marketplace-versus-other choice comes from today. A ruling on
#: HIMRKS changes :func:`choose_acs_direct_purchase_types` and this name.
ACS_DIRECT_PURCHASE_CHOICE_SOURCE = (
    "transfer_draws_else_donor_asec_type_shares_by_age_band"
)
_SOURCE_LABELS: Mapping[str, str] = {
    _EMPLOYER_ITEM: "Insurance through a current or former employer or union",
    _DIRECT_ITEM: "Insurance purchased directly from an insurance company",
}
_NOT_LOADED: Mapping[str, str] = {
    "HIMRKS": (
        "Subsidized Marketplace Coverage (2024 PUMS: 0 categorically ineligible "
        "for subsidy, 1 direct purchase with premium subsidy, 2 without); it "
        "could split the direct-purchase types and awaits a ruling"
    ),
}


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
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The Marketplace-versus-other choice for ACS direct purchasers.

    The one function a ruling on HIMRKS (Subsidized Marketplace Coverage,
    which the loader does not read) changes; the HINS2 == 2 rule, ``has_esi``
    and the rest of the stage stay. Today the shared transfer's draws stand
    where either flag is True. Where both are False, one draw keyed on
    ``acs_2024_1yr:SERIALNO:SPORDER`` and ``seed`` picks the type at the
    donor ASEC role's shares in the person's age band (``donor_shares``, from
    :func:`acs_local_direct_purchase_donor_shares`); a band without shares
    leaves both False and counts the row as unresolved.

    Args:
        rows: The HINS2 == 1 person rows, with ``age``, ``SPORDER`` and
            ``person_household_id``.
        household: The household table, for ``SERIALNO``.
        drawn_marketplace: The transferred Marketplace flag of ``rows``.
        drawn_other: The transferred non-Marketplace direct-purchase flag of
            ``rows``.
        donor_shares: The donor's type shares by age band.
        seed: The build seed.

    Returns:
        The two flags for ``rows``, in their order, and the receipt of the
        choice.

    Raises:
        ValueError: If a row needs a draw and its key cannot be built.
    """

    drawn_marketplace = np.asarray(drawn_marketplace, dtype=bool)
    drawn_other = np.asarray(drawn_other, dtype=bool)
    marketplace = drawn_marketplace.copy()
    other = drawn_other.copy()
    to_draw = ~drawn_marketplace & ~drawn_other
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
    receipt: dict[str, Any] = {
        "source": ACS_DIRECT_PURCHASE_CHOICE_SOURCE,
        "rows": int(len(rows)),
        "draw_kept_rows": int((~to_draw).sum()),
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
            f"ACS coverage anchors require person column(s) {missing}: HINS1 and "
            "HINS2 from the pinned ACS PUMS source, the flags from the shared "
            f"transfer ({ACS_LOCAL_COVERAGE_ANCHORS_ISSUE})."
        )
    employer = _nullable_source_codes(person[_EMPLOYER_ITEM], minimum=1, maximum=2)
    direct = _nullable_source_codes(person[_DIRECT_ITEM], minimum=1, maximum=2)
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
    # The Marketplace-versus-other choice among direct purchasers is the one
    # place a HIMRKS ruling would change.
    yes_rows = np.flatnonzero(yes_direct)
    chosen_marketplace, chosen_other, choice = choose_acs_direct_purchase_types(
        person.iloc[yes_rows],
        frame.table("household"),
        drawn_marketplace=drawn_marketplace[yes_rows],
        drawn_other=drawn_other[yes_rows],
        donor_shares=acs_local_direct_purchase_donor_shares(donor),
        seed=seed,
    )
    marketplace[yes_rows] = chosen_marketplace
    other[yes_rows] = chosen_other

    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables.update({name: frame.link(name) for name in frame.links})
    updated = person.copy()
    updated[_ESI] = esi.astype(bool)
    updated[_MARKETPLACE] = marketplace.astype(bool)
    updated[_OTHER_DIRECT] = other.astype(bool)
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
        "not_loaded": dict(_NOT_LOADED),
    }
    return result, _json_ready(receipt)
