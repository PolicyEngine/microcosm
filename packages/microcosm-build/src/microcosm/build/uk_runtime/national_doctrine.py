"""UK national calibration doctrine: reviewed solve constants for #623.

The national release path exposes calibration knobs only as receipted per-run
overrides on the calibration seam driver; defaults remain the reviewed
constants. Every solver option that the first Ledger-backed national
calibration currently inherits is declared here so a future default change
must edit this module and its pinned tests, forcing review before any armed
run.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from types import MappingProxyType

import numpy as np

__all__ = [
    "UK_NATIONAL_LEARNING_RATE",
    "UK_NATIONAL_L0_LAMBDA",
    "UK_NATIONAL_LOSS_FAMILIES",
    "UK_NATIONAL_MASS_RULE",
    "UK_NATIONAL_MAX_WEIGHT_RATIO",
    "UK_NATIONAL_SEED",
    "UK_NATIONAL_SOLVE_DOCTRINE",
    "UK_NATIONAL_SOLVE_EPOCHS",
    "UK_NATIONAL_TARGET_LOSS_CAP",
    "UK_NATIONAL_TARGET_WEIGHT_RULE",
    "uk_doctrine_with_overrides",
    "uk_national_loss_families",
    "uk_national_target_loss_weights",
    "UKNationalSolveDoctrine",
]

UK_NATIONAL_SOLVE_EPOCHS = 1500
UK_NATIONAL_LEARNING_RATE = 0.02
UK_NATIONAL_MAX_WEIGHT_RATIO = 10.0
UK_NATIONAL_SEED = 0
UK_NATIONAL_TARGET_LOSS_CAP = 10.0
UK_NATIONAL_L0_LAMBDA = 0.0
UK_NATIONAL_MASS_RULE = "free"
UK_NATIONAL_TARGET_WEIGHT_RULE = "loss_family_equal"

#: The reviewed split of a register family into loss families, by contract
#: target (María's ruling of 2026-11-25, microcosm#1710). DWP's Universal
#: Credit family bundles eleven Stat-Xplore tables, so under family_equal each
#: of its composition rows weighed under 1/4,000 of the objective and the
#: solver gave them away. Its loss families follow what the rows measure: what
#: households are paid (the award amount bands by family type), who claims
#: (households in total, by family type and by number of children, and people
#: aged 16 to 21 and 22 to 24), and which elements they carry (the housing
#: element by tenure, with the single-row tables). A bound row of a split
#: family whose contract target is not listed is refused, so a new target in
#: the family is weighted only after review. Only the national loss weights
#: read this map: the register family stays every row's family everywhere else.
UK_NATIONAL_LOSS_FAMILIES: Mapping[str, Mapping[str, tuple[str, ...]]] = (
    MappingProxyType(
        {
            "dwp_universal_credit": MappingProxyType(
                {
                    "dwp_universal_credit_award_bands": (
                        "dwp.uc.payment_distribution_single",
                        "dwp.uc.payment_distribution_lone_parent",
                        "dwp.uc.payment_distribution_couple_no_children",
                        "dwp.uc.payment_distribution_couple_with_children",
                    ),
                    "dwp_universal_credit_composition": (
                        "dwp.uc.households",
                        "dwp.uc.households_single_no_children",
                        "dwp.uc.households_single_with_children",
                        "dwp.uc.households_couple_no_children",
                        "dwp.uc.households_couple_with_children",
                        "dwp.uc.households_children_1",
                        "dwp.uc.households_children_2",
                        "dwp.uc.households_children_3",
                        "dwp.uc.households_children_4",
                        "dwp.uc.people_aged_16_to_21",
                        "dwp.uc.people_aged_22_to_24",
                    ),
                    "dwp_universal_credit_housing_and_single_tables": (
                        "dwp.uc.households_housing_element_social_rented",
                        "dwp.uc.households_housing_element_private_rented",
                        "dwp.uc.households_lcwra_element",
                        "dwp.uc.households_carer_element",
                        "dwp.uc.households_childcare_element",
                        "dwp.uc.households_with_deduction",
                        "dwp.uc.scotland_households_child_under_1",
                        "dwp.uc.people_aged_16_to_24_lcwra_element",
                    ),
                }
            )
        }
    )
)

_ALLOWED_SCALE_RULES = ("default_target_loss_scales",)

# "family_equal" gives every declared target family one equal share of the
# objective, split evenly within the family — the reviewed rule name the
# local doctrine reserved for exactly this ("a future family-level weighting
# would be a new reviewed rule name here — never a per-target vector").
#
# Vocabulary admitted on 2026-08-23 from the first armed-run receipts: under
# "uniform" the hmrc SPI income-band family is 57% of the target surface and
# supplied 101 of the 102 references that start past the loss cap; pursuing
# them annihilated 24 population and OBR targets that were never past cap
# themselves (objective gain from annihilating a segment: +4.48 uniform vs
# +0.68 family_equal at cap 10, negative at cap 2). María's ruling
# (2026-08-24) kept "uniform" as the default while the weighting doctrine was
# being measured, with family_equal reachable only as an explicit, receipted
# per-run override (its own run-9 receipts show a 9-member family carrying
# 5.5x the per-reference weight of a 50-member one).
#
# María's ruling (2026-09-20, acknowledged on microcosm#965): every campaign
# national
# run since v18 has overridden to family_equal at 1,500 epochs — the v20
# certified-cut posture (spine-s, loss 0.0104, 6/6 seam gates) carried the
# receipt {epochs 256 → 1500, target_weight_rule uniform → family_equal} —
# so those two values are now the doctrine and a certified national cut
# records no overrides. "uniform" stays vocabulary: a receipted per-run
# override, never silently reachable.
#
# María's ruling (2026-11-25, microcosm#1710): the doctrine moves to
# "loss_family_equal", family_equal over loss families. A row's loss family is
# its register family, unless UK_NATIONAL_LOSS_FAMILIES splits that family by
# contract target, as it splits DWP's Universal Credit family into three. Only
# the national loss weights change: the register family keeps its place on
# every other surface (source months, holds, exclusions, the support floor,
# gates, diagnostics and the mass reason), and the local doctrine's rules read
# register families as before. "family_equal" stays vocabulary: a receipted
# per-run override that reproduces the earlier weighting.
_ALLOWED_TARGET_WEIGHT_RULES = ("uniform", "family_equal", "loss_family_equal")
_ALLOWED_MASS_RULES = ("free",)
_OVERRIDABLE_FIELDS = frozenset(
    {"epochs", "learning_rate", "target_weight_rule", "target_loss_cap"}
)


@dataclass(frozen=True)
class UKNationalSolveDoctrine:
    """The declared constants of the UK national calibration solve."""

    epochs: int = UK_NATIONAL_SOLVE_EPOCHS
    learning_rate: float = UK_NATIONAL_LEARNING_RATE
    max_weight_ratio: float | None = UK_NATIONAL_MAX_WEIGHT_RATIO
    seed: int = UK_NATIONAL_SEED
    target_loss_cap: float = UK_NATIONAL_TARGET_LOSS_CAP
    scale_rule: str = "default_target_loss_scales"
    target_weight_rule: str = UK_NATIONAL_TARGET_WEIGHT_RULE
    mass_rule: str = UK_NATIONAL_MASS_RULE
    l0_lambda: float = UK_NATIONAL_L0_LAMBDA

    def __post_init__(self) -> None:
        if (
            not isinstance(self.epochs, int)
            or isinstance(self.epochs, bool)
            or self.epochs <= 0
        ):
            raise ValueError(
                f"doctrine epochs must be a positive integer, got {self.epochs!r}."
            )
        if (
            not isinstance(self.learning_rate, int | float)
            or isinstance(self.learning_rate, bool)
            or not np.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError(
                "doctrine learning_rate must be a positive finite number, "
                f"got {self.learning_rate!r}."
            )
        if self.max_weight_ratio is not None and (
            not isinstance(self.max_weight_ratio, int | float)
            or isinstance(self.max_weight_ratio, bool)
            or not np.isfinite(self.max_weight_ratio)
            or self.max_weight_ratio <= 1
        ):
            raise ValueError(
                "doctrine max_weight_ratio must be None or a finite number "
                f"greater than 1, got {self.max_weight_ratio!r}."
            )
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or self.seed < 0
        ):
            raise ValueError(
                f"doctrine seed must be a non-negative integer, got {self.seed!r}."
            )
        if (
            not isinstance(self.target_loss_cap, int | float)
            or isinstance(self.target_loss_cap, bool)
            or not np.isfinite(self.target_loss_cap)
            or self.target_loss_cap <= 0
        ):
            raise ValueError(
                "doctrine target_loss_cap must be a positive finite number, "
                f"got {self.target_loss_cap!r}."
            )
        if self.scale_rule not in _ALLOWED_SCALE_RULES:
            raise ValueError(
                f"doctrine scale_rule must be one of {_ALLOWED_SCALE_RULES}, "
                f"got {self.scale_rule!r}."
            )
        if self.target_weight_rule not in _ALLOWED_TARGET_WEIGHT_RULES:
            raise ValueError(
                "doctrine target_weight_rule must be one of "
                f"{_ALLOWED_TARGET_WEIGHT_RULES}, got "
                f"{self.target_weight_rule!r}."
            )
        if self.mass_rule not in _ALLOWED_MASS_RULES:
            raise ValueError(
                f"doctrine mass_rule must be one of {_ALLOWED_MASS_RULES}, "
                f"got {self.mass_rule!r}."
            )
        if (
            not isinstance(self.l0_lambda, int | float)
            or isinstance(self.l0_lambda, bool)
            or not np.isfinite(self.l0_lambda)
            or self.l0_lambda < 0
        ):
            raise ValueError(
                "doctrine l0_lambda must be a non-negative finite number, "
                f"got {self.l0_lambda!r}."
            )


UK_NATIONAL_SOLVE_DOCTRINE = UKNationalSolveDoctrine()


def uk_doctrine_with_overrides(
    **overrides: object,
) -> tuple[UKNationalSolveDoctrine, dict[str, dict[str, object]]]:
    """Return the national doctrine plus a receipt of effective overrides."""

    fields_by_name = {field.name for field in fields(UKNationalSolveDoctrine)}
    unknown = sorted(set(overrides) - fields_by_name)
    if unknown:
        raise ValueError(
            "unknown UK national doctrine field(s) "
            f"{unknown}; frozen fields are {sorted(fields_by_name)}."
        )
    frozen = sorted(set(overrides) - _OVERRIDABLE_FIELDS)
    if frozen:
        raise ValueError(
            "UK national doctrine field(s) are reviewed constants, not knobs: "
            f"{frozen}; overridable fields are {sorted(_OVERRIDABLE_FIELDS)}."
        )
    doctrine = replace(UK_NATIONAL_SOLVE_DOCTRINE, **overrides)
    receipt: dict[str, dict[str, object]] = {}
    for name in sorted(_OVERRIDABLE_FIELDS):
        default = getattr(UK_NATIONAL_SOLVE_DOCTRINE, name)
        effective = getattr(doctrine, name)
        if effective != default:
            receipt[name] = {"default": default, "effective": effective}
    return doctrine, receipt


def uk_national_loss_families(
    families: Sequence[str],
    contract_target_ids: Sequence[str | None],
) -> list[str]:
    """Each row's loss family: its register family, or the declared split.

    A row of a family :data:`UK_NATIONAL_LOSS_FAMILIES` splits takes the loss
    family its contract target is listed under; a row of a split family whose
    contract target is not listed is refused, never left in a remainder.
    """

    if len(families) != len(contract_target_ids):
        raise ValueError(
            f"{len(families)} families against {len(contract_target_ids)} "
            "contract targets: the two must align row for row."
        )
    listed: dict[tuple[str, str], str] = {}
    for family, splits in UK_NATIONAL_LOSS_FAMILIES.items():
        for loss_family, targets in splits.items():
            for target in targets:
                if (family, target) in listed:
                    raise ValueError(
                        f"{target!r} is listed under two loss families of {family!r}."
                    )
                listed[(family, target)] = loss_family
    labels: list[str] = []
    unlisted: dict[str, set[str]] = {}
    for family, target in zip(families, contract_target_ids, strict=True):
        family = str(family)
        if family not in UK_NATIONAL_LOSS_FAMILIES:
            labels.append(family)
            continue
        loss_family = listed.get((family, str(target or "")))
        if loss_family is None:
            unlisted.setdefault(family, set()).add(str(target or "<none>"))
            continue
        labels.append(loss_family)
    if unlisted:
        named = "; ".join(
            f"{family}: {sorted(targets)}"
            for family, targets in sorted(unlisted.items())
        )
        raise ValueError(
            "loss_family_equal splits these families by contract target, but bound "
            f"rows name contract targets it does not list ({named}); add them to "
            "UK_NATIONAL_LOSS_FAMILIES after review."
        )
    return labels


def uk_national_target_loss_weights(
    families: Sequence[str],
    *,
    rule: str = UK_NATIONAL_TARGET_WEIGHT_RULE,
    contract_target_ids: Sequence[str | None] | None = None,
) -> np.ndarray | None:
    """The declared target-weight vector for one compiled register.

    ``uniform`` returns ``None`` — the kernel's own equal weighting, and the
    identity this doctrine shipped with. ``family_equal`` gives each declared
    family one equal share of the objective, divided evenly among its
    members, so an over-supplied family cannot outvote the rest of the
    surface by sheer count. ``loss_family_equal`` does the same over loss
    families (:func:`uk_national_loss_families`), and needs each row's
    ``contract_target_ids`` entry to place the rows of a split family.

    The vector is derived from the compiled register's declared families and
    contract targets and nothing else: it is a rule, not a per-target vector,
    which is the property the local doctrine requires of any weighting.
    """

    if rule not in _ALLOWED_TARGET_WEIGHT_RULES:
        raise ValueError(
            f"target_weight_rule must be one of {_ALLOWED_TARGET_WEIGHT_RULES}, "
            f"got {rule!r}."
        )
    if rule == "uniform":
        return None
    labels = [str(family) for family in families]
    if not labels:
        raise ValueError(f"{rule} weighting requires a non-empty register.")
    if any(not label for label in labels):
        raise ValueError(f"every target must declare a family for {rule}.")
    if rule == "loss_family_equal":
        if contract_target_ids is None:
            raise ValueError(
                "loss_family_equal needs each row's contract target to place the "
                "rows of a split family."
            )
        labels = uk_national_loss_families(labels, contract_target_ids)
    counts = Counter(labels)
    family_count = len(counts)
    return np.asarray(
        [1.0 / (family_count * counts[label]) for label in labels],
        dtype=np.float64,
    )
