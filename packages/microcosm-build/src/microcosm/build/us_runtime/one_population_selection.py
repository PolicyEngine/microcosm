"""The national one-population selection: every CPS household plus ACS households.

d2283's national file keeps every CPS source household of the parent's income
years and fills the rest of a fixed household budget with ACS households, in
the support-mix bake-off's nested order. This module holds the rule as pure
functions, with no frame I/O:

- unit keys (:func:`acs_unit_key`, :func:`cps_unit_key`), in the bake-off's
  forms, refusing values a ``str()`` cast would otherwise let through;
- eligibility flags (:func:`spm_zero_adult`, :func:`group_quarters`) and the
  rule that turns them into an eligible set (:class:`Eligibility`);
- the plan (:func:`plan_selection`): rank every ACS key with
  :func:`~microcosm.build.us_runtime.support_mix.acs_selection_order`, drop
  the ineligible households, and take replicate ``r``'s block of the rest
  (replicate 0's block is the first ``n``), as the bake-off's arm path does;
- starting weights (:func:`starting_weights`): the bake-off's
  :func:`~microcosm.build.us_runtime.support_mix.arm_initial_weights` with no
  clones, scaled so weighted persons equal a person benchmark;
- a manifest (:class:`SelectionManifest`) that records the rule, the counts,
  the threshold draw and digests of the selected keys. No rank, draw or key
  column is ever stored on a frame.

Every function refuses an input it cannot honour (a duplicate key, a budget
the CPS households exceed, an eligible set shorter than the fill) and never
clamps.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.demographics import CENSUS_NATIONAL_AGE_BENCHMARK
from microcosm.build.us_runtime.support_mix import (
    ArmCounts,
    acs_selection_order,
    arm_initial_weights,
    hash_uniform,
    replicate_block,
)

__all__ = [
    "ACS_ORDER_SALT",
    "DEFAULT_PERSON_BENCHMARK",
    "RULE_VERSION",
    "SPM_ROLE_COLUMN",
    "AcsEligibility",
    "Eligibility",
    "SelectionManifest",
    "SelectionPlan",
    "StartingWeights",
    "acs_unit_key",
    "cps_unit_key",
    "group_quarters",
    "keys_sha256",
    "plan_selection",
    "spm_zero_adult",
    "starting_weights",
]

#: Names this module's rule. A change to what the plan selects, or to how the
#: manifest is written, takes a new version.
RULE_VERSION = "us.one_population.selection.v1"

#: The bake-off's ACS order salt (``tools/bakeoff_us_support_mix.py``
#: ``ACS_SELECTION_SALT``), kept so production selects the households the
#: bake-off scored. A test pins the two together.
ACS_ORDER_SALT = "microcosm.us.support_mix_bakeoff.v1.acs_order"

#: The stored SPM role column, as the bake-off's flag reads it.
SPM_ROLE_COLUMN = "is_spm_independent_minor_role"

#: The national person benchmark: the sum of the Census age bands, which is
#: the sum the release tool takes for its base population repair.
DEFAULT_PERSON_BENCHMARK = float(sum(CENSUS_NATIONAL_AGE_BENCHMARK.values()))

_ACS_PREFIX = "acs:"
_CPS_KEY = re.compile(r"([1-9][0-9]{3})\|cps:(.+)", re.DOTALL)
#: Text a ``str()`` cast of a float, a null or a bool produces, plus
#: e-notation with an unsigned exponent (``"17E6"``), which Python never
#: writes but other software may. Refusing it errs on the safe side: ACS
#: SERIALNOs and CPS PERIDNUMs are digits, with letters only in the ACS
#: housing-unit or group-quarters code, so neither takes that shape.
_COERCED_NUMBER = re.compile(
    r"[+-]?(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    r"|[+-]?[0-9]+[eE][+-]?[0-9]+"
    r"|[+-][0-9]+"
)
_COERCED_WORDS = frozenset(
    {
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "true",
        "false",
    }
)
_WHITESPACE = re.compile(r"\s")
#: The key forms' own separators, refused inside an identifier.
_SEPARATORS = (":", "|")
_TYPEHUGQ_CODES = (1.0, 2.0, 3.0)


# ------------------------------------------------------------------- keys


def _check_identifier(value: object, *, what: str) -> str:
    """``value`` if it is a clean source identifier, else :class:`ValueError`.

    Refuses a non-string, an empty value, any whitespace or non-printable
    character (a newline would make the sorted-key digest ambiguous), the
    key separators ``:`` and ``|``, and text that a ``str()`` cast of a
    float, a null or a bool produces (``"123.0"``, ``"1e+12"``, ``"nan"``,
    ``"None"``, ``"True"``). E-notation is refused with or without a signed
    exponent (``"17E6"`` too), on purpose: text of that shape is a number
    some tool rewrote, not a source identifier. A run of digits stays valid:
    older ACS serial numbers and CPS identifiers are digits.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{what} must be a string, got {type(value).__name__} {value!r}."
        )
    if not value or not value.strip():
        raise ValueError(f"{what} must not be empty or blank, got {value!r}.")
    if not value.isprintable() or _WHITESPACE.search(value):
        raise ValueError(
            f"{what} must not hold whitespace or control characters, got {value!r}."
        )
    if any(separator in value for separator in _SEPARATORS):
        raise ValueError(f"{what} must not hold ':' or '|', got {value!r}.")
    if _COERCED_NUMBER.fullmatch(value) or value.lower() in _COERCED_WORDS:
        raise ValueError(
            f"{what} {value!r} looks like a number, null or bool cast to text; "
            "pass the source's own string identifier."
        )
    return value


def acs_unit_key(serialno: str) -> str:
    """The ACS household's unit key, ``"acs:" + SERIALNO`` (the bake-off's form).

    Raises:
        ValueError: If ``serialno`` is not a clean string identifier (see
            :func:`_check_identifier`).
    """
    return _ACS_PREFIX + _check_identifier(serialno, what="ACS SERIALNO")


def _check_income_year(income_year: object) -> int:
    if isinstance(income_year, bool) or not isinstance(income_year, numbers.Integral):
        raise ValueError(f"CPS income year must be an integer, got {income_year!r}.")
    year = int(income_year)
    if not 1000 <= year <= 9999:
        raise ValueError(f"CPS income year must have four digits, got {year!r}.")
    return year


def cps_unit_key(income_year: int, unit_id: str) -> str:
    """The CPS source household's key, ``f"{income_year}|cps:{unit_id}"``.

    This is the bake-off's ``source_key``: one ASEC household-year. Every CPS
    household of the named years is selected, so the key serves counting and
    the manifest digest only.

    Raises:
        ValueError: If ``income_year`` is not a four-digit integer or
            ``unit_id`` is not a clean string identifier.
    """
    year = _check_income_year(income_year)
    return f"{year}|cps:{_check_identifier(unit_id, what='CPS unit id')}"


def _parse_cps_key(key: object) -> int:
    """The income year of a valid CPS key, else :class:`ValueError`."""
    if not isinstance(key, str):
        raise ValueError(f"CPS key must be a string, got {type(key).__name__} {key!r}.")
    match = _CPS_KEY.fullmatch(key)
    if match is None:
        raise ValueError(
            f"CPS key {key!r} is not of the form '<income year>|cps:<unit id>'."
        )
    _check_identifier(match.group(2), what=f"unit id of CPS key {key!r}")
    return int(match.group(1))


def _check_acs_key(key: object) -> str:
    if not isinstance(key, str):
        raise ValueError(f"ACS key must be a string, got {type(key).__name__} {key!r}.")
    if not key.startswith(_ACS_PREFIX):
        raise ValueError(f"ACS key {key!r} does not start with {_ACS_PREFIX!r}.")
    _check_identifier(key[len(_ACS_PREFIX) :], what=f"SERIALNO of ACS key {key!r}")
    return key


def _refuse_duplicates(keys: Sequence[str], *, side: str) -> None:
    if len(set(keys)) != len(keys):
        counts: dict[str, int] = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        repeated = sorted(key for key, count in counts.items() if count > 1)
        raise ValueError(
            f"{side} keys repeat: {repeated[:5]} ({len(repeated)} in all)."
        )


def keys_sha256(keys: Iterable[str]) -> str:
    """sha256 of the sorted keys joined by newline (keys hold no newline).

    The digest :class:`SelectionManifest` records for each selected key set.
    It is public so a later check over a frame's keys (the population
    contract) digests them by the same rule.
    """
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


# ------------------------------------------------------------ eligibility


def _bool_values(series: pd.Series, *, what: str) -> np.ndarray:
    """A column of booleans (or 0/1) as a bool array; anything else refuses."""
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)
    values = series.to_numpy(dtype=object)
    if all(isinstance(value, bool | np.bool_) for value in values):
        return values.astype(bool)
    # Text refuses, "0" and "1" included: the bake-off's boolean cast reads
    # any nonempty text as true, so a numeric reading would disagree with it.
    text = next((value for value in values if isinstance(value, str | bytes)), None)
    if text is not None:
        raise ValueError(
            f"{what} must hold booleans (or 0/1), not text such as {text!r}."
        )
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and numeric.isin([0, 1]).all():
        return numeric.to_numpy() == 1
    raise ValueError(f"{what} must hold booleans (or 0/1), got dtype {series.dtype}.")


def _role(person: pd.DataFrame) -> np.ndarray:
    """The SPM independent-minor role the pinned engine would read.

    The stored column when every row has a value; the formula
    ``is_household_head | is_household_spouse`` when the column is absent or
    all null; a refusal when it is partly null, or when the formula has
    neither of its columns (15-17-year-old heads would silently count as
    minors).
    """
    stored = person[SPM_ROLE_COLUMN] if SPM_ROLE_COLUMN in person.columns else None
    if stored is not None and stored.notna().all():
        return _bool_values(stored, what=f"person.{SPM_ROLE_COLUMN}")
    if stored is not None and stored.notna().any():
        raise ValueError(
            f"person.{SPM_ROLE_COLUMN} is partially null ({int(stored.isna().sum())} rows)."
        )
    columns = [
        c for c in ("is_household_head", "is_household_spouse") if c in person.columns
    ]
    if not columns:
        raise ValueError(
            "person table has neither is_household_head nor is_household_spouse."
        )
    role = np.zeros(len(person), dtype=bool)
    for column in columns:
        values = person[column]
        # A missing head or spouse value counts as not head or spouse, as in
        # the bake-off's formula. A dtype that cannot hold False (nullable
        # Int64 with a missing value) refuses there too; refuse it here with
        # this module's ValueError rather than pandas' TypeError.
        try:
            filled = values.where(values.notna(), False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"person.{column} has missing values that dtype {values.dtype} "
                f"cannot fill with False: {error}"
            ) from error
        role |= _bool_values(filled, what=f"person.{column}")
    return role


def _ids(frame: pd.DataFrame, column: str, *, table: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"{table} table has no {column} column.")
    values = frame[column]
    if values.isna().any():
        raise ValueError(
            f"{table}.{column} has {int(values.isna().sum())} null values."
        )
    return values.to_numpy()


def spm_zero_adult(person: pd.DataFrame, household: pd.DataFrame) -> np.ndarray:
    """Households holding an SPM unit the pinned engine counts no adult in.

    A member counts as an adult when ``age >= 18``, or ``age >= 15`` and in
    the SPM independent-minor role (see :func:`_role`). This is the bake-off
    tool's ``spm_zero_adult_households`` rule; unlike the tool it refuses a
    missing or non-numeric age instead of reading it as 0, an infinite age,
    a role, head or spouse column that is not boolean or 0/1, and a
    person whose household is not in ``household``.

    Returns:
        One flag per ``household`` row, in its order.
    """
    household_ids = _ids(household, "household_id", table="household")
    if len(pd.unique(household_ids)) != len(household_ids):
        raise ValueError("household.household_id repeats.")
    person_household = _ids(person, "person_household_id", table="person")
    person_unit = _ids(person, "person_spm_unit_id", table="person")
    if "age" not in person.columns:
        raise ValueError("person table has no age column.")
    age = pd.to_numeric(person["age"], errors="coerce")
    unusable = ~np.isfinite(age.to_numpy(dtype=np.float64, na_value=np.nan))
    if unusable.any():
        raise ValueError(
            f"person.age has {int(unusable.sum())} missing, non-numeric or infinite "
            "values; the adult rule needs every age."
        )
    outside = ~np.isin(person_household, household_ids)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} persons belong to households missing from the household table."
        )
    age = age.to_numpy(dtype=np.float64)
    adult = (age >= 18) | ((age >= 15) & _role(person))
    unit_adults = pd.Series(adult).groupby(person_unit).sum()
    empty_units = unit_adults.index[unit_adults.to_numpy() == 0]
    flagged = person_household[np.isin(person_unit, empty_units.to_numpy())]
    return np.isin(household_ids, flagged)


def group_quarters(typehugq: Sequence | np.ndarray | pd.Series) -> np.ndarray:
    """ACS group-quarters flags from ``TYPEHUGQ``: true where it is not 1.

    A missing value counts as not group quarters, as in the bake-off. A
    value that is not a missing value or one of the PUMS codes 1, 2, 3
    refuses.
    """
    raw = pd.Series(typehugq) if not isinstance(typehugq, pd.Series) else typehugq
    numeric = pd.to_numeric(raw, errors="coerce")
    unparsed = numeric.isna() & raw.notna()
    if unparsed.any():
        raise ValueError(f"TYPEHUGQ has {int(unparsed.sum())} non-numeric values.")
    present = numeric.dropna()
    unknown = ~present.isin(list(_TYPEHUGQ_CODES))
    if unknown.any():
        raise ValueError(
            f"TYPEHUGQ holds codes outside 1, 2, 3: {sorted(set(present[unknown].tolist()))[:5]}."
        )
    return numeric.fillna(1).to_numpy(dtype=np.float64) != 1


def _flag_array(values: object, *, what: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(
            f"{what} flags must be one-dimensional, got shape {array.shape}."
        )
    if array.dtype != np.bool_:
        raise ValueError(f"{what} flags must be booleans, got dtype {array.dtype}.")
    out = array.copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class Eligibility:
    """Which ACS households may be selected.

    The defaults are d2283's measured arm: group-quarters households stay in
    and households with a zero-adult SPM unit go out.
    """

    include_group_quarters: bool = True
    drop_spm_zero_adult: bool = True

    def __post_init__(self) -> None:
        for name in ("include_group_quarters", "drop_spm_zero_adult"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}.")

    def evaluate(
        self, *, group_quarters: object, spm_zero_adult: object
    ) -> AcsEligibility:
        """The eligible set for per-household flags in ACS-key order."""
        return AcsEligibility(
            rule=self, group_quarters=group_quarters, spm_zero_adult=spm_zero_adult
        )


@dataclass(frozen=True, eq=False)
class AcsEligibility:
    """An :class:`Eligibility` applied to one ACS household set.

    Holds the flags, one per ACS key in the order the keys are passed to
    :func:`plan_selection`; the eligible mask is derived from them, so the
    counts a manifest records cannot disagree with the mask.
    """

    rule: Eligibility
    group_quarters: np.ndarray
    spm_zero_adult: np.ndarray
    eligible: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.rule, Eligibility):
            raise ValueError(f"rule must be an Eligibility, got {self.rule!r}.")
        gq = _flag_array(self.group_quarters, what="group-quarters")
        zero = _flag_array(self.spm_zero_adult, what="zero-adult")
        if gq.shape != zero.shape:
            raise ValueError(
                f"group-quarters and zero-adult flags must align, got {gq.shape} and {zero.shape}."
            )
        eligible = np.ones(gq.shape, dtype=bool)
        if not self.rule.include_group_quarters:
            eligible &= ~gq
        if self.rule.drop_spm_zero_adult:
            eligible &= ~zero
        eligible.setflags(write=False)
        object.__setattr__(self, "group_quarters", gq)
        object.__setattr__(self, "spm_zero_adult", zero)
        object.__setattr__(self, "eligible", eligible)

    @property
    def n_households(self) -> int:
        return int(len(self.eligible))

    @property
    def n_eligible(self) -> int:
        return int(self.eligible.sum())

    @property
    def n_group_quarters(self) -> int:
        return int(self.group_quarters.sum())

    @property
    def n_spm_zero_adult(self) -> int:
        return int(self.spm_zero_adult.sum())


# ------------------------------------------------------------------ plan


def _check_count(value: object, *, what: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{what} must be an integer, got {value!r}.")
    if int(value) < minimum:
        raise ValueError(f"{what} must be at least {minimum}, got {int(value)!r}.")
    return int(value)


@dataclass(frozen=True, eq=False)
class SelectionPlan:
    """What :func:`plan_selection` selected, and why.

    Attributes:
        rule_version: :data:`RULE_VERSION`.
        salt: The ACS order salt.
        replicate: The block of the eligible order taken (0 is the prefix).
        budget: Source households in the selection.
        cps_income_years: The named CPS income years, ascending.
        cps_keys: Every CPS key, sorted; all are selected.
        n_cps_by_income_year: ``(year, households)`` per named year.
        eligibility: The ACS eligibility the plan applied.
        acs_positions: Positions in the input ACS keys of the selected
            households, in selection (rank) order. Read-only.
        acs_keys: The selected ACS keys, in selection order.
        threshold_draw: The last selected household's draw, or ``None`` when
            the CPS households fill the budget.
    """

    rule_version: str
    salt: str
    replicate: int
    budget: int
    cps_income_years: tuple[int, ...]
    cps_keys: tuple[str, ...]
    n_cps_by_income_year: tuple[tuple[int, int], ...]
    eligibility: AcsEligibility
    acs_positions: np.ndarray
    acs_keys: tuple[str, ...]
    threshold_draw: float | None

    @property
    def n_cps(self) -> int:
        return len(self.cps_keys)

    @property
    def n_acs(self) -> int:
        return len(self.acs_keys)

    @property
    def n_acs_input(self) -> int:
        return self.eligibility.n_households

    @property
    def counts(self) -> ArmCounts:
        """The plan's counts in :mod:`support_mix` form (no clones)."""
        return ArmCounts(cps=self.n_cps, acs=self.n_acs, clones=0)


def plan_selection(
    *,
    cps_keys: Sequence[str],
    cps_income_years: Iterable[int],
    acs_keys: Sequence[str],
    acs_eligible: AcsEligibility,
    budget: int,
    salt: str = ACS_ORDER_SALT,
    replicate: int = 0,
) -> SelectionPlan:
    """Select every CPS household and fill the budget with ranked ACS households.

    Every ACS key is ranked with
    :func:`~microcosm.build.us_runtime.support_mix.acs_selection_order`
    (ascending draw, ties by key), as the bake-off's ``acs_rank`` ranks all
    rows. The ineligible households are then dropped and replicate
    ``replicate``'s block of the rest is taken (``replicate_block``): for
    replicate 0 that is the first ``budget - len(cps_keys)`` eligible
    households, so selections nest across budgets. Each draw depends only on
    the salt and the key, so ranking before or after the drop gives the same
    set.

    Args:
        cps_keys: One :func:`cps_unit_key` per CPS source household.
        cps_income_years: The income years the CPS households come from.
            Every key's year must be named and every named year must hold a
            key.
        acs_keys: One :func:`acs_unit_key` per ACS household.
        acs_eligible: Eligibility flags aligned with ``acs_keys``.
        budget: Source households (CPS plus ACS) to select.
        salt: The ACS order salt.
        replicate: Which disjoint block of the eligible order to take.

    Raises:
        ValueError: On an invalid or duplicate key on either side, a key of
            an unnamed year or a named year with no key, flags that do not
            align with the ACS keys, CPS households above the budget, or too
            few eligible ACS households for the block. Nothing is clamped.
    """
    budget = _check_count(budget, what="budget", minimum=1)
    replicate = _check_count(replicate, what="replicate", minimum=0)
    if not isinstance(salt, str) or not salt:
        raise ValueError(f"salt must be a non-empty string, got {salt!r}.")
    if isinstance(cps_keys, str) or isinstance(acs_keys, str):
        raise ValueError(
            "cps_keys and acs_keys must be sequences of keys, not one string."
        )
    years = tuple(_check_income_year(year) for year in cps_income_years)
    if len(set(years)) != len(years):
        raise ValueError(f"cps_income_years repeats a year: {years}.")
    years = tuple(sorted(years))

    cps = list(cps_keys)
    key_years = [_parse_cps_key(key) for key in cps]
    _refuse_duplicates(cps, side="CPS")
    unnamed = sorted(set(key_years) - set(years))
    if unnamed:
        raise ValueError(
            f"CPS keys come from income years {unnamed} that are not named in {years}."
        )
    by_year = {year: 0 for year in years}
    for year in key_years:
        by_year[year] += 1
    empty = [year for year, count in by_year.items() if count == 0]
    if empty:
        raise ValueError(f"no CPS households for named income year(s) {empty}.")

    acs = [_check_acs_key(key) for key in acs_keys]
    _refuse_duplicates(acs, side="ACS")
    if not isinstance(acs_eligible, AcsEligibility):
        raise ValueError(
            "acs_eligible must be an AcsEligibility (Eligibility().evaluate(...)), "
            f"got {type(acs_eligible).__name__}."
        )
    if acs_eligible.n_households != len(acs):
        raise ValueError(
            f"{acs_eligible.n_households:,} eligibility flags for {len(acs):,} ACS keys."
        )

    n_cps = len(cps)
    if n_cps > budget:
        raise ValueError(
            f"{n_cps:,} CPS source households exceed the {budget:,} budget."
        )
    fill = budget - n_cps
    needed = (replicate + 1) * fill
    if needed > acs_eligible.n_eligible:
        raise ValueError(
            f"replicate {replicate} of an ACS fill of {fill:,} needs {needed:,} eligible ACS "
            f"households, but only {acs_eligible.n_eligible:,} of {len(acs):,} are eligible."
        )

    order = acs_selection_order(acs, salt=salt)
    ranked_eligible = order[acs_eligible.eligible[order]]
    block = np.array(replicate_block(ranked_eligible, fill, replicate), dtype=np.int64)
    block.setflags(write=False)
    selected = tuple(acs[position] for position in block)
    threshold = hash_uniform(selected[-1], salt=salt) if selected else None
    return SelectionPlan(
        rule_version=RULE_VERSION,
        salt=salt,
        replicate=replicate,
        budget=budget,
        cps_income_years=years,
        cps_keys=tuple(sorted(cps)),
        n_cps_by_income_year=tuple(sorted(by_year.items())),
        eligibility=acs_eligible,
        acs_positions=block,
        acs_keys=selected,
        threshold_draw=threshold,
    )


# --------------------------------------------------------------- weights


@dataclass(frozen=True, eq=False)
class StartingWeights:
    """Starting household weights for a plan.

    Attributes:
        cps: One weight per CPS physical row, in the order passed.
        acs: One weight per selected ACS household, in
            ``plan.acs_positions`` (selection) order.
        mass_factor: The factor that took the unscaled weights to the
            person benchmark.
        person_benchmark: Weighted persons the weights carry.
        unscaled_total_mass: The total mass given to ``arm_initial_weights``
            (the budget: a mean of 1 per source household).
    """

    cps: np.ndarray
    acs: np.ndarray
    mass_factor: float
    person_benchmark: float
    unscaled_total_mass: float

    @property
    def total_mass(self) -> float:
        return self.unscaled_total_mass * self.mass_factor


def _positive_array(values: object, *, what: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{what} must be one-dimensional, got shape {array.shape}.")
    if not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError(f"{what} must be finite and positive.")
    return array


def starting_weights(
    plan: SelectionPlan,
    *,
    cps_design_weights: object,
    cps_row_income_year: object,
    acs_design_weights: object,
    persons_per_cps_row: object,
    persons_per_acs_row: object,
    person_benchmark: float | None = None,
) -> StartingWeights:
    """The plan's starting weights, scaled so weighted persons equal a benchmark.

    The unscaled weights are
    :func:`~microcosm.build.us_runtime.support_mix.arm_initial_weights` with
    no clones and a total mass of ``plan.budget``: each CPS income year gets
    an equal share of the CPS side, the CPS and ACS sides split the mass by
    source-household counts, and design-weight ratios within a year and
    within the ACS are kept. Every weight is then multiplied by one factor
    so that weighted persons equal ``person_benchmark``.

    Args:
        plan: The selection.
        cps_design_weights: Design weight per CPS physical row (every row of
            every selected CPS household; a household may span several rows).
        cps_row_income_year: Income year per CPS physical row.
        acs_design_weights: Design weight per ACS household, aligned with the
            ACS keys passed to :func:`plan_selection` (all of them).
        persons_per_cps_row: Persons per CPS physical row.
        persons_per_acs_row: Persons per ACS household, aligned like
            ``acs_design_weights``.
        person_benchmark: Weighted persons to reach; defaults to
            :data:`DEFAULT_PERSON_BENCHMARK`.

    Raises:
        ValueError: On misaligned arrays, a non-finite or non-positive weight,
            person count or benchmark, CPS rows from an unnamed year, a named
            year with fewer rows than households, or CPS rows given for a
            plan without CPS households.
    """
    if not isinstance(plan, SelectionPlan):
        raise ValueError(f"plan must be a SelectionPlan, got {type(plan).__name__}.")
    benchmark = (
        DEFAULT_PERSON_BENCHMARK if person_benchmark is None else person_benchmark
    )
    if isinstance(benchmark, bool) or not isinstance(benchmark, numbers.Real):
        raise ValueError(f"person_benchmark must be a number, got {benchmark!r}.")
    benchmark = float(benchmark)
    if not (math.isfinite(benchmark) and benchmark > 0):
        raise ValueError(
            f"person_benchmark must be finite and positive, got {benchmark!r}."
        )

    cps_w = _positive_array(cps_design_weights, what="CPS design weights")
    cps_persons = _positive_array(persons_per_cps_row, what="CPS persons per row")
    years = np.asarray(cps_row_income_year)
    if years.ndim != 1 or not (len(years) == len(cps_w) == len(cps_persons)):
        raise ValueError("CPS design weights, income years and persons must align.")
    if len(years) and not np.issubdtype(years.dtype, np.integer):
        raise ValueError(f"CPS income years must be integers, got dtype {years.dtype}.")
    rows_by_year = dict(zip(*np.unique(years, return_counts=True), strict=True))
    unnamed = sorted(
        int(year) for year in rows_by_year if int(year) not in plan.cps_income_years
    )
    if unnamed:
        raise ValueError(
            f"CPS rows come from income years {unnamed} the plan does not name."
        )
    for year, households in plan.n_cps_by_income_year:
        rows = int(rows_by_year.get(year, 0))
        if rows < households:
            raise ValueError(
                f"income year {year} has {rows:,} CPS rows for {households:,} source households."
            )

    acs_all = _positive_array(acs_design_weights, what="ACS design weights")
    acs_persons_all = _positive_array(
        persons_per_acs_row, what="ACS persons per household"
    )
    if not (len(acs_all) == len(acs_persons_all) == plan.n_acs_input):
        raise ValueError(
            f"ACS weights and persons must align with the plan's {plan.n_acs_input:,} input "
            f"ACS keys, got {len(acs_all):,} and {len(acs_persons_all):,}."
        )
    acs_w = acs_all[plan.acs_positions]
    acs_persons = acs_persons_all[plan.acs_positions]

    unscaled_total = float(plan.budget)
    cps_out, acs_out = arm_initial_weights(
        cps_design_weights=cps_w,
        cps_income_year=years,
        cps_copies=np.zeros(len(cps_w), dtype=np.int64),
        acs_design_weights=acs_w,
        counts=plan.counts,
        total_mass=unscaled_total,
    )
    weighted_persons = float(cps_persons @ cps_out + acs_persons @ acs_out)
    factor = benchmark / weighted_persons
    if not (math.isfinite(factor) and factor > 0):
        raise ValueError(
            f"the person scaling factor is not finite and positive: {factor!r}."
        )
    return StartingWeights(
        cps=cps_out * factor,
        acs=acs_out * factor,
        mass_factor=factor,
        person_benchmark=benchmark,
        unscaled_total_mass=unscaled_total,
    )


# --------------------------------------------------------------- manifest


def _check_digests(
    input_digests: Mapping[str, str] | None,
) -> tuple[tuple[str, str], ...]:
    if input_digests is None:
        return ()
    if not isinstance(input_digests, Mapping):
        raise ValueError(
            f"input_digests must be a mapping, got {type(input_digests).__name__}."
        )
    out = []
    for name, digest in input_digests.items():
        for what, value in (("input name", name), ("input digest", digest)):
            if (
                not isinstance(value, str)
                or not value
                or any(c.isspace() for c in value)
            ):
                raise ValueError(
                    f"{what} must be a non-empty string without whitespace, got {value!r}."
                )
        out.append((name, digest))
    return tuple(sorted(out))


_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class SelectionManifest:
    """The receipt of one selection; no rank, draw or key column is stored.

    ``to_json`` is canonical (sorted keys, no whitespace, ASCII), and
    ``digest`` is its sha256. Construction refuses a field of the wrong type
    and a tuple field out of order, so two manifests share a digest exactly
    when every field agrees. :meth:`from_plan` is the constructor to use.

    The digest moves with the selected key sets (ACS and CPS), the salt, the
    replicate, the rule version and the budget, and also with what the
    manifest records beside them: the eligibility flags, the ACS input,
    eligible, zero-adult and group-quarters counts, the CPS counts by year
    and the caller's input digests. It never moves with the order of the
    inputs.
    """

    rule_version: str
    salt: str
    replicate: int
    budget: int
    include_group_quarters: bool
    drop_spm_zero_adult: bool
    acs_households: int
    acs_eligible: int
    acs_spm_zero_adult: int
    acs_group_quarters: int
    cps_income_years: tuple[int, ...]
    n_cps: int
    n_acs: int
    n_cps_by_income_year: tuple[tuple[int, int], ...]
    threshold_draw: float | None
    selected_acs_keys_sha256: str
    cps_keys_sha256: str
    input_digests: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        # Each field takes the one type from_plan gives it: a budget of 15.0
        # equals 15 but is written differently, and a numpy integer is not
        # written at all. The JSON writes the year and digest tuples as
        # sorted mappings, so they must already be sorted and unrepeated.
        for name in ("rule_version", "salt"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string, got {value!r}.")
        for name in (
            "replicate",
            "budget",
            "acs_households",
            "acs_eligible",
            "acs_spm_zero_adult",
            "acs_group_quarters",
            "n_cps",
            "n_acs",
        ):
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}.")
        for name in ("include_group_quarters", "drop_spm_zero_adult"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool, got {getattr(self, name)!r}.")
        years = self.cps_income_years
        if (
            not isinstance(years, tuple)
            or not all(_is_int(year) for year in years)
            or years != tuple(sorted(set(years)))
        ):
            raise ValueError(
                f"cps_income_years must be a sorted tuple of distinct ints, got {years!r}."
            )
        by_year = self.n_cps_by_income_year
        if (
            not isinstance(by_year, tuple)
            or not all(
                isinstance(pair, tuple)
                and len(pair) == 2
                and _is_int(pair[0])
                and _is_int(pair[1])
                and pair[1] >= 0
                for pair in by_year
            )
            or [year for year, _n in by_year] != sorted({year for year, _n in by_year})
        ):
            raise ValueError(
                "n_cps_by_income_year must be a tuple of (year, households) int pairs "
                f"sorted by distinct year, got {by_year!r}."
            )
        draw = self.threshold_draw
        if draw is not None and not (
            isinstance(draw, float)
            and 0.0 <= draw < 1.0
            and math.copysign(1.0, draw) > 0
        ):
            raise ValueError(
                f"threshold_draw must be None or a float in [0, 1), got {draw!r}."
            )
        for name in ("selected_acs_keys_sha256", "cps_keys_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256_HEX.fullmatch(value):
                raise ValueError(
                    f"{name} must be a lowercase sha256 hex digest, got {value!r}."
                )
        digests = self.input_digests
        if (
            not isinstance(digests, tuple)
            or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in digests)
            or _check_digests(dict(digests)) != digests
        ):
            raise ValueError(
                "input_digests must be a tuple of (name, digest) string pairs sorted "
                f"by distinct name, got {digests!r}."
            )

    @classmethod
    def from_plan(
        cls, plan: SelectionPlan, *, input_digests: Mapping[str, str] | None = None
    ) -> SelectionManifest:
        """The manifest of ``plan``, with the caller's input digests."""
        if not isinstance(plan, SelectionPlan):
            raise ValueError(
                f"plan must be a SelectionPlan, got {type(plan).__name__}."
            )
        eligibility = plan.eligibility
        return cls(
            rule_version=plan.rule_version,
            salt=plan.salt,
            replicate=plan.replicate,
            budget=plan.budget,
            include_group_quarters=eligibility.rule.include_group_quarters,
            drop_spm_zero_adult=eligibility.rule.drop_spm_zero_adult,
            acs_households=eligibility.n_households,
            acs_eligible=eligibility.n_eligible,
            acs_spm_zero_adult=eligibility.n_spm_zero_adult,
            acs_group_quarters=eligibility.n_group_quarters,
            cps_income_years=plan.cps_income_years,
            n_cps=plan.n_cps,
            n_acs=plan.n_acs,
            n_cps_by_income_year=plan.n_cps_by_income_year,
            threshold_draw=plan.threshold_draw,
            selected_acs_keys_sha256=keys_sha256(plan.acs_keys),
            cps_keys_sha256=keys_sha256(plan.cps_keys),
            input_digests=_check_digests(input_digests),
        )

    def to_payload(self) -> dict:
        return {
            "rule_version": self.rule_version,
            "salt": self.salt,
            "replicate": self.replicate,
            "budget": self.budget,
            "eligibility": {
                "include_group_quarters": self.include_group_quarters,
                "drop_spm_zero_adult": self.drop_spm_zero_adult,
            },
            "acs_households": self.acs_households,
            "acs_eligible": self.acs_eligible,
            "acs_spm_zero_adult": self.acs_spm_zero_adult,
            "acs_group_quarters": self.acs_group_quarters,
            "cps_income_years": list(self.cps_income_years),
            "n_by_origin": {"cps": self.n_cps, "acs": self.n_acs},
            "n_cps_by_income_year": {
                str(year): n for year, n in self.n_cps_by_income_year
            },
            "threshold_draw": self.threshold_draw,
            "selected_acs_keys_sha256": self.selected_acs_keys_sha256,
            "cps_keys_sha256": self.cps_keys_sha256,
            "input_digests": dict(self.input_digests),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("ascii")).hexdigest()
