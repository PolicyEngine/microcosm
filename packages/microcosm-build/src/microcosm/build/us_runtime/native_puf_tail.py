"""Pure native-tail declarations and selection over invented role projections.

No input here is authenticated source/person evidence. The only implemented
provenance mode is an explicitly declared invented fixture. In particular the
native return donor's technical person rows cannot qualify as observed people.
This module neither chooses a person-allocation model nor expands a population.

Union-arm and AGI-only thinning algebra adapted from PolicyEngine/microcosm
PR964, commit f7df78b2a00421f9b90305a9b7db192444075ae0, puf_agi_tail.py.
The native profile preserves independent SS, SCF and three late-owned leaves;
it is not the reference implementation's full65-minus-three transfer profile.
Two deliberate divergences from that reference are documented where they
occur: :func:`_rescale_exact` corrects a largest cell weight rather than the
last one, and the AGI floor decision never routes through a float64 total.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.frame import WeightKind

from .operator_column_contracts import US_QBI_BOOLEAN_OUTPUT_COLUMNS
from .us_late_overlap_ownership import US_LATE_OVERLAP_OWNERSHIP_TARGETS

PROTOCOL = "microcosm.us.native-puf-tail-declared-fixture/1"
REFERENCE_COMMIT = "f7df78b2a00421f9b90305a9b7db192444075ae0"
AGI_FLOOR = 5_000_000
MAX_AGI_ONLY_DONORS = 3_000
# This first contract is deliberately bounded to invented development fixtures.
# A complete actual source requires a different, independently reviewed owner.
MAX_DECLARED_RETURNS = 10_000
MAX_DECLARED_PERSONS = 100_000
# The donor-side build-period income locator. It is the reference selector's
# own 20-leaf donor proxy, not the seven-column recipient-side AGI-band locator
# `puf_capital_gains_tail._RECIPIENT_AGI_PROXY_COLUMNS`, which places recipient
# households into SOI bands. Neither list substitutes for the other.
PROXY_AGI_COMPONENTS = (
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "sstb_self_employment_income_before_lsr",
    "taxable_interest_income",
    "qualified_dividend_income",
    "non_qualified_dividend_income",
    "short_term_capital_gains",
    "long_term_capital_gains_before_response",
    "non_sch_d_capital_gains",
    "taxable_private_pension_income",
    "taxable_ira_distributions",
    "partnership_income",
    "s_corp_income",
    "rental_income",
    "farm_income",
    "farm_rent_income",
    "miscellaneous_income",
    "estate_income",
    "alimony_income",
    "salt_refund_income",
)
# A selected donor either kept its declared return weight untouched or had that
# weight rescaled inside a thinned arm-two cell. Unchanged numeric values do
# not retain DESIGN population semantics after conditional support selection:
# the whole selection is IMPORTANCE support either way. These labels record
# which of the two happened, never a population mass claim.
WEIGHT_TREATMENT_UNCHANGED = "unchanged_declared_return_weight"
WEIGHT_TREATMENT_RESCALED = "cell_rescaled_to_preserve_declared_cell_weight_sum"
_ROLES = frozenset(("head", "spouse", "dependent", "unclassified", "ambiguous"))
# Money storage is narrowed to the two exactly supported physical widths.
# Narrower or wider integers, float16/float32 and non-native byte order are
# refused rather than silently reinterpreted at a precision this contract
# never demonstrated.
_INT64 = np.dtype("int64")
_FLOAT64 = np.dtype("float64")
_AMOUNT_DTYPES = (_INT64, _FLOAT64)
_SNAPSHOT_DTYPES = frozenset(dtype.str for dtype in (np.dtype(bool), *_AMOUNT_DTYPES))


def _require(condition: bool, code: str) -> None:
    if not condition:
        # Never include a donor id, vector, mask or source value in an error.
        raise ValueError("NATIVE_PUF_TAIL_" + code)


def _json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _filing_status_domain() -> tuple[int, ...]:
    """Return the admitted filing-status codes from the existing PUF authority.

    ``puf_support._FILING_STATUS_CODES`` is the canonical decode map the US
    runtime already uses to turn engine filing-status names into the integer
    codes this contract receives, and ``puf_capital_gains_tail`` derives its
    integer-keyed inverse view from the same five entries. Reading the
    canonical map here keeps SURVIVING_SPOUSE (5) admissible without a second
    local enumeration; see docs/shared-constants.md. The import is deferred
    for the same reason the profile derivation defers its own.
    """
    from .puf_support import _FILING_STATUS_CODES

    values = tuple(_FILING_STATUS_CODES.values())
    _require(
        len(values) == len(set(values))
        and all(
            type(value) in (int, float) and float(value).is_integer() and value > 0
            for value in values
        ),
        "FILING_STATUS_AUTHORITY",
    )
    return tuple(sorted(int(value) for value in values))


@dataclass(frozen=True, slots=True)
class NativeTailProfile:
    """An exact output roster; not a donor admission or a transfer operation."""

    person_outputs: tuple[str, ...]
    tax_unit_outputs: tuple[str, ...]
    boolean_outputs: tuple[str, ...]
    late_owned: tuple[tuple[str, str], ...]

    @property
    def targets(self) -> tuple[str, ...]:
        return (*self.person_outputs, *self.tax_unit_outputs)


def native_puf_tail_profile() -> NativeTailProfile:
    """Derive the 52-field native profile from its existing ownership authorities."""
    from .full_puf_enrichment import PUF55_SURVEY_SS

    late = tuple(US_LATE_OVERLAP_OWNERSHIP_TARGETS)
    _require(len(late) == 3 and len(set(late)) == 3, "LATE_OWNER_ROSTER")
    person = tuple(
        name for name in PUF55_SURVEY_SS.person_outputs if ("person", name) not in late
    )
    tax_unit = tuple(
        name
        for name in PUF55_SURVEY_SS.tax_unit_outputs
        if ("tax_unit", name) not in late
    )
    booleans = tuple(name for name in person if name in US_QBI_BOOLEAN_OUTPUT_COLUMNS)
    profile = NativeTailProfile(person, tax_unit, booleans, late)
    _require(len(profile.targets) == len(set(profile.targets)) == 52, "PROFILE_ROSTER")
    _require(set(PROXY_AGI_COMPONENTS) <= set(profile.targets), "PROXY_ROSTER")
    return profile


@dataclass(frozen=True, slots=True)
class TailProjectionProvenance:
    """Descriptive fixture status. Real source/model modes are unimplemented."""

    basis: str = "invented_fixture"
    method: str = "explicit_fixture_roles"
    observed_person_authority: bool = False
    source_admission_issued: bool = False
    release_eligible: bool = False


@dataclass(frozen=True, slots=True)
class _Column:
    """A typed payload plus its all-known witness.

    This contract is all-known only. ``_known`` refuses any mask that is not
    entirely true, so ``known`` is always ``b"\\x01" * rows`` and carries no
    information beyond the row count; ``_restore`` requires exactly that
    constant. The field is an explicit witness of that limitation, not a
    partial-knownness channel. Admitting partial knownness would first require
    threading the caller's real mask through ``_snapshot``; until that happens
    the constant may not be reinterpreted as measured knownness.
    """

    name: str
    dtype: str
    payload: bytes
    known: bytes


@dataclass(frozen=True, slots=True)
class _Table:
    rows: int
    columns: tuple[_Column, ...]


@dataclass(frozen=True, slots=True)
class DeclaredPufTailRoleProjection:
    """Immutable typed fixture bytes. Copies/digests never issue source authority."""

    returns: _Table
    persons: _Table
    roles: tuple[str, ...]
    role_known: bytes
    provenance: TailProjectionProvenance
    profile: NativeTailProfile
    weight_kind: WeightKind

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256(PROTOCOL.encode())
        for table in (self.returns, self.persons):
            digest.update(_json({"rows": table.rows}))
            for column in table.columns:
                for value in (
                    _json((column.name, column.dtype)),
                    column.payload,
                    column.known,
                ):
                    digest.update(len(value).to_bytes(8, "little"))
                    digest.update(value)
        digest.update(_json(self.roles))
        digest.update(self.role_known)
        digest.update(
            _json(
                (
                    self.provenance.basis,
                    self.provenance.method,
                    self.provenance.observed_person_authority,
                    self.provenance.source_admission_issued,
                    self.provenance.release_eligible,
                )
            )
        )
        digest.update(
            _json(
                (
                    self.profile.person_outputs,
                    self.profile.tax_unit_outputs,
                    self.profile.boolean_outputs,
                    self.profile.late_owned,
                )
            )
        )
        digest.update(_json(self.weight_kind.value))
        return digest.hexdigest()


def _provenance(value: TailProjectionProvenance) -> None:
    _require(type(value) is TailProjectionProvenance, "PROVENANCE_TYPE")
    _require(
        value.basis == "invented_fixture"
        and value.method == "explicit_fixture_roles"
        and value.observed_person_authority is False
        and value.source_admission_issued is False
        and value.release_eligible is False,
        "PROVENANCE_UNIMPLEMENTED_AUTHORITY",
    )


def _columns(profile: NativeTailProfile) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        ("donor_recid", "weight", "filing_status_code", *profile.targets),
        ("person_id", "donor_recid", "role", *profile.person_outputs),
    )


def _validate_tables(
    returns: pd.DataFrame, persons: pd.DataFrame, profile: NativeTailProfile
) -> None:
    expected = _columns(profile)
    for table, names, limit in zip(
        (returns, persons),
        expected,
        (MAX_DECLARED_RETURNS, MAX_DECLARED_PERSONS),
        strict=True,
    ):
        _require(
            type(table) is pd.DataFrame
            and table.columns.is_unique
            and set(table.columns) == set(names),
            "TABLE_ROSTER",
        )
        _require(0 < len(table) <= limit and table.index.is_unique, "TABLE_ROWS")
        for name in names:
            if name == "role":
                _require(
                    isinstance(table[name].dtype, pd.StringDtype)
                    and not table[name].isna().any()
                    and all(type(v) is str and v in _ROLES for v in table[name]),
                    "ROLE_TYPE",
                )
                continue
            dtype = table[name].dtype
            _require(
                isinstance(dtype, np.dtype)
                and dtype.kind in "biuf"
                and dtype.itemsize <= 8,
                "NUMERIC_DTYPE",
            )
            _require(bool(np.isfinite(table[name].to_numpy()).all()), "NONFINITE")
            if name in ("person_id", "donor_recid", "filing_status_code"):
                _require(dtype == _INT64, "IDENTITY_DTYPE")
            elif table is persons and name in profile.boolean_outputs:
                _require(dtype == np.dtype(bool), "BOOLEAN_DTYPE")
            else:
                # Exactly int64 or float64: a narrower float would store money
                # at a precision the proxy and reconciliation never assume.
                _require(dtype in _AMOUNT_DTYPES, "AMOUNT_DTYPE")
    _require(
        returns.donor_recid.gt(0).all()
        and returns.donor_recid.is_unique
        and persons.person_id.gt(0).all()
        and persons.person_id.is_unique,
        "IDENTITY_DOMAIN",
    )
    _require(persons.donor_recid.isin(returns.donor_recid).all(), "ORPHAN_PERSON")
    _require(
        returns.filing_status_code.isin(_filing_status_domain()).all(),
        "FILING_STATUS_DOMAIN",
    )
    _require(
        returns.weight.dtype == _FLOAT64
        and returns.weight.ge(0).all()
        and np.isfinite(returns.weight.sum()),
        "WEIGHT_DOMAIN",
    )


def _known(table: pd.DataFrame, known: pd.DataFrame) -> None:
    # All-known only, by design: a partially known declaration refuses here
    # rather than being snapshotted as if every cell had been observed.
    _require(
        type(known) is pd.DataFrame
        and known.index.equals(table.index)
        and known.columns.equals(table.columns)
        and all(dtype == np.dtype(bool) for dtype in known.dtypes)
        and known.to_numpy().all(),
        "KNOWNNESS",
    )


def _snapshot(table: pd.DataFrame, names: tuple[str, ...]) -> _Table:
    # The constant witness is sound only because `_known` already refused every
    # non-all-true mask. Do not relax `_known` without threading a real mask.
    return _Table(
        len(table),
        tuple(
            _Column(
                name,
                table[name].dtype.str,
                table[name].to_numpy().tobytes(),
                b"\x01" * len(table),
            )
            for name in names
            if name != "role"
        ),
    )


def declare_puf_tail_role_projection(
    returns: pd.DataFrame,
    persons: pd.DataFrame,
    *,
    return_known: pd.DataFrame,
    person_known: pd.DataFrame,
    provenance: TailProjectionProvenance,
    weight_kind: WeightKind,
) -> DeclaredPufTailRoleProjection:
    """Snapshot complete known fixture cells without inferring/qualifying persons.

    Return values use tax-return grain. Person booleans remain physical bool;
    return-side booleans are aggregated numeric inputs, not person flags. All
    unknown cells refuse; zero can only arrive as an explicit known value.
    Money is stored at exactly int64 or float64; no other width is admitted.
    """
    _provenance(provenance)
    _require(weight_kind is WeightKind.DESIGN, "RETURN_WEIGHT_KIND")
    profile = native_puf_tail_profile()
    _validate_tables(returns, persons, profile)
    _known(returns, return_known)
    _known(persons, person_known)
    rcols, pcols = _columns(profile)
    result = DeclaredPufTailRoleProjection(
        _snapshot(returns, rcols),
        _snapshot(persons, pcols),
        tuple(persons.role),
        b"\x01" * len(persons),
        provenance,
        profile,
        weight_kind,
    )
    projection_tables(result)
    return result


def _restore(table: _Table, names: tuple[str, ...], limit: int) -> pd.DataFrame:
    _require(
        type(table) is _Table and type(table.rows) is int and 0 < table.rows <= limit,
        "SNAPSHOT_ROWS",
    )
    _require(
        type(table.columns) is tuple
        and all(type(c) is _Column for c in table.columns)
        and tuple(c.name for c in table.columns)
        == tuple(n for n in names if n != "role"),
        "SNAPSHOT_COLUMNS",
    )
    values = {}
    for column in table.columns:
        # The admitted set is the native-order physical bool/int64/float64
        # triple `_snapshot` can produce. A forged narrower or byte-swapped
        # dtype refuses instead of reinterpreting the same payload bytes.
        _require(
            type(column.dtype) is str and column.dtype in _SNAPSHOT_DTYPES,
            "SNAPSHOT_DTYPE",
        )
        dtype = np.dtype(column.dtype)
        _require(
            type(column.payload) is bytes
            and len(column.payload) == table.rows * dtype.itemsize
            and type(column.known) is bytes
            and column.known == b"\x01" * table.rows,
            "SNAPSHOT_PAYLOAD_OR_KNOWNNESS",
        )
        values[column.name] = np.frombuffer(column.payload, dtype=dtype).copy()
    return pd.DataFrame(values)


def projection_tables(
    projection: DeclaredPufTailRoleProjection,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Revalidate descriptive bytes and return detached tables, never authority."""
    _require(type(projection) is DeclaredPufTailRoleProjection, "PROJECTION_TYPE")
    _provenance(projection.provenance)
    _require(projection.weight_kind is WeightKind.DESIGN, "RETURN_WEIGHT_KIND")
    profile = native_puf_tail_profile()
    _require(
        type(projection.profile) is NativeTailProfile and projection.profile == profile,
        "PROFILE_CHANGED",
    )
    names = _columns(profile)
    returns = _restore(projection.returns, names[0], MAX_DECLARED_RETURNS)
    people = _restore(projection.persons, names[1], MAX_DECLARED_PERSONS)
    _require(
        type(projection.roles) is tuple
        and len(projection.roles) == len(people)
        and all(type(role) is str and role in _ROLES for role in projection.roles)
        and type(projection.role_known) is bytes
        and projection.role_known == b"\x01" * len(people),
        "ROLE_PAYLOAD_OR_KNOWNNESS",
    )
    people.insert(2, "role", pd.Series(projection.roles, dtype="string"))
    _validate_tables(returns, people, profile)
    return returns, people


@dataclass(frozen=True, slots=True)
class SelectedTailDonor:
    """One selected donor return. Descriptive only; no household is implied.

    Attributes:
        donor_recid: The declared return identifier.
        arm: 1 capital-gains only, 2 AGI only, 3 both.
        proxy_agi: The float64 diagnostic proxy total. The inclusive floor
            decision does not route through this value; see :func:`_proxy_agi`.
        weight: The selected support weight, in the treatment named below.
        spouse_payload_nonzero: Whether any declared spouse cell of this donor
            is nonzero, or ``None`` when the donor was never role-screened
            (arm 1 is outside the declared role-eligibility scope). This is a
            description of the declared payload, not a structural requirement:
            an all-zero spouse row on a joint return still says nothing about
            whether a later population owner must construct a spouse, and that
            owner alone determines household structure.
        weight_treatment: ``WEIGHT_TREATMENT_UNCHANGED`` or
            ``WEIGHT_TREATMENT_RESCALED``.
    """

    donor_recid: int
    arm: int
    proxy_agi: float
    weight: float
    spouse_payload_nonzero: bool | None
    weight_treatment: str


@dataclass(frozen=True, slots=True)
class NativeTailSelection:
    """A deterministic fixture selection; no population/source admission."""

    projection_sha256: str
    selection_input_sha256: str
    donors: tuple[SelectedTailDonor, ...]
    receipt: bytes

    @property
    def source_admission_issued(self) -> bool:
        return False

    @property
    def release_eligible(self) -> bool:
        return False

    @property
    def weight_kind(self) -> WeightKind:
        """Selected donor-return support, not household population mass.

        Thinned arm-two cells carry rescaled weights and every other selected
        weight is its unchanged declared value, but conditional support
        selection does not retain DESIGN population semantics either way, so
        the whole selection is IMPORTANCE. Per-donor ``weight_treatment``
        records which of the two happened.
        """
        return WeightKind.IMPORTANCE


def _mask(values: np.ndarray, rows: int) -> np.ndarray:
    _require(
        type(values) is np.ndarray
        and values.dtype == np.dtype(bool)
        and values.shape == (rows,),
        "MASK",
    )
    return values.copy()


def _role_eligibility(
    returns: pd.DataFrame,
    position: int,
    persons: pd.DataFrame,
    profile: NativeTailProfile,
) -> tuple[str | None, bool]:
    """Screen one AGI-arm donor and describe its declared spouse payload.

    The returned flag is descriptive: it reports whether any declared spouse
    cell is nonzero. It is not a structural claim about the household a later
    population owner builds, and it does not consult ``filing_status_code``.
    """
    roles = persons.role
    if (
        roles.eq("head").sum() != 1
        or roles.eq("spouse").sum() > 1
        or not roles.isin(("head", "spouse", "dependent")).all()
    ):
        return "unrepresentable_person_roles", False
    monetary = tuple(
        c for c in profile.person_outputs if c not in profile.boolean_outputs
    )
    if (persons.loc[roles.eq("dependent"), list(monetary)].to_numpy() != 0).any():
        return "dependent_monetary_value", False
    # A mixed-type row Series coerces exact int64 identities/amounts to float.
    # Sum Python scalars so signed integer reconciliation cannot wrap or round.
    # The sum is also person-row-order independent: by the two checks above an
    # eligible donor has exactly one head, at most one spouse and exactly zero
    # dependent money, so at most two of the summed scalars are nonzero and
    # float64 addition of one or two values among zeros carries no order-
    # dependent rounding.
    if any(
        sum(persons[column].tolist()) != returns[column].iat[position].item()
        for column in monetary
    ):
        return "person_return_monetary_mismatch", False
    return None, bool(
        (
            persons.loc[roles.eq("spouse"), list(profile.person_outputs)].to_numpy()
            != 0
        ).any()
    )


def _proxy_agi(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return the float64 diagnostic proxy total and the exact floor decision.

    The inclusive floor decision never routes through the float64 total. A
    row's int64 components are summed as Python integers, which is exact at
    any magnitude, and its float64 components are summed with ``math.fsum``,
    which returns the correctly rounded sum of the doubles it was given --
    that is a stable summation, not universally exact real arithmetic. The
    comparison ``float_part >= AGI_FLOOR - integer_part`` is then an exact
    CPython float/int comparison, so an int64 component above 2**53 cannot
    move a donor across the floor by rounding. The returned float64 total is a
    diagnostic view used for the receipt masses and the rank-decile order; it
    is not calculated AGI and is not the threshold authority.
    """
    integer_names = [
        name for name in PROXY_AGI_COMPONENTS if returns[name].dtype == _INT64
    ]
    integer_set = set(integer_names)
    float_names = [name for name in PROXY_AGI_COMPONENTS if name not in integer_set]
    rows = len(returns)
    if integer_names:
        integer_parts = [
            sum(values)
            for values in zip(
                *(returns[name].tolist() for name in integer_names), strict=True
            )
        ]
    else:
        integer_parts = [0] * rows
    try:
        if float_names:
            float_parts = [
                math.fsum(values)
                for values in zip(
                    *(returns[name].tolist() for name in float_names), strict=True
                )
            ]
        else:
            float_parts = [0.0] * rows
    except OverflowError:
        raise ValueError("NATIVE_PUF_TAIL_PROXY_OVERFLOW") from None
    proxy = np.asarray(
        [
            integer + decimal
            for integer, decimal in zip(integer_parts, float_parts, strict=True)
        ],
        dtype=np.float64,
    )
    _require(bool(np.isfinite(proxy).all()), "PROXY_OVERFLOW")
    above_floor = np.asarray(
        [
            decimal >= AGI_FLOOR - integer
            for integer, decimal in zip(integer_parts, float_parts, strict=True)
        ],
        dtype=bool,
    )
    return proxy, above_floor


def _mass(rows: pd.DataFrame) -> dict[str, int | float]:
    """Float64 diagnostic totals. Not an exact reconciliation of the amounts."""
    weights = rows.weight.to_numpy(dtype=np.float64)
    proxy = rows.proxy_agi.to_numpy(dtype=np.float64)
    values = {
        "count": len(rows),
        "weight": float(weights.sum()),
        "proxy_agi_sum": float(proxy.sum()),
        "weighted_proxy_agi": float(np.dot(weights, proxy)),
    }
    _require(all(np.isfinite(v) for v in values.values()), "MASS_OVERFLOW")
    return values


def _cell_quotas(counts: np.ndarray) -> np.ndarray:
    budget = min(MAX_AGI_ONLY_DONORS, int(counts.sum()))
    # An inexpensive invariant rather than a live guard: groupby cells are
    # non-empty, so len(counts) <= counts.sum() and the budget covers them.
    _require(len(counts) <= budget, "CELL_BUDGET")
    if budget == int(counts.sum()):
        return counts.copy()
    quotas = np.ones(len(counts), dtype=np.int64)
    capacity = counts - quotas
    remaining = budget - len(counts)
    ideal = capacity * (remaining / float(capacity.sum()))
    extra = np.floor(ideal).astype(np.int64)
    quotas += extra
    missing = budget - int(quotas.sum())
    # Also invariants, not fixes for an observed counterexample: a negative
    # remainder would bump every cell but one, and the exit checks below are
    # the postconditions the largest-remainder allocation actually owes.
    _require(0 <= missing <= len(counts), "CELL_QUOTA_REMAINDER")
    quotas[np.argsort(-(ideal - extra), kind="stable")[:missing]] += 1
    _require(
        int(quotas.sum()) == budget
        and bool((quotas >= 1).all())
        and bool((quotas <= counts).all()),
        "CELL_QUOTA_POSTCONDITION",
    )
    return quotas


def _rescale_exact(weights: np.ndarray, total: float) -> np.ndarray:
    """Rescale a kept cell so its float64 weight sum is exactly ``total``.

    The reference selector forced the residual onto the last row. A cell with
    dispersed weights can have a last row orders of magnitude smaller than the
    correction, which either drives that weight non-positive or leaves the
    ``nextafter`` walk unable to move the sum at all, refusing a cell that is
    perfectly representable. Correcting a largest weight keeps each step's
    resolution at or above the resolution of the sum being corrected.
    Positivity, finiteness and exact conservation remain mandatory: a cell
    mass that still cannot be represented refuses instead of being approximated.
    """
    result = weights * (total / float(weights.sum()))
    target = int(np.argmax(result))
    for _ in range(8):
        observed = float(result.sum())
        if observed == total:
            break
        result[target] += total - observed
    else:
        for _ in range(64):
            observed = float(result.sum())
            if observed == total:
                break
            result[target] = np.nextafter(
                result[target], np.inf if observed < total else -np.inf
            )
    _require(
        bool(np.isfinite(result).all())
        and bool((result > 0).all())
        and float(result.sum()) == total,
        "EXACT_CELL_WEIGHT",
    )
    return result


def _thin(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object], frozenset[int]]:
    ranked = rows.sort_values(["proxy_agi", "donor_recid"], kind="stable").copy()
    ranked["decile"] = (
        np.arange(len(ranked), dtype=np.int64) * 10 // max(len(ranked), 1)
    )
    cells = list(ranked.groupby(["filing_status_code", "decile"], sort=True))
    quotas = _cell_quotas(np.asarray([len(cell) for _, cell in cells], dtype=np.int64))
    kept_cells, receipts, rescaled = [], [], set()
    for ((status, decile), cell), quota in zip(cells, quotas, strict=True):
        ordered = cell.sort_values("donor_recid", kind="stable").copy()
        before = _mass(ordered)
        if quota == len(ordered):
            kept = ordered.copy()
            treatment = WEIGHT_TREATMENT_UNCHANGED
        else:
            indices = np.floor((np.arange(quota) + 0.5) * len(ordered) / quota).astype(
                int
            )
            kept = ordered.iloc[indices].copy()
            kept["weight"] = _rescale_exact(
                kept.weight.to_numpy(dtype=np.float64), float(before["weight"])
            )
            treatment = WEIGHT_TREATMENT_RESCALED
            rescaled.update(int(recid) for recid in kept.donor_recid)
        after = _mass(kept)
        receipts.append(
            {
                "filing_status_code": int(status),
                "proxy_agi_decile": int(decile),
                "count_before": len(ordered),
                "kept_count": len(kept),
                "dropped_count": len(ordered) - len(kept),
                "weight_treatment": treatment,
                **{
                    name + "_before": before[name]
                    for name in ("weight", "proxy_agi_sum", "weighted_proxy_agi")
                },
                **{
                    name + "_after": after[name]
                    for name in ("weight", "proxy_agi_sum", "weighted_proxy_agi")
                },
                "kept_donor_recids": kept.donor_recid.tolist(),
            }
        )
        kept_cells.append(kept.drop(columns="decile"))
    kept = (
        pd.concat(kept_cells, ignore_index=True) if kept_cells else rows.iloc[:0].copy()
    )
    return (
        kept,
        {
            "maximum_count": MAX_AGI_ONLY_DONORS,
            "count_before": len(rows),
            "kept_count": len(kept),
            "dropped_count": len(rows) - len(kept),
            "rescaled_cell_count": sum(
                1
                for cell in receipts
                if cell["weight_treatment"] == WEIGHT_TREATMENT_RESCALED
            ),
            "amount_mass_conserved": False,
            "cells": receipts,
        },
        frozenset(rescaled),
    )


def select_native_puf_tail(
    projection: DeclaredPufTailRoleProjection,
    *,
    capital_gains_mask: np.ndarray,
    source_eligible: np.ndarray,
) -> NativeTailSelection:
    """Select union arms and thin only AGI-only donors, without a population.

    Capital-gains eligibility/quantiles are a separate, unimplemented source
    qualification: this function takes a declared aligned fixture mask, never
    derives a source-qualified CG mask from fixture values or a preferred score.
    """
    returns, people = projection_tables(projection)
    gains = _mask(capital_gains_mask, len(returns))
    eligible = _mask(source_eligible, len(returns))
    selection_input_sha256 = hashlib.sha256(
        _json(
            {
                "protocol": PROTOCOL,
                "reference_commit": REFERENCE_COMMIT,
                "projection_sha256": projection.sha256,
                "capital_gains_mask_sha256": hashlib.sha256(
                    gains.tobytes()
                ).hexdigest(),
                "source_eligible_sha256": hashlib.sha256(
                    eligible.tobytes()
                ).hexdigest(),
                "agi_floor": AGI_FLOOR,
                "maximum_agi_only_count": MAX_AGI_ONLY_DONORS,
                "proxy_components": PROXY_AGI_COMPONENTS,
            }
        )
    ).hexdigest()
    weights = returns.weight.to_numpy(dtype=np.float64)
    _require(
        not (gains & (~eligible | (weights <= 0))).any(), "CAPITAL_GAINS_ELIGIBILITY"
    )
    proxy, above_floor = _proxy_agi(returns)
    agi = eligible & (weights > 0) & above_floor
    rows = returns.loc[:, ["donor_recid", "weight", "filing_status_code"]].copy()
    rows["proxy_agi"] = proxy
    rows["arm"] = gains.astype(np.int8) + 2 * agi.astype(np.int8)
    groups = {
        int(key): group for key, group in people.groupby("donor_recid", sort=False)
    }
    reasons: dict[str, list[int]] = {}
    # Only role-screened donors get a spouse-payload description. A donor
    # absent from this mapping was never examined, which is not the same as a
    # donor examined and found to have an all-zero spouse payload.
    spouse_payload: dict[int, bool] = {}
    for position in np.flatnonzero(agi):
        recid = int(returns.donor_recid.iat[position])
        reason, spouse_rows_nonzero = _role_eligibility(
            returns,
            int(position),
            groups.get(recid, people.iloc[:0]),
            projection.profile,
        )
        if reason is not None:
            reasons.setdefault(reason, []).append(int(position))
        else:
            spouse_payload[recid] = spouse_rows_nonzero
    skipped_positions = sorted(p for positions in reasons.values() for p in positions)
    keep = gains | agi
    keep[skipped_positions] = False
    candidates = rows.iloc[np.flatnonzero(keep)]
    thinned, thinning, rescaled = _thin(candidates.loc[candidates.arm.eq(2)])
    selected = pd.concat(
        [candidates.loc[candidates.arm.ne(2)], thinned], ignore_index=True
    ).sort_values("donor_recid", kind="stable")
    donors = tuple(
        SelectedTailDonor(
            int(row.donor_recid),
            int(row.arm),
            float(row.proxy_agi),
            float(row.weight),
            spouse_payload.get(int(row.donor_recid)),
            WEIGHT_TREATMENT_RESCALED
            if int(row.donor_recid) in rescaled
            else WEIGHT_TREATMENT_UNCHANGED,
        )
        for row in selected.itertuples(index=False)
    )
    skipped = rows.iloc[skipped_positions].sort_values("donor_recid", kind="stable")
    skipped_index = np.asarray(skipped_positions, dtype=np.int64)
    agi_selected = sum(row.arm in (2, 3) for row in donors)
    capital_gains_selected = sum(row.arm in (1, 3) for row in donors)
    both_selected = sum(row.arm == 3 for row in donors)
    # Every skipped position came out of the AGI loop, so a capital-gains
    # donor can only be skipped when it is also an AGI candidate; CG-only
    # support is never role-screened and never thinned.
    capital_gains_skipped = int(gains[skipped_index].sum())
    both_skipped = int((gains & agi)[skipped_index].sum())
    rescaled_selected = sum(
        row.weight_treatment == WEIGHT_TREATMENT_RESCALED for row in donors
    )
    _require(
        int(agi.sum()) == agi_selected + len(skipped) + thinning["dropped_count"],
        "SELECTION_COUNT_EQUATION",
    )
    _require(
        int(gains.sum()) == capital_gains_selected + capital_gains_skipped,
        "CAPITAL_GAINS_COUNT_EQUATION",
    )
    _require(
        int((gains & agi).sum()) == both_selected + both_skipped,
        "BOTH_ARM_COUNT_EQUATION",
    )
    _require(rescaled_selected == len(rescaled), "WEIGHT_TREATMENT_ACCOUNTING")
    _require(
        all((row.spouse_payload_nonzero is None) == (row.arm == 1) for row in donors),
        "SPOUSE_PAYLOAD_SCOPE",
    )
    receipt = {
        "protocol": PROTOCOL,
        "reference_commit": REFERENCE_COMMIT,
        "projection_sha256": projection.sha256,
        "selection_input_sha256": selection_input_sha256,
        "basis": "invented_fixture",
        "source_admission_issued": False,
        "release_eligible": False,
        "profile": "native52_excluding_late_ss_scf",
        "input_weight_kind": projection.weight_kind.value,
        "selected_weight_kind": WeightKind.IMPORTANCE.value,
        "weight_scope": "donor_return_support_not_population_households",
        "weight_treatment_counts": {
            WEIGHT_TREATMENT_UNCHANGED: len(donors) - rescaled_selected,
            WEIGHT_TREATMENT_RESCALED: rescaled_selected,
        },
        "agi_floor": AGI_FLOOR,
        "comparison": "greater_than_or_equal",
        "filing_status_domain": list(_filing_status_domain()),
        "proxy_components": PROXY_AGI_COMPONENTS,
        "proxy_is_calculated_agi": False,
        "proxy_floor_arithmetic": (
            "exact_integer_part_with_correctly_rounded_float_part"
        ),
        "proxy_totals_arithmetic": (
            "float64_diagnostic_sums_not_exact_amount_reconciliation"
        ),
        "agi_candidate_count": int(agi.sum()),
        "agi_selected_count": agi_selected,
        "capital_gains_candidate_count": int(gains.sum()),
        "capital_gains_selected_count": capital_gains_selected,
        "capital_gains_skipped_count": capital_gains_skipped,
        "both_candidate_count": int((gains & agi).sum()),
        "both_selected_count": both_selected,
        "both_skipped_count": both_skipped,
        "dependent_boolean_policy": "ignored_explicitly",
        "role_eligibility_scope": "agi_only_and_both_arms",
        "spouse_payload_flag": (
            "descriptive_nonzero_declared_spouse_cells_not_a_structural_requirement"
        ),
        "spouse_payload_scope": "null_outside_role_eligibility_scope",
        "capital_gains_selection": "caller_declared_fixture_mask_not_source_qualification",
        "skipped": {
            **_mass(skipped),
            "donor_recids": skipped.donor_recid.tolist(),
            "by_reason": {
                reason: _mass(
                    rows.iloc[positions].sort_values("donor_recid", kind="stable")
                )
                for reason, positions in sorted(reasons.items())
            },
        },
        "thinning": thinning,
    }
    return NativeTailSelection(
        projection.sha256, selection_input_sha256, donors, _json(receipt)
    )
