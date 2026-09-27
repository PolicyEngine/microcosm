"""Weighted paired empirical hurdle model; no country/source authority.

This pooled two-target operator has no predictors and does not run QRF. Model
bytes are canonical plain data. Callers retain source authority and supply
uniforms (the graph adapter uses the maintained keyed-uniform protocol).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import fsum, isfinite

import numpy as np
import pandas as pd

from microcosm.fit import model as weight_model
from microcosm.frame import Frame

PROTOCOL = "microcosm.fit.joint-empirical.v1"
PATTERNS = ((False, False), (True, False), (False, True), (True, True))
MAX_MODEL_BYTES = 64 * 1024**2
MAX_DONORS = 1_048_576


def _require(condition, code):
    if not condition:
        raise ValueError("JOINT_EMPIRICAL_" + code)


def _real(value, code, *, minimum=0):
    _require(type(value) in (int, float), code)
    try:
        result = float(value)
    except OverflowError:
        raise ValueError("JOINT_EMPIRICAL_" + code) from None
    _require(isfinite(result) and result >= minimum, code)
    return result


def _json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _pairs(items):
    out = {}
    for name, value in items:
        _require(name not in out, "DUPLICATE_JSON_KEY")
        out[name] = value
    return out


def _key(value):
    _require(type(value) is tuple and bool(value), "IDENTITY_TUPLE")
    result = []
    for item in value:
        if isinstance(item, np.integer) and not isinstance(item, np.bool_):
            item = int(item)
        _require(type(item) in (int, str), "IDENTITY_TYPE")
        result.append(["int" if type(item) is int else "str", str(item)])
    return result


def _decode_key(value):
    _require(type(value) is list and bool(value), "IDENTITY_ENCODING")
    out = []
    for item in value:
        _require(
            type(item) is list and len(item) == 2 and type(item[1]) is str,
            "IDENTITY_ENCODING",
        )
        kind, text = item
        _require(kind in ("int", "str"), "IDENTITY_ENCODING")
        if kind == "int":
            try:
                number = int(text)
            except ValueError:
                raise ValueError("JOINT_EMPIRICAL_IDENTITY_INTEGER") from None
            _require(str(number) == text, "IDENTITY_INTEGER")
            out.append(number)
        else:
            out.append(text)
    return tuple(out)


@dataclass(frozen=True)
class SupportThreshold:
    """Declared minimum positive-mass support, not an adequacy claim."""

    rows: int
    households: int
    person_ess: float
    household_ess: float

    def document(self):
        _require(type(self.rows) is int and self.rows >= 1, "MIN_ROWS")
        _require(
            type(self.households) is int and self.households >= 1, "MIN_HOUSEHOLDS"
        )
        person = _real(self.person_ess, "MIN_PERSON_ESS", minimum=1)
        household = _real(self.household_ess, "MIN_HOUSEHOLD_ESS", minimum=1)
        return dict(
            rows=self.rows,
            households=self.households,
            person_ess=person,
            household_ess=household,
        )


@dataclass(frozen=True)
class SupportRequirements:
    """Mandatory policy; no production/native support default is supplied."""

    policy_id: str
    scope: str
    overall: SupportThreshold
    positive: SupportThreshold
    pattern: SupportThreshold
    required_patterns: tuple[int, ...]

    def document(self):
        _require(type(self.policy_id) is str and bool(self.policy_id), "POLICY_ID")
        _require(
            type(self.scope) is str and self.scope in ("test", "candidate"),
            "POLICY_SCOPE",
        )
        _require(
            all(
                type(v) is SupportThreshold
                for v in (self.overall, self.positive, self.pattern)
            ),
            "THRESHOLD_TYPE",
        )
        _require(
            type(self.required_patterns) is tuple
            and all(type(p) is int and p in range(4) for p in self.required_patterns)
            and tuple(sorted(set(self.required_patterns))) == self.required_patterns,
            "REQUIRED_PATTERNS",
        )
        return dict(
            policy_id=self.policy_id,
            scope=self.scope,
            overall=self.overall.document(),
            positive=self.positive.document(),
            pattern=self.pattern.document(),
            required_patterns=list(self.required_patterns),
        )


def _support(document):
    _require(
        type(document) is dict
        and set(document)
        == {
            "policy_id",
            "scope",
            "overall",
            "positive",
            "pattern",
            "required_patterns",
        },
        "SUPPORT_DOCUMENT",
    )
    thresholds = []
    for name in ("overall", "positive", "pattern"):
        item = document[name]
        _require(
            type(item) is dict
            and set(item) == {"rows", "households", "person_ess", "household_ess"},
            "THRESHOLD_DOCUMENT",
        )
        thresholds.append(SupportThreshold(**item))
    _require(type(document["required_patterns"]) is list, "REQUIRED_PATTERNS")
    result = SupportRequirements(
        document["policy_id"],
        document["scope"],
        *thresholds,
        tuple(document["required_patterns"]),
    )
    _require(result.document() == document, "SUPPORT_DOCUMENT")
    return result


@dataclass(frozen=True)
class JointTransport:
    """Scalar any-positive mass transport and two positive amount factors.

    Reference is (1, (1, 1)). Receipt factor zero is a model-zero stress and
    still requires the same fitted source/support-qualified model.
    """

    receipt_factor: float
    amount_factors: tuple[float, float]

    def document(self):
        receipt = _real(self.receipt_factor, "RECEIPT_FACTOR")
        _require(
            type(self.amount_factors) is tuple and len(self.amount_factors) == 2,
            "AMOUNT_FACTORS",
        )
        factors = tuple(_real(v, "AMOUNT_FACTOR") for v in self.amount_factors)
        _require(all(v > 0 for v in factors), "AMOUNT_FACTOR")
        return dict(receipt_factor=receipt, amount_factors=list(factors))


def _vector(value, length, name):
    array = np.asarray(value)
    _require(
        array.ndim == 1 and len(array) == length and array.dtype.kind in "iuf", name
    )
    # This operator's arithmetic is binary64. Refuse wider floating inputs
    # before narrowing can erase a nonzero value or hide a negative amount.
    _require(array.dtype.kind != "f" or array.dtype.itemsize <= 8, name + "_DTYPE")
    result = array.astype(np.float64)
    _require(np.isfinite(result).all(), name)
    return result


def _statistics(weights, households, selected):
    positions = np.flatnonzero(selected & (weights > 0))
    if not len(positions):
        return dict(rows=0, households=0, weight=0.0, person_ess=0.0, household_ess=0.0)
    values = weights[positions]
    scaled = values / values.max()
    _require((scaled > 0).all(), "WEIGHT_UNDERFLOW")
    total = fsum(scaled)
    groups = {}
    for i, weight in zip(positions, scaled, strict=True):
        groups.setdefault(households[i], []).append(float(weight))
    group_weights = [fsum(items) for items in groups.values()]
    try:
        original_mass = fsum(values)
    except OverflowError:
        raise ValueError("JOINT_EMPIRICAL_WEIGHT_OVERFLOW") from None
    _require(isfinite(original_mass), "WEIGHT_OVERFLOW")
    return dict(
        rows=len(positions),
        households=len(groups),
        weight=original_mass,
        person_ess=total**2 / fsum(float(v) ** 2 for v in scaled),
        household_ess=total**2 / fsum(v**2 for v in group_weights),
    )


def _enforce(stats, threshold, name):
    declared = threshold.document()
    for measure in declared:
        _require(stats[measure] >= declared[measure], "SUPPORT:" + name + ":" + measure)


def _cdf(weights):
    """Normalize cumulative positive masses; refuse lost positive intervals.

    The final endpoint is one because the cumulative vector is divided by its
    own finite positive endpoint. No probability parameter is clipped.
    """
    _require(
        len(weights) > 0 and np.isfinite(weights).all() and (weights > 0).all(),
        "CDF_MASS",
    )
    scaled = weights / weights.max()
    _require((scaled > 0).all(), "WEIGHT_UNDERFLOW")
    cumulative = np.cumsum(scaled, dtype=np.float64)
    _require(
        np.isfinite(cumulative).all() and (np.diff(np.r_[0.0, cumulative]) > 0).all(),
        "LOST_POSITIVE_INTERVAL",
    )
    result = cumulative / cumulative[-1]
    _require(
        (np.diff(np.r_[0.0, result]) > 0).all() and result[-1] == 1,
        "LOST_POSITIVE_INTERVAL",
    )
    return result


def _prepare(document):
    _require(
        type(document) is dict
        and set(document)
        == {"protocol", "targets", "support", "donors", "weight_kind"},
        "MODEL_DOCUMENT",
    )
    _require(document["protocol"] == PROTOCOL, "MODEL_PROTOCOL")
    _require(
        type(document["weight_kind"]) is str
        and document["weight_kind"]
        in ("explicit", "design", "importance", "calibrated"),
        "WEIGHT_KIND",
    )
    targets = document["targets"]
    _require(
        type(targets) is list
        and len(targets) == 2
        and all(type(v) is str and v for v in targets)
        and targets[0] != targets[1],
        "TARGETS",
    )
    support = _support(document["support"])
    rows = document["donors"]
    _require(type(rows) is list and 1 <= len(rows) <= MAX_DONORS, "DONOR_COUNT")
    keys, households, values, weights = [], [], [], []
    for row in rows:
        _require(
            type(row) is dict and set(row) == {"key", "household", "values", "weight"},
            "DONOR_DOCUMENT",
        )
        key, household = _decode_key(row["key"]), _decode_key(row["household"])
        _require(
            type(row["values"]) is list and len(row["values"]) == 2, "TARGET_VALUES"
        )
        pair = [_real(v, "TARGET_VALUE") for v in row["values"]]
        weight = _real(row["weight"], "WEIGHT")
        keys.append(key)
        households.append(_json(_key(household)))
        values.append(pair)
        weights.append(weight)
    order_keys = [_json(_key(key)) for key in keys]
    _require(order_keys == sorted(set(order_keys)), "DONOR_ORDER_OR_DUPLICATE")
    values, weights = (
        np.asarray(values, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )
    _require((weights > 0).any(), "ZERO_WEIGHT_POOL")
    patterns = (values[:, 0] > 0).astype(np.int8) + 2 * (values[:, 1] > 0).astype(
        np.int8
    )
    overall = _statistics(weights, households, np.ones(len(weights), dtype=bool))
    positive = _statistics(weights, households, patterns > 0)
    stats = [_statistics(weights, households, patterns == g) for g in range(4)]
    _enforce(overall, support.overall, "overall")
    _enforce(positive, support.positive, "positive")
    for g, item in enumerate(stats):
        if g in support.required_patterns or item["rows"]:
            _enforce(item, support.pattern, "pattern" + str(g))
    masses = np.asarray([s["weight"] for s in stats])
    total = fsum(masses)
    probabilities = masses / total
    _require(
        np.isfinite(probabilities).all()
        and ((masses == 0) | (probabilities > 0)).all(),
        "PATTERN_UNDERFLOW",
    )
    cdfs = {
        g: _cdf(weights[(patterns == g) & (weights > 0)])
        for g in range(4)
        if masses[g] > 0
    }
    diagnostic = dict(
        weight_kind=document["weight_kind"],
        raw_rows=len(weights),
        raw_zero_pair_rows=int((patterns == 0).sum()),
        raw_positive_pair_rows=int((patterns > 0).sum()),
        zero_weight_rows=int((weights == 0).sum()),
        positive_weight_zero_pair_rows=int(((weights > 0) & (patterns == 0)).sum()),
        positive_weight_positive_pair_rows=int(((weights > 0) & (patterns > 0)).sum()),
        overall=overall,
        positive=positive,
        patterns=stats,
        pattern_probabilities=probabilities.tolist(),
        absent_patterns=[g for g in range(4) if masses[g] == 0],
    )
    return keys, values, weights, patterns, cdfs, probabilities, diagnostic


@dataclass(frozen=True)
class JointEmpiricalModel:
    """Descriptive model data; constructing/copying grants no source authority."""

    payload: bytes

    def _checked(self):
        _require(
            type(self.payload) is bytes and 0 < len(self.payload) <= MAX_MODEL_BYTES,
            "MODEL_BYTES",
        )
        try:
            document = json.loads(self.payload, object_pairs_hook=_pairs)
        except (UnicodeError, json.JSONDecodeError):
            raise ValueError("JOINT_EMPIRICAL_MODEL_JSON") from None
        result = _prepare(document)
        _require(_json(document) == self.payload, "MODEL_CANONICAL_BYTES")
        return result

    @classmethod
    def from_bytes(cls, payload: bytes) -> JointEmpiricalModel:
        result = cls(payload)
        result._checked()
        return result

    def to_bytes(self) -> bytes:
        self._checked()
        return self.payload

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @property
    def diagnostics(self) -> dict:
        return self._checked()[-1]


@dataclass(frozen=True)
class JointEmpiricalDraw:
    """One pair per supplied uniform coordinate; never observed source amounts."""

    values: np.ndarray
    patterns: np.ndarray
    donor_keys: tuple[tuple | None, ...]
    model_sha256: str
    transport: bytes


def fit_joint_empirical(
    frame_or_df: Frame | pd.DataFrame,
    *,
    targets: tuple[str, str],
    donor_keys: Sequence[tuple],
    household_keys: Sequence[tuple],
    support: SupportRequirements,
    weights=weight_model.DESIGN_WEIGHTS,
) -> JointEmpiricalModel:
    """Fit a pooled paired empirical law using explicit/typed supplied weights."""
    _require(
        type(targets) is tuple
        and len(targets) == 2
        and targets[0] != targets[1]
        and all(type(v) is str and v for v in targets),
        "TARGETS",
    )
    _require(type(support) is SupportRequirements, "SUPPORT_TYPE")
    entity = None
    if isinstance(frame_or_df, Frame):
        owners = {frame_or_df.column_entity(v) for v in targets}
        _require(len(owners) == 1, "TARGET_ENTITY")
        entity = next(iter(owners))
        frame = frame_or_df.table(entity)
        _vector(frame_or_df.resolve_weights(entity).values, len(frame), "WEIGHTS")
        resolved = weight_model.resolve_fit_weights(frame_or_df, entity, weights)
    else:
        _require(type(frame_or_df) is pd.DataFrame, "FIT_INPUT")
        frame = frame_or_df
        raw_weights = (
            frame[weights] if isinstance(weights, str) and weights in frame else weights
        )
        if not isinstance(raw_weights, str):
            _vector(raw_weights, len(frame), "WEIGHTS")
        resolved = weight_model.resolve_dataframe_fit_weights(
            frame, weights, predictors=[], targets=list(targets)
        )
    _require(frame.columns.is_unique and set(targets) <= set(frame), "TARGET_COLUMNS")
    _require(resolved is not None, "WEIGHTS_REQUIRED")
    _require(
        1 <= len(frame) <= MAX_DONORS
        and len(donor_keys) == len(household_keys) == len(frame),
        "DONOR_AXIS",
    )
    values = np.column_stack(
        [_vector(frame[name], len(frame), "TARGET_VALUES") for name in targets]
    )
    _require((values >= 0).all(), "NEGATIVE_TARGET")
    resolved = _vector(resolved, len(frame), "WEIGHTS")
    _require((resolved >= 0).all(), "NEGATIVE_WEIGHT")
    records = [
        dict(key=_key(k), household=_key(h), values=y.tolist(), weight=float(w))
        for k, h, y, w in zip(donor_keys, household_keys, values, resolved, strict=True)
    ]
    records.sort(key=lambda row: _json(row["key"]))
    weight_kind = weight_model.resolved_weight_kind(frame_or_df, entity, resolved)
    payload = _json(
        dict(
            protocol=PROTOCOL,
            targets=list(targets),
            support=support.document(),
            donors=records,
            weight_kind=weight_kind,
        )
    )
    return JointEmpiricalModel.from_bytes(payload)


def draw_joint_empirical(
    model: JointEmpiricalModel,
    *,
    pattern_uniforms,
    donor_uniforms,
    transport: JointTransport,
) -> JointEmpiricalDraw:
    """Draw a pattern and one paired donor using caller-supplied [0,1) uniforms."""
    _require(
        type(model) is JointEmpiricalModel and type(transport) is JointTransport,
        "DRAW_TYPE",
    )
    keys, values, weights, patterns, cdfs, probabilities, _ = model._checked()
    specification = transport.document()
    receipt = specification["receipt_factor"]
    positive = fsum(probabilities[1:])
    transported_positive = receipt * positive
    _require(
        isfinite(transported_positive) and 0 <= transported_positive <= 1,
        "TRANSPORT_PROBABILITY",
    )
    # The reference law already carries its measured zero-pattern mass. Its
    # complement may round to zero even when that leading interval is
    # representable (for example masses 2**-54 and 1). Retain all reference
    # masses and use the same declared cumulative-endpoint normalization.
    masses = (
        probabilities.copy()
        if receipt == 1
        else np.r_[1 - transported_positive, receipt * probabilities[1:]]
    )
    _require(np.isfinite(masses).all() and (masses >= 0).all(), "TRANSPORT_PROBABILITY")
    _require(
        not receipt or ((probabilities[1:] == 0) | (masses[1:] > 0)).all(),
        "TRANSPORT_UNDERFLOW",
    )
    active = np.flatnonzero(masses > 0)
    pattern_cdf = _cdf(masses[active])
    u = np.asarray(pattern_uniforms)
    _require(u.ndim == 1, "UNIFORMS")
    u = _vector(u, len(u), "PATTERN_UNIFORMS")
    v = _vector(donor_uniforms, len(u), "DONOR_UNIFORMS")
    _require(((u >= 0) & (u < 1)).all() and ((v >= 0) & (v < 1)).all(), "UNIFORM_RANGE")
    selected = active[np.searchsorted(pattern_cdf, u, side="right")]
    result = np.zeros((len(u), 2))
    selected_keys = [None] * len(u)
    factors = np.asarray(specification["amount_factors"])
    for g in range(1, 4):
        recipients = np.flatnonzero(selected == g)
        if not len(recipients):
            continue
        _require(g in cdfs, "UNSUPPORTED_PATTERN")
        positions = np.flatnonzero((patterns == g) & (weights > 0))
        donors = positions[np.searchsorted(cdfs[g], v[recipients], side="right")]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            transported = values[donors] * factors
        _require(
            np.isfinite(transported).all()
            and np.array_equal(transported > 0, values[donors] > 0),
            "AMOUNT_TRANSPORT_RANGE",
        )
        result[recipients] = transported
        for recipient, donor in zip(recipients, donors, strict=True):
            selected_keys[recipient] = keys[donor]
    frozen_values = np.frombuffer(result.tobytes(), dtype=np.float64).reshape((-1, 2))
    frozen_patterns = np.frombuffer(selected.astype(np.int8).tobytes(), dtype=np.int8)
    return JointEmpiricalDraw(
        frozen_values,
        frozen_patterns,
        tuple(selected_keys),
        hashlib.sha256(model.payload).hexdigest(),
        _json(specification),
    )
