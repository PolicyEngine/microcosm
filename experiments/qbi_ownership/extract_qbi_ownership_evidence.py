"""Extract QBI amount-margin ownership evidence from the failed-attempt checkpoints.

Read-only forensic extraction over the SHA-verified f025 failed-attempt stacked
checkpoints cited by ``experiments/battery_burndown/ADJUDICATION.md`` (line 228).
No build, no artifact mutation, no logbook interaction.

For each of the four red QBI amount legs
(``qualified_bdc_income``, ``qualified_reit_and_ptp_income``,
``unadjusted_basis_qualified_property``, ``w2_wages_from_qualified_business``)
this script:

1. verifies the simulated checkpoint's SHA-256 against the adjudication pin and
   records the transferred/assembled digests beside it;
2. reproduces the terminal by-origin battery numbers (positive-leg incidence
   ratio, carrier counts, QED) on the simulated frame with the battery's exact
   math (`stacked_spine.py` `_battery_sign_separated_comparison`,
   `_battery_conditional_quantiles`, `_battery_quantile_envelope_distance`)
   and asserts agreement with the adjudication's quoted values;
3. recomputes the same statistics at the pre-reconciliation ``transferred``
   stage and inside the clone-1 producer scope, so each red clone-0 margin
   decomposes into producer, late-transfer, and QBI-reconciliation stages;
4. recomputes the realized QRF regime of every late QBI availability pattern
   from the frozen ASEC clone-1 donor support (``microcosm.fit.qrf.detect_regime``
   with the banked ``zero_atol``), cross-checking donor-row counts and
   per-pattern donor-index digests against the banked chain states; and
5. reruns the nine coupled QBI invariants at ``1e-8`` on both stages.

Output: a machine-readable JSON evidence file plus a console summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from microcosm.build.frame_checkpoint import _read_index, _read_series
from microcosm.fit.qrf import detect_regime

CHECKPOINT_DIR = Path(
    "/Users/maxghenis/PolicyEngine/_buildo-runtime/out/stacked-f025-r1/"
    "populace_us_2024_stacked_pool.checkpoints/stacked/"
    "9be8ecdf82356f38998e8b620ee36d9134f554fe89a8eafd8406f438e2b5aad6"
)

#: Adjudication-pinned digest for the simulated checkpoint (ADJUDICATION.md:228).
SIMULATED_SHA256 = "5b47eb0ded02f4031e235b7a6e07506b5bd38f87827644752d26f4263e492f5a"
ASSEMBLED_SHA256 = "3b50bbd6abca781ea5dc23c63e6128e4ee934042068fb32b94afb98eeb4d2540"
TRANSFERRED_SHA256 = "bdc9355d92659bb28d58b1ddcd647ec303f2ad217661e17d5b4b0984e04532e8"
ASSEMBLED_IDENTITY_SHA256 = (
    "4ad8dfdcf7b31655deaa24f05176ae39b1dae06d9265409fbab4104ace8cda80"
)
TRANSFERRED_IDENTITY_SHA256 = (
    "7847224d1daa6b21cb4d063b1b54b62516b24936a302513143a685b61b7cf8e4"
)
SIMULATED_IDENTITY_SHA256 = (
    "55c9c86613a91c12affc38e7b238d6a35b1fa6ee81340afdeeec1ed1db2c5ee7"
)
QBI_TRANSITION_AUTHORITY_SHA256 = (
    "524656612a137acfd236753c51b5ddec98ff58d57de1ad262594763919fdfb6d"
)
PUBLICATION_RELEASE_ID = (
    "populace-us-2024-stacked-f025-s578-asec42213-acs382903-20260816T145820Z-80e26cb5"
)
PUBLICATION_RUN_ID = "378f7af26eb24667be35de7cfe595d27"
PUBLICATION_STATUS = "gate_failed"
PUBLICATION_SIMULATION_READY = False
PUBLICATION_MANIFEST_SHA256 = (
    "e341b027bfd6f18996936caedd60f05fef073db8cb52d68da7418a0549c88e3a"
)
PUBLICATION_GATES_SHA256 = (
    "685cad63d4dc62234da72501c5a3ce9ec5a81fcd3f7b412b61474a9c1d8b423b"
)
LATE_PRODUCER_OPERATIONAL_RECEIPT_SHA256 = (
    "1b1867c29124a52b853923ea8f42ee591062363a6b72011580e8b9cdd70e4c4b"
)
POOL_BASENAME = "populace_us_2024_stacked_pool"
CANONICAL_OUTPUT = Path(__file__).resolve().parent / "evidence.json"

CHECKPOINT_BINDINGS = {
    "assembled": {
        "sha256": ASSEMBLED_SHA256,
        "identity_sha256": ASSEMBLED_IDENTITY_SHA256,
    },
    "transferred": {
        "sha256": TRANSFERRED_SHA256,
        "identity_sha256": TRANSFERRED_IDENTITY_SHA256,
    },
    "simulated": {
        "sha256": SIMULATED_SHA256,
        "identity_sha256": SIMULATED_IDENTITY_SHA256,
    },
}

#: Battery constants mirrored from stacked_spine.py:13892-13894 and the
#: support profile at stacked_spine.py:3177-3181.
INCIDENCE_BOUNDS = (0.8, 1.25)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
QED_TOLERANCE = 0.25
MIN_EFFECTIVE_SUPPORT = 5

#: Support-provenance column names (support_provenance.py:333-344) and
#: channel/clone labels (support_provenance.py:31-34).
PERSON_CHANNEL = "person_support_channel"
PERSON_CLONE = "person_support_clone_index"
ASEC = "asec"
ACS = "acs"
PUF_TAX_DETAIL_CLONE_INDEX = 1

AMOUNT_TARGETS = (
    "qualified_bdc_income",
    "qualified_reit_and_ptp_income",
    "unadjusted_basis_qualified_property",
    "w2_wages_from_qualified_business",
)
SSTB_BOOLEANS = (
    "sstb_self_employment_income_would_be_qualified",
    "business_is_sstb",
)

#: Full-precision baseline values for adjudication rows 57--64. These are kept
#: as numeric pins: formatting to the battery's six display digits would erase
#: precisely the binding this forensic extractor is meant to establish.
ADJUDICATED_LEGS = {
    "qualified_bdc_income": {
        "ratio_acs_over_asec": 0.3289281071143448,
        "asec_incidence": 0.002557056591401217,
        "acs_incidence": 0.0008410877843938609,
        "quantile_envelope_distance": 0.751361328298312,
        "carriers_asec": 228,
        "carriers_acs": 863,
    },
    "qualified_reit_and_ptp_income": {
        "ratio_acs_over_asec": 0.44658482815239,
        "asec_incidence": 0.017372315783429463,
        "acs_incidence": 0.007758212658751899,
        "quantile_envelope_distance": 1.1576246415399343,
        "carriers_asec": 1740,
        "carriers_acs": 7781,
    },
    "unadjusted_basis_qualified_property": {
        "ratio_acs_over_asec": 0.7745478326179687,
        "asec_incidence": 0.05877937845587233,
        "acs_incidence": 0.04552744018562724,
        "quantile_envelope_distance": 0.5901547288221703,
        "carriers_asec": 6137,
        "carriers_acs": 41761,
    },
    "w2_wages_from_qualified_business": {
        "ratio_acs_over_asec": 0.7609745871477333,
        "asec_incidence": 0.004911295390127669,
        "acs_incidence": 0.003737370981862969,
        "quantile_envelope_distance": 1.3371284036777606,
        "carriers_asec": 509,
        "carriers_acs": 3442,
    },
}

ADJUDICATED_CHECKS = {
    ("person/puf_tax_itemization/qualified_bdc_income[clone_0]/positive#incidence"): {
        "target": "qualified_bdc_income",
        "criterion": "incidence",
        "criterion_value": 0.3289281071143448,
        "first_failing_stage": "late_cross_role_transfer",
    },
    ("person/puf_tax_itemization/qualified_bdc_income[clone_0]/positive#quantile"): {
        "target": "qualified_bdc_income",
        "criterion": "quantile",
        "criterion_value": 0.751361328298312,
        "first_failing_stage": "clone1_puf_producer",
    },
    (
        "person/puf_tax_itemization/qualified_reit_and_ptp_income[clone_0]/"
        "positive#incidence"
    ): {
        "target": "qualified_reit_and_ptp_income",
        "criterion": "incidence",
        "criterion_value": 0.44658482815239,
        "first_failing_stage": "late_cross_role_transfer",
    },
    (
        "person/puf_tax_itemization/qualified_reit_and_ptp_income[clone_0]/"
        "positive#quantile"
    ): {
        "target": "qualified_reit_and_ptp_income",
        "criterion": "quantile",
        "criterion_value": 1.1576246415399343,
        "first_failing_stage": "clone1_puf_producer",
    },
    (
        "person/puf_tax_itemization/unadjusted_basis_qualified_property[clone_0]/"
        "positive#incidence"
    ): {
        "target": "unadjusted_basis_qualified_property",
        "criterion": "incidence",
        "criterion_value": 0.7745478326179687,
        "first_failing_stage": "late_cross_role_transfer",
    },
    (
        "person/puf_tax_itemization/unadjusted_basis_qualified_property[clone_0]/"
        "positive#quantile"
    ): {
        "target": "unadjusted_basis_qualified_property",
        "criterion": "quantile",
        "criterion_value": 0.5901547288221703,
        "first_failing_stage": "late_cross_role_transfer",
    },
    (
        "person/puf_tax_itemization/w2_wages_from_qualified_business[clone_0]/"
        "positive#incidence"
    ): {
        "target": "w2_wages_from_qualified_business",
        "criterion": "incidence",
        "criterion_value": 0.7609745871477333,
        "first_failing_stage": "late_cross_role_transfer",
    },
    (
        "person/puf_tax_itemization/w2_wages_from_qualified_business[clone_0]/"
        "positive#quantile"
    ): {
        "target": "w2_wages_from_qualified_business",
        "criterion": "quantile",
        "criterion_value": 1.3371284036777606,
        "first_failing_stage": "clone1_puf_producer",
    },
}

TERMINAL_ORIGIN_CHANNELS = frozenset({"qrf_transfer"})
FAILING_STAGES = frozenset(
    {"clone1_puf_producer", "late_cross_role_transfer", "qbi_reconciliation"}
)
QRF_REGIMES = frozenset(
    {
        "three_sign",
        "zero_inflated_positive",
        "zero_inflated_negative",
        "sign_only",
        "positive_only",
        "negative_only",
        "degenerate_zero",
    }
)
QBI_INVARIANT_NAMES = (
    "sstb_rows_with_non_sstb_income",
    "non_sstb_rows_with_sstb_income",
    "sstb_w2_split_mismatches",
    "sstb_ubia_split_mismatches",
    "self_employment_qualification_overlap",
    "sstb_qualification_route_mismatches",
    "non_sstb_qualification_route_mismatches",
    "qualified_bdc_exposure_mismatches",
    "qualified_reit_ptp_exposure_mismatches",
)
TOP_LEVEL_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "checkpoint_dir",
        "adjudication_pin_simulated_sha256",
        "artifact_bindings",
        "row_counts",
        "bank_patterns",
        "amount_legs",
        "ownership_checks",
        "ownership_summary",
        "sstb_booleans",
        "recipient_context",
        "realized_regimes",
        "invariants",
        "adjudication_mismatches",
    }
)
ADJUDICATED_SSTB_CLONE1 = {
    "sstb_self_employment_income_would_be_qualified": {
        "asec_incidence": "0.0350747",
        "acs_incidence": "0.0372869",
        "ratio": "1.06307",
        "carriers_asec": 3735,
        "carriers_acs": 33902,
    },
    "business_is_sstb": {
        "asec_incidence": "0.0352413",
        "acs_incidence": "0.0375576",
        "ratio": "1.06573",
        "carriers_asec": 3749,
        "carriers_acs": 34166,
    },
}

BATCH_4 = "puf_tax_itemization__batch_4"
BATCH_5 = "puf_tax_itemization__batch_5"
BATCH_FAMILY_TARGETS = {
    BATCH_4: (
        "farm_rent_income_would_be_qualified",
        "partnership_s_corp_income_would_be_qualified",
        "rental_income_would_be_qualified",
        "self_employment_income_would_be_qualified",
        "sstb_self_employment_income_would_be_qualified",
        "business_is_sstb",
        "qualified_bdc_income",
        "qualified_reit_and_ptp_income",
    ),
    BATCH_5: (
        "sstb_self_employment_income_before_lsr",
        "sstb_unadjusted_basis_qualified_property",
        "sstb_w2_wages_from_qualified_business",
        "unadjusted_basis_qualified_property",
        "w2_wages_from_qualified_business",
    ),
}

EXTRA_COLUMNS = (
    "self_employment_income_before_lsr",
    "non_qualified_dividend_income",
    "partnership_income",
    "s_corp_income",
    "person_household_id",
    PERSON_CHANNEL,
    PERSON_CLONE,
    "age",
    "employment_income_before_lsr",
)

#: Source columns behind the late QBI transfer's predictor surface and the
#: reconciliation cap bases, profiled per channel to diagnose the
#: recipient-gate collapse (`evidence.json.recipient_context`).
CONTEXT_COLUMNS = (
    "age",
    "employment_income_before_lsr",
    "self_employment_income_before_lsr",
    "non_qualified_dividend_income",
    "partnership_income",
    "s_corp_income",
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(value: dict[str, object]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path) -> dict[str, object]:
    with path.open() as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    """Durably replace ``path`` only after the complete JSON is serialized."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            temporary = Path(fh.name)
            json.dump(value, fh, indent=1, sort_keys=True, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def index_identity(index: pd.Index) -> dict[str, object]:
    """Mirror ``microcosm.fit.qrf._index_identity`` exactly."""

    hashes = pd.util.hash_pandas_object(
        index,
        index=False,
        categorize=False,
    ).to_numpy(dtype="<u8", copy=False)
    digest = hashlib.sha256()
    digest.update(hashes.tobytes(order="C"))
    return {
        "length": len(index),
        "class_name": type(index).__name__,
        "dtype": str(index.dtype),
        "names_repr": repr(tuple(index.names)),
        "sha256": digest.hexdigest(),
    }


def validated_index_identity(value: object, *, boundary: str) -> dict[str, object]:
    """Return a complete banked QRF index identity or fail closed."""

    if not isinstance(value, dict):
        raise TypeError(f"{boundary} must be an object")
    expected_keys = {"length", "class_name", "dtype", "names_repr", "sha256"}
    if set(value) != expected_keys:
        raise ValueError(
            f"{boundary} fields changed: got {sorted(value)}, "
            f"expected {sorted(expected_keys)}"
        )
    length = value["length"]
    if not isinstance(length, int) or isinstance(length, bool) or length < 0:
        raise ValueError(f"{boundary}.length must be a nonnegative integer")
    for key in ("class_name", "dtype", "names_repr"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError(f"{boundary}.{key} must be a nonempty string")
    digest = value["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{boundary}.sha256 is malformed")
    return dict(value)


def exact_numeric_agreement(observed: object, expected: object) -> bool:
    """Compare counts and binary64 results exactly."""

    if isinstance(expected, int):
        return observed == expected
    if not isinstance(observed, (float, int)) or not isinstance(expected, float):
        return False
    return float(observed) == expected


def criterion_value(stats: dict[str, object], criterion: str) -> float:
    if criterion == "incidence":
        return float(stats["incidence_ratio_acs_over_asec"])
    if criterion == "quantile":
        return float(stats["quantile_envelope_distance"])
    raise ValueError(f"unsupported battery criterion {criterion!r}")


def criterion_failed(stats: dict[str, object], criterion: str) -> bool:
    if criterion == "incidence":
        return not bool(stats["incidence_in_band"])
    if criterion == "quantile":
        return not bool(stats["qed_within_tolerance"])
    raise ValueError(f"unsupported battery criterion {criterion!r}")


class CheckpointColumns:
    """Targeted positional column reader for a `_populace_frame_checkpoint` h5."""

    def __init__(self, path: Path):
        self.path = path
        self._h5 = h5py.File(path, "r")
        root = self._h5["_populace_frame_checkpoint"]
        self._meta = json.loads(bytes(root["metadata_json"][()]).decode("utf-8"))
        self._tables = root["tables"]
        self._weights_group = root["weights"]
        self._table_lookup: dict[str, tuple[int, dict[str, tuple[int, dict]]]] = {}
        self._indices: dict[str, pd.Index] = {}
        for t_index, spec in enumerate(self._meta["tables"]):
            columns = {
                column_spec["name"]: (c_index, column_spec)
                for c_index, column_spec in enumerate(spec["columns"])
            }
            self._table_lookup[spec["name"]] = (t_index, columns)
            table_group = self._tables[f"t{t_index:05d}"]
            self._indices[spec["name"]] = _read_index(
                table_group,
                spec["index"],
                self.path,
            )

    def close(self) -> None:
        self._h5.close()

    def column(self, table: str, column: str) -> pd.Series:
        t_index, columns = self._table_lookup[table]
        if column not in columns:
            raise KeyError(f"{self.path.name}:{table} has no column {column!r}")
        c_index, spec = columns[column]
        group = self._tables[f"t{t_index:05d}"]["columns"][f"c{c_index:05d}"]
        series = _read_series(group, spec, self.path, label=f"{table}.{column}")
        series.index = self._indices[table]
        return series

    def has_column(self, table: str, column: str) -> bool:
        return column in self._table_lookup[table][1]

    def weights(self, entity: str) -> np.ndarray:
        for w_index, spec in enumerate(self._meta["weights"]):
            if spec["entity"] == entity:
                return np.asarray(
                    self._weights_group[f"w{w_index:05d}"][()], dtype=np.float64
                )
        raise KeyError(f"{self.path.name} stores no weights for {entity!r}")


def person_weights(reader: CheckpointColumns) -> np.ndarray:
    """Broadcast household importance weights to persons through membership.

    Mirrors Frame.resolve_weights person inheritance (microcosm-frame
    bundle.py:516-546) and the adjudication's extraction ("mapped person
    households to w00000").
    """

    household_ids = reader.column("household", "household_id").to_numpy()
    household_weights = reader.weights("household")
    if len(household_ids) != len(household_weights):
        raise ValueError("household weights do not align to the household table")
    lookup = pd.Series(household_weights, index=household_ids)
    if not lookup.index.is_unique:
        raise ValueError("household ids are not unique")
    memberships = reader.column("person", "person_household_id").to_numpy()
    mapped = lookup.reindex(memberships).to_numpy(dtype=np.float64)
    if not np.isfinite(mapped).all():
        raise ValueError("person household membership resolved nonfinite weights")
    if (mapped < 0.0).any():
        raise ValueError("person household membership resolved negative weights")
    return mapped


def to_float(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Float64").to_numpy(dtype=np.float64, na_value=np.nan)


def weighted_incidence(mask: np.ndarray, weights: np.ndarray) -> float:
    """stacked_spine.py `_weighted_mask_incidence` (denominator = scope weight)."""

    total = float(weights.sum())
    if total <= 0.0:
        return 0.0
    return float(weights[mask].sum() / total)


def battery_quantiles(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """stacked_spine.py `_battery_conditional_quantiles` (inverse ECDF)."""

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    positions = np.minimum(
        np.searchsorted(cumulative, np.asarray(QUANTILES), side="left"),
        len(sorted_values) - 1,
    )
    return sorted_values[positions]


def quantile_envelope_distance(left: np.ndarray, right: np.ndarray) -> float:
    """stacked_spine.py `_battery_quantile_envelope_distance`."""

    denominator = np.abs(left) + np.abs(right)
    distances = np.divide(
        2.0 * np.abs(left - right),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    return float(np.max(distances))


def positive_leg_stats(
    values: np.ndarray,
    scope: np.ndarray,
    channel: np.ndarray,
    weights: np.ndarray,
) -> dict[str, object]:
    """Positive-leg incidence + QED within one clone scope, split asec/acs.

    Mirrors `_battery_sign_separated_comparison` for the positive leg only:
    incidence denominators are the full origin scopes, carriers are raw row
    counts, and the QED runs on abs(values) over carriers when both origins
    have at least MIN_EFFECTIVE_SUPPORT carriers.
    """

    left_rows = scope & (channel == ASEC)
    right_rows = scope & (channel == ACS)
    compared = left_rows | right_rows
    if not np.isfinite(values[compared]).all():
        raise ValueError("battery amount scope contains nonfinite values")
    leg = values > 0.0
    left_leg = leg & left_rows
    right_leg = leg & right_rows
    left_incidence = weighted_incidence(left_leg[left_rows], weights[left_rows])
    right_incidence = weighted_incidence(right_leg[right_rows], weights[right_rows])
    record: dict[str, object] = {
        "asec_incidence": left_incidence,
        "acs_incidence": right_incidence,
        "carriers_asec": int(left_leg.sum()),
        "carriers_acs": int(right_leg.sum()),
        "scope_rows_asec": int(left_rows.sum()),
        "scope_rows_acs": int(right_rows.sum()),
    }
    if left_incidence == 0.0 and right_incidence == 0.0:
        record["status"] = "absent_on_both_origins"
        return record
    ratio = math.inf if left_incidence == 0.0 else right_incidence / left_incidence
    record["incidence_ratio_acs_over_asec"] = ratio
    record["incidence_in_band"] = bool(
        INCIDENCE_BOUNDS[0] <= ratio <= INCIDENCE_BOUNDS[1]
    )
    if (
        record["carriers_asec"] >= MIN_EFFECTIVE_SUPPORT
        and record["carriers_acs"] >= MIN_EFFECTIVE_SUPPORT
    ):
        left_q = battery_quantiles(np.abs(values[left_leg]), weights[left_leg])
        right_q = battery_quantiles(np.abs(values[right_leg]), weights[right_leg])
        distance = quantile_envelope_distance(left_q, right_q)
        record["quantiles_asec"] = [float(q) for q in left_q]
        record["quantiles_acs"] = [float(q) for q in right_q]
        record["quantile_envelope_distance"] = distance
        record["qed_within_tolerance"] = bool(distance <= QED_TOLERANCE)
    else:
        record["quantile_envelope"] = "leg_insufficient_support"
    return record


def boolean_stats(
    values: np.ndarray,
    scope: np.ndarray,
    channel: np.ndarray,
    weights: np.ndarray,
) -> dict[str, object]:
    """Boolean-incidence comparison mirroring `_battery_incidence_comparison`."""

    left_rows = scope & (channel == ASEC)
    right_rows = scope & (channel == ACS)
    compared = left_rows | right_rows
    if not np.isfinite(values[compared]).all():
        raise ValueError("battery boolean scope contains nonfinite values")
    nonzero = values != 0.0
    left = weighted_incidence(nonzero[left_rows], weights[left_rows])
    right = weighted_incidence(nonzero[right_rows], weights[right_rows])
    record: dict[str, object] = {
        "asec_incidence": left,
        "acs_incidence": right,
        "carriers_asec": int((nonzero & left_rows).sum()),
        "carriers_acs": int((nonzero & right_rows).sum()),
    }
    if left == 0.0 and right == 0.0:
        record["status"] = "dead_comparison"
        return record
    ratio = math.inf if left == 0.0 else right / left
    record["incidence_ratio_acs_over_asec"] = ratio
    record["incidence_in_band"] = bool(
        INCIDENCE_BOUNDS[0] <= ratio <= INCIDENCE_BOUNDS[1]
    )
    return record


def fmt(value: float) -> str:
    return f"{value:.6g}"


def load_stage(reader: CheckpointColumns) -> dict[str, object]:
    person_columns = {}
    membership = reader.column("person", "person_household_id")
    person_columns["__person_index__"] = membership.index.copy()
    needed = set(EXTRA_COLUMNS)
    for family_targets in BATCH_FAMILY_TARGETS.values():
        needed.update(family_targets)
    for column in sorted(needed):
        if column in (PERSON_CHANNEL,):
            person_columns[column] = (
                reader.column("person", column).astype(str).to_numpy()
            )
        elif column == PERSON_CLONE:
            person_columns[column] = pd.to_numeric(
                reader.column("person", column), errors="raise"
            ).to_numpy(dtype=np.int64)
        elif column == "person_household_id":
            person_columns[column] = membership.to_numpy()
        elif reader.has_column("person", column):
            person_columns[column] = to_float(reader.column("person", column))
        else:
            person_columns[column] = None
    person_columns["__weights__"] = person_weights(reader)
    return person_columns


def optional(person: dict[str, object], column: str) -> np.ndarray:
    values = person.get(column)
    if values is None:
        return np.zeros(len(np.asarray(person["__weights__"])), dtype=np.float64)
    numeric = np.asarray(values, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"optional QBI input {column!r} contains nonfinite values")
    return numeric


def required_finite(person: dict[str, object], column: str) -> np.ndarray:
    values = person.get(column)
    if values is None:
        raise KeyError(f"required QBI input {column!r} is absent")
    numeric = np.asarray(values, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"required QBI input {column!r} contains nonfinite values")
    return numeric


def nine_invariants(person: dict[str, object]) -> dict[str, int]:
    """Mirror the nine coupled QBI invariants and structural source scope.

    QBI reconciliation applies every derived identity to the whole pool, but
    its base self-employment source is structurally absent for ACS people under
    15. Production zeroes that source outside the active universe before the
    first identity check (`qbi_inputs.py` `_qbi_reconciliation_scope` and
    `us_qbi_inputs_summary`). All other invariant inputs must be finite on all
    rows; active base-source values must also be finite.
    """

    atol = 1e-8
    channel = np.asarray(person[PERSON_CHANNEL], dtype=str)
    if not np.isin(channel, (ASEC, ACS)).all():
        raise ValueError("QBI invariant scope contains an unknown support channel")
    age = required_finite(person, "age")
    structural = (channel == ACS) & (age < 15.0)
    active = ~structural

    raw_self_employment = np.asarray(
        person["self_employment_income_before_lsr"], dtype=np.float64
    )
    if not np.isfinite(raw_self_employment[active]).all():
        raise ValueError(
            "base self-employment contains nonfinite values in its active universe"
        )
    self_employment = np.where(active, raw_self_employment, 0.0)

    business = required_finite(person, "business_is_sstb") > 0.0
    sstb_self_employment = required_finite(
        person, "sstb_self_employment_income_before_lsr"
    )
    w2 = required_finite(person, "w2_wages_from_qualified_business")
    sstb_w2 = required_finite(person, "sstb_w2_wages_from_qualified_business")
    ubia = required_finite(person, "unadjusted_basis_qualified_property")
    sstb_ubia = required_finite(person, "sstb_unadjusted_basis_qualified_property")
    self_qualified = (
        required_finite(person, "self_employment_income_would_be_qualified") > 0.0
    )
    sstb_qualified = (
        required_finite(person, "sstb_self_employment_income_would_be_qualified") > 0.0
    )
    non_qualified_dividends = np.maximum(
        optional(person, "non_qualified_dividend_income"), 0.0
    )
    partnership_s_corp = optional(person, "partnership_income") + optional(
        person, "s_corp_income"
    )
    bdc = required_finite(person, "qualified_bdc_income")
    reit = required_finite(person, "qualified_reit_and_ptp_income")
    return {
        "sstb_rows_with_non_sstb_income": int(
            np.count_nonzero(business & ~np.isclose(self_employment, 0.0))
        ),
        "non_sstb_rows_with_sstb_income": int(
            np.count_nonzero(~business & ~np.isclose(sstb_self_employment, 0.0))
        ),
        "sstb_w2_split_mismatches": int(
            np.count_nonzero(
                ~np.isclose(sstb_w2, np.where(business, w2, 0.0), atol=atol)
            )
        ),
        "sstb_ubia_split_mismatches": int(
            np.count_nonzero(
                ~np.isclose(sstb_ubia, np.where(business, ubia, 0.0), atol=atol)
            )
        ),
        "self_employment_qualification_overlap": int(
            np.count_nonzero(self_qualified & sstb_qualified)
        ),
        "sstb_qualification_route_mismatches": int(
            np.count_nonzero(sstb_qualified & ~business)
        ),
        "non_sstb_qualification_route_mismatches": int(
            np.count_nonzero(self_qualified & business)
        ),
        "qualified_bdc_exposure_mismatches": int(
            np.count_nonzero(bdc > non_qualified_dividends + atol)
        ),
        "qualified_reit_ptp_exposure_mismatches": int(
            np.count_nonzero(
                reit
                > non_qualified_dividends + np.maximum(partnership_s_corp, 0.0) + atol
            )
        ),
    }


def bank_pattern_evidence(
    checkpoint_dir: Path,
    *,
    group_receipts: dict[str, object],
) -> dict[str, object]:
    """Read and validate banked chain identities for both QBI batches."""

    acs_transfer = checkpoint_dir / "acs-transfer"
    bank_roots = [
        candidate for candidate in sorted(acs_transfer.iterdir()) if candidate.is_dir()
    ]
    if len(bank_roots) != 1:
        raise FileNotFoundError(
            f"Expected exactly one bank root under {acs_transfer}, got {bank_roots}"
        )
    bank_root = bank_roots[0]
    if tuple(group_receipts) != tuple(BATCH_FAMILY_TARGETS):
        raise AssertionError("publication QBI target-bank group membership changed")
    evidence: dict[str, object] = {
        "bank_root": str(bank_root),
        "receipt_bindings": {},
    }
    for batch, expected_targets in BATCH_FAMILY_TARGETS.items():
        target_dir = bank_root / "late_producer_dag" / "person" / batch / "targets"
        group_receipt = group_receipts[batch]
        if not isinstance(group_receipt, dict):
            raise TypeError(
                f"{batch}: publication bank-group receipt must be an object"
            )
        receipt_root = Path(str(group_receipt["root"])).resolve()
        if receipt_root != target_dir.parent.resolve():
            raise AssertionError(
                f"{batch}: bank root {target_dir.parent} differs from publication "
                f"receipt {receipt_root}"
            )
        receipt_targets = group_receipt["targets"]
        if not isinstance(receipt_targets, dict):
            raise TypeError(f"{batch}: publication bank targets must be an object")
        group_identity = group_receipt["identity"]
        group_identity_sha256 = group_receipt["identity_sha256"]
        evidence["receipt_bindings"][batch] = {
            "identity_sha256": group_identity_sha256,
            "root": str(receipt_root),
            "target_count": len(receipt_targets),
            "all_target_checkpoint_bytes_verified": True,
        }
        batch_record: dict[str, object] = {}
        for target_file in sorted(target_dir.glob("*.h5")):
            with h5py.File(target_file, "r") as fh:
                meta = json.loads(bytes(fh["metadata_json"][()]).decode("utf-8"))
            if not isinstance(meta, dict):
                raise TypeError(f"{target_file}: metadata_json must be an object")
            target_binding = meta.get("target")
            if not isinstance(target_binding, dict):
                raise TypeError(f"{target_file}: target binding must be an object")
            model_target = target_binding.get("model_target")
            if model_target not in expected_targets:
                raise ValueError(
                    f"{target_file}: unexpected model target {model_target!r}"
                )
            if model_target in batch_record:
                raise ValueError(f"{batch}: duplicate bank target {model_target!r}")
            if tuple(target_binding.get("model_targets", ())) != expected_targets:
                raise ValueError(
                    f"{batch}/{model_target}: banked model-target order changed"
                )
            target_index = target_binding.get("target_index")
            if (
                not isinstance(target_index, int)
                or isinstance(target_index, bool)
                or target_index < 0
            ):
                raise ValueError(f"{batch}/{model_target}: invalid target index")
            target_receipt = receipt_targets.get(str(target_index))
            if not isinstance(target_receipt, dict):
                raise AssertionError(
                    f"{batch}/{model_target}: no publication target-bank receipt"
                )
            computed_checkpoint_sha256 = sha256_of(target_file)
            expected_checkpoint_sha256 = target_receipt.get("checkpoint_sha256")
            if computed_checkpoint_sha256 != expected_checkpoint_sha256:
                raise AssertionError(
                    f"{batch}/{model_target}: target-bank checkpoint SHA changed"
                )
            if (
                Path(str(target_receipt.get("path"))).resolve() != target_file.resolve()
                or target_receipt.get("size_bytes") != target_file.stat().st_size
                or target_receipt.get("descriptor") != target_binding
                or target_receipt.get("content_metadata_sha256")
                != meta.get("content_metadata_sha256")
                or target_receipt.get("raw_draw_sha256") != meta.get("raw_draw_sha256")
                or meta.get("identity") != group_identity
                or meta.get("identity_sha256") != group_identity_sha256
            ):
                raise AssertionError(
                    f"{batch}/{model_target}: target-bank publication binding changed"
                )
            content = {
                key: value
                for key, value in meta.items()
                if key != "content_metadata_sha256"
            }
            if mapping_sha256(content) != meta.get("content_metadata_sha256"):
                raise AssertionError(
                    f"{batch}/{model_target}: target-bank content digest changed"
                )
            steps = []
            seen_patterns: set[str] = set()
            for step in meta["pattern_steps"]:
                pattern = step.get("pattern")
                if not isinstance(pattern, str) or not pattern:
                    raise ValueError(
                        f"{batch}/{model_target}: pattern name must be nonempty"
                    )
                if pattern in seen_patterns:
                    raise ValueError(
                        f"{batch}/{model_target}: duplicate pattern {pattern!r}"
                    )
                seen_patterns.add(pattern)
                state = step["state_after"]
                if not isinstance(state, dict):
                    raise TypeError(
                        f"{batch}/{model_target}/{pattern}: state_after must be an object"
                    )
                donor_identity = validated_index_identity(
                    state.get("donor_index"),
                    boundary=f"{batch}/{model_target}/{pattern}.donor_index",
                )
                recipient_identity = validated_index_identity(
                    state.get("recipient_index"),
                    boundary=f"{batch}/{model_target}/{pattern}.recipient_index",
                )
                model_config = state.get("model_config")
                if not isinstance(model_config, dict):
                    raise TypeError(
                        f"{batch}/{model_target}/{pattern}: model_config must be an object"
                    )
                zero_atol = model_config.get("zero_atol")
                if (
                    not isinstance(zero_atol, (int, float))
                    or isinstance(zero_atol, bool)
                    or not math.isfinite(float(zero_atol))
                    or float(zero_atol) < 0.0
                ):
                    raise ValueError(
                        f"{batch}/{model_target}/{pattern}: invalid zero_atol"
                    )
                persisted_regime = step.get("regime")
                if persisted_regime is not None and not isinstance(
                    persisted_regime, str
                ):
                    raise TypeError(
                        f"{batch}/{model_target}/{pattern}: regime must be a string"
                    )
                steps.append(
                    {
                        "pattern": pattern,
                        "predictors": list(state["predictors"]),
                        "donor_index": donor_identity,
                        "recipient_index": recipient_identity,
                        "weight_kind": state["weight_kind"],
                        "weight_sha256": state["weight_sha256"],
                        "model_config": dict(model_config),
                        "persisted_regime": persisted_regime,
                        "regime_persisted": persisted_regime is not None,
                    }
                )
            batch_record[str(model_target)] = {
                "artifact_path": str(target_file),
                "schema_version": meta.get("schema_version"),
                "materializer_version": meta.get("materializer_version"),
                "identity_sha256": meta.get("identity_sha256"),
                "content_metadata_sha256": meta.get("content_metadata_sha256"),
                "raw_draw_sha256": meta.get("raw_draw_sha256"),
                "checkpoint_sha256": computed_checkpoint_sha256,
                "publication_checkpoint_sha256": expected_checkpoint_sha256,
                "checkpoint_bytes_verified": True,
                "target_binding": dict(target_binding),
                "recipient_rows_total": int(meta["recipient_rows"]),
                "pattern_steps": steps,
            }
        if tuple(batch_record) != expected_targets:
            raise ValueError(
                f"{batch}: bank target order/membership changed: "
                f"got {list(batch_record)}, expected {list(expected_targets)}"
            )
        evidence[batch] = batch_record
    return evidence


def artifact_bindings(
    checkpoint_dir: Path,
    *,
    skip_sha: bool,
) -> dict[str, object]:
    """Validate checkpoint declarations, bytes, and failed publication pins."""

    pool_roots = [
        parent
        for parent in checkpoint_dir.parents
        if (parent / f"{POOL_BASENAME}.manifest.json").is_file()
    ]
    if not pool_roots:
        raise FileNotFoundError(
            f"Could not locate {POOL_BASENAME}.manifest.json above {checkpoint_dir}"
        )
    pool_root = pool_roots[0]
    stages: dict[str, object] = {}
    for stage, expected in CHECKPOINT_BINDINGS.items():
        checkpoint_path = checkpoint_dir / f"{stage}.checkpoint.h5"
        manifest_path = checkpoint_dir / f"{stage}.checkpoint.manifest.json"
        manifest = read_json(manifest_path)
        declared_checkpoint = manifest.get("checkpoint")
        if not isinstance(declared_checkpoint, dict):
            raise TypeError(f"{manifest_path}: checkpoint binding must be an object")
        observed = {
            "stage": manifest.get("stage"),
            "identity_sha256": manifest.get("identity_sha256"),
            "checkpoint_filename": declared_checkpoint.get("filename"),
            "checkpoint_sha256": declared_checkpoint.get("sha256"),
        }
        required = {
            "stage": stage,
            "identity_sha256": expected["identity_sha256"],
            "checkpoint_filename": checkpoint_path.name,
            "checkpoint_sha256": expected["sha256"],
        }
        if observed != required:
            raise AssertionError(
                f"{stage} checkpoint manifest binding changed: "
                f"observed={observed}, required={required}"
            )
        computed_sha256 = None if skip_sha else sha256_of(checkpoint_path)
        if computed_sha256 is not None and computed_sha256 != expected["sha256"]:
            raise AssertionError(
                f"{stage} checkpoint byte SHA-256 changed: {computed_sha256}"
            )
        frame_metadata = manifest.get("frame_metadata")
        late_transition = (
            frame_metadata.get("us_late_producer_transition_authority")
            if isinstance(frame_metadata, dict)
            else None
        )
        stage_record = {
            "manifest_path": str(manifest_path),
            "checkpoint_path": str(checkpoint_path),
            "manifest_checkpoint": dict(declared_checkpoint),
            "identity_sha256": manifest["identity_sha256"],
            "computed_sha256": computed_sha256,
            "byte_sha_verification": "skipped_debug" if skip_sha else "verified",
            "materializer_version": manifest.get("materializer_version"),
            "late_producer_transition_authority_sha256": manifest.get(
                "late_producer_transition_authority_sha256"
            ),
            "qbi_transition_authority_sha256": manifest.get(
                "qbi_transition_authority_sha256"
            ),
            "late_producer_operational_receipt_sha256": (
                late_transition.get("receipt_sha256")
                if isinstance(late_transition, dict)
                else None
            ),
        }
        stages[stage] = stage_record

    simulated = stages["simulated"]
    if simulated["qbi_transition_authority_sha256"] != QBI_TRANSITION_AUTHORITY_SHA256:
        raise AssertionError(
            "simulated checkpoint QBI transition authority does not match "
            "the adjudication pin"
        )
    transferred_stage = stages["transferred"]
    if (
        transferred_stage["late_producer_operational_receipt_sha256"]
        != LATE_PRODUCER_OPERATIONAL_RECEIPT_SHA256
    ):
        raise AssertionError("transferred late-producer operational receipt changed")

    publication_path = pool_root / f"{POOL_BASENAME}.manifest.json"
    publication_sha256 = sha256_of(publication_path)
    if publication_sha256 != PUBLICATION_MANIFEST_SHA256:
        raise AssertionError("failed publication manifest SHA-256 changed")
    publication = read_json(publication_path)
    publication_observed = {
        "release_id": publication.get("release_id"),
        "publication_run_id": publication.get("publication_run_id"),
        "status": publication.get("status"),
        "simulation_ready": publication.get("simulation_ready"),
    }
    publication_required = {
        "release_id": PUBLICATION_RELEASE_ID,
        "publication_run_id": PUBLICATION_RUN_ID,
        "status": PUBLICATION_STATUS,
        "simulation_ready": PUBLICATION_SIMULATION_READY,
    }
    if publication_observed != publication_required:
        raise AssertionError(
            "failed publication binding changed: "
            f"observed={publication_observed}, required={publication_required}"
        )

    try:
        impute_receipts = publication["stage_receipts"]["impute"]
        late_groups = impute_receipts["acs_qrf_transfer"]["target_bank"][
            "late_producer_groups"
        ]
        post_puf = impute_receipts["stacked_post_puf_transfer"]
    except (KeyError, TypeError) as error:
        raise AssertionError(
            "publication manifest is missing QBI transfer receipts"
        ) from error
    if not isinstance(late_groups, dict) or not isinstance(post_puf, dict):
        raise TypeError("publication QBI transfer receipts must be objects")

    qbi_bank_groups: dict[str, object] = {}
    for batch, expected_targets in BATCH_FAMILY_TARGETS.items():
        receipt_key = f"transfer:person/{batch}"
        group = late_groups.get(receipt_key)
        if not isinstance(group, dict):
            raise AssertionError(
                f"publication has no target-bank receipt {receipt_key}"
            )
        identity = group.get("identity")
        identity_digest = group.get("identity_sha256")
        if (
            not isinstance(identity, dict)
            or not isinstance(identity_digest, str)
            or mapping_sha256(identity) != identity_digest
        ):
            raise AssertionError(f"{receipt_key}: target-bank identity digest changed")
        targets = group.get("targets")
        if not isinstance(targets, dict) or tuple(targets) != tuple(
            str(index) for index in range(len(expected_targets))
        ):
            raise AssertionError(f"{receipt_key}: target-bank receipt order changed")
        qbi_bank_groups[batch] = {
            "artifact_kind": group.get("artifact_kind"),
            "schema_version": group.get("schema_version"),
            "materializer_version": group.get("materializer_version"),
            "identity": identity,
            "identity_sha256": identity_digest,
            "root": group.get("root"),
            "targets": targets,
        }

    post_targets = post_puf.get("targets")
    if not isinstance(post_targets, dict):
        raise TypeError("stacked post-PUF transfer targets receipt must be an object")
    qbi_post_targets: dict[str, object] = {}
    for target in AMOUNT_TARGETS:
        receipt_key = f"person/puf_tax_itemization/{target}"
        target_receipt = post_targets.get(receipt_key)
        if not isinstance(target_receipt, dict):
            raise AssertionError(f"publication has no post-PUF receipt {receipt_key}")
        qbi_post_targets[target] = dict(target_receipt)
    qbi_post_puf_receipt = {
        "donor_selection": post_puf.get("donor_selection"),
        "donor_channel": post_puf.get("donor_channel"),
        "donor_clone_index": post_puf.get("donor_clone_index"),
        "recipient_selection": post_puf.get("recipient_selection"),
        "resolved_donor_channel": post_puf.get("resolved_donor_channel"),
        "targets": qbi_post_targets,
    }
    expected_post_puf_contract = {
        "donor_selection": "owner_projection_of_asec_origin_clone_1",
        "donor_channel": ASEC,
        "donor_clone_index": PUF_TAX_DETAIL_CLONE_INDEX,
        "recipient_selection": ("target_specific_complement_of_declared_producer_rows"),
        "resolved_donor_channel": None,
    }
    if {
        key: qbi_post_puf_receipt[key] for key in expected_post_puf_contract
    } != expected_post_puf_contract:
        raise AssertionError("stacked post-PUF transfer contract receipt changed")
    expected_target_completion = {
        "authorized_null_rows": 964699,
        "imputed_rows": 964699,
        "producer_roles": ["puf_clone"],
        "producer_rows": 1006274,
        "residual_null_rows": 0,
        "unmodeled_rows": 0,
    }
    for target, target_receipt in qbi_post_targets.items():
        if target_receipt != expected_target_completion:
            raise AssertionError(
                f"stacked post-PUF QBI target completion changed for {target}"
            )

    gates_path = pool_root / f"{POOL_BASENAME}.gates.json"
    gates_sha256 = sha256_of(gates_path)
    if gates_sha256 != PUBLICATION_GATES_SHA256:
        raise AssertionError("publication gates SHA-256 changed")
    gates = read_json(gates_path)
    gates_observed = {
        "release_id": gates.get("release_id"),
        "publication_run_id": gates.get("publication_run_id"),
        "simulation_ready": gates.get("simulation_ready"),
    }
    gates_required = {
        "release_id": PUBLICATION_RELEASE_ID,
        "publication_run_id": PUBLICATION_RUN_ID,
        "simulation_ready": PUBLICATION_SIMULATION_READY,
    }
    if gates_observed != gates_required:
        raise AssertionError(
            "publication-gates binding changed: "
            f"observed={gates_observed}, required={gates_required}"
        )
    return {
        "sha_verification": "skipped_debug" if skip_sha else "verified",
        "stages": stages,
        "publication_manifest": {
            "path": str(publication_path),
            "sha256": publication_sha256,
            **publication_observed,
        },
        "publication_gates": {
            "path": str(gates_path),
            "sha256": gates_sha256,
            **gates_observed,
        },
        "qbi_target_bank_groups": qbi_bank_groups,
        "qbi_post_puf_transfer": qbi_post_puf_receipt,
    }


def validate_evidence(
    evidence: dict[str, object],
    *,
    skip_sha: bool,
) -> dict[str, object]:
    """Validate the complete closed evidence schema before atomic output."""

    if set(evidence) != TOP_LEVEL_EVIDENCE_KEYS:
        raise AssertionError(
            "QBI evidence top-level schema changed: "
            f"got {sorted(evidence)}, expected {sorted(TOP_LEVEL_EVIDENCE_KEYS)}"
        )

    ownership = evidence["ownership_checks"]
    if not isinstance(ownership, dict) or tuple(ownership) != tuple(ADJUDICATED_CHECKS):
        raise AssertionError(
            "QBI ownership checks must contain the exact eight adjudication IDs "
            "in source order"
        )
    for check_id, record in ownership.items():
        if not isinstance(record, dict):
            raise TypeError(f"{check_id}: ownership record must be an object")
        if record.get("full_precision_value_agrees") is not True:
            raise AssertionError(f"{check_id}: full-precision value did not agree")
        if record.get("terminal_value_origin_channel") not in TERMINAL_ORIGIN_CHANNELS:
            raise AssertionError(f"{check_id}: terminal value origin is unrecorded")
        if record.get("criterion_first_failing_stage") not in FAILING_STAGES:
            raise AssertionError(f"{check_id}: first failing stage is unrecorded")
        contributing = record.get("contributing_stages")
        if (
            not isinstance(contributing, list)
            or not contributing
            or any(stage not in FAILING_STAGES for stage in contributing)
        ):
            raise AssertionError(f"{check_id}: contributing stages are unrecorded")
        origin_evidence = record.get("terminal_value_origin_evidence")
        if (
            not isinstance(origin_evidence, dict)
            or origin_evidence.get("bank_target_binding_valid") is not True
            or origin_evidence.get("post_puf_target_receipt_valid") is not True
            or origin_evidence.get("attribution_kind") != "code_plus_receipt_inference"
            or origin_evidence.get("explicit_origin_channel_field_persisted")
            is not False
        ):
            raise AssertionError(f"{check_id}: terminal origin has no bank binding")

    bank_patterns = evidence["bank_patterns"]
    if not isinstance(bank_patterns, dict):
        raise TypeError("bank_patterns must be an object")
    bank_receipts = bank_patterns.get("receipt_bindings")
    if not isinstance(bank_receipts, dict) or tuple(bank_receipts) != tuple(
        BATCH_FAMILY_TARGETS
    ):
        raise AssertionError("QBI target-bank receipt bindings changed")
    bank_target_file_count = 0
    for batch, expected_targets in BATCH_FAMILY_TARGETS.items():
        receipt = bank_receipts[batch]
        if receipt.get("all_target_checkpoint_bytes_verified") is not True:
            raise AssertionError(f"{batch}: target-bank bytes are unverified")
        target_records = bank_patterns.get(batch)
        if not isinstance(target_records, dict) or tuple(target_records) != (
            expected_targets
        ):
            raise AssertionError(f"{batch}: banked target membership changed")
        for target, target_record in target_records.items():
            if target_record.get(
                "checkpoint_bytes_verified"
            ) is not True or target_record.get(
                "checkpoint_sha256"
            ) != target_record.get("publication_checkpoint_sha256"):
                raise AssertionError(
                    f"{batch}/{target}: banked checkpoint bytes are unverified"
                )
            bank_target_file_count += 1

    regimes = evidence["realized_regimes"]
    if not isinstance(regimes, dict) or tuple(regimes) != tuple(BATCH_FAMILY_TARGETS):
        raise AssertionError("QBI regime batches changed")
    regime_cell_count = 0
    for batch, expected_targets in BATCH_FAMILY_TARGETS.items():
        batch_record = regimes[batch]
        pattern_catalog = batch_record.get("availability_patterns")
        if not isinstance(pattern_catalog, list) or not pattern_catalog:
            raise AssertionError(f"{batch}: availability-pattern catalog is empty")
        pattern_order = tuple(record.get("pattern") for record in pattern_catalog)
        if any(
            not isinstance(pattern, str) or not pattern for pattern in pattern_order
        ):
            raise AssertionError(f"{batch}: availability-pattern names are invalid")
        realized = batch_record.get("realized_regimes_by_target_and_pattern")
        if not isinstance(realized, dict) or tuple(realized) != expected_targets:
            raise AssertionError(f"{batch}: realized-regime targets changed")
        for target, target_patterns in realized.items():
            if not isinstance(target_patterns, dict) or tuple(target_patterns) != (
                pattern_order
            ):
                raise AssertionError(
                    f"{batch}/{target}: target-pattern regime coverage changed"
                )
            for pattern, cell in target_patterns.items():
                if (
                    not isinstance(cell, dict)
                    or cell.get("bank_identity_match") is not True
                    or cell.get("realized_regime") not in QRF_REGIMES
                ):
                    raise AssertionError(
                        f"{batch}/{target}/{pattern}: regime cell is unverified"
                    )
                regime_cell_count += 1

    invariants = evidence["invariants"]
    if not isinstance(invariants, dict) or set(invariants) != {
        "transferred",
        "simulated",
    }:
        raise AssertionError("QBI invariant stage coverage changed")
    for stage, counts in invariants.items():
        if not isinstance(counts, dict) or tuple(counts) != QBI_INVARIANT_NAMES:
            raise AssertionError(f"{stage}: QBI invariant names/order changed")
    if any(invariants["simulated"].values()):
        raise AssertionError("terminal QBI invariants are not all zero")

    bindings = evidence["artifact_bindings"]
    if not isinstance(bindings, dict):
        raise TypeError("artifact_bindings must be an object")
    expected_sha_status = "skipped_debug" if skip_sha else "verified"
    if bindings.get("sha_verification") != expected_sha_status:
        raise AssertionError("artifact SHA verification status is unrecorded")
    stages = bindings.get("stages")
    if not isinstance(stages, dict) or tuple(stages) != tuple(CHECKPOINT_BINDINGS):
        raise AssertionError("artifact checkpoint-stage bindings changed")
    for stage, expected in CHECKPOINT_BINDINGS.items():
        record = stages[stage]
        if (
            record.get("identity_sha256") != expected["identity_sha256"]
            or record["manifest_checkpoint"].get("sha256") != expected["sha256"]
            or record.get("byte_sha_verification") != expected_sha_status
        ):
            raise AssertionError(f"{stage}: artifact pin validation is incomplete")
        if not skip_sha and record.get("computed_sha256") != expected["sha256"]:
            raise AssertionError(f"{stage}: checkpoint bytes were not SHA-verified")
    publication = bindings.get("publication_manifest")
    if not isinstance(publication, dict) or {
        key: publication.get(key)
        for key in ("release_id", "publication_run_id", "status", "simulation_ready")
    } != {
        "release_id": PUBLICATION_RELEASE_ID,
        "publication_run_id": PUBLICATION_RUN_ID,
        "status": PUBLICATION_STATUS,
        "simulation_ready": PUBLICATION_SIMULATION_READY,
    }:
        raise AssertionError("failed-publication artifact pin validation is incomplete")
    if publication.get("sha256") != PUBLICATION_MANIFEST_SHA256:
        raise AssertionError("publication manifest bytes are unverified")
    gates = bindings.get("publication_gates")
    if not isinstance(gates, dict) or gates.get("sha256") != PUBLICATION_GATES_SHA256:
        raise AssertionError("publication gates bytes are unverified")
    qbi_post_puf = bindings.get("qbi_post_puf_transfer")
    if not isinstance(qbi_post_puf, dict) or set(
        qbi_post_puf.get("targets", {})
    ) != set(AMOUNT_TARGETS):
        raise AssertionError("QBI post-PUF transfer receipts are incomplete")

    return {
        "schema_version": 1,
        "status": "passed",
        "top_level_schema_before_validation": sorted(TOP_LEVEL_EVIDENCE_KEYS),
        "ordered_check_ids": list(ownership),
        "ownership_check_count": len(ownership),
        "full_precision_agreements": len(ownership),
        "origin_attributions_recorded": len(ownership),
        "first_failing_stages_recorded": len(ownership),
        "target_pattern_regime_cells_verified": regime_cell_count,
        "bank_target_checkpoint_files_sha_verified": bank_target_file_count,
        "post_puf_target_receipts_verified": len(AMOUNT_TARGETS),
        "terminal_invariant_names": list(QBI_INVARIANT_NAMES),
        "terminal_invariant_nonzero_count": 0,
        "artifact_pin_status": expected_sha_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=CANONICAL_OUTPUT,
    )
    parser.add_argument(
        "--skip-sha",
        action="store_true",
        help="Skip streaming SHA-256 verification (iteration only).",
    )
    args = parser.parse_args()
    checkpoint_dir: Path = args.checkpoint_dir.resolve()
    output: Path = args.output.resolve()
    if args.skip_sha and output == CANONICAL_OUTPUT:
        parser.error(
            "--skip-sha is debug-only and cannot overwrite the canonical evidence.json; "
            "pass an explicit noncanonical --output"
        )

    evidence: dict[str, object] = {
        "schema_version": 2,
        "checkpoint_dir": str(checkpoint_dir),
        "adjudication_pin_simulated_sha256": SIMULATED_SHA256,
        "artifact_bindings": artifact_bindings(
            checkpoint_dir,
            skip_sha=args.skip_sha,
        ),
    }

    print("== loading transferred stage columns ==", flush=True)
    transferred_reader = CheckpointColumns(checkpoint_dir / "transferred.checkpoint.h5")
    transferred = load_stage(transferred_reader)
    transferred_reader.close()
    print("== loading simulated stage columns ==", flush=True)
    simulated_reader = CheckpointColumns(checkpoint_dir / "simulated.checkpoint.h5")
    simulated = load_stage(simulated_reader)
    simulated_reader.close()

    # Row alignment: the two stages must describe the same physical rows.
    for column in (
        "__person_index__",
        PERSON_CHANNEL,
        PERSON_CLONE,
        "person_household_id",
    ):
        if not np.array_equal(transferred[column], simulated[column]):
            raise AssertionError(f"stage row alignment failed on {column}")
    if not np.array_equal(transferred["__weights__"], simulated["__weights__"]):
        raise AssertionError("stage weights differ between transferred and simulated")

    channel = np.asarray(simulated[PERSON_CHANNEL], dtype=str)
    clone = np.asarray(simulated[PERSON_CLONE], dtype=np.int64)
    weights = np.asarray(simulated["__weights__"], dtype=np.float64)
    if not np.isin(channel, (ASEC, ACS)).all():
        raise AssertionError("checkpoint contains unknown person support channels")
    scope_clone0 = (clone == 0) & (weights > 0.0)
    scope_clone1 = (clone == PUF_TAX_DETAIL_CLONE_INDEX) & (weights > 0.0)
    evidence["row_counts"] = {
        "person_rows": int(len(weights)),
        "clone0_rows": int(scope_clone0.sum()),
        "clone1_rows": int(scope_clone1.sum()),
        "clone_indices": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(clone, return_counts=True), strict=True)
        },
    }

    # The failed attempt predates origin/regime receipts, so its banked QRF
    # target artifacts are the artifact-level origin evidence available here.
    # The code-level producer-complement contract is identified explicitly in
    # each check below; it is not mislabeled as a persisted origin receipt.
    artifact_receipts = evidence["artifact_bindings"]
    if not isinstance(artifact_receipts, dict):
        raise TypeError("artifact bindings must be an object")
    group_receipts = artifact_receipts.get("qbi_target_bank_groups")
    if not isinstance(group_receipts, dict):
        raise TypeError("QBI target-bank group receipts must be an object")
    post_puf_receipt = artifact_receipts.get("qbi_post_puf_transfer")
    if not isinstance(post_puf_receipt, dict) or not isinstance(
        post_puf_receipt.get("targets"), dict
    ):
        raise TypeError("QBI post-PUF target receipts must be an object")
    bank = bank_pattern_evidence(
        checkpoint_dir,
        group_receipts=group_receipts,
    )
    evidence["bank_patterns"] = bank

    # Battery replication + stage decomposition for the four amount legs.
    legs: dict[str, object] = {}
    mismatches: list[str] = []
    for target in AMOUNT_TARGETS:
        simulated_values = np.asarray(simulated[target], dtype=np.float64)
        transferred_values = np.asarray(transferred[target], dtype=np.float64)
        record: dict[str, object] = {}
        record["simulated_clone0"] = positive_leg_stats(
            simulated_values, scope_clone0, channel, weights
        )
        record["transferred_clone0"] = positive_leg_stats(
            transferred_values, scope_clone0, channel, weights
        )
        record["simulated_clone1"] = positive_leg_stats(
            simulated_values, scope_clone1, channel, weights
        )
        record["transferred_clone1"] = positive_leg_stats(
            transferred_values, scope_clone1, channel, weights
        )

        # Reconciliation row deltas per channel within each clone scope.
        deltas: dict[str, object] = {}
        for scope_name, scope in (("clone0", scope_clone0), ("clone1", scope_clone1)):
            for origin in (ASEC, ACS):
                rows = scope & (channel == origin)
                before = transferred_values[rows]
                after = simulated_values[rows]
                changed = ~np.isclose(before, after, rtol=0.0, atol=1e-9)
                deltas[f"{scope_name}_{origin}"] = {
                    "rows": int(rows.sum()),
                    "changed_rows": int(changed.sum()),
                    "positive_to_zero_rows": int(
                        ((before > 0.0) & (after == 0.0)).sum()
                    ),
                    "zero_to_positive_rows": int(
                        ((before == 0.0) & (after > 0.0)).sum()
                    ),
                    "sum_before": float(before.sum()),
                    "sum_after": float(after.sum()),
                }
        record["reconciliation_deltas"] = deltas

        # Exposure-cap attribution for the two capped targets
        # (qbi_inputs.py:1342-1359): count carriers the cap must kill or trim.
        if target == "qualified_bdc_income":
            cap_base = np.maximum(
                optional(transferred, "non_qualified_dividend_income"), 0.0
            )
        elif target == "qualified_reit_and_ptp_income":
            cap_base = np.maximum(
                optional(transferred, "non_qualified_dividend_income"), 0.0
            ) + np.maximum(
                optional(transferred, "partnership_income")
                + optional(transferred, "s_corp_income"),
                0.0,
            )
        else:
            cap_base = None
        if cap_base is not None:
            cap_record: dict[str, object] = {}
            for scope_name, scope in (
                ("clone0", scope_clone0),
                ("clone1", scope_clone1),
            ):
                for origin in (ASEC, ACS):
                    rows = scope & (channel == origin)
                    before = transferred_values[rows]
                    base = cap_base[rows]
                    cap_record[f"{scope_name}_{origin}"] = {
                        "carriers_before": int((before > 0.0).sum()),
                        "carriers_with_zero_cap_base": int(
                            ((before > 0.0) & (base <= 0.0)).sum()
                        ),
                        "carriers_trimmed_not_killed": int(
                            ((before > 0.0) & (base > 0.0) & (before > base)).sum()
                        ),
                    }
            record["exposure_cap_attribution"] = cap_record

        # Adjudication agreement on the simulated clone-0 battery numbers.
        stats = record["simulated_clone0"]
        expected = ADJUDICATED_LEGS[target]
        observed = {
            "ratio_acs_over_asec": stats["incidence_ratio_acs_over_asec"],
            "asec_incidence": stats["asec_incidence"],
            "acs_incidence": stats["acs_incidence"],
            "quantile_envelope_distance": stats["quantile_envelope_distance"],
            "carriers_asec": stats["carriers_asec"],
            "carriers_acs": stats["carriers_acs"],
        }
        agreement = {
            key: exact_numeric_agreement(observed[key], expected[key])
            for key in expected
        }
        record["adjudication_agreement"] = {
            "expected": expected,
            "observed": observed,
            "agreed": agreement,
        }
        if not all(agreement.values()):
            mismatches.append(f"{target}: {agreement} observed={observed}")
        legs[target] = record
    evidence["amount_legs"] = legs

    # Exact check-level ownership attribution. Terminal provenance and first
    # failure are intentionally separate: a qrf_transfer-owned clone-0 cell can
    # carry a distributional mismatch already visible on the clone-1 producer.
    ownership_checks: dict[str, object] = {}
    for check_id, adjudicated in ADJUDICATED_CHECKS.items():
        target = str(adjudicated["target"])
        criterion = str(adjudicated["criterion"])
        leg = legs[target]
        producer_stats = leg["transferred_clone1"]
        transfer_stats = leg["transferred_clone0"]
        terminal_stats = leg["simulated_clone0"]
        if criterion_failed(producer_stats, criterion):
            first_failing_stage = "clone1_puf_producer"
        elif criterion_failed(transfer_stats, criterion):
            first_failing_stage = "late_cross_role_transfer"
        elif criterion_failed(terminal_stats, criterion):
            first_failing_stage = "qbi_reconciliation"
        else:
            first_failing_stage = None

        expected_stage = adjudicated["first_failing_stage"]
        if first_failing_stage != expected_stage:
            mismatches.append(
                f"{check_id}: first failing stage {first_failing_stage!r}, "
                f"expected {expected_stage!r}"
            )
        observed_terminal = criterion_value(terminal_stats, criterion)
        adjudicated_value = float(adjudicated["criterion_value"])
        value_agrees = exact_numeric_agreement(observed_terminal, adjudicated_value)
        if not value_agrees:
            mismatches.append(
                f"{check_id}: terminal value {observed_terminal!r}, "
                f"expected {adjudicated_value!r}"
            )
        producer_first = first_failing_stage == "clone1_puf_producer"
        contributing_stages = (
            ["clone1_puf_producer", "late_cross_role_transfer"]
            if producer_first
            else ["late_cross_role_transfer"]
        )
        reconciliation_changed_value = not exact_numeric_agreement(
            criterion_value(transfer_stats, criterion),
            criterion_value(terminal_stats, criterion),
        )
        secondary_effect_stages = (
            ["qbi_reconciliation"] if reconciliation_changed_value else []
        )
        target_batches = [
            batch
            for batch, targets in BATCH_FAMILY_TARGETS.items()
            if target in targets
        ]
        if len(target_batches) != 1:
            raise AssertionError(
                f"{check_id}: target must bind to exactly one QRF batch, got "
                f"{target_batches}"
            )
        target_batch = target_batches[0]
        bank_target = bank[target_batch][target]
        target_binding = bank_target["target_binding"]
        bank_binds_target = (
            target_binding.get("model_target") == target
            and target in target_binding.get("exported_targets", ())
            and bool(bank_target["pattern_steps"])
            and bank_target.get("checkpoint_bytes_verified") is True
        )
        post_target_receipt = post_puf_receipt["targets"].get(target)
        post_receipt_binds_target = isinstance(post_target_receipt, dict) and (
            post_target_receipt.get("producer_roles") == ["puf_clone"]
            and post_target_receipt.get("authorized_null_rows") == 964699
            and post_target_receipt.get("imputed_rows") == 964699
            and post_target_receipt.get("residual_null_rows") == 0
            and post_target_receipt.get("unmodeled_rows") == 0
        )
        terminal_value_origin_channel = (
            "qrf_transfer" if bank_binds_target and post_receipt_binds_target else None
        )
        if terminal_value_origin_channel is None:
            mismatches.append(f"{check_id}: terminal origin channel is unrecorded")
        attribution = {
            "check_id": check_id,
            "target": target,
            "criterion": criterion,
            "adjudicated_value": adjudicated_value,
            "observed_terminal_value": observed_terminal,
            "full_precision_value_agrees": value_agrees,
            "terminal_value_origin_channel": terminal_value_origin_channel,
            "terminal_value_origin_evidence": {
                "evidence_kind": (
                    "banked_qrf_bytes_plus_post_puf_target_receipt_plus_code_contract"
                ),
                "attribution_kind": "code_plus_receipt_inference",
                "bank_batch": target_batch,
                "bank_artifact_path": bank_target["artifact_path"],
                "bank_checkpoint_sha256": bank_target["checkpoint_sha256"],
                "bank_model_target": target_binding.get("model_target"),
                "bank_exported_targets": target_binding.get("exported_targets"),
                "bank_availability_pattern_count": len(bank_target["pattern_steps"]),
                "bank_target_binding_valid": bank_binds_target,
                "post_puf_target_receipt": post_target_receipt,
                "post_puf_target_receipt_valid": post_receipt_binds_target,
                "failed_attempt_post_puf_target_receipt_persisted": True,
                "explicit_origin_channel_field_persisted": False,
                "code_inference": (
                    "post-PUF transfer fills the target-specific complement of "
                    "producer-owned clone>0 rows"
                ),
            },
            "criterion_first_failing_stage": first_failing_stage,
            "contributing_stages": contributing_stages,
            "secondary_effect_stages": secondary_effect_stages,
            "value_path_stages": [
                *contributing_stages,
                *secondary_effect_stages,
            ],
            "ownership_class": (
                "producer_first_coupled" if producer_first else "transfer_first"
            ),
            "stage_values": {
                "clone1_puf_producer": criterion_value(producer_stats, criterion),
                "late_cross_role_transfer": criterion_value(transfer_stats, criterion),
                "qbi_reconciliation_terminal": criterion_value(
                    terminal_stats, criterion
                ),
            },
            "stage_failures": {
                "clone1_puf_producer": criterion_failed(producer_stats, criterion),
                "late_cross_role_transfer": criterion_failed(transfer_stats, criterion),
                "qbi_reconciliation_terminal": criterion_failed(
                    terminal_stats, criterion
                ),
            },
        }
        if attribution["terminal_value_origin_channel"] not in TERMINAL_ORIGIN_CHANNELS:
            raise AssertionError(f"{check_id}: unrecognized terminal origin channel")
        if attribution["criterion_first_failing_stage"] not in FAILING_STAGES:
            raise AssertionError(f"{check_id}: first failing stage is unrecorded")
        if not attribution["contributing_stages"] or any(
            stage not in FAILING_STAGES for stage in attribution["contributing_stages"]
        ):
            raise AssertionError(f"{check_id}: contributing stages are unrecorded")
        ownership_checks[check_id] = attribution

    transfer_first = sum(
        record["ownership_class"] == "transfer_first"
        for record in ownership_checks.values()
    )
    producer_first = sum(
        record["ownership_class"] == "producer_first_coupled"
        for record in ownership_checks.values()
    )
    if (transfer_first, producer_first) != (5, 3):
        mismatches.append(
            "ownership split changed: "
            f"transfer_first={transfer_first}, producer_first={producer_first}"
        )
    evidence["ownership_checks"] = ownership_checks
    evidence["ownership_summary"] = {
        "check_count": len(ownership_checks),
        "transfer_first_checks": transfer_first,
        "producer_first_coupled_checks": producer_first,
        "terminal_value_origin_channel": "qrf_transfer",
    }

    # SSTB boolean evidence: clone-0 dead + clone-1 live, both stages.
    booleans: dict[str, object] = {}
    for target in SSTB_BOOLEANS:
        simulated_values = np.asarray(simulated[target], dtype=np.float64)
        transferred_values = np.asarray(transferred[target], dtype=np.float64)
        record = {
            "simulated_clone0": boolean_stats(
                simulated_values, scope_clone0, channel, weights
            ),
            "simulated_clone1": boolean_stats(
                simulated_values, scope_clone1, channel, weights
            ),
            "transferred_clone0": boolean_stats(
                transferred_values, scope_clone0, channel, weights
            ),
            "transferred_clone1": boolean_stats(
                transferred_values, scope_clone1, channel, weights
            ),
        }
        stats = record["simulated_clone1"]
        expected = ADJUDICATED_SSTB_CLONE1[target]
        observed = {
            "asec_incidence": fmt(stats["asec_incidence"]),
            "acs_incidence": fmt(stats["acs_incidence"]),
            "ratio": fmt(stats["incidence_ratio_acs_over_asec"]),
            "carriers_asec": stats["carriers_asec"],
            "carriers_acs": stats["carriers_acs"],
        }
        agreement = {key: observed[key] == expected[key] for key in expected}
        record["adjudication_agreement"] = {
            "expected": expected,
            "observed": observed,
            "agreed": agreement,
        }
        if not all(agreement.values()):
            mismatches.append(f"{target}: {agreement} observed={observed}")
        if record["simulated_clone0"].get("status") != "dead_comparison":
            mismatches.append(f"{target}: simulated clone-0 comparison is not dead")
        booleans[target] = record
    evidence["sstb_booleans"] = booleans

    # Recipient-context profiles: the transfer's predictor sources and the
    # reconciliation cap bases per channel, contrasting donors with the two
    # clone-0 recipient populations (transferred-stage values).
    context: dict[str, object] = {}
    donor_rows = (channel == ASEC) & (clone == PUF_TAX_DETAIL_CLONE_INDEX)
    context_cells = {
        "donor_asec_clone1": donor_rows,
        "recipient_asec_clone0": scope_clone0 & (channel == ASEC),
        "recipient_acs_clone0": scope_clone0 & (channel == ACS),
    }
    for column in CONTEXT_COLUMNS:
        values = transferred.get(column)
        if values is None:
            continue
        numeric_values = np.asarray(values, dtype=np.float64)
        if np.isinf(numeric_values).any():
            raise ValueError(f"recipient context {column!r} contains infinity")
        column_record: dict[str, object] = {}
        for cell_name, rows in context_cells.items():
            raw_cell_values = numeric_values[rows]
            cell_values = np.nan_to_num(raw_cell_values, nan=0.0)
            cell_weights = weights[rows]
            positive = cell_values > 0.0
            column_record[cell_name] = {
                "rows": int(rows.sum()),
                "missing_rows": int(np.isnan(raw_cell_values).sum()),
                "weighted_positive_share": weighted_incidence(positive, cell_weights),
                "positive_rows": int(positive.sum()),
                "positive_p50": (
                    float(
                        battery_quantiles(
                            cell_values[positive], cell_weights[positive]
                        )[2]
                    )
                    if int(positive.sum())
                    else 0.0
                ),
            }
        context[column] = column_record
    evidence["recipient_context"] = context

    # Realized-regime recomputation from frozen donor support, per pattern.
    print("== recomputing realized regimes from frozen donor support ==", flush=True)
    donor_mask = (channel == ASEC) & (clone == PUF_TAX_DETAIL_CLONE_INDEX)
    person_index = transferred["__person_index__"]
    if not isinstance(person_index, pd.Index):
        raise TypeError("checkpoint person index was not preserved as a pandas Index")
    regimes: dict[str, object] = {}
    for batch, family_targets in BATCH_FAMILY_TARGETS.items():
        target_complete = donor_mask.copy()
        for target in family_targets:
            target_values = np.asarray(transferred[target], dtype=np.float64)
            target_complete &= np.isfinite(target_values)
        batch_bank = bank[batch]
        if not isinstance(batch_bank, dict):
            raise TypeError(f"{batch}: bank evidence must be an object")
        physical_support_identity = index_identity(person_index[target_complete])
        # `_model_frame` intentionally copies the selected support onto an
        # explicit int64 Index from `np.arange(n)` before QRF sees it. Recreate
        # that exact model-front-door identity, not the checkpoint table index.
        reconstructed_identity = index_identity(
            pd.Index(np.arange(int(target_complete.sum()), dtype=np.int64))
        )
        first_target_record = batch_bank[family_targets[0]]
        pattern_order = tuple(
            step["pattern"] for step in first_target_record["pattern_steps"]
        )
        if not pattern_order:
            raise AssertionError(f"{batch}: no banked availability patterns")
        canonical_pattern_bindings = {
            step["pattern"]: {
                "predictors": step["predictors"],
                "donor_index": step["donor_index"],
                "recipient_index": step["recipient_index"],
            }
            for step in first_target_record["pattern_steps"]
        }

        recomputed: dict[str, object] = {}
        persisted_flags: list[bool] = []
        for target in family_targets:
            target_record = batch_bank[target]
            steps = target_record["pattern_steps"]
            observed_order = tuple(step["pattern"] for step in steps)
            if observed_order != pattern_order:
                raise AssertionError(
                    f"{batch}/{target}: availability-pattern order changed: "
                    f"{observed_order} != {pattern_order}"
                )
            values = np.asarray(transferred[target], dtype=np.float64)[target_complete]
            if not np.isfinite(values).all():
                raise AssertionError(
                    f"{batch}/{target}: reconstructed donor target is nonfinite"
                )
            target_patterns: dict[str, object] = {}
            for step in steps:
                pattern = step["pattern"]
                donor_identity = step["donor_index"]
                if donor_identity != reconstructed_identity:
                    raise AssertionError(
                        f"{batch}/{target}/{pattern}: reconstructed frozen donor "
                        f"identity {reconstructed_identity} does not match banked "
                        f"identity {donor_identity}"
                    )
                canonical = canonical_pattern_bindings[pattern]
                if (
                    step["predictors"] != canonical["predictors"]
                    or step["donor_index"] != canonical["donor_index"]
                    or step["recipient_index"] != canonical["recipient_index"]
                ):
                    raise AssertionError(
                        f"{batch}/{target}/{pattern}: banked pattern identities "
                        "differ across chained targets"
                    )
                model_config = step["model_config"]
                zero_atol = float(model_config["zero_atol"])
                realized = detect_regime(values, zero_atol=zero_atol)
                persisted = step["persisted_regime"]
                persisted_flags.append(bool(step["regime_persisted"]))
                if persisted is not None and persisted != realized:
                    raise AssertionError(
                        f"{batch}/{target}/{pattern}: persisted regime {persisted!r} "
                        f"does not match recomputed regime {realized!r}"
                    )
                target_patterns[pattern] = {
                    "realized_regime": realized,
                    "zero_atol": zero_atol,
                    "donor_index": dict(reconstructed_identity),
                    "bank_identity_match": True,
                    "persisted_regime": persisted,
                    "persisted_regime_match": (
                        None if persisted is None else persisted == realized
                    ),
                }
            recomputed[target] = target_patterns

        if any(persisted_flags) and not all(persisted_flags):
            raise AssertionError(f"{batch}: only some target-pattern regimes persist")
        regimes[batch] = {
            "donor_rows_reconstructed": int(target_complete.sum()),
            "physical_checkpoint_support_index": physical_support_identity,
            "reconstructed_donor_index": reconstructed_identity,
            "availability_patterns": [
                {
                    "pattern": pattern,
                    **canonical_pattern_bindings[pattern],
                }
                for pattern in pattern_order
            ],
            "all_banked_donor_identities_match_reconstruction": True,
            "regimes_persisted_in_failed_attempt_bank": all(persisted_flags),
            "failed_attempt_receipt_gap": (
                None if all(persisted_flags) else "realized_qrf_regime_not_persisted"
            ),
            "realized_regimes_by_target_and_pattern": recomputed,
        }
    evidence["realized_regimes"] = regimes

    # Nine coupled invariants on both stages.
    print("== rerunning the nine coupled invariants ==", flush=True)
    evidence["invariants"] = {
        "simulated": nine_invariants(simulated),
        "transferred": nine_invariants(transferred),
    }
    nonzero_simulated = {
        name: count
        for name, count in evidence["invariants"]["simulated"].items()
        if count
    }
    if nonzero_simulated:
        mismatches.append(
            f"simulated invariants unexpectedly nonzero: {nonzero_simulated}"
        )

    evidence["adjudication_mismatches"] = mismatches
    if mismatches:
        details = "\n".join(f"- {mismatch}" for mismatch in mismatches)
        raise AssertionError(
            "QBI ownership extraction disagrees with adjudicated evidence; "
            f"output was not written:\n{details}"
        )
    evidence["validation"] = validate_evidence(
        evidence,
        skip_sha=args.skip_sha,
    )
    if set(evidence) != {*TOP_LEVEL_EVIDENCE_KEYS, "validation"}:
        raise AssertionError("QBI evidence final top-level schema changed")
    write_json_atomic(output, evidence)

    print(f"\nevidence written to {output}")
    print("adjudication mismatches: 0")
    for target, record in legs.items():
        s0 = record["simulated_clone0"]
        t0 = record["transferred_clone0"]
        t1 = record["transferred_clone1"]
        print(f"\n{target}:")
        print(
            f"  clone1 transferred (producer): ratio="
            f"{fmt(t1.get('incidence_ratio_acs_over_asec', float('nan')))} "
            f"qed={fmt(t1.get('quantile_envelope_distance', float('nan')))}"
        )
        print(
            f"  clone0 transferred (post-transfer, pre-reconciliation): ratio="
            f"{fmt(t0.get('incidence_ratio_acs_over_asec', float('nan')))} "
            f"qed={fmt(t0.get('quantile_envelope_distance', float('nan')))}"
        )
        print(
            f"  clone0 simulated (terminal battery): ratio="
            f"{fmt(s0.get('incidence_ratio_acs_over_asec', float('nan')))} "
            f"qed={fmt(s0.get('quantile_envelope_distance', float('nan')))}"
        )


if __name__ == "__main__":
    main()
