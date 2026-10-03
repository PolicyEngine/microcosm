#!/usr/bin/env python3
"""Write a seeded, stratified whole-household subsample of a US export H5.

Route A's attempt on build 310842b98 ran 49,304 s and then failed its
post-export reform-coverage smoke on four stale probe definitions (fixed in
#1046); the post-export stages after the smoke (reform validation,
demographics, source coverage, take-up participation) have never run on a
Route A export. A code or definition failure in those stages does not depend
on the population's size, so this tool writes a household subsample of an
export H5 for ``tools/probe_us_post_export.py`` to run them on in minutes.

Design (a stratified sample with a take-all stratum and a ratio estimator):

- **Strata.** Households are stratified by ``household_support_channel`` and,
  where the export stores it, the person-level ``source_year`` (constant
  within a household, or the tool refuses). The Route A export stores its
  ASEC-channel households first and its PUF tax-detail households after
  them, so a first-N sample would be one channel only; the draw is random
  within each stratum instead.
- **Certainty (take-all) stratum**, kept at source weight (inclusion
  probability 1), with every household's reasons receipted:

  - *Thin probes.* For every reform-coverage smoke probe, the households
    carrying any nonzero binding input are its pool carriers. When
    ``fraction * carriers`` is below ``--certainty-threshold`` (default 30),
    a draw could miss the probe's support or see too little of it, so every
    carrier is kept.
  - *Dominant households.* A weighted total that a few records dominate is
    usually underestimated by a sample that missed them, with a standard
    error that is too small for the same reason. So, probe by probe, every
    carrier whose weighted input mass is at least
    ``--size-certainty-multiplier`` (default 2) times the mass one sampled
    carrier would represent is kept, repeated on the rest until none
    qualifies (:func:`size_certainty_households`).

- **Draw.** Within each stratum, ``min(N, max(2, floor(fraction * N)))`` of
  its ``N`` non-certainty households are drawn without replacement, one
  ``numpy.random.default_rng(seed)`` stream over the sorted household ids of
  the sorted strata, so the draw depends on ids, not row order.
- **Weights.** A certainty household keeps its weight. A drawn household in
  stratum ``h`` gets ``weight * X_h / x_h``, with ``X_h`` the stratum's
  non-certainty weight mass and ``x_h`` the drawn households': a ratio
  estimator with the source weight as its auxiliary, which conserves every
  stratum's weight total exactly (``sum of sampled weights == sum of source
  weights``). The Horvitz-Thompson factor ``N_h / n_h`` is receipted beside
  it; the ratio form trades an O(1/n) bias for exact totals.

The defaults were calibrated on the published Route A export
(``docs/evidence/us-export-subsample-design``). So the subsample is larger
than ``fraction`` of the pool: at 0.05 on that export, about 7%.

Whole households enter the sample together: a household's persons and every
group unit they reference, and nothing else. The source must nest every group
unit in one household (the post-export scorer's batching premise), or the
tool refuses before sampling.

Memory: the tool never loads the whole export. It reads the household table,
the person membership columns and the binding-input columns in one chunked
pass, then each table's selected rows in a second chunked pass. The subsample
is written through ``PolicyEngineUSEngine().write_dataset``, the release's own
writer, and the written file is re-read and checked against the selection:
ids, weights, per-stratum weight totals, per-entity row counts, the stored
tables and columns, and the column dtype kinds.

Usage::

    .venv/bin/python tools/sample_us_export_households.py \\
        --export <populace_us_2024.h5> --fraction 0.05 --seed 0 --out <dir>

writes ``<dir>/populace_us_2024.h5`` and ``<dir>/sample_receipt.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Bump with any change to the draw, the weights or the receipt layout.
SAMPLE_RECEIPT_SCHEMA_VERSION = 1

US_PERSON_ENTITY = "person"
US_GROUP_ENTITIES = ("household", "tax_unit", "spm_unit", "family", "marital_unit")
US_ENTITIES = (US_PERSON_ENTITY, *US_GROUP_ENTITIES)
HOUSEHOLD_WEIGHT_COLUMN = "household_weight"
TIME_PERIOD_KEY = "_time_period"
DEFAULT_STRATUM_COLUMNS = ("household_support_channel", "source_year")
#: A probe expecting fewer sampled carriers than this is taken whole. At 5,
#: the Route A probes with 101 and 394 carrier households were still drawn,
#: and the standard errors of their input mass missed by more than 3 SE in
#: 44% and 58% of seeds (design 5:0 in docs/evidence/us-export-subsample-
#: design/coverage.json); 30 takes their carriers whole, 492 households more
#: than threshold 5 keeps (505 certainty households instead of 13).
DEFAULT_CERTAINTY_THRESHOLD = 30.0
#: Size-certainty cutoff, in units of the mass one sampled carrier represents
#: (see :func:`size_certainty_households`); 0 disables the rule.
DEFAULT_SIZE_CERTAINTY_MULTIPLIER = 2.0
#: Rows per chunked read are sized to about this many stored bytes.
DEFAULT_CHUNK_BYTES = 128 * 2**20
#: Fewest rows per chunked read when the size comes from ``chunk_bytes``.
MINIMUM_CHUNK_ROWS = 1024
RECEIPT_FILENAME = "sample_receipt.json"

#: The per-stratum draw rule, declared in every receipt.
NONCERTAINTY_COUNT_RULE = (
    "min(N, max(2, floor(fraction * N))) of a stratum's N eligible "
    "non-certainty households (N when N < 2), so every sampled stratum has a "
    "within-stratum variance; 0 for a stratum whose households are all "
    "certainty households"
)
WEIGHT_RULE = (
    "certainty households keep their source weight (inclusion probability 1); "
    "a drawn household in stratum h gets weight * X_h / x_h, X_h the stratum's "
    "eligible non-certainty weight mass and x_h the drawn households' (a ratio "
    "estimator with the source weight as its auxiliary, calibrated to every "
    "stratum's weight total: exact totals, O(1/n) bias). The inclusion "
    "probability of a drawn household is n_h / N_h; its Horvitz-Thompson "
    "factor N_h / n_h is receipted beside the ratio factor"
)
DRAW_RULE = (
    "numpy.random.default_rng(seed).choice(sorted non-certainty household ids, "
    "size, replace=False) per stratum, strata in sorted label order; a stratum "
    "taking every eligible household consumes no draws"
)
CERTAINTY_RULE = (
    "a probe's carriers are the households with a nonzero value of any of its "
    "binding inputs (booleans count as 1, missing as 0); every carrier of a "
    "probe with 0 < fraction * carriers < threshold is a certainty household"
)
SIZE_CERTAINTY_RULE = (
    "a household's mass in a probe is its weight times the absolute values of "
    "the probe's binding inputs; probe by probe, every carrier with mass >= "
    "multiplier * residual mass / (fraction * residual carriers) is a "
    "certainty household, repeated on the rest until none qualifies (the "
    "probability-proportional-to-size certainty rule)"
)

#: Relative tolerance of the per-stratum weight-total conservation check.
CONSERVATION_RTOL = 1e-9

#: The microcosm frame-table codec attribute (``microcosm.frame.materialize``).
#: A table carrying it stores nullable booleans the chunked reader does not
#: restore, so such a source is refused rather than resampled lossily.
_FRAME_TABLE_CODEC_ATTR = "_microcosm_frame_table_codec"


def id_column(entity: str) -> str:
    return f"{entity}_id"


def membership_column(entity: str) -> str:
    return f"person_{entity}_id"


PERSON_MEMBERSHIP_COLUMNS = tuple(
    membership_column(entity) for entity in US_GROUP_ENTITIES
)


# ---------------------------------------------------------------------------
# Pure selection core (no H5, no engine)
# ---------------------------------------------------------------------------


def numeric_support_values(values: pd.Series | np.ndarray) -> np.ndarray:
    """A leaf column as float64, the release preflight's support semantics.

    Mirrors ``release_gate_preflight._numeric_column``: booleans become their
    1.0/0.0 mass, and missing or non-numeric values count as 0, so "nonzero"
    means the same thing here as in the smoke-probe support audit.
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy(dtype=np.float64)
    return pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)


def _label_value(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, (np.integer,)):
        return str(int(value))
    return str(value)


def assert_units_nest_in_households(person: pd.DataFrame) -> dict[str, int]:
    """Refuse a person table whose group units span more than one household.

    Whole-household sampling keeps a unit whole only if the unit lies inside
    one household; the post-export scorer refuses the same frame
    (``_assert_post_export_batching_premises``). Returns each group entity's
    unit count.
    """
    household = person[membership_column("household")].to_numpy()
    counts: dict[str, int] = {}
    for entity in US_GROUP_ENTITIES:
        if entity == "household":
            counts[entity] = int(len(np.unique(household)))
            continue
        pairs = pd.DataFrame(
            {"unit": person[membership_column(entity)].to_numpy(), "hh": household}
        ).drop_duplicates()
        spanning = pairs["unit"].duplicated(keep=False)
        if spanning.any():
            examples = sorted(pairs.loc[spanning, "unit"].unique().tolist())[:5]
            raise ValueError(
                f"{int(pairs.loc[spanning, 'unit'].nunique())} {entity} unit(s) "
                f"span more than one household (e.g. {examples}); a "
                "whole-household subsample would split them, and the "
                "post-export scorer refuses such a frame."
            )
        counts[entity] = int(pairs["unit"].nunique())
    return counts


def household_stratum_labels(
    household: pd.DataFrame,
    person: pd.DataFrame,
    stratum_columns: Sequence[str],
) -> tuple[np.ndarray, dict[str, object]]:
    """One stratum label per household row, in household-table order.

    A stratum column is read from the household table when it is there, else
    from the person table, where it must be constant within each household.
    Absent columns are skipped and receipted. With no column present every
    household is in the single stratum ``"all"``.
    """
    household_ids = household[id_column("household")].to_numpy()
    used: list[str] = []
    absent: list[str] = []
    parts: list[list[str]] = []
    for column in stratum_columns:
        if column in household.columns:
            values = household[column].to_numpy()
            used.append(column)
        elif column in person.columns:
            pairs = pd.DataFrame(
                {
                    "hh": person[membership_column("household")].to_numpy(),
                    "value": person[column].to_numpy(),
                }
            ).drop_duplicates()
            mixed = pairs["hh"].duplicated(keep=False)
            if mixed.any():
                raise ValueError(
                    f"Stratum column {column!r} varies within "
                    f"{int(pairs.loc[mixed, 'hh'].nunique())} household(s); a "
                    "person-level stratum must be constant within a household."
                )
            values = pairs.set_index("hh")["value"].reindex(household_ids).to_numpy()
            used.append(column)
        else:
            absent.append(column)
            continue
        if pd.isna(values).any():
            raise ValueError(
                f"Stratum column {column!r} is missing for "
                f"{int(pd.isna(values).sum())} household(s)."
            )
        parts.append([f"{column}={_label_value(value)}" for value in values])
    if parts:
        labels = np.asarray(
            ["|".join(pieces) for pieces in zip(*parts, strict=True)], dtype=object
        )
    else:
        labels = np.full(len(household_ids), "all", dtype=object)
    return labels, {"requested": list(stratum_columns), "used": used, "absent": absent}


@dataclass(frozen=True)
class ProbeCarriers:
    """A smoke probe's pool carriers and whether they form a take-all set."""

    probe_id: str
    binding_inputs: tuple[str, ...]
    input_carriers: Mapping[str, int]
    input_entities: Mapping[str, str]
    absent_inputs: tuple[str, ...]
    carrier_household_ids: np.ndarray
    expected_sampled_carriers: float
    certainty: bool

    def record(self) -> dict[str, object]:
        return {
            "probe": self.probe_id,
            "binding_inputs": list(self.binding_inputs),
            "input_entities": dict(self.input_entities),
            "input_carrier_households": dict(self.input_carriers),
            "absent_inputs": list(self.absent_inputs),
            "carrier_households": int(len(self.carrier_household_ids)),
            "expected_sampled_carriers": self.expected_sampled_carriers,
            "certainty": self.certainty,
        }


def probe_carriers(
    probes: Iterable[Any],
    leaf_households: Mapping[str, tuple[str, np.ndarray, np.ndarray]],
    *,
    fraction: float,
    threshold: float,
) -> tuple[list[ProbeCarriers], dict[int, list[str]]]:
    """Pool carriers per probe and the certainty households they imply.

    ``leaf_households`` maps a binding input stored by the export to
    ``(entity, values, household id of each row)``. Returns the per-probe
    records and ``{certainty household id: [probe ids]}``.
    """
    records: list[ProbeCarriers] = []
    reasons: dict[int, list[str]] = {}
    carrier_cache: dict[str, np.ndarray] = {}
    for probe in probes:
        inputs = tuple(str(leaf) for leaf in probe.binding_inputs)
        per_input: dict[str, int] = {}
        entities: dict[str, str] = {}
        absent: list[str] = []
        union: list[np.ndarray] = []
        for leaf in inputs:
            if leaf not in leaf_households:
                absent.append(leaf)
                continue
            if leaf not in carrier_cache:
                entity, values, households = leaf_households[leaf]
                nonzero = numeric_support_values(values) != 0.0
                carrier_cache[leaf] = np.unique(np.asarray(households)[nonzero])
            entities[leaf] = leaf_households[leaf][0]
            per_input[leaf] = int(len(carrier_cache[leaf]))
            union.append(carrier_cache[leaf])
        carriers = (
            np.unique(np.concatenate(union)) if union else np.asarray([], np.int64)
        )
        expected = float(fraction) * len(carriers)
        certainty = bool(len(carriers) > 0 and expected < float(threshold))
        if certainty:
            for household_id in carriers.tolist():
                reasons.setdefault(int(household_id), []).append(str(probe.id))
        records.append(
            ProbeCarriers(
                probe_id=str(probe.id),
                binding_inputs=inputs,
                input_carriers=per_input,
                input_entities=entities,
                absent_inputs=tuple(absent),
                carrier_household_ids=carriers,
                expected_sampled_carriers=expected,
                certainty=certainty,
            )
        )
    return records, reasons


def size_certainty_households(
    probes: Iterable[Any],
    leaf_households: Mapping[str, tuple[str, np.ndarray, np.ndarray]],
    household_ids: np.ndarray,
    weights: np.ndarray,
    *,
    fraction: float,
    multiplier: float,
) -> tuple[dict[int, list[str]], dict[str, dict[str, object]]]:
    """Households kept with certainty for their size in a probe's input mass.

    A household's mass in a probe is its weight times the absolute values of
    the probe's binding inputs over its rows (booleans count as 1). Probe by
    probe, every carrier whose mass is at least ``multiplier * residual mass /
    (fraction * residual carriers)`` becomes a certainty household, and the
    rule repeats on the rest until none qualifies. At multiplier 1 the cutoff
    is the mass one sampled carrier would have to represent: the classical
    certainty rule of probability-proportional-to-size sampling. Afterwards
    no remaining carrier holds more than ``multiplier / fraction`` times the
    mean residual mass, which bounds the skewness an equal-probability draw
    of the rest (and its variance estimate) has to cope with. A weighted
    total dominated by a few records is otherwise usually underestimated,
    with a standard error that is too small because the sample missed them.

    Returns ``{household id: ["<probe id>:size", ...]}`` and a record per
    probe. A non-positive ``multiplier`` disables the rule.
    """
    reasons: dict[int, list[str]] = {}
    records: dict[str, dict[str, object]] = {}
    if not (multiplier and multiplier > 0):
        return reasons, records
    # Work in household-id order, so sums (and so the cutoffs) do not depend
    # on the household table's row order.
    order = np.argsort(np.asarray(household_ids), kind="stable")
    household_ids = np.asarray(household_ids)[order]
    weights = np.asarray(weights, dtype=np.float64)[order]
    index = pd.Index(household_ids)
    leaf_mass: dict[str, np.ndarray] = {}
    for probe in probes:
        mass = np.zeros(len(household_ids), dtype=np.float64)
        for leaf in (str(leaf) for leaf in probe.binding_inputs):
            if leaf not in leaf_households:
                continue
            if leaf not in leaf_mass:
                _, values, households = leaf_households[leaf]
                positions = index.get_indexer(np.asarray(households))
                if (positions < 0).any():
                    raise ValueError(
                        f"binding input {leaf!r} has rows in households the "
                        "household table does not list."
                    )
                total = np.zeros(len(household_ids), dtype=np.float64)
                np.add.at(total, positions, np.abs(numeric_support_values(values)))
                leaf_mass[leaf] = total
            mass += leaf_mass[leaf]
        mass *= weights
        residual = mass > 0.0
        total_mass = float(mass.sum())
        certain = np.zeros(len(household_ids), dtype=bool)
        cutoff = None
        while residual.any():
            expected = float(fraction) * int(residual.sum())
            cutoff = (
                float(multiplier) * float(mass[residual].sum()) / max(expected, 1.0)
            )
            large = residual & (mass >= cutoff)
            if not large.any():
                break
            certain |= large
            residual &= ~large
        for household_id in household_ids[certain].tolist():
            reasons.setdefault(int(household_id), []).append(f"{probe.id}:size")
        records[str(probe.id)] = {
            "households": int(certain.sum()),
            "final_cutoff": cutoff,
            "mass_share": 0.0
            if total_mass <= 0.0
            else float(mass[certain].sum() / total_mass),
        }
    return reasons, records


@dataclass(frozen=True)
class HouseholdDraw:
    """The selected households, their adjusted weights and the receipt."""

    selected_ids: np.ndarray  # sorted
    adjusted_weights: np.ndarray  # aligned to selected_ids
    source_weights: np.ndarray  # aligned to selected_ids
    certainty: np.ndarray  # bool, aligned to selected_ids
    labels: np.ndarray  # stratum label, aligned to selected_ids
    strata: dict[str, dict[str, object]] = field(default_factory=dict)

    def weight_of(self) -> pd.Series:
        return pd.Series(self.adjusted_weights, index=self.selected_ids)


def validate_fraction(fraction: float) -> float:
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not 0.0 < float(fraction) <= 1.0
    ):
        raise ValueError(
            f"fraction must be a finite number in (0, 1]; got {fraction!r}."
        )
    return float(fraction)


def validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer; got {seed!r}.")
    return int(seed)


def draw_households(
    household_ids: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    certainty_ids: Iterable[int],
    *,
    fraction: float,
    seed: int,
) -> HouseholdDraw:
    """Draw the stratified subsample and its conserving weights.

    Every input is aligned to the household table; its row order does not
    affect the result (ids are sorted before the draw). See the module
    docstring for the rules; they are also receipted verbatim.
    """
    fraction = validate_fraction(fraction)
    seed = validate_seed(seed)
    household_ids = np.asarray(household_ids)
    weights = np.asarray(weights, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    if not all(isinstance(label, str) for label in labels.tolist()):
        raise ValueError(
            "stratum labels must be strings (household_stratum_labels builds "
            "them); a numeric or missing label would key its stratum ambiguously."
        )
    if not np.issubdtype(household_ids.dtype, np.integer):
        raise ValueError(
            f"household ids must be integer-typed; got {household_ids.dtype}."
        )
    if not (len(household_ids) == len(weights) == len(labels)):
        raise ValueError("household ids, weights and labels must be aligned.")
    if len(household_ids) == 0:
        raise ValueError("the export has no households to sample.")
    if len(np.unique(household_ids)) != len(household_ids):
        raise ValueError("household ids must be unique.")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(
            "household weights must be finite and non-negative for a "
            "Horvitz-Thompson subsample."
        )
    certainty_list = [int(value) for value in certainty_ids]
    certain = np.isin(household_ids, np.asarray(certainty_list, dtype=np.int64))
    missing = set(certainty_list) - set(household_ids.tolist())
    if missing:
        raise ValueError(
            f"{len(missing)} certainty household id(s) are not in the export."
        )
    order = np.argsort(household_ids, kind="stable")
    ids_sorted = household_ids[order]
    weights_sorted = weights[order]
    labels_sorted = labels[order]
    certain_sorted = certain[order]

    rng = np.random.default_rng(seed)
    selected_positions: list[np.ndarray] = []
    factors = np.zeros(len(ids_sorted), dtype=np.float64)
    strata: dict[str, dict[str, object]] = {}
    for label in sorted(set(labels_sorted.tolist())):
        in_stratum = labels_sorted == label
        stratum_cert = np.flatnonzero(in_stratum & certain_sorted)
        eligible = np.flatnonzero(in_stratum & ~certain_sorted)
        n_eligible = int(len(eligible))
        requested = min(n_eligible, max(2, int(math.floor(fraction * n_eligible))))
        if requested == n_eligible:
            drawn = eligible
        else:
            drawn = np.sort(rng.choice(eligible, size=requested, replace=False))
        eligible_mass = float(weights_sorted[eligible].sum())
        drawn_mass = float(weights_sorted[drawn].sum())
        if eligible_mass > 0.0 and drawn_mass <= 0.0:
            raise ValueError(
                f"Stratum {label!r} drew {requested} non-certainty household(s) "
                f"that all weigh 0 against {eligible_mass:,.3f} of eligible "
                "weight mass; the draw cannot carry that mass. Raise the "
                "fraction or change the seed."
            )
        factor = eligible_mass / drawn_mass if eligible_mass > 0.0 else 1.0
        factors[stratum_cert] = 1.0
        factors[drawn] = factor
        selected_positions.extend([stratum_cert, drawn])
        source_mass = float(weights_sorted[in_stratum].sum())
        certainty_mass = float(weights_sorted[stratum_cert].sum())
        sampled_mass = certainty_mass + drawn_mass * factor
        strata[str(label)] = {
            "households": int(in_stratum.sum()),
            "certainty_households": int(len(stratum_cert)),
            "eligible_noncertainty_households": n_eligible,
            "drawn_noncertainty_households": int(len(drawn)),
            "sampled_households": int(len(stratum_cert) + len(drawn)),
            "design_fraction": (
                None if n_eligible == 0 else float(len(drawn) / n_eligible)
            ),
            "expansion_factor": (
                None if len(drawn) == 0 else float(n_eligible / len(drawn))
            ),
            "source_weight_total": source_mass,
            "certainty_weight_total": certainty_mass,
            "eligible_noncertainty_weight_total": eligible_mass,
            "drawn_noncertainty_weight_total_before_adjustment": drawn_mass,
            "drawn_weight_share": (
                None if eligible_mass <= 0.0 else float(drawn_mass / eligible_mass)
            ),
            "noncertainty_weight_factor": float(factor),
            "sampled_weight_total": float(sampled_mass),
            "conservation_abs_error": float(sampled_mass - source_mass),
        }
    positions = (
        np.sort(np.concatenate(selected_positions))
        if selected_positions
        else np.asarray([], dtype=np.int64)
    )
    selected_ids = ids_sorted[positions]
    source_selected = weights_sorted[positions]
    adjusted = source_selected * factors[positions]
    # Recompute each stratum's realized total from the adjusted vector itself,
    # so the receipt's conservation line is the vector the writer gets.
    selected_labels = labels_sorted[positions]
    for label, record in strata.items():
        realized = float(adjusted[selected_labels == label].sum())
        source = float(record["source_weight_total"])
        record["sampled_weight_total"] = realized
        record["conservation_abs_error"] = realized - source
        record["conservation_rel_error"] = (
            0.0 if source == 0.0 else (realized - source) / source
        )
        if not math.isclose(realized, source, rel_tol=CONSERVATION_RTOL, abs_tol=0.0):
            raise AssertionError(
                f"Stratum {label!r} weight total {realized!r} does not "
                f"conserve its source total {source!r}."
            )
    return HouseholdDraw(
        selected_ids=selected_ids,
        adjusted_weights=adjusted,
        source_weights=source_selected,
        certainty=certain_sorted[positions],
        labels=selected_labels,
        strata=strata,
    )


def entity_row_masks(
    person_memberships: pd.DataFrame,
    group_ids: Mapping[str, np.ndarray],
    selected_household_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Row masks per entity table for the selected households.

    Persons of a selected household, and every group row those persons
    reference: :meth:`microcosm.frame.Frame.select`'s semantics over the
    stored tables, without building the frame.
    """
    selected = np.asarray(selected_household_ids)
    person_mask = np.isin(
        person_memberships[membership_column("household")].to_numpy(), selected
    )
    masks = {US_PERSON_ENTITY: person_mask}
    for entity in US_GROUP_ENTITIES:
        referenced = np.unique(
            person_memberships[membership_column(entity)].to_numpy()[person_mask]
        )
        masks[entity] = np.isin(np.asarray(group_ids[entity]), referenced)
    return masks


def household_of_rows(
    entity: str,
    person_memberships: pd.DataFrame,
    entity_ids: np.ndarray,
) -> np.ndarray:
    """Household id of each row of ``entity`` (whose rows carry ``entity_ids``)."""
    person_household = person_memberships[membership_column("household")]
    if entity == "household":
        return np.asarray(entity_ids)
    if entity == US_PERSON_ENTITY:
        return person_household.to_numpy()
    first = (
        pd.DataFrame(
            {
                "unit": person_memberships[membership_column(entity)].to_numpy(),
                "hh": person_household.to_numpy(),
            }
        )
        .drop_duplicates("unit")
        .set_index("unit")["hh"]
    )
    mapped = first.reindex(np.asarray(entity_ids))
    if mapped.isna().any():
        raise ValueError(
            f"{int(mapped.isna().sum())} {entity} row(s) are referenced by no person."
        )
    return mapped.to_numpy(dtype=np.int64)


# ---------------------------------------------------------------------------
# Frame reference path (tests and small inputs)
# ---------------------------------------------------------------------------


def sample_frame(frame, draw: HouseholdDraw):
    """The same subsample built through :meth:`Frame.select` in memory.

    The chunked H5 path never builds the source frame; this reference does,
    so a test can check that both paths realize the same tables and weights.
    """
    from microcosm.frame import MassChange, Weights

    person_mask = (
        frame.table(US_PERSON_ENTITY)[membership_column("household")]
        .isin(draw.selected_ids)
        .to_numpy()
    )
    sampled = frame.select(person_mask)
    sampled_ids = sampled.table("household")[id_column("household")].to_numpy()
    if set(sampled_ids.tolist()) != set(draw.selected_ids.tolist()):
        raise AssertionError("Frame.select realized a different household set.")
    new_values = draw.weight_of().reindex(sampled_ids).to_numpy(dtype=np.float64)
    existing = sampled.weights_for("household")
    old_total = float(existing.total)
    new_total = float(new_values.sum())
    return sampled.with_weights(
        "household",
        Weights(new_values, existing.kind),
        mass=MassChange(
            factor=None if old_total == 0.0 else new_total / old_total,
            reason="stratified Horvitz-Thompson export subsample reweighting",
        ),
    )


# ---------------------------------------------------------------------------
# Chunked H5 access (needs PyTables)
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _storer(store: pd.HDFStore, key: str):
    storer = store.get_storer(key)
    if storer is None:
        raise KeyError(key)
    return storer


def _refuse_frame_table_codec(store: pd.HDFStore, key: str) -> None:
    attrs = _storer(store, key).attrs
    if getattr(attrs, _FRAME_TABLE_CODEC_ATTR, None) is not None:
        raise ValueError(
            f"Table {key!r} stores nullable booleans under the microcosm frame "
            "codec, which the chunked subsample reader does not restore; "
            "resample it through the Frame path instead."
        )


def table_nrows(store: pd.HDFStore, key: str) -> int:
    return int(_storer(store, key).nrows)


def table_columns(store: pd.HDFStore, key: str) -> list[str]:
    return list(store.select(key, start=0, stop=0).columns)


def read_table_rows(
    store: pd.HDFStore,
    key: str,
    *,
    columns: Sequence[str] | None = None,
    row_mask: np.ndarray | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    chunk_rows: int | None = None,
) -> pd.DataFrame:
    """Read ``key`` in row chunks, keeping ``columns`` and the masked rows.

    Each chunk is read with ``HDFStore.select(start=, stop=)``, the reader
    ``USSingleYearDataset`` uses, so values and dtypes are what the release
    loader sees. A chunk holds ``chunk_rows`` rows, or by default about
    ``chunk_bytes`` stored bytes (at least :data:`MINIMUM_CHUNK_ROWS` rows).
    Peak memory is one chunk plus the rows kept: every kept part is copied
    out of its chunk, because the columns pandas returns from a
    ``format="table"`` select are views into the chunk's whole record buffer,
    and a kept view would pin every column of every chunk until the concat.
    """
    _refuse_frame_table_codec(store, key)
    storer = _storer(store, key)
    nrows = int(storer.nrows)
    if chunk_rows is None:
        rowsize = int(getattr(getattr(storer, "table", None), "rowsize", 0) or 1024)
        chunk_rows = max(MINIMUM_CHUNK_ROWS, int(chunk_bytes) // max(1, rowsize))
    chunk_rows = int(chunk_rows)
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be positive; got {chunk_rows}.")
    if row_mask is not None:
        row_mask = np.asarray(row_mask, dtype=bool)
        if row_mask.shape != (nrows,):
            raise ValueError(
                f"row mask for {key!r} has shape {row_mask.shape}, the table "
                f"has {nrows} rows."
            )
    wanted = None if columns is None else list(columns)
    parts: list[pd.DataFrame] = []
    for start in range(0, nrows, chunk_rows):
        stop = min(nrows, start + chunk_rows)
        local = None if row_mask is None else row_mask[start:stop]
        if local is not None and not local.any():
            continue
        part = store.select(key, start=start, stop=stop)
        if wanted is not None:
            part = part[wanted]
        if local is not None:
            # A positional take copies just the kept rows.
            part = part.iloc[np.flatnonzero(local)]
        elif wanted is not None:
            part = part.copy()
        parts.append(part)
    if not parts:
        empty = store.select(key, start=0, stop=0)
        return (empty if wanted is None else empty[wanted]).reset_index(drop=True)
    return pd.concat(parts).reset_index(drop=True)


def _rss_peak_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux.
    return int(peak if sys.platform == "darwin" else peak * 1024)


class _PhaseClock:
    """Wall, CPU and peak-RSS per phase, for the receipt."""

    def __init__(self) -> None:
        self.phases: dict[str, dict[str, float]] = {}

    def run(self, name: str, fn: Callable[[], Any]) -> Any:
        wall = time.perf_counter()
        cpu = time.process_time()
        try:
            return fn()
        finally:
            self.phases[name] = {
                "wall_seconds": round(time.perf_counter() - wall, 3),
                "cpu_seconds": round(time.process_time() - cpu, 3),
                "process_peak_rss_gib": round(_rss_peak_bytes() / 2**30, 3),
            }


def _library_versions() -> dict[str, str | None]:
    import importlib.metadata as metadata

    versions: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "tables", "h5py", "policyengine-us"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _tool_commit() -> dict[str, object]:
    """The commit this tool was loaded from, whether ``tools/`` differed from
    it, and this file's sha256 (a check on the commit)."""
    root = Path(__file__).resolve().parents[1]
    sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", "tools"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tools_dirty": None, "sha256": sha256}
    return {"commit": head, "tools_dirty": bool(dirty), "sha256": sha256}


# Read once, as the module loads: the worktree's HEAD can move while a long
# draw runs.
_TOOL_SOURCE = _tool_commit()


def _default_probes() -> tuple[Any, ...]:
    from microcosm.build.us_runtime.release_input_coverage import (
        us_release_reform_coverage_probes,
    )

    return us_release_reform_coverage_probes()


def _default_write_dataset(frame, path: Path, period: int) -> None:
    from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine

    PolicyEngineUSEngine().write_dataset(frame, path, period=period)


def _stored_dtype_kinds(path: Path) -> dict[str, dict[str, str]]:
    """``{table: {column: numpy dtype kind}}`` from HDF metadata only."""
    import h5py

    kinds: dict[str, dict[str, str]] = {}
    with h5py.File(path, "r") as h5:
        for key in h5:
            table = h5[key].get("table") if isinstance(h5[key], h5py.Group) else None
            names = getattr(getattr(table, "dtype", None), "names", None)
            if not names:
                continue
            kinds[key] = {
                name: table.dtype[name].kind for name in names if name != "index"
            }
    return kinds


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------


def sample_export(
    export_path: Path | str,
    out_dir: Path | str,
    *,
    fraction: float,
    seed: int,
    certainty_threshold: float = DEFAULT_CERTAINTY_THRESHOLD,
    size_certainty_multiplier: float = DEFAULT_SIZE_CERTAINTY_MULTIPLIER,
    stratum_columns: Sequence[str] = DEFAULT_STRATUM_COLUMNS,
    probes: Iterable[Any] | None = None,
    write_dataset: Callable[[Any, Path, int], None] | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    chunk_rows: int | None = None,
    dataset_filename: str | None = None,
    refuse_denied: bool = True,
) -> dict[str, object]:
    """Write the subsample H5 and its receipt; return the receipt."""
    from microcosm.frame import Frame, WeightKind, Weights
    from microcosm.frame.units import US_SCHEMA

    fraction = validate_fraction(fraction)
    seed = validate_seed(seed)
    if not (math.isfinite(float(certainty_threshold)) and certainty_threshold >= 0):
        raise ValueError("certainty_threshold must be finite and non-negative.")
    export_path = Path(export_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (dataset_filename or export_path.name)
    if out_path == export_path:
        raise ValueError("The subsample would overwrite its source export.")
    clock = _PhaseClock()
    write = write_dataset or _default_write_dataset

    def import_release_modules() -> tuple[Any, ...]:
        # With policyengine-us installed, importing microcosm.build.us_runtime
        # (the shipped probes, the deny-list boundary) builds the engine's
        # tax-benefit system, and so does the release writer: minutes of CPU
        # booked here rather than under whichever phase first imports them.
        loaded = tuple(_default_probes() if probes is None else probes)
        if refuse_denied:
            import microcosm.build.us_runtime.h5_io  # noqa: F401
        if write_dataset is None:
            import policyengine_us  # noqa: F401
        return loaded

    probes = clock.run("import_release_modules", import_release_modules)
    consumer = "US export household subsampler (tools/sample_us_export_households.py)"

    def identify() -> str:
        if refuse_denied:
            from microcosm.build.us_runtime.h5_io import refuse_denied_pool_h5

            return refuse_denied_pool_h5(export_path, consumer=consumer)
        return sha256_file(export_path)

    source_sha256 = clock.run("hash_source", identify)

    binding_inputs = sorted(
        {str(leaf) for probe in probes for leaf in probe.binding_inputs}
    )

    def read_selection_inputs():
        with pd.HDFStore(str(export_path), mode="r") as store:
            keys = {key.strip("/") for key in store.keys()}
            missing = [entity for entity in US_ENTITIES if entity not in keys]
            if missing:
                raise ValueError(f"{export_path} lacks entity table(s) {missing}.")
            time_period = int(store[TIME_PERIOD_KEY].iloc[0])
            columns = {entity: table_columns(store, entity) for entity in US_ENTITIES}
            nrows = {entity: table_nrows(store, entity) for entity in US_ENTITIES}
            household = read_table_rows(
                store, "household", chunk_bytes=chunk_bytes, chunk_rows=chunk_rows
            )
            person_columns = [
                *PERSON_MEMBERSHIP_COLUMNS,
                *[
                    column
                    for column in stratum_columns
                    if column in columns[US_PERSON_ENTITY]
                    and column not in columns["household"]
                ],
                *[leaf for leaf in binding_inputs if leaf in columns[US_PERSON_ENTITY]],
            ]
            person = read_table_rows(
                store,
                US_PERSON_ENTITY,
                columns=list(dict.fromkeys(person_columns)),
                chunk_bytes=chunk_bytes,
                chunk_rows=chunk_rows,
            )
            groups: dict[str, pd.DataFrame] = {"household": household}
            for entity in US_GROUP_ENTITIES[1:]:
                wanted = [id_column(entity)] + [
                    leaf for leaf in binding_inputs if leaf in columns[entity]
                ]
                groups[entity] = read_table_rows(
                    store,
                    entity,
                    columns=wanted,
                    chunk_bytes=chunk_bytes,
                    chunk_rows=chunk_rows,
                )
        return time_period, columns, nrows, person, groups

    time_period, source_columns, source_nrows, person, groups = clock.run(
        "read_selection_inputs", read_selection_inputs
    )
    household = groups["household"]
    if HOUSEHOLD_WEIGHT_COLUMN not in household.columns:
        raise ValueError(f"{export_path} stores no {HOUSEHOLD_WEIGHT_COLUMN}.")
    unit_counts = assert_units_nest_in_households(person)
    household_ids = household[id_column("household")].to_numpy()
    source_weights = household[HOUSEHOLD_WEIGHT_COLUMN].to_numpy(dtype=np.float64)

    labels, strata_columns = household_stratum_labels(
        household, person, stratum_columns
    )

    leaf_households: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    for leaf in binding_inputs:
        for entity in US_ENTITIES:
            table = person if entity == US_PERSON_ENTITY else groups[entity]
            if leaf in table.columns:
                ids = (
                    None
                    if entity == US_PERSON_ENTITY
                    else table[id_column(entity)].to_numpy()
                )
                leaf_households[leaf] = (
                    entity,
                    table[leaf].to_numpy(),
                    household_of_rows(entity, person, ids),
                )
                break
    probe_records, certainty_reasons = probe_carriers(
        probes, leaf_households, fraction=fraction, threshold=certainty_threshold
    )
    size_reasons, size_records = size_certainty_households(
        probes,
        leaf_households,
        household_ids,
        source_weights,
        fraction=fraction,
        multiplier=size_certainty_multiplier,
    )
    for household_id, why in size_reasons.items():
        certainty_reasons.setdefault(household_id, []).extend(why)
    draw = clock.run(
        "draw",
        lambda: draw_households(
            household_ids,
            source_weights,
            labels,
            sorted(certainty_reasons),
            fraction=fraction,
            seed=seed,
        ),
    )
    group_ids = {
        entity: groups[entity][id_column(entity)].to_numpy()
        for entity in US_GROUP_ENTITIES
    }
    masks = entity_row_masks(person, group_ids, draw.selected_ids)
    del person, groups, leaf_households

    def read_selected_rows() -> dict[str, pd.DataFrame]:
        with pd.HDFStore(str(export_path), mode="r") as store:
            return {
                entity: read_table_rows(
                    store,
                    entity,
                    row_mask=masks[entity],
                    chunk_bytes=chunk_bytes,
                    chunk_rows=chunk_rows,
                )
                for entity in US_ENTITIES
            }

    tables = clock.run("read_selected_rows", read_selected_rows)
    if refuse_denied:
        from microcosm.build.us_runtime.h5_io import assert_h5_unchanged

        assert_h5_unchanged(export_path, source_sha256, consumer=consumer)
    else:
        if sha256_file(export_path) != source_sha256:
            raise ValueError(f"{export_path} changed while being read.")

    household_table = tables["household"]
    adjusted = (
        draw.weight_of()
        .reindex(household_table[id_column("household")].to_numpy())
        .to_numpy(dtype=np.float64)
    )
    if np.isnan(adjusted).any():
        raise AssertionError("A selected household row has no drawn weight.")
    household_table = household_table.drop(columns=[HOUSEHOLD_WEIGHT_COLUMN])
    tables["household"] = household_table
    frame = Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(adjusted, WeightKind.CALIBRATED)},
    )
    del tables
    if out_path.exists():
        out_path.unlink()
    written_rows = {entity: int(frame.n(entity)) for entity in US_ENTITIES}
    clock.run("write_subsample", lambda: write(frame, out_path, time_period))
    frame = None

    verification = clock.run(
        "verify_subsample",
        lambda: _verify_written_subsample(
            out_path,
            draw=draw,
            expected_rows=written_rows,
            source_path=export_path,
            chunk_bytes=chunk_bytes,
            chunk_rows=chunk_rows,
        ),
    )
    output_sha256 = sha256_file(out_path)

    certainty_ids = sorted(certainty_reasons)
    receipt: dict[str, object] = {
        "schema_version": SAMPLE_RECEIPT_SCHEMA_VERSION,
        "tool": "tools/sample_us_export_households.py",
        "tool_source": dict(_TOOL_SOURCE),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            # Generator streams may change across NumPy releases (NEP 19), so
            # the draw is reproducible given the seed and these versions.
            "libraries": _library_versions(),
        },
        "source": {
            "path": str(export_path),
            "sha256": source_sha256,
            "size_bytes": export_path.stat().st_size,
            "time_period": time_period,
            "rows": source_nrows,
            "stored_columns": {
                entity: len(columns) for entity, columns in source_columns.items()
            },
            "group_units": unit_counts,
            "household_weight_total": float(source_weights.sum()),
        },
        "design": {
            "fraction": fraction,
            "seed": seed,
            "certainty_threshold": float(certainty_threshold),
            "size_certainty_multiplier": float(size_certainty_multiplier),
            "size_certainty_rule": SIZE_CERTAINTY_RULE,
            "strata_columns": strata_columns,
            "certainty_rule": CERTAINTY_RULE,
            "noncertainty_count_rule": NONCERTAINTY_COUNT_RULE,
            "draw_rule": DRAW_RULE,
            "weight_rule": WEIGHT_RULE,
            "conservation_rtol": CONSERVATION_RTOL,
        },
        "strata": draw.strata,
        "certainty": {
            "households": len(certainty_ids),
            "probes_with_certainty": sorted(
                record.probe_id for record in probe_records if record.certainty
            ),
            "household_ids": certainty_ids,
            "household_reasons": {
                str(household_id): certainty_reasons[household_id]
                for household_id in certainty_ids
            },
        },
        "probes": [
            {
                **record.record(),
                "size_certainty": size_records.get(record.probe_id),
                "sampled_carrier_households": int(
                    np.isin(record.carrier_household_ids, draw.selected_ids).sum()
                ),
            }
            for record in probe_records
        ],
        "selection": {
            "households": int(len(draw.selected_ids)),
            "certainty_households": int(draw.certainty.sum()),
            "drawn_households": int((~draw.certainty).sum()),
            "selected_household_ids_sha256": hashlib.sha256(
                json.dumps(
                    [int(value) for value in draw.selected_ids.tolist()],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "rows": written_rows,
            "household_weight_total": float(draw.adjusted_weights.sum()),
        },
        "output": {
            "path": str(out_path),
            "sha256": output_sha256,
            "size_bytes": out_path.stat().st_size,
        },
        "verification": verification,
        "timing": clock.phases,
        "process_peak_rss_gib": round(_rss_peak_bytes() / 2**30, 3),
    }
    (out_dir / RECEIPT_FILENAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def _verify_written_subsample(
    out_path: Path,
    *,
    draw: HouseholdDraw,
    expected_rows: Mapping[str, int],
    source_path: Path,
    chunk_bytes: int,
    chunk_rows: int | None = None,
) -> dict[str, object]:
    """Re-read the written subsample and check it against the selection."""
    from microcosm.data.stored_inputs import h5_stored_tables

    with pd.HDFStore(str(out_path), mode="r") as store:
        household = read_table_rows(
            store,
            "household",
            columns=[id_column("household"), HOUSEHOLD_WEIGHT_COLUMN],
            chunk_bytes=chunk_bytes,
            chunk_rows=chunk_rows,
        )
        person = read_table_rows(
            store,
            US_PERSON_ENTITY,
            columns=list(PERSON_MEMBERSHIP_COLUMNS),
            chunk_bytes=chunk_bytes,
            chunk_rows=chunk_rows,
        )
        rows = {entity: table_nrows(store, entity) for entity in US_ENTITIES}
    failures: list[str] = []
    if rows != dict(expected_rows):
        failures.append(f"written rows {rows} != selected rows {dict(expected_rows)}")
    written = household.set_index(id_column("household"))[HOUSEHOLD_WEIGHT_COLUMN]
    if sorted(written.index.tolist()) != draw.selected_ids.tolist():
        failures.append("written household ids differ from the selection")
    else:
        expected = draw.weight_of()
        if not np.array_equal(
            written.reindex(expected.index).to_numpy(dtype=np.float64),
            expected.to_numpy(dtype=np.float64),
        ):
            failures.append("written household weights differ from the draw")
    labels = pd.Series(draw.labels, index=draw.selected_ids)
    stratum_totals: dict[str, float] = {}
    for label, record in draw.strata.items():
        ids = labels.index[labels.to_numpy() == label]
        total = float(written.reindex(ids).sum())
        stratum_totals[label] = total
        source = float(record["source_weight_total"])
        if not math.isclose(total, source, rel_tol=CONSERVATION_RTOL, abs_tol=0.0):
            failures.append(
                f"stratum {label!r}: written weight total {total!r} != source "
                f"{source!r}"
            )
    try:
        assert_units_nest_in_households(person)
    except ValueError as error:
        failures.append(f"written person table: {error}")
    person_households = set(
        np.unique(person[membership_column("household")].to_numpy()).tolist()
    )
    if person_households != set(draw.selected_ids.tolist()):
        failures.append("written persons do not cover exactly the selected households")
    source_tables = h5_stored_tables(source_path)
    written_tables = h5_stored_tables(out_path)
    column_differences = {
        table: {
            "missing": sorted(set(source_tables.get(table, ())) - set(columns)),
            "added": sorted(set(columns) - set(source_tables.get(table, ()))),
        }
        for table, columns in written_tables.items()
        if set(columns) != set(source_tables.get(table, ()))
    }
    if set(written_tables) != set(source_tables):
        column_differences["_tables"] = {
            "missing": sorted(set(source_tables) - set(written_tables)),
            "added": sorted(set(written_tables) - set(source_tables)),
        }
    if column_differences:
        failures.append(f"stored columns differ from the source: {column_differences}")
    source_kinds = _stored_dtype_kinds(source_path)
    written_kinds = _stored_dtype_kinds(out_path)
    kind_differences = sorted(
        f"{table}.{column}: {kind}->{written_kinds.get(table, {}).get(column)}"
        for table, columns in source_kinds.items()
        for column, kind in columns.items()
        if column in written_kinds.get(table, {})
        and written_kinds[table][column] != kind
    )
    if kind_differences:
        failures.append(
            f"stored dtype kinds differ from the source: {kind_differences}"
        )
    if failures:
        raise AssertionError(
            "The written subsample failed verification: " + "; ".join(failures)
        )
    return {
        "passed": True,
        "rows": rows,
        "stratum_weight_totals": stratum_totals,
        "stored_tables_match_source": True,
        "dtype_kinds_match_source": True,
        "group_units_nest_in_households": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--export", type=Path, required=True, help="source export H5")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--certainty-threshold",
        type=float,
        default=DEFAULT_CERTAINTY_THRESHOLD,
        help="keep every carrier of a probe with fraction x carriers below this",
    )
    parser.add_argument(
        "--size-certainty-multiplier",
        type=float,
        default=DEFAULT_SIZE_CERTAINTY_MULTIPLIER,
        help="keep every household whose weighted input mass in a probe is at "
        "least this many times the mass one sampled carrier represents (0: off)",
    )
    parser.add_argument(
        "--strata",
        default=",".join(DEFAULT_STRATUM_COLUMNS),
        help="comma-separated stratum columns (household or person level)",
    )
    parser.add_argument("--chunk-mib", type=int, default=DEFAULT_CHUNK_BYTES // 2**20)
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=None,
        help="rows per chunked read (default: sized from --chunk-mib)",
    )
    parser.add_argument(
        "--dataset-filename",
        default=None,
        help="output H5 name (default: the source's file name)",
    )
    args = parser.parse_args(argv)
    receipt = sample_export(
        args.export,
        args.out,
        fraction=args.fraction,
        seed=args.seed,
        certainty_threshold=args.certainty_threshold,
        size_certainty_multiplier=args.size_certainty_multiplier,
        stratum_columns=tuple(
            column.strip() for column in args.strata.split(",") if column.strip()
        ),
        chunk_bytes=int(args.chunk_mib) * 2**20,
        chunk_rows=args.chunk_rows,
        dataset_filename=args.dataset_filename,
    )
    selection = receipt["selection"]
    print(
        f"Wrote {selection['households']:,} households "
        f"({selection['certainty_households']:,} certainty) to "
        f"{receipt['output']['path']}; receipt "
        f"{Path(args.out) / RECEIPT_FILENAME}.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
