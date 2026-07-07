"""UK output-area-anchored geography-ladder assignment.

The UK artifact's geography spine is anchored at the 2021 Census output area
(OA) — the UK equivalent of the US 2020 census tabulation block
(``us_runtime/geography_ladder.py``). Every household is placed at one OA and
every other layer of the ladder is a deterministic function of that OA:

1. **Region** is already assigned — the FRS carries it and it is a calibrated
   dimension of the national build (the analogue of the US household's
   pre-assigned ``state_fips``).
2. **A 2024 Westminster parliamentary constituency is sampled within the
   region**, proportional to constituency household counts. The constituency
   is the politically salient rung and the rung where per-area calibrated
   weights (populace #146 long weights) later bind — exactly as the US anchors
   its ladder inside the assigned congressional district.
3. **One OA is sampled within that constituency**, proportional to 2021 Census
   OA usual-resident population (the ONS OA21 -> PCON24 best-fit lookup defines
   constituency membership).
4. **Every other layer derives from the OA** with no independent randomness:
   LSOA and MSOA by structural census nesting; local authority district, ward,
   and region by ONS Open Geography Portal best-fit lookups; ITL2/ITL1 as the
   4- and 3-character prefixes of the ITL3 code. UK geographies famously do not
   nest cleanly (constituencies can cross LAD boundaries), so the best-fit
   lookups are carried **in the ladder artifact itself**, sha-pinned, the way
   the US ladder artifact carries its block crosswalks.

One national dataset, filter by geography, at any grain (populace #275; no
per-area files, the standing rule).

Vintage discipline follows the country-spec geography-spine schema
(``vintage_policy: "error"``): the ladder artifact records one vintage per
derived layer, the loader refuses an artifact that omits any of them, and
:func:`assign_uk_geography_ladder` refuses a ladder whose constituency vintage
differs from the vintage the households' constituencies were assigned under. A
silent partial join on mismatched geography vintages is the failure these
checks exist to forbid (the #205 lesson).

Column names are policyengine-uk household *inputs* where one exists
(``region`` is the pre-assigned enum input; the ladder never overwrites it),
and plain data columns otherwise: policyengine-uk has no OA/LSOA/MSOA/ward/
constituency/ITL input variable, so those ride as ``*_code`` data columns. The
exported artifact never carries a formula output — ``country`` recomputes from
``region`` in the engine, so it is never persisted (the #34 regression).

England & Wales is the first milestone. Scotland and Northern Ireland are a
NotImplemented path, not a silent absence: a household whose region is absent
from the ladder (a Scottish household against an England-&-Wales ladder) raises
naming the missing region, never a silent partial assignment.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.gates import GateResult
from populace.build.uk_runtime.rowwise_geography import FRS_REGION_TO_REGION_CODE

#: Derived (lookup-sourced) layers the ladder artifact must carry a vintage
#: for. ITL2 and ITL1 are structural prefixes of the ITL3 code and share the
#: ITL layer's own vintage; LSOA and MSOA are the census structural nesting of
#: the OA anchor.
UK_OA_LADDER_DERIVED_LAYERS = (
    "constituency",
    "lsoa",
    "msoa",
    "local_authority",
    "ward",
    "itl",
    "region",
)

#: Household spine columns the ladder assignment writes, in write order. Every
#: name is either a policyengine-uk household input (none finer than ``region``
#: exists, and ``region`` is pre-assigned, so it is not rewritten) or a plain
#: ``*_code`` data column carrying an ONS GSS code.
UK_GEOGRAPHY_LADDER_COLUMNS = (
    "oa_code",
    "lsoa_code",
    "msoa_code",
    "local_authority_code",
    "ward_code",
    "constituency_code",
    "region_code",
    "itl3_code",
    "itl2_code",
    "itl1_code",
)

#: The nine English regions carry real ``E12`` GSS codes; Wales rides the
#: ``W92000004`` country code in ONS lookups but the FRS-calibrated region
#: dimension uses the ``W99999999`` pseudo-code, so the artifact builder
#: normalizes Welsh OAs to it (matching ``build_england_wales_crosswalk``).
#: This is the set an England-&-Wales ladder can host.
UK_ENGLAND_WALES_REGION_CODES = (
    "E12000001",  # North East
    "E12000002",  # North West
    "E12000003",  # Yorkshire and The Humber
    "E12000004",  # East Midlands
    "E12000005",  # West Midlands
    "E12000006",  # East of England
    "E12000007",  # London
    "E12000008",  # South East
    "E12000009",  # South West
    "W99999999",  # Wales
)

#: London's GSS region code — the ladder's coverage anchor for the assignment
#: summary (London holds roughly 13% of England & Wales household weight, so a
#: collapse to zero is caught the way the US ladder catches an NYC collapse).
UK_LONDON_REGION_CODE = "E12000007"

UK_OA_LADDER_SCHEMA_VERSION = 1
UK_OA_LADDER_KIND = "uk_oa_ladder"

GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR = "populace_uk_geography_ladder_artifact_sha256"
GEOGRAPHY_LADDER_VINTAGES_ATTR = "populace_uk_geography_ladder_vintages"

#: OA-grain arrays every ladder artifact must carry, aligned to ``oa_code``.
_REQUIRED_ARRAY_KEYS = (
    "oa_code",
    "population",
    "households",
    "constituency_code",
    "region_code",
    "lsoa_code",
    "msoa_code",
    "local_authority_code",
    "ward_code",
    "itl3_code",
)

_GSS_CODE_PATTERN = re.compile(r"^[ENSWK]\d{8}$")
_ITL_CODE_PATTERN = re.compile(r"^TL[A-Z](\d)?(\d)?$")


@dataclass(frozen=True)
class UkOaLadder:
    """The national OA ladder: one row per populated 2021 census output area.

    Attributes:
        oa_code: 2021 output-area GSS codes (``E00``/``W00`` for England &
            Wales), unique, as a unicode array.
        population: 2021 Census usual-resident population per OA — the
            within-constituency (stage-two) sampling weight, strictly positive
            (an OA with no residents cannot host a household and is excluded at
            artifact build time).
        households: 2021 Census household count per OA — summed to the
            constituency to form the within-region (stage-one) sampling weight,
            non-negative and finite.
        constituency_code: 2024 Westminster parliamentary constituency GSS code
            (``PCON24``) the OA best-fits into.
        region_code: GSS region code (``E12`` for England, ``W99999999`` for
            Wales) — the calibrated dimension the household arrives with.
        lsoa_code: 2021 lower-layer super output area GSS code (structural).
        msoa_code: 2021 middle-layer super output area GSS code (structural).
        local_authority_code: local authority district GSS code (April 2023
            boundaries, the vintage the long-weights solver keys on).
        ward_code: electoral ward GSS code (best-fit lookup).
        itl3_code: ITL3 code (``TL``-prefixed); ITL2 and ITL1 are its 4- and
            3-character prefixes.
        metadata: the artifact's embedded metadata, including one vintage per
            derived layer.
    """

    oa_code: np.ndarray
    population: np.ndarray
    households: np.ndarray
    constituency_code: np.ndarray
    region_code: np.ndarray
    lsoa_code: np.ndarray
    msoa_code: np.ndarray
    local_authority_code: np.ndarray
    ward_code: np.ndarray
    itl3_code: np.ndarray
    metadata: Mapping[str, Any]

    def __len__(self) -> int:
        return len(self.oa_code)

    @property
    def layer_vintages(self) -> dict[str, str]:
        """Vintage per derived layer, plus the OA anchor's own vintage."""
        layers = self.metadata["layers"]
        vintages = {
            layer: str(layers[layer]["vintage"])
            for layer in UK_OA_LADDER_DERIVED_LAYERS
        }
        vintages["oa"] = str(self.metadata["oa_vintage"])
        return vintages


def load_uk_oa_ladder(path: str | Path) -> UkOaLadder:
    """Load and validate a UK OA-ladder artifact (NPZ).

    The artifact is a single national file (never per-area files) built by
    ``tools/build_uk_oa_ladder_artifact.py`` from primary ONS/Nomis sources.
    Every validation failure raises — a ladder that loads is a ladder every
    assignment invariant holds for.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"UK OA ladder artifact not found: {source}")
    with np.load(source, allow_pickle=False) as payload:
        missing = [key for key in _REQUIRED_ARRAY_KEYS if key not in payload.files]
        if missing:
            raise ValueError(
                f"UK OA ladder artifact is missing required key(s): {missing}."
            )
        if "metadata_json" not in payload.files:
            raise ValueError(
                "UK OA ladder artifact is missing required key 'metadata_json'."
            )
        arrays = {key: np.asarray(payload[key]) for key in _REQUIRED_ARRAY_KEYS}
        metadata = _metadata_from_scalar(payload["metadata_json"])

    _validate_ladder_metadata(metadata)
    oa = _gss_code_array(arrays["oa_code"], label="oa_code")
    n = len(oa)
    if n == 0:
        raise ValueError("UK OA ladder artifact has zero output areas.")
    for key in _REQUIRED_ARRAY_KEYS:
        if len(arrays[key]) != n:
            raise ValueError(
                f"UK OA ladder array {key!r} has length {len(arrays[key])}, "
                f"expected {n} (aligned to oa_code)."
            )
    unique_oa = np.unique(oa)
    if len(unique_oa) != n:
        raise ValueError(
            f"oa_code values must be unique; {n - len(unique_oa)} duplicate row(s)."
        )

    population = np.asarray(arrays["population"], dtype=np.float64)
    if not np.isfinite(population).all() or (population <= 0).any():
        raise ValueError(
            "population must be positive and finite for every OA; "
            "unpopulated output areas belong outside the artifact."
        )
    households = np.asarray(arrays["households"], dtype=np.float64)
    if not np.isfinite(households).all() or (households < 0).any():
        raise ValueError("households must be non-negative and finite for every OA.")

    constituency = _gss_code_array(
        arrays["constituency_code"], label="constituency_code"
    )
    region = _gss_code_array(arrays["region_code"], label="region_code")
    lsoa = _gss_code_array(arrays["lsoa_code"], label="lsoa_code")
    msoa = _gss_code_array(arrays["msoa_code"], label="msoa_code")
    local_authority = _gss_code_array(
        arrays["local_authority_code"], label="local_authority_code"
    )
    ward = _gss_code_array(arrays["ward_code"], label="ward_code")
    itl3 = _itl_code_array(arrays["itl3_code"], label="itl3_code")

    # Every region that hosts households must carry at least one household to
    # sample: a region whose OAs sum to zero households cannot seat a
    # constituency draw (the stage-one weight would be undefined).
    region_households = pd.Series(households).groupby(pd.Series(region)).sum()
    empty_regions = sorted(region_households[region_households <= 0].index.tolist())
    if empty_regions:
        raise ValueError(
            f"region(s) {empty_regions} carry zero household weight; the "
            "stage-one constituency draw would be undefined."
        )

    return UkOaLadder(
        oa_code=oa,
        population=population,
        households=households,
        constituency_code=constituency,
        region_code=region,
        lsoa_code=lsoa,
        msoa_code=msoa,
        local_authority_code=local_authority,
        ward_code=ward,
        itl3_code=itl3,
        metadata=metadata,
    )


def assign_uk_geography_ladder(
    household: pd.DataFrame,
    ladder: UkOaLadder,
    *,
    seed: int = 0,
    expected_constituency_vintage: str | None = None,
    region_column: str = "region",
) -> pd.DataFrame:
    """Assign each household one OA and the derived ladder columns.

    Two seeded draws under the build's seed discipline: a 2024 constituency is
    sampled within the household's calibrated region proportional to
    constituency household counts, then an OA is sampled within that
    constituency proportional to 2021 Census OA population. Every finer and
    coarser layer then derives from the OA, so the calibrated region marginal
    is preserved exactly while every grain becomes filterable.

    Requires region assignment to have run first (the FRS carries it). A
    household region absent from the ladder is an error, never a silent partial
    join — for an England-&-Wales ladder that is exactly how a Scottish or
    Northern Irish household is refused until those rungs are pinned.
    """

    if region_column not in household.columns:
        raise ValueError(
            f"household table must contain {region_column!r} before geography-"
            "ladder assignment (assign regions first)."
        )
    if expected_constituency_vintage is not None:
        ladder_vintage = ladder.layer_vintages["constituency"]
        if ladder_vintage != expected_constituency_vintage:
            raise ValueError(
                f"UK OA ladder constituency vintage {ladder_vintage!r} does not "
                "match the vintage household constituencies were assigned under "
                f"({expected_constituency_vintage!r})."
            )

    region_codes = _household_region_codes(household[region_column], label=region_column)
    ladder_regions = set(np.unique(ladder.region_code).tolist())
    missing_regions = sorted(set(region_codes.tolist()) - ladder_regions)
    if missing_regions:
        raise ValueError(
            "UK OA ladder has no output areas for household region(s): "
            f"{missing_regions}. The household regions and the ladder artifact "
            "must share coverage (Scotland and Northern Ireland rungs are not "
            "in an England-&-Wales ladder)."
        )

    assigned_index = _sample_oa_indices(
        region_codes.to_numpy(),
        ladder=ladder,
        seed=seed,
    )

    assigned_region = ladder.region_code[assigned_index]
    mismatched = assigned_region != region_codes.to_numpy()
    if mismatched.any():
        examples = [
            f"region {want} -> OA region {got}"
            for want, got in zip(
                region_codes.to_numpy()[mismatched][:5].tolist(),
                assigned_region[mismatched][:5].tolist(),
                strict=True,
            )
        ]
        raise ValueError(
            "Assigned OA region disagrees with household region "
            f"(sampling is inconsistent): {examples}."
        )

    itl3 = ladder.itl3_code[assigned_index]
    assigned = household.copy()
    assigned["oa_code"] = ladder.oa_code[assigned_index].astype(object)
    assigned["lsoa_code"] = ladder.lsoa_code[assigned_index].astype(object)
    assigned["msoa_code"] = ladder.msoa_code[assigned_index].astype(object)
    assigned["local_authority_code"] = ladder.local_authority_code[
        assigned_index
    ].astype(object)
    assigned["ward_code"] = ladder.ward_code[assigned_index].astype(object)
    assigned["constituency_code"] = ladder.constituency_code[assigned_index].astype(
        object
    )
    assigned["region_code"] = assigned_region.astype(object)
    assigned["itl3_code"] = itl3.astype(object)
    assigned["itl2_code"] = _itl_prefix(itl3, width=4)
    assigned["itl1_code"] = _itl_prefix(itl3, width=3)
    return assigned


def uk_geography_ladder_assignment_summary(
    household: pd.DataFrame,
    ladder: UkOaLadder,
    *,
    weight_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize ladder assignment coverage for provenance logs.

    Weighted shares use household weights when provided (the spot-check surface
    for comparing filtered grains against ONS population/household totals);
    otherwise unweighted row shares.
    """

    applied = all(column in household.columns for column in UK_GEOGRAPHY_LADDER_COLUMNS)
    weights = (
        np.asarray(weight_values, dtype=np.float64)
        if weight_values is not None
        else np.ones(len(household), dtype=np.float64)
    )
    if len(weights) != len(household):
        raise ValueError(
            f"weight_values length {len(weights)} does not match household "
            f"rows {len(household)}."
        )
    summary: dict[str, Any] = {
        "applied": applied,
        "household_rows": int(len(household)),
        "ladder_output_areas": int(len(ladder)),
        "layer_vintages": ladder.layer_vintages,
        "constituency_sampling_basis": str(
            ladder.metadata["constituency_sampling_basis"]
        ),
        "oa_sampling_basis": str(ladder.metadata["oa_sampling_basis"]),
    }
    if not applied:
        return summary
    total = float(weights.sum())
    for column in UK_GEOGRAPHY_LADDER_COLUMNS:
        values = household[column].astype(str).to_numpy()
        nonempty = values != ""
        summary[f"assigned_{column}_values"] = int(
            pd.Series(values[nonempty]).nunique()
        )
        summary[f"{column}_nonempty_weighted_share"] = (
            float(weights[nonempty].sum() / total) if total > 0 else 0.0
        )
    region = household["region_code"].astype(str).to_numpy()
    london = region == UK_LONDON_REGION_CODE
    summary["london_weighted_household_share"] = (
        float(weights[london].sum() / total) if total > 0 else 0.0
    )
    return summary


def uk_geography_ladder_gate(
    household: pd.DataFrame,
    weight_values: np.ndarray,
    *,
    region_column: str = "region",
    london_share_bounds: tuple[float, float] = (0.08, 0.20),
    ward_share_bounds: tuple[float, float] = (0.98, 1.0),
    itl_share_bounds: tuple[float, float] = (0.98, 1.0),
    constituency_min_share: float = 0.99,
) -> GateResult:
    """Gate the exported geography ladder: structure, coverage, and London mass.

    The structural checks are the permanent form of the #205 partial-join risk:
    every derived layer must be present and consistent with the OA anchor's
    nesting (ITL2/ITL1 are prefixes of ITL3). The London mass check is the UK
    analogue of the US ladder's NYC check — recomputing region-conditioned
    quantities must never silently collapse the capital, which holds roughly
    13% of England & Wales household weight, so the default bounds fail on
    collapse (0%) and on gross misassignment, not on calibration noise. Ward,
    ITL, and constituency coverage are ~100% in England & Wales because every
    populated OA best-fits into all three.
    """

    failures: list[str] = []
    details: dict[str, object] = {}
    weights = np.asarray(weight_values, dtype=np.float64)
    if len(weights) != len(household):
        raise ValueError(
            f"weight_values length {len(weights)} does not match household "
            f"rows {len(household)}."
        )

    missing_columns = [
        column
        for column in (*UK_GEOGRAPHY_LADDER_COLUMNS, region_column)
        if column not in household.columns
    ]
    if missing_columns:
        failures.append(
            f"household table is missing geography column(s): {missing_columns}"
        )
        return GateResult(
            name="uk_geography_ladder",
            passed=False,
            failures=tuple(failures),
            details=details,
        )

    for label, pattern in (
        ("oa_code", _GSS_CODE_PATTERN),
        ("lsoa_code", _GSS_CODE_PATTERN),
        ("msoa_code", _GSS_CODE_PATTERN),
        ("local_authority_code", _GSS_CODE_PATTERN),
        ("ward_code", _GSS_CODE_PATTERN),
        ("constituency_code", _GSS_CODE_PATTERN),
        ("region_code", _GSS_CODE_PATTERN),
    ):
        values = household[label].astype(str).to_numpy()
        bad = np.array([bool(value) and pattern.match(value) is None for value in values])
        if bad.any():
            failures.append(
                f"{label}: {int(bad.sum())}/{len(values)} row(s) are not valid "
                f"GSS codes; examples {sorted(set(values[bad]))[:5]}"
            )

    itl3 = household["itl3_code"].astype(str).to_numpy()
    itl2 = household["itl2_code"].astype(str).to_numpy()
    itl1 = household["itl1_code"].astype(str).to_numpy()
    for label, values, expected in (
        ("itl2_code", itl2, _itl_prefix(itl3, width=4)),
        ("itl1_code", itl1, _itl_prefix(itl3, width=3)),
    ):
        mismatched = values != expected
        if mismatched.any():
            failures.append(
                f"{label}: {int(mismatched.sum())}/{len(values)} row(s) disagree "
                "with the ITL3 prefix"
            )

    region = household["region_code"].astype(str).to_numpy()
    unknown_region = np.array(
        [value not in UK_ENGLAND_WALES_REGION_CODES for value in region]
    )
    if unknown_region.any():
        failures.append(
            f"region_code: {int(unknown_region.sum())}/{len(region)} row(s) are "
            "not England & Wales region codes; "
            f"examples {sorted(set(region[unknown_region]))[:5]}"
        )

    total = float(weights.sum())
    if total <= 0:
        failures.append("household weights sum to zero; shares are undefined")
        return GateResult(
            name="uk_geography_ladder",
            passed=False,
            failures=tuple(failures),
            details=details,
        )

    london_mask = region == UK_LONDON_REGION_CODE
    london_share = float(weights[london_mask].sum() / total)
    details["london_weighted_household_share"] = london_share
    low, high = london_share_bounds
    if not low <= london_share <= high:
        failures.append(
            f"London weighted household share {london_share:.4f} outside "
            f"[{low}, {high}] (the region-collapse regression)"
        )

    for label, bounds in (
        ("ward_code", ward_share_bounds),
        ("itl3_code", itl_share_bounds),
    ):
        values = household[label].astype(str).to_numpy()
        share = float(weights[values != ""].sum() / total)
        details[f"{label}_nonempty_weighted_share"] = share
        low, high = bounds
        if not low <= share <= high:
            failures.append(
                f"{label} nonempty weighted share {share:.4f} outside [{low}, {high}]"
            )

    constituency = household["constituency_code"].astype(str).to_numpy()
    constituency_share = float(weights[constituency != ""].sum() / total)
    details["constituency_nonempty_weighted_share"] = constituency_share
    if constituency_share < constituency_min_share:
        failures.append(
            f"constituency nonempty weighted share {constituency_share:.4f} below "
            f"{constituency_min_share}"
        )

    return GateResult(
        name="uk_geography_ladder",
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )


def _sample_oa_indices(
    region_codes: np.ndarray,
    *,
    ladder: UkOaLadder,
    seed: int,
) -> np.ndarray:
    """Two-stage seeded OA draw: constituency within region, then OA within it.

    Stage one samples a constituency per household proportional to constituency
    household counts (summed OA household counts) within the household's region;
    stage two samples an OA within the chosen (region, constituency) group
    proportional to OA population. Draws are consumed in a fixed sorted group
    order, so the result is reproducible from ``seed`` alone.
    """

    rng = np.random.default_rng(seed)
    n = len(region_codes)
    assigned_index = np.full(n, -1, dtype=np.int64)

    frame = pd.DataFrame(
        {
            "constituency_code": ladder.constituency_code,
            "region_code": ladder.region_code,
            "population": ladder.population,
            "households": ladder.households,
            "ladder_index": np.arange(len(ladder), dtype=np.int64),
        }
    )
    household_positions = pd.Series(np.arange(n, dtype=np.int64))

    for region_code, region_positions in household_positions.groupby(
        pd.Series(region_codes), sort=True
    ):
        region_rows = frame[frame["region_code"] == region_code]
        # Stage one: constituency household-count weights within the region.
        constituency_weight = (
            region_rows.groupby("constituency_code", sort=True)["households"]
            .sum()
        )
        constituencies = constituency_weight.index.to_numpy()
        weights = constituency_weight.to_numpy(dtype=np.float64)
        if weights.sum() <= 0:
            raise ValueError(
                f"region {region_code!r} has zero household weight for the "
                "constituency draw."
            )
        positions = region_positions.to_numpy()
        chosen_constituency = rng.choice(
            constituencies,
            size=len(positions),
            replace=True,
            p=weights / weights.sum(),
        )
        # Stage two: OA population weights within each chosen constituency.
        chosen_series = pd.Series(chosen_constituency)
        for constituency_code, local_positions in pd.Series(positions).groupby(
            chosen_series, sort=True
        ):
            oa_rows = region_rows[
                region_rows["constituency_code"] == constituency_code
            ]
            oa_indices = oa_rows["ladder_index"].to_numpy()
            oa_weights = oa_rows["population"].to_numpy(dtype=np.float64)
            drawn = rng.choice(
                oa_indices,
                size=len(local_positions),
                replace=True,
                p=oa_weights / oa_weights.sum(),
            )
            assigned_index[local_positions.to_numpy()] = drawn

    if (assigned_index < 0).any():
        raise ValueError("internal error: some households received no OA draw.")
    return assigned_index


def _metadata_from_scalar(value: np.ndarray) -> dict[str, Any]:
    if value.shape != ():
        raise ValueError("metadata_json must be a scalar JSON string.")
    raw = value.item()
    if isinstance(raw, bytes):
        raw = raw.decode()
    if not isinstance(raw, str):
        raise ValueError("metadata_json must be a scalar JSON string.")
    metadata = json.loads(raw)
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to an object.")
    return metadata


def _validate_ladder_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("schema_version") != UK_OA_LADDER_SCHEMA_VERSION:
        raise ValueError(
            "UK OA ladder metadata schema_version must be "
            f"{UK_OA_LADDER_SCHEMA_VERSION}, got {metadata.get('schema_version')!r}."
        )
    if metadata.get("kind") != UK_OA_LADDER_KIND:
        raise ValueError(
            f"UK OA ladder metadata kind must be {UK_OA_LADDER_KIND!r}, "
            f"got {metadata.get('kind')!r}."
        )
    if not str(metadata.get("oa_vintage") or ""):
        raise ValueError("UK OA ladder metadata must record oa_vintage.")
    for basis in ("constituency_sampling_basis", "oa_sampling_basis"):
        if not str(metadata.get(basis) or ""):
            raise ValueError(f"UK OA ladder metadata must record {basis}.")
    layers = metadata.get("layers")
    if not isinstance(layers, Mapping):
        raise ValueError("UK OA ladder metadata must carry a layers mapping.")
    missing = [layer for layer in UK_OA_LADDER_DERIVED_LAYERS if layer not in layers]
    if missing:
        raise ValueError(
            f"UK OA ladder metadata layers are missing: {missing}. Every derived "
            "layer records its own vintage (vintage_policy: error)."
        )
    for layer in UK_OA_LADDER_DERIVED_LAYERS:
        spec = layers[layer]
        if not isinstance(spec, Mapping) or not str(spec.get("vintage") or ""):
            raise ValueError(
                f"UK OA ladder layer {layer!r} must record a non-empty vintage."
            )
        if not str(spec.get("source") or ""):
            raise ValueError(
                f"UK OA ladder layer {layer!r} must record a non-empty source "
                "citation."
            )


def _gss_code_array(values: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in ("U", "S", "O"):
        raise ValueError(
            f"{label} must be a string array, got dtype {array.dtype}."
        )
    array = array.astype("U", copy=False)
    stripped = np.char.strip(array)
    blank = stripped == ""
    if blank.any():
        raise ValueError(f"{label} contains {int(blank.sum())} blank code(s).")
    bad = np.array(
        [_GSS_CODE_PATTERN.match(value) is None for value in stripped.tolist()]
    )
    if bad.any():
        raise ValueError(
            f"{label} codes must be 9-character GSS codes; invalid: "
            f"{sorted(set(stripped[bad].tolist()))[:5]}."
        )
    return stripped


def _itl_code_array(values: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in ("U", "S", "O"):
        raise ValueError(f"{label} must be a string array, got dtype {array.dtype}.")
    array = array.astype("U", copy=False)
    stripped = np.char.strip(array)
    blank = stripped == ""
    if blank.any():
        raise ValueError(f"{label} contains {int(blank.sum())} blank code(s).")
    bad = np.array(
        [_ITL_CODE_PATTERN.match(value) is None for value in stripped.tolist()]
    )
    if bad.any():
        raise ValueError(
            f"{label} codes must be ``TL``-prefixed ITL3 codes; invalid: "
            f"{sorted(set(stripped[bad].tolist()))[:5]}."
        )
    return stripped


def _itl_prefix(itl3: np.ndarray, *, width: int) -> np.ndarray:
    return np.array(
        [value[:width] for value in itl3.astype("U").tolist()], dtype=object
    )


def _household_region_codes(values: Any, *, label: str) -> pd.Series:
    series = pd.Series(values).reset_index(drop=True)
    if series.isna().any():
        raise ValueError(f"{label} contains missing values.")
    mapped = series.map(_region_code_for_value)
    if mapped.isna().any():
        bad = series[mapped.isna()].astype(str).unique().tolist()
        raise ValueError(
            f"{label} contains unrecognised UK region value(s): {bad[:5]}."
        )
    return mapped


def _region_code_for_value(value: Any) -> str | None:
    text = str(value).strip()
    if not text:
        return None
    # Already a GSS region/country code (E12..., W99999999, S99999999, ...).
    if _GSS_CODE_PATTERN.match(text):
        return text
    key = text.upper().replace("-", "_").replace(" ", "_")
    # Collapse enum-value spellings such as "Yorkshire and the Humber".
    if key not in FRS_REGION_TO_REGION_CODE:
        key = _REGION_VALUE_ALIASES.get(text.strip().upper(), key)
    return FRS_REGION_TO_REGION_CODE.get(key)


#: policyengine-uk ``Region`` enum *values* (human labels) that differ from the
#: enum member names ``FRS_REGION_TO_REGION_CODE`` keys on.
_REGION_VALUE_ALIASES = {
    "YORKSHIRE AND THE HUMBER": "YORKSHIRE",
    "EAST OF ENGLAND": "EAST_OF_ENGLAND",
}


__all__ = [
    "GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR",
    "GEOGRAPHY_LADDER_VINTAGES_ATTR",
    "UK_ENGLAND_WALES_REGION_CODES",
    "UK_GEOGRAPHY_LADDER_COLUMNS",
    "UK_LONDON_REGION_CODE",
    "UK_OA_LADDER_DERIVED_LAYERS",
    "UK_OA_LADDER_KIND",
    "UK_OA_LADDER_SCHEMA_VERSION",
    "UkOaLadder",
    "assign_uk_geography_ladder",
    "load_uk_oa_ladder",
    "uk_geography_ladder_assignment_summary",
    "uk_geography_ladder_gate",
]
