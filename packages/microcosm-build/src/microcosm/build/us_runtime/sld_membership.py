"""State-legislative-district membership at the 2024 boundary vintage.

The local artifact's baked ``sldu``/``sldl`` columns are 2020-BAF vintage
(2010-cycle districts) and donor-spine-only. The SLD layer (populace#625)
derives district membership itself, at the vintage the ACS 5-year targets
tabulate on (the Census 2024 SLD BEFs — district names in the witnessed ACS
responses carry the "(2024)" label), so calibration targets and membership
share one declared boundary vintage.

One uniform seeded operator, conditioning on the finest geography each row
carries:

- rows with a ``tract_geoid`` (donor spine): exact lookup where the tract
  lies wholly in one district pair, population-weighted block draw within
  the tract where split;
- rows without (ACS spine): population-weighted draw over the block
  overlap of ``(puma, congressional_district_geoid, county_fips)``, falling
  back through coarser conditioning cells only when the row's independently
  seeded geography names an empty cell, with the method recorded per row.

Both chambers are read off the same drawn overlap row, so a household's
SLDU and SLDL assignments are mutually coherent and coherent with the
geography it already carries. Draws come from one seeded generator consumed
in a fixed order, so an assignment is reproducible from its seed.

The overlap tables are assembled from one national block pass in
``tools/build_us_sld_membership_ladder_artifact.py`` (pinned sources:
NationalSLDU24/NationalSLDL24 BEFs, 2020 P.L. 94-171 block populations, the
2020 tract-to-PUMA relationship, and the 119th CD BEF — the same source
family the existing ladders pin).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "NO_LOWER_CHAMBER_STATE_FIPS",
    "SLD_ASSIGNMENT_METHODS",
    "SLD_MEMBERSHIP_COLUMNS",
    "UsSldMembershipLadder",
    "assign_us_sld_membership",
    "load_us_sld_membership_ladder",
    "parse_national_sld24_bef",
    "us_sld_membership_gate",
]

#: States with no lower legislative chamber in the Census SLD universe:
#: Nebraska's unicameral legislature and the District of Columbia (whose
#: council wards are tabulated as the upper chamber only).
NO_LOWER_CHAMBER_STATE_FIPS = frozenset({"31", "11"})

#: Census "assigned to no district" markers in SLD BEFs (water blocks).
_SLD_UNASSIGNED_MARKERS = frozenset({"ZZZ", "ZZ", ""})

#: Assignment-method vocabulary, finest conditioning first. ``tract_exact``
#: means the row's tract lies wholly in one (SLDU, SLDL) pair — no draw.
SLD_ASSIGNMENT_METHODS = (
    "tract_exact",
    "tract_split_draw",
    "puma_cd_county_draw",
    "puma_county_draw",
    "puma_cd_draw",
    "puma_draw",
    "unassigned",
)

#: Columns :func:`assign_us_sld_membership` appends.
SLD_MEMBERSHIP_COLUMNS = (
    "sld_upper_code",
    "sld_lower_code",
    "sld_assignment_method",
)


def parse_national_sld24_bef(
    lines: Iterable[str],
    *,
    chamber: str,
) -> dict[int, str]:
    """Parse a national 2024 SLD BEF into block geoid -> district code.

    Format matches the CD BEF family: comma-delimited ``GEOID,SLDUST`` (upper)
    or ``GEOID,SLDLST`` (lower). Blocks marked unassigned (``ZZZ``) are
    dropped — they carry no population that a district tabulates.
    """
    if chamber not in ("upper", "lower"):
        raise ValueError(f"chamber must be 'upper' or 'lower', got {chamber!r}.")
    expected_header = "GEOID,SLDUST" if chamber == "upper" else "GEOID,SLDLST"
    iterator = iter(lines)
    try:
        header = next(iterator).strip().lstrip("﻿")
    except StopIteration:
        raise ValueError("National SLD BEF is empty.") from None
    if header.upper() != expected_header:
        raise ValueError(
            f"National SLD BEF header must be {expected_header!r}, got {header!r}."
        )
    result: dict[int, str] = {}
    for line_number, line in enumerate(iterator, start=2):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"National SLD BEF line {line_number} must be GEOID,code, "
                f"got {stripped!r}."
            )
        geoid, code = parts[0].strip(), parts[1].strip()
        if len(geoid) != 15 or not geoid.isdigit():
            raise ValueError(
                f"National SLD BEF line {line_number} block geoid must be 15 "
                f"digits, got {geoid!r}."
            )
        if code.upper() in _SLD_UNASSIGNED_MARKERS:
            continue
        if len(code) != 3:
            raise ValueError(
                f"National SLD BEF line {line_number} district code must be "
                f"3 characters, got {code!r}."
            )
        result[int(geoid)] = code
    return result


#: The boundary vintage this module's assignment semantics target. A ladder
#: artifact must declare the same value in its embedded metadata — a stale
#: or differently-sourced NPZ with the right array names is refused, not
#: silently reported as 2024.
SLD_MEMBERSHIP_BOUNDARY_VINTAGE = "2024_state_legislative_districts"

#: The pinned source family the boundary vintage comes from.
SLD_MEMBERSHIP_SOURCE_KIND = "census_2024_sld_bef"


@dataclass(frozen=True)
class UsSldMembershipLadder:
    """Block-population overlap tables for 2024-vintage SLD membership.

    ``tract_*`` arrays hold one row per ``tract x (sldu, sldl)`` combination;
    ``cell_*`` arrays one row per ``(puma, cd, county) x (sldu, sldl)``
    combination. ``population`` is the 2020 P.L. 94-171 block population the
    combination carries; draws are proportional to it. ``boundary_vintage``
    and ``source_kind`` are the artifact's own declaration, validated at
    load against this module's constants.
    """

    tract_geoid: np.ndarray
    tract_sldu: np.ndarray
    tract_sldl: np.ndarray
    tract_population: np.ndarray
    cell_puma: np.ndarray
    cell_cd: np.ndarray
    cell_county: np.ndarray
    cell_sldu: np.ndarray
    cell_sldl: np.ndarray
    cell_population: np.ndarray
    boundary_vintage: str = SLD_MEMBERSHIP_BOUNDARY_VINTAGE
    source_kind: str = SLD_MEMBERSHIP_SOURCE_KIND

    def district_codes(self, chamber: str) -> dict[str, frozenset[str]]:
        """Observed district codes per state FIPS for one chamber."""
        if chamber == "upper":
            codes = self.cell_sldu
        elif chamber == "lower":
            codes = self.cell_sldl
        else:
            raise ValueError(f"chamber must be 'upper' or 'lower', got {chamber!r}.")
        states = (self.cell_county // 1000).astype(np.int64)
        by_state: dict[str, set[str]] = {}
        for state, code in zip(states.tolist(), codes.tolist(), strict=True):
            if code:
                by_state.setdefault(f"{state:02d}", set()).add(code)
        return {state: frozenset(codes) for state, codes in by_state.items()}


def _string_array(values: np.ndarray, *, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in ("U", "S", "O"):
        raise ValueError(f"{label} must be a string array, got {array.dtype}.")
    return array.astype(str)


def load_us_sld_membership_ladder(path: str | Path) -> UsSldMembershipLadder:
    """Load and validate the membership-ladder NPZ artifact.

    The artifact must embed ``meta_boundary_vintage`` and
    ``meta_source_kind`` matching this module's constants: array names alone
    do not prove the districts are the 2024-BEF vintage the ACS targets
    tabulate on.
    """
    with np.load(Path(path), allow_pickle=False) as arrays:
        required = (
            "tract_geoid",
            "tract_sldu",
            "tract_sldl",
            "tract_population",
            "cell_puma",
            "cell_cd",
            "cell_county",
            "cell_sldu",
            "cell_sldl",
            "cell_population",
            "meta_boundary_vintage",
            "meta_source_kind",
        )
        missing = [name for name in required if name not in arrays]
        if missing:
            raise ValueError(f"Membership ladder is missing arrays: {missing}.")
        boundary_vintage = str(np.asarray(arrays["meta_boundary_vintage"]).item())
        source_kind = str(np.asarray(arrays["meta_source_kind"]).item())
        if boundary_vintage != SLD_MEMBERSHIP_BOUNDARY_VINTAGE:
            raise ValueError(
                f"Membership ladder declares boundary vintage "
                f"{boundary_vintage!r}; this module assigns "
                f"{SLD_MEMBERSHIP_BOUNDARY_VINTAGE!r} districts."
            )
        if source_kind != SLD_MEMBERSHIP_SOURCE_KIND:
            raise ValueError(
                f"Membership ladder declares source kind {source_kind!r}; "
                f"expected {SLD_MEMBERSHIP_SOURCE_KIND!r}."
            )
        ladder = UsSldMembershipLadder(
            tract_geoid=arrays["tract_geoid"].astype(np.int64),
            tract_sldu=_string_array(arrays["tract_sldu"], label="tract_sldu"),
            tract_sldl=_string_array(arrays["tract_sldl"], label="tract_sldl"),
            tract_population=arrays["tract_population"].astype(np.int64),
            cell_puma=arrays["cell_puma"].astype(np.int64),
            cell_cd=arrays["cell_cd"].astype(np.int64),
            cell_county=arrays["cell_county"].astype(np.int64),
            cell_sldu=_string_array(arrays["cell_sldu"], label="cell_sldu"),
            cell_sldl=_string_array(arrays["cell_sldl"], label="cell_sldl"),
            cell_population=arrays["cell_population"].astype(np.int64),
            boundary_vintage=boundary_vintage,
            source_kind=source_kind,
        )
    if len(ladder.tract_geoid) == 0 or len(ladder.cell_puma) == 0:
        raise ValueError("Membership ladder tables must be non-empty.")
    if (ladder.tract_population < 0).any() or (ladder.cell_population < 0).any():
        raise ValueError("Membership ladder populations must be non-negative.")
    tract_total = int(ladder.tract_population.sum())
    cell_total = int(ladder.cell_population.sum())
    if tract_total != cell_total:
        raise ValueError(
            "Membership ladder population is not conserved across tables: "
            f"tract total {tract_total:,} vs cell total {cell_total:,}. Both "
            "tables come from the same block pass; a mismatch means a "
            "corrupted artifact."
        )
    return ladder


@dataclass(frozen=True)
class _GroupIndex:
    """Row indices of an overlap table grouped by a key tuple."""

    groups: Mapping[tuple, np.ndarray]

    @classmethod
    def build(cls, *keys: np.ndarray) -> _GroupIndex:
        frame = pd.DataFrame({i: key for i, key in enumerate(keys)})
        by = list(frame.columns) if len(keys) > 1 else frame.columns[0]
        groups = {
            key if isinstance(key, tuple) else (key,): indices.to_numpy()
            for key, indices in frame.groupby(by, sort=True).groups.items()
        }
        return cls(groups=groups)


def _draw(
    indices: np.ndarray,
    populations: np.ndarray,
    generator: np.random.Generator,
) -> int | None:
    """Population-weighted draw; ``None`` when the cell has no population.

    Draws are population-proportional by doctrine — a zero-population cell
    is not support, so the caller falls through to the next coarser
    conditioning level instead of drawing uniformly from it.
    """
    weights = populations[indices].astype(np.float64)
    total = weights.sum()
    if total <= 0:
        return None
    return int(generator.choice(indices, p=weights / total))


def assign_us_sld_membership(
    households: pd.DataFrame,
    ladder: UsSldMembershipLadder,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """Assign 2024-vintage SLDU/SLDL membership to every household row.

    Requires ``state_fips``; conditions on ``tract_geoid`` where present and
    non-null, else on ``puma`` / ``congressional_district_geoid`` /
    ``county_fips`` as available. Geography components that name a
    different state than the row's ``state_fips`` are ignored (the drawn
    district always belongs to the row's own state — district codes repeat
    across states, so a cross-state key would otherwise assign a
    plausible-looking wrong district). Zero-population conditioning cells
    fall through to the next coarser level.

    Returns a copy with the :data:`SLD_MEMBERSHIP_COLUMNS` appended. Rows
    are processed in a fixed order (stable index order) from one seeded
    generator, so results are reproducible from ``seed`` given identical
    input bytes and row order.
    """
    if "state_fips" not in households.columns:
        raise ValueError("households must carry state_fips.")
    generator = np.random.default_rng(seed)
    tract_index = _GroupIndex.build(ladder.tract_geoid)
    cell_full = _GroupIndex.build(ladder.cell_puma, ladder.cell_cd, ladder.cell_county)
    cell_puma_county = _GroupIndex.build(
        ladder.cell_puma,
        ladder.cell_county,
    )
    cell_puma_cd = _GroupIndex.build(ladder.cell_puma, ladder.cell_cd)
    cell_puma = _GroupIndex.build(ladder.cell_puma)

    def _tract_value(row: pd.Series, state: int) -> int | None:
        if "tract_geoid" not in row.index:
            return None
        value = row["tract_geoid"]
        if pd.isna(value) or value in ("", 0):
            return None
        tract = int(value)
        return tract if tract // 10**9 == state else None

    def _int_value(
        row: pd.Series,
        column: str,
        *,
        state: int,
        state_divisor: int,
    ) -> int | None:
        if column not in row.index:
            return None
        value = row[column]
        if pd.isna(value) or value in ("", 0):
            return None
        number = int(value)
        return number if number // state_divisor == state else None

    upper_codes: list[str] = []
    lower_codes: list[str] = []
    methods: list[str] = []
    for _, row in households.iterrows():
        state = int(row["state_fips"])
        tract = _tract_value(row, state)
        chosen: int | None = None
        method = "unassigned"
        if tract is not None and (tract,) in tract_index.groups:
            indices = tract_index.groups[(tract,)]
            if len(indices) == 1:
                chosen = int(indices[0])
                method = "tract_exact"
            else:
                chosen = _draw(indices, ladder.tract_population, generator)
                method = "tract_split_draw"
            if chosen is not None:
                upper_codes.append(str(ladder.tract_sldu[chosen]))
                lower_codes.append(str(ladder.tract_sldl[chosen]))
                methods.append(method)
                continue
        puma = _int_value(row, "puma", state=state, state_divisor=100_000)
        cd = _int_value(
            row,
            "congressional_district_geoid",
            state=state,
            state_divisor=100,
        )
        county = _int_value(row, "county_fips", state=state, state_divisor=1_000)
        fallbacks: list[tuple[str, _GroupIndex, tuple]] = []
        if puma is not None:
            if cd is not None and county is not None:
                fallbacks.append(("puma_cd_county_draw", cell_full, (puma, cd, county)))
            if county is not None:
                fallbacks.append(("puma_county_draw", cell_puma_county, (puma, county)))
            if cd is not None:
                fallbacks.append(("puma_cd_draw", cell_puma_cd, (puma, cd)))
            fallbacks.append(("puma_draw", cell_puma, (puma,)))
        chosen = None
        for fallback_method, index, key in fallbacks:
            if key in index.groups:
                drawn = _draw(
                    index.groups[key],
                    ladder.cell_population,
                    generator,
                )
                if drawn is None:
                    continue
                chosen = drawn
                method = fallback_method
                break
        if chosen is None:
            upper_codes.append("")
            lower_codes.append("")
            methods.append("unassigned")
        else:
            drawn_state = int(ladder.cell_county[chosen]) // 1_000
            if drawn_state != state:  # pragma: no cover - defense in depth
                raise ValueError(
                    f"drawn cell belongs to state {drawn_state:02d} for a "
                    f"state {state:02d} row; the ladder keys are corrupt."
                )
            upper_codes.append(str(ladder.cell_sldu[chosen]))
            lower_codes.append(str(ladder.cell_sldl[chosen]))
            methods.append(method)

    assigned = households.copy()
    assigned["sld_upper_code"] = upper_codes
    assigned["sld_lower_code"] = lower_codes
    assigned["sld_assignment_method"] = methods
    return assigned


@dataclass(frozen=True)
class UsSldMembershipGate:
    """Pass/fail summary for an SLD membership assignment."""

    passed: bool
    failures: tuple[str, ...]
    details: dict


def us_sld_membership_gate(
    assigned: pd.DataFrame,
    ladder: UsSldMembershipLadder,
    *,
    max_unassigned_share: float = 0.001,
) -> UsSldMembershipGate:
    """Gate an assignment: coverage, code validity, chamber expectations."""
    for column in SLD_MEMBERSHIP_COLUMNS:
        if column not in assigned.columns:
            return UsSldMembershipGate(
                passed=False,
                failures=(f"assignment is missing column {column}",),
                details={},
            )
    failures: list[str] = []
    methods = assigned["sld_assignment_method"].astype(str)
    method_counts = methods.value_counts().to_dict()
    unknown_methods = set(method_counts) - set(SLD_ASSIGNMENT_METHODS)
    if unknown_methods:
        failures.append(f"unknown assignment methods: {sorted(unknown_methods)}")
    unassigned_mask = methods == "unassigned"
    unassigned_share = float(unassigned_mask.mean())
    if "household_weight" in assigned.columns:
        weights = pd.to_numeric(assigned["household_weight"], errors="coerce").fillna(
            0.0
        )
        total_weight = float(weights.sum())
        unassigned_weight_share = (
            float(weights[unassigned_mask].sum()) / total_weight
            if total_weight > 0
            else 0.0
        )
    else:
        unassigned_weight_share = unassigned_share
    if unassigned_share > max_unassigned_share:
        failures.append(
            f"unassigned row share {unassigned_share:.4%} exceeds "
            f"{max_unassigned_share:.4%}"
        )
    if unassigned_weight_share > max_unassigned_share:
        failures.append(
            f"unassigned weight share {unassigned_weight_share:.4%} exceeds "
            f"{max_unassigned_share:.4%}"
        )

    states = assigned["state_fips"].map(lambda value: f"{int(value):02d}")
    upper_by_state = ladder.district_codes("upper")
    lower_by_state = ladder.district_codes("lower")
    invalid_upper = 0
    invalid_lower = 0
    missing_upper = 0
    missing_lower_outside_unicameral = 0
    unexpected_lower_in_unicameral = 0
    tract_degraded = 0
    has_tract = (
        assigned["tract_geoid"]
        if "tract_geoid" in assigned.columns
        else pd.Series([None] * len(assigned), index=assigned.index)
    )
    for state, upper, lower, method, tract in zip(
        states,
        assigned["sld_upper_code"].astype(str),
        assigned["sld_lower_code"].astype(str),
        methods,
        has_tract,
        strict=True,
    ):
        if method == "unassigned":
            continue
        if method.startswith("puma") and tract is not None and not pd.isna(tract):
            if tract not in ("", 0):
                tract_degraded += 1
        if not upper:
            missing_upper += 1
        elif upper not in upper_by_state.get(state, frozenset()):
            invalid_upper += 1
        if lower:
            if state in NO_LOWER_CHAMBER_STATE_FIPS:
                unexpected_lower_in_unicameral += 1
            elif lower not in lower_by_state.get(state, frozenset()):
                invalid_lower += 1
        elif state not in NO_LOWER_CHAMBER_STATE_FIPS:
            missing_lower_outside_unicameral += 1
    if missing_upper:
        failures.append(f"{missing_upper} assigned rows lack an upper-chamber code")
    if invalid_upper:
        failures.append(
            f"{invalid_upper} rows carry an upper-chamber code outside their "
            "state's district set"
        )
    if invalid_lower:
        failures.append(
            f"{invalid_lower} rows carry a lower-chamber code outside their "
            "state's district set"
        )
    if missing_lower_outside_unicameral:
        failures.append(
            f"{missing_lower_outside_unicameral} assigned rows lack a "
            "lower-chamber code outside the no-lower-chamber states"
        )
    if unexpected_lower_in_unicameral:
        failures.append(
            f"{unexpected_lower_in_unicameral} rows carry a lower-chamber "
            "code in a state with no lower chamber"
        )
    details = {
        "method_counts": {key: int(value) for key, value in method_counts.items()},
        "unassigned_share": unassigned_share,
        "unassigned_weight_share": unassigned_weight_share,
        "tract_degraded_rows": int(tract_degraded),
        "n_rows": int(len(assigned)),
    }
    return UsSldMembershipGate(
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )


def assemble_us_sld_membership_ladder(
    *,
    block_population: Mapping[int, int],
    sldu_by_block: Mapping[int, str],
    sldl_by_block: Mapping[int, str],
    cd_by_block: Mapping[int, int],
    tract_to_puma: Mapping[int, int],
) -> dict[str, np.ndarray]:
    """One block pass into the two overlap tables the ladder stores.

    The block universe is the P.L. 94-171 population file. Blocks assigned
    to no district in either chamber (water) are dropped; blocks with a
    district but no CD BEF row keep ``cd == 0`` (they still support the
    coarser fallback draws). A populated block whose tract has no PUMA
    mapping is a source-inconsistency error, never a silent drop.
    """
    tract_cells: dict[tuple[int, str, str], int] = {}
    overlap_cells: dict[tuple[int, int, int, str, str], int] = {}
    n_blocks = 0
    n_no_cd = 0
    n_dropped_blocks = 0
    dropped_population = 0
    for block, population in block_population.items():
        sldu = sldu_by_block.get(block, "")
        sldl = sldl_by_block.get(block, "")
        if not sldu and not sldl:
            n_dropped_blocks += 1
            dropped_population += int(population)
            continue
        tract = block // 10**4
        county = int(block // 10**10)
        puma = tract_to_puma.get(tract)
        if puma is None:
            if population == 0:
                continue
            raise ValueError(
                f"Block {block} (population {population}) has a district "
                "assignment but its tract has no PUMA mapping; the pinned "
                "sources are inconsistent."
            )
        cd = int(cd_by_block.get(block, 0))
        if cd == 0:
            n_no_cd += 1
        n_blocks += 1
        tract_key = (int(tract), str(sldu), str(sldl))
        tract_cells[tract_key] = tract_cells.get(tract_key, 0) + int(population)
        cell_key = (int(puma), cd, county, str(sldu), str(sldl))
        overlap_cells[cell_key] = overlap_cells.get(cell_key, 0) + int(population)
    if not tract_cells:
        raise ValueError("No district-assigned blocks; sources are empty.")
    tract_rows = sorted(tract_cells.items())
    cell_rows = sorted(overlap_cells.items())
    return {
        "tract_geoid": np.array([key[0] for key, _ in tract_rows], dtype=np.int64),
        "tract_sldu": np.array([key[1] for key, _ in tract_rows], dtype="U3"),
        "tract_sldl": np.array([key[2] for key, _ in tract_rows], dtype="U3"),
        "tract_population": np.array(
            [population for _, population in tract_rows], dtype=np.int64
        ),
        "cell_puma": np.array([key[0] for key, _ in cell_rows], dtype=np.int64),
        "cell_cd": np.array([key[1] for key, _ in cell_rows], dtype=np.int64),
        "cell_county": np.array([key[2] for key, _ in cell_rows], dtype=np.int64),
        "cell_sldu": np.array([key[3] for key, _ in cell_rows], dtype="U3"),
        "cell_sldl": np.array([key[4] for key, _ in cell_rows], dtype="U3"),
        "cell_population": np.array(
            [population for _, population in cell_rows], dtype=np.int64
        ),
        "n_assigned_blocks": np.array([n_blocks], dtype=np.int64),
        "n_blocks_without_cd": np.array([n_no_cd], dtype=np.int64),
        "n_dropped_blocks": np.array([n_dropped_blocks], dtype=np.int64),
        "dropped_population": np.array([dropped_population], dtype=np.int64),
        "meta_boundary_vintage": np.array(SLD_MEMBERSHIP_BOUNDARY_VINTAGE),
        "meta_source_kind": np.array(SLD_MEMBERSHIP_SOURCE_KIND),
    }


def write_membership_provenance(
    path: str | Path,
    *,
    sources: dict,
    totals: dict,
) -> None:
    """Record pinned-source and national-total provenance beside the NPZ."""
    payload = {"sources": sources, "totals": totals}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
