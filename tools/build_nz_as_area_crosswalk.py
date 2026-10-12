"""Build MC9's centroid/population bridge from hash-pinned local Stats NZ pages.

No network or country engine is used. Supply the published staged bytes using
``--sources DIR`` or ``NZ_AS_AREA_SOURCES=DIR``. The spec contains the input
pins; receipts in DIR are never trusted as replacement pins. Example::

    python tools/build_nz_as_area_crosswalk.py --sources .hub-sources \
        --out packages/microcosm-build/src/microcosm/build/nz/as_area_crosswalk.json \
        --report .hub-scratch/as-area-report.json

AU membership uses staged SA1 interior points, otherwise centroids; this is an
approximation to a population overlay. TA membership uses the official meshblock
concordance. Queenstown's Frankton East extent remains a separate sensitivity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SA1_CODE = "SA12023_V1_00"
AU_CODE = "AU2017_V1_00"
AU_NAME = "AU2017_V1_00_NAME"
TA_CODE = "TA2025_V1_00"
TA_NAME = "TA2025_V1_00_NAME"
CONCORDANCE_SA1_CODE = "SA12026_V1_00"
CONCORDANCE_TA_CODE = "TA2026_V1_00"
CONCORDANCE_TA_NAME = "TA2026_V1_00_NAME"
WATER_AU_PREFIXES = ("Inlet", "Inlets", "Inland Water", "Oceanic")
POPULATION_VARIABLE = "VAR_1_3"
POPULATION_LAYER = "2023_Census_totals_by_topic_for_individuals_by_SA1"
CENSUS_DATES = {"VAR_1_3": "2023-03-07", "VAR_1_2": "2018-03-06"}
EDGE_EPSILON = 1e-10
INTERIOR_POINT_TOLERANCE = 0.00002
EARTH_RADIUS_M = 6371008.8
DEFAULT_SPEC = (
    Path(__file__).resolve().parents[1]
    / "packages/microcosm-build/src/microcosm/build/nz/as_area_crosswalk.json"
)


def _parts(geometry):
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    raise ValueError("Only Polygon and MultiPolygon geometries are supported")


def _ring_status(points, ring):
    """Return -1 outside, 0 on an edge, 1 inside, using an even-odd ray."""
    vertices = np.asarray(ring, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 4:
        raise ValueError("A ring requires at least four two-dimensional vertices")
    if not np.array_equal(vertices[0], vertices[-1]):
        raise ValueError("Polygon ring is not closed")
    a, b = vertices[:-1], vertices[1:]
    span = float(vertices[:, 1].max() - vertices[:, 1].min())
    if len(a) > 512 and len(points) > 32 and span > 0:
        # Latitude bands retain every edge that can cross a point's ray or
        # contain it. This avoids an enormous points-by-coastline matrix.
        lower = float(vertices[:, 1].min())
        band_ids = np.clip(((points[:, 1] - lower) / span * 128).astype(int), 0, 127)
        result = np.full(len(points), -1, dtype=np.int8)
        edge_low, edge_high = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
        for band in np.unique(band_ids):
            indices = np.flatnonzero(band_ids == band)
            band_low = lower + span * band / 128 - EDGE_EPSILON
            band_high = lower + span * (band + 1) / 128 + EDGE_EPSILON
            edges = (edge_high >= band_low) & (edge_low <= band_high)
            if edges.any():
                result[indices] = _edge_status(points[indices], a[edges], b[edges])
        return result
    return _edge_status(points, a, b)


def _edge_status(points, a, b):
    dx, dy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
    length = np.hypot(dx, dy)
    result = np.full(len(points), -1, dtype=np.int8)
    chunk = max(1, 500000 // len(a))
    for offset in range(0, len(points), chunk):
        group = points[offset : offset + chunk]
        x, y = group[:, 0, None], group[:, 1, None]
        cross = (x - a[:, 0]) * dy - (y - a[:, 1]) * dx
        edge = (
            (np.abs(cross) <= EDGE_EPSILON * length)
            & (x >= np.minimum(a[:, 0], b[:, 0]) - EDGE_EPSILON)
            & (x <= np.maximum(a[:, 0], b[:, 0]) + EDGE_EPSILON)
            & (y >= np.minimum(a[:, 1], b[:, 1]) - EDGE_EPSILON)
            & (y <= np.maximum(a[:, 1], b[:, 1]) + EDGE_EPSILON)
        ).any(axis=1)
        # Horizontal edges cannot cross the ray; avoid dividing them by zero.
        intersection = a[:, 0] + (y - a[:, 1]) * dx / np.where(dy == 0, 1, dy)
        crosses = ((a[:, 1] > y) != (b[:, 1] > y)) & (x < intersection)
        inside = np.count_nonzero(crosses, axis=1) % 2 == 1
        result[offset : offset + len(group)] = np.where(
            edge, 0, np.where(inside, 1, -1)
        )
    return result


def _geometry_status(points, geometry):
    # ArcGIS GeoJSON can place an enclave's hole in a separate MultiPolygon
    # part (e.g. AU500208 encloses AU500100). Apply even-odd parity across all
    # rings of one feature; disjoint parts have the same result as their union.
    inside = np.zeros(len(points), dtype=bool)
    edge = np.zeros(len(points), dtype=bool)
    for rings in _parts(geometry):
        for ring in rings:
            vertices = np.asarray(ring, dtype=float)
            candidates = np.flatnonzero(
                (
                    (points >= vertices.min(axis=0) - EDGE_EPSILON)
                    & (points <= vertices.max(axis=0) + EDGE_EPSILON)
                ).all(axis=1)
            )
            if len(candidates):
                values = _ring_status(points[candidates], ring)
                inside[candidates] ^= values == 1
                edge[candidates] |= values == 0
    return np.where(edge, 0, np.where(inside, 1, -1))


def point_in_geometry(point, geometry):
    """Boundary-inclusive even-odd membership, including holes and multipolygons."""
    return bool(_geometry_status(np.asarray([point], dtype=float), geometry)[0] >= 0)


def interior_point(geometry, tolerance=INTERIOR_POINT_TOLERANCE):
    """Return a deterministic strictly interior pole-of-inaccessibility point.

    Best-first subdivision maximizes distance from every ring boundary in
    coordinate degrees, within ``tolerance`` (default 0.00002 degrees). Each
    polygon part's bounding box starts a search cell; even-odd membership is
    evaluated across all rings, including holes encoded as multipart rings.
    Cells use the signed boundary distance plus their half diagonal as an
    upper bound. The tolerance only stops refinement after a strictly interior
    candidate exists, so narrow polygons never fall back to a boundary point.
    Coordinate ordering breaks ties without randomness.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Interior-point tolerance must be finite and positive")
    rings, bounds = [], []
    for part in _parts(geometry):
        if not part:
            raise ValueError("Polygon parts must contain a ring")
        vertices_in_part = []
        for ring in part:
            vertices = np.asarray(ring, dtype=float)
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 2
                or len(vertices) < 4
                or not np.isfinite(vertices).all()
                or not np.array_equal(vertices[0], vertices[-1])
            ):
                raise ValueError("Interior-point rings must be finite and closed")
            rings.append(vertices)
            vertices_in_part.append(vertices)
        vertices = np.concatenate(vertices_in_part)
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        if (high > low).all():
            bounds.append((low, high))
    has_area = any(
        np.sum(
            (ring[:-1, 0] - ring[0, 0]) * (ring[1:, 1] - ring[0, 1])
            - (ring[1:, 0] - ring[0, 0]) * (ring[:-1, 1] - ring[0, 1])
        )
        != 0
        for ring in rings
    )
    if not bounds or not has_area:
        raise ValueError("Geometry has no two-dimensional polygon interior")
    a = np.concatenate([ring[:-1] for ring in rings])
    b = np.concatenate([ring[1:] for ring in rings])
    delta = b - a
    norm = (delta * delta).sum(axis=1)
    queue = []
    best_point, best_distance = None, 0.0

    def add_cell(x, y, half_width, half_height):
        nonlocal best_point, best_distance
        point = np.asarray([x, y])
        fraction = np.clip(
            ((point - a) * delta).sum(axis=1) / np.where(norm == 0, 1, norm),
            0,
            1,
        )
        distance = float(
            np.linalg.norm(a + fraction[:, None] * delta - point, axis=1).min()
        )
        status = int(_geometry_status(point[None, :], geometry)[0])
        distance = distance if status == 1 else -distance if status == -1 else 0.0
        candidate = (float(x), float(y))
        if status == 1 and (
            best_point is None
            or distance > best_distance
            or (distance == best_distance and candidate < best_point)
        ):
            best_point, best_distance = candidate, distance
        upper = distance + math.hypot(half_width, half_height)
        heapq.heappush(queue, (-upper, x, y, half_width, half_height))

    for low, high in bounds:
        center, half_size = (low + high) / 2, (high - low) / 2
        add_cell(*center, *half_size)
    while queue:
        negative_upper, x, y, half_width, half_height = heapq.heappop(queue)
        if best_point is not None and -negative_upper <= best_distance + tolerance:
            break
        half_width, half_height = half_width / 2, half_height / 2
        if x - half_width == x or y - half_height == y:
            continue
        for dx in (-half_width, half_width):
            for dy in (-half_height, half_height):
                add_cell(x + dx, y + dy, half_width, half_height)
    if best_point is None:
        raise ValueError("Geometry has no strictly interior point")
    if _geometry_status(np.asarray([best_point]), geometry)[0] != 1:
        raise ValueError("Interior-point search did not produce a strict interior")
    return best_point


def _vertices(geometry):
    return np.concatenate(
        [np.asarray(ring, dtype=float) for part in _parts(geometry) for ring in part]
    )


def _boundary_distance(point, geometry):
    """Local equirectangular distance to every ring segment, in metres."""
    scale = np.array([math.cos(math.radians(point[1])), 1])
    scale *= math.pi / 180 * EARTH_RADIUS_M
    best = math.inf
    for rings in _parts(geometry):
        for ring in rings:
            vertices = (np.asarray(ring, dtype=float) - point) * scale
            a, b = vertices[:-1], vertices[1:]
            delta = b - a
            norm = (delta * delta).sum(axis=1)
            fraction = np.clip(
                -(a * delta).sum(axis=1) / np.where(norm == 0, 1, norm), 0, 1
            )
            best = min(
                best, float(np.linalg.norm(a + fraction[:, None] * delta, axis=1).min())
            )
    return best


def assign_points(
    points,
    polygons,
    code_field,
    *,
    max_distance_m=20000,
    populations=None,
    max_overlap_m=0,
):
    """Assign exactly once; refuse overlaps and distant populated misses.

    All polygon hits are inspected. Interior overlaps refuse by default. A
    declared max_overlap_m permits a sliver within that distance of every
    candidate boundary, choosing the uniquely deepest interior and reporting
    all distances. Shared boundaries use ascending code and are reported.
    No hits use the nearest boundary within max_distance_m;
    supplying a zero population explicitly permits a more distant assignment.
    Equal nearest distances refuse rather than select the first polygon.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Points must be a finite Nx2 array")
    ordered = sorted(polygons, key=lambda p: p["properties"][code_field])
    codes = [p["properties"][code_field] for p in ordered]
    if not ordered or len(codes) != len(set(codes)):
        raise ValueError("Polygon codes must be nonempty and unique")
    hits = [[] for _ in points]
    bounds = []
    for feature in ordered:
        vertices = _vertices(feature["geometry"])
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        bounds.append((low, high))
        candidates = np.flatnonzero(
            ((points >= low - EDGE_EPSILON) & (points <= high + EDGE_EPSILON)).all(
                axis=1
            )
        )
        values = _geometry_status(points[candidates], feature["geometry"])
        for index, status in zip(candidates, values, strict=True):
            if status >= 0:
                hits[index].append((feature["properties"][code_field], int(status)))
    assigned = []
    diagnostics = {
        "no_polygon": [],
        "multiple_boundary_hits": [],
        "multiple_interior_hits": [],
    }
    geometry_by_code = {
        feature["properties"][code_field]: feature["geometry"] for feature in ordered
    }
    low = np.asarray([pair[0] for pair in bounds])
    high = np.asarray([pair[1] for pair in bounds])
    for index, matches in enumerate(hits):
        if len(matches) > 1:
            if any(status == 1 for _, status in matches):
                distances = sorted(
                    [
                        (
                            code,
                            _boundary_distance(points[index], geometry_by_code[code]),
                        )
                        for code, _ in matches
                    ],
                    key=lambda pair: (-pair[1], pair[0]),
                )
                if (
                    max_overlap_m <= 0
                    or distances[0][1] > max_overlap_m
                    or abs(distances[0][1] - distances[1][1]) <= 1e-6
                ):
                    raise ValueError(
                        f"Point {index} is in multiple polygons: {matches}; boundary distances {distances}"
                    )
                diagnostics["multiple_interior_hits"].append(
                    {
                        "index": index,
                        "candidate_boundary_distances_m": dict(sorted(distances)),
                        "assigned_code": distances[0][0],
                    }
                )
                matches = [(distances[0][0], 1)]
            else:
                diagnostics["multiple_boundary_hits"].append(
                    {
                        "index": index,
                        "candidate_codes": [code for code, _ in matches],
                        "assigned_code": matches[0][0],
                    }
                )
        if matches:
            assigned.append(matches[0][0])
            continue
        point = points[index]
        scale = np.array([math.cos(math.radians(point[1])), 1])
        scale *= math.pi / 180 * EARTH_RADIUS_M
        delta = np.maximum(np.maximum(low - point, point - high), 0) * scale
        lower_distance = np.linalg.norm(delta, axis=1)
        best, nearest = math.inf, []
        for candidate in np.argsort(lower_distance, kind="stable"):
            if lower_distance[candidate] > best + 1e-6:
                break
            distance = _boundary_distance(point, ordered[candidate]["geometry"])
            if distance < best - 1e-6:
                best, nearest = distance, [codes[candidate]]
            elif abs(distance - best) <= 1e-6:
                nearest.append(codes[candidate])
        if len(nearest) != 1:
            raise ValueError(f"Point {index} has tied nearest polygons: {nearest}")
        if best > max_distance_m and (populations is None or populations[index] != 0):
            raise ValueError(
                f"Point {index} misses every polygon; nearest is {best:.3f} m, beyond {max_distance_m} m"
            )
        assigned.append(nearest[0])
        diagnostics["no_polygon"].append(
            {"index": index, "assigned_code": nearest[0], "distance_m": best}
        )
    return assigned, diagnostics


def population_counts(features, variable=POPULATION_VARIABLE):
    """Exclude nulls and negative sentinel codes, never add them as counts.

    The layer description names -997 (Not available) and -999 (Confidential).
    Other negative codes are also excluded and enumerated for review.
    Non-integral, boolean or nonnumeric counts refuse.
    """
    values, excluded = [], []
    for index, feature in enumerate(features):
        value = feature["attributes"][variable]
        if value is None or (type(value) is int and value < 0):
            excluded.append(
                {
                    "index": index,
                    "sa1_code": feature["attributes"].get(SA1_CODE),
                    "value": value,
                }
            )
            values.append(0)
        elif type(value) is not int:
            raise ValueError(
                f"{variable}: population must be an integer, null or negative sentinel"
            )
        else:
            values.append(value)
    return np.asarray(values, dtype=np.int64), {
        "variable": variable,
        "excluded": excluded,
        "count": len(excluded),
    }


def exact_area_mapping(
    definitions,
    au_features,
    au_population=None,
    definition_name_aliases=None,
    resolved_definition_entries=None,
):
    """Match full names or declared code aliases; validate every resolution."""
    au_population = au_population or {}
    by_name = defaultdict(list)
    by_code = {}
    assigned = {}
    for feature in au_features:
        properties = feature["properties"]
        by_name[properties[AU_NAME]].append(properties[AU_CODE])
        by_code[properties[AU_CODE]] = properties[AU_NAME]
        assigned[properties[AU_CODE]] = 4
    aliases = {}
    definition_names = Counter(
        name
        for area in ("1", "2", "3")
        for location in definitions["areas"][area]["locations"]
        for name in location["statistical_area_units_2017"]
    )
    for alias in definition_name_aliases or []:
        name, code = alias["definition_name"], alias["au_code"]
        if code not in by_code or by_code[code] != alias["au_name"]:
            raise ValueError(f"Alias code/name does not match AU layer: {alias}")
        if name in aliases or definition_names[name] != 1 or not alias["evidence"]:
            raise ValueError(
                f"Alias must identify one definition with evidence: {alias}"
            )
        aliases[name] = code
    resolved = set()
    for entry in resolved_definition_entries or []:
        key = (entry["name"], entry["heading"], entry["as_area"])
        if entry["au_codes"] or not entry["evidence"] or key in resolved:
            raise ValueError(f"Unsupported or duplicate definition resolution: {entry}")
        resolved.add(key)
    seen_resolutions = set()
    unmatched = []
    for area in ("1", "2", "3"):
        for location in definitions["areas"][area]["locations"]:
            for name in location["statistical_area_units_2017"]:
                key = (name, location["heading"], int(area))
                if key in resolved:
                    seen_resolutions.add(key)
                    continue
                candidates = (
                    [aliases[name]] if name in aliases else sorted(by_name[name])
                )
                if len(candidates) == 1:
                    assigned[candidates[0]] = int(area)
                else:
                    unmatched.append(
                        {
                            "name": name,
                            "heading": location["heading"],
                            "as_area": int(area),
                            "reason": "no exact match"
                            if not candidates
                            else "multiple exact matches",
                            "candidates": [
                                {"code": code, "population": au_population.get(code, 0)}
                                for code in candidates
                            ],
                        }
                    )
    if seen_resolutions != resolved:
        raise ValueError(
            f"Resolution does not match definition: {resolved - seen_resolutions}"
        )
    return assigned, unmatched


def concordance_ta_codes(sa1_features, ta_features, meshblock_features):
    """Use the official SA1-to-TA mapping, refusing incomplete/inconsistent data."""
    by_sa1 = defaultdict(set)
    ta_set = set()
    for feature in meshblock_features:
        attributes = feature["attributes"]
        pair = (attributes[CONCORDANCE_TA_CODE], attributes[CONCORDANCE_TA_NAME])
        by_sa1[attributes[CONCORDANCE_SA1_CODE]].add(pair)
        ta_set.add(pair)
    expected = {
        (f["properties"][TA_CODE], f["properties"][TA_NAME]) for f in ta_features
    }
    if ta_set != expected:
        raise ValueError(
            f"TA-set mismatch: missing {sorted(expected - ta_set)}; extra {sorted(ta_set - expected)}"
        )
    multiple = {
        code: sorted(pairs) for code, pairs in by_sa1.items() if len(pairs) != 1
    }
    if multiple:
        raise ValueError(f"SA1 with multiple TAs in concordance: {multiple}")
    codes = [f["attributes"][SA1_CODE] for f in sa1_features]
    missing = sorted(set(codes) - by_sa1.keys())
    if missing:
        raise ValueError(f"Missing SA1 codes in concordance: {missing}")
    return [next(iter(by_sa1[code]))[0] for code in codes]


def _shares(counts):
    total = sum(counts)
    if total == 0:
        raise ValueError("A population share requires a positive total")
    best = max(counts)
    winners = [area for area, value in enumerate(counts, 1) if value == best]
    if len(winners) != 1:
        raise ValueError(f"tied modal areas: {winners}")
    return [
        {"as_area": area, "population_share": value / total}
        for area, value in enumerate(counts, 1)
    ], winners[0]


def aggregate_rows(
    ta_features,
    sa1_features,
    au_codes,
    ta_codes,
    area_by_au,
    variable=POPULATION_VARIABLE,
):
    populations, _ = population_counts(sa1_features, variable)
    totals = {feature["properties"][TA_CODE]: [0, 0, 0, 0] for feature in ta_features}
    if len(totals) != len(ta_features):
        raise ValueError("Duplicate territorial authority code")
    for population, au, ta in zip(populations, au_codes, ta_codes, strict=True):
        totals[ta][area_by_au[au] - 1] += int(population)
    rows = []
    for feature in sorted(ta_features, key=lambda f: f["properties"][TA_CODE]):
        properties = feature["properties"]
        counts = totals[properties[TA_CODE]]
        if sum(counts) == 0:
            if properties[TA_CODE] != "999":
                raise ValueError(f"TA {properties[TA_CODE]} has zero usable population")
            shares, modal = _shares([0, 0, 0, 1])
        else:
            shares, modal = _shares(counts)
        rows.append(
            {
                "ta_code": properties[TA_CODE],
                "ta_name": properties[TA_NAME],
                "shares": shares,
                "population_source": f"Stats NZ {POPULATION_LAYER}; {variable}; Census {CENSUS_DATES[variable]}; SA1 2023 interior/centroid AU bridge; Meshblock Higher Geographies 2026 TA concordance",
                "population_total": sum(counts),
                "modal_area": modal,
            }
        )
    return rows


def _load_sources(source_dir, provenance):
    groups = {"sa1": [], "au": [], "ta": [], "mbhg": [], "sa1poly": []}
    # Verify every file before interpreting any geographic or population data.
    pins = list(provenance["pages"]) + [
        {"file": layer["metadata_file"], "sha256": layer["metadata_sha256"]}
        for layer in provenance["layers"]
    ]
    payloads = {}
    for pin in pins:
        path = source_dir / pin["file"]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != pin["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {pin['file']}")
        if "bytes" in pin and len(payload) != pin["bytes"]:
            raise ValueError(f"Byte count mismatch: {pin['file']}")
        payloads[pin["file"]] = payload
    for pin in provenance["pages"]:
        document = json.loads(payloads[pin["file"]])
        if len(document["features"]) != pin["features"]:
            raise ValueError(f"Feature count mismatch: {pin['file']}")
        group = pin["file"].split("-")[0]
        groups[group].extend(document["features"])
    census = next(
        layer for layer in provenance["layers"] if layer["name"] == POPULATION_LAYER
    )
    metadata = json.loads(payloads[census["metadata_file"]])
    aliases = {field["name"]: field["alias"] for field in metadata["fields"]}
    for variable, date in CENSUS_DATES.items():
        alias = aliases[variable]
        if (
            "Census usually resident population count" not in alias
            or f"Year: {date[:4]}" not in alias
        ):
            raise ValueError(f"Unexpected Census field alias: {variable}: {alias}")
    for group, code_field in (("sa1", SA1_CODE), ("au", AU_CODE), ("ta", TA_CODE)):
        attributes = "attributes" if group == "sa1" else "properties"
        groups[group].sort(key=lambda f: f[attributes][code_field])
        codes = [feature[attributes][code_field] for feature in groups[group]]
        if len(set(codes)) != len(codes):
            raise ValueError(f"Duplicate {group} codes")
    groups["sa1poly"].sort(key=lambda f: f["properties"][SA1_CODE])
    polygon_codes = [f["properties"][SA1_CODE] for f in groups["sa1poly"]]
    if len(set(polygon_codes)) != len(polygon_codes):
        raise ValueError("Duplicate SA1 polygon codes")
    return groups, {variable: aliases[variable] for variable in CENSUS_DATES}


def _assignment_report(diagnostics, features, populations):
    for kind in ("no_polygon", "multiple_boundary_hits", "multiple_interior_hits"):
        for row in diagnostics[kind]:
            index = row.pop("index")
            row.update(
                sa1_code=features[index]["attributes"][SA1_CODE],
                population=int(populations[index]),
            )
    diagnostics["no_polygon_count"] = len(diagnostics["no_polygon"])
    diagnostics["no_polygon_population"] = sum(
        row["population"] for row in diagnostics["no_polygon"]
    )
    return diagnostics


def _heading_conflicts(
    definitions, au_features, assigned_tas, expectations, definition_name_aliases=None
):
    by_name = defaultdict(list)
    for feature in au_features:
        properties = feature["properties"]
        by_name[properties[AU_NAME]].append(properties[AU_CODE])
    aliases = {
        row["definition_name"]: row["au_code"] for row in definition_name_aliases or []
    }
    conflicts, checks = [], []
    for area in ("1", "2", "3"):
        for location in definitions["areas"][area]["locations"]:
            heading = location["heading"]
            allowed = expectations[heading]
            for name in location["statistical_area_units_2017"]:
                for code in [aliases[name]] if name in aliases else by_name[name]:
                    observed = assigned_tas.get(code, {})
                    checks.append(
                        {
                            "name": name,
                            "au_code": code,
                            "heading": heading,
                            "observed_ta_population": dict(sorted(observed.items())),
                            "check": "non-geographic heading"
                            if allowed is None
                            else "no SA1 centroids; unchecked"
                            if not observed
                            else "primary TA review",
                        }
                    )
                    if allowed is None:
                        continue
                    unexpected = {
                        ta: population
                        for ta, population in sorted(observed.items())
                        if ta not in allowed
                    }
                    if unexpected:
                        conflicts.append(
                            {
                                "name": name,
                                "au_code": code,
                                "heading": heading,
                                "as_area": int(area),
                                "expected_tas": allowed,
                                "unexpected_ta_population": unexpected,
                                "all_ta_population": dict(sorted(observed.items())),
                            }
                        )
    return conflicts, checks


def generate_crosswalk(source_dir, spec, report=None):
    """Return a lean spec; fill a separate report with detailed audit evidence."""
    groups, aliases = _load_sources(Path(source_dir), spec["provenance"])
    sa1, au, ta = groups["sa1"], groups["au"], groups["ta"]
    populations, suppression = population_counts(sa1)
    points = np.asarray(
        [(feature["centroid"]["x"], feature["centroid"]["y"]) for feature in sa1]
    )
    tolerance = spec["provenance"]["assignment_rule"]["max_distance_m"]
    overlap_tolerance = spec["provenance"]["assignment_rule"].get("max_overlap_m", 0)
    old_au_codes, au_diagnostics = assign_points(
        points,
        au,
        AU_CODE,
        max_distance_m=tolerance,
        populations=populations,
        max_overlap_m=overlap_tolerance,
    )
    ta_codes = concordance_ta_codes(sa1, ta, groups["mbhg"])
    centroid_ta_codes, ta_diagnostics = assign_points(
        points,
        ta,
        TA_CODE,
        max_distance_m=tolerance,
        populations=populations,
        max_overlap_m=overlap_tolerance,
    )
    au_names = {f["properties"][AU_CODE]: f["properties"][AU_NAME] for f in au}
    indices_by_sa1 = {f["attributes"][SA1_CODE]: i for i, f in enumerate(sa1)}
    polygons_by_sa1 = {f["properties"][SA1_CODE]: f for f in groups["sa1poly"]}
    unknown = sorted(polygons_by_sa1.keys() - indices_by_sa1.keys())
    if unknown:
        raise ValueError(f"Staged polygons have unknown SA1 codes: {unknown}")
    missing_water = [
        {
            "sa1_code": f["attributes"][SA1_CODE],
            "au_code": code,
            "au_name": au_names[code],
            "population": int(population),
        }
        for f, code, population in zip(sa1, old_au_codes, populations, strict=True)
        if au_names[code].startswith(WATER_AU_PREFIXES)
        and f["attributes"][SA1_CODE] not in polygons_by_sa1
    ]
    if missing_water:
        raise ValueError(
            f"Water AU centroids require staged SA1 polygons: {json.dumps(missing_water, sort_keys=True)}"
        )
    flagged_indices = [indices_by_sa1[code] for code in sorted(polygons_by_sa1)]
    au_codes = list(old_au_codes)
    interior_points = []
    if flagged_indices:
        interior_points = [
            interior_point(
                polygons_by_sa1[sa1[i]["attributes"][SA1_CODE]]["geometry"],
                tolerance=spec["provenance"]["assignment_rule"]["interior_point"][
                    "tolerance_degrees"
                ],
            )
            for i in flagged_indices
        ]
        flagged_codes, flagged_diagnostics = assign_points(
            np.asarray(interior_points),
            au,
            AU_CODE,
            max_distance_m=tolerance,
            populations=populations[flagged_indices],
            max_overlap_m=overlap_tolerance,
        )
        for index, code in zip(flagged_indices, flagged_codes, strict=True):
            au_codes[index] = code
        for kind in ("no_polygon", "multiple_boundary_hits", "multiple_interior_hits"):
            au_diagnostics[kind] = [
                r for r in au_diagnostics[kind] if r["index"] not in flagged_indices
            ]
            for row in flagged_diagnostics[kind]:
                row["index"] = flagged_indices[row["index"]]
                au_diagnostics[kind].append(row)
    au_population = Counter()
    au_ta_population = defaultdict(Counter)
    ta_names = {
        feature["properties"][TA_CODE]: feature["properties"][TA_NAME] for feature in ta
    }
    for population, au_code, ta_code in zip(
        populations, au_codes, ta_codes, strict=True
    ):
        au_population[au_code] += int(population)
        ta_name = ta_names[ta_code]
        au_ta_population[au_code][ta_name] += int(population)
    areas, unmatched = exact_area_mapping(
        spec["area_definitions"],
        au,
        au_population,
        spec.get("definition_name_aliases"),
        spec.get("resolved_definition_entries"),
    )
    before_alias_areas, _ = exact_area_mapping(
        spec["area_definitions"],
        au,
        au_population,
        resolved_definition_entries=spec.get("resolved_definition_entries"),
    )
    listed_names = {
        name
        for area in ("1", "2", "3")
        for location in spec["area_definitions"]["areas"][area]["locations"]
        for name in location["statistical_area_units_2017"]
    }
    rows = aggregate_rows(ta, sa1, au_codes, ta_codes, areas)
    previous = aggregate_rows(ta, sa1, au_codes, ta_codes, areas, variable="VAR_1_2")
    changes = [
        (
            abs(s["population_share"] - p["population_share"]),
            row["ta_code"],
            s["as_area"],
        )
        for row, prior in zip(rows, previous, strict=True)
        for s, p in zip(row["shares"], prior["shares"], strict=True)
    ]
    largest = max(changes)
    queenstown = next(
        row for row in rows if row["ta_name"] == "Queenstown-Lakes District"
    )
    sensitivity_areas = dict(areas)
    sensitivity_codes = []
    for feature in au:
        properties = feature["properties"]
        code, name = properties[AU_CODE], properties[AU_NAME]
        if (
            areas[code] == 4
            and name not in listed_names
            and ("Frankton" in name or "Queenstown" in name)
            and "Queenstown-Lakes District" in au_ta_population.get(code, {})
        ):
            sensitivity_areas[code] = 1
            sensitivity_codes.append(
                {
                    "au_code": code,
                    "au_name": name,
                    "population": au_ta_population[code]["Queenstown-Lakes District"],
                }
            )
    # Only move SA1s assigned to the district; an AU may straddle its boundary.
    moved = sum(
        int(pop)
        for pop, ac, tc in zip(populations, au_codes, ta_codes, strict=True)
        if tc == queenstown["ta_code"] and areas[ac] != sensitivity_areas[ac]
    )
    sensitivity_counts = [
        round(s["population_share"] * queenstown["population_total"])
        for s in queenstown["shares"]
    ]
    sensitivity_counts[0] += moved
    sensitivity_counts[3] -= moved
    sensitivity_shares, sensitivity_modal = _shares(sensitivity_counts)
    document = copy.deepcopy(spec)
    document["status"] = "harvested_concordance_bridge"
    document["status_reason"] = (
        "Rows use hash-pinned Stats NZ SA1 2023 Census URP, interior points for staged SA1 polygons and centroids otherwise for AU2017 membership, and the Meshblock Higher Geographies 2026 SA1-to-TA concordance. TA2026 code/name support is verified equal to TA2025. Whole-SA1 AU allocation remains an approximation. Whakarewarewa maps to Whaka by declared code alias; Queenstown urban area resolves to no additional unit from the hub's Urban Area 2014 sampled overlay. Frankton East remains sensitivity only. Detailed diagnostics are generated separately with --report."
    )
    document["rows"] = rows
    document["row_shape"].update(
        population_total="sum of usable 2023 Census URP assigned to this TA; zero only for explicitly declared TA 999 placeholder",
        modal_area="unique largest-population AS area, for scenario S3; ties refuse",
    )
    document["unmatched_definition_entries"] = unmatched
    conflicts, checks = _heading_conflicts(
        spec["area_definitions"],
        au,
        au_ta_population,
        spec["heading_ta_expectations"],
        spec.get("definition_name_aliases"),
    )
    document.pop("definition_heading_conflicts", None)
    document.pop("definition_heading_checks", None)
    document["queenstown_sensitivity"] = {
        "label": "sensitivity only; not the central assignment rule",
        "definition": "Within Queenstown-Lakes District, otherwise unlisted AU2017 names containing Frankton or Queenstown are assigned Area 1",
        "ta_code": queenstown["ta_code"],
        "area_units": sensitivity_codes,
        "moved_population": moved,
        "moved_population_share": moved / queenstown["population_total"],
        "shares": sensitivity_shares,
        "modal_area": sensitivity_modal,
    }
    diagnostics = {
        "field_aliases": aliases,
        "sa1_count": len(sa1),
        "au_count": len(au),
        "ta_count": len(ta),
        "au_assignment": _assignment_report(au_diagnostics, sa1, populations),
        "ta_assignment": _assignment_report(ta_diagnostics, sa1, populations),
        "suppressed_values": suppression,
        "national_layer_population": int(populations.sum()),
        "assigned_population": sum(row["population_total"] for row in rows),
        "zero_population_tas": [
            row["ta_code"] for row in rows if row["population_total"] == 0
        ],
        "queenstown_areas_1_and_4_only": {
            s["as_area"] for s in queenstown["shares"] if s["population_share"] > 0
        }
        == {1, 4},
        "stability_2018": {
            "variable": "VAR_1_2",
            "census_date": CENSUS_DATES["VAR_1_2"],
            "national_layer_population": int(
                population_counts(sa1, "VAR_1_2")[0].sum()
            ),
            "suppressed_values": population_counts(sa1, "VAR_1_2")[1],
            "rows": previous,
            "largest_absolute_share_change": largest[0],
            "ta_code": largest[1],
            "as_area": largest[2],
        },
    }
    disagreements = [
        {
            "sa1_code": f["attributes"][SA1_CODE],
            "population": int(population),
            "centroid_ta_code": old,
            "centroid_ta_name": ta_names[old],
            "concordance_ta_code": new,
            "concordance_ta_name": ta_names[new],
        }
        for f, population, old, new in zip(
            sa1, populations, centroid_ta_codes, ta_codes, strict=True
        )
        if old != new
    ]
    flagged_changes = [
        {
            "sa1_code": sa1[i]["attributes"][SA1_CODE],
            "population": int(populations[i]),
            "ta_code": ta_codes[i],
            "ta_name": ta_names[ta_codes[i]],
            "old_au_code": old_au_codes[i],
            "old_au_name": au_names[old_au_codes[i]],
            "new_au_code": au_codes[i],
            "new_au_name": au_names[au_codes[i]],
            "as_area_before": before_alias_areas[old_au_codes[i]],
            "as_area_after": areas[au_codes[i]],
            "interior_point": list(point),
        }
        for i, point in zip(flagged_indices, interior_points, strict=True)
    ]
    remaining_water = [
        row
        for row in flagged_changes
        if row["new_au_name"].startswith(WATER_AU_PREFIXES)
    ]
    area_changes = defaultdict(Counter)
    for population, old, new, tc in zip(
        populations, old_au_codes, au_codes, ta_codes, strict=True
    ):
        before, after = before_alias_areas[old], areas[new]
        if before != after:
            area_changes[tc][(before, after)] += int(population)
    change_rows = [
        {
            "ta_code": code,
            "ta_name": ta_names[code],
            "changed_population": sum(counts.values()),
            "transitions": [
                {
                    "as_area_before": before,
                    "as_area_after": after,
                    "population": population,
                }
                for (before, after), population in sorted(counts.items())
            ],
        }
        for code, counts in sorted(area_changes.items())
    ]
    before_alias_rows = aggregate_rows(ta, sa1, au_codes, ta_codes, before_alias_areas)
    alias_comparison = {
        "before": next(
            (r for r in before_alias_rows if r["ta_name"] == "Rotorua District"), None
        ),
        "after": next((r for r in rows if r["ta_name"] == "Rotorua District"), None),
    }
    details = {
        "definition_heading_conflicts": conflicts,
        "definition_heading_checks": checks,
        "diagnostics": diagnostics,
        "ta_concordance_disagreements": disagreements,
        "flagged_sa1_changes": flagged_changes,
        "remaining_water_au_sa1s": remaining_water,
        "as_area_changes_by_ta": change_rows,
        "rotorua_alias_comparison": alias_comparison,
        "rows": rows,
    }
    document["diagnostics"] = {
        "sa1_count": len(sa1),
        "meshblock_count": len(groups["mbhg"]),
        "au_count": len(au),
        "ta_count": len(ta),
        "national_layer_population": diagnostics["national_layer_population"],
        "assigned_population": diagnostics["assigned_population"],
        "ta_concordance_disagreement_count": len(disagreements),
        "ta_concordance_disagreement_population": sum(
            r["population"] for r in disagreements
        ),
        "flagged_sa1_count": len(flagged_changes),
        "flagged_sa1_population": sum(r["population"] for r in flagged_changes),
        "remaining_water_au_sa1_count": len(remaining_water),
        "remaining_water_au_population": sum(r["population"] for r in remaining_water),
        "as_area_changed_population": sum(r["changed_population"] for r in change_rows),
        "definition_heading_check_count": len(checks),
        "definition_heading_conflict_count": len(conflicts),
        "au_no_polygon_count": diagnostics["au_assignment"]["no_polygon_count"],
        "au_no_polygon_population": diagnostics["au_assignment"][
            "no_polygon_population"
        ],
        "suppressed_population_count": suppression["count"],
    }
    if report is not None:
        report.clear()
        report.update(details)
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", type=Path, default=os.environ.get("NZ_AS_AREA_SOURCES")
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        help="Audit JSON path (default: beside --out, with .report.json suffix)",
    )
    args = parser.parse_args(argv)
    if args.sources is None or not args.sources.is_dir():
        print(
            "SKIPPED: supply pinned staged pages with --sources DIR or NZ_AS_AREA_SOURCES=DIR; no network fetch is performed"
        )
        return
    spec = json.loads(args.spec.read_text())
    names = [page["file"] for page in spec["provenance"]["pages"]]
    names.extend(layer["metadata_file"] for layer in spec["provenance"]["layers"])
    missing = [name for name in names if not (args.sources / name).is_file()]
    if missing:
        print(f"SKIPPED: pinned staged source set is incomplete; missing {missing[0]}")
        return
    print(
        "Verifying pinned pages and computing the concordance/interior bridge",
        flush=True,
    )
    report = {}
    document = generate_crosswalk(args.sources, spec, report)
    args.out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    report_path = args.report or args.out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "rows": len(document["rows"]),
                "assigned_population": document["diagnostics"]["assigned_population"],
                "au_misses": document["diagnostics"]["au_no_polygon_count"],
                "ta_disagreements": document["diagnostics"][
                    "ta_concordance_disagreement_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
