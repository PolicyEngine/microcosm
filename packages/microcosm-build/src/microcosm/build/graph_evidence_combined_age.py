"""Pinned aggregate-only evidence for combined-survey national age calibration.

Execute by file path: importing this module never imports an execution engine.
The externally pinned root binding authorizes input JSON identities, not a run.
Checkpoint bytes, source stores and record arrays are never opened.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_combined_age_evidence_base", Path(__file__).with_name("graph_evidence.py")
)
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)
EvidenceError = _base.EvidenceError
require, digest, count = _base.require, _base.digest, _base.count
identifier, safe_text = _base.identifier, _base.safe_text
canonical_json, sha256 = _base.canonical_json, _base.sha256

PROFILE = "us_combined_asec_acs_national_age_v1"
BINDING_SCHEMA = "microcosm.us_combined_age_evidence_binding.v1"
IDS = (
    "composed_population.prepare",
    "composed_population",
    "assign_us_puma_ladder.lookup",
    "assign_us_puma_ladder.boundary",
    "assign_us_puma_ladder",
    "assign_us_puma_ladder.gate",
    "composed_population.asec_bind",
    "composed_population.asec_cps_carried_current",
    "composed_population.asec_reported_income",
    "national.age_counts",
    "national.demographic_calibration",
)
LABELS = (
    "Prepare ASEC and ACS sources",
    "Compose whole survey households",
    "Load geography crosswalk",
    "Resolve geography boundary",
    "Assign geography",
    "Check geography integrity",
    "Bind ASEC source records",
    "Carry current income",
    "Account for reported income",
    "Count observed ages",
    "Calibrate national ages",
)
ROLES = (
    "composition_acceptance",
    "composition_run",
    "composition_cold_receipt",
    "composition_warm_receipt",
    "composition_cold_manifest",
    "composition_warm_manifest",
    "calibration_acceptance",
    "calibration_run",
    "calibration_extension_receipt",
    "calibration_replay_receipt",
    "calibration_extension_manifest",
    "calibration_replay_manifest",
    "materialization_receipt",
    "aggregate_verification",
    "checkpoint_readback",
    "diagnostics",
    "calibrated_manifest",
)
IDENTITIES = (
    "composition_head",
    "calibration_head",
    "composition_runner_sha256",
    "calibration_runner_sha256",
    "materializer_head",
    "materializer_sha256",
    "exporter_sha256",
    "base_exporter_sha256",
    "writer_sha256",
)
BANDS = tuple(
    (f"S0101_C01_{i + 2:03}", f"people_age_{i * 5}_{i * 5 + 4}", i * 5, i * 5 + 4)
    for i in range(17)
) + (("S0101_C01_019", "people_age_85_plus", 85, None),)
SUPPORT = (
    "supported_households",
    "unweighted_people_in_band",
    "maximum_band_count_in_a_household",
)
CELL_FIELDS = (
    "published_value",
    "published_se",
    "compiled_target",
    "initial_estimate",
    "final_estimate",
    "initial_deviation",
    "final_deviation",
    "initial_relative_deviation",
    "final_relative_deviation",
)
REASON = "Local development age-calibration experiment; launch validation remains incomplete."
MISSING = [
    "Group-quarters representativeness",
    "Sex and household-reference mapping",
    "Source observation-date harmonization",
    "Housing transfer",
    "PUF enrichment",
    "National launch validation",
    "Congressional-district validation",
    "Pruned analysis",
]
CONVENTION = "max(1, ceil(0.01 * all household rows))"


def number(value, *, nonnegative=False):
    require(
        type(value) in (int, float) and math.isfinite(value),
        "Nonfinite or nonnumeric aggregate.",
    )
    require(not nonnegative or value >= 0, "Negative aggregate.")
    return value


def close(left, right, label):
    require(
        math.isclose(number(left), number(right), rel_tol=1e-9, abs_tol=1e-7), label
    )


def tree(value, depth=0, *, public=False):
    require(depth <= 24, "JSON nesting limit.")
    if isinstance(value, dict):
        require(len(value) <= (1024 if public else 20000), "JSON map limit.")
        for key, child in value.items():
            require(isinstance(key, str) and len(key) <= 1024, "JSON key limit.")
            tree(child, depth + 1, public=public)
    elif isinstance(value, list):
        require(len(value) <= (4096 if public else 20000), "JSON list limit.")
        for child in value:
            tree(child, depth + 1, public=public)
    elif isinstance(value, str):
        if public:
            safe_text(value)
        else:
            require(len(value) <= 65536, "JSON string limit.")
    elif value is not None and type(value) is not bool:
        number(value)


def read_json(path):
    require(
        Path(path).suffix == ".json",
        "Only explicit aggregate JSON inputs are accepted.",
    )
    raw = _base._file_bytes(Path(path), maximum=16 * 1024 * 1024)

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key.")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise EvidenceError("Invalid aggregate JSON.") from exc
    tree(value)
    require(isinstance(value, dict), "Expected JSON object.")
    return value, raw


def roster(rows, key, expected):
    require(
        isinstance(rows, list) and len(rows) == len(expected),
        "Exact target roster required.",
    )
    require(
        all(isinstance(row, dict) and key in row for row in rows),
        "Malformed target row.",
    )
    result = {row[key]: row for row in rows}
    require(
        len(result) == len(rows) and set(result) == set(expected),
        "Missing or duplicate target.",
    )
    return result


def trajectory_summary(values):
    require(
        isinstance(values, list) and 1 <= len(values) <= 4096,
        "Recorded trajectory required.",
    )
    for value in values:
        number(value, nonnegative=True)
    decreasing = sum(
        right <= left for left, right in zip(values, values[1:], strict=False)
    )
    tail = values[-min(10, len(values)) :]
    return dict(
        epochs_recorded=len(values),
        first=values[0],
        last=values[-1],
        minimum=min(values),
        argmin_epoch=values.index(min(values)),
        monotone_nonincreasing_steps=decreasing,
        total_steps=len(values) - 1,
        monotone_nonincreasing=decreasing == len(values) - 1,
        last10_relative_change=None
        if tail[0] == 0
        else (tail[-1] - tail[0]) / abs(tail[0]),
    )


def verify_manifest(manifest, ids, keys, implementations, hits):
    require(
        manifest.get("schema_version") == 3 and manifest.get("country") == "us",
        "Manifest schema.",
    )
    nodes = manifest["nodes"]
    require(set(nodes) == set(ids), "Exact node roster required.")
    content = {
        "nodes": {
            k: {f: v for f, v in n.items() if f not in ("hit", "wall_time")}
            for k, n in nodes.items()
        },
        "tier": manifest["tier"],
    }
    require(
        manifest["content_addressed"] == content,
        "Manifest content projection mismatch.",
    )
    require(
        manifest["key"] == sha256(b"manifest\0" + canonical_json(content)),
        "Manifest content key mismatch.",
    )
    require(
        manifest["tier"] is None and manifest.get("decisions") == [],
        "Release decisions are outside scope.",
    )
    require(
        set(manifest.get("known_failures", [])) <= {"assign_us_puma_ladder.gate"},
        "Unexpected known failures.",
    )
    public = []
    for node_id in ids:
        n = nodes[node_id]
        require(digest(n["key"]) == keys[node_id], "Node key mismatch.")
        require(
            digest(n["kernel_impl_hash"]) == implementations[node_id],
            "Implementation mismatch.",
        )
        require(
            type(n["hit"]) is bool and n["hit"] == hits[node_id],
            "Cache provenance mismatch.",
        )
        caps, receipt = n["capabilities"], n["receipt"]
        require(
            caps["consumes_se"] is False and caps["tolerance"] is None,
            "Unsupported SE or tolerance contract.",
        )
        typed = n.get("typed_artifacts", {"inputs": {}, "outputs": {}})
        inputs, outputs = typed["inputs"], typed["outputs"]
        for name, edge in outputs.items():
            require(
                edge["producer"] == node_id and edge["producer_key"] == n["key"],
                "Typed output producer mismatch.",
            )
            require(
                edge["artifact"] == name and edge["key"] == n["opaque_artifacts"][name],
                "Typed output identity mismatch.",
            )
        for edge in inputs.values():
            producer = edge["producer"]
            require(
                producer in nodes and producer != node_id, "Unknown typed producer."
            )
            require(
                ids.index(producer) < ids.index(node_id), "Non-topological typed edge."
            )
            require(
                edge
                == nodes[producer]["typed_artifacts"]["outputs"].get(edge["artifact"]),
                "Typed input provenance mismatch.",
            )
        outcome = receipt.get("outcome")
        failed = outcome == "fail"
        require(
            not failed or node_id in manifest["known_failures"],
            "Unacknowledged failed gate.",
        )
        require(outcome in (None, "pass", "fail"), "Unsupported receipt outcome.")
        counts = Counter(identifier(a["entity"]) for a in n["artifacts"])
        gate = None
        if caps["role"] == "gate":
            gate = {
                "outcome": outcome,
                "scope": identifier(receipt["scope"]),
                "passed": receipt["evidence"]["passed"],
                "checks": [
                    {
                        "name": identifier(k),
                        "passed": v["passed"],
                        "failures": [safe_text(t) for t in v["failures"]],
                        "details": [
                            {"key": identifier(dk), "value": number(dv)}
                            for dk, dv in v.get("details", {}).items()
                        ],
                    }
                    for k, v in receipt["evidence"]["gates"].items()
                ],
            }
            require(gate["passed"] is (outcome == "pass"), "Gate outcome mismatch.")
        public.append(
            dict(
                id=node_id,
                label=LABELS[IDS.index(node_id)],
                group="calibration" if node_id in IDS[-2:] else "composition",
                group_label="National age calibration"
                if node_id in IDS[-2:]
                else "Survey composition",
                role=identifier(caps["role"]),
                receipt_present=True,
                failed=failed,
                blocked=False,
                gate_exception=False,
                execution_record=None,
                receipt_outcome=outcome,
                cache="hit" if n["hit"] else "computed",
                node_key=n["key"],
                kernel_ref=identifier(n["kernel_ref"], kernel=True),
                kernel_impl_hash=n["kernel_impl_hash"],
                determinism=identifier(caps["determinism"]),
                numeric=identifier(caps["numeric"]),
                tolerance=None,
                runtime_dependencies=[identifier(x) for x in caps["dependencies"]],
                seed=str(n["seed"]),
                wall_time_seconds=_base.duration(n["wall_time"]),
                frame_key=digest(n["frame_key"]) if n["frame_key"] else None,
                weight_key=digest(n["weight_key"]) if n["weight_key"] else None,
                phase=identifier(receipt.get("phase", node_id)),
                typed_inputs=[
                    _base._typed(v, k, is_input=True) for k, v in inputs.items()
                ],
                typed_outputs=[
                    _base._typed(v, k, is_input=False) for k, v in outputs.items()
                ],
                opaque_artifact_keys=[
                    {"name": identifier(k), "key": digest(v)}
                    for k, v in n["opaque_artifacts"].items()
                ],
                column_artifact_counts=[
                    {"entity": k, "count": v} for k, v in sorted(counts.items())
                ],
                column_artifact_total=sum(counts.values()),
                gate=gate,
                source_pins=[],
                measures=[],
            )
        )
    return public


def verify_cells(result, accepted, aggregate, diagnostics, households, persons):
    expected = [b[0] for b in BANDS]
    rows = roster(result["cells"], "cell", expected)
    require(
        result["cells"] == accepted["cells"] == aggregate["cells"],
        "Accepted cell reports disagree.",
    )
    support = roster(aggregate["independently_recomputed_support"], "cell", expected)
    activation = result["activation"]
    require(
        activation["activated_cell_count"]
        == activation["standard_errors_present"]
        == 18
        and activation["standard_errors_consumed_by_loss"] is False,
        "Activation scope mismatch.",
    )
    active = roster(activation["cells"], "name", expected)
    targets = roster(diagnostics["targets"], "target_name", expected)
    require(
        diagnostics["target_surface"]["n_targets"] == 18
        and diagnostics["target_surface"]["n_records"] == households,
        "Diagnostics target surface mismatch.",
    )
    published = []
    for cell, measure, lower, upper in BANDS:
        row, s, a, t = rows[cell], support[cell], active[cell], targets[cell]
        require(
            row["measure"]
            == s["measure"]
            == a["measure"]
            == t["measure"]["name"]
            == measure,
            "Age measure mismatch.",
        )
        require(
            t["measure"]["kind"] == "column"
            and t["name"] == cell + "@2024"
            and a["period"] == t["period"] == "2024"
            and a["entity"] == t["entity"] == "household"
            and a["family"] == t["registry"]["family"] == "acs.S0101",
            "Target identity mismatch.",
        )
        for f in CELL_FIELDS:
            number(
                row[f],
                nonnegative=f
                not in (
                    "initial_deviation",
                    "final_deviation",
                    "initial_relative_deviation",
                    "final_relative_deviation",
                ),
            )
        require(
            row["published_value"] > 0
            and row["published_value"]
            == row["compiled_target"]
            == a["value"]
            == t["target"]
            == t["compiled_target"],
            "Published and compiled targets disagree.",
        )
        require(
            row["published_se"] == a["se"] == t["registry"]["se"],
            "Published SE mismatch.",
        )
        require(
            row["within_tolerance"] is None and t["within_tolerance"] is None,
            "Undeclared tolerance.",
        )
        for phase in ("initial", "final"):
            close(
                row[f"{phase}_estimate"],
                t[f"{phase}_estimate"],
                "Diagnostics estimate mismatch.",
            )
            close(
                row[f"{phase}_deviation"],
                row[f"{phase}_estimate"] - row["compiled_target"],
                "Deviation arithmetic mismatch.",
            )
            close(
                row[f"{phase}_relative_deviation"],
                row[f"{phase}_deviation"] / row["compiled_target"],
                "Relative deviation arithmetic mismatch.",
            )
        close(
            row["final_relative_deviation"],
            t["relative_error"],
            "Diagnostics relative error mismatch.",
        )
        close(
            row["reported_relative_error"],
            t["relative_error"],
            "Reported relative error mismatch.",
        )
        for f in SUPPORT:
            require(count(row[f]) == count(s[f]), "Independent support mismatch.")
        h, p, m = (row[f] for f in SUPPORT)
        require(
            0 <= h <= households
            and h <= p <= persons
            and m <= p
            and (h == p == m == 0 or 1 <= m and m + h - 1 <= p <= h * m),
            "Invalid support arithmetic.",
        )
        if h == 0:
            require(
                row["initial_estimate"] == row["final_estimate"] == 0,
                "Zero-support estimate must remain zero.",
            )
        published.append(
            dict(
                cell=cell,
                measure=measure,
                label=f"Age {lower}+" if upper is None else f"Ages {lower}–{upper}",
                minimum_age=lower,
                maximum_age=upper,
                **{k: row[k] for k in (*CELL_FIELDS, *SUPPORT)},
                support_status="supported" if h else "zero_support",
                within_tolerance=None,
            )
        )
    require(
        sum(row["unweighted_people_in_band"] for row in published) == persons,
        "Age bands do not partition persons.",
    )
    close(
        sum(row["published_value"] for row in published),
        activation["published_partition_total"],
        "Published partition mismatch.",
    )
    return published


def verify_weights(result, accepted, aggregate, diagnostics, households):
    public = {}
    for role, field in (
        ("initial_importance", "importance_weights"),
        ("calibrated", "calibrated_weights"),
    ):
        w, c = result[field], aggregate["weight_concentration"][role]
        require(w == accepted[field], "Accepted weight diagnostics disagree.")
        require(
            count(w["records"]) == count(c["records"]) == households,
            "Weight record count mismatch.",
        )
        require(
            count(w["positive_records"]) + count(w["zero_records"]) == households,
            "Weight sign counts mismatch.",
        )
        for key in ("total", "minimum", "maximum", "maximum_weight_share"):
            number(w[key], nonnegative=True)
        require(
            0 < w["total"] and w["minimum"] <= w["maximum"] <= w["total"],
            "Weight extrema mismatch.",
        )
        close(
            w["maximum"] / w["total"],
            w["maximum_weight_share"],
            "Maximum share arithmetic mismatch.",
        )
        require(
            (w["minimum"] == 0) is (w["zero_records"] > 0)
            and w["minimum"] <= w["total"] / households <= w["maximum"] * (1 + 1e-12),
            "Weight extrema or zero-count arithmetic mismatch.",
        )
        close(w["total"], c["sum_weights"], "Weight sum mismatch.")
        close(
            w["maximum_weight_share"],
            c["maximum_weight_share"],
            "Maximum share mismatch.",
        )
        require(
            number(c["sum_squared_weights"], nonnegative=True) > 0,
            "Empty squared weight sum.",
        )
        ess = c["sum_weights"] ** 2 / c["sum_squared_weights"]
        close(ess, c["weight_concentration_ess"], "ESS arithmetic mismatch.")
        close(
            ess / households,
            c["weight_concentration_ess_fraction"],
            "ESS fraction mismatch.",
        )
        require(
            0 < ess <= w["positive_records"] + 1e-7,
            "ESS outside positive record bound.",
        )
        require(
            c["top_1pct_count"] == max(1, math.ceil(0.01 * households))
            and c["top_1pct_convention"] == CONVENTION,
            "Top one-percent convention mismatch.",
        )
        require(
            w["maximum_weight_share"] <= number(c["top_1pct_weight_share"]) <= 1,
            "Top weight share out of range.",
        )
        public[role] = {
            **{
                k: w[k]
                for k in (
                    "records",
                    "positive_records",
                    "zero_records",
                    "total",
                    "minimum",
                    "maximum",
                    "maximum_weight_share",
                )
            },
            "effective_sample_size": ess,
            "effective_sample_size_fraction": ess / households,
            **{
                k: c[k]
                for k in (
                    "top_1pct_count",
                    "top_1pct_weight_share",
                    "top_1pct_convention",
                )
            },
        }
    final = public["calibrated"]
    # Solver float32 diagnostics may differ slightly from independent float64 sums.
    require(
        math.isclose(
            final["effective_sample_size"],
            diagnostics["effective_sample_size"],
            rel_tol=2e-6,
        ),
        "Solver ESS mismatch.",
    )
    require(
        diagnostics["n_records"] == households
        and diagnostics["n_nonzero"] == final["positive_records"],
        "Solver weight counts mismatch.",
    )
    solver = accepted["solver"]
    close(
        solver["effective_sample_size"],
        diagnostics["effective_sample_size"],
        "Accepted solver ESS mismatch.",
    )
    close(
        solver["top_1pct_weight_share"],
        diagnostics["top_1pct_weight_share"],
        "Accepted solver top weight share mismatch.",
    )
    require(
        solver["n_records"] == households
        and solver["n_nonzero"] == final["positive_records"],
        "Accepted solver record counts mismatch.",
    )
    for field in (
        "effective_sample_size",
        "effective_sample_size_fraction",
        "top_1pct_weight_share",
    ):
        require(
            math.isclose(
                number(solver[field]), final[field], rel_tol=2e-6, abs_tol=1e-10
            ),
            "Accepted solver concentration mismatch.",
        )
    return public


def _export_combined_age_evidence(
    *, binding_path: Path, binding_sha256: str, output_root: Path
):
    """Verify pinned aggregate JSON inputs and seal candidate attachments; no index."""
    binding, raw = read_json(binding_path)
    require(sha256(raw) == digest(binding_sha256), "Root binding digest mismatch.")
    _base.obj(
        binding,
        {
            "schema",
            "verdict",
            "profile",
            "invented_sources",
            "candidate_release_id",
            "run_id",
            "created_at",
            "files",
            "identities",
            "node_keys",
            "implementation_hashes",
            "source_vintages",
        },
    )
    require(
        binding["schema"] == BINDING_SCHEMA
        and binding["verdict"] == "APPROVE"
        and binding["profile"] == PROFILE,
        "Root export approval required.",
    )
    require(type(binding["invented_sources"]) is bool, "Explicit source kind required.")
    require(
        set(binding["files"]) == set(ROLES), "Exact aggregate input roster required."
    )
    identities = _base.obj(binding["identities"], set(IDENTITIES))
    for key, value in identities.items():
        if key.endswith("_head"):
            require(
                isinstance(value, str) and re.fullmatch("[a-f0-9]{40}", value),
                "Invalid source revision.",
            )
        else:
            digest(value)
    writer = _base._load_writer()
    for role, path in (
        ("exporter", __file__),
        ("base_exporter", _base.__file__),
        ("writer", writer.__file__),
    ):
        require(
            sha256(_base._file_bytes(Path(path))) == identities[role + "_sha256"],
            "Exporter or writer implementation mismatch.",
        )
    docs, hashes, sizes = {}, {}, {}
    for role in ROLES:
        row = _base.obj(binding["files"][role], {"path", "sha256"})
        docs[role], data = read_json(row["path"])
        hashes[role], sizes[role] = sha256(data), len(data)
        require(
            hashes[role] == digest(row["sha256"]),
            "Pinned aggregate digest mismatch: " + role,
        )
    require(
        set(binding["node_keys"]) == set(binding["implementation_hashes"]) == set(IDS),
        "Root node roster mismatch.",
    )
    keys = {k: digest(v) for k, v in binding["node_keys"].items()}
    impl = {k: digest(v) for k, v in binding["implementation_hashes"].items()}
    ca, cr = docs["composition_acceptance"], docs["composition_run"]
    aa, ar = docs["calibration_acceptance"], docs["calibration_run"]
    require(
        ca["status"] == "ACCEPT_LOCAL_COMPOSED_ASEC_ACS_ENGINEERING_DRILL"
        and aa["status"]
        == "ACCEPT_LOCAL_DEVELOPMENT_NATIONAL_AGE_CALIBRATION_EXECUTION",
        "Completion acceptance required.",
    )
    require(
        ca["run_receipt_sha256"] == hashes["composition_run"]
        and aa["run_receipt_sha256"] == hashes["calibration_run"],
        "Acceptance run binding mismatch.",
    )
    require(
        ca["head"]
        == cr["head"]
        == ar["source_head"]
        == aa["source_head"]
        == identities["composition_head"],
        "Composition source identity mismatch.",
    )
    require(
        aa["head"] == ar["head"] == identities["calibration_head"],
        "Calibration source identity mismatch.",
    )
    require(
        cr["runner_sha256"] == identities["composition_runner_sha256"]
        and ar["runner_sha256"]
        == aa["runner_sha256"]
        == identities["calibration_runner_sha256"],
        "Runner identity mismatch.",
    )
    require(aa["pins_sha256"] == ar["pins_sha256"], "Calibration pins mismatch.")
    for run, status in ((cr, "PASS"), (ar, "ok")):
        require(
            run["status"] == status
            and run["release_eligible"] is False
            and run["invented_sources"] is binding["invented_sources"]
            and run["genuine"] is (not binding["invented_sources"]),
            "Run status or source-kind mismatch.",
        )
    require(
        cr["age_date_harmonized"] is False and cr["asec_sex_or_head_bound"] is False,
        "Source scope changed; new review required.",
    )
    require(
        ca["final_frame"]["age_period_harmonized"] is False, "Age date scope mismatch."
    )
    require(
        aa["extension_cache_hits"] == 9
        and aa["extension_computed_nodes"] == 2
        and aa["fresh_warm_cache_hits"] == 11
        and aa["fresh_warm_computed_nodes"] == 0
        and aa["fresh_warm_is_second_solve"] is False,
        "Accepted cache-origin mismatch.",
    )
    executions = []
    results = {}
    modes = (
        "composition_cold",
        "composition_warm",
        "calibration_extension",
        "calibration_replay",
    )
    for mode in modes:
        doc, manifest = docs[mode + "_receipt"], docs[mode + "_manifest"]
        composed = mode.startswith("composition")
        phase = (
            mode.split("_")[-1]
            if composed
            else ("calibrate" if mode.endswith("extension") else "warm")
        )
        require(
            doc["phase"] == phase
            and doc["status"] == ("PASS" if composed else "ok")
            and doc["head"]
            == identities["composition_head" if composed else "calibration_head"],
            "Phase identity mismatch.",
        )
        require(
            doc["genuine"] is (not binding["invented_sources"]),
            "Phase source kind mismatch.",
        )
        if composed:
            for role, name in (
                (mode + "_receipt", phase + "-receipt.json"),
                (mode + "_manifest", phase + "-manifest.json"),
            ):
                require(
                    cr["outputs"][name]["sha256"] == hashes[role]
                    and cr["outputs"][name]["bytes"] == sizes[role],
                    "Composition output binding mismatch.",
                )
        else:
            require(
                doc["source_head"] == identities["composition_head"]
                and ar["phases"][phase]["receipt_sha256"] == hashes[mode + "_receipt"]
                and ar["phases"][phase]["exit_code"] == 0
                and ar["phases"][phase]["watchdog"] is None,
                "Calibration phase binding mismatch.",
            )
        r = results[mode] = doc["result"]
        ids = IDS[:9] if composed else IDS
        hits = {
            k: mode != "composition_cold"
            and (mode != "calibration_extension" or k in IDS[:9])
            for k in ids
        }
        require(
            r["node_keys"] == {k: keys[k] for k in ids}
            and r["node_hits" if composed else "cache_hits"] == hits,
            "Phase node identities or cache counts mismatch.",
        )
        require(
            r["manifest_key"] == manifest["key"], "Phase manifest binding mismatch."
        )
        if composed:
            require(
                r["kernel_call_count"] == (9 if phase == "cold" else 0),
                "Composition calls mismatch.",
            )
            require(
                r["whole_source_reads"] == (1 if phase == "cold" else 0),
                "Source preparation calls mismatch.",
            )
            require(
                r["entity_rows"] == ca["entity_rows"]
                and r["arm_rows"] == ca["arm_rows"],
                "Composition counts mismatch.",
            )
            require(
                r["checkpoint_sha256"] == ca["final_frame"]["checkpoint_sha256"],
                "Composition checkpoint mismatch.",
            )
        else:
            require(
                r["implementation_hashes"] == impl and r["sentinel_trips"] == [],
                "Calibration implementation or sentinel mismatch.",
            )
            require(
                r["limits"]["release_eligible"] is False
                and r["limits"]["standard_errors_consumed"] is False,
                "Calibration eligibility or SE scope mismatch.",
            )
        nodes = verify_manifest(manifest, ids, keys, impl, hits)
        executions.append(
            dict(
                mode=mode,
                label={
                    "composition_cold": "Composition cold run",
                    "composition_warm": "Composition fresh replay",
                    "calibration_extension": "Age-calibration extension",
                    "calibration_replay": "Calibration fresh replay",
                }[mode],
                description="Recorded execution; cached composition ancestry is distinct from newly computed calibration nodes.",
                execution_manifest_sha256=hashes[mode + "_manifest"],
                computation_content_key=manifest["key"],
                kernel_calls=sum(not x for x in hits.values()),
                source_preparation_calls=1 if mode == "composition_cold" else 0,
                computed_count=sum(not x for x in hits.values()),
                cache_hit_count=sum(hits.values()),
                failed_node_ids=[n["id"] for n in nodes if n["failed"]],
                blocked_node_ids=[],
                gate_exception_node_ids=[n["id"] for n in nodes if n["gate_exception"]],
                nodes=nodes,
                cache_origin="empty_composition_store"
                if mode == "composition_cold"
                else "same_composition_store"
                if mode == "composition_warm"
                else "mirror_of_accepted_composition_store"
                if mode == "calibration_extension"
                else "same_calibration_store",
                source_composition_receipt_sha256=hashes["composition_run"],
            )
        )
    require(
        results["composition_warm"]["fresh_process_parity_with_cold"] is True
        and results["calibration_replay"]["fresh_process_parity_with_added_pass"]
        is True,
        "Fresh replay parity attestation missing.",
    )
    for left, right in (
        ("composition_cold", "composition_warm"),
        ("calibration_extension", "calibration_replay"),
    ):
        require(
            docs[left + "_manifest"]["content_addressed"]
            == docs[right + "_manifest"]["content_addressed"],
            "Replay content mismatch.",
        )
    extension = docs["calibration_extension_manifest"]["content_addressed"]["nodes"]
    for k in IDS[:9]:
        require(
            extension[k]
            == docs["composition_cold_manifest"]["content_addressed"]["nodes"][k],
            "Inherited composition changed.",
        )
    result = results["calibration_replay"]
    for field in (
        "manifest_key",
        "node_keys",
        "implementation_hashes",
        "source_content_keys",
        "diagnostics_sha256",
        "diagnostics_artifact_key",
        "partition",
        "cells",
        "activation",
        "importance_weights",
        "calibrated_weights",
        "caps",
    ):
        require(
            result[field] == results["calibration_extension"][field],
            "Extension and replay mismatch: " + field,
        )
    require(
        result["manifest_key"] == aa["manifest_key"]
        and result["node_keys"] == aa["node_keys"]
        and result["partition"] == aa["partition"],
        "Accepted calibrated identity mismatch.",
    )
    material, aggregate, readback = (
        docs["materialization_receipt"],
        docs["aggregate_verification"],
        docs["checkpoint_readback"],
    )
    require(
        material["schema"] == "microcosm.us_national_age_cache_materializer.v1"
        and material["status"] == "ok"
        and material["root_only_materialization"] is True
        and material["original_store_unchanged"] is True
        and material["release_eligible"] is False
        and material["invented_sources"] is binding["invented_sources"],
        "Materialization status mismatch.",
    )
    require(
        material["helper_head"] == identities["materializer_head"]
        and material["helper_sha256"] == identities["materializer_sha256"]
        and material["production_head"] == identities["calibration_head"]
        and material["runner_sha256"] == identities["calibration_runner_sha256"],
        "Materializer code identity mismatch.",
    )
    require(
        material["completion_acceptance_sha256"]
        == hashes["calibration_acceptance"]
        == aggregate["completion_acceptance_sha256"],
        "Materialization acceptance mismatch.",
    )
    for role, name in (
        ("aggregate_verification", "aggregate-verification.json"),
        ("diagnostics", "diagnostics.json"),
        ("checkpoint_readback", "checkpoint-readback.json"),
        ("calibrated_manifest", "calibrated-manifest.json"),
    ):
        require(
            material["outputs"][name] == {"bytes": sizes[role], "sha256": hashes[role]},
            "Materializer output pin mismatch.",
        )
    mr = material["result"]
    require(
        mr["cache_hits"] == 11
        and mr["kernel_calls"] == mr["source_parse_calls"] == 0
        and mr["sentinel_trips"] == [],
        "Materialization was not cache-only.",
    )
    require(
        aggregate["schema"]
        == "microcosm.us_national_age_cache_materializer.v1.aggregate"
        and aggregate["kernel_calls"] == aggregate["source_data_parse_calls"] == 0
        and aggregate["initial_calibrated_household_ids_and_band_counts_exact"] is True
        and aggregate["release_eligible"] is False,
        "Aggregate materialization verification mismatch.",
    )
    require(
        aggregate["calibrate_receipt_sha256"] == hashes["calibration_extension_receipt"]
        and aggregate["warm_receipt_sha256"] == hashes["calibration_replay_receipt"],
        "Materialization phase mismatch.",
    )
    composition = {
        "checkpoint_sha256": digest(ca["final_frame"]["checkpoint_sha256"]),
        "run_receipt_sha256": hashes["composition_run"],
        "acceptance_sha256": hashes["composition_acceptance"],
        "computation_content_key": docs["composition_cold_manifest"]["key"],
    }
    for record in (material["composition"], aggregate["composition"]):
        require(
            record["checkpoint_sha256"] == composition["checkpoint_sha256"]
            and record["run_receipt_sha256"] == composition["run_receipt_sha256"]
            and record["acceptance"]["sha256"] == composition["acceptance_sha256"],
            "Materializer composition ancestry mismatch.",
        )
    checkpoint = digest(aggregate["population_sha256"])
    require(
        checkpoint
        not in {
            digest(result["importance_weights"]["bytes_sha256"]),
            digest(result["calibrated_weights"]["bytes_sha256"]),
            composition["checkpoint_sha256"],
        },
        "Checkpoint identity must be distinct from weight and composition identities.",
    )
    require(
        checkpoint
        == mr["population_sha256"]
        == material["outputs"]["calibrated-frame.checkpoint.h5"]["sha256"],
        "Checkpoint file identity mismatch.",
    )
    require(
        count(material["outputs"]["calibrated-frame.checkpoint.h5"]["bytes"]) > 0,
        "Empty checkpoint attestation.",
    )
    for field, expected in (
        ("manifest_key", result["manifest_key"]),
        ("calibrated_weights_sha256", result["calibrated_weights"]["bytes_sha256"]),
        ("diagnostics_sha256", hashes["diagnostics"]),
    ):
        require(
            aggregate[field] == mr[field] == aa[field] == expected,
            "Calibrated output identity mismatch.",
        )
    require(
        result["diagnostics_sha256"] == hashes["diagnostics"]
        and aggregate["diagnostics_artifact_key"]
        == result["diagnostics_artifact_key"]
        == extension[IDS[-1]]["opaque_artifacts"]["diagnostics"],
        "Diagnostics artifact identity mismatch.",
    )
    require(
        docs["calibrated_manifest"]["content_addressed"]
        == docs["calibration_replay_manifest"]["content_addressed"],
        "Saved calibrated manifest mismatch.",
    )
    verify_manifest(
        docs["calibrated_manifest"], IDS, keys, impl, {k: True for k in IDS}
    )
    entities = {identifier(k): count(v) for k, v in ca["entity_rows"].items()}
    households, persons = entities["household"], entities["person"]
    require(
        households > 0
        and persons > 0
        and result["partition"]["households"] == households
        and result["partition"]["all_person_partition_count"] == persons
        and aggregate["partition"] == result["partition"],
        "Calibrated count mismatch.",
    )
    require(
        readback == mr["checkpoint_readback"]
        and readback["all_tables_ids_strata_and_weights_exact"] is True
        and readback["entity_counts"] == entities
        and readback["weights"]["household"]
        == {
            "kind": "calibrated",
            "records": households,
            "sha256": aggregate["calibrated_weights_sha256"],
        },
        "Checkpoint readback mismatch.",
    )
    arms = []
    require(
        set(binding["source_vintages"]) == {"asec", "acs"}, "Two source arms required."
    )
    for arm, label in (("asec", "CPS ASEC"), ("acs", "ACS")):
        vintages = binding["source_vintages"][arm]
        require(
            isinstance(vintages, list) and 1 <= len(vintages) <= 12,
            "Recorded source vintages required.",
        )
        arms.append(
            dict(
                id=arm,
                label=label,
                households=count(ca["arm_rows"]["household"][arm]),
                persons=count(ca["arm_rows"]["person"][arm]),
                source_vintages=[safe_text(v) for v in vintages],
            )
        )
    for entity, total in entities.items():
        require(
            ca["arm_rows"][entity]["other"] == 0
            and sum(count(ca["arm_rows"][entity][arm]) for arm in ("asec", "acs"))
            == total,
            "Source-arm count mismatch.",
        )
    diag = docs["diagnostics"]
    cells = verify_cells(result, aa, aggregate, diag, households, persons)
    weights = verify_weights(result, aa, aggregate, diag, households)
    solver = aa["solver"]
    require(
        solver["convergence_flag_available_from_api"] is False,
        "Unsupported convergence claim.",
    )
    for field in ("method", "epochs", "learning_rate", "seed", "mass"):
        require(
            solver[field] == diag["options"][field], "Solver configuration mismatch."
        )
    for field in ("initial_loss", "final_loss", "diagnostic_warnings"):
        require(solver[field] == diag[field], "Recorded solver diagnostic mismatch.")
    summary = trajectory_summary(diag["loss_trajectory"])
    require(
        summary == solver["loss_trajectory"]
        and len(diag["loss_trajectory"]) == solver["epochs"],
        "Recorded trajectory summary mismatch.",
    )
    caps = {}
    require(result["caps"] == aa["caps"], "Accepted cap mismatch.")
    for role, anchor in (
        ("executor", "original_design"),
        ("solver", "incoming_importance"),
    ):
        c = result["caps"][role]
        require(
            c["anchor"] == anchor == diag["build"][role + "_weight_anchor"],
            "Weight cap anchor mismatch.",
        )
        require(
            c["max_weight_ratio"] == diag["build"][role + "_max_weight_ratio"],
            "Weight cap mismatch.",
        )
        realized = c["realized"] if role == "executor" else c["realized_recomputed"]
        require(
            0 <= number(realized) <= number(c["max_weight_ratio"]) * (1 + 2e-6),
            "Realized cap exceeds bound.",
        )
        if role == "solver":
            close(
                c["realized_from_diagnostics"],
                diag["realized_max_weight_ratio"],
                "Solver cap diagnostic mismatch.",
            )
            require(
                math.isclose(
                    realized,
                    diag["realized_max_weight_ratio"],
                    rel_tol=2e-6,
                    abs_tol=1e-7,
                ),
                "Recomputed solver cap mismatch.",
            )
        caps[role] = dict(
            anchor=anchor, max_weight_ratio=c["max_weight_ratio"], realized=realized
        )

    def error_summary(rows):
        return dict(
            denominator=len(rows),
            excluded_cells=18 - len(rows),
            maximum_absolute_relative_deviation=max(
                (abs(r["final_relative_deviation"]) for r in rows), default=None
            ),
        )

    provenance = {
        **identities,
        **{k + "_sha256": v for k, v in hashes.items()},
        "root_binding_sha256": binding_sha256,
        "calibrated_weights_sha256": aggregate["calibrated_weights_sha256"],
    }
    common = dict(
        country="us",
        candidate_release_id=identifier(binding["candidate_release_id"]),
        run_id=identifier(binding["run_id"]),
        profile=PROFILE,
        genuine=not binding["invented_sources"],
        invented_sources=binding["invented_sources"],
        provenance=provenance,
        execution_status="completed",
        release_eligibility={"status": "excluded", "reason": REASON},
    )
    graph = dict(
        schema_version="microcosm.graph_receipts.v1",
        **common,
        population_sha256=checkpoint,
        period="2024",
        entity="household",
        period_basis="target_year",
        period_reason="2024 is the national target year. Ages remain those observed at each source interview; observation dates are not harmonized.",
        scope=dict(
            households=households,
            persons=persons,
            entity_rows=[{"entity": k, "rows": v} for k, v in sorted(entities.items())],
            source_arms=arms,
        ),
        computation_content_key=result["manifest_key"],
        composition=composition,
        quality=dict(
            status="incomplete",
            basis="age_only_development_calibration",
            score=None,
            missing=MISSING,
            note="Prior wages are outside this cross-sectional experiment. Engineering completion does not establish launch quality.",
        ),
        implementation_set={
            "sha256": _base.implementation_set_digest(
                {
                    k: {
                        "kernel_ref": extension[k]["kernel_ref"],
                        "kernel_impl_hash": impl[k],
                    }
                    for k in IDS
                }
            ),
            "by_node": {
                k: {
                    "kernel_ref": extension[k]["kernel_ref"],
                    "kernel_impl_hash": impl[k],
                }
                for k in IDS
            },
        },
        age_calibration=dict(
            target_year=2024,
            consumes_standard_errors=False,
            tolerance=None,
            tolerance_status="unavailable",
            cells=cells,
            error_summary={
                "all_cells": error_summary(cells),
                "supported_cells": error_summary(
                    [r for r in cells if r["supported_households"]]
                ),
            },
            weights=weights,
            caps=caps,
            solver={
                **{
                    f: solver[f]
                    for f in (
                        "method",
                        "epochs",
                        "learning_rate",
                        "seed",
                        "mass",
                        "initial_loss",
                        "final_loss",
                        "diagnostic_warnings",
                    )
                },
                "convergence_status": "unavailable",
            },
            trajectory={
                "kind": "complete_recorded",
                "summary": summary,
                "values": diag["loss_trajectory"],
            },
            column_provenance=[
                {
                    "columns": ["published_value", "published_se", "compiled_target"],
                    "source_role": "calibration_replay_receipt",
                    "source_sha256": hashes["calibration_replay_receipt"],
                },
                {
                    "columns": ["initial_estimate", "final_estimate", "trajectory"],
                    "source_role": "diagnostics",
                    "source_sha256": hashes["diagnostics"],
                },
                {
                    "columns": [*SUPPORT, "weights"],
                    "source_role": "aggregate_verification",
                    "source_sha256": hashes["aggregate_verification"],
                },
            ],
        ),
        executions=executions,
    )
    tree(graph, public=True)
    graph_raw = canonical_json(graph) + b"\n"
    build = dict(
        schema_version="microcosm.graph_build_summary.v1",
        **common,
        graph_receipts_sha256=sha256(graph_raw),
        checks={
            k: True
            for k in (
                "pinned_aggregate_inputs",
                "accepted_composition_ancestry",
                "extension_and_replay_cache_origin",
                "all_18_targets_reconciled",
                "independent_support_reconciled",
                "checkpoint_file_identity_attested",
                "recorded_trajectory_verified",
                "no_source_or_population_reads",
                "release_excluded",
            )
        },
    )
    tree(build, public=True)
    build_raw = canonical_json(build) + b"\n"
    require(
        max(len(graph_raw), len(build_raw)) <= _base.MAX_JSON_BYTES,
        "Public attachment size limit.",
    )
    metadata = dict(
        country="us",
        candidate_release_id=common["candidate_release_id"],
        run_id=common["run_id"],
        created_at=binding["created_at"],
        execution_status="completed",
        release_eligibility=common["release_eligibility"],
        population={"sha256": checkpoint, "period": "2024", "entity": "household"},
        provenance=provenance,
    )
    writer._metadata(metadata)
    try:
        Path(output_root).mkdir(parents=False, exist_ok=False)
    except OSError as exc:
        raise EvidenceError("Exclusive output directory required.") from exc
    with tempfile.TemporaryDirectory(prefix="combined-age-evidence-") as temporary:
        p = Path(temporary)
        (p / "graph.json").write_bytes(graph_raw)
        (p / "build.json").write_bytes(build_raw)
        identity = {k: common[k] for k in ("country", "candidate_release_id", "run_id")}
        attachments = {
            "graph_receipts": {
                "status": "available",
                "scope": "population",
                "source": str(p / "graph.json"),
                "identity": {**identity, "population_sha256": checkpoint},
            },
            "build_manifest": {
                "status": "available",
                "scope": "run",
                "source": str(p / "build.json"),
                "identity": identity,
            },
        }
        for name in writer.EXPECTED_ATTACHMENTS:
            if name not in attachments:
                attachments[name] = {
                    "status": "missing",
                    "reason": "Not evaluated for this development age-calibration experiment.",
                }
        reference = writer.write_candidate_quality(
            metadata, attachments, Path(output_root), update_index=False
        )
    writer.read_candidate_quality(Path(output_root), reference["sha256"])
    return {
        **reference,
        "profile": PROFILE,
        "graph_receipts_sha256": sha256(graph_raw),
        "build_summary_sha256": sha256(build_raw),
        "population_sha256": checkpoint,
        "households": households,
        "persons": persons,
        "index_updated": False,
    }


def export_combined_age_evidence(
    *, binding_path: Path, binding_sha256: str, output_root: Path
):
    """Verify a root-pinned aggregate chain and seal two attachments without an index.

    Missing fields or malformed structures fail closed with a path-free error.
    No checkpoint, store, source or person-level artifact is opened.
    """
    try:
        return _export_combined_age_evidence(
            binding_path=binding_path,
            binding_sha256=binding_sha256,
            output_root=output_root,
        )
    except EvidenceError:
        raise
    except (
        KeyError,
        ValueError,
        TypeError,
        IndexError,
        AttributeError,
        ZeroDivisionError,
        OverflowError,
    ) as exc:
        raise EvidenceError("Malformed aggregate schema.") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = export_combined_age_evidence(
            binding_path=args.binding,
            binding_sha256=args.binding_sha256,
            output_root=args.output,
        )
    except (EvidenceError, KeyError, TypeError) as exc:
        raise SystemExit(
            "Combined age evidence refused: "
            + (
                str(exc)
                if isinstance(exc, EvidenceError)
                else "Malformed aggregate schema."
            )
        ) from None
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
