"""Compare the seam (A) and graph (B) national builds on one spine.

Usage (from a tree with the uk extra): uv run --no-sync python compare_national_ab.py <a-dir> <b-dir>

Reports as JSON: the solve through the target-support sidecars both sides
write before the battery (row names, target vector, design and final weights,
the CSR matrix), the diagnostics minus the build block and the build block
minus its operational fields, the gate report outcomes and details, the
Logbook rows' identity and pin digests, the H5 payload when both sides wrote
one, and the wall time.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from microcosm.build.logbook import load_spool_rows

A, B = Path(sys.argv[1]), Path(sys.argv[2])
DATASET = "microcosm_uk_2024_25.h5"
GATES = "microcosm_uk_2024_25.terminal_gates.json"
OPERATIONAL = {
    "build_id",
    "created_at",
    "code_pin",
    "runtime",
    "git_commit",
    "git_dirty",
    "staging_delivery",
    "path",
    "graph",
    "rowwise_driver_parameters",
    "certification",
    "sha256",
    "size_bytes",
    "bytes",
    "elapsed_seconds",
    "seconds",
    "evaluated_at",
    "timestamp",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip(payload, drop=OPERATIONAL):
    if isinstance(payload, dict):
        return {k: strip(v, drop) for k, v in payload.items() if k not in drop}
    if isinstance(payload, list):
        return [strip(v, drop) for v in payload]
    return payload


def diff_paths(a, b, prefix=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(
                    f"{prefix}/{key}: {'only in A' if key in a else 'only in B'}"
                )
            else:
                out.extend(diff_paths(a[key], b[key], f"{prefix}/{key}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{prefix}: list length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                out.extend(diff_paths(x, y, f"{prefix}[{i}]"))
    elif a != b:
        out.append(f"{prefix}: {str(a)[:80]!r} vs {str(b)[:80]!r}")
    return out


report = {"a": str(A), "b": str(B)}

# 1. The solve, from the target-support sidecars.
va = np.load(A / "target_support_vectors.npz", allow_pickle=True)
vb = np.load(B / "target_support_vectors.npz", allow_pickle=True)
ma, mb = (
    sp.load_npz(A / "target_support_matrix.npz"),
    sp.load_npz(B / "target_support_matrix.npz"),
)
fa, fb = va["final_weights"], vb["final_weights"]
report["solve"] = {
    "targets": (int(ma.shape[0]), int(mb.shape[0])),
    "households": (int(ma.shape[1]), int(mb.shape[1])),
    "row_names_equal": bool(np.array_equal(va["names"], vb["names"])),
    "target_vector_equal": bool(
        np.array_equal(va["target_vector"], vb["target_vector"])
    ),
    "matrix_identical": bool(ma.shape == mb.shape and (ma != mb).nnz == 0),
    "design_weights_equal": bool(
        np.array_equal(va["design_weights"], vb["design_weights"])
    ),
    "household_ids_equal": bool(
        np.array_equal(va["household_ids"], vb["household_ids"])
    ),
    "final_weights_identical": bool(np.array_equal(fa, fb)),
    "final_weights_max_abs_diff": float(np.max(np.abs(fa - fb)))
    if fa.shape == fb.shape
    else None,
    "final_weight_totals": (float(fa.sum()), float(fb.sum())),
}

# 2. The diagnostics.
da = json.loads((A / "calibration_diagnostics.json").read_text())
db = json.loads((B / "calibration_diagnostics.json").read_text())
report["diagnostics"] = {
    "scalars": {
        k: (da.get(k), db.get(k), da.get(k) == db.get(k))
        for k in (
            "schema_version",
            "initial_loss",
            "final_loss",
            "n_nonzero",
            "n_records",
            "realized_max_weight_ratio",
            "fraction_within_10pct",
        )
    },
    "diff_minus_build": diff_paths(
        strip(da, OPERATIONAL | {"build"}), strip(db, OPERATIONAL | {"build"})
    ),
    "build_diff": diff_paths(strip(da.get("build", {})), strip(db.get("build", {}))),
    "bytes_identical": sha(A / "calibration_diagnostics.json")
    == sha(B / "calibration_diagnostics.json"),
}

# 3. The gate report.
ga, gb = json.loads((A / GATES).read_text()), json.loads((B / GATES).read_text())
report["gate_report"] = {
    "blocked_at_phase": (ga.get("blocked_at_phase"), gb.get("blocked_at_phase")),
    "statuses": {
        k: (ga["gates"][k]["status"], gb["gates"][k]["status"]) for k in ga["gates"]
    },
    "diff_minus_attestation": diff_paths(
        strip(ga, OPERATIONAL | {"attestation", "release_evidence"}),
        strip(gb, OPERATIONAL | {"attestation", "release_evidence"}),
    ),
    "signed": (
        bool(ga.get("attestation", {}).get("signature")),
        bool(gb.get("attestation", {}).get("signature")),
    ),
}

# 4. The Logbook rows.
ra, rb = (
    load_spool_rows(A / "logbook-spool")[0].to_mapping(),
    load_spool_rows(B / "logbook-spool")[0].to_mapping(),
)
report["logbook"] = {
    k: (ra[k], rb[k], ra[k] == rb[k])
    for k in (
        "pipeline",
        "identity_digest",
        "input_pins_digest",
        "disposition",
        "rung",
        "seed",
    )
}
report["logbook"]["phases"] = (ra["phases_reached"], rb["phases_reached"])

# 5. The H5 and the build record, when both sides wrote them.
if (A / DATASET).is_file() and (B / DATASET).is_file():
    from microcosm.build.uk_runtime.national_frame import load_uk_national_frame

    fra, _ = load_uk_national_frame(A / DATASET)
    frb, _ = load_uk_national_frame(B / DATASET)
    h5 = {"bytes_identical": sha(A / DATASET) == sha(B / DATASET), "tables": {}}
    for entity in fra.entities:
        ta, tb = fra.table(entity), frb.table(entity)
        entry = {
            "rows": (len(ta), len(tb)),
            "columns_equal": list(ta.columns) == list(tb.columns),
        }
        unequal, dtype_only = [], []
        for col in [c for c in ta.columns if c in tb.columns]:
            sa, sb = ta[col], tb[col]
            if str(sa.dtype) != str(sb.dtype):
                dtype_only.append(f"{col}: {sa.dtype} vs {sb.dtype}")
            try:
                same = np.array_equal(sa.to_numpy(), sb.to_numpy()) or sa.astype(
                    object
                ).equals(sb.astype(object))
            except Exception:
                same = sa.astype(str).equals(sb.astype(str))
            if not same:
                unequal.append(col)
        entry["value_unequal_columns"] = unequal
        entry["dtype_only_differences"] = dtype_only
        h5["tables"][entity] = entry
    h5["weights_identical"] = bool(
        np.array_equal(
            fra.weights_for("household").values, frb.weights_for("household").values
        )
    )
    h5["mass_log_identical"] = fra.mass_log == frb.mass_log
    report["h5"] = h5
else:
    report["h5"] = {"written": ((A / DATASET).is_file(), (B / DATASET).is_file())}
if (A / "build_record.json").is_file() and (B / "build_record.json").is_file():
    bra, brb = (
        json.loads((A / "build_record.json").read_text()),
        json.loads((B / "build_record.json").read_text()),
    )
    report["build_record"] = {"diff": diff_paths(strip(bra), strip(brb))}
else:
    report["build_record"] = {
        "written": (
            (A / "build_record.json").is_file(),
            (B / "build_record.json").is_file(),
        )
    }


# 6. Wall time from the run logs.
def wall(path: Path):
    for line in path.read_text().splitlines():
        if line.strip().endswith("real") or " real " in line:
            return line.strip().split()[0]
    return None


report["wall_seconds"] = {"a": wall(A / "run.log"), "b": wall(B / "run.log")}
print(json.dumps(report, indent=2, default=str))
