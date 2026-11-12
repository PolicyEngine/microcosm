"""Top-1 share of every QRF output in the certified Route A release.

Runs the PR branch's own ``_qrf_tail_concentration_gate`` on the release's
published H5 (QRF columns only, read with the post-export probe's loader),
first at the release's calibrated household weights, then at the initial
weights from the release's calibration sidecar.

Fidelity checks before anything is reported:

* the H5 and the recorded ``qrf_tail_concentration.json`` hash to the
  release manifest's sha256s;
* at calibrated weights the branch's gate reproduces the recorded tail
  evidence exactly: verdict, surface buckets, and the gate's details
  (every top-100 share and carrier count) after a JSON round trip, and the
  probe's ``compare_qrf_tail`` finds no difference;
* the sidecar's ``household_weight`` equals the H5's household weights
  element for element, so its ``initial_household_weight`` aligns by row.

Usage: route_a_top1.py <worktree> <release-out dir> <out dir>
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
RELEASE_OUT = Path(sys.argv[2])
OUT = Path(sys.argv[3])
RELEASE_ID = RELEASE_OUT.name
RELEASE_DIR = RELEASE_OUT / "releases" / RELEASE_ID
H5 = RELEASE_OUT / "artifacts" / "populace_us_2024.h5"
SIDECAR = RELEASE_OUT / "artifacts" / "populace_us_2024_calibration.npz"
THRESHOLDS = (0.05, 0.10)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


builder = _load("build_us_fiscal_refresh_release", "build_us_fiscal_refresh_release.py")
probe = _load("probe_us_post_export", "probe_us_post_export.py")
sampler = probe.load_sampler()
from microcosm.frame import Frame, WeightKind, Weights  # noqa: E402
from microcosm.frame.units import US_SCHEMA  # noqa: E402

manifest = json.loads((RELEASE_DIR / "release_manifest.json").read_text())
artifacts = manifest["artifacts"]
h5_sha = _sha256(H5)
tail_sha = _sha256(RELEASE_DIR / "qrf_tail_concentration.json")
assert h5_sha == artifacts["populace_us_2024"]["sha256"], h5_sha
assert tail_sha == artifacts["qrf_tail_concentration"]["sha256"], tail_sha

recorded = json.loads((RELEASE_DIR / "qrf_tail_concentration.json").read_text())
register = dict(recorded["surface"]["reviewed_exclusions"])
qrf_outputs = sorted(builder._qrf_imputed_source_outputs())
frame = probe.source_columns_frame(
    H5, qrf_outputs, sampler=sampler, chunk_bytes=sampler.DEFAULT_CHUNK_BYTES
)

# --- Fidelity at the release's calibrated weights ------------------------
gate, surface = builder._qrf_tail_concentration_gate(
    frame, reviewed_exclusions=register
)
fidelity: dict[str, object] = {}
roundtrip = json.loads(json.dumps(dict(gate.details)))
fidelity["gate_details_equal_recorded"] = (
    roundtrip == recorded["tail_concentration"]["details"]
)
fidelity["verdict_equal_recorded"] = (
    gate.passed == recorded["tail_concentration"]["passed"]
)
for bucket in (
    "checked_sparse_columns",
    "dense_columns",
    "absent_columns",
    "non_numeric_columns",
):
    fidelity[f"{bucket}_equal_recorded"] = sorted(surface[bucket]) == sorted(
        recorded["surface"][bucket]
    )
comparison = probe.compare_qrf_tail(
    {"passed": gate.passed, "surface": surface, "details": dict(gate.details)},
    recorded,
)
fidelity["probe_compare_differences"] = comparison["differences"]
fidelity["probe_columns_compared"] = comparison["columns_compared"]
assert all(v is True for k, v in fidelity.items() if k.endswith("_recorded")), fidelity
assert not comparison["differences"], comparison
block = surface["single_record_concentration"]
assert block["unmeasured"] == {}, block["unmeasured"]
for column, share in gate.details["top_share"].items():
    assert block["columns"][column]["top_k_share"] == min(share, 1.0), column

# --- Initial weights from the calibration sidecar ------------------------
with np.load(SIDECAR, allow_pickle=False) as sidecar:
    final_weights = sidecar["household_weight"]
    initial_weights = sidecar["initial_household_weight"]
h5_weights = np.asarray(frame.resolve_weights("household").values, dtype=np.float64)
fidelity["sidecar_final_weights_equal_h5"] = bool(
    np.array_equal(final_weights, h5_weights)
)
assert fidelity["sidecar_final_weights_equal_h5"]
initial_frame = Frame(
    {entity: frame.table(entity) for entity in sampler.US_ENTITIES},
    US_SCHEMA,
    {"household": Weights(initial_weights, WeightKind.DESIGN)},
)
initial_gate, initial_surface = builder._qrf_tail_concentration_gate(
    initial_frame, reviewed_exclusions=register
)
initial_block = initial_surface["single_record_concentration"]


# --- The largest record itself, for the report ----------------------------
def largest_record(frame_: Frame, column: str) -> dict[str, float] | None:
    entity = frame_.column_entity(column)
    values = frame_.table(entity)[column].to_numpy(dtype=np.float64)
    weights = np.asarray(frame_.resolve_weights(entity).values, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(weights)
    mass = np.where(finite, np.abs(values) * weights, 0.0)
    mass = np.where(mass > 0.0, mass, 0.0)
    if not mass.any():
        return None
    index = int(np.argmax(mass))
    table = frame_.table(entity)
    household = (
        table["person_household_id"].iloc[index]
        if entity == "person"
        else table["household_id"].iloc[index]
        if entity == "household"
        else None
    )
    return {
        "entity": entity,
        "record_id": int(table[f"{entity}_id"].iloc[index]),
        "household_id": None if household is None else int(household),
        "value": float(values[index]),
        "weight": float(weights[index]),
        "mass": float(mass[index]),
        "column_absolute_mass": float(mass.sum()),
    }


def classify(column: str, surface_: dict, details: dict) -> str:
    if column in surface_["dense_columns"]:
        return "dense"
    if column in details["thin_columns"]:
        return "thin"
    return "checked"


rows = []
for column in sorted(block["columns"]):
    final = block["columns"][column]
    initial = initial_block["columns"][column]
    record = largest_record(frame, column)
    initial_record = largest_record(initial_frame, column)
    if record is not None:
        assert record["weight"] == final["top_1_weight"] or np.isclose(
            record["mass"] / record["column_absolute_mass"], final["top_1_share"]
        ), column
    rows.append(
        {
            "column": column,
            "class": classify(column, surface, gate.details),
            "waived": column in register,
            "entity": None if record is None else record["entity"],
            "carriers": final["carriers"],
            "final": final,
            "initial": initial,
            "final_largest_record": record,
            "initial_largest_record": initial_record,
        }
    )


def over(threshold: float, key: str) -> list[str]:
    return [
        row["column"]
        for row in sorted(rows, key=lambda r: -(r[key]["top_1_share"] or 0.0))
        if (row[key]["top_1_share"] or 0.0) > threshold
    ]


summary = {
    "release_id": RELEASE_ID,
    "h5_sha256": h5_sha,
    "qrf_tail_concentration_sha256": tail_sha,
    "branch_commit": __import__("subprocess")
    .run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    .stdout.strip(),
    "fidelity": fidelity,
    "qrf_outputs": len(qrf_outputs),
    "measured_columns": len(rows),
    "over": {
        f"{key}_weights_over_{int(t * 100)}pct": over(t, key)
        for key in ("final", "initial")
        for t in THRESHOLDS
    },
    "rows": rows,
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "route_a_top1.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def pct(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def money(value):
    return "n/a" if value is None else f"${value / 1e9:,.2f}B"


lines = [
    f"# Route A single-record concentration ({RELEASE_ID})",
    "",
    f"H5 sha256 `{h5_sha}`; recorded tail evidence `{tail_sha}`.",
    "",
    "| column | class | waived | carriers | top-1 final | top-1 weight final "
    "| top-100 final | largest record final (value × weight) | top-1 initial "
    "| top-1 weight initial |",
    "|---|---|---|---:|---:|---:|---:|---|---:|---:|",
]


def weight(value):
    return "n/a" if value is None else f"{value:,.1f}"


def described(record):
    if record is None:
        return "n/a"
    return (
        f"${record['value']:,.0f} × {record['weight']:,.1f} = {money(record['mass'])}"
    )


for row in sorted(rows, key=lambda r: -(r["final"]["top_1_share"] or 0.0)):
    lines.append(
        f"| `{row['column']}` | {row['class']} | {'yes' if row['waived'] else ''} "
        f"| {row['carriers']:,} | {pct(row['final']['top_1_share'])} "
        f"| {weight(row['final']['top_1_weight'])} "
        f"| {pct(row['final']['top_k_share'])} "
        f"| {described(row['final_largest_record'])} "
        f"| {pct(row['initial']['top_1_share'])} "
        f"| {weight(row['initial']['top_1_weight'])} |"
    )
lines += ["", "## Over the thresholds", ""]
for key, columns in summary["over"].items():
    lines.append(
        f"- **{key}** ({len(columns)}): " + ", ".join(f"`{c}`" for c in columns)
    )
(OUT / "route_a_top1.md").write_text("\n".join(lines) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
