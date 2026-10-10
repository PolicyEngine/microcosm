"""Run the pool tool's ACS input path on the pinned ACS 2024 1-year archives.

This makes the three calls ``tools/build_us_multispine_pool.py`` makes on its
ACS arm before spine assembly, through the tool module's own names:
``build_acs_pums_unit_frame``, ``map_acs_native_inputs`` and
``assert_operator_free_source_frame`` with the native-input receipt. Both
archives must match the checked-in ACS source manifest.

The receipt records whether the boundary admitted the frame, the usual-hours
receipt, which rows the mapping leaves blank, and a preview of the by-origin
battery's usual-hours comparison against the public ASEC file. It builds no
pool and writes nothing but the receipt.

``--repo-root`` selects the checkout whose tool is loaded, so one script can
measure origin/main and a fix. The interpreter decides which ``microcosm``
sources are imported; the receipt records both paths.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_USUAL_HOURS = "weekly_hours_worked_before_lsr"
_BOUNDARY_LABEL = "ACS native-mapped stacked input"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_tool(root: Path):
    spec = importlib.util.spec_from_file_location(
        "build_us_multispine_pool_acs_input_path",
        root / "tools" / "build_us_multispine_pool.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _asec_usual_hours(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``HRSWK`` as the ASEC hours operator maps it, at household weights."""

    person = pd.read_hdf(path, "person")
    household = pd.read_hdf(path, "household")
    weights = person["PH_SEQ"].map(household.set_index("H_SEQ")["HSUP_WGT"])
    hours = pd.to_numeric(person["HRSWK"], errors="coerce").fillna(0.0)
    return (
        np.maximum(hours.to_numpy(dtype=np.float64), 0.0),
        weights.to_numpy(dtype=np.float64),
    )


def _battery_preview(
    *,
    acs_hours: np.ndarray,
    acs_weights: np.ndarray,
    asec_hours: np.ndarray,
    asec_weights: np.ndarray,
) -> dict[str, object]:
    """The battery's positive-leg comparison, with its own functions."""

    import microcosm.build.us_runtime.stacked_spine as stacked_spine

    incidence = {
        "asec": stacked_spine._weighted_mask_incidence(asec_hours > 0, asec_weights),
        "acs": stacked_spine._weighted_mask_incidence(acs_hours > 0, acs_weights),
    }
    quantiles = {
        "asec": stacked_spine._battery_conditional_quantiles(
            asec_hours[asec_hours > 0], asec_weights[asec_hours > 0]
        ),
        "acs": stacked_spine._battery_conditional_quantiles(
            acs_hours[acs_hours > 0], acs_weights[acs_hours > 0]
        ),
    }
    lower, upper = stacked_spine._BATTERY_INCIDENCE_RATIO_BOUNDS
    ratio = incidence["acs"] / incidence["asec"]
    distance = stacked_spine._battery_quantile_envelope_distance(
        quantiles["asec"], quantiles["acs"]
    )
    tolerance = stacked_spine._BATTERY_QUANTILE_ENVELOPE_TOLERANCE
    return {
        "metric": "monetary_sign_separated, positive leg",
        "quantiles": list(stacked_spine._BATTERY_QUANTILES),
        "positive_incidence": incidence,
        "incidence_ratio_acs_over_asec": ratio,
        "incidence_ratio_bounds": [lower, upper],
        "conditional_quantiles": {
            origin: values.tolist() for origin, values in quantiles.items()
        },
        "quantile_envelope_distance": distance,
        "quantile_envelope_tolerance": tolerance,
        "within_tolerance": bool(lower <= ratio <= upper and distance <= tolerance),
        "caveats": [
            "A preview, not a battery run: no pool was built.",
            "ACS blanks count as zero hours here; the pool fills them from "
            "ASEC donors, which are zero below age 15.",
            "ASEC is the public Census file named below at household "
            "supplement weights, not the pool's pooled raw-stage artifact.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path.home() / ".cache/microcosm/acs-pums/2024-1yr",
    )
    parser.add_argument("--asec-h5", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    root = args.repo_root.resolve()
    tool = _load_tool(root)
    import microcosm.build.us_runtime.operator_boundary as operator_boundary

    receipt: dict[str, object] = {
        "code": {
            "repo_root_head": _git(root, "rev-parse", "HEAD"),
            "repo_root_dirty_paths": _git(root, "status", "--porcelain").splitlines(),
            "tool": str(Path(tool.__file__).resolve()),
            "operator_boundary": str(Path(operator_boundary.__file__).resolve()),
        },
        "archives": {},
    }
    manifest = tool.load_acs_source_manifest()
    paths: dict[str, Path] = {}
    for artifact in manifest.artifacts:
        path = args.archive_dir / artifact.filename
        actual = _sha256(path)
        if actual != artifact.sha256:
            raise SystemExit(f"{path} does not match the ACS source manifest.")
        paths[artifact.role] = path
        receipt["archives"][artifact.role] = {
            "filename": artifact.filename,
            "sha256": actual,
            "matches_source_manifest": True,
        }

    started = time.monotonic()
    acs_frame, _acs_build = tool.build_acs_pums_unit_frame(
        tool.AcsPumsSource(
            household_zip=paths["household"],
            person_zip=paths["person"],
            vintage=manifest.vintage,
        )
    )
    mapped = tool.map_acs_native_inputs(acs_frame)
    receipt["load_and_map_seconds"] = round(time.monotonic() - started, 1)
    receipt["native_outputs"] = sorted(mapped.native_inputs)
    receipt["usual_hours_receipt"] = dict(mapped.native_inputs[_USUAL_HOURS])
    try:
        tool.assert_operator_free_source_frame(
            mapped.frame,
            label=_BOUNDARY_LABEL,
            native_inputs=mapped.native_inputs,
        )
    except ValueError as error:
        receipt["boundary"] = {"admitted": False, "error": str(error)}
    else:
        receipt["boundary"] = {"admitted": True, "error": None}

    person = mapped.frame.table("person")
    weights = np.asarray(mapped.frame.resolve_weights("person").values, dtype=float)
    hours = pd.to_numeric(person[_USUAL_HOURS], errors="coerce")
    age = pd.to_numeric(person["AGEP"], errors="coerce")
    blank = hours.isna()
    receipt["usual_hours_rows"] = {
        "persons": len(person),
        "blank": int(blank.sum()),
        "blank_under_15": int((blank & age.lt(15)).sum()),
        "blank_age_15": int((blank & age.eq(15)).sum()),
        "blank_age_16_plus": int((blank & age.ge(16)).sum()),
        "observed_under_16": int((~blank & age.lt(16)).sum()),
        "observed_zero": int(hours.eq(0).sum()),
        "observed_positive": int(hours.gt(0).sum()),
    }
    if args.asec_h5 is not None:
        asec_hours, asec_weights = _asec_usual_hours(args.asec_h5)
        receipt["battery_preview"] = {
            "asec_file": {
                "name": args.asec_h5.name,
                "sha256": _sha256(args.asec_h5),
                "persons": len(asec_hours),
            },
            **_battery_preview(
                acs_hours=hours.fillna(0.0).to_numpy(dtype=np.float64),
                acs_weights=weights,
                asec_hours=asec_hours,
                asec_weights=asec_weights,
            ),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["boundary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
