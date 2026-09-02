#!/usr/bin/env python3
"""Extract aggregate local weights and metric surfaces from an incumbent UK H5."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from microcosm.build.uk_runtime import compute_household_metrics

WEIGHTS_FILENAME = "incumbent_local_weights.csv"
METRICS_FILENAME = "incumbent_household_metrics.csv"
MANIFEST_FILENAME = "incumbent_local_surface_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _decode_codes(values: Any) -> list[str]:
    array = np.asarray(values).reshape(-1)
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in array.tolist()
    ]


def _roster_codes(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    if "code" not in frame.columns:
        raise ValueError(f"{path}: roster CSV must contain a code column.")
    codes = frame["code"].astype(str).tolist()
    if not codes or len(set(codes)) != len(codes):
        raise ValueError(f"{path}: roster codes must be non-empty and unique.")
    return codes


def _h5_array_and_codes(
    path: Path,
    *,
    period: int,
) -> tuple[np.ndarray, list[str] | None]:
    with h5py.File(path, "r") as root:
        name = str(period)
        if name not in root or not isinstance(root[name], h5py.Dataset):
            raise ValueError(
                f"{path}: expected a two-dimensional {name!r} weight dataset."
            )
        dataset = root[name]
        if dataset.ndim != 2:
            raise ValueError(
                f"{path}: {name!r} weight dataset must be two-dimensional."
            )
        values = np.asarray(dataset[...], dtype=np.float64)
        codes = None
        for owner in (dataset, root):
            if "area_codes" in owner.attrs:
                codes = _decode_codes(owner.attrs["area_codes"])
                break
        if codes is None:
            for candidate in ("area_codes", f"{name}_area_codes"):
                if candidate in root and isinstance(root[candidate], h5py.Dataset):
                    codes = _decode_codes(root[candidate][...])
                    break
    return values, codes


def _load_weight_surface(
    path: Path,
    *,
    household_ids: pd.Series,
    roster_csv: Path | None,
    period: int,
) -> pd.DataFrame:
    roster = _roster_codes(roster_csv)
    frame: pd.DataFrame | None = None
    try:
        with pd.HDFStore(path, mode="r") as store:
            candidates = []
            for key in store.keys():
                value = store[key]
                if isinstance(value, pd.DataFrame) and value.shape[1] >= 1:
                    candidates.append(value)
            if len(candidates) == 1:
                frame = candidates[0].copy()
    except (KeyError, TypeError, ValueError):
        frame = None
    if frame is not None:
        if "household_id" in frame.columns:
            observed_ids = frame.pop("household_id")
            if not observed_ids.reset_index(drop=True).equals(
                household_ids.reset_index(drop=True)
            ):
                raise ValueError(
                    f"{path}: household_id does not align to incumbent H5."
                )
        codes = [str(column) for column in frame.columns]
        values = frame.to_numpy(dtype=np.float64)
    else:
        values, embedded_codes = _h5_array_and_codes(path, period=period)
        codes = embedded_codes or roster
        if codes is None:
            raise ValueError(
                f"{path}: weight H5 carries no area-code index; supply its roster CSV."
            )
    if roster is not None and codes != roster:
        raise ValueError(f"{path}: embedded area-code order does not match roster CSV.")
    expected_wide = (len(codes), len(household_ids))
    expected_tabular = (len(household_ids), len(codes))
    if frame is None and values.shape == expected_wide:
        values = values.T
    if values.shape != expected_tabular:
        raise ValueError(
            f"{path}: weight shape {values.shape} does not match "
            f"the incumbent layout {expected_wide} (areas x households)."
        )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{path}: weights must be finite and non-negative.")
    return pd.DataFrame(values, columns=codes)


def extract_incumbent_surface(
    *,
    incumbent_h5: Path,
    constituency_weights_h5: Path,
    local_authority_weights_h5: Path,
    constituency_codes_csv: Path | None,
    local_authority_codes_csv: Path | None,
    period: int,
    out_dir: Path,
    microsimulation_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    """Write aggregate incumbent weights, metrics, and their pinned manifest."""

    with pd.HDFStore(incumbent_h5, mode="r") as store:
        if "/household" not in store.keys():
            raise ValueError(f"{incumbent_h5}: missing household table.")
        household = store["household"]
    if "household_id" not in household.columns:
        raise ValueError(f"{incumbent_h5}: household table lacks household_id.")
    household_ids = household["household_id"]
    if household_ids.isna().any() or household_ids.duplicated().any():
        raise ValueError(f"{incumbent_h5}: household_id must be complete and unique.")
    constituency = _load_weight_surface(
        constituency_weights_h5,
        household_ids=household_ids,
        roster_csv=constituency_codes_csv,
        period=period,
    )
    local_authority = _load_weight_surface(
        local_authority_weights_h5,
        household_ids=household_ids,
        roster_csv=local_authority_codes_csv,
        period=period,
    )
    overlap = sorted(set(constituency.columns) & set(local_authority.columns))
    if overlap:
        raise ValueError(f"Area codes collide across incumbent grains: {overlap[:10]}.")

    if microsimulation_factory is None:
        from policyengine_uk import Microsimulation

        microsimulation_factory = Microsimulation
    simulation = microsimulation_factory(dataset=str(incumbent_h5))
    metric_frames = [
        compute_household_metrics(
            simulation,
            grain,
            period=period,
            household_ids=household_ids,
        )
        for grain in ("constituency", "la")
    ]
    metrics = metric_frames[0].copy()
    for frame in metric_frames[1:]:
        shared = sorted(set(metrics.columns) & set(frame.columns))
        for column in shared:
            if not np.array_equal(
                metrics[column].to_numpy(), frame[column].to_numpy(), equal_nan=True
            ):
                raise ValueError(
                    f"Metric {column!r} differs between constituency and LA extraction."
                )
        metrics = metrics.join(frame.drop(columns=shared), how="outer")
    weights = pd.concat([constituency, local_authority], axis=1)
    weights.insert(0, "household_id", household_ids.to_numpy())
    metrics = metrics.reset_index(drop=True)
    metrics.insert(0, "household_id", household_ids.to_numpy())

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / WEIGHTS_FILENAME
    metrics_path = out_dir / METRICS_FILENAME
    manifest_path = out_dir / MANIFEST_FILENAME
    weights.to_csv(weights_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    manifest = {
        "schema_version": 1,
        "kind": "uk_local_incumbent_surface",
        "period": int(period),
        "households": len(household_ids),
        "geography": {
            "constituency_areas": len(constituency.columns),
            "local_authority_areas": len(local_authority.columns),
        },
        "inputs": {
            "incumbent_h5": _artifact(incumbent_h5),
            "constituency_weights_h5": _artifact(constituency_weights_h5),
            "local_authority_weights_h5": _artifact(local_authority_weights_h5),
            **(
                {"constituency_codes_csv": _artifact(constituency_codes_csv)}
                if constituency_codes_csv is not None
                else {}
            ),
            **(
                {"local_authority_codes_csv": _artifact(local_authority_codes_csv)}
                if local_authority_codes_csv is not None
                else {}
            ),
        },
        "outputs": {
            "weights": _artifact(weights_path),
            "metrics": _artifact(metrics_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-h5", type=Path, required=True)
    parser.add_argument("--constituency-weights-h5", type=Path, required=True)
    parser.add_argument("--local-authority-weights-h5", type=Path, required=True)
    parser.add_argument("--constituency-codes-csv", type=Path)
    parser.add_argument("--local-authority-codes-csv", type=Path)
    parser.add_argument("--period", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    extract_incumbent_surface(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
