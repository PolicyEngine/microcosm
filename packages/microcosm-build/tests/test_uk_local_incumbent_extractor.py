from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest


def _tool_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "tools/extract_uk_local_incumbent_surface.py"
    )
    spec = importlib.util.spec_from_file_location("extract_uk_local_incumbent", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_incumbent_surface_from_tiny_h5_and_stub_simulation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pytest.importorskip("tables")
    tool = _tool_module()
    incumbent = tmp_path / "incumbent.h5"
    with pd.HDFStore(incumbent, mode="w") as store:
        store["household"] = pd.DataFrame({"household_id": [10, 20]})
    constituency_h5 = tmp_path / "constituency.h5"
    with h5py.File(constituency_h5, mode="w") as root:
        root.create_dataset("2025", data=np.asarray([[1.0, 2.0]]))
        root.attrs["area_codes"] = np.asarray(["C1"], dtype="S2")
    la_h5 = tmp_path / "la.h5"
    with h5py.File(la_h5, mode="w") as root:
        root.create_dataset("2025", data=np.asarray([[3.0, 4.0]]))
    la_roster = tmp_path / "la.csv"
    pd.DataFrame({"code": ["L1"], "name": ["Local one"]}).to_csv(la_roster, index=False)

    class StubSimulation:
        pass

    def fake_metrics(simulation, grain, *, period, household_ids):
        assert isinstance(simulation, StubSimulation)
        assert period == 2025
        assert list(household_ids) == [10, 20]
        common = {"households": [1.0, 1.0]}
        if grain == "constituency":
            common["constituency_only"] = [5.0, 6.0]
        else:
            common["la_only"] = [7.0, 8.0]
        return pd.DataFrame(common, index=household_ids)

    monkeypatch.setattr(tool, "compute_household_metrics", fake_metrics)
    out_dir = tmp_path / "out"
    manifest = tool.extract_incumbent_surface(
        incumbent_h5=incumbent,
        constituency_weights_h5=constituency_h5,
        local_authority_weights_h5=la_h5,
        constituency_codes_csv=None,
        local_authority_codes_csv=la_roster,
        period=2025,
        out_dir=out_dir,
        microsimulation_factory=lambda **_: StubSimulation(),
    )

    weights = pd.read_csv(out_dir / tool.WEIGHTS_FILENAME)
    metrics = pd.read_csv(out_dir / tool.METRICS_FILENAME)
    assert weights.to_dict("list") == {
        "household_id": [10, 20],
        "C1": [1.0, 2.0],
        "L1": [3.0, 4.0],
    }
    assert metrics.to_dict("list") == {
        "household_id": [10, 20],
        "households": [1.0, 1.0],
        "constituency_only": [5.0, 6.0],
        "la_only": [7.0, 8.0],
    }
    assert manifest["geography"] == {
        "constituency_areas": 1,
        "local_authority_areas": 1,
    }
    persisted = json.loads((out_dir / tool.MANIFEST_FILENAME).read_text())
    assert (
        persisted["outputs"]["weights"]["sha256"]
        == manifest["outputs"]["weights"]["sha256"]
    )
