import importlib.util
from pathlib import Path

import pytest


def _load_support_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_puf_support_base.py"
    spec = importlib.util.spec_from_file_location("build_us_puf_support_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cd_vintage_crosswalk_requires_cd_assignment() -> None:
    builder = _load_support_builder_module()

    with pytest.raises(SystemExit) as exc:
        builder._parse_args(
            [
                "--base-h5",
                "base.h5",
                "--puf-h5",
                "puf.h5",
                "--out",
                "out",
                "--congressional-district-vintage-crosswalk",
                "crosswalk.csv",
            ]
        )

    assert exc.value.code == 2
