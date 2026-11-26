"""tools/modal_rebuild_uk_chronicle_artifact.py's image, against a stub ``modal``.

The app's credential tests are in ``test_modal_uk_cut.py``; these check the
image the rebuild runs in. Nothing here builds an image or contacts Modal.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[5]
REBUILD_TOOL = ROOT / "tools" / "modal_rebuild_uk_chronicle_artifact.py"
US_PLAN_TOOL = ROOT / "tools" / "modal_us_stage_plan.py"


def _load_rebuild_app(monkeypatch):
    """The rebuild app as the launcher sees it: ``modal.is_local()`` is true."""

    modal_stub = MagicMock(name="modal")
    modal_stub.is_local.return_value = True
    modal_stub.App.return_value.function.side_effect = lambda **options: (
        lambda function: setattr(function, "modal_options", options) or function
    )
    modal_stub.App.return_value.local_entrypoint.side_effect = lambda **_: (
        lambda function: function
    )
    monkeypatch.setitem(sys.modules, "modal", modal_stub)
    spec = importlib.util.spec_from_file_location(
        "modal_rebuild_uk_chronicle_artifact_under_test", REBUILD_TOOL
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, modal_stub


def _load_us_plan(monkeypatch):
    """The US Modal stage runner's plan module, loaded by path like the tools."""

    spec = importlib.util.spec_from_file_location(
        "modal_us_stage_plan_beside_chronicle_rebuild", US_PLAN_TOOL
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolve string annotations through sys.modules.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _image_chain(modal_stub):
    """Each image-definition call as (step name, arguments, options), in order."""

    return [
        (call_name.rsplit(".", 1)[-1], call_args, call_options)
        for call_name, call_args, call_options in modal_stub.mock_calls
        if call_name.startswith("Image.debian_slim")
    ]


def test_the_rebuild_image_trusts_no_compromised_certificate_authority(monkeypatch):
    """As the US Modal image does, before anything else is downloaded.

    emSign ECC Root CA - G3, the eMudhra root removed after the 2026-10-20
    key disclosure: uv before 0.13.4 compiles it in, the cached base's pip 25.3
    bundles it, and the base's CA store, which git's clone of Chronicle and the
    standard library read, holds it until it is upgraded. The upload's Hub
    client checks the Hub's certificate against certifi on the image's own
    Python.
    """

    us_plan = _load_us_plan(monkeypatch)
    module, modal_stub = _load_rebuild_app(monkeypatch)
    chain = _image_chain(modal_stub)
    assert [step_name for step_name, _, _ in chain] == [
        "debian_slim",
        "apt_install",
        "run_commands",
        "pip_install",
        "run_commands",
        "env",
    ]
    assert "ca-certificates" in chain[1][1]
    # The hash-pinned pip replaces the base's before pip downloads anything else.
    assert chain[2][1] == (us_plan.image_pip_bootstrap_command(),)
    assert chain[3][1] == (
        f"uv=={us_plan.IMAGE_UV_VERSION}",
        f"huggingface_hub=={module.HF_HUB_VERSION}",
        f"certifi>={us_plan.RUNNER_CERTIFI_MINIMUM_VERSION}",
    )
    # The rebuild's own Python comes from the upgraded uv.
    assert chain[4][1] == (f"uv python install {module.PYTHON_VERSION}",)
    # No step before the environment passes options such as another index or
    # a trusted host.
    assert [step_options for _, _, step_options in chain[1:5]] == [{}, {}, {}, {}]
