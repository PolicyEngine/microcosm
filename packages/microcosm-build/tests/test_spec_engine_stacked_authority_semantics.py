"""Live-byte oracles for pure stacked authority and checkpoint projections."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from microcosm.build.spec_engine.loader import load_bundle
from microcosm.build.spec_engine.model import ResolvedSpec, thaw_json
from microcosm.build.spec_engine.stacked_authority_semantics import (
    project_stacked_authority_receipt,
    project_stacked_checkpoint_base_identity,
    project_stacked_checkpoint_static_components,
    stacked_identity_bytes,
)
from microcosm.build.us_runtime.stacked_spine import (
    stacked_spine_authority_receipt,
)

pytest.importorskip(
    "policyengine_us",
    reason="live-engine oracle: the wheels gate's venv installs no engine",
    exc_type=ModuleNotFoundError,
)

ROOT = Path(__file__).resolve().parents[3]
MODULE = (
    ROOT / "packages/microcosm-build/src/microcosm/build/spec_engine/"
    "stacked_authority_semantics.py"
)


@pytest.fixture(scope="module")
def resolved_us_spec() -> ResolvedSpec:
    return load_bundle("us")


@pytest.fixture(scope="module")
def pool_tool() -> ModuleType:
    path = ROOT / "tools" / "build_us_multispine_pool.py"
    spec = importlib.util.spec_from_file_location(
        "spec_engine_stacked_identity_live_oracle",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_authority_projection_is_field_and_byte_identical_to_live_generation_zero(
    resolved_us_spec: ResolvedSpec,
) -> None:
    projected = project_stacked_authority_receipt(resolved_us_spec)
    live = dict(stacked_spine_authority_receipt())

    assert projected == live
    assert stacked_identity_bytes(projected) == _canonical_bytes(live)
    assert projected["sha256"] == (
        "6761af0fa25e80f15eb960dc75a2838c2b3a09fda3e487bb4cf9ada5dcabf6db"
    )
    assert {
        name: component["sha256"] for name, component in projected["components"].items()
    } == {
        "declared_surface": (
            "3cd7bd2a96595c6a836029db16580a05aa727751ba58dd9a43962eb8783f0ec1"
        ),
        "gap_fill_plan": (
            "f41319a95750a441676bc6599b1de6bb49a87b45d83b9263d627c402cfe8e750"
        ),
        "joint_metric_registry": (
            "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
        ),
        "late_producer_schedule": (
            "5308a5644ab9d74009c7dace075d23d4b9f378288cdee07c82b01d0c1dbb6a1a"
        ),
        "metric_registry": (
            "abb2e8c9ec8982ac93d0a33ebc9888a27fd2642ab86833bffa1a4411c98e2b00"
        ),
        "post_puf_transfer_surface": (
            "cb9d878ed03fdc3dfcacd1b3516650a6afe830ca2910d572cf21978498636b58"
        ),
        "post_transfer_calibration": (
            "141519684c72ab84a077ae0f5716a0416f1e19da57262948e459633cbe560576"
        ),
        "puf_capital_gains_tail_support_contract": (
            "91bc9272cb7f28c6271fb9695ddf6ec05fe55b4458070fc47ae4a1d9607f3c89"
        ),
        "support_profile": (
            "fd8b92353f53f7e562a829b4d7c82b888d3ce25a436195d3b334b14132b73e46"
        ),
    }


def test_checkpoint_projection_is_field_and_byte_identical_to_live_oracle(
    resolved_us_spec: ResolvedSpec,
    pool_tool: ModuleType,
) -> None:
    verified = {
        "zeta": pool_tool._VerifiedInput(
            role="zeta",
            path=Path("unused-zeta"),
            expected_sha256="b" * 64,
            actual_sha256="b" * 64,
            size_bytes=23,
        ),
        "alpha": pool_tool._VerifiedInput(
            role="alpha",
            path=Path("unused-alpha"),
            expected_sha256="a" * 64,
            actual_sha256="a" * 64,
            size_bytes=17,
        ),
        "puma_ladder": pool_tool._VerifiedInput(
            role="puma_ladder",
            path=Path("unused-puma-ladder"),
            expected_sha256=pool_tool._STACKED_PUMA_LADDER_SHA256,
            actual_sha256=pool_tool._STACKED_PUMA_LADDER_SHA256,
            size_bytes=29,
        ),
        "congressional_district_vintage_crosswalk": pool_tool._VerifiedInput(
            role="congressional_district_vintage_crosswalk",
            path=Path("unused-cd-crosswalk"),
            expected_sha256=pool_tool._STACKED_CD_CROSSWALK_SHA256,
            actual_sha256=pool_tool._STACKED_CD_CROSSWALK_SHA256,
            size_bytes=77_935,
        ),
    }
    pins = {
        role: {"sha256": pin.actual_sha256, "size_bytes": pin.size_bytes}
        for role, pin in verified.items()
    }
    stack_receipt = {
        "sample_seed": 578,
        "sample_fraction": 0.25,
        "survey_samples": {
            "asec": {"realized_household_count": 3},
            "acs": {"realized_household_count": 5},
        },
        "unicode_probe": "Caf\u00e9",
    }
    generated = thaw_json(resolved_us_spec.generated_authorities)
    assert isinstance(generated, dict)
    engine_lock = generated["engine_abi_lock"]
    assert isinstance(engine_lock, dict)
    engine = engine_lock["engine"]
    assert isinstance(engine, dict)

    projected = project_stacked_checkpoint_base_identity(
        resolved_us_spec,
        input_pins=pins,
        stack_receipt=stack_receipt,
        sample_fraction=0.25,
        sample_seed=578,
        clone_attachment_fraction=0.4,
        clone_attachment_seed=991,
    )
    live = pool_tool._stacked_checkpoint_base_identity(
        verified,
        stack_receipt=stack_receipt,
        sample_fraction=0.25,
        sample_seed=578,
        clone_attachment_fraction=0.4,
        clone_attachment_seed=991,
        policyengine_us_version=str(engine["version"]),
    )

    assert projected == live
    assert stacked_identity_bytes(projected) == _canonical_bytes(live)
    assert list(projected["inputs"]) == [
        "alpha",
        "congressional_district_vintage_crosswalk",
        "puma_ladder",
        "zeta",
    ]
    assert (
        projected["pool_code"]["late_producer_schedule"]["schedule_sha256"]
        == "1fa79f7d95660fa9c5e33d153d547238d321eac7550fc43c4f0b3d87fd99a185"
    )


def test_static_projection_selects_exact_defaulted_live_identity_components(
    resolved_us_spec: ResolvedSpec,
    pool_tool: ModuleType,
) -> None:
    generated = thaw_json(resolved_us_spec.generated_authorities)
    assert isinstance(generated, dict)
    engine_lock = generated["engine_abi_lock"]
    assert isinstance(engine_lock, dict)
    engine = engine_lock["engine"]
    assert isinstance(engine, dict)
    live = pool_tool._stacked_checkpoint_base_identity(
        {},
        stack_receipt={"sample_fraction": 1.0, "sample_seed": 578},
        sample_fraction=1.0,
        sample_seed=578,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
        policyengine_us_version=str(engine["version"]),
    )
    selected_keys = (
        "artifact_kind",
        "schema_version",
        "materializer_version",
        "pipeline",
        "period",
        "model_seed",
        "policyengine_us_version",
        "geography_assignment",
        "stacked_authority",
        "pool_code",
    )
    expected = {key: live[key] for key in selected_keys}

    projected = project_stacked_checkpoint_static_components(resolved_us_spec)

    assert projected == expected
    assert stacked_identity_bytes(projected) == _canonical_bytes(expected)
    assert not {"inputs", "sampling", "clone_attachment"}.intersection(projected)


def test_production_projector_has_no_executor_or_frozen_evidence_dependency() -> None:
    source = MODULE.read_text(encoding="utf-8")
    forbidden_import_fragments = (
        "from tools",
        "import tools",
        "country_spec",
        "us_runtime",
        "take_up_contract.json",
        "source_stages.json",
        "support_spine.json",
    )
    assert all(fragment not in source for fragment in forbidden_import_fragments)
