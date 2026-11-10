"""The runtime identity a build's first telemetry event carries."""

from __future__ import annotations

import json

import pytest

from microcosm.build import telemetry_identity

REQUEST = {"class": "heavy", "cpu": 4.0, "cpu_limit": None, "memory_mib": 131072}


def test_a_modal_stage_reports_its_container_s_resource_request(monkeypatch) -> None:
    monkeypatch.setenv("MICROCOSM_MODAL_RESOURCES", json.dumps(REQUEST))
    identity = telemetry_identity.runtime_identity()
    assert identity.get("modal_resources") == REQUEST


def test_a_build_outside_modal_reports_no_resource_request(monkeypatch) -> None:
    monkeypatch.delenv("MICROCOSM_MODAL_RESOURCES", raising=False)
    identity = telemetry_identity.runtime_identity()
    assert "modal_resources" in identity
    assert identity["modal_resources"] is None


@pytest.mark.parametrize(
    "value",
    [
        "not JSON",
        "[]",
        json.dumps({**REQUEST, "extra": 1}),
        json.dumps({**REQUEST, "cpu": "four"}),
        json.dumps({key: value for key, value in REQUEST.items() if key != "class"}),
    ],
    ids=["not JSON", "a list", "another key", "a text CPU", "no class"],
)
def test_an_unusable_resource_request_is_left_out(monkeypatch, value: str) -> None:
    monkeypatch.setenv("MICROCOSM_MODAL_RESOURCES", value)
    identity = telemetry_identity.runtime_identity()
    assert "modal_resources" in identity
    assert identity["modal_resources"] is None
