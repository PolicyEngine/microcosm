"""Import-boundary regressions for the US runtime package."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from importlib import import_module

import pytest

_SPINE_MODULE = "microcosm.build.us_runtime.spine_agreement"
_SPINE_EXPORTS = (
    "DEFAULT_CATEGORICAL_TOTAL_VARIATION_TOLERANCE",
    "DEFAULT_INCIDENCE_RATIO_BOUNDS",
    "DEFAULT_QUANTILE_ENVELOPE_TOLERANCE",
    "DEFAULT_SPINE_AGREEMENT_QUANTILES",
    "US_SPINE_AGREEMENT_REGISTRY",
    "SpineAgreementSpec",
    "default_spine_agreement_registry",
    "normalize_transfer_family_name",
    "spine_agreement_gate",
    "validate_spine_agreement_registry",
)


def test_us_trade_does_not_import_spine_agreement() -> None:
    script = textwrap.dedent(
        f"""
        import sys

        import microcosm.build.us_runtime.us_trade
        import microcosm.build.us_runtime as us_runtime

        assert {_SPINE_MODULE!r} not in sys.modules
        assert set({_SPINE_EXPORTS!r}) <= set(dir(us_runtime))
        assert {_SPINE_MODULE!r} not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_spine_agreement_exports_preserve_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import microcosm.build.us_runtime as us_runtime

    spine_agreement = import_module(_SPINE_MODULE)
    sentinel_registry = object()
    spine_agreement.__dict__.pop("US_SPINE_AGREEMENT_REGISTRY", None)
    monkeypatch.setattr(
        spine_agreement,
        "default_spine_agreement_registry",
        lambda: sentinel_registry,
    )
    assert "US_SPINE_AGREEMENT_REGISTRY" in dir(spine_agreement)
    assert "US_SPINE_AGREEMENT_REGISTRY" not in spine_agreement.__dict__

    direct = {name: getattr(us_runtime, name) for name in _SPINE_EXPORTS}
    assert direct["US_SPINE_AGREEMENT_REGISTRY"] is sentinel_registry
    for name, value in direct.items():
        assert value is getattr(spine_agreement, name)
        assert us_runtime.__dict__[name] is value

    for name in _SPINE_EXPORTS:
        us_runtime.__dict__.pop(name)

    from microcosm.build.us_runtime import (
        DEFAULT_CATEGORICAL_TOTAL_VARIATION_TOLERANCE,
        DEFAULT_INCIDENCE_RATIO_BOUNDS,
        DEFAULT_QUANTILE_ENVELOPE_TOLERANCE,
        DEFAULT_SPINE_AGREEMENT_QUANTILES,
        US_SPINE_AGREEMENT_REGISTRY,
        SpineAgreementSpec,
        default_spine_agreement_registry,
        normalize_transfer_family_name,
        spine_agreement_gate,
        validate_spine_agreement_registry,
    )

    imported = {
        "DEFAULT_CATEGORICAL_TOTAL_VARIATION_TOLERANCE": (
            DEFAULT_CATEGORICAL_TOTAL_VARIATION_TOLERANCE
        ),
        "DEFAULT_INCIDENCE_RATIO_BOUNDS": DEFAULT_INCIDENCE_RATIO_BOUNDS,
        "DEFAULT_QUANTILE_ENVELOPE_TOLERANCE": (
            DEFAULT_QUANTILE_ENVELOPE_TOLERANCE
        ),
        "DEFAULT_SPINE_AGREEMENT_QUANTILES": DEFAULT_SPINE_AGREEMENT_QUANTILES,
        "US_SPINE_AGREEMENT_REGISTRY": US_SPINE_AGREEMENT_REGISTRY,
        "SpineAgreementSpec": SpineAgreementSpec,
        "default_spine_agreement_registry": default_spine_agreement_registry,
        "normalize_transfer_family_name": normalize_transfer_family_name,
        "spine_agreement_gate": spine_agreement_gate,
        "validate_spine_agreement_registry": validate_spine_agreement_registry,
    }
    for name, value in imported.items():
        assert value is direct[name]
        assert value is getattr(spine_agreement, name)
        assert us_runtime.__dict__[name] is value

    for name in _SPINE_EXPORTS:
        us_runtime.__dict__.pop(name)
    spine_agreement.__dict__.pop("US_SPINE_AGREEMENT_REGISTRY", None)


def test_default_gate_uses_lazy_canonical_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spine_agreement = import_module(_SPINE_MODULE)
    spine_agreement.__dict__.pop("US_SPINE_AGREEMENT_REGISTRY", None)
    monkeypatch.setattr(
        spine_agreement,
        "default_spine_agreement_registry",
        lambda: (),
    )
    monkeypatch.setattr(
        spine_agreement,
        "validate_assembly_provenance",
        lambda *args, **kwargs: None,
    )

    class EmptyFrame:
        entities: tuple[str, ...] = ()

    result = spine_agreement.spine_agreement_gate(EmptyFrame())

    assert result.passed
    assert spine_agreement.__dict__["US_SPINE_AGREEMENT_REGISTRY"] == ()
    spine_agreement.__dict__.pop("US_SPINE_AGREEMENT_REGISTRY")
