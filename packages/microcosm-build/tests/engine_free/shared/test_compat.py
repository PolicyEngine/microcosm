"""The constellation mechanism: the kernel-compatibility gate at import.

DESIGN.md requires each shard to assert kernel compatibility at import so a
resolver that ignores ``[tool.uv.sources]`` cannot silently assemble an
incompatible pair. microcosm-build reads the installed frame version from
distribution metadata, so the gate does not load the kernel's numpy and pandas
stack. These tests exercise the gate directly (the real import already ran it
successfully, or the suite would not have loaded).
"""

from __future__ import annotations

from importlib import metadata

import pytest
from hypothesis import given
from hypothesis import strategies as st

import microcosm.build
import microcosm.frame
from microcosm.build import _assert_frame_compatible, _installed_frame_version


def test_build_declares_its_own_version() -> None:
    """The shard exposes its version for the constellation matrix."""
    assert microcosm.build.__version__ == "0.1.0"


def test_gate_reads_the_installed_frame_distribution() -> None:
    """The gate's version is the installed distribution's version."""
    assert _installed_frame_version() == metadata.version("microcosm-frame")


def test_installed_distribution_matches_the_kernel_module() -> None:
    """Metadata and ``frame.__version__`` are hand-written separately; keep them equal.

    The gate reads metadata while microcosm-fit and microcosm-calibrate read the
    module attribute, so both sources must name the same kernel.
    """
    assert metadata.version("microcosm-frame") == microcosm.frame.__version__


def test_gate_falls_back_to_the_module_without_a_distribution(monkeypatch) -> None:
    """A frame importable from the path but not installed still gets checked."""

    def not_installed(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(microcosm.build._metadata, "version", not_installed)
    assert _installed_frame_version() == microcosm.frame.__version__


def test_compat_gate_accepts_the_matching_series() -> None:
    """The installed kernel passes the gate (this is the live configuration)."""
    _assert_frame_compatible("0.1.0", (0, 1))
    _assert_frame_compatible("0.1.5", (0, 1))


def test_compat_gate_rejects_a_too_old_or_too_new_kernel() -> None:
    """A kernel outside the required 0.x minor series is refused at import."""
    with pytest.raises(ImportError, match="requires microcosm-frame 0.1.x"):
        _assert_frame_compatible("0.0.9", (0, 1))
    with pytest.raises(ImportError, match="0.2.0 is installed"):
        _assert_frame_compatible("0.2.0", (0, 1))


def test_compat_gate_uses_major_only_from_1_0() -> None:
    """From 1.0 on, the gate matches the major and tolerates any minor."""
    _assert_frame_compatible("1.4.2", (1, 0))
    with pytest.raises(ImportError, match="requires microcosm-frame 2.x"):
        _assert_frame_compatible("1.9.9", (2, 0))


def test_compat_gate_rejects_an_unparseable_version() -> None:
    """A version string the gate cannot parse is a clear ImportError."""
    with pytest.raises(ImportError, match="cannot parse"):
        _assert_frame_compatible("not-a-version", (0, 1))


_PART = st.integers(min_value=0, max_value=30)


@given(major=_PART, minor=_PART, patch=_PART, required_major=_PART, required=_PART)
def test_compat_gate_accepts_exactly_the_required_series(
    major: int, minor: int, patch: int, required_major: int, required: int
) -> None:
    """Pre-1.0 the gate pins the minor series; from 1.0 it pins the major only."""
    version = f"{major}.{minor}.{patch}"
    series = (required_major, required)
    accepted = (
        (major, minor) == series if required_major == 0 else major == required_major
    )
    if accepted:
        _assert_frame_compatible(version, series)
    else:
        with pytest.raises(ImportError, match=f"{version} is installed"):
            _assert_frame_compatible(version, series)
