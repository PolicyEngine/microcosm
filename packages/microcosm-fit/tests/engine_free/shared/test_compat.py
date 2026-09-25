"""The constellation mechanism: the kernel-compatibility gate at import.

DESIGN.md requires each shard to assert kernel compatibility at import — a cheap
``frame.__version__`` check — so a resolver that ignores ``[tool.uv.sources]``
cannot silently assemble an incompatible pair. These tests exercise the gate
directly (the real import already ran the gate successfully, or the suite would
not have loaded).
"""

from __future__ import annotations

import pytest

import microcosm.fit
from microcosm.fit import __version__ as fit_version
from microcosm.fit import _assert_frame_compatible


def test_namespace_has_no_top_level_init() -> None:
    """microcosm stays a PEP 420 namespace; the fit shard ships no __init__ for it.

    A shard clobbering ``microcosm/__init__.py`` would break side-by-side install
    of microcosm-frame and microcosm-calibrate. (The fit *subpackage* has its own
    ``__init__``; the *namespace* must not.)
    """
    import microcosm

    assert getattr(microcosm, "__file__", None) is None
    assert hasattr(microcosm, "__path__")


def test_fit_declares_its_own_version() -> None:
    """The shard exposes its version for the constellation matrix."""
    assert fit_version == "0.1.0"
    assert microcosm.fit.__version__ == "0.1.0"


def test_compat_gate_accepts_the_matching_series() -> None:
    """The installed kernel passes the gate (this is the live configuration)."""
    # Exact series match.
    _assert_frame_compatible("0.1.0", (0, 1))
    # Patch differences within the series are fine.
    _assert_frame_compatible("0.1.5", (0, 1))


def test_compat_gate_rejects_a_too_old_or_too_new_kernel() -> None:
    """A kernel outside the required 0.x minor series is refused at import.

    Pre-1.0, minors may break compatibility, so 0.0.x and 0.2.x are both
    incompatible with a shard built for 0.1.x. The error names both versions.
    """
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
