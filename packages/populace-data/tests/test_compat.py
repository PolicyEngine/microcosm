"""Namespace and public-API guarantees for the data shard.

Unlike the operator shards, populace-data has no Frame kernel in its dependency
closure, so it carries no kernel-compat gate. What it must still uphold: it
stays a PEP 420 namespace contributor (no top-level ``populace/__init__.py``)
so it installs side by side with the other shards, and it exposes exactly the
documented public surface.
"""

from __future__ import annotations

import populace.data
from populace.data import __version__ as data_version


def test_namespace_has_no_top_level_init() -> None:
    """populace stays a PEP 420 namespace; the data shard ships no __init__.

    A shard clobbering ``populace/__init__.py`` would break side-by-side install
    of populace-frame/fit/calibrate/data. (The data *subpackage* has its own
    ``__init__``; the *namespace* must not.)
    """
    import populace

    assert getattr(populace, "__file__", None) is None
    assert hasattr(populace, "__path__")


def test_data_declares_its_own_version() -> None:
    assert data_version == "0.1.0"
    assert populace.data.__version__ == "0.1.0"


def test_public_api_surface() -> None:
    """The shard exports exactly the documented public names."""
    from populace.data import (
        DEFAULT_VARIANT,
        REGISTRY,
        DatasetSpec,
        available,
        available_variants,
        download,
        latest_year,
        load,
        register,
        resolve,
    )

    for fn in (
        load,
        download,
        available,
        available_variants,
        resolve,
        latest_year,
        register,
    ):
        assert callable(fn)
    assert isinstance(DatasetSpec, type)
    assert isinstance(REGISTRY, dict)
    assert DEFAULT_VARIANT == "compact"
