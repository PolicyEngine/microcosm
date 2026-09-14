"""Explicit packaged-resource and installed microunit byte verification.

This module never imports microunit or runs units. The lower-level packaged
function permits explicit inspected-contract injection for pure synthetic tests;
it does not certify an installed distribution or authenticate source data.
"""

import platform
import sys
from importlib import metadata, resources

import numpy as np
import pandas as pd

from .asec_current_money import (
    MICROUNIT_SHA256,
    MICROUNIT_VERSION,
    CurrentMoneyResources,
    MoneyRefusalError,
    _json,
    _require,
    _sha,
)

RESOURCE_NAMES = (
    "asec_current_money_domains_v1.json",
    "asec_current_money_price_basis_v1.json",
    "asec_current_money_consumers_v1.json",
)
MODULE_NAMES = (
    "asec_current_money.py",
    "_asec_current_money_codec.py",
    "asec_current_money_resources.py",
)


def packaged_current_money_resources(
    *, inspected_version: str, inspected_module_sha256: str
) -> CurrentMoneyResources:
    """Read package resources with explicit claims, for synthetic fixture injection.

    Use load_current_money_resources to verify the actual installed reader.
    Neither resource-reading function creates authenticated source authority.
    """
    package = resources.files(__package__)
    identity = _json(
        {
            "modules": {
                name: _sha(package.joinpath(name).read_bytes()) for name in MODULE_NAMES
            },
            "dependencies": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "byteorder": sys.byteorder,
                "python_implementation": platform.python_implementation(),
            },
        }
    )
    return CurrentMoneyResources(
        *(package.joinpath(name).read_bytes() for name in RESOURCE_NAMES),
        identity,
        inspected_version,
        inspected_module_sha256,
    )


def load_current_money_resources(
    *, microunit_search_path: tuple[str, ...] | None = None
) -> CurrentMoneyResources:
    """Verify pinned installed version/module bytes without importing microunit.

    An explicit site-packages search path supports a separately pinned local
    environment. Paths are operational and never enter the normative identity.
    """
    try:
        if microunit_search_path is None:
            distribution = metadata.distribution("microunit")
        else:
            _require(
                type(microunit_search_path) is tuple
                and all(type(p) is str for p in microunit_search_path),
                "MICROUNIT_SEARCH_PATH",
            )
            matches = [
                d
                for d in metadata.distributions(path=microunit_search_path)
                if d.metadata.get("Name", "").lower() == "microunit"
            ]
            _require(len(matches) == 1, "MICROUNIT_DISTRIBUTION_COUNT")
            distribution = matches[0]
        _require(distribution.version == MICROUNIT_VERSION, "MICROUNIT_VERSION")
        relative = "microunit/tax_unit_construction.py"
        _require(
            distribution.files is not None
            and any(str(f) == relative for f in distribution.files),
            "MICROUNIT_MODULE_INVENTORY",
        )
        path = distribution.locate_file(relative)
        _require(path.stat().st_size <= 1024 * 1024, "MICROUNIT_MODULE_SIZE")
        module = path.read_bytes()
        _require(
            len(module) <= 1024 * 1024 and _sha(module) == MICROUNIT_SHA256,
            "MICROUNIT_MODULE_SHA256",
        )
    except (metadata.PackageNotFoundError, OSError) as exc:
        raise MoneyRefusalError("MICROUNIT_UNAVAILABLE") from exc
    return packaged_current_money_resources(
        inspected_version=distribution.version, inspected_module_sha256=_sha(module)
    )
