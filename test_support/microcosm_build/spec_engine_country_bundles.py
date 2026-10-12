"""Shared-core country compile proofs for the F0 CountrySpec seam."""

# ruff: noqa: F401

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from microcosm.build.country_spec import country_stage_plan, load_country_spec
from microcosm.build.spec_engine import load_bundle
from microcosm.build.spec_engine.compiler_ir import compile_spec
from microcosm.build.spec_engine.resolver import (
    F0_CONTRACT_ONLY_KERNEL_IDS,
    F0_IMPLEMENTED_KERNEL_IDS,
    F0_KERNEL_REGISTRY,
    KernelRegistry,
)

EXPECTED_RESOURCES = {
    "bundle",
    "catalogs",
    "geography",
    "sources",
    "spine",
    "vintages",
}

__all__ = [name for name in globals() if not name.startswith("__")]
