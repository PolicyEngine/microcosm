"""The reviewed registers of the US fiscal target compile (microcosm#1728).

``us/fiscal_target_registers.json`` holds the registers that
:mod:`microcosm.build.us_runtime.fiscal_targets` and
:mod:`microcosm.build.us_runtime.source_vintage` read: the facts the compile
drops and why, the source columns a package reads under the wrong measure, the
restamped source packages, the SOI variable maps, the direct and indicator
targets, and the M-CHIP states. Each register keeps the comments that stood
beside it in the code as its description and notes. The shared
:mod:`microcosm.build.declared_registers` reads them.
"""

from __future__ import annotations

from typing import Any

from microcosm.build.declared_registers import package_register

__all__ = ["US_FISCAL_TARGET_REGISTERS_RESOURCE", "us_fiscal_target_register"]

US_FISCAL_TARGET_REGISTERS_RESOURCE = "fiscal_target_registers.json"


def us_fiscal_target_register(name: str) -> Any:
    """One register of the file: a dict in entry order, or a frozenset of keys."""

    return package_register(
        "microcosm.build.us", US_FISCAL_TARGET_REGISTERS_RESOURCE, name
    )
