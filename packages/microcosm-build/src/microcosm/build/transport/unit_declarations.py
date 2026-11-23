"""CREATE version identities and their shared unit-declaration parser.

This module depends only on the Frame declaration framework, so population
kernels, target compilation, composition and CLI preparation share one contract
without importing each other's execution or registration modules.
"""

from collections.abc import Mapping

from microcosm.frame.unit_construction import BenefitUnitRule, MultiUnitRule

__all__ = [
    "CREATE_V1_REF",
    "CREATE_V2_REF",
    "CREATE_REFS",
    "create_unit_declaration",
]

CREATE_V1_REF = "transport.create@1"
CREATE_V2_REF = "transport.create@2"
CREATE_REFS = frozenset({CREATE_V1_REF, CREATE_V2_REF})


def create_unit_declaration(kernel: str, document: Mapping) -> MultiUnitRule:
    """Lift legacy benefit declarations or read the versioned declaration list."""
    if kernel == CREATE_V1_REF:
        rule = BenefitUnitRule.from_spec(document)
        if rule.entity != "family":
            raise ValueError("CREATE's declared schema requires family benefit units.")
        return MultiUnitRule.from_benefit_unit(rule)
    if kernel == CREATE_V2_REF:
        return MultiUnitRule.from_spec(document)
    raise ValueError(f"Unknown transport CREATE kernel {kernel!r}.")
