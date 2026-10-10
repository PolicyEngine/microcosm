"""The modules whose source decides UK target values.

Both UK target kernels (the full graph's ``_TargetKernel`` and the national
graph's ``_NationalKernel``) hash this one tuple, so an edit to any module
that compiles, uprates or reconciles a target re-keys every node that
consumes the compiled surface. Before microcosm#1123 each kernel listed its
own modules and neither listed the shared resolver (``build.ledger_targets``,
where ``linear_combination`` and the period rules live) nor the HMRC uprating
appliers, so an edit there could reuse a cached surface computed by the old
code.

``EXCLUDED_DIRECT_IMPORTS`` names every ``microcosm.build`` module a listed
module imports but the tuple leaves out, with the reason. A test pins that
the two together cover every direct import of every listed module, so the
coverage holds transitively: a new import must be listed or excluded.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import MappingProxyType, ModuleType

UK_TARGET_COMPILE_MODULE_NAMES: tuple[str, ...] = (
    "microcosm.build.cross_grain",
    "microcosm.build.ledger_targets",
    "microcosm.build.target_materialization",
    "microcosm.build.target_reference_authoring",
    "microcosm.build.uk_runtime.cgt_calibration",
    "microcosm.build.uk_runtime.chronicle_feed",
    "microcosm.build.uk_runtime.cross_grain_declarations",
    "microcosm.build.uk_runtime.geography_ladder",
    "microcosm.build.uk_runtime.hmrc_uprating",
    "microcosm.build.uk_runtime.ledger_fact_vendoring",
    "microcosm.build.uk_runtime.ledger_targets",
    "microcosm.build.uk_runtime.local_target_census",
    "microcosm.build.uk_runtime.local_targets",
    "microcosm.build.uk_runtime.national_reconciliation",
    "microcosm.build.uk_runtime.tenure_constants",
    "microcosm.build.uk_runtime.uc_relationships",
    "microcosm.build.uk_runtime.uc_source_periods",
    "microcosm.build.uk_runtime.tenure_drift",
    "microcosm.build.uk_runtime.uprating_holds",
)

EXCLUDED_DIRECT_IMPORTS: Mapping[str, str] = MappingProxyType(
    {
        "microcosm.build.country_spec": (
            "the kernels hash the UK country-spec fingerprint, which covers "
            "every packaged resource the spec loads"
        ),
        "microcosm.build.uk_runtime.frs_relationships": (
            "imported only for the ons_household_type value domain, which "
            "refuses an undeclared condition and never moves a value"
        ),
        "microcosm.build.chronicle_epoch": (
            "the feed-pin epoch arithmetic chronicle_feed checks; the pinned "
            "facts themselves are node inputs hashed by content"
        ),
        "microcosm.build.gates": (
            "imported for the GateResult and TargetCoverageRequirement types only"
        ),
        "microcosm.build.uk_runtime.fiscal_targets": (
            "cgt_calibration imports UK_CGT_REQUIRED_COLUMNS for the spine-side "
            "CGT calibration; the compile uses only uk_cgt_annual_exempt_amount"
        ),
        "microcosm.build.uk_runtime.national_frame": (
            "cgt_calibration imports the frame's weight-kind and period helpers "
            "for the spine-side CGT calibration, not for any target value"
        ),
        "microcosm.build.uk_runtime.local_authority_input": (
            "geography_ladder imports the local-authority vintage constants and "
            "the engine-key resolver for the ladder gate and the engine column"
        ),
        "microcosm.build.uk_runtime.rowwise_geography": (
            "geography_ladder imports the fixed FRS-region code map "
            "FRS_REGION_TO_REGION_CODE, pinned by its own tests"
        ),
        "microcosm.build.uk_runtime.hmrc_income": (
            "local_target_census imports HMRC_SPI_TARGET_RECORD_COUNT for the "
            "census text only"
        ),
        "microcosm.build.uk_runtime.hmrc_replay": (
            "local_target_census imports FULL_FRS_TI_BAND_FENCE_ID for the "
            "census text only"
        ),
    }
)


def uk_target_compile_modules() -> tuple[ModuleType, ...]:
    """The modules of :data:`UK_TARGET_COMPILE_MODULE_NAMES`, imported."""

    return tuple(
        importlib.import_module(name) for name in UK_TARGET_COMPILE_MODULE_NAMES
    )
