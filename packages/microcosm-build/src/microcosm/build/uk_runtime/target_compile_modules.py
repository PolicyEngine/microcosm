"""The modules whose source decides UK target values.

Both UK target kernels (the full graph's ``_TargetKernel`` and the national
graph's ``_NationalKernel``) hash this one tuple, so an edit to any module
that compiles, uprates or reconciles a target re-keys every node that
consumes the compiled surface. Before microcosm#1123 each kernel listed its
own modules and neither listed the shared resolver (``build.ledger_targets``,
where ``linear_combination`` and the period rules live) nor the HMRC uprating
appliers, so an edit there could reuse a cached surface computed by the old
code.

``EXCLUDED_DIRECT_IMPORTS`` names every first-party (``microcosm.*``) module
a listed module imports but the tuple leaves out, with the reason. A test
pins that the two together cover every direct import of every listed module,
so the coverage holds transitively: a new import must be listed or excluded.
Until microcosm#1492 that test read only ``microcosm.build`` imports, so it
never saw ``microcosm.calibrate.geography_constants``, the geography labels
of the target hierarchies, nor the ``microcosm.calibrate`` registry and
hierarchy classes the compiled surface is built from (with
``microcosm.calibrate.target``, which the registry imports). The transport
compile modules already hashed those three.

The exclusions cover only the listed modules' own imports. A second test
walks every module each target and problem kernel hashes, its own module and
the modules hashed beside this tuple included, and records the imports no
review has classified yet (microcosm#1497). This module is not hashed, so an
edit to a reason re-keys nothing; an edit to the tuple re-keys the kernels.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import MappingProxyType, ModuleType

UK_TARGET_COMPILE_MODULE_NAMES: tuple[str, ...] = (
    "microcosm.build.cross_grain",
    "microcosm.build.ledger_artifact",
    "microcosm.build.ledger_targets",
    "microcosm.build.target_materialization",
    "microcosm.build.target_reference_authoring",
    "microcosm.build.uk_runtime.bus_fare_pricing",
    "microcosm.build.uk_runtime.cgt_calibration",
    "microcosm.build.uk_runtime.chronicle_feed",
    "microcosm.build.uk_runtime.council_tax_constants",
    "microcosm.build.uk_runtime.cross_grain_declarations",
    "microcosm.build.uk_runtime.derived_facts",
    "microcosm.build.uk_runtime.desnz_constituency_energy",
    "microcosm.build.uk_runtime.domestic_energy_targets",
    "microcosm.build.uk_runtime.energy_pricing",
    "microcosm.build.uk_runtime.frs_release",
    "microcosm.build.uk_runtime.geography_ladder",
    "microcosm.build.uk_runtime.high_value_homes",
    "microcosm.build.uk_runtime.hmrc_uprating",
    "microcosm.build.uk_runtime.ledger_fact_vendoring",
    "microcosm.build.uk_runtime.ledger_surface",
    "microcosm.build.uk_runtime.ledger_targets",
    "microcosm.build.uk_runtime.local_target_census",
    "microcosm.build.uk_runtime.local_targets",
    "microcosm.build.uk_runtime.national_reconciliation",
    "microcosm.build.uk_runtime.off_surface_cells",
    "microcosm.build.uk_runtime.scotland_brma_la_overlap",
    "microcosm.build.uk_runtime.tenure_constants",
    "microcosm.build.uk_runtime.uc_relationships",
    "microcosm.build.uk_runtime.uc_source_periods",
    "microcosm.build.uk_runtime.tenure_drift",
    "microcosm.build.uk_runtime.uprating_holds",
    "microcosm.calibrate.geography_constants",
    "microcosm.calibrate.hierarchy",
    "microcosm.calibrate.registry",
    "microcosm.calibrate.target",
)

EXCLUDED_DIRECT_IMPORTS: Mapping[str, str] = MappingProxyType(
    {
        "microcosm.build.country_spec": (
            "the full graph's kernels hash the UK solve fingerprint, every "
            "packaged resource but the gate-only registers, which "
            "test_uk_solve_scope pins that no compile module reads "
            "(microcosm#1301); the national graph's hash the full fingerprint"
        ),
        "microcosm.build.uk_runtime.frs_relationships": (
            "imported only for the ons_household_type value domain, which "
            "refuses an undeclared condition and never moves a value"
        ),
        "microcosm.build.chronicle_epoch": (
            "the feed-pin epoch arithmetic chronicle_feed checks, and the "
            "consumer schema gate and epoch labels ledger_artifact applies; "
            "none moves a fact value, and the pinned facts themselves are node "
            "inputs hashed by content"
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
            "geography_ladder and bus_fare_pricing import the fixed FRS-region "
            "code map FRS_REGION_TO_REGION_CODE, pinned by its own tests"
        ),
        "microcosm.build.uk_runtime.hmrc_income": (
            "local_target_census imports HMRC_SPI_TARGET_RECORD_COUNT for the "
            "census text only"
        ),
        "microcosm.build.uk_runtime.hmrc_replay": (
            "local_target_census imports FULL_FRS_TI_BAND_FENCE_ID for the "
            "census text only"
        ),
        "microcosm.build.raking": (
            "energy_pricing imports the proportional fit for the spine's NEED "
            "rake; the compile reads only its published level and shares"
        ),
        "microcosm.build.source_manifest": (
            "energy_pricing imports SourceStageSpec as a type; the declared "
            "stage is part of the country-spec fingerprint"
        ),
        "microcosm.build.stochastic_assignment": (
            "energy_pricing imports the identity-keyed uniforms for the "
            "spine's gas connection walk, which no target value reads"
        ),
        "microcosm.frame.bundle": (
            "the population container: cgt_calibration builds the spine-side "
            "CGT calibration frame with it, ledger_targets' UKFrameTargetAdapter "
            "reads and rebuilds the national calibration's population with it, "
            "and calibrate.target measures a target on one; no target value "
            "comes from it"
        ),
        "microcosm.frame.schema": (
            "cgt_calibration imports EntitySchema for the spine-side CGT "
            "calibration frame; the compile uses only uk_cgt_annual_exempt_amount "
            "and uk_cgt_dated_measure_year"
        ),
        "microcosm.frame.weights": (
            "cgt_calibration imports Weights for the spine-side CGT calibration "
            "frame; the compile uses only uk_cgt_annual_exempt_amount and "
            "uk_cgt_dated_measure_year"
        ),
    }
)


def uk_target_compile_modules() -> tuple[ModuleType, ...]:
    """The modules of :data:`UK_TARGET_COMPILE_MODULE_NAMES`, imported."""

    return tuple(
        importlib.import_module(name) for name in UK_TARGET_COMPILE_MODULE_NAMES
    )
