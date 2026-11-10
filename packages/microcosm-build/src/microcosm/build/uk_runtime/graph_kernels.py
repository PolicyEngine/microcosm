"""Kernel family for the UK FRS spine graph.

Real-build kernels reconstruct a minimal UK :class:`~microcosm.frame.Frame`
from only their declared ``KernelContext.tables``, invoke the existing stage
transform unchanged, and project the owned cells back out.  Structural stages
return new-row source lineage through :attr:`KernelResult.expand`, cell
overlays through ``columns``, and explicit weights; the graph executor carries
the rows and remaps memberships.

Production execution binds transforms through the caller.  The hermetic H2
bundle carries a data-only marker from which an otherwise unbound registry
lazily reconstructs those same transforms.  Without that marker, unbound stage
kernels still fail explicitly rather than replaying recorded deltas: such a
replay would make parity agree with itself.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from microcosm.calibrate import geography_constants
from microcosm.fit import categorical as fit_categorical
from microcosm.fit import donors as fit_donors
from microcosm.fit import held_out as fit_held_out
from microcosm.fit import model as fit_model
from microcosm.fit import qrf as fit_qrf
from microcosm.frame import Frame, MassChangeRecord, WeightKind
from microcosm.frame import materialize as frame_materialize
from microcosm.frame import rules as frame_rules
from microcosm.frame.adapters import engine_population
from microcosm.frame.adapters import policyengine_uk as uk_engine_adapter
from microcosm.frame.adapters import (
    policyengine_uk_population as uk_engine_population_declaration,
)
from microcosm.graph import (
    Capabilities,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Numeric,
    SeedSource,
    StructuralDelta,
    load_source,
    source_hash,
)
from microcosm.graph import canonical as graph_canonical
from microcosm.graph import randomness as graph_randomness
from microcosm.graph.population import dtype_for_token

from .. import (
    chronicle_epoch,
    gates,
    ledger_targets,
    raking,
    source_lineage,
    stage_evidence,
    stochastic_assignment,
    target_materialization,
    trace,
)
from . import (
    advani_summers,
    annuity_trust_column,
    birth_month,
    bus_fare_pricing,
    bus_support_pricing,
    bus_use_incidence,
    cgt_asset_type,
    cgt_imputation,
    cgt_structure,
    cgt_support,
    chronicle_feed,
    copy_id_table,
    council_tax_constants,
    dfe_spring_term,
    donor_uprating,
    employer_pension_contributions,
    energy_pricing,
    etb_vat,
    frs_annual_income,
    frs_brma,
    frs_disability,
    frs_education,
    frs_hmrc_source,
    frs_release,
    frs_spine,
    frs_take_up,
    gambling_participation,
    graph_frames,
    hmrc_capital_gains,
    hmrc_company_car_statistics,
    hmrc_income,
    hmrc_isa_statistics,
    hmrc_property_rental,
    hmrc_replay,
    hmrc_source_contract,
    hmrc_uprating,
    household_income,
    household_lineage,
    ledger_fact_vendoring,
    legacy_closure_take_up,
    national_frame,
    nts_car_attributes,
    occupation_constants,
    pension_coherence,
    property_localisation,
    qrf_gates,
    receipt_counts,
    release_input_coverage,
    rowwise_geography,
    spi_band_donors,
    spi_income,
    spi_rank_transport,
    spi_support,
    stage_health,
    support_clip,
    take_up_contract,
    tenure_constants,
    terminal_gates,
    uc_capital_coherence,
    uc_deduction_attributes,
    uc_relationships,
    was_wealth,
)
from .graph_frames import cast_owned, normalize_create_frame
from .national_frame import UK_NATIONAL_SCHEMA

if TYPE_CHECKING:
    from microcosm.build.source_manifest import SourceStageSpec

__all__ = [
    "UKClaimKernel",
    "UKCreateKernel",
    "UKExpandStageKernel",
    "UKIdentityKernel",
    "UKStageKernel",
    "build_uk_registry",
    "fixture_stage_plan_inputs",
]


_STAGE_MODULES = {
    "frs_spine": "frs_spine",
    "frs_relationships": "frs_relationships",
    "frs_employment": "frs_employment",
    "frs_council_tax": "frs_council_tax",
    "frs_disability": "frs_disability",
    "frs_education": "frs_education",
    "frs_legacy_proxies": "frs_legacy_proxies",
    "frs_education_grant_split": "frs_education_grants",
    "frs_take_up": "frs_take_up",
    "frs_person_draws": "frs_person_draws",
    "frs_household_draws": "frs_household_draws",
    "frs_brma": "frs_brma",
    "was_wealth": "was_wealth",
    "was_lisa": "was_lisa",
    "nts_bus_travel": "nts_bus_travel",
    "regional_property_uprating": "regional_uprating",
    "property_components": "property_components",
    "property_purchases": "property_purchases",
    "lcfs_consumption": "lcfs_consumption",
    "etb_vat": "etb_vat",
    "etb_services": "etb_services",
    "frs_hmrc_spine_leaves": "spi_spine",
    "frs_annual_income_spells": "frs_annual_income",
    "spi_support_channel": "spi_spine",
    "spi_income_band_donors": "spi_band_donors",
    "hmrc_spi_income_spine": "spi_spine",
    "spi_employment_shell": "spi_employment_shell",
    "aps_occupation": "aps_occupation",
    "spi_housing_shell": "spi_housing_shell",
    "uc_reporter_redraw": "uc_reporter_redraw",
    "spi_benefit_coherence": "spi_benefit_coherence",
    "uc_capital_coherence": "uc_capital_coherence",
    "uc_take_up": "uc_take_up",
    "uc_legacy_closure_take_up": "legacy_closure_take_up",
    "pension_credit_take_up": "pension_credit_take_up",
    "child_benefit_take_up": "child_benefit_take_up",
    "carers_allowance_take_up": "carers_allowance_take_up",
    "uc_deduction_attributes": "uc_deduction_attributes",
    "high_value_home_support": "high_value_home_support",
    "regional_home_values": "regional_home_values",
    "cgt_support_split": "cgt_support",
    "cgt_incidence_clone": "cgt_structure",
    "cgt_incidence_anchor": "cgt_structure",
    "hmrc_cgt_gains_spine": "cgt_imputation",
    "cgt_residential_split": "cgt_residential_split",
    "hmrc_cgt_asset_type_spine": "cgt_asset_type",
    "salary_sacrifice": "salary_sacrifice",
    "student_loans": "student_loans",
    "private_school_attendance": "private_school_attendance",
    "isa_subscriptions": "isa_subscriptions",
    "trading_allowance_receipts": "trading_allowance_receipts",
    "company_cars": "company_cars",
    "childcare_copy_take_up": "childcare_copy_take_up",
    "age_tail": "age_tail",
}

# What every stage that calls the engine computes through: the adapter, which
# runs a call whose frame carries several clones one clone at a time, and the
# shared engine-block machinery with the UK declaration it judges each batch
# against (microcosm#931). An edit to any of them moves the hash of every such
# stage; test_uk_graph derives the stages from their engine calls and from
# their transforms' engine parameter. Every stage also hashes the adapter
# beside national_frame (_implementation_hash).
_ENGINE_MODULES = (
    uk_engine_adapter,
    engine_population,
    uk_engine_population_declaration,
)
# What else the injected engine computes through when those stages call it:
# the adapter builds each simulation's dataset through frame.materialize's
# engine_tables. test_uk_graph derives the group from a second walk, rooted at
# the PolicyEngineUKEngine the builds inject and followed from the methods the
# engine stages call on it (microcosm#931). It is kept apart from
# _ENGINE_MODULES, which no other stage binds, because the SPI spine's three
# stages and the band donors bind frame.materialize for their own tables.
_ENGINE_COMPUTE_MODULES = (frame_materialize,)
# What every stage that calls the engine binds.
_ENGINE_STAGE_MODULES = (*_ENGINE_MODULES, *_ENGINE_COMPUTE_MODULES)

# Imported modules are not traversed by ``source_hash``, so each stage binds
# every first-party module holding code its transform computes through,
# beyond its own module (microcosm#931). test_uk_graph derives that set: it
# follows what the transform names into the module defining it, in any
# workspace package, to a fixed point, and for a stage that calls the engine,
# what the methods the stage calls on the injected engine name. It needs no
# binding for a listed refusal-only check, nor for the core modules every node
# runs through (the Frame, the graph executor and the modules that compute
# node keys), which it lists with their reasons. Where it stops at a single
# string, a column-name function or a record, the module is hashed by the
# stage or a stage upstream of it, or the name is listed with its reason; a
# collection of strings, like a number, is a value.
# Every stage whose module or bound helpers import the shared keyed-draw
# helpers (``microcosm.build.stochastic_assignment``), or import another
# module's function that draws through them, binds them as well, so an edit
# to how a uniform is keyed or a binary is placed re-keys every stage it
# moves; test_uk_graph derives that set from the imports and names.

# A vendored ledger fact is read through ledger_fact_vendoring, which checks
# it against the pinned chronicle feed and the feed's accepted schema epochs.
_FEED_FACT_MODULES = (ledger_fact_vendoring, chronicle_feed, chronicle_epoch)
# A pinned source tab is read through frs_spine's tab reader, which checks the
# tab's digest through trace.
_PINNED_TAB_MODULES = (frs_spine, trace)
# The QRF donor stages fit and draw through microcosm.fit's regime-gated QRF,
# which resolves its fit columns and weights through fit.model. The
# categorical fitters resolve theirs through fit.model too, and their held-out
# scores and folds use the QRF's gate calibration and folds (microcosm#931).
_QRF_MODULES = (fit_qrf, fit_model)

_STAGE_HELPER_MODULES = {
    # The three FRS draw stages' receipts measure their take-up flags through
    # frs_take_up's signal measurement and count support by the shared
    # disclosure rules (microcosm#1095). The flags draw on the take-up
    # contract's rates and age children by the DfE's spring-term rule
    # (microcosm#931).
    "frs_take_up": (
        dfe_spring_term,
        receipt_counts,
        take_up_contract,
        stochastic_assignment,
    ),
    "frs_person_draws": (
        dfe_spring_term,
        frs_take_up,
        receipt_counts,
        take_up_contract,
        stochastic_assignment,
    ),
    "frs_household_draws": (
        dfe_spring_term,
        frs_take_up,
        receipt_counts,
        take_up_contract,
        stochastic_assignment,
    ),
    # The FRS stages that read raw tabs and normalise their ids through
    # frs_spine; the employment stage maps occupations to the shared codes, and
    # the disability flags resolve the benefit year's rules from the FRS
    # release and annualise on frs_spine's week count (microcosm#931).
    "frs_relationships": (*_PINNED_TAB_MODULES, stochastic_assignment),
    "frs_employment": (
        *_PINNED_TAB_MODULES,
        occupation_constants,
        stochastic_assignment,
    ),
    "frs_council_tax": (*_PINNED_TAB_MODULES, stochastic_assignment),
    "frs_disability": (frs_release, frs_spine, stochastic_assignment),
    "frs_education": (*_PINNED_TAB_MODULES, stochastic_assignment),
    # The leaves' employer pension contributions are drawn on keyed uniforms
    # through employer_pension_contributions, from the ASHE rates it reads as
    # vendored facts; the weekly leaves annualise on frs_spine's week count
    # (microcosm#931). Their module, spi_spine, also mints the support
    # channel's copies, so the leaves and the SPI income stage bind the copy-id
    # table as the support channel does. Its three stages, like the band
    # donors, copy their frames' tables through microcosm.frame's
    # engine_tables (microcosm#931).
    "frs_hmrc_spine_leaves": (
        frs_hmrc_source,
        copy_id_table,
        employer_pension_contributions,
        *_FEED_FACT_MODULES,
        frs_spine,
        frame_materialize,
        stochastic_assignment,
    ),
    # The annual income descriptors' receipt follows the shared disclosure rules
    # (microcosm#1164); the spells read their tabs and interview dates through
    # frs_spine (microcosm#931).
    "frs_annual_income_spells": (
        receipt_counts,
        *_PINNED_TAB_MODULES,
        stochastic_assignment,
    ),
    # The SPI income stage's imputation, band-donor resample and receipt live
    # in spi_income, so an edit there moves the stage's hash (microcosm#1278),
    # and its twin transport in spi_rank_transport (microcosm#840). The
    # imputation refreshes the disability flags, coheres the pension
    # contributions on the ASHE rate bands, uprates savings income, floors pay
    # on frs_spine's week count and reads its age model and band targets as
    # vendored facts and HMRC tables; the stage's parameters bound the band
    # donors and the coverage floor, and its result carries the fits' weight
    # kinds, gate outcomes and the HMRC replay report (microcosm#931). It
    # binds the copy-id table as the leaves do. The imputation refreshes the
    # annual income descriptors on frs_annual_income's share columns, and the
    # evidence shares iterate HMRC's distributional inputs. It zeros a drawn
    # row's annuity and trust descriptor, whose name it reads from
    # annuity_trust_column, as the root does (microcosm#931, ruled 9 Nov
    # (late)).
    "hmrc_spi_income_spine": (
        spi_income,
        spi_rank_transport,
        annuity_trust_column,
        copy_id_table,
        employer_pension_contributions,
        *_FEED_FACT_MODULES,
        frs_annual_income,
        frs_disability,
        frs_release,
        frs_spine,
        frs_take_up,
        gates,
        hmrc_income,
        hmrc_replay,
        hmrc_source_contract,
        hmrc_uprating,
        pension_coherence,
        qrf_gates,
        release_input_coverage,
        spi_band_donors,
        spi_support,
        target_materialization,
        frame_materialize,
        stochastic_assignment,
    ),
    # The spine's raw ids sit below the copy-id table's first field, and every
    # copy layer mints its ids in its own field of the table; the two SPI
    # support stages mint through the support channel's clone helper, and the
    # high-value split weighs its copies through the CGT split's exact-total
    # correction (microcosm#931). The executor reads each id-offset stage's
    # declared radix from the table too. The root reads its tabs through
    # trace's digest, ages people by birth month, caps pension contributions
    # with their receipt, and codes council tax bands in the shared band order
    # (microcosm#931). It writes the annuity and trust descriptor under the
    # name annuity_trust_column holds (microcosm#931, ruled 9 Nov (late)).
    "frs_spine": (
        uc_relationships,
        annuity_trust_column,
        copy_id_table,
        birth_month,
        council_tax_constants,
        pension_coherence,
        receipt_counts,
        trace,
        stochastic_assignment,
    ),
    # The support channel draws its keyed share and bounds its zero-weight
    # gate through the keyed-draw helpers, reads the take-up population's ages
    # and gates its zero-weight strata (microcosm#931).
    "spi_support_channel": (
        spi_support,
        copy_id_table,
        frs_take_up,
        gates,
        terminal_gates,
        frame_materialize,
        stochastic_assignment,
    ),
    # The band donors read HMRC's ITL band taxpayers as vendored facts and the
    # donors' age model through spi_income (microcosm#931).
    "spi_income_band_donors": (
        spi_support,
        copy_id_table,
        *_FEED_FACT_MODULES,
        frs_take_up,
        spi_income,
        frame_materialize,
        stochastic_assignment,
    ),
    # The exact-total correction the CGT split hands the high-value split is
    # spi_support's, and the support stratum is property_localisation's, whose
    # owners are the tenure map's (microcosm#931). spi_support draws the SPI
    # share through the keyed-draw helpers, so the stages that bind it bind
    # them too.
    "high_value_home_support": (
        cgt_support,
        copy_id_table,
        property_localisation,
        spi_support,
        tenure_constants,
        stochastic_assignment,
    ),
    # The CGT stages read HMRC's CGT tables as vendored facts and the
    # Advani-Summers tail from advani_summers, take the CGT policy parameters
    # and income proxy from cgt_imputation (whose loss relief evaluates a named
    # term), and weigh their copies through spi_support's exact-total
    # correction; the support split also takes the adult age and oldest-adult
    # rule from cgt_structure, and the residential split its logistic from
    # cgt_asset_type; the incidence clone sums cgt_imputation's income proxy
    # components into the income behind its propensity; the gains spine names
    # HMRC's regional conditioning rows by microcosm.calibrate's UK region tier,
    # and it and the asset types key each walk's offset on its label through
    # the shared keyed draws (microcosm#931).
    "cgt_support_split": (
        copy_id_table,
        *_FEED_FACT_MODULES,
        cgt_imputation,
        cgt_structure,
        hmrc_capital_gains,
        hmrc_income,
        spi_support,
        target_materialization,
        stochastic_assignment,
    ),
    "cgt_incidence_clone": (
        rowwise_geography,
        copy_id_table,
        advani_summers,
        cgt_imputation,
        spi_support,
        stochastic_assignment,
    ),
    "cgt_incidence_anchor": (
        rowwise_geography,
        copy_id_table,
        advani_summers,
        cgt_imputation,
        hmrc_capital_gains,
        stochastic_assignment,
    ),
    "cgt_residential_split": (
        copy_id_table,
        *_FEED_FACT_MODULES,
        cgt_asset_type,
        cgt_imputation,
        hmrc_capital_gains,
        spi_support,
        stochastic_assignment,
    ),
    "hmrc_cgt_gains_spine": (
        *_FEED_FACT_MODULES,
        advani_summers,
        geography_constants,
        hmrc_capital_gains,
        hmrc_income,
        target_materialization,
        stochastic_assignment,
    ),
    "hmrc_cgt_asset_type_spine": (
        *_FEED_FACT_MODULES,
        cgt_imputation,
        hmrc_capital_gains,
        stochastic_assignment,
    ),
    # The legacy proxies read raw tabs, and the grant split resolves its
    # year's rules from the FRS release (microcosm#931).
    "frs_legacy_proxies": (
        *_ENGINE_STAGE_MODULES,
        *_PINNED_TAB_MODULES,
        stochastic_assignment,
    ),
    "frs_education_grant_split": (*_ENGINE_STAGE_MODULES, frs_release),
    "frs_brma": (*_ENGINE_STAGE_MODULES, stochastic_assignment),
    # The donor stages read their donor tabs through frs_spine, record each
    # fit's weight kind, gate the fits and clip donor support to the donor's
    # range (microcosm#931); the WAS recipients' predictors map benefit units to
    # households, count family roles, group tenure by the shared tenure map and
    # name the HBAI income predictor. The WAS stage, and the LISA, bus, LCFS,
    # ETB, landlord, reporter and housing stages below, fit through the QRF
    # modules (microcosm#931).
    "was_wealth": (
        *_ENGINE_STAGE_MODULES,
        *_PINNED_TAB_MODULES,
        *_QRF_MODULES,
        frs_brma,
        gates,
        household_income,
        qrf_gates,
        support_clip,
        tenure_constants,
        uc_relationships,
        stochastic_assignment,
    ),
    # The LISA stage reads its household donor through the WAS cleaning and its
    # household predictors through the WAS recipient surface; the anchor counts
    # HMRC's payers through the typed ISA facts (microcosm#1415).
    "was_lisa": (
        *_ENGINE_STAGE_MODULES,
        was_wealth,
        hmrc_isa_statistics,
        *_PINNED_TAB_MODULES,
        *_FEED_FACT_MODULES,
        *_QRF_MODULES,
        frs_brma,
        gates,
        household_income,
        support_clip,
        tenure_constants,
        uc_relationships,
        stochastic_assignment,
    ),
    # Landlords' receipts read the vendored PRIS rows through their typed
    # loader, the tape through the income stage's preparation and uprating,
    # and the support channel's names (microcosm#1106); the loader checks the
    # feed, the preparation reads the donors' age model over the take-up
    # population and uprates savings income, and the fits record their weight
    # kinds (microcosm#931).
    "property_components": (
        hmrc_property_rental,
        spi_income,
        spi_support,
        *_FEED_FACT_MODULES,
        *_QRF_MODULES,
        frs_take_up,
        gates,
        hmrc_uprating,
        stochastic_assignment,
    ),
    # The regional values rank as the dense node does and read its knots
    # (microcosm#1112), over the tenure map's owners; their keyed uniforms are
    # microcosm.graph's, which encodes each draw's coordinates as canonical
    # JSON (microcosm#931).
    "regional_home_values": (
        property_localisation,
        rowwise_geography,
        tenure_constants,
        graph_randomness,
        graph_canonical,
    ),
    # The purchases read the vendored HMRC Stamp Tax rows through the feed-pinned
    # reader and mark SDLT's regions from the FRS region map (microcosm#1095).
    "property_purchases": (
        *_FEED_FACT_MODULES,
        rowwise_geography,
        stochastic_assignment,
    ),
    # The bus travel donors also read published car availability as vendored
    # facts, and the recipients' HBAI gross income (microcosm#931).
    "nts_bus_travel": (
        *_ENGINE_STAGE_MODULES,
        bus_use_incidence,
        *_PINNED_TAB_MODULES,
        *_FEED_FACT_MODULES,
        *_QRF_MODULES,
        gates,
        household_income,
        qrf_gates,
        support_clip,
        stochastic_assignment,
    ),
    # The energy level and prices are energy_pricing's (microcosm#1418): a
    # code-only change to them must re-key the stage (the independent review
    # of microcosm#1420, finding 5). The LCFS and ETB donors are uprated
    # through donor_uprating on vendored indices, the recipients' household
    # income and family roles are predictors, and bus fares are priced over
    # calendar and fiscal-year windows; the LCFS stage also rakes its energy,
    # assigns car attributes, tops up gambling and reads the DfT boardings of
    # each recipient's region through the FRS region map (microcosm#931).
    "lcfs_consumption": (
        *_ENGINE_STAGE_MODULES,
        energy_pricing,
        *_PINNED_TAB_MODULES,
        *_FEED_FACT_MODULES,
        *_QRF_MODULES,
        bus_fare_pricing,
        donor_uprating,
        gambling_participation,
        gates,
        household_income,
        ledger_targets,
        nts_car_attributes,
        qrf_gates,
        raking,
        rowwise_geography,
        support_clip,
        stochastic_assignment,
    ),
    "etb_vat": (
        *_ENGINE_STAGE_MODULES,
        *_PINNED_TAB_MODULES,
        *_FEED_FACT_MODULES,
        *_QRF_MODULES,
        donor_uprating,
        gates,
        household_income,
        support_clip,
        uc_relationships,
        stochastic_assignment,
    ),
    # The services stage counts its recipients' family roles as the VAT stage
    # does, from etb_vat's role table (microcosm#931).
    "etb_services": (
        *_ENGINE_STAGE_MODULES,
        *_PINNED_TAB_MODULES,
        *_FEED_FACT_MODULES,
        *_QRF_MODULES,
        bus_fare_pricing,
        bus_support_pricing,
        donor_uprating,
        energy_pricing,
        etb_vat,
        gates,
        ledger_targets,
        qrf_gates,
        support_clip,
        uc_relationships,
        stochastic_assignment,
    ),
    # The reporter redraw gates its fits and maps household weights to benefit
    # units as the UC capital stage does (microcosm#931).
    "uc_reporter_redraw": (
        uc_relationships,
        *_ENGINE_STAGE_MODULES,
        *_QRF_MODULES,
        qrf_gates,
        uc_capital_coherence,
        stochastic_assignment,
    ),
    # The SPI benefit pass restores from the FRS twin through the support
    # channel's lineage, re-derives the disability flags and the own-right
    # rule, draws over the take-up population and maps household weights to
    # benefit units as the UC capital stage does (microcosm#1095); the flags
    # resolve their year's rules and week count, and the draws read the
    # take-up contract (microcosm#931).
    "spi_benefit_coherence": (
        spi_support,
        frs_disability,
        frs_education,
        frs_take_up,
        uc_capital_coherence,
        frs_release,
        frs_spine,
        take_up_contract,
        stochastic_assignment,
    ),
    # Unavailable capital is frs_spine's sentinel (microcosm#931).
    "uc_capital_coherence": (uc_relationships, frs_spine, stochastic_assignment),
    # The Universal Credit redraw reads the award through the adapter, the
    # vendored DWP counts, the closure stage's rule and rates, the deduction
    # attributes, the take-up population and the family-type roles, and counts
    # its pools and receipt on the source lineage (microcosm#1334); the
    # attributes map benefit units to households and judge their realisation
    # by the shared tolerance, and the year's rules come from the FRS release;
    # it types its engine as the adapter's ProjectingRulesEngine, a protocol
    # that extends microcosm.frame.rules' RulesEngine (microcosm#931).
    "uc_take_up": (
        *_ENGINE_STAGE_MODULES,
        frame_rules,
        *_FEED_FACT_MODULES,
        legacy_closure_take_up,
        uc_deduction_attributes,
        frs_take_up,
        uc_relationships,
        uc_capital_coherence,
        receipt_counts,
        frs_brma,
        frs_release,
        stage_health,
        stochastic_assignment,
    ),
    # The Move to UC claim draw reuses the UC stage's household-to-benefit-unit
    # weight mapping and reads the vendored DWP rows (microcosm#1095).
    "uc_legacy_closure_take_up": (
        uc_capital_coherence,
        *_FEED_FACT_MODULES,
        stochastic_assignment,
    ),
    # The Pension Credit redraw reuses the UC stage's household-to-benefit-unit
    # weight mapping and the adapter's engine materialization; it reads the
    # take-up contract and the year's rules, and types its engine by the same
    # protocol as the UC redraw (microcosm#931).
    "pension_credit_take_up": (
        uc_capital_coherence,
        *_ENGINE_STAGE_MODULES,
        frame_rules,
        frs_release,
        take_up_contract,
        stochastic_assignment,
    ),
    # The Child Benefit redraw shares the weight mapping and the adapter, and
    # reads the vendored Child Benefit statistics (microcosm#931).
    "child_benefit_take_up": (
        uc_capital_coherence,
        *_ENGINE_STAGE_MODULES,
        *_FEED_FACT_MODULES,
        stochastic_assignment,
    ),
    # The Carer's Allowance draw reads the earnings test through the adapter's
    # engine materialization and counts its receipt's support on the source
    # lineage (microcosm#1319), and balances its draws through the shared
    # placement helper (microcosm#1418); it reads the take-up contract and the
    # year's rules (microcosm#931).
    "carers_allowance_take_up": (
        *_ENGINE_STAGE_MODULES,
        receipt_counts,
        frs_release,
        take_up_contract,
        stochastic_assignment,
    ),
    # The stored attendance is one engine materialization (microcosm#1095),
    # whose year's rules come from the FRS release (microcosm#931).
    "private_school_attendance": (*_ENGINE_STAGE_MODULES, frs_release),
    # The housing shell keeps council tax bands within the nation, whose
    # regions are microcosm.calibrate's UK region tier, and gates its fits;
    # the deduction attributes map benefit units to households and
    # judge their realisation by the shared tolerance; the salary sacrifice
    # reads its bonus anchor as vendored facts and gates its fits; the
    # student loans resolve their year's rules (microcosm#931).
    "spi_housing_shell": (
        council_tax_constants,
        geography_constants,
        *_QRF_MODULES,
        qrf_gates,
        stochastic_assignment,
    ),
    "uc_deduction_attributes": (frs_brma, stage_health, stochastic_assignment),
    "salary_sacrifice": (*_FEED_FACT_MODULES, qrf_gates, stochastic_assignment),
    "student_loans": (frs_release, stochastic_assignment),
    # The subscriptions walk the CGT bands on its income proxy (with the
    # proxy's loss relief), read HMRC's ISA tables through their typed loader
    # and round their receipts' support (microcosm#1415); the loss relief
    # evaluates its named term, the donor tabs are pinned, and the regions are
    # microcosm.calibrate's UK region tier (microcosm#931).
    "isa_subscriptions": (
        cgt_imputation,
        geography_constants,
        hmrc_income,
        hmrc_isa_statistics,
        receipt_counts,
        *_PINNED_TAB_MODULES,
        *_FEED_FACT_MODULES,
        target_materialization,
        stochastic_assignment,
    ),
    # The claimant draw is keyed on identity and its receipt rounds its
    # support (microcosm#1428).
    "trading_allowance_receipts": (receipt_counts, stochastic_assignment),
    # The company cars read their carriers' names and codes from frs_spine,
    # HMRC's company car tables through their typed loader, the engine's CPI
    # index and contribution cap, the SPI channel's EPB and tape-row flag, and
    # price the cars twice through the engine adapter; their draws are keyed on
    # identity, and their receipts' support is rounded
    # (PolicyEngine/uk-autumn-budget-dashboard-2026#316). The loader checks the
    # feed, and the donor tab is pinned (microcosm#931).
    "company_cars": (
        frs_spine,
        hmrc_company_car_statistics,
        hmrc_uprating,
        receipt_counts,
        spi_income,
        spi_support,
        trace,
        *_FEED_FACT_MODULES,
        stochastic_assignment,
        *_ENGINE_STAGE_MODULES,
    ),
    # The copies' childcare re-draw draws through frs_take_up's childcare
    # helper, reads each household's clone path, and measures its flags for
    # the stage-time signal gate by the shared disclosure rules (microcosm#1095);
    # it maps benefit units to households, ages children by the DfE's rule and
    # reads the take-up contract (microcosm#931).
    "childcare_copy_take_up": (
        frs_take_up,
        household_lineage,
        receipt_counts,
        dfe_spring_term,
        frs_brma,
        take_up_contract,
        stochastic_assignment,
    ),
    # The employment shell fits, scores and draws through microcosm.fit's
    # categorical classifier, held-out scores and keyed donors, reads the
    # shared occupation codes, and counts its receipt's support by the shared
    # disclosure rules (microcosm#1490); the classifier and held-out scores
    # compute through the QRF modules (microcosm#931).
    "spi_employment_shell": (
        fit_categorical,
        fit_donors,
        fit_held_out,
        *_QRF_MODULES,
        occupation_constants,
        receipt_counts,
        stochastic_assignment,
    ),
    # The sub-major draw fits and scores through microcosm.fit's categorical
    # classifier and held-out scores, reads the shared occupation codes, and
    # counts its receipt's support by the shared disclosure rules
    # (microcosm#1490); it reads its donor tab through trace's digest and
    # records its fit's weight kind, and its classifier and held-out scores
    # compute through the QRF modules (microcosm#931).
    "aps_occupation": (
        fit_categorical,
        fit_held_out,
        *_QRF_MODULES,
        occupation_constants,
        receipt_counts,
        gates,
        trace,
        stochastic_assignment,
    ),
}

# The receipts' counting rules and the shared lineage readers: every floor
# counts true source households through receipt_counts, and the FRS-trained
# fits fold geographic clones through household_lineage, both reading
# source_lineage on household_lineage's declaration (microcosm#931). An edit
# to any of the three moves the hash of every stage whose module or helpers
# import one of them; test_uk_graph derives this set from the imports.
_COUNTING_MODULES = (receipt_counts, household_lineage, source_lineage)
_COUNTING_STAGES = frozenset(
    {
        "frs_spine",
        "frs_take_up",
        "frs_person_draws",
        "frs_household_draws",
        # spi_spine, stage 14's module, rounds the support channel's receipt
        # (microcosm#931).
        "frs_hmrc_spine_leaves",
        "frs_annual_income_spells",
        "hmrc_spi_income_spine",
        "spi_support_channel",
        "spi_income_band_donors",
        "company_cars",
        "spi_housing_shell",
        "high_value_home_support",
        "regional_home_values",
        "property_components",
        "property_purchases",
        "nts_bus_travel",
        "lcfs_consumption",
        "etb_vat",
        "etb_services",
        "uc_reporter_redraw",
        "spi_benefit_coherence",
        "uc_capital_coherence",
        "uc_take_up",
        "uc_legacy_closure_take_up",
        "pension_credit_take_up",
        "child_benefit_take_up",
        "salary_sacrifice",
        "carers_allowance_take_up",
        "childcare_copy_take_up",
        "uc_deduction_attributes",
        "isa_subscriptions",
        "trading_allowance_receipts",
        "spi_employment_shell",
        "aps_occupation",
    }
)

_COMPUTE = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
)
_FILTER = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
    structural=StructuralDelta.FILTER,
)
_CREATE = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
    structural=StructuralDelta.CREATE,
)
_EXPAND = Capabilities(
    determinism=Determinism.DETERMINISTIC,
    numeric=Numeric.BITWISE,
    seed_source=SeedSource.NONE,
    structural=StructuralDelta.EXPAND,
)


def _stage_module(stage: str):
    try:
        leaf = _STAGE_MODULES[stage]
    except KeyError as error:
        raise ValueError(
            f"No implementation module is registered for {stage!r}."
        ) from error
    return importlib.import_module(f"microcosm.build.uk_runtime.{leaf}")


def _implementation_hash(kernel: object, stage: str, transform: object | None) -> str:
    from . import graph_evidence

    # The stage module is the behavior-bearing source in every mode. Hashing
    # an injected transform's dynamic test wrapper would make
    # hermetic registries unhashable and, more importantly, would fail to bind
    # production edits made elsewhere in that stage's module. A transform may
    # still declare extra behavior-bearing sources through
    # ``graph_implementation_dependencies``. Every kernel rebuilds its frame
    # on national_frame's schema, and the transforms build, validate and read
    # the time period of their frames through it (microcosm#931). It stamps
    # each frame it builds with the engine adapter's ISA income contract and
    # savings income projection, so every kernel hashes the adapter with it.
    # Every kernel casts the cells it owns, and the root normalises the frame
    # it creates, through graph_frames.
    dependencies = getattr(transform, "graph_implementation_dependencies", None)
    return source_hash(
        type(kernel),
        stage_evidence,
        graph_evidence,
        national_frame,
        uk_engine_adapter,
        graph_frames,
        _stage_artifacts,
        _mass_log_payload,
        _stage_module(stage),
        *_STAGE_HELPER_MODULES.get(stage, ()),
        *(_COUNTING_MODULES if stage in _COUNTING_STAGES else ()),
        *(dependencies() if callable(dependencies) else ()),
    )


def _mass_log_payload(before: Frame, after: Frame) -> list[dict[str, object]]:
    prefix_length = len(before.mass_log)
    if after.mass_log[:prefix_length] != before.mass_log:
        raise ValueError("A UK stage replaced its incumbent Frame mass records.")
    return [
        {
            "entity": record.entity,
            "old_total": record.old_total,
            "new_total": record.new_total,
            "declared_factor": record.declared_factor,
            "reason": record.reason,
        }
        for record in after.mass_log[prefix_length:]
    ]


def _stage_artifacts(
    stage: str,
    transform: object | None,
    before: Frame | None,
    after: Frame,
    *,
    release: bool = False,
) -> dict[str, bytes]:
    document = stage_evidence.snapshot_stage_evidence(stage, transform)
    document["frame_mass_log_append"] = (
        [
            {
                "entity": record.entity,
                "old_total": record.old_total,
                "new_total": record.new_total,
                "declared_factor": record.declared_factor,
                "reason": record.reason,
            }
            for record in after.mass_log
        ]
        if before is None
        else _mass_log_payload(before, after)
    )
    if before is None:
        document["frame_context"] = {"metadata": dict(after.metadata)}
    artifacts = {"stage_evidence": stage_evidence.encode_stage_evidence(document)}
    if release:
        # The bytes above are all that later steps read of this stage, so the
        # result it kept, output Frame included, need not live for the run.
        stage_evidence.release_stage_results(transform)
    return artifacts


def _invoke_transform(transform: object, frame: Frame, context: KernelContext):
    """Invoke a stage, giving context-bound adapters only declared sources."""

    run_with_sources = getattr(transform, "run_with_sources", None)
    if callable(run_with_sources):
        return run_with_sources(frame, context.sources)
    return transform(frame)


def _json_mapping(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read UK parity fixture {label} at {path}.") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"UK parity fixture {label} at {path} must be an object.")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"UK parity fixture {label} must be an object.")
    return value


def _fixture_input(
    source: Path,
    inputs: Mapping[str, object],
    name: str,
) -> Path:
    relative = inputs.get(name)
    if not isinstance(relative, str) or not relative:
        raise ValueError(
            f"UK parity fixture inputs.{name} must be a non-empty relative path."
        )
    if Path(relative).is_absolute():
        raise ValueError(f"UK parity fixture inputs.{name} must be relative.")
    root = source.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"UK parity fixture inputs.{name} escapes its source directory."
        ) from error
    if not path.exists():
        raise ValueError(f"UK parity fixture input {name!r} is absent at {path}.")
    return path


def _fixture_hmrc_income_targets(path: Path):
    from .hmrc_income import (
        HMRCIncomeBandTargetRecord,
        HMRCIncomeSourceProvenance,
        HMRCIncomeTargetSet,
    )

    payload = _json_mapping(path, label="HMRC income targets")
    raw_source = dict(_mapping(payload.get("source"), label="HMRC source"))
    local_path = raw_source.get("local_path")
    table_names = raw_source.get("table_names")
    if not isinstance(local_path, str):
        raise ValueError("UK parity fixture HMRC source.local_path must be a string.")
    if not isinstance(table_names, list) or not all(
        isinstance(value, str) for value in table_names
    ):
        raise ValueError(
            "UK parity fixture HMRC source.table_names must be a string list."
        )
    raw_source["local_path"] = Path(local_path)
    raw_source["table_names"] = tuple(table_names)
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list):
        raise ValueError("UK parity fixture HMRC targets must be a list.")
    return HMRCIncomeTargetSet(
        source=HMRCIncomeSourceProvenance(**raw_source),
        targets=tuple(
            HMRCIncomeBandTargetRecord(
                **dict(_mapping(value, label="HMRC target record"))
            )
            for value in raw_targets
        ),
    )


def _fixture_cgt_distribution(path: Path):
    from .hmrc_capital_gains import (
        HMRCCapitalGainsBandTotal,
        HMRCCapitalGainsCell,
        HMRCCapitalGainsIncomeTotal,
        HMRCCapitalGainsJointDistribution,
        HMRCCapitalGainsSourceProvenance,
    )

    payload = _json_mapping(path, label="CGT distribution")
    raw_source = dict(_mapping(payload.get("source"), label="CGT source"))

    def records(name: str) -> list[Mapping[str, object]]:
        raw = payload.get(name)
        if not isinstance(raw, list):
            raise ValueError(f"UK parity fixture CGT {name} must be a list.")
        return [_mapping(value, label=f"CGT {name} record") for value in raw]

    return HMRCCapitalGainsJointDistribution(
        cells=tuple(HMRCCapitalGainsCell(**dict(value)) for value in records("cells")),
        band_totals=tuple(
            HMRCCapitalGainsBandTotal(**dict(value)) for value in records("band_totals")
        ),
        income_totals=tuple(
            HMRCCapitalGainsIncomeTotal(**dict(value))
            for value in records("income_totals")
        ),
        source=HMRCCapitalGainsSourceProvenance(**raw_source),
        total_individuals=float(payload["total_individuals"]),
        total_gains=float(payload["total_gains"]),
    )


def _fixture_property_rental_facts(path: Path):
    from .hmrc_property_rental import HMRCPropertyRentalBand, HMRCPropertyRentalFacts

    payload = dict(_json_mapping(path, label="PRIS property rental facts"))
    raw_bands = payload.pop("receipts_bands")
    if not isinstance(raw_bands, list):
        raise ValueError("UK parity fixture PRIS receipts_bands must be a list.")
    return HMRCPropertyRentalFacts(
        **payload,
        receipts_bands=tuple(
            HMRCPropertyRentalBand(**dict(_mapping(row, label="Table 13 band")))
            for row in raw_bands
        ),
    )


def _fixture_asset_type_facts(path: Path):
    from .cgt_asset_type import (
        HMRCCGTAssetTypeFacts,
        HMRCCGTBADRBand,
        HMRCCGTTable7Type,
    )

    payload = dict(_json_mapping(path, label="CGT asset-type facts"))
    raw_rows = payload.pop("table7_types")
    raw_bands = payload.pop("table4_bands")
    if not isinstance(raw_rows, list) or not isinstance(raw_bands, list):
        raise ValueError(
            "UK parity fixture CGT asset-type table7_types and table4_bands must "
            "be lists."
        )
    return HMRCCGTAssetTypeFacts(
        **payload,
        table7_types=tuple(
            HMRCCGTTable7Type(**dict(_mapping(row, label="Table 7 row")))
            for row in raw_rows
        ),
        table4_bands=tuple(
            HMRCCGTBADRBand(**dict(_mapping(row, label="Table 4.1 band")))
            for row in raw_bands
        ),
    )


def _fixture_descriptor(
    source: Path,
) -> tuple[Mapping[str, object], dict[str, SourceStageSpec]]:
    """Parse the H2 bundle's descriptor and its stage specs, keyed by name."""

    from microcosm.build.source_manifest import SourceStageSpec

    descriptor = _json_mapping(source / "fixture.json", label="descriptor")
    if descriptor.get("schema_version") != "uk-spine-parity-fixture.v1":
        raise ValueError(
            "UK parity fixture schema_version must be 'uk-spine-parity-fixture.v1'."
        )
    stage_payloads = _mapping(descriptor.get("stages"), label="stages")
    stages: dict[str, SourceStageSpec] = {}
    for name, payload in stage_payloads.items():
        if not isinstance(name, str):
            raise ValueError("UK parity fixture stage names must be strings.")
        stage = SourceStageSpec.from_mapping(_mapping(payload, label=f"stage {name!r}"))
        if stage.stage != name:
            raise ValueError(
                f"UK parity fixture stage key {name!r} contains {stage.stage!r}."
            )
        stages[name] = stage
    if set(stages) != set(_STAGE_MODULES):
        missing = sorted(set(_STAGE_MODULES) - set(stages))
        extra = sorted(set(stages) - set(_STAGE_MODULES))
        raise ValueError(
            "UK parity fixture must describe the current 39-stage spine "
            f"(missing={missing}, extra={extra})."
        )
    return descriptor, stages


def _fixture_implementations(source: Path) -> Mapping[str, object]:
    """Reconstruct current transforms from one data-only H2 descriptor."""

    from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

    from .age_tail import UKAgeTailStageTransform
    from .aps_occupation import UKAPSOccupationStageTransform
    from .carers_allowance_take_up import UKCarersAllowanceTakeUpStageTransform
    from .cgt_asset_type import UKCGTAssetTypeStageTransform, UKCGTBADRParameters
    from .cgt_imputation import UKCGTPolicyParameters, uk_cgt_spine_stage_transform
    from .cgt_residential_split import UKCGTResidentialSplitStageTransform
    from .cgt_structure import (
        UKCGTIncidenceAnchorStageTransform,
        UKCGTIncidenceCloneStageTransform,
    )
    from .cgt_support import UKCGTSupportSplitStageTransform
    from .child_benefit_take_up import UKChildBenefitTakeUpStageTransform
    from .childcare_copy_take_up import UKChildcareCopyTakeUpStageTransform
    from .company_cars import UKCompanyCarsStageTransform
    from .etb_services import UKETBServicesStageTransform
    from .etb_vat import UKETBVATStageTransform
    from .frs_annual_income import UKFRSAnnualIncomeStageTransform
    from .frs_brma import UKFRSBRMAStageTransform
    from .frs_council_tax import UKFRSCouncilTaxStageTransform
    from .frs_disability import UKFRSDisabilityStageTransform
    from .frs_education import UKFRSEducationStageTransform
    from .frs_education_grants import UKFRSEducationGrantSplitStageTransform
    from .frs_employment import UKFRSEmploymentStageTransform
    from .frs_household_draws import UKFRSHouseholdDrawsStageTransform
    from .frs_legacy_proxies import UKFRSLegacyProxiesStageTransform
    from .frs_person_draws import UKFRSPersonDrawsStageTransform
    from .frs_relationships import UKFRSRelationshipsStageTransform
    from .frs_take_up import UKFRSTakeUpStageTransform
    from .high_value_home_support import UKHighValueHomeSupportStageTransform
    from .isa_subscriptions import UKISASubscriptionsStageTransform
    from .lcfs_consumption import UKLCFSConsumptionStageTransform
    from .legacy_closure_take_up import UKLegacyClosureTakeUpStageTransform
    from .nts_bus_travel import UKNTSBusTravelStageTransform
    from .pension_credit_take_up import UKPensionCreditTakeUpStageTransform
    from .private_school_attendance import UKPrivateSchoolAttendanceStageTransform
    from .property_components import UKPropertyComponentsStageTransform
    from .property_purchases import (
        UKPropertyPurchasesStageTransform,
        UKPropertyPurchaseTargets,
    )
    from .regional_home_values import UKRegionalHomeValuesStageTransform
    from .regional_uprating import UKRegionalPropertyUpratingStageTransform
    from .salary_sacrifice import UKSalarySacrificeStageTransform
    from .spi_band_donors import UKSPIIncomeBandDonorStageTransform
    from .spi_benefit_coherence import UKSPIBenefitCoherenceStageTransform
    from .spi_employment_shell import UKSPIEmploymentShellStageTransform
    from .spi_housing_shell import UKSPIHousingShellStageTransform
    from .spi_spine import (
        UKFRSHMRCSpineLeavesStageTransform,
        UKSPIIncomeSpineStageTransform,
        UKSPISupportChannelStageTransform,
    )
    from .student_loans import UKStudentLoansStageTransform
    from .take_up_contract import load_uk_take_up_contract
    from .trading_allowance_receipts import UKTradingAllowanceReceiptsStageTransform
    from .uc_capital_coherence import UKUCCapitalCoherenceStageTransform
    from .uc_deduction_attributes import UKUCDeductionAttributesStageTransform
    from .uc_reporter_redraw import UKUCReporterRedrawStageTransform
    from .uc_take_up import UKUCTakeUpStageTransform
    from .was_lisa import UKWASLISAStageTransform
    from .was_wealth import UKWASWealthStageTransform

    descriptor, stages = _fixture_descriptor(source)

    config = _mapping(descriptor.get("config"), label="config")
    inputs = _mapping(descriptor.get("inputs"), label="inputs")
    raw_dir = _fixture_input(source, inputs, "frs_raw")
    was = pd.read_csv(
        _fixture_input(source, inputs, "was"), float_precision="round_trip"
    )
    was_person = pd.read_csv(
        _fixture_input(source, inputs, "was_person"), float_precision="round_trip"
    )
    lcfs_household = pd.read_csv(
        _fixture_input(source, inputs, "lcfs_household"),
        float_precision="round_trip",
    )
    lcfs_person = pd.read_csv(
        _fixture_input(source, inputs, "lcfs_person"),
        float_precision="round_trip",
    )
    etb = pd.read_csv(
        _fixture_input(source, inputs, "etb"), float_precision="round_trip"
    )
    aps = pd.read_csv(
        _fixture_input(source, inputs, "aps"), float_precision="round_trip"
    )
    nts_household = pd.read_csv(
        _fixture_input(source, inputs, "nts_household"), float_precision="round_trip"
    )
    nts_individual = pd.read_csv(
        _fixture_input(source, inputs, "nts_individual"), float_precision="round_trip"
    )
    nts_trip = pd.read_csv(
        _fixture_input(source, inputs, "nts_trip"), float_precision="round_trip"
    )
    nts_stage = pd.read_csv(
        _fixture_input(source, inputs, "nts_stage"), float_precision="round_trip"
    )
    nts_vehicle = pd.read_csv(
        _fixture_input(source, inputs, "nts_vehicle"), float_precision="round_trip"
    )
    nts_ticket = pd.read_csv(
        _fixture_input(source, inputs, "nts_ticket"), float_precision="round_trip"
    )
    spi_path = _fixture_input(source, inputs, "spi_donor")
    spi_donor = pd.read_csv(spi_path, float_precision="round_trip")
    hmrc_targets_path = _fixture_input(source, inputs, "hmrc_income_targets")
    income_targets = _fixture_hmrc_income_targets(hmrc_targets_path)
    cgt_distribution = _fixture_cgt_distribution(
        _fixture_input(source, inputs, "cgt_distribution")
    )
    cgt_asset_type_facts = _fixture_asset_type_facts(
        _fixture_input(source, inputs, "cgt_asset_type_facts")
    )
    property_rental_facts = _fixture_property_rental_facts(
        _fixture_input(source, inputs, "property_rental_facts")
    )
    property_purchase_targets = UKPropertyPurchaseTargets.from_payload(
        _json_mapping(
            _fixture_input(source, inputs, "property_purchase_targets"),
            label="property purchase targets",
        )
    )
    cgt_parameters = UKCGTPolicyParameters(
        **dict(_mapping(descriptor.get("cgt_parameters"), label="CGT parameters"))
    )
    cgt_badr_parameters = UKCGTBADRParameters(
        **dict(
            _mapping(descriptor.get("cgt_badr_parameters"), label="CGT BADR parameters")
        )
    )

    engine = PolicyEngineUKEngine()
    contract = load_uk_take_up_contract()
    qrf_estimators = int(config["qrf_estimators"])
    donor_sample_size = int(config["spi_donor_sample_size"])
    sample_fraction = float(config["spi_sample_fraction"])
    calibration_year = int(config["student_loans_calibration_year"])
    return MappingProxyType(
        {
            "frs_relationships": UKFRSRelationshipsStageTransform(
                raw_dir, stage=stages["frs_relationships"]
            ),
            "frs_employment": UKFRSEmploymentStageTransform(
                raw_dir, stage=stages["frs_employment"]
            ),
            "frs_council_tax": UKFRSCouncilTaxStageTransform(
                raw_dir, stage=stages["frs_council_tax"]
            ),
            "frs_disability": UKFRSDisabilityStageTransform(
                stage=stages["frs_disability"]
            ),
            "frs_education": UKFRSEducationStageTransform(
                raw_dir, stage=stages["frs_education"]
            ),
            "frs_legacy_proxies": UKFRSLegacyProxiesStageTransform(
                raw_dir,
                stage=stages["frs_legacy_proxies"],
                engine=engine,
            ),
            "frs_education_grant_split": UKFRSEducationGrantSplitStageTransform(
                stage=stages["frs_education_grant_split"], engine=engine
            ),
            "frs_take_up": UKFRSTakeUpStageTransform(
                contract=contract, stage=stages["frs_take_up"]
            ),
            "frs_person_draws": UKFRSPersonDrawsStageTransform(
                contract=contract, stage=stages["frs_person_draws"]
            ),
            "frs_household_draws": UKFRSHouseholdDrawsStageTransform(
                contract=contract, stage=stages["frs_household_draws"]
            ),
            "frs_brma": UKFRSBRMAStageTransform(
                stage=stages["frs_brma"], engine=engine
            ),
            "was_wealth": UKWASWealthStageTransform(
                stage=stages["was_wealth"], engine=engine, donor=was
            ),
            "was_lisa": UKWASLISAStageTransform(
                stage=stages["was_lisa"],
                engine=engine,
                donor_household=was,
                donor_person=was_person,
            ),
            "nts_bus_travel": UKNTSBusTravelStageTransform(
                stage=stages["nts_bus_travel"],
                engine=engine,
                nts_household=nts_household,
                nts_individual=nts_individual,
                nts_trip=nts_trip,
                nts_stage=nts_stage,
                nts_ticket=nts_ticket,
            ),
            "regional_property_uprating": UKRegionalPropertyUpratingStageTransform(
                stage=stages["regional_property_uprating"]
            ),
            "property_components": UKPropertyComponentsStageTransform(
                stage=stages["property_components"],
                spi_tab_path=spi_path,
                donor_table=spi_donor,
                facts=property_rental_facts,
            ),
            "property_purchases": UKPropertyPurchasesStageTransform(
                stage=stages["property_purchases"],
                targets=property_purchase_targets,
            ),
            "regional_home_values": UKRegionalHomeValuesStageTransform(
                stage=stages["regional_home_values"]
            ),
            "lcfs_consumption": UKLCFSConsumptionStageTransform(
                stage=stages["lcfs_consumption"],
                engine=engine,
                lcfs_household=lcfs_household,
                lcfs_person=lcfs_person,
                nts_household=nts_household,
                nts_vehicle=nts_vehicle,
            ),
            "etb_vat": UKETBVATStageTransform(
                stage=stages["etb_vat"], engine=engine, donor=etb
            ),
            "etb_services": UKETBServicesStageTransform(
                stage=stages["etb_services"], engine=engine, donor=etb
            ),
            "frs_hmrc_spine_leaves": UKFRSHMRCSpineLeavesStageTransform(
                raw_dir, stage=stages["frs_hmrc_spine_leaves"]
            ),
            "frs_annual_income_spells": UKFRSAnnualIncomeStageTransform(
                raw_dir, stage=stages["frs_annual_income_spells"]
            ),
            "spi_support_channel": UKSPISupportChannelStageTransform(
                stage=stages["spi_support_channel"],
                sample_fraction=sample_fraction,
            ),
            "spi_income_band_donors": UKSPIIncomeBandDonorStageTransform(
                spi_path,
                stage=stages["spi_income_band_donors"],
                sample_fraction=sample_fraction,
                donor_table=spi_donor,
            ),
            "hmrc_spi_income_spine": UKSPIIncomeSpineStageTransform(
                spi_path,
                hmrc_targets_path,
                stage=stages["hmrc_spi_income_spine"],
                qrf_estimators=qrf_estimators,
                donor_sample_size=donor_sample_size,
                sampled_rung=True,
                donor_table=spi_donor,
                source_targets=income_targets,
            ),
            "spi_employment_shell": UKSPIEmploymentShellStageTransform(
                stage=stages["spi_employment_shell"], boosted_max_iter=qrf_estimators
            ),
            "aps_occupation": UKAPSOccupationStageTransform(
                stage=stages["aps_occupation"],
                donor=aps,
                boosted_max_iter=qrf_estimators,
            ),
            "spi_housing_shell": UKSPIHousingShellStageTransform(
                stage=stages["spi_housing_shell"], n_estimators=qrf_estimators
            ),
            "uc_reporter_redraw": UKUCReporterRedrawStageTransform(
                stage=stages["uc_reporter_redraw"], engine=engine
            ),
            "spi_benefit_coherence": UKSPIBenefitCoherenceStageTransform(
                stage=stages["spi_benefit_coherence"], contract=contract
            ),
            "uc_capital_coherence": UKUCCapitalCoherenceStageTransform(
                stage=stages["uc_capital_coherence"]
            ),
            "uc_take_up": UKUCTakeUpStageTransform(
                stage=stages["uc_take_up"], engine=engine
            ),
            "uc_legacy_closure_take_up": UKLegacyClosureTakeUpStageTransform(
                stage=stages["uc_legacy_closure_take_up"]
            ),
            "pension_credit_take_up": UKPensionCreditTakeUpStageTransform(
                stage=stages["pension_credit_take_up"], engine=engine
            ),
            "child_benefit_take_up": UKChildBenefitTakeUpStageTransform(
                stage=stages["child_benefit_take_up"], engine=engine
            ),
            "carers_allowance_take_up": UKCarersAllowanceTakeUpStageTransform(
                stage=stages["carers_allowance_take_up"], engine=engine
            ),
            "uc_deduction_attributes": UKUCDeductionAttributesStageTransform(
                stage=stages["uc_deduction_attributes"]
            ),
            "high_value_home_support": UKHighValueHomeSupportStageTransform(
                stage=stages["high_value_home_support"]
            ),
            "cgt_support_split": UKCGTSupportSplitStageTransform(
                stage=stages["cgt_support_split"],
                distribution=cgt_distribution,
                parameters=cgt_parameters,
            ),
            "cgt_incidence_clone": UKCGTIncidenceCloneStageTransform(
                stage=stages["cgt_incidence_clone"]
            ),
            "hmrc_cgt_gains_spine": uk_cgt_spine_stage_transform(
                stages["hmrc_cgt_gains_spine"],
                distribution=cgt_distribution,
                parameters=cgt_parameters,
            ),
            "hmrc_cgt_asset_type_spine": UKCGTAssetTypeStageTransform(
                stage=stages["hmrc_cgt_asset_type_spine"],
                facts=cgt_asset_type_facts,
                parameters=cgt_parameters,
                badr_parameters=cgt_badr_parameters,
            ),
            "cgt_incidence_anchor": UKCGTIncidenceAnchorStageTransform(
                stage=stages["cgt_incidence_anchor"],
                parameters=cgt_parameters,
            ),
            "cgt_residential_split": UKCGTResidentialSplitStageTransform(
                stage=stages["cgt_residential_split"],
                facts=cgt_asset_type_facts,
                parameters=cgt_parameters,
            ),
            "salary_sacrifice": UKSalarySacrificeStageTransform(
                stage=stages["salary_sacrifice"]
            ),
            "student_loans": UKStudentLoansStageTransform(
                stage=stages["student_loans"], calibration_year=calibration_year
            ),
            "private_school_attendance": UKPrivateSchoolAttendanceStageTransform(
                stage=stages["private_school_attendance"],
                engine=engine,
                calibration_year=calibration_year,
            ),
            "isa_subscriptions": UKISASubscriptionsStageTransform(
                stage=stages["isa_subscriptions"],
                engine=engine,
                donor_household=was,
                donor_person=was_person,
            ),
            "trading_allowance_receipts": UKTradingAllowanceReceiptsStageTransform(
                stage=stages["trading_allowance_receipts"]
            ),
            "company_cars": UKCompanyCarsStageTransform(
                stage=stages["company_cars"],
                engine=engine,
                donor_person=was_person,
            ),
            "childcare_copy_take_up": UKChildcareCopyTakeUpStageTransform(
                contract=contract, stage=stages["childcare_copy_take_up"]
            ),
            "age_tail": UKAgeTailStageTransform(stage=stages["age_tail"]),
        }
    )


def fixture_stage_plan_inputs(
    source: Path,
) -> tuple[tuple[SourceStageSpec, ...], Mapping[str, object]]:
    """The H2 bundle as legacy StagePlan inputs: ordered specs, every transform.

    The hermetic registry's transforms (``_fixture_implementations``) carry
    no ``frs_spine`` transform, so its CREATE node, with no root transform
    bound, loads the captured root frame. This mapping adds one built from
    the same ``frs_raw`` tables: the legacy oracle applies it, and H2 parity
    binds the graph's CREATE kernel to it, the root's production path. The
    descriptor keys stages by name, so the committed UK manifest supplies the
    order the legacy plan runs them in.
    """

    from microcosm.build.country_spec import load_country_spec

    from .frs_spine import UKFRSSpineStageTransform

    descriptor, stages = _fixture_descriptor(source)
    committed = load_country_spec("uk")
    if committed.sources is None:
        raise ValueError("The committed UK country spec declares no sources.")
    ordered = tuple(
        stages[stage.stage]
        for stage in committed.sources.stages
        if stage.stage in stages
    )
    if len(ordered) != len(stages):
        unordered = sorted(set(stages) - {stage.stage for stage in ordered})
        raise ValueError(
            "UK parity fixture stages missing from the committed spine order: "
            f"{unordered}."
        )
    inputs = _mapping(descriptor.get("inputs"), label="inputs")
    raw_dir = _fixture_input(source, inputs, "frs_raw")
    implementations = dict(_fixture_implementations(source))
    implementations["frs_spine"] = UKFRSSpineStageTransform(
        raw_dir, stage=stages["frs_spine"]
    )
    return ordered, MappingProxyType(implementations)


class _FixtureTransformResolver:
    """Lazily bind the H2 fixture marker without weakening normal registries."""

    def __init__(self) -> None:
        self._by_source: dict[Path, Mapping[str, object]] = {}

    def resolve(self, stage: str, context: KernelContext) -> object | None:
        raw_source = context.sources.get("frs")
        if raw_source is None:
            return None
        source = Path(raw_source).resolve()
        if not (source / "fixture.json").is_file():
            return None
        implementations = self._by_source.get(source)
        if implementations is None:
            implementations = _fixture_implementations(source)
            self._by_source[source] = implementations
        try:
            return implementations[stage]
        except KeyError as error:
            raise ValueError(
                f"UK parity fixture has no transform for stage {stage!r}."
            ) from error


def _minimal_frame(context: KernelContext) -> Frame:
    missing = set(UK_NATIONAL_SCHEMA.entities) - context.tables.keys()
    if missing:
        raise ValueError(
            f"UK stage {context.node.id!r} lacks declared slices for {sorted(missing)}."
        )
    if "household" not in context.weights:
        raise ValueError(f"UK stage {context.node.id!r} lacks household weights.")
    mass_log: tuple[MassChangeRecord, ...] = ()
    if context.params.get("stage") == "hmrc_spi_income_spine":
        # KernelContext deliberately has no legacy Frame.mass_log channel.  The
        # SPI income transform consumes only the immediately preceding support
        # allocation reason, whose exact current-spine record is reconstructible
        # from the conserved household total and reviewed public reason, the
        # declared allocation's own sentence (microcosm#1069 c6).
        from microcosm.build.country_spec import load_country_spec

        from .spi_spine import uk_spi_support_mass_change_reason

        sources = load_country_spec("uk").sources
        if sources is None:
            raise ValueError("The committed UK country spec declares no sources.")
        reason = uk_spi_support_mass_change_reason(
            sources.stage_map()["spi_support_channel"]
        )
        total = context.weights["household"].total
        mass_log = (
            MassChangeRecord(
                entity="household",
                old_total=total,
                new_total=total,
                declared_factor=1.0,
                reason=reason,
            ),
        )
    return Frame(
        {
            entity: context.tables[entity].copy(deep=True)
            for entity in UK_NATIONAL_SCHEMA.entities
        },
        UK_NATIONAL_SCHEMA,
        {"household": context.weights["household"]},
        context.strata.copy(deep=True),
        mass_log=mass_log,
        metadata={"time_period": str(context.params.get("time_period", "2024"))},
    )


def _id_index(frame: Frame, entity: str) -> pd.Index:
    id_column = frame.schema.entity_id_column(entity)
    return pd.Index(frame.table(entity)[id_column].to_numpy(copy=True), name=id_column)


def _owned_series(frame: Frame, entity: str, column: str, dtype: str) -> pd.Series:
    values = cast_owned(frame.table(entity)[column].copy(deep=True), dtype)
    values.index = _id_index(frame, entity)
    values.name = column
    return values


class UKCreateKernel(KernelBase):
    """Load fixture CSV tables or invoke the bound FRS root transform."""

    ref = "uk.create@1"
    capabilities = _CREATE

    def __init__(
        self,
        transform: object | None = None,
        *,
        release_stage_results: bool = False,
    ) -> None:
        self.transform = transform
        self.release_stage_results = release_stage_results

    def implementation_hash(self) -> str:
        return _implementation_hash(self, "frs_spine", self.transform)

    def run(self, context: KernelContext) -> KernelResult:
        source = context.sources["frs"]
        if self.transform is None:
            frame = load_source("csv-tables", source)
        else:
            from .frs_spine import uk_frs_spine_seed_frame

            frame = _invoke_transform(
                self.transform, uk_frs_spine_seed_frame(), context
            )
        if not isinstance(frame, Frame):
            raise TypeError(
                f"The UK root transform returned {type(frame).__name__}, not Frame."
            )
        return KernelResult(
            frame=normalize_create_frame(frame, context),
            artifacts=_stage_artifacts(
                "frs_spine",
                self.transform,
                None,
                frame,
                release=self.release_stage_results,
            ),
        )


class UKIdentityKernel(KernelBase):
    """Keep every person, creating a new ownership/population boundary."""

    ref = "uk.identity@1"
    capabilities = _FILTER

    def implementation_hash(self) -> str:
        # The person ids it keeps are read on national_frame's schema
        # (microcosm#931).
        return source_hash(
            type(self), national_frame, dependencies=self.capabilities.dependencies
        )

    def run(self, context: KernelContext) -> KernelResult:
        person = context.tables[UK_NATIONAL_SCHEMA.person_entity]
        id_column = UK_NATIONAL_SCHEMA.person_id_column
        ids = pd.Index(person[id_column].to_numpy(copy=True), name=id_column)
        return KernelResult(keep=pd.Series(True, index=ids, dtype="bool"))


class UKClaimKernel(KernelBase):
    """Claim executor-materialized incumbent cells in a new population."""

    ref = "uk.claim@1"
    capabilities = _COMPUTE

    def implementation_hash(self) -> str:
        # The claimed cells are indexed by national_frame's schema's id
        # columns and cast through graph_frames (microcosm#931).
        return source_hash(
            type(self),
            national_frame,
            graph_frames,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        columns = {
            (owned.entity, owned.column): pd.Series(
                cast_owned(
                    context.tables[owned.entity][owned.column].copy(deep=True),
                    owned.dtype,
                ).array,
                index=pd.Index(
                    context.tables[owned.entity][
                        UK_NATIONAL_SCHEMA.entity_id_column(owned.entity)
                    ].to_numpy(copy=True),
                    name=UK_NATIONAL_SCHEMA.entity_id_column(owned.entity),
                ),
                name=owned.column,
                dtype=dtype_for_token(owned.dtype),
            )
            for owned in context.node.outputs
        }
        return KernelResult(columns=MappingProxyType(columns))


class UKStageKernel(KernelBase):
    """One ordinary UK stage, parameterized by its manifest stage name."""

    capabilities = _COMPUTE

    def __init__(
        self,
        stage: str,
        transform: object | None = None,
        fixture_resolver: _FixtureTransformResolver | None = None,
        *,
        release_stage_results: bool = False,
    ) -> None:
        self.stage = stage
        self.transform = transform
        self.fixture_resolver = fixture_resolver
        self.release_stage_results = release_stage_results
        self.ref = f"uk.stage.{stage}@1"

    def implementation_hash(self) -> str:
        return _implementation_hash(self, self.stage, self.transform)

    def run(self, context: KernelContext) -> KernelResult:
        from .graph_evidence import require_uk_spine_gate_admission

        require_uk_spine_gate_admission(context)
        transform = self.transform
        if transform is None and self.fixture_resolver is not None:
            transform = self.fixture_resolver.resolve(self.stage, context)
        if transform is None:
            raise RuntimeError(
                f"UK stage {self.stage!r} has no bound production transform; "
                "recorded fixture deltas are not accepted as parity evidence."
            )
        before = _minimal_frame(context)
        after = _invoke_transform(transform, before, context)
        if not isinstance(after, Frame):
            raise TypeError(
                f"UK stage {self.stage!r} returned {type(after).__name__}, not Frame."
            )
        columns = {
            (owned.entity, owned.column): _owned_series(
                after, owned.entity, owned.column, owned.dtype
            )
            for owned in context.node.outputs
        }
        return KernelResult(
            columns=MappingProxyType(columns),
            artifacts=_stage_artifacts(
                self.stage,
                transform,
                before,
                after,
                release=self.release_stage_results,
            ),
            receipt={
                "stage": self.stage,
                "frame_mass_log_append": _mass_log_payload(before, after),
            },
        )


def _expand_cells(context: KernelContext) -> tuple[tuple[str, str, str], ...]:
    raw = context.params.get("expand_cells")
    if not isinstance(raw, tuple):
        raise ValueError(f"UK EXPAND stage {context.node.id!r} has no cell contract.")
    return tuple(tuple(str(part) for part in item) for item in raw)  # type: ignore[arg-type]


#: The copy-id field each id-offset structural stage mints (microcosm#931):
#: copy ``k`` takes ``source + k x radix(field)`` on every id column, and
#: every incumbent id lies below the field's radix, so the copy index is the
#: integer quotient. The SPI support stages name each copy's source in
#: ``{entity}_source_id`` instead.
_COPY_ID_FIELDS = MappingProxyType(
    {
        "high_value_home_support": "high_value_support",
        "cgt_support_split": "cgt_support_split",
        "cgt_incidence_clone": "cgt_incidence_clone",
        "cgt_residential_split": "cgt_residential_split",
    }
)


def _declared_copy_radix(stage: str, before: Frame) -> int | None:
    """The declared radix an id-offset stage mints at, or None for the others.

    Every incumbent id and membership must sit below it, or the quotient
    would name the wrong parent; the refusal prints counts only.
    """

    field = _COPY_ID_FIELDS.get(stage)
    if field is None:
        return None
    person = before.table(before.schema.person_entity)
    columns = [
        (before.table(entity), before.schema.entity_id_column(entity))
        for entity in before.entities
    ] + [
        (person, before.schema.membership_column(group))
        for group in before.schema.group_entities
    ]
    for table, column in columns:
        copy_id_table.require_below(
            table[column],
            field,
            what=f"UK stage {stage!r} incumbent {column} values",
        )
    return copy_id_table.radix(field)


def _exact_ids(values: pd.Series, *, label: str) -> np.ndarray:
    """An id column as exact int64: ids reach 2**53 and above (microcosm#931)."""

    if not pd.api.types.is_integer_dtype(values.dtype):
        raise TypeError(f"{label} must be an integer id column, not {values.dtype}.")
    return values.to_numpy(dtype=np.int64)


def _source_lineage(
    before: Frame,
    after: Frame,
    entity: str,
    *,
    id_offset: int | None,
) -> pd.Series:
    """Derive new target-to-immediate-source ids from a structural result.

    Vectorised on exact int64 ids, so it is exact on any frame. A row-by-row
    walk reads a row whose columns hold only integers and floats as float64,
    which is inexact above 2**53 (an all-integer row stays int64, and a row
    with a string or boolean column comes back as objects). The refusals
    print counts, never an id.
    """

    id_column = before.schema.entity_id_column(entity)
    before_table = before.table(entity)
    after_table = after.table(entity)
    dtype = before_table[id_column].dtype
    before_ids = pd.Index(before_table[id_column])
    after_ids = _exact_ids(after_table[id_column], label=f"{entity}.{id_column}")
    new = ~pd.Index(after_ids).isin(before_ids)
    # A weights-only structural stage (the #970 incidence anchor) adds no
    # row: every target is an incumbent, so the lineage is empty.
    if not new.any():
        return pd.Series(
            [],
            index=pd.Index([], name=id_column, dtype=dtype),
            dtype=dtype,
            name=id_column,
        )
    targets = after_ids[new]
    source_column = f"{entity}_source_id"
    if id_offset is not None:
        # CGT and high-value stages retain long-lived source-id provenance
        # from the SPI support stage, so their immediate lineage is the
        # stage's declared copy-id radix, not that older provenance column.
        copy_index = targets // int(id_offset)
        sources = targets - copy_index * int(id_offset)
        unmatched = int(((copy_index < 1) | ~pd.Index(sources).isin(before_ids)).sum())
        if unmatched:
            raise ValueError(
                f"UK stage produced {unmatched} new {entity!r} row(s) whose id "
                "less its copy ordinal times the stage's declared copy-id radix "
                "is not an incumbent id."
            )
    elif source_column in after_table:
        # Only the new rows' sources are read, as the row walk read them.
        sources = _exact_ids(
            after_table[source_column][new], label=f"{entity}.{source_column}"
        )
        unmatched = int((~pd.Index(sources).isin(before_ids)).sum())
        if unmatched:
            raise ValueError(
                f"UK stage produced {unmatched} new {entity!r} row(s) whose "
                f"{source_column!r} is not an incumbent id."
            )
    else:
        raise ValueError(
            f"UK stage produced {int(new.sum())} new {entity!r} row(s) without an "
            f"explicit {source_column!r} or a declared copy-id lineage rule."
        )
    return pd.Series(
        sources,
        index=pd.Index(targets, name=id_column, dtype=dtype),
        dtype=dtype,
        name=id_column,
    )


class UKExpandStageKernel(KernelBase):
    """A UK row-expanding stage returning lineage rather than a Frame."""

    capabilities = _EXPAND

    def __init__(
        self,
        stage: str,
        transform: object | None = None,
        fixture_resolver: _FixtureTransformResolver | None = None,
        *,
        release_stage_results: bool = False,
    ) -> None:
        self.stage = stage
        self.transform = transform
        self.fixture_resolver = fixture_resolver
        self.release_stage_results = release_stage_results
        self.ref = f"uk.stage.expand.{stage}@1"

    def implementation_hash(self) -> str:
        return _implementation_hash(self, self.stage, self.transform)

    def run(self, context: KernelContext) -> KernelResult:
        from .graph_evidence import require_uk_spine_gate_admission

        require_uk_spine_gate_admission(context)
        transform = self.transform
        if transform is None and self.fixture_resolver is not None:
            transform = self.fixture_resolver.resolve(self.stage, context)
        if transform is None:
            raise RuntimeError(
                f"UK stage {self.stage!r} has no bound production transform; "
                "recorded fixture deltas are not accepted as parity evidence."
            )
        before = _minimal_frame(context)
        after = _invoke_transform(transform, before, context)
        if not isinstance(after, Frame):
            raise TypeError(
                f"UK stage {self.stage!r} returned {type(after).__name__}, not Frame."
            )
        cells = _expand_cells(context)
        id_offset = _declared_copy_radix(self.stage, before)
        expand: dict[str, pd.Series] = {}
        for entity in before.entities:
            expand[entity] = _source_lineage(
                before,
                after,
                entity,
                id_offset=id_offset,
            )
        columns: dict[tuple[str, str], pd.Series] = {}
        for entity, column, dtype in cells:
            columns[(entity, column)] = _owned_series(after, entity, column, dtype)
        weight_entity = str(context.params["expand_weight_entity"])
        after_weights = after.weights_for(weight_entity)
        declared_kind = WeightKind(str(context.params["expand_weight_kind"]))
        if after_weights.kind is not declared_kind:
            raise ValueError(
                f"UK EXPAND stage {self.stage!r} returned weight kind "
                f"{after_weights.kind.value!r}, not declared "
                f"{declared_kind.value!r}."
            )
        receipt: dict[str, object] = {
            "stage": self.stage,
            "frame_mass_log_append": _mass_log_payload(before, after),
        }
        if context.node.mass == "declared":
            receipt["mass"] = _declared_mass_receipt(
                before, after, stage=self.stage, weight_entity=weight_entity
            )
        return KernelResult(
            columns=MappingProxyType(columns),
            expand=MappingProxyType(expand),
            weights=after_weights,
            artifacts=_stage_artifacts(
                self.stage,
                transform,
                before,
                after,
                release=self.release_stage_results,
            ),
            receipt=receipt,
        )


#: Relative tolerance for the weight-entity mass invariant a ``declared`` UK
#: expansion must still hold; equals the executor's own ledger tolerance.
_WEIGHT_ENTITY_MASS_RTOL = 1e-9


def _declared_mass_receipt(
    before: Frame,
    after: Frame,
    *,
    stage: str,
    weight_entity: str,
) -> dict[str, object]:
    """State the person-mass ledger of a ``declared`` expansion.

    The executor's mass ledger is weighted person mass per stratum, which an
    expansion that changes household composition cannot conserve even when it
    conserves the mass of the entity it reweights.  A ``declared`` UK expansion
    therefore states the person-mass ledger for the executor to verify and
    asserts here the invariant that is actually its contract: the weight
    entity's total mass is unchanged.
    """

    before_entity = float(before.weights_for(weight_entity).total)
    after_entity = float(after.weights_for(weight_entity).total)
    if not np.isclose(
        after_entity, before_entity, rtol=_WEIGHT_ENTITY_MASS_RTOL, atol=0.0
    ):
        raise ValueError(
            f"UK EXPAND stage {stage!r} declares its person-mass change but must "
            f"conserve {weight_entity!r} mass: {before_entity!r} -> {after_entity!r}."
        )
    before_mass = before.stratum_mass()
    after_mass = after.stratum_mass()
    return {
        "policy": "declared",
        "before": float(before_mass.sum()),
        "after": float(after_mass.sum()),
        "stratum_before": {key: float(value) for key, value in before_mass.items()},
        "stratum_after": {key: float(value) for key, value in after_mass.items()},
        "weight_entity": weight_entity,
        "weight_entity_mass_before": before_entity,
        "weight_entity_mass_after": after_entity,
    }


def build_uk_registry(
    graph: Graph,
    implementations: Mapping[str, object],
    *,
    release_stage_results: bool = False,
) -> KernelRegistry:
    """Bind graph refs to supplied transforms or the hermetic H2 descriptor.

    With ``release_stage_results``, each stage kernel drops the result its
    transform kept as soon as the stage's evidence is captured
    (:func:`~microcosm.build.stage_evidence.release_stage_results`). The
    default keeps it, for a caller that reads ``last_result`` after the run.
    The setting enters no key: it changes what a live transform holds, never
    what a node computes or stores.
    """

    stage_names = {
        str(node.params["stage"]) for node in graph.nodes if "stage" in node.params
    }
    unknown = set(implementations) - (stage_names | {"frs_spine"})
    if unknown:
        raise ValueError(f"Unknown UK graph stage implementations: {sorted(unknown)}.")
    if implementations:
        missing = stage_names - implementations.keys()
        if missing:
            raise ValueError(
                f"Missing UK graph stage implementations: {sorted(missing)}."
            )

    registry = KernelRegistry()
    registry.register(
        UKCreateKernel(
            implementations.get("frs_spine"),
            release_stage_results=release_stage_results,
        )
    )
    registry.register(UKIdentityKernel())
    registry.register(UKClaimKernel())
    fixture_resolver = _FixtureTransformResolver()
    for stage in sorted(stage_names):
        transform = implementations.get(stage)
        if stage in {
            "spi_support_channel",
            "spi_income_band_donors",
            "high_value_home_support",
            "cgt_support_split",
            "cgt_incidence_clone",
            "cgt_incidence_anchor",
            "cgt_residential_split",
        }:
            registry.register(
                UKExpandStageKernel(
                    stage,
                    transform,
                    fixture_resolver,
                    release_stage_results=release_stage_results,
                )
            )
        else:
            registry.register(
                UKStageKernel(
                    stage,
                    transform,
                    fixture_resolver,
                    release_stage_results=release_stage_results,
                )
            )

    # Gate bindings carry the live rules engine and are registered separately
    # after population-stage construction by the composing build.
    required = {
        node.kernel for node in graph.nodes if node.kernel != "uk.spine-gates@1"
    }
    if set(registry.refs()) != required:
        missing = sorted(required - set(registry.refs()))
        extra = sorted(set(registry.refs()) - required)
        raise ValueError(
            f"UK registry does not exactly cover its graph (missing={missing}, "
            f"extra={extra})."
        )
    return registry
