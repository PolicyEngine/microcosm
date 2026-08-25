"""Immutable F0 seed protocols and the exhaustive legacy-v1 draw ledger.

F0 resolves this compiler-owned protocol from the authored
``seed_protocol: legacy-v1`` selector.  It does not broker or execute a draw;
that is F1.  The ledger is nevertheless normative now so every constants-era
site, seed derivation, ordering rule, and reset boundary enters the emitted
locks and compiled IR instead of surviving as ambient Python knowledge.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

from .canonical import canonical_json_bytes

MAX_SEED = 2**64 - 1


def _distribution_version(name: str) -> str:
    """Return the installed, lock-resolved distribution version."""

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError as error:  # pragma: no cover - packaging gate
        raise RuntimeError(
            f"seed protocol dependency is not installed: {name}"
        ) from error


def source_inventory_sha256(module_names: tuple[str, ...]) -> str:
    """Hash installed Python sources without including machine-local paths.

    The inventory hashes the logical module name and exact installed source
    bytes.  Loader origins are deliberately excluded, so an editable checkout,
    a clean virtual environment, and an installed wheel attest the same code.
    """

    rows: list[dict[str, str]] = []
    for module_name in module_names:
        if module_name.startswith("microcosm.build."):
            # Avoid importing ``microcosm.build.us_runtime`` while spec_engine
            # itself initializes.  This path is also valid in an installed
            # wheel; it is used only to read bytes and never enters the digest.
            relative = module_name.removeprefix("microcosm.build.").replace(".", "/")
            source = Path(__file__).resolve().parents[1] / f"{relative}.py"
            try:
                raw = source.read_bytes()
            except OSError as error:  # pragma: no cover - packaging gate
                raise RuntimeError(
                    f"cannot read seed-kernel source {module_name!r}"
                ) from error
        else:
            spec = find_spec(module_name)
            if spec is None or spec.origin is None or spec.loader is None:
                raise RuntimeError(f"cannot locate seed-kernel source {module_name!r}")
            get_data = getattr(spec.loader, "get_data", None)
            if get_data is None:
                raise RuntimeError(f"seed-kernel loader cannot read {module_name!r}")
            raw = get_data(spec.origin)
        rows.append(
            {
                "module": module_name,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return hashlib.sha256(canonical_json_bytes(rows)).hexdigest()


@dataclass(frozen=True, slots=True)
class KernelAttestation:
    """Path-independent digest of the code that consumes one seed contract."""

    id: str
    source_modules: tuple[str, ...]
    source_sha256: str

    def __post_init__(self) -> None:
        if not self.id or not self.source_modules:
            raise ValueError("kernel attestation requires an id and source modules")
        if self.source_modules != tuple(sorted(set(self.source_modules))):
            raise ValueError("kernel source modules must be sorted and unique")
        if len(self.source_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_sha256
        ):
            raise ValueError("kernel attestation requires a SHA-256 digest")

    def to_wire(self) -> dict[str, object]:
        return {
            "id": self.id,
            "source_modules": list(self.source_modules),
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class DrawSiteProtocol:
    """One deterministic draw-site contract in a selected seed protocol."""

    id: str
    stream: str
    value_source: str
    default: int | None
    rng_family: str
    rng_version: str
    kernel: str
    seed_material: tuple[str, ...]
    consumption_order: tuple[str, ...]
    reset_boundary: str
    draw_condition: str
    derivation: str

    def __post_init__(self) -> None:
        if self.default is not None and not 0 <= self.default <= MAX_SEED:
            raise ValueError(f"draw site {self.id!r} seed is outside uint64")
        if (
            not self.id
            or not self.stream
            or not self.rng_family
            or not self.rng_version
        ):
            raise ValueError("draw sites require ids, streams, and typed RNG metadata")
        if not self.kernel:
            raise ValueError(f"draw site {self.id!r} has no kernel attestation")
        if not self.seed_material or not self.consumption_order:
            raise ValueError(f"draw site {self.id!r} has an incomplete draw contract")

    def to_wire(self) -> dict[str, object]:
        return {
            "id": self.id,
            "stream": f"stream:{self.stream}",
            "value_source": self.value_source,
            "default": self.default,
            "rng_family": self.rng_family,
            "rng_version": self.rng_version,
            "kernel": self.kernel,
            "seed_material": list(self.seed_material),
            "consumption_order": list(self.consumption_order),
            "reset_boundary": self.reset_boundary,
            "draw_condition": self.draw_condition,
            "derivation": self.derivation,
        }


@dataclass(frozen=True, slots=True)
class SeedProtocol:
    """A content-attested protocol selected by one authored scalar."""

    id: str
    implementation_id: str
    kernels: tuple[KernelAttestation, ...]
    sites: tuple[DrawSiteProtocol, ...]

    def __post_init__(self) -> None:
        site_ids = tuple(site.id for site in self.sites)
        kernel_ids = tuple(kernel.id for kernel in self.kernels)
        if len(site_ids) != len(set(site_ids)):
            raise ValueError("seed-protocol draw-site ids are not unique")
        if len(kernel_ids) != len(set(kernel_ids)):
            raise ValueError("seed-protocol kernel ids are not unique")
        unknown = sorted({site.kernel for site in self.sites} - set(kernel_ids))
        if unknown:
            raise ValueError(
                f"seed-protocol sites reference unknown kernels: {unknown}"
            )

    @property
    def streams(self) -> frozenset[str]:
        return frozenset(site.stream for site in self.sites)

    @property
    def implementation_sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "id": self.id,
                    "implementation_id": self.implementation_id,
                    "boundary": "mirror-only-until-f1",
                    "kernels": [kernel.to_wire() for kernel in self.kernels],
                    "sites": [site.to_wire() for site in self.sites],
                }
            )
        ).hexdigest()

    def to_wire(self) -> dict[str, object]:
        wire = {
            "id": self.id,
            "implementation_id": self.implementation_id,
            "implementation_sha256": self.implementation_sha256,
            "boundary": "mirror-only-until-f1",
            "kernels": [kernel.to_wire() for kernel in self.kernels],
            "streams": sorted(self.streams),
            "sites": [site.to_wire() for site in self.sites],
        }
        validate_seed_protocol_wire(wire)
        return wire

    def site(self, site_id: str) -> DrawSiteProtocol:
        matches = [site for site in self.sites if site.id == site_id]
        if len(matches) != 1:
            raise KeyError(site_id)
        return matches[0]


def validate_seed_protocol_wire(value: Mapping[str, object]) -> None:
    """Validate cross-row invariants and the content digest of a lock payload."""

    required = {
        "id",
        "implementation_id",
        "implementation_sha256",
        "boundary",
        "kernels",
        "streams",
        "sites",
    }
    if set(value) != required:
        raise ValueError("seed protocol wire fields do not match the closed contract")
    kernels = value["kernels"]
    sites = value["sites"]
    streams = value["streams"]
    if (
        not isinstance(kernels, list)
        or not isinstance(sites, list)
        or not isinstance(streams, list)
    ):
        raise ValueError("seed protocol kernels, streams, and sites must be arrays")
    kernel_ids = [row.get("id") for row in kernels if isinstance(row, Mapping)]
    site_ids = [row.get("id") for row in sites if isinstance(row, Mapping)]
    if len(kernel_ids) != len(kernels) or len(kernel_ids) != len(set(kernel_ids)):
        raise ValueError("seed protocol kernel ids are not unique")
    if len(site_ids) != len(sites) or len(site_ids) != len(set(site_ids)):
        raise ValueError("seed protocol draw-site ids are not unique")
    site_streams: set[str] = set()
    for row in sites:
        assert isinstance(row, Mapping)  # guarded by site_ids construction above
        default = row.get("default")
        if default is not None and (
            not isinstance(default, int)
            or isinstance(default, bool)
            or not 0 <= default <= MAX_SEED
        ):
            raise ValueError(f"draw site {row.get('id')!r} seed is outside uint64")
        kernel = row.get("kernel")
        if kernel not in kernel_ids:
            raise ValueError(f"draw site {row.get('id')!r} references unknown kernel")
        stream = row.get("stream")
        if not isinstance(stream, str) or not stream.startswith("stream:"):
            raise ValueError(f"draw site {row.get('id')!r} has an invalid stream")
        site_streams.add(stream.removeprefix("stream:"))
    if streams != sorted(site_streams):
        raise ValueError("seed protocol stream index does not match draw sites")
    body = {
        "id": value["id"],
        "implementation_id": value["implementation_id"],
        "boundary": value["boundary"],
        "kernels": kernels,
        "sites": sites,
    }
    expected = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if value["implementation_sha256"] != expected:
        raise ValueError("seed protocol implementation_sha256 does not match its body")


def _site(
    id: str,
    stream: str,
    *,
    value_source: str,
    default: int | None,
    rng_family: str = "numpy.random.Generator(PCG64)",
    rng_version: str | None = None,
    kernel: str = "legacy_v1_direct_draws",
    seed_material: tuple[str, ...] = ("seed",),
    consumption_order: tuple[str, ...] = ("declared_row_order",),
    reset_boundary: str = "fresh_generator_per_call",
    draw_condition: str = "always",
    derivation: str = "direct_integer_seed",
) -> DrawSiteProtocol:
    return DrawSiteProtocol(
        id=id,
        stream=stream,
        value_source=value_source,
        default=default,
        rng_family=rng_family,
        rng_version=rng_version or f"numpy=={_distribution_version('numpy')}",
        kernel=kernel,
        seed_material=seed_material,
        consumption_order=consumption_order,
        reset_boundary=reset_boundary,
        draw_condition=draw_condition,
        derivation=derivation,
    )


_BLAKE2B_DRAW = (
    "blake2b(utf8('{seed}:{salt}:{stable_key}'),digest_size=8); "
    "big_endian_unsigned / 2**64"
)
_X31_MIX64 = (
    "utf8 polynomial_x31 modulo 2**64; xor_shift_33; "
    "multiply_0xff51afd7ed558ccd modulo 2**64; xor_shift_33; modulo 2**63"
)
_DIRECT_KERNEL_MODULES = (
    "microcosm.build.frame_sampling",
    "microcosm.build.source_runtime",
    "microcosm.build.us_runtime.acs_transfer",
    "microcosm.build.us_runtime.adult_care",
    "microcosm.build.us_runtime.child_support",
    "microcosm.build.us_runtime.childcare",
    "microcosm.build.us_runtime.congressional_district_geography",
    "microcosm.build.us_runtime.disability_benefits",
    "microcosm.build.us_runtime.energy_subsidy",
    "microcosm.build.us_runtime.geography_ladder",
    "microcosm.build.us_runtime.housing_inputs",
    "microcosm.build.us_runtime.immigration",
    "microcosm.build.us_runtime.medicaid_take_up",
    "microcosm.build.us_runtime.other_health_insurance",
    "microcosm.build.us_runtime.pregnancy",
    "microcosm.build.us_runtime.prior_year_income",
    "microcosm.build.us_runtime.puf_aggregate_records",
    "microcosm.build.us_runtime.puf_capital_gains_tail",
    "microcosm.build.us_runtime.puf_qrf_chain",
    "microcosm.build.us_runtime.puf_source_agi",
    "microcosm.build.us_runtime.puf_support",
    "microcosm.build.us_runtime.puma_ladder",
    "microcosm.build.us_runtime.retirement_contributions",
    "microcosm.build.us_runtime.retirement_distributions",
    "microcosm.build.us_runtime.scf_auto_loans",
    "microcosm.build.us_runtime.scf_wealth",
    "microcosm.build.us_runtime.sipp_financial_assets",
    "microcosm.build.us_runtime.sipp_tips",
    "microcosm.build.us_runtime.sipp_vehicles",
    "microcosm.build.us_runtime.snap_discretionary_exemption",
    "microcosm.build.us_runtime.snap_state_take_up",
    "microcosm.build.us_runtime.snap_take_up",
    "microcosm.build.us_runtime.source_runtime",
    "microcosm.build.us_runtime.ssi_disability_criteria",
    "microcosm.build.us_runtime.ssi_take_up",
    "microcosm.build.us_runtime.take_up",
    "microcosm.build.us_runtime.weeks_unemployed",
    "microcosm.build.us_runtime.wic_claim",
    "microcosm.build.us_runtime.workers_compensation",
    "microcosm.calibrate.exact_k",
    "microcosm.calibrate.solve",
)
_QRF_KERNEL_MODULES = (
    "microcosm.build.us_runtime.acs_transfer",
    "microcosm.build.us_runtime.puf_qrf_chain",
    "microcosm.build.us_runtime.scf_auto_loans",
    "microcosm.build.us_runtime.scf_wealth",
    "microcosm.build.us_runtime.sipp_financial_assets",
    "microcosm.build.us_runtime.sipp_vehicles",
    "microcosm.build.us_runtime.ssi_disability_criteria",
    "microcosm.fit.qrf",
)

LEGACY_V1_KERNELS = (
    KernelAttestation(
        id="legacy_v1_direct_draws",
        source_modules=_DIRECT_KERNEL_MODULES,
        source_sha256=source_inventory_sha256(_DIRECT_KERNEL_MODULES),
    ),
    KernelAttestation(
        id="regime_gated_qrf",
        source_modules=_QRF_KERNEL_MODULES,
        source_sha256=source_inventory_sha256(_QRF_KERNEL_MODULES),
    ),
)

_PYTHON_HASHLIB_VERSION = "BLAKE2b-RFC7693;SHA-256-FIPS180-4"
_QRF_RNG_VERSION = ";".join(
    (
        f"microcosm-fit=={_distribution_version('microcosm-fit')}",
        f"numpy=={_distribution_version('numpy')}",
        f"scikit-learn=={_distribution_version('scikit-learn')}",
        f"quantile-forest=={_distribution_version('quantile-forest')}",
    )
)
_BUILD_CAP_SITES = (
    "prior_year_income_training_cap",
    "childcare_training_cap",
    "retirement_contributions_training_cap",
    "disability_benefits_training_cap",
    "housing_inputs_training_cap",
    "workers_compensation_training_cap",
    "retirement_distributions_training_cap",
    "child_support_training_cap",
    "energy_subsidy_training_cap",
    "other_health_insurance_training_cap",
    "weeks_unemployed_training_cap",
)


def _stable_site(
    site_id: str,
    *,
    salt: str,
    key_grammar: tuple[str, ...],
    candidate_universe: str,
    draw_condition: str = "always",
) -> DrawSiteProtocol:
    """Build one explicit current BLAKE2b site; no generic salt is permitted."""

    return _site(
        site_id,
        "stable_entity_draw",
        value_source="run_request.build_model_seed",
        default=0,
        rng_family="hashlib.blake2b stateless uniform",
        rng_version=_PYTHON_HASHLIB_VERSION,
        seed_material=(
            "build_model_seed",
            f"literal_salt={salt}",
            *key_grammar,
        ),
        consumption_order=(candidate_universe, "one_stateless_hash_per_row"),
        reset_boundary="stateless_per_entity",
        draw_condition=draw_condition,
        derivation=_BLAKE2B_DRAW,
    )


LEGACY_V1_SITES = (
    _site(
        "survey_sample_asec",
        "sampling_asec",
        value_source="run_request.sample_seed",
        default=578,
        seed_material=("sample_seed",),
        consumption_order=("sorted_strata", "sorted_entity_ids"),
        draw_condition="fraction_below_one_only",
    ),
    _site(
        "survey_sample_acs",
        "sampling_acs",
        value_source="run_request.sample_seed",
        default=578,
        seed_material=("sample_seed",),
        consumption_order=("sorted_strata", "sorted_entity_ids"),
        draw_condition="fraction_below_one_only",
    ),
    _site(
        "puf_clone_attachment",
        "puf_clone_attachment",
        value_source="run_request.clone_attachment_seed",
        default=578,
        seed_material=("clone_attachment_seed",),
        consumption_order=("sorted_source_household_ids",),
        draw_condition="clone_attachment_fraction_below_one_only",
    ),
    _site(
        "puf_archived_aggregate_disaggregation",
        "puf_archived_disaggregation",
        value_source="literal",
        default=42,
        consumption_order=(
            "one_shared_generator",
            "three_bounded_buckets_in_declared_order",
        ),
        reset_boundary="fresh_generator_per_archived_processed_puf",
    ),
    _site(
        "puf_live_aggregate_disaggregation",
        "puf_live_disaggregation",
        value_source="run_request.build_model_seed",
        default=0,
        seed_material=("build_model_seed",),
        consumption_order=(
            "one_shared_generator",
            "aggregate_recids_in_runtime_declared_order",
            "per_bucket_assign_s006_choice_then_donor_choice",
        ),
        reset_boundary="fresh_generator_per_live_raw_puf_disaggregation",
        derivation="runtime_seed_overrides_library_default_42",
    ),
    _site(
        "ssi_weighted_replacement_training",
        "ssi_weighted_replacement",
        value_source="literal",
        default=8_386_123_572_872_638_692,
        consumption_order=("donor_row_order", "weighted_choice_with_replacement"),
        draw_condition="min(20000,donor_rows)_draws_replace_true",
    ),
    _site(
        "ssi_archived_qrf_model",
        "ssi_model",
        value_source="literal",
        default=42,
        rng_family="microcosm.fit.QRF SeedSequence(PCG64)",
        rng_version=_QRF_RNG_VERSION,
        kernel="regime_gated_qrf",
        seed_material=("archived_model_seed",),
        consumption_order=("asec_support", "puf_tax_detail_support"),
        reset_boundary="deepcopy_pristine_fitted_model_per_support_channel",
        derivation="build_seed_argument_ignored_for_archived_model_seed",
    ),
    _site(
        "sipp_vehicle_training_cap",
        "sipp_training_cap",
        value_source="stable_string",
        default=None,
        seed_material=(
            "calibration_sipp_vehicle_training_sample",
            "target_or_fill_salt",
        ),
        consumption_order=("vehicles_owned", "vehicles_value", "fill"),
        derivation=_X31_MIX64,
    ),
    _site(
        "sipp_vehicle_qrf_model",
        "qrf_fit_draw",
        value_source="literal",
        default=42,
        rng_family="microcosm.fit.QRF SeedSequence(PCG64)",
        rng_version=_QRF_RNG_VERSION,
        kernel="regime_gated_qrf",
        consumption_order=("vehicles_owned", "vehicles_value"),
        reset_boundary="one_model_chain_per_vehicle_stage",
        derivation="QRF SeedSequence(seed).spawn(2) fit_child_0 draw_child_1",
    ),
    _site(
        "sipp_financial_asset_training_cap",
        "sipp_training_cap",
        value_source="stable_string",
        default=None,
        seed_material=("named_training_sample", "target_or_fill_salt"),
        consumption_order=("bank_account_assets", "stock_assets", "bond_assets"),
        derivation=_X31_MIX64,
    ),
    _site(
        "sipp_financial_asset_qrf_models",
        "qrf_fit_draw",
        value_source="run_request.build_model_seed",
        default=0,
        rng_family="microcosm.fit.QRF SeedSequence(PCG64)",
        rng_version=_QRF_RNG_VERSION,
        kernel="regime_gated_qrf",
        seed_material=("build_model_seed", "literal_374"),
        consumption_order=("bank_account_assets", "stock_assets", "bond_assets"),
        reset_boundary="three_spawned_child_sequences_one_per_declared_target",
        derivation="SeedSequence([base_seed,374]).spawn(3)",
    ),
    _site(
        "acs_rent_archived_training_cap",
        "sipp_training_cap",
        value_source="stable_string",
        default=None,
        seed_material=("named_training_sample", "target_or_fill_salt"),
        consumption_order=("rent", "real_estate_tax"),
        derivation=_X31_MIX64,
    ),
    _site(
        "sipp_tip_training_cap",
        "sipp_training_cap",
        value_source="literal",
        default=5_559_651_045_748_063_828,
        seed_material=("calibration_sipp_tip_training_sample:tip_income",),
        consumption_order=("choice_without_replacement", "sorted_positions"),
        draw_condition="donor_rows_above_10000",
    ),
    _site(
        "scf_household_source_selector",
        "build_model",
        value_source="run_request.build_model_seed",
        default=0,
        seed_material=("build_model_seed", "time_period", "literal_374"),
        consumption_order=("sorted_household_ids", "one_uniform_per_household"),
        draw_condition="select_scf_when_uniform_below_0.5",
        derivation="default_rng(SeedSequence([seed,time_period,374]))",
    ),
    _site(
        "scf_financial_asset_qrf_model",
        "qrf_fit_draw",
        value_source="run_request.build_model_seed",
        default=0,
        rng_family="microcosm.fit.QRF SeedSequence(PCG64)",
        rng_version=_QRF_RNG_VERSION,
        kernel="regime_gated_qrf",
        seed_material=("build_model_seed",),
        consumption_order=("bank_account_assets", "stock_assets", "bond_assets"),
        reset_boundary="fresh_qrf_chain_for_scf_financial_asset_vector",
        derivation="SeedSequence(seed).spawn(2): fit_child_0 draw_child_1",
    ),
    _site(
        "scf_net_worth_qrf_model",
        "qrf_fit_draw",
        value_source="run_request.build_model_seed",
        default=0,
        rng_family="microcosm.fit.QRF SeedSequence(PCG64)",
        rng_version=_QRF_RNG_VERSION,
        kernel="regime_gated_qrf",
        seed_material=("build_model_seed",),
        consumption_order=("net_worth",),
        reset_boundary="fresh_qrf_chain_separate_from_scf_financial_assets",
        derivation="SeedSequence(seed).spawn(2): fit_child_0 draw_child_1",
    ),
    _site(
        "scf_auto_loan_qrf_model",
        "qrf_fit_draw",
        value_source="run_request.build_model_seed",
        default=0,
        rng_family="microcosm.fit.QRF SeedSequence(PCG64)",
        rng_version=_QRF_RNG_VERSION,
        kernel="regime_gated_qrf",
        seed_material=("build_model_seed",),
        consumption_order=("auto_loan_balance", "auto_loan_interest"),
        reset_boundary="fresh_qrf_chain_per_scf_auto_loan_stage",
        derivation="SeedSequence(seed).spawn(2): fit_child_0 draw_child_1",
    ),
    _site(
        "acs_transfer_family_seed",
        "qrf_fit_draw",
        value_source="run_request.build_model_seed",
        default=0,
        rng_family="SHA-256 derived integer",
        rng_version=_PYTHON_HASHLIB_VERSION,
        seed_material=("build_model_seed", "entity", "family"),
        consumption_order=("declared_family_order",),
        reset_boundary="derived_once_per_entity_family",
        derivation=(
            "little_endian_unsigned(first_4_bytes(sha256(utf8("
            "base+'\\0'+entity+'\\0'+family))))"
        ),
    ),
    _site(
        "acs_transfer_pattern_seed",
        "qrf_fit_draw",
        value_source="run_request.build_model_seed",
        default=0,
        rng_family="SHA-256 derived integer",
        rng_version=_PYTHON_HASHLIB_VERSION,
        seed_material=(
            "build_model_seed",
            "entity",
            "family",
            "nul_joined_ordered_optional_predictors",
        ),
        consumption_order=("declared_availability_pattern_order",),
        reset_boundary="derived_once_per_availability_pattern",
        derivation=(
            "little_endian_unsigned(first_4_bytes(sha256(utf8(base+'\\0'+entity+"
            "'\\0'+family+'\\0'+nul_joined_ordered_optional_predictors)))); "
            "pattern_id='pattern_'+two_digit_position+'_'+first_8_lower_hex_of_"
            "sha256(utf8(nul_joined_ordered_optional_predictors))"
        ),
    ),
    _site(
        "primary_qrf_fit_draw",
        "qrf_fit_draw",
        value_source="run_request.build_model_seed",
        default=0,
        rng_family="microcosm.fit.QRF SeedSequence(PCG64)",
        rng_version=_QRF_RNG_VERSION,
        kernel="regime_gated_qrf",
        seed_material=("build_model_seed",),
        consumption_order=(
            "declared_target_order",
            "append_each_drawn_target_to_later_predictors",
            "ordered_quantile_then_gated_sign_uniform",
        ),
        reset_boundary="one_shared_fit_rng_and_one_shared_draw_rng_per_chain",
        derivation="SeedSequence(seed).spawn(2): fit_child_0 draw_child_1",
    ),
    _site(
        "acs_qrf_fit_draw",
        "qrf_fit_draw",
        value_source="derived_acs_pattern_seed",
        default=0,
        rng_family="microcosm.fit.QRF SeedSequence(PCG64)",
        rng_version=_QRF_RNG_VERSION,
        kernel="regime_gated_qrf",
        seed_material=("acs_transfer_pattern_seed",),
        consumption_order=(
            "declared_target_order",
            "append_each_drawn_target_to_later_predictors",
            "ordered_quantile_then_gated_sign_uniform",
        ),
        reset_boundary="one_shared_fit_rng_and_one_shared_draw_rng_per_fit_group",
        derivation="SeedSequence(seed).spawn(2): fit_child_0 draw_child_1",
    ),
    _stable_site(
        "source_aca_assignment",
        salt="aca:{output}",
        key_grammar=("tax_unit_id_else_positional_row_index",),
        candidate_universe="all_frame_rows_before_eligibility_mask",
        draw_condition="only_when_declared_draw_column_is_absent",
    ),
    _stable_site(
        "source_count_calibration",
        salt="calibrate:{variable}",
        key_grammar=("tax_unit_id_else_positional_row_index",),
        candidate_universe="all_frame_rows_before_domain_and_anchor_masks",
        draw_condition="only_when_declared_draw_column_is_absent",
    ),
    _stable_site(
        "source_joint_count_calibration",
        salt="joint-calibrate:{variable}",
        key_grammar=("tax_unit_id_else_positional_row_index",),
        candidate_universe="all_frame_rows_before_domain_and_anchor_masks",
        draw_condition="only_when_declared_draw_column_is_absent",
    ),
    _stable_site(
        "snap_take_up_assignment",
        salt="snap_take_up",
        key_grammar=(
            "source_year:source_household_id:source_person_id_if_complete",
            "else_spm_unit_membership_id",
        ),
        candidate_universe="all_derived_spm_units_before_rate_comparison",
    ),
    _stable_site(
        "pregnancy_assignment",
        salt="pregnancy",
        key_grammar=(
            "source_year:source_household_id:source_person_id_if_complete",
            "else_person_id",
        ),
        candidate_universe="all_person_rows_before_sex_and_age_eligibility_mask",
    ),
    _stable_site(
        "wic_claim_assignment",
        salt="would_claim_wic",
        key_grammar=(
            "source_year:source_household_id:source_person_id_if_complete",
            "else_support:person_support_source_id",
            "else_person:person_id",
        ),
        candidate_universe="all_person_rows_before_category_rate_comparison",
    ),
    _stable_site(
        "snap_discretionary_exemption_assignment",
        salt="snap_abawd_discretionary_exemption",
        key_grammar=(
            "source_year:source_household_id:source_person_id_if_complete",
            "else_person_id",
        ),
        candidate_universe="all_person_rows_before_abawd_eligibility_mask",
    ),
    *(
        _stable_site(
            f"immigration_humanitarian_{label.replace(':', '_')}_assignment",
            salt=f"immigration:{label}",
            key_grammar=("source_year:source_person_id_if_present", "else_person_id"),
            candidate_universe=(
                "all_person_rows_then_category_origin_window_pool_mask"
            ),
        )
        for label in (
            "paroled_one_year:afghanistan",
            "paroled_one_year:ukraine",
            "paroled_one_year:nicaragua",
            "paroled_one_year:venezuela",
            "refugee",
            "asylee",
            "deportation_withheld",
            "tps:venezuela",
            "tps:el_salvador",
            "tps:honduras",
            "tps:nicaragua",
            "tps:nepal",
            "tps:other_designated",
        )
    ),
    _stable_site(
        "immigration_ead_workers_assignment",
        salt="immigration:ead_workers",
        key_grammar=("source_year:source_person_id_if_present", "else_person_id"),
        candidate_universe="all_person_rows_then_worker_candidate_mask",
    ),
    _stable_site(
        "immigration_ead_students_assignment",
        salt="immigration:ead_students",
        key_grammar=("source_year:source_person_id_if_present", "else_person_id"),
        candidate_universe="all_person_rows_then_student_candidate_mask",
    ),
    _stable_site(
        "ssi_take_up_assignment",
        salt="takes_up_ssi_if_eligible",
        key_grammar=("person_support_source_id",),
        candidate_universe="source_person_ids_sorted_by_groupby_before_age_band_threshold",
    ),
    _stable_site(
        "medicaid_take_up_assignment",
        salt="takes_up_medicaid_if_eligible",
        key_grammar=(
            "complete_source_identity_overrides_support_source_id",
            "support_source_id_overrides_person_id",
        ),
        candidate_universe="all_person_units_before_eligibility_and_state_calibration",
    ),
    _stable_site(
        "snap_state_take_up_assignment",
        salt="takes_up_snap_if_eligible",
        key_grammar=(
            "complete_source_identity_overrides_support_source_id",
            "support_source_id_overrides_spm_unit_id",
        ),
        candidate_universe="all_spm_units_before_eligibility_and_state_calibration",
    ),
    _stable_site(
        "tanf_take_up_assignment",
        salt="takes_up_tanf",
        key_grammar=(
            "complete_source_identity_overrides_support_source_id",
            "support_source_id_overrides_entity_id",
        ),
        candidate_universe="all_declared_tanf_program_units_before_rate_comparison",
        draw_condition="only_for_missing_or_legacy_constant_output_cells",
    ),
    _stable_site(
        "eitc_take_up_assignment",
        salt="takes_up_eitc",
        key_grammar=(
            "complete_source_identity_overrides_support_source_id",
            "support_source_id_overrides_entity_id",
        ),
        candidate_universe="all_tax_units_before_child_count_rate_comparison",
        draw_condition="only_for_missing_or_legacy_constant_output_cells",
    ),
    _site(
        "adult_care_weighted_prefix_assignment",
        "build_model",
        value_source="run_request.build_model_seed",
        default=0,
        seed_material=("build_model_seed",),
        consumption_order=(
            "eligible_unit_ids_stable_sorted",
            "one_permutation",
            "permuted_weight_prefix_through_usage_target",
        ),
        reset_boundary="fresh_generator_per_adult_care_stage",
    ),
    _site(
        "capital_gains_tail_random_rank",
        "build_model",
        value_source="run_request.build_model_seed",
        default=0,
        seed_material=("build_model_seed",),
        consumption_order=(
            "mergesort_recipient_household_source_id_then_tax_unit_source_id",
            "one_uniform_per_sorted_recipient_tax_unit",
        ),
        reset_boundary="fresh_generator_per_tail_stage",
    ),
    _site(
        "torch_calibration_reseed",
        "calibration",
        value_source="run_request.build_model_seed",
        default=0,
        rng_family="torch.manual_seed",
        rng_version=f"torch=={_distribution_version('torch')}",
        seed_material=("build_model_seed",),
        consumption_order=("optimizer_initialization_then_declared_iterations",),
        reset_boundary="manual_seed_at_solver_entry",
        derivation="torch_manual_seed(integer_seed)",
    ),
    _site(
        "exact_k_pcg64_selection",
        "exact_k_selection",
        value_source="run_request.selection_seed",
        default=None,
        seed_material=("selection_seed",),
        consumption_order=("exact_k_candidate_order", "sampford_draw_order"),
        reset_boundary="fresh_pcg64_per_exact_k_selection",
        draw_condition="only_when_exact_k_selection_is_requested",
        derivation="PCG64 plus Sampford fixed-size selection",
    ),
    *tuple(
        _site(
            site_id,
            "build_model",
            value_source="run_request.build_model_seed",
            default=0,
            rng_family="pandas.DataFrame.sample RandomState(MT19937)",
            rng_version=(
                f"pandas=={_distribution_version('pandas')};"
                f"numpy=={_distribution_version('numpy')}"
            ),
            seed_material=("build_model_seed", "stage_training_cap"),
            consumption_order=(
                "source_dataframe_row_order",
                "sorted_selected_positions",
            ),
            reset_boundary="fresh_generator_per_source_stage",
            draw_condition="donor_rows_above_5000",
            derivation="pandas_sample(n=5000,random_state=build_model_seed)",
        )
        for site_id in _BUILD_CAP_SITES
    ),
    _site(
        "legacy_geography_ladder",
        "geography_legacy",
        value_source="run_request.build_model_seed",
        default=0,
        seed_material=("build_model_seed",),
        consumption_order=("stable_household_or_person_order",),
        reset_boundary="fresh_generator_per_geography_stage",
    ),
    _site(
        "legacy_puma_ladder",
        "geography_legacy",
        value_source="run_request.build_model_seed",
        default=0,
        seed_material=("build_model_seed",),
        consumption_order=("stable_support_row_order", "declared_ladder_order"),
        reset_boundary="fresh_generator_per_puma_assignment",
    ),
    _site(
        "legacy_congressional_district_assignment",
        "geography_legacy",
        value_source="run_request.build_model_seed",
        default=0,
        seed_material=("build_model_seed",),
        consumption_order=("stable_support_row_order", "district_weight_order"),
        reset_boundary="fresh_generator_per_district_assignment",
    ),
)


def _validated_legacy_v1() -> SeedProtocol:
    protocol = SeedProtocol(
        id="legacy-v1",
        implementation_id="microcosm.seed-protocol.legacy-v1",
        kernels=LEGACY_V1_KERNELS,
        sites=LEGACY_V1_SITES,
    )
    return protocol


LEGACY_V1_PROTOCOL = _validated_legacy_v1()


__all__ = [
    "DrawSiteProtocol",
    "KernelAttestation",
    "LEGACY_V1_KERNELS",
    "LEGACY_V1_PROTOCOL",
    "LEGACY_V1_SITES",
    "SeedProtocol",
    "source_inventory_sha256",
]
