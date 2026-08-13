"""Inventory of production serializers that can encounter Frame dtypes.

This registry is deliberately code, rather than review prose.  Its test scans
every production Python source for writable HDF handles and fails when a new
physical sink has not been classified.  A serializer entry denotes a logical
Frame/table-collection sink; an exclusion denotes an HDF mutation that cannot
receive a Frame column (raw arrays or root attributes only).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HdfWriteSite:
    """One function that opens an HDF file for mutation."""

    path: str
    function: str

    @property
    def key(self) -> str:
        return f"{self.path}::{self.function}"


@dataclass(frozen=True)
class FrameSerializerSpec:
    """A physical serializer whose input includes one or more Frame tables."""

    serializer_id: str
    writer: HdfWriteSite
    backend: str
    routes: tuple[str, ...]
    version_owner: str
    nullable_boolean_storage: str
    direct_hdf_open: bool = True


@dataclass(frozen=True)
class HdfWriteExclusion:
    """A writable HDF site that provably does not serialize Frame columns."""

    exclusion_id: str
    writer: HdfWriteSite
    reason: str


FRAME_TABLE_SERIALIZERS = (
    FrameSerializerSpec(
        serializer_id="frame_checkpoint",
        writer=HdfWriteSite(
            "packages/microcosm-build/src/microcosm/build/frame_checkpoint.py",
            "write_frame_checkpoint",
        ),
        backend="h5py",
        routes=("generic Frame checkpoints", "US pool stage checkpoints"),
        version_owner="FRAME_CHECKPOINT_SCHEMA_VERSION",
        nullable_boolean_storage="bool_values_optional_uint8_mask",
    ),
    FrameSerializerSpec(
        serializer_id="nullable_us_h5",
        writer=HdfWriteSite(
            "packages/microcosm-build/src/microcosm/build/us_runtime/h5_io.py",
            "_write_nullable_us_h5_file",
        ),
        backend="pandas.HDFStore fixed",
        routes=(
            "stacked terminal pool publication",
            "current legacy two-spine publication facade",
            "US ACS calibrated release",
            "US L0 refit export",
        ),
        version_owner="US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION",
        nullable_boolean_storage="numpy_bool_or_object_pd_na_v1",
    ),
    FrameSerializerSpec(
        serializer_id="uk_single_year_h5",
        writer=HdfWriteSite(
            "packages/microcosm-build/src/microcosm/build/uk_runtime/national_build.py",
            "_write_uk_single_year_tables",
        ),
        backend="pandas.HDFStore table",
        routes=(
            "UK national publication",
            "UK rowwise publication",
            "UK ladder-rowwise publication",
        ),
        version_owner="UK single-year payload contract",
        nullable_boolean_storage="numpy_bool_or_object_pd_na_v1",
    ),
    FrameSerializerSpec(
        serializer_id="axiom_entity_tables",
        writer=HdfWriteSite(
            "packages/microcosm-frame/src/microcosm/frame/adapters/axiom.py",
            "save",
        ),
        backend="pandas.HDFStore table",
        routes=("Axiom adapter entity-table dataset",),
        version_owner="AxiomEntityTableDataset payload contract",
        nullable_boolean_storage="numpy_bool_or_object_pd_na_v1",
    ),
    FrameSerializerSpec(
        serializer_id="policyengine_us_dataset",
        writer=HdfWriteSite(
            "packages/microcosm-frame/src/microcosm/frame/adapters/policyengine_us.py",
            "_write_and_verify",
        ),
        backend="pandas.HDFStore table",
        routes=("PolicyEngine-US adapter export",),
        version_owner="PolicyEngineUSAdapter payload contract",
        nullable_boolean_storage="numpy_bool_or_object_pd_na_v1",
    ),
    FrameSerializerSpec(
        serializer_id="legacy_us_two_spine",
        writer=HdfWriteSite(
            "tools/_legacy/build_us_acs_multispine_base.py",
            "_write_dataset",
        ),
        backend="pandas.HDFStore fixed",
        routes=("preserved directly executable legacy two-spine builder",),
        version_owner="legacy schema-4 publication contract",
        nullable_boolean_storage="numpy_bool_or_object_pd_na_v1",
    ),
    FrameSerializerSpec(
        serializer_id="acs_local_lean_checkpoint",
        writer=HdfWriteSite(
            "tools/build_us_acs_local_release.py",
            "write_lean_checkpoint",
        ),
        backend="pandas.HDFStore fixed",
        routes=("US ACS local lean target-frame checkpoint",),
        version_owner="ACS local checkpoint payload contract",
        nullable_boolean_storage="numpy_bool_or_object_pd_na_v1",
    ),
    FrameSerializerSpec(
        serializer_id="fiscal_target_frame_checkpoint",
        writer=HdfWriteSite(
            "tools/build_us_fiscal_refresh_release.py",
            "_write_target_frame_checkpoint",
        ),
        backend="h5py",
        routes=("US fiscal-refresh target-frame checkpoint",),
        version_owner="TARGET_FRAME_CHECKPOINT_SCHEMA_VERSION",
        nullable_boolean_storage="bool_values_optional_uint8_mask",
    ),
)


HDF_WRITE_EXCLUSIONS = (
    HdfWriteExclusion(
        exclusion_id="l0_refit_root_attrs",
        writer=HdfWriteSite(
            "packages/microcosm-build/src/microcosm/build/us_runtime/"
            "l0_refit_export.py",
            "copy_microcosm_root_attrs",
        ),
        reason="Copies Microcosm-owned root attributes only.",
    ),
    HdfWriteExclusion(
        exclusion_id="acs_transfer_raw_draw_bank",
        writer=HdfWriteSite(
            "packages/microcosm-build/src/microcosm/build/us_runtime/"
            "acs_transfer_bank.py",
            "write_target",
        ),
        reason="Writes canonical JSON bytes and raw numeric draw bits only.",
    ),
    HdfWriteExclusion(
        exclusion_id="puf_qrf_raw_draw_checkpoint",
        writer=HdfWriteSite(
            "packages/microcosm-build/src/microcosm/build/us_runtime/puf_qrf_chain.py",
            "_write_target_checkpoint",
        ),
        reason="Writes canonical JSON bytes and raw numeric draw bits only.",
    ),
    HdfWriteExclusion(
        exclusion_id="uk_weight_root_attrs",
        writer=HdfWriteSite(
            "packages/microcosm-build/src/microcosm/build/uk_runtime/national_build.py",
            "_write_weight_metadata",
        ),
        reason="Adds weight-kind and mass-log root attributes only.",
    ),
    HdfWriteExclusion(
        exclusion_id="puf_equivalence_raw_draw_observer",
        writer=HdfWriteSite(
            "tools/build_us_puf_support_base.py",
            "observe_primary_qrf",
        ),
        reason="Optional equivalence observer writes float draw bits and attrs only.",
    ),
    HdfWriteExclusion(
        exclusion_id="puf_monolith_geography_attrs",
        writer=HdfWriteSite(
            "tools/build_us_puf_support_base.py",
            "_run_all",
        ),
        reason="Adds geography provenance root attributes only.",
    ),
    HdfWriteExclusion(
        exclusion_id="puf_staged_geography_attrs",
        writer=HdfWriteSite(
            "tools/build_us_puf_support_base.py",
            "_export_staged_result",
        ),
        reason="Adds geography provenance root attributes only.",
    ),
)


__all__ = [
    "FRAME_TABLE_SERIALIZERS",
    "HDF_WRITE_EXCLUSIONS",
    "FrameSerializerSpec",
    "HdfWriteExclusion",
    "HdfWriteSite",
]
