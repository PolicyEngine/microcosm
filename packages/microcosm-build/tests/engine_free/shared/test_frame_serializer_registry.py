"""Tests split from packages/microcosm-build/tests/test_frame_serializer_registry.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.frame_serializer_registry import *


def test_registry_classifies_every_writable_production_hdf_site() -> None:
    classified = {
        spec.writer.key for spec in FRAME_TABLE_SERIALIZERS if spec.direct_hdf_open
    }
    classified.update(exclusion.writer.key for exclusion in HDF_WRITE_EXCLUSIONS)
    assert _discover_writable_hdf_sites() == classified


def test_registry_has_exactly_ten_unique_frame_table_serializers() -> None:
    assert len(FRAME_TABLE_SERIALIZERS) == 10
    assert len({spec.serializer_id for spec in FRAME_TABLE_SERIALIZERS}) == 10
    assert len({spec.writer.key for spec in FRAME_TABLE_SERIALIZERS}) == 10


def test_round_trip_adapter_registry_exactly_matches_serializer_registry() -> None:
    assert set(ROUND_TRIP_ADAPTERS) == {
        spec.serializer_id for spec in FRAME_TABLE_SERIALIZERS
    }


@pytest.mark.parametrize(
    "serializer",
    ENGINE_FREE_SERIALIZERS,
    ids=lambda serializer: serializer.serializer_id,
)
@pytest.mark.parametrize("nullable_case", ("mixed", "all_missing"))
def test_registered_serializer_round_trips_nullable_boolean_dtype_family(
    serializer: FrameSerializerSpec,
    nullable_case: str,
    tmp_path: Path,
) -> None:
    assert_registered_serializer_round_trip(serializer, nullable_case, tmp_path)


def test_policyengine_us_adapter_owns_its_registered_hdf_boundary() -> None:
    (spec,) = (
        candidate
        for candidate in FRAME_TABLE_SERIALIZERS
        if candidate.serializer_id == "policyengine_us_single_year"
    )
    assert spec.direct_hdf_open is True
    assert spec.writer.key in _discover_writable_hdf_sites()


def test_no_production_dataframe_to_hdf_sink_bypasses_registry() -> None:
    sites: list[str] = []
    for root in PRODUCTION_ROOTS:
        for path in root.rglob("*.py"):
            if "tests" in path.parts:
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "to_hdf"
                ):
                    sites.append(
                        f"{path.relative_to(REPOSITORY_ROOT).as_posix()}:{node.lineno}"
                    )
    assert sites == []
