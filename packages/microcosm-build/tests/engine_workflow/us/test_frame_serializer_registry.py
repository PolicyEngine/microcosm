"""Tests split from packages/microcosm-build/tests/test_frame_serializer_registry.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.frame_serializer_registry import *


@pytest.mark.parametrize(
    "serializer",
    ENGINE_REQUIRED_SERIALIZERS,
    ids=lambda serializer: serializer.serializer_id,
)
@pytest.mark.parametrize("nullable_case", ("mixed", "all_missing"))
def test_registered_serializer_round_trips_nullable_boolean_dtype_family(
    serializer: FrameSerializerSpec,
    nullable_case: str,
    tmp_path: Path,
) -> None:
    assert_registered_serializer_round_trip(serializer, nullable_case, tmp_path)


def test_annual_serializer_preserves_supported_complete_boolean_columns(tmp_path):
    observation = _round_trip_us_annual_static_aging(tmp_path, "complete")
    for column in (NATIVE_COLUMN, COMPLETE_COLUMN, MISSING_COLUMN):
        assert observation.loaded[column].dtype == np.dtype(np.bool_)
        np.testing.assert_array_equal(
            observation.loaded[column],
            observation.source[column].to_numpy(dtype=np.bool_),
        )
