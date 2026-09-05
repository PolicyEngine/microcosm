"""Lookup artifact bytes retain the file loaders' actual geography contracts."""

import pandas as pd
import pytest

from microcosm.build.us_runtime.congressional_district_vintage import (
    decode_congressional_district_vintage_crosswalk,
    default_congressional_district_vintage_crosswalk_path,
    load_default_congressional_district_vintage_crosswalk,
    validate_packaged_congressional_district_vintage_crosswalk,
)
from microcosm.build.us_runtime.puma_ladder import (
    decode_us_puma_ladder,
)


def test_crosswalk_bytes_equal_validated_national_file():
    payload = default_congressional_district_vintage_crosswalk_path().read_bytes()
    decoded = decode_congressional_district_vintage_crosswalk(payload)
    pd.testing.assert_frame_equal(
        decoded, load_default_congressional_district_vintage_crosswalk()
    )
    validate_packaged_congressional_district_vintage_crosswalk(decoded)


@pytest.mark.parametrize(
    "bad",
    [
        b"source_geography_id,target_geography_id,weight\n5001700US0101,5001900US0200,1\n",
        b"source_geography_id,target_geography_id,weight,pair_population\n5001700US0101,5001900US0101,0.9,50\n5001700US0101,5001900US0102,0.1,50\n",
    ],
)
def test_crosswalk_bytes_refuse_invalid_geometry_or_population_weights(bad):
    with pytest.raises(ValueError):
        decode_congressional_district_vintage_crosswalk(bad)


@pytest.mark.parametrize(
    "decoder", [decode_us_puma_ladder, decode_congressional_district_vintage_crosswalk]
)
def test_lookup_decoders_require_immutable_bytes(decoder):
    with pytest.raises(TypeError, match="immutable bytes"):
        decoder(bytearray(b"mutable"))
