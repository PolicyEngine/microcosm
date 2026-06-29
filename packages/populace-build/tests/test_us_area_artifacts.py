import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from populace.build.us_runtime.area_artifacts import (
    AreaArtifactResult,
    AreaArtifactSpec,
    assert_complete_area_artifacts,
    congressional_district_artifact_specs,
    select_household_area,
    state_artifact_specs,
    write_area_artifacts,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights


def test_area_specs_match_policyengine_release_key_conventions() -> None:
    frame = _area_frame()

    state_specs = state_artifact_specs(frame, require_complete=False)
    district_specs = congressional_district_artifact_specs(
        frame, require_complete=False
    )

    assert [spec.key for spec in state_specs] == ["states/AK", "states/CA"]
    assert [spec.path for spec in state_specs] == ["states/AK.h5", "states/CA.h5"]
    assert [spec.key for spec in district_specs] == [
        "districts/AK-01",
        "districts/CA-01",
        "districts/CA-02",
    ]
    assert [spec.selector_value for spec in district_specs] == [200, 601, 602]


def test_select_household_area_prunes_every_entity_and_weight_vector() -> None:
    frame = _area_frame()
    spec = AreaArtifactSpec(
        key="states/CA",
        path="states/CA.h5",
        kind="state_microdata",
        selector_column="state_fips",
        selector_value=6,
    )

    selected = select_household_area(frame, spec)

    assert selected.table("household")["household_id"].tolist() == [10, 20]
    assert selected.person["person_id"].tolist() == [1, 2, 3]
    assert selected.table("tax_unit")["tax_unit_id"].tolist() == [100, 200]
    assert selected.table("spm_unit")["spm_unit_id"].tolist() == [1000, 2000]
    assert selected.table("family")["family_id"].tolist() == [10000, 20000]
    assert selected.table("marital_unit")["marital_unit_id"].tolist() == [
        100000,
        200000,
    ]
    assert selected.weights_for("household").values.tolist() == [50.0, 60.0]


def test_write_area_artifacts_returns_manifest_ready_metadata(tmp_path: Path) -> None:
    frame = _area_frame()
    specs = state_artifact_specs(frame, require_complete=False)[:1]

    def fake_writer(area: Frame, path: Path, period: int) -> None:
        assert period == 2024
        path.write_bytes(
            f"{area.n('household')} households {area.n('person')} persons".encode()
        )

    results = write_area_artifacts(
        frame,
        specs,
        output_root=tmp_path,
        period=2024,
        writer=fake_writer,
    )

    assert len(results) == 1
    result = results[0]
    assert result.key == "states/AK"
    assert result.path == "states/AK.h5"
    assert result.kind == "state_microdata"
    assert result.n_households == 1
    assert result.n_persons == 1
    assert result.sha256 == hashlib.sha256(b"1 households 1 persons").hexdigest()


def test_area_artifact_release_contract_rejects_partial_surface() -> None:
    try:
        assert_complete_area_artifacts(
            (
                AreaArtifactResult(
                    key="states/CA",
                    path="states/CA.h5",
                    kind="state_microdata",
                    sha256="a" * 64,
                    n_households=1,
                    n_persons=1,
                ),
            )
        )
    except ValueError as exc:
        assert "Incomplete area artifact keys" in str(exc)
        assert "missing=" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected partial area artifact release to fail.")


def test_area_artifact_release_contract_rejects_wrong_path() -> None:
    artifacts = [
        AreaArtifactResult(
            key=spec.key,
            path=spec.path,
            kind=spec.kind,
            sha256="a" * 64,
            n_households=1,
            n_persons=1,
        )
        for spec in (
            *_all_state_specs(),
            *_all_congressional_district_specs(),
        )
    ]
    artifacts[0] = AreaArtifactResult(
        key=artifacts[0].key,
        path="states/not-the-public-key.h5",
        kind=artifacts[0].kind,
        sha256=artifacts[0].sha256,
        n_households=artifacts[0].n_households,
        n_persons=artifacts[0].n_persons,
    )

    try:
        assert_complete_area_artifacts(tuple(artifacts))
    except ValueError as exc:
        assert "has path" in str(exc)
        assert "expected" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected wrong area artifact path to fail.")


def test_at_large_98_proxy_maps_to_public_district_01() -> None:
    frame = _area_frame(congressional_district_geoids=[601, 602, 298])

    district_specs = congressional_district_artifact_specs(
        frame, require_complete=False
    )

    assert [spec.key for spec in district_specs] == [
        "districts/AK-01",
        "districts/CA-01",
        "districts/CA-02",
    ]
    assert [spec.selector_value for spec in district_specs] == [298, 601, 602]


def test_complete_state_specs_require_full_release_surface() -> None:
    frame = _area_frame()

    try:
        state_artifact_specs(frame)
    except ValueError as exc:
        assert "Incomplete state_fips" in str(exc)
        assert "missing=" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected sparse state frame to fail completeness.")


def test_complete_district_specs_require_full_release_surface() -> None:
    frame = _area_frame()

    try:
        congressional_district_artifact_specs(frame)
    except ValueError as exc:
        assert "Incomplete congressional district artifact keys" in str(exc)
        assert "missing=" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected sparse district frame to fail completeness.")


def test_district_specs_reject_duplicate_public_artifact_keys() -> None:
    frame = _area_frame(
        state_fips=[2, 2, 6],
        congressional_district_geoids=[200, 298, 601],
    )

    try:
        congressional_district_artifact_specs(frame, require_complete=False)
    except ValueError as exc:
        assert "districts/AK-01" in str(exc)
        assert "200" in str(exc)
        assert "298" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected duplicate public district key to fail.")


def test_at_large_proxy_is_invalid_for_multi_district_states() -> None:
    frame = _area_frame(congressional_district_geoids=[600, 602, 200])

    try:
        congressional_district_artifact_specs(frame, require_complete=False)
    except ValueError as exc:
        assert "only valid for states with one district" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected CA at-large proxy to fail.")


def test_district_specs_reject_state_geoid_mismatch() -> None:
    frame = _area_frame(
        state_fips=[6, 2, 6],
        congressional_district_geoids=[3601, 200, 601],
    )

    try:
        congressional_district_artifact_specs(frame, require_complete=False)
    except ValueError as exc:
        assert "state_fips=6" in str(exc)
        assert "congressional_district_geoid=3601" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected mismatched state/CD geoid to fail.")


def test_legacy_117th_district_key_is_invalid_for_current_release() -> None:
    frame = _area_frame(congressional_district_geoids=[653, 601, 200])

    try:
        congressional_district_artifact_specs(frame, require_complete=False)
    except ValueError as exc:
        assert "Congressional district 53" in str(exc)
        assert "CA" in str(exc)
        assert "52 district" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected stale CA-53 district key to fail.")


def _area_frame(
    *,
    state_fips: list[int] | None = None,
    congressional_district_geoids: list[int] | None = None,
) -> Frame:
    state_fips = state_fips or [6, 6, 2]
    congressional_district_geoids = congressional_district_geoids or [601, 602, 200]
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": np.asarray([1, 2, 3, 4], dtype="int64"),
                "person_household_id": np.asarray([10, 20, 20, 30], dtype="int64"),
                "person_tax_unit_id": np.asarray([100, 200, 200, 300], dtype="int64"),
                "person_spm_unit_id": np.asarray(
                    [1000, 2000, 2000, 3000], dtype="int64"
                ),
                "person_family_id": np.asarray(
                    [10000, 20000, 20000, 30000], dtype="int64"
                ),
                "person_marital_unit_id": np.asarray(
                    [100000, 200000, 200000, 300000],
                    dtype="int64",
                ),
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": np.asarray([10, 20, 30], dtype="int64"),
                "state_fips": np.asarray(state_fips, dtype="int64"),
                "congressional_district_geoid": np.asarray(
                    congressional_district_geoids,
                    dtype="int64",
                ),
            }
        ),
        "tax_unit": pd.DataFrame(
            {"tax_unit_id": np.asarray([100, 200, 300], dtype="int64")}
        ),
        "spm_unit": pd.DataFrame(
            {"spm_unit_id": np.asarray([1000, 2000, 3000], dtype="int64")}
        ),
        "family": pd.DataFrame(
            {"family_id": np.asarray([10000, 20000, 30000], dtype="int64")}
        ),
        "marital_unit": pd.DataFrame(
            {
                "marital_unit_id": np.asarray(
                    [100000, 200000, 300000],
                    dtype="int64",
                )
            }
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([50.0, 60.0, 70.0]),
                WeightKind.CALIBRATED,
            )
        },
    )


def _all_state_specs() -> tuple[AreaArtifactSpec, ...]:
    return state_artifact_specs(_complete_area_frame())


def _all_congressional_district_specs() -> tuple[AreaArtifactSpec, ...]:
    return congressional_district_artifact_specs(_complete_area_frame())


def _complete_area_frame() -> Frame:
    state_fips: list[int] = []
    congressional_district_geoids: list[int] = []
    from populace.build.us_runtime.area_artifacts import (
        CONGRESSIONAL_DISTRICT_COUNT_BY_FIPS,
    )

    for fips, district_count in CONGRESSIONAL_DISTRICT_COUNT_BY_FIPS.items():
        state_fips.extend([fips] * district_count)
        congressional_district_geoids.extend(
            fips * 100 + district_number
            for district_number in range(1, district_count + 1)
        )
    household_count = len(state_fips)
    ids = np.arange(1, household_count + 1, dtype="int64")
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": ids,
                "person_household_id": ids,
                "person_tax_unit_id": ids,
                "person_spm_unit_id": ids,
                "person_family_id": ids,
                "person_marital_unit_id": ids,
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": ids,
                "state_fips": np.asarray(state_fips, dtype="int64"),
                "congressional_district_geoid": np.asarray(
                    congressional_district_geoids, dtype="int64"
                ),
            }
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids}),
        "family": pd.DataFrame({"family_id": ids}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(household_count, dtype="float64"),
                WeightKind.CALIBRATED,
            )
        },
    )
