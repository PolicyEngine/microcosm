"""The local-annual release contract (microcosm#922, the County Files).

A build of ``microcosm-us-2024-local-annual`` carries a build of
``microcosm-us-2024-local`` unchanged and adds one H5 per projected year.
Every H5 here is a real pandas fixed-format file with the six US entity
tables, as the local-area writer stores it, so the contract's h5py reader of
that layout is tested against the layout itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from microcosm.data import annual_projections, contract
from microcosm.data.contract import (
    US_LOCAL_ANNUAL_RELEASE_ID,
    US_LOCAL_RELEASE_ID,
    US_NATIONAL_RELEASE_ID,
    US_SPM_MEASUREMENT_EVIDENCE_KEY,
    US_STORED_INPUTS_EVIDENCE_KEY,
    ReleaseContractError,
    validate_release_dir,
)
from test_support.microcosm_data.annual_projections import (  # noqa: F401
    annual_build_folder,
    artifact_root,
    national_build,
    release_dir,
    rewrite_annual_evidence,
    rewrite_annual_release_manifest,
)
from test_support.microcosm_data.local_annual import (
    LOCAL_BUILD,
    LOCAL_DATASET_KEY,
    LOCAL_REPO,
    local_annual_build_folder_of,
    local_build_folder,
    local_tables,
    projected_key,
    read_json,
    replace_projected_h5,
    reseal,
    rewrite_json,
    rewrite_local_annual_evidence,
    sha256,
    write_json,
    write_local_h5,
)

_PROPERTY = settings(max_examples=60, deadline=None)
_FOLDER_PROPERTY = settings(max_examples=25, deadline=None)


@pytest.fixture
def local(tmp_path):
    """A local-area build folder and its artifact root."""
    return local_build_folder(tmp_path), tmp_path


@pytest.fixture
def annual(local):
    """A local-annual build of :func:`local` for 2026 and 2027, and its root."""
    folder, root = local
    return local_annual_build_folder_of(folder, root), root


def _failures(annual_dir: Path, root: Path) -> str:
    with pytest.raises(ReleaseContractError) as excinfo:
        validate_release_dir(annual_dir, artifact_root=root)
    return "\n".join(excinfo.value.failures)


def test_a_local_annual_build_passes_the_contract(annual) -> None:
    folder, root = annual

    assert validate_release_dir(folder, artifact_root=root) == []
    manifest = read_json(folder / "release_manifest.json")
    assert manifest["release"] == US_LOCAL_ANNUAL_RELEASE_ID
    assert manifest["default_datasets"] == {}
    # The County Files' years skip 2025: the local-annual release has no
    # rule that every year from the base through the last is present.
    assert manifest["metadata"]["dataset_years"] == {
        LOCAL_DATASET_KEY: {
            "2024": LOCAL_DATASET_KEY,
            "2026": "microcosm_us_2026_acs_local",
            "2027": "microcosm_us_2027_acs_local",
        }
    }
    extension = annual_projections.validate_annual_projection_extension(
        folder, manifest, artifact_root=root
    )
    assert extension.revision is None
    assert extension.projected_artifacts == {
        "microcosm_us_2026_acs_local",
        "microcosm_us_2027_acs_local",
    }


def test_the_local_base_build_passes_its_own_contract(local) -> None:
    folder, root = local

    assert validate_release_dir(folder, artifact_root=root) == []


def test_only_the_annual_releases_declare_annual_years() -> None:
    years = {"metadata": {"dataset_years": {"x": {"2024": "x"}}}}

    assert annual_projections.annual_release_refusal(US_LOCAL_RELEASE_ID, {}) is None
    assert "cannot" in annual_projections.annual_release_refusal(
        US_LOCAL_RELEASE_ID, years
    )
    assert "must declare" in annual_projections.annual_release_refusal(
        US_LOCAL_ANNUAL_RELEASE_ID, {}
    )
    assert (
        annual_projections.annual_release_refusal(US_LOCAL_ANNUAL_RELEASE_ID, years)
        is None
    )
    assert annual_projections.is_annual_build(Path("x") / US_LOCAL_ANNUAL_RELEASE_ID)
    assert annual_projections.is_annual_build(Path("x") / "microcosm-us-2024-annual")
    assert not annual_projections.is_annual_build(Path("x") / US_LOCAL_RELEASE_ID)


def _set_reference(field, value):
    def mutate(manifest):
        manifest["metadata"]["annual_projection_base"][field] = value

    return mutate


def _rename_projected_file(folder: Path, root: Path) -> None:
    """Publish 2026 under the national annual file name."""
    (root / "microcosm_us_2026_acs_local.h5").rename(root / "microcosm_us_2026.h5")

    def mutate(manifest):
        manifest["artifacts"]["microcosm_us_2026_acs_local"]["path"] = (
            "microcosm_us_2026.h5"
        )

    rewrite_json(folder, root, "release_manifest.json", mutate)


def _extra_microdata(folder: Path, root: Path) -> None:
    extra = write_local_h5(root / "extra.h5", 2024)

    def mutate(manifest):
        manifest["artifacts"]["extra"] = {
            "kind": "microdata",
            "path": extra.name,
            "repo_id": LOCAL_REPO,
            "sha256": sha256(extra),
            "role": "dataset",
        }

    rewrite_json(folder, root, "release_manifest.json", mutate)


def _gate_sha(key):
    def apply(folder: Path, root: Path) -> None:
        def gate(document):
            document["gates"][key]["artifact_sha256"] = "0" * 64

        rewrite_json(folder, root, "gate_summary.json", gate)
        build = read_json(folder / "build_manifest.json")
        build["gates"][key]["artifact_sha256"] = "0" * 64
        write_json(folder / "build_manifest.json", build)
        reseal(folder, root)

    return apply


def _acceptance(mutate):
    def apply(folder: Path, root: Path) -> None:
        rewrite_json(folder, root, "annual_acceptance.json", mutate)

    return apply


def _drop_projected_years(folder: Path, root: Path) -> None:
    only_base = {LOCAL_DATASET_KEY: {"2024": LOCAL_DATASET_KEY}}

    def evidence(document):
        document["metadata"]["dataset_years"] = only_base

    rewrite_local_annual_evidence(folder, root, evidence)

    def manifest(document):
        document["metadata"]["dataset_years"] = only_base
        for key in [*document["artifacts"]]:
            if key.startswith(
                ("microcosm_us_2026", "microcosm_us_2027", "projection_")
            ):
                del document["artifacts"][key]

    rewrite_json(folder, root, "release_manifest.json", manifest)


def _projected(year, **options):
    def apply(folder: Path, root: Path) -> None:
        stored_year = options.pop("stored_year", None)
        tables = local_tables(3, weight_scale=1.02)
        for entity, change in options.items():
            change(tables[entity])
        replace_projected_h5(folder, root, year, tables, stored_year=stored_year)

    return apply


def _permute(column):
    def change(table):
        table[column] = table[column].to_numpy()[::-1].copy()

    return change


def _set(column, value):
    def change(table):
        table[column] = value

    return change


def _object_ids(table):
    table["person_id"] = table["person_id"].astype(str).astype(object)


@pytest.mark.parametrize(
    ("problem", "refusal"),
    [
        (
            lambda f, r: rewrite_json(
                f,
                r,
                "release_manifest.json",
                _set_reference("release", US_NATIONAL_RELEASE_ID),
            ),
            f"projects a build of {US_LOCAL_RELEASE_ID!r}",
        ),
        (
            lambda f, r: rewrite_json(
                f, r, "release_manifest.json", _set_reference("build", "other-build")
            ),
            "does not start with its base build",
        ),
        (_rename_projected_file, "'microcosm_us_2026_acs_local.h5'"),
        (
            lambda f, r: rewrite_json(
                f,
                r,
                "release_manifest.json",
                lambda m: m.update(default_datasets={"national": LOCAL_DATASET_KEY}),
            ),
            "claims no default dataset",
        ),
        (_extra_microdata, "besides its projected years"),
        (
            _acceptance(lambda a: a["years"].pop("2027")),
            "2027 acceptance must be an object",
        ),
        (
            _acceptance(
                lambda a: a["years"].update(
                    {"2025": {"dataset": "x", "sha256": "0" * 64, "checks": {}}}
                )
            ),
            "acceptance coverage differs",
        ),
        (
            _acceptance(lambda a: a["years"]["2026"]["checks"].update(runtime="no")),
            "lacks passing acceptance",
        ),
        (_projected(2026, person=_permute("person_id")), "person_id differs"),
        (
            _projected(2027, person=_permute("person_household_id")),
            "person_household_id differs",
        ),
        (
            _projected(2026, household=_permute("household_id")),
            "household_id differs",
        ),
        (
            _projected(2026, household=_set("household_weight", 0.0)),
            "strictly positive",
        ),
        (
            _projected(2026, household=_set("household_weight", np.nan)),
            "finite",
        ),
        (_projected(2026, stored_year=2025), "stored year does not match 2026"),
        (_projected(2026, person=_set("extra_column", 1.0)), "columns differ"),
        (_projected(2026, person=_object_ids), "not in a numeric block"),
        (_drop_projected_years, "at least one projected year"),
        (_gate_sha(US_STORED_INPUTS_EVIDENCE_KEY), "certifies artifact_sha256"),
        (_gate_sha(US_SPM_MEASUREMENT_EVIDENCE_KEY), "certifies artifact_sha256"),
    ],
)
def test_the_contract_refuses_a_local_annual_build_that_does_not_hold(
    annual, problem, refusal
) -> None:
    folder, root = annual
    problem(folder, root)

    assert refusal in _failures(folder, root)


def test_a_changed_row_count_is_refused(annual) -> None:
    folder, root = annual
    replace_projected_h5(folder, root, 2026, local_tables(4))

    assert "row count differs" in _failures(folder, root)


def test_the_gates_still_certify_the_base_h5_when_several_microdata_are_declared(
    annual,
) -> None:
    """The package stage's gates certify one H5. In a local-annual build they
    are the base build's and are held to its base H5, not skipped because the
    build declares several microdata artifacts."""
    folder, root = annual
    manifest = read_json(folder / "release_manifest.json")
    gated = contract._us_local_area_gated_microdata(
        manifest, US_LOCAL_ANNUAL_RELEASE_ID
    )

    assert [entry["path"] for entry in gated] == [f"{LOCAL_DATASET_KEY}.h5"]
    # Any other release keeps every microdata artifact in scope.
    assert len(contract._us_local_area_gated_microdata(manifest, None)) == 3


def test_a_national_annual_build_still_needs_every_year(
    annual_build_folder,  # noqa: F811
    artifact_root,  # noqa: F811
) -> None:
    """The national annual release keeps its rule that every year from the
    base through the last is present; only the local-annual release drops it."""
    mapping = {"microcosm_us_2024": {"2024": "microcosm_us_2024", "2026": "x"}}
    rewrite_annual_evidence(
        annual_build_folder,
        lambda evidence: evidence["metadata"].update(dataset_years=mapping),
    )
    rewrite_annual_release_manifest(
        annual_build_folder,
        lambda manifest: manifest["metadata"].update(dataset_years=mapping),
    )

    with pytest.raises(ReleaseContractError, match="every year from its base"):
        validate_release_dir(annual_build_folder, artifact_root=artifact_root)


class _LocalBaseOnTheHub:
    """A Hub serving the local base build's release manifest from its folder."""

    def __init__(self, directory: Path, manifest: dict) -> None:
        self.directory = directory
        self.manifest = manifest
        self.requests: list[tuple] = []

    def hf_hub_download(self, *, repo_id, filename, repo_type, revision):
        self.requests.append((repo_id, filename, repo_type, revision))
        local = self.directory / filename
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(json.dumps(self.manifest))
        return str(local)


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        (None, None),
        (
            lambda m: m["artifacts"][LOCAL_DATASET_KEY].update(sha256="d" * 64),
            "projects exactly the published base build's bytes",
        ),
        (
            lambda m: m["artifacts"]["county_reliability"].update(sha256="d" * 64),
            "county_reliability",
        ),
        (
            lambda m: (
                m["artifacts"]["extra"].update(sha256="d" * 64)
                if "extra" in m["artifacts"]
                else m["artifacts"].update(
                    extra={
                        "kind": "diagnostics",
                        "path": "extra.json",
                        "repo_id": LOCAL_REPO,
                        "sha256": "a" * 64,
                        "role": "extra",
                    }
                )
            ),
            "extra",
        ),
        (
            lambda m: m["artifacts"].update(
                second={
                    "kind": "microdata",
                    "path": "second.h5",
                    "repo_id": LOCAL_REPO,
                    "sha256": "a" * 64,
                    "role": "dataset",
                }
            ),
            "projects exactly the published base build's bytes",
        ),
    ],
)
def test_a_local_annual_build_needs_its_published_local_base(
    local, tmp_path, change, refusal
) -> None:
    """Before a local-annual build is handed over, its base build's manifest
    is read from releases/microcosm-us-2024-local/<base build>/ at main. Its
    one microdata artifact must be the base-year H5, and every other file it
    declares must be carried unchanged."""
    folder, root = local
    published = read_json(folder / "release_manifest.json")
    annual_dir = local_annual_build_folder_of(folder, root)
    if change is not None:
        change(published)
    hub = _LocalBaseOnTheHub(tmp_path / "hub", published)

    def check():
        annual_projections.require_base_publication(hub, annual_dir, repo_id=LOCAL_REPO)

    if refusal is None:
        check()
    else:
        with pytest.raises(ValueError, match=refusal):
            check()
    assert hub.requests == [
        (
            LOCAL_REPO,
            f"releases/{US_LOCAL_RELEASE_ID}/{LOCAL_BUILD}/release_manifest.json",
            "dataset",
            "main",
        )
    ]


def test_a_local_annual_build_naming_a_national_base_is_not_looked_up(annual) -> None:
    folder, root = annual
    rewrite_json(
        folder,
        root,
        "release_manifest.json",
        _set_reference("release", US_NATIONAL_RELEASE_ID),
    )
    hub = _LocalBaseOnTheHub(root / "hub", {})

    with pytest.raises(ValueError, match="base build is a build of"):
        annual_projections.require_base_publication(hub, folder, repo_id=LOCAL_REPO)
    assert hub.requests == []


_YEARS = st.sets(st.integers(min_value=2025, max_value=2040), min_size=1, max_size=4)


def _manifest_for(years, *, base_file=f"{LOCAL_DATASET_KEY}.h5"):
    family = {"2024": LOCAL_DATASET_KEY}
    artifacts = {
        LOCAL_DATASET_KEY: {"kind": "microdata", "path": base_file},
        "gate_summary": {"kind": "diagnostics", "path": "gate_summary.json"},
    }
    for year in years:
        family[str(year)] = projected_key(year)
        artifacts[projected_key(year)] = {
            "kind": "microdata",
            "path": f"{projected_key(year)}.h5",
        }
    return {
        "metadata": {"dataset_years": {LOCAL_DATASET_KEY: family}},
        "artifacts": artifacts,
    }


def _dataset_file_failures(manifest) -> list[str]:
    failures: list[str] = []
    contract._check_us_release_dataset_file(
        manifest["artifacts"],
        US_LOCAL_ANNUAL_RELEASE_ID,
        failures,
        manifest=manifest,
    )
    return failures


@_PROPERTY
@given(
    years=_YEARS, base_file=st.sampled_from(contract.US_LOCAL_ANNUAL_BASE_DATASET_FILES)
)
def test_any_set_of_projected_years_with_their_names_passes_the_file_check(
    years, base_file
) -> None:
    manifest = _manifest_for(years, base_file=base_file)

    assert _dataset_file_failures(manifest) == []
    assert contract.us_local_annual_projected_keys(manifest) == {
        projected_key(year): str(year) for year in years
    }
    assert [
        entry["path"]
        for entry in contract._us_local_area_gated_microdata(
            manifest, US_LOCAL_ANNUAL_RELEASE_ID
        )
    ] == [base_file]


@_PROPERTY
@given(
    years=_YEARS,
    data=st.data(),
    wrong=st.sampled_from(["national_key", "national_path", "other_year", "kind"]),
)
def test_any_misnamed_projected_year_fails_the_file_check(years, data, wrong) -> None:
    manifest = _manifest_for(years)
    year = data.draw(st.sampled_from(sorted(years)))
    key = projected_key(year)
    family = manifest["metadata"]["dataset_years"][LOCAL_DATASET_KEY]
    entry = manifest["artifacts"][key]
    if wrong == "national_key":
        manifest["artifacts"][f"microcosm_us_{year}"] = manifest["artifacts"].pop(key)
        family[str(year)] = f"microcosm_us_{year}"
    elif wrong == "national_path":
        entry["path"] = f"microcosm_us_{year}.h5"
    elif wrong == "other_year":
        entry["path"] = f"{projected_key(year + 100)}.h5"
    else:
        entry["kind"] = "diagnostics"

    assert any(str(year) in failure for failure in _dataset_file_failures(manifest))


@_PROPERTY
@given(years=_YEARS, base_file=st.sampled_from(["microcosm_us_2024.h5", "x.h5"]))
def test_a_base_outside_the_local_names_fails_the_file_check(years, base_file) -> None:
    assert any(
        "base local-area build's dataset" in failure
        for failure in _dataset_file_failures(_manifest_for(years, base_file=base_file))
    )


@_FOLDER_PROPERTY
@given(years=_YEARS)
def test_any_set_of_projected_years_builds_a_valid_local_annual_build(
    tmp_path_factory, years
) -> None:
    root = tmp_path_factory.mktemp("local-annual")
    folder = local_annual_build_folder_of(local_build_folder(root), root, years=years)

    assert validate_release_dir(folder, artifact_root=root) == []


@_PROPERTY
@given(
    households=st.integers(min_value=2, max_value=6),
    entity=st.sampled_from(
        ["person", "household", "tax_unit", "spm_unit", "family", "marital_unit"]
    ),
    data=st.data(),
)
def test_the_fixed_layout_check_refuses_every_reordered_identity_column(
    tmp_path_factory, households, entity, data
) -> None:
    """For any population size, entity and non-identity reordering of that
    entity's IDs, the projected year differs from its base and is refused;
    the unchanged tables pass."""
    directory = tmp_path_factory.mktemp("fixed")
    base = write_local_h5(directory / "base.h5", 2024, households=households)
    tables = local_tables(households)
    rows = {name: len(table) for name, table in tables.items()}
    same = write_local_h5(directory / "same.h5", 2026, local_tables(households))
    annual_projections._check_fixed_native_identity(
        base, same, 2026, 2024, {"rows": rows}
    )
    n = rows[entity]
    order = data.draw(st.permutations(range(n)))
    if list(order) == list(range(n)):
        order = list(range(1, n)) + [0]
    column = f"{entity}_id"
    tables[entity][column] = tables[entity][column].to_numpy()[list(order)]
    changed = write_local_h5(directory / "changed.h5", 2026, tables)

    with pytest.raises(ValueError, match=f"{column} differs"):
        annual_projections._check_fixed_native_identity(
            base, changed, 2026, 2024, {"rows": rows}
        )


@_PROPERTY
@given(
    weights=st.lists(
        st.floats(
            min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
        min_size=2,
        max_size=6,
    ),
    bad_weight=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        st.sampled_from([np.nan, np.inf, -np.inf]),
    ),
    row=st.integers(min_value=0, max_value=5),
)
@example(weights=[1.0, 2.0], bad_weight=0.0, row=1)
@example(weights=[1.0, 2.0], bad_weight=-1.0, row=0)
@example(weights=[1.0, 2.0], bad_weight=np.nan, row=1)
@example(weights=[1.0, 2.0], bad_weight=np.inf, row=0)
@example(weights=[1.0, 2.0], bad_weight=-np.inf, row=1)
def test_the_fixed_layout_check_refuses_any_single_invalid_household_weight(
    tmp_path_factory, weights, bad_weight, row
) -> None:
    """Positive weights elsewhere cannot hide a zero, negative or nonfinite row."""
    directory = tmp_path_factory.mktemp("fixed-weight")
    base = write_local_h5(
        directory / "base.h5", 2024, weights=weights, households=len(weights)
    )
    tables = local_tables(weights=weights, households=len(weights))
    rows = {entity: len(table) for entity, table in tables.items()}
    valid = write_local_h5(directory / "valid.h5", 2026, tables)
    annual_projections._check_fixed_native_identity(
        base, valid, 2026, 2024, {"rows": rows}
    )
    tables["household"].loc[row % len(weights), "household_weight"] = bad_weight
    invalid = write_local_h5(directory / "invalid.h5", 2026, tables)

    with pytest.raises(ValueError, match="finite and strictly positive"):
        annual_projections._check_fixed_native_identity(
            base, invalid, 2026, 2024, {"rows": rows}
        )
