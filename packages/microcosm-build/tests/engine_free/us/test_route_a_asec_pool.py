"""Route A pools main's default ASEC years, with the registries' pins.

Max's d2603 (2026-11-07) moved Route A's driver (``tools/route_a/route_a.sh``)
and its Modal base plan template
(``docs/us-modal-stage-route-a-rerun-base-plan.json``) from income years
2022-2024 to main's default pool,
:data:`microcosm.build.us_runtime.asec_sources.ASEC_DEFAULT_POOL_INCOME_YEARS`
(income years 2023-2025, ASEC 2024-2026). Both write their years and pins out
instead of deriving them, so these tests hold them to the registries:

- The driver's and the template's pooled years equal the default pool. A PR
  that pins a newer year moves the default pool, so it fails here until it
  moves Route A as well.
- Every pooled year sits in all four year-keyed registries.
- Every ASEC pin the driver or the template carries is its registry's: the
  digest, the byte count and the file name, and the LKWEEKS archive's.
- The driver's base command is the default-pool registration's command, flag
  for flag and value for value except file locations (a differential test).
- ``check_commit`` refuses a build commit whose registries lack a pooled pin.
- The default-pool registration differs from the rerun registration only in
  its pool and its seeded LKWEEKS archive, and its argv builder maps any pool
  year for year (a Hypothesis property).
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.us_runtime import weeks_unemployed
from microcosm.build.us_runtime.asec_sources import (
    ASEC_DEFAULT_POOL_INCOME_YEARS,
    ASEC_SOURCE_ARTIFACTS,
)
from microcosm.build.us_runtime.education_assistance_source import (
    ASEC_EDUCATION_ASSISTANCE_ARCHIVES,
)
from microcosm.build.us_runtime.public_assistance_type_source import (
    ASEC_PUBLIC_ASSISTANCE_TYPE_AUDIT_PINS,
)
from microcosm.build.us_runtime.spm_role_source import ASEC_SPM_ROLE_SOURCES
from test_support.git_repositories import NO_BACKGROUND_MAINTENANCE
from test_support.microcosm_build.route_a_driver import (
    ROUTE_A_TOOLS,
    run_driver_settings,
)
from test_support.microcosm_build.us_modal_stage_plan_tool import ROOT, plan_lib

TEMPLATE = ROOT / "docs" / "us-modal-stage-route-a-rerun-base-plan.json"
DEFAULT_POOL_TOOL = plan_lib.US_PUF_SUPPORT_BASE_2023_2025
SEED = plan_lib.LKWEEKS_ARCHIVE_SEED
ASEC_PIN_REL = "packages/microcosm-build/src/microcosm/build/us_runtime/asec_sources.py"
EDU_PIN_REL = (
    "packages/microcosm-build/src/microcosm/build/us_runtime/"
    "education_assistance_source.py"
)
# Where these tests tell route_a.sh its inputs live.
_INPUT_ROOT = "/route-a"
_STORAGE = f"{_INPUT_ROOT}/storage"
_EDU = f"{_INPUT_ROOT}/edu"
_ASEC_ROLE = re.compile(r"asec_(\d{4})_h5")
_EDUCATION_ROLE = re.compile(r"asec_education_(\d{4})_zip")
_MAPPING_FLAGS = ("--asec-h5", "--asec-h5-sha256", "--asec-education-source")


def _template_plan() -> plan_lib.Plan:
    data = json.loads(TEMPLATE.read_text())
    # The template refuses validation until an operator picks the commit.
    data["source"]["commit"] = "0" * 40
    return plan_lib.parse_plan(data)


def _years(roles, pattern: re.Pattern[str]) -> list[int]:
    return [int(match.group(1)) for role in roles if (match := pattern.fullmatch(role))]


def _archive_name(url: str) -> str:
    return PurePosixPath(urlsplit(url).path).name


def _driver_settings() -> dict[str, str]:
    names = {role: ref.filename for role, ref in _template_plan().inputs.items()}
    return {
        "STORAGE": _STORAGE,
        "EDU": _EDU,
        "FEED": f"{_INPUT_ROOT}/feed/{names['base_ledger_facts']}",
        "LADDER": f"{_INPUT_ROOT}/ladder/{names['block_ladder_npz']}",
    }


def _driver_rows(tmp_path: Path) -> dict[str, list[str]]:
    """input_rows(): role -> [path, sha256, bytes, where main pins it]."""

    result = run_driver_settings(tmp_path, _driver_settings(), "\ninput_rows\n")
    assert result.returncode == 0, result.stderr.decode()
    rows = {}
    for line in result.stdout.decode().splitlines():
        role, *fields = line.split("|")
        rows[role] = fields
    return rows


def _driver_base_argv(tmp_path: Path) -> list[str]:
    """The base command of route_a.sh section 4, as bash expands it."""

    source = (ROUTE_A_TOOLS / "route_a.sh").read_text(encoding="utf-8")
    match = re.search(r'(?ms)^  write_config base .*?--out "\$BASE_OUT"$', source)
    assert match is not None, "route_a.sh has no base write_config"
    state = f"{plan_lib.WORK_ROOT}/state"
    commands = "\n".join(
        [
            "",
            # The driver's write_config takes eight settings and '--', then argv.
            "write_config() { shift 9; printf '%s\\0' \"$@\"; }",
            f"PY={plan_lib.IMAGE_VENV}/bin/python",
            f"W={plan_lib.IMAGE_REPO_ROOT}",
            f"BASE_CKPT={state}/{plan_lib.PUF_SUPPORT_BASE_CHECKPOINTS}",
            f"BASE_OUT={state}/{plan_lib.PUF_SUPPORT_BASE_OUT}",
            "base_disk=26",
            "BUILD_ENV='{}'",
            match.group(0),
            "",
        ]
    )
    result = run_driver_settings(tmp_path, _driver_settings(), commands)
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout.decode().rstrip("\0").split("\0")


def _as_staged(argv: list[str], plan: plan_lib.Plan) -> list[str]:
    """Move each input the driver names to where the Modal runner stages it."""

    staged = {
        ref.filename: plan_lib.input_local_path(plan_lib.WORK_ROOT, ref)
        for ref in plan.inputs.values()
    }

    def move(value: str) -> str:
        year, eq, rest = value.partition("=")
        if eq and year.isdigit():
            return f"{year}={move(rest)}"
        name = PurePosixPath(value).name
        if value.startswith(_INPUT_ROOT + "/") and name in staged:
            return staged[name]
        return value

    return [move(value) for value in argv]


def _mapping(argv: list[str], flag: str) -> list[tuple[int, str]]:
    pairs = []
    for index, value in enumerate(argv[:-1]):
        if value == flag:
            year, _, rest = argv[index + 1].partition("=")
            pairs.append((int(year), rest))
    return pairs


def test_route_a_pools_main_s_default_asec_years(tmp_path: Path) -> None:
    default = set(ASEC_DEFAULT_POOL_INCOME_YEARS)
    rows = _driver_rows(tmp_path)
    assert set(_years(rows, _ASEC_ROLE)) == default
    assert set(_years(rows, _EDUCATION_ROLE)) == default
    argv = _driver_base_argv(tmp_path)
    for flag in _MAPPING_FLAGS:
        assert {year for year, _ in _mapping(argv, flag)} == default, flag

    plan = _template_plan()
    assert plan.tool is DEFAULT_POOL_TOOL
    assert set(_years(plan.inputs, _ASEC_ROLE)) == default
    assert set(_years(plan.inputs, _EDUCATION_ROLE)) == default
    pool = plan_lib.PUF_SUPPORT_BASE_POOLS[plan.tool.name]
    assert tuple(sorted(pool.asec_years)) == ASEC_DEFAULT_POOL_INCOME_YEARS
    # Route A pools newest first, as it did for 2022-2024; the order is part of
    # the build because the tool pools and locks the years in the order given.
    assert [year for year, _ in _mapping(argv, "--asec-h5")] == list(pool.asec_years)
    assert list(pool.asec_years) == sorted(default, reverse=True)


def test_every_pooled_year_sits_in_all_four_registries(tmp_path: Path) -> None:
    rows = _driver_rows(tmp_path)
    plan = _template_plan()
    pooled = {
        *_years(rows, _ASEC_ROLE),
        *_years(rows, _EDUCATION_ROLE),
        *_years(plan.inputs, _ASEC_ROLE),
        *_years(plan.inputs, _EDUCATION_ROLE),
    }
    assert pooled
    for registry in (
        ASEC_SOURCE_ARTIFACTS,
        ASEC_EDUCATION_ASSISTANCE_ARCHIVES,
        ASEC_SPM_ROLE_SOURCES,
        ASEC_PUBLIC_ASSISTANCE_TYPE_AUDIT_PINS,
    ):
        assert pooled <= set(registry)


def test_every_driver_asec_pin_is_its_registry_s(tmp_path: Path) -> None:
    rows = _driver_rows(tmp_path)
    for year in _years(rows, _ASEC_ROLE):
        path, sha256, size, pinned_by = rows[f"asec_{year}_h5"]
        artifact = ASEC_SOURCE_ARTIFACTS[year]
        assert (path, sha256, int(size)) == (
            f"{_STORAGE}/{artifact.filename}",
            artifact.sha256,
            artifact.size_bytes,
        )
        assert pinned_by.endswith("asec_sources.py ASEC_SOURCE_ARTIFACTS")
    for year in _years(rows, _EDUCATION_ROLE):
        path, sha256, size, pinned_by = rows[f"asec_education_{year}_zip"]
        archive = ASEC_EDUCATION_ASSISTANCE_ARCHIVES[year]
        assert (path, sha256, int(size)) == (
            f"{_EDU}/{_archive_name(archive.zip_url)}",
            archive.zip_sha256,
            archive.zip_size_bytes,
        )
        assert pinned_by.endswith(
            "education_assistance_source.py ASEC_EDUCATION_ASSISTANCE_ARCHIVES"
        )


def test_every_template_asec_pin_is_its_registry_s() -> None:
    plan = _template_plan()
    pool = plan_lib.PUF_SUPPORT_BASE_POOLS[plan.tool.name]
    expected = {}
    for year in pool.asec_years:
        artifact = ASEC_SOURCE_ARTIFACTS[year]
        expected[f"asec_{year}_h5"] = (artifact.sha256, artifact.filename)
    for year in pool.education_years:
        archive = ASEC_EDUCATION_ASSISTANCE_ARCHIVES[year]
        expected[f"asec_education_{year}_zip"] = (
            archive.zip_sha256,
            _archive_name(archive.zip_url),
        )
    lkweeks = (
        weeks_unemployed.ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256,
        _archive_name(weeks_unemployed.ASEC_2023_WEEKS_UNEMPLOYED_ZIP_URL),
    )
    expected[SEED.input] = lkweeks
    for role, (sha256, name) in expected.items():
        ref = plan.inputs[role]
        assert ref.uri == f"volume://{plan_lib.cas_volume_path(sha256, name)}", role
        assert ref.sha256 == sha256, role
    # The LKWEEKS archive is the income-2022 Census archive, which the tool
    # pins once for both uses; the runner seeds the tool's cache with it.
    assert lkweeks == (
        ASEC_EDUCATION_ASSISTANCE_ARCHIVES[2022].zip_sha256,
        _archive_name(ASEC_EDUCATION_ASSISTANCE_ARCHIVES[2022].zip_url),
    )
    assert SEED.sha256 == lkweeks[0]
    assert PurePosixPath(SEED.path).name == lkweeks[1]


def test_the_driver_s_base_command_is_the_default_pool_registration_s(
    tmp_path: Path,
) -> None:
    plan = _template_plan()
    modal = plan_lib.tokenize_stage_argv(plan_lib.planned_argv(plan))
    driver = plan_lib.tokenize_stage_argv(_as_staged(_driver_base_argv(tmp_path), plan))
    assert driver == modal
    # Every path became a token, and every staged input but the seeded LKWEEKS
    # archive is named: the same files, digests, years and order on both sides.
    assert not [value for value in driver if value.startswith(_INPUT_ROOT)]
    named = {
        match for value in modal for match in re.findall(r"<input:([a-z0-9_]+)>", value)
    }
    assert named == set(plan.inputs) - {SEED.input}


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *NO_BACKGROUND_MAINTENANCE, "-C", str(repo), *args], text=True
    ).strip()


@pytest.mark.parametrize("dropped", [None, *ASEC_DEFAULT_POOL_INCOME_YEARS])
def test_check_commit_refuses_a_commit_without_a_pooled_pin(
    tmp_path: Path, dropped: int | None
) -> None:
    repo = tmp_path / "main-repo"
    for rel in (ASEC_PIN_REL, EDU_PIN_REL):
        text = (ROOT / rel).read_text(encoding="utf-8")
        if dropped is not None:
            text = text.replace(ASEC_SOURCE_ARTIFACTS[dropped].sha256, "0" * 64)
            text = text.replace(
                ASEC_EDUCATION_ASSISTANCE_ARCHIVES[dropped].zip_sha256, "0" * 64
            )
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    _git(tmp_path, "init", "--quiet", str(repo))
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.com",
        "commit",
        "--quiet",
        "-m",
        "registries",
    )
    head = _git(repo, "rev-parse", "HEAD")
    result = run_driver_settings(
        tmp_path,
        {"MAIN": str(repo)},
        '\ncheck_commit HEAD HEAD\nprintf "%s\\n" '
        '${COMMIT_BLOCKERS[@]+"${COMMIT_BLOCKERS[@]}"}\n',
    )
    assert result.returncode == 0, result.stderr.decode()
    blockers = [
        line
        for line in result.stdout.decode().splitlines()
        if "asec_sources.py" in line or "education_assistance_source.py" in line
    ]
    if dropped is None:
        assert blockers == []
    else:
        assert blockers == [
            f"asec_sources.py at {head} lacks canonical ASEC pin "
            f"{ASEC_SOURCE_ARTIFACTS[dropped].sha256} (the base would refuse "
            "the CLI pin)",
            f"education_assistance_source.py at {head} lacks the Census archive "
            f"pin {ASEC_EDUCATION_ASSISTANCE_ARCHIVES[dropped].zip_sha256} (the "
            "base would refuse that year's archive)",
        ]


def test_the_default_pool_registration_changes_only_the_pool_and_its_seed() -> None:
    tool = DEFAULT_POOL_TOOL
    rerun = plan_lib.US_PUF_SUPPORT_BASE_RERUN
    assert plan_lib.TOOLS[tool.name] is tool
    assert (
        dataclasses.replace(
            tool,
            name=rerun.name,
            inputs=rerun.inputs,
            stages=rerun.stages,
            argv_builder=rerun.argv_builder,
            home_seeds=rerun.home_seeds,
        )
        == rerun
    )
    stage = tool.stages["all"]
    assert stage.required_inputs == tool.inputs
    assert (
        dataclasses.replace(stage, required_inputs=rerun.stages["all"].required_inputs)
        == rerun.stages["all"]
    )
    # The seed is the historical seed's file under a role of its own, because
    # this pool has no income-2022 education archive to seed it from.
    assert tool.home_seeds == (SEED,)
    (historical,) = rerun.home_seeds
    assert (SEED.path, SEED.sha256) == (historical.path, historical.sha256)
    assert tool.inputs == (
        *plan_lib.puf_support_base_inputs(plan_lib.PUF_SUPPORT_BASE_POOL_2023_2025),
        SEED.input,
    )
    # Every registration of the base tool names its pool, and the two that
    # came first keep 2022-2024, so earlier plans still validate.
    assert set(plan_lib.PUF_SUPPORT_BASE_POOLS) == {
        name
        for name, registered in plan_lib.TOOLS.items()
        if registered.script == "tools/build_us_puf_support_base.py"
    }
    for name in (plan_lib.US_PUF_SUPPORT_BASE.name, rerun.name):
        assert plan_lib.PUF_SUPPORT_BASE_POOLS[name] == (
            plan_lib.AsecPool((2024, 2023, 2022), (2022, 2023, 2024))
        )


@pytest.mark.parametrize(
    ("asec_years", "education_years"),
    [
        ((2025, 2024, 2025), (2024, 2025, 2025)),
        ((2025, 2024, 2023), (2025, 2024, 2023)),
        ((2025, 2024, 2023), (2023, 2024)),
        ((2025, 2024), (2023, 2024, 2025)),
    ],
)
def test_an_asec_pool_refuses_a_repeated_or_unmatched_year(
    asec_years: tuple[int, ...], education_years: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="names each year once"):
        plan_lib.AsecPool(asec_years, education_years)


_POOLS = (
    st.lists(st.integers(2000, 2040), min_size=1, max_size=5, unique=True)
    .flatmap(st.permutations)
    .map(lambda years: plan_lib.AsecPool(tuple(years), tuple(sorted(years))))
)


@settings(max_examples=100, deadline=None)
@given(pool=_POOLS, digest_seed=st.integers(0, 2**32))
def test_the_argv_builder_maps_any_pool_year_for_year(
    pool: plan_lib.AsecPool, digest_seed: int
) -> None:
    roles = plan_lib.puf_support_base_inputs(pool)
    paths = {role: f"/work/inputs/{role}/{role}.bin" for role in roles}
    digests = {
        role: f"{(digest_seed + index) % 2**32:064x}"
        for index, role in enumerate(roles)
    }
    plan = SimpleNamespace(
        stage="all",
        inputs={
            role: SimpleNamespace(sha256=digest) for role, digest in digests.items()
        },
    )
    argv = plan_lib.puf_support_base_argv_builder(pool)(plan, paths, "/work/state")
    assert _mapping(argv, "--asec-h5") == [
        (year, paths[f"asec_{year}_h5"]) for year in pool.asec_years
    ]
    assert _mapping(argv, "--asec-h5-sha256") == [
        (year, digests[f"asec_{year}_h5"]) for year in pool.asec_years
    ]
    assert _mapping(argv, "--asec-education-source") == [
        (year, paths[f"asec_education_{year}_zip"]) for year in pool.education_years
    ]
    # The flags are the registration's, and every input is named exactly once.
    flags = [value for value in argv if value.startswith("--")]
    assert set(flags) == plan_lib.PUF_SUPPORT_BASE_BUILDER_FLAGS
    values = [value.partition("=")[2] or value for value in argv]
    for role, path in paths.items():
        assert values.count(path) == 1, role
    # The historical builder is this builder at the 2022-2024 pool.
    if pool == plan_lib.PUF_SUPPORT_BASE_POOL_2022_2024:
        assert argv == plan_lib.US_PUF_SUPPORT_BASE.argv_builder(
            plan, paths, "/work/state"
        )
