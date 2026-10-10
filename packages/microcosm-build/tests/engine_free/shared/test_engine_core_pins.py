"""Each country keeps its own policyengine-core pin (microcosm#1086, option 2).

Nothing imports both engines and CI installs one country extra per job, so
every extra that installs policyengine-us is declared in conflict with every
extra that installs policyengine-uk, and uv locks the two sides in separate
forks. The US side stays on the core the certified US default was built with
until its next certified build. The resolution checks read the committed
``uv.lock`` through ``uv export --frozen``, uv's own reading of what an install
path gets, without the network or the environment.
"""

from __future__ import annotations

import itertools
import shutil
import subprocess
import tomllib

import pytest
from packaging.requirements import Requirement

from test_support.paths import REPOSITORY_ROOT

#: What every US install path resolves until the next certified US build.
US_ENGINE = {
    "policyengine-core": "3.32.5",
    "policyengine-us": "2.2.1",
    "spm-calculator": "1.0.0",
}
ENGINE_PACKAGES = ("microcosm-build", "microcosm-data", "microcosm-frame")


def _optional_dependencies(package: str) -> dict[str, list[str]]:
    pyproject = REPOSITORY_ROOT / "packages" / package / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]
    return project.get("optional-dependencies", {})


def _extras_installing(engine: str) -> list[tuple[str, str]]:
    return [
        (package, extra)
        for package in ENGINE_PACKAGES
        for extra, requirements in _optional_dependencies(package).items()
        if any(Requirement(requirement).name == engine for requirement in requirements)
    ]


US_EXTRAS = _extras_installing("policyengine-us")
UK_EXTRAS = _extras_installing("policyengine-uk")


def _uv_export(*args: str) -> subprocess.CompletedProcess[str]:
    uv = shutil.which("uv")
    assert uv is not None, "uv reads the lock as an install would; CI sets it up."
    return subprocess.run(
        [uv, "export", "--frozen", "--no-hashes", "--no-header"]
        + ["--no-emit-workspace", *args],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _resolved(requirements: str) -> dict[str, str]:
    versions = {}
    for line in requirements.splitlines():
        pin = line.split(";")[0].strip()
        if "==" in pin and not pin.startswith(("#", "-")):
            name, version = pin.split("==", 1)
            versions[name.strip()] = version.strip()
    return versions


def test_the_rule_covers_every_engine_extra() -> None:
    # A new extra that installs either engine must join the conflicts below.
    assert US_EXTRAS == [
        ("microcosm-build", "us"),
        ("microcosm-data", "us"),
        ("microcosm-frame", "policyengine"),
    ]
    assert UK_EXTRAS == [
        ("microcosm-build", "uk"),
        ("microcosm-data", "uk"),
        ("microcosm-frame", "uk"),
    ]


@pytest.mark.parametrize(("package", "extra"), US_EXTRAS)
def test_every_us_engine_extra_pins_the_us_core(package: str, extra: str) -> None:
    requirements = _optional_dependencies(package)[extra]
    assert f"policyengine-core=={US_ENGINE['policyengine-core']}" in requirements


def _us_install_paths() -> list[tuple[str, ...]]:
    # Workspace installs name each US extra alone, then all of them together.
    extra_names = sorted({extra for _, extra in US_EXTRAS})
    return [
        *(("--all-packages", "--extra", extra) for extra in extra_names),
        (
            "--all-packages",
            *(arg for extra in extra_names for arg in ("--extra", extra)),
        ),
        *(("--package", package, "--extra", extra) for package, extra in US_EXTRAS),
    ]


@pytest.mark.parametrize("args", _us_install_paths(), ids=" ".join)
def test_every_us_install_path_resolves_the_certified_us_engine(
    args: tuple[str, ...],
) -> None:
    result = _uv_export(*args)

    assert result.returncode == 0, result.stderr
    resolved = _resolved(result.stdout)
    assert {name: resolved.get(name) for name in US_ENGINE} == US_ENGINE
    assert "policyengine-uk" not in resolved


def test_every_us_and_uk_pair_is_declared_and_locked_as_a_conflict() -> None:
    expected = {
        frozenset({us, uk}) for us, uk in itertools.product(US_EXTRAS, UK_EXTRAS)
    }

    def pairs(groups: list[list[dict[str, str]]]) -> set[frozenset[tuple[str, str]]]:
        return {
            frozenset((item["package"], item["extra"]) for item in group)
            for group in groups
        }

    workspace = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text())
    lock = tomllib.loads((REPOSITORY_ROOT / "uv.lock").read_text())
    assert pairs(workspace["tool"]["uv"]["conflicts"]) == expected
    assert pairs(lock["conflicts"]) == expected


def _us_and_uk_combinations() -> list[tuple[str, ...]]:
    same_package = [
        ("--package", us_package, "--extra", us_extra, "--extra", uk_extra)
        for (us_package, us_extra), (uk_package, uk_extra) in itertools.product(
            US_EXTRAS, UK_EXTRAS
        )
        if us_package == uk_package
    ]
    extra_names = sorted(
        {
            (us_extra, uk_extra)
            for (_, us_extra), (_, uk_extra) in itertools.product(US_EXTRAS, UK_EXTRAS)
        }
    )
    workspace = [
        ("--all-packages", "--extra", us_extra, "--extra", uk_extra)
        for us_extra, uk_extra in extra_names
    ]
    return same_package + workspace


@pytest.mark.parametrize("args", _us_and_uk_combinations(), ids=" ".join)
def test_uv_refuses_every_us_and_uk_combination(args: tuple[str, ...]) -> None:
    result = _uv_export(*args)

    assert result.returncode != 0
    assert "incompatible with the declared conflicts" in result.stderr
