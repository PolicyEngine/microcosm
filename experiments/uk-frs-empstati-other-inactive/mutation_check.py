"""Mutation check for the EMPSTATI port: each mutation must fail the tests.

Runs against a copy of microcosm-build's sources placed first on PYTHONPATH, so
the worktree itself is never mutated. Prints one line per mutation. Usage (UK
engine environment)::

    uv run --no-sync python experiments/uk-frs-empstati-other-inactive/mutation_check.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COPY = Path(tempfile.mkdtemp(prefix="empstati-mutants-")) / "src"
EMPLOYMENT = "microcosm/build/uk_runtime/frs_employment.py"
PROXIES = "microcosm/build/uk_runtime/frs_legacy_proxies.py"
TESTS = [
    "packages/microcosm-build/tests/engine_free/uk/test_uk_frs_employment.py",
    "packages/microcosm-build/tests/engine_free/uk/test_uk_frs_legacy_proxies.py",
    "packages/microcosm-build/tests/engine/uk/test_uk_frs_employment.py",
]

MUTATIONS = {
    "11 back to LONG_TERM_DISABLED": (
        EMPLOYMENT,
        '11: "OTHER_INACTIVE",  # Other inactive',
        '11: "LONG_TERM_DISABLED",',
    ),
    "unknown adult codes default to LONG_TERM_DISABLED": (
        EMPLOYMENT,
        "    unknown = adult & pd.isna(adult_status)\n",
        '    adult_status = pd.Series(adult_status).fillna("LONG_TERM_DISABLED")'
        ".to_numpy(dtype=object)\n    unknown = adult & pd.isna(adult_status)\n",
    ),
    "every row treated as an adult": (
        EMPLOYMENT,
        "    adult = np.asarray(is_adult_record, dtype=bool)\n",
        "    adult = np.ones(len(codes), dtype=bool)\n",
    ),
    "adult records by code presence, not adult.tab membership": (
        EMPLOYMENT,
        'aligned["empstati"], person["person_id"].isin(adult["person_id"])',
        'aligned["empstati"], aligned["empstati"].notna().to_numpy()',
    ),
    "9 and 10 swapped": (
        EMPLOYMENT,
        '9: "LONG_TERM_DISABLED",  # Permanently sick/disabled\n'
        '        10: "SHORT_TERM_DISABLED",',
        '9: "SHORT_TERM_DISABLED",  # Permanently sick/disabled\n'
        '        10: "LONG_TERM_DISABLED",',
    ),
    "small counts printed": (
        EMPLOYMENT,
        "if count >= 10 else",
        "if count >= 0 else",
    ),
    "children get a non-CHILD status": (
        EMPLOYMENT,
        "np.where(adult, adult_status, CHILD_EMPLOYMENT_STATUS)",
        'np.where(adult, adult_status, "OTHER_INACTIVE")',
    ),
    "code 0 accepted for adults": (
        EMPLOYMENT,
        '        1: "FT_EMPLOYED",',
        '        0: "CHILD",\n        1: "FT_EMPLOYED",',
    ),
    "non-integer codes truncated": (
        EMPLOYMENT,
        'np.asarray(pd.to_numeric(empstati, errors="coerce"), dtype="float64")',
        'np.floor(np.asarray(pd.to_numeric(empstati, errors="coerce"), dtype="float64"))',
    ),
    "OTHER_INACTIVE added to the ESA health statuses": (
        PROXIES,
        'ESA_HEALTH_EMPLOYMENT_STATUSES = ("LONG_TERM_DISABLED", "SHORT_TERM_DISABLED")',
        'ESA_HEALTH_EMPLOYMENT_STATUSES = ("LONG_TERM_DISABLED", "SHORT_TERM_DISABLED", '
        '"OTHER_INACTIVE")',
    ),
}


def run(env: dict[str, str]) -> int:
    return subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore",
            "-m",
            "pytest",
            *TESTS,
            "-x",
            "-q",
            "-p",
            "no:cacheprovider",
            "--hypothesis-seed=0",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode


def main() -> int:
    source = ROOT / "packages/microcosm-build/src"
    env = {**os.environ, "PYTHONPATH": str(COPY)}
    shutil.rmtree(COPY, ignore_errors=True)
    shutil.copytree(source, COPY)
    resolved = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore",
            "-c",
            "import microcosm.build.uk_runtime.frs_employment as m; print(m.__file__)",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert resolved.startswith(str(COPY)), resolved
    baseline = run(env)
    print(f"baseline (unmutated copy): rc={baseline}")
    assert baseline == 0
    killed = 0
    for name, (relative, old, new) in MUTATIONS.items():
        path = COPY / relative
        original = (source / relative).read_text()
        assert original.count(old) == 1, name
        path.write_text(original.replace(old, new))
        rc = run(env)
        path.write_text(original)
        verdict = "killed" if rc != 0 else "SURVIVED"
        killed += rc != 0
        print(f"{verdict}: {name} (rc={rc})")
    print(f"{killed}/{len(MUTATIONS)} mutations killed")
    shutil.rmtree(COPY.parent, ignore_errors=True)
    return 0 if killed == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
