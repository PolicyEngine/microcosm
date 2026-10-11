"""Seed nine faults in the rebase reference and check a property test catches each.

Run from the repository root:
    uv run python docs/evidence/us-puf-donor-uprating/proposal/mutation_check.py

Each fault is one edit to ``rebase_reference.py``, applied to a copy in a
temporary folder with the tests and the archived sources beside it. A fault
that no test catches is reported as SURVIVED and the script exits nonzero.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MUTATIONS = {
    "rebase does not divide out the carried factor": (
        "        factor = table.growth(leg.series, SOURCE_YEAR, build_year) / leg.carried\n",
        "        factor = table.growth(leg.series, SOURCE_YEAR, build_year)\n",
    ),
    "the stage accepts an already rebased donor": (
        "    if donor.lineage != ARCHIVED_LINEAGE or donor.price_year is not None:\n",
        "    if False:\n",
    ),
    "the positive leg is applied to every value": (
        "    if leg.sign > 0:\n        return values > 0\n",
        "    if leg.sign > 0:\n        return np.ones(values.shape, dtype=bool)\n",
    ),
    "verification always passes": (
        "    expected = source_values * carried\n    return bool(",
        "    expected = source_values * carried\n    return True or bool(",
    ),
    "non-qualified dividends follow the qualified series": (
        "        non_qualified_series, SOURCE_YEAR, build_year\n",
        "        qualified_series, SOURCE_YEAR, build_year\n",
    ),
    "recovered dividends are not snapped to whole dollars": (
        "    snapped = tuple(np.rint(values) for values in recovered)\n",
        "    snapped = recovered\n",
    ),
    "the splice applies the projection index without rebasing it to the pivot": (
        "            spliced[year] = actuals[pivot] * index / projection[pivot]\n",
        "            spliced[year] = actuals[pivot] * index\n",
    ),
    "derived columns are left unscaled": (
        "    return values * ratio\n",
        "    return values\n",
    ),
    "untouched columns are changed": (
        "        untouched={name: values.copy() for name, values in donor.untouched.items()},\n",
        "        untouched={name: values * 2 for name, values in donor.untouched.items()},\n",
    ),
}


def main() -> int:
    reference = (HERE / "rebase_reference.py").read_text()
    survived = 0
    for name, (old, new) in MUTATIONS.items():
        if old not in reference:
            print(f"{name}: the line to mutate is not in the reference")
            return 2
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "proposal").mkdir()
            shutil.copytree(HERE.parent / "archived", root / "archived")
            (root / "proposal" / "rebase_reference.py").write_text(
                reference.replace(old, new)
            )
            shutil.copy(HERE / "test_rebase_properties.py", root / "proposal")
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(root / "proposal" / "test_rebase_properties.py"),
                    "-q",
                    "-x",
                    "-p",
                    "no:cacheprovider",
                    "--rootdir",
                    str(root),
                ],
                capture_output=True,
                text=True,
            )
        caught = re.findall(r"FAILED \S+::(\S+)", run.stdout)
        if caught:
            print(f"{name}: caught by {caught[0]}")
        else:
            survived += 1
            print(f"{name}: SURVIVED")
    return 1 if survived else 0


if __name__ == "__main__":
    raise SystemExit(main())
