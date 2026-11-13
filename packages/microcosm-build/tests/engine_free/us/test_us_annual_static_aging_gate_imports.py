"""The annual lane's parent-flag check never imports ``us_runtime``.

Importing any ``microcosm.build.us_runtime`` module runs that package's
``__init__``, which builds the spine-agreement registry; that loads the
take-up contract, which asserts the engine ABI lock wherever policyengine-us
is installed. The annual static-aging lane checks its parent's
local-jurisdiction flags before any model work, so a check that went through
``us_runtime`` would refuse every parent under any policyengine-us other than
the locked one. The check therefore imports
:mod:`microcosm.build.us_local_jurisdictions`, which sits outside
``us_runtime``.

Invariants:

- in a fresh interpreter, the lane run up to its model-provenance step, on a
  parent that carries its flags and on one that lacks them, leaves no
  ``microcosm.build.us_runtime`` module in ``sys.modules``. It must be a fresh
  interpreter: this test process has usually imported ``us_runtime`` already;
- ``us_runtime.local_jurisdictions`` re-exports every public name of the
  module as the same object, so the geography lines that import it there
  see one implementation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from microcosm.build import us_annual_static_aging as annual
from microcosm.build import us_local_jurisdictions
from microcosm.build.us_runtime import local_jurisdictions as us_runtime_re_export
from test_support.paths import paths_for

#: Runs the lane in a fresh interpreter on two parents and prints what it
#: loaded. Each parent is a Yonkers and a Denver household on their blocks.
_GATE_PROBE = r"""
import json
import sys
from pathlib import Path

import pandas as pd

from microcosm.build import us_annual_static_aging as annual
from microcosm.frame import US_SCHEMA
from microcosm.frame.materialize import put_frame_table

work = Path(sys.argv[1])
HOUSEHOLD = {
    "state_fips": [36, 8],
    "county_fips": ["36119", "08031"],
    "place_fips": ["84000", "20000"],
    "block_geoid": ["361190001001000", "080310001001000"],
}
FLAGS = {
    "in_denver": ("household", [False, True]),
    "in_wilmington": ("household", [False, False]),
    "in_yonkers": ("household", [True, False]),
    "pa_philadelphia_wage_tax_resident": ("person", [False, False]),
}


def parent(name, flags):
    ids = [1, 2]
    tables = {
        entity: pd.DataFrame({US_SCHEMA.entity_id_column(entity): ids})
        for entity in US_SCHEMA.entities
    }
    for group in US_SCHEMA.group_entities:
        tables["person"][US_SCHEMA.membership_column(group)] = ids
    tables["person"]["age"] = [40, 40]
    tables["person"]["is_female"] = [False, True]
    tables["household"]["household_weight"] = [1.0, 2.0]
    for column, values in HOUSEHOLD.items():
        tables["household"][column] = values
    for flag in flags:
        entity, values = FLAGS[flag]
        tables[entity][flag] = values
    path = work / f"{name}.h5"
    with pd.HDFStore(path, "w") as store:
        for entity, table in tables.items():
            put_frame_table(
                store, entity, table, preferred_format="table", data_columns=True
            )
        store.put("_time_period", pd.Series([2024]), format="table")
    return path


class ReachedModelProvenance(Exception):
    pass


def reached(**kwargs):
    raise ReachedModelProvenance


annual._model_provenance = reached
ssa = work / "ssa.csv"
pd.DataFrame(
    {"Year": [2024, 2025], "Age": [40, 40], "M Tot": [4, 5], "F Tot": [2, 3]}
).to_csv(ssa, index=False)
outcomes = {}
for name, flags in (("flagged", tuple(FLAGS)), ("unflagged", ())):
    base = parent(name, flags)
    try:
        annual.build_annual_static_aging(
            base_h5=base,
            base_sha256=annual._sha256(base),
            parent_release="pinned-parent",
            ssa_csv=ssa,
            ssa_sha256=annual._sha256(ssa),
            model_source=work,
            model_commit="a" * 40,
            model_version="test",
            output_dir=work / f"{name}-candidate",
            base_year=2024,
            end_year=2025,
            epochs=1,
        )
    except ReachedModelProvenance:
        outcomes[name] = "reached model provenance"
    except ValueError as error:
        outcomes[name] = str(error)
print(
    json.dumps(
        {
            "module": annual.__file__,
            "outcomes": outcomes,
            "us_runtime_modules": sorted(
                module
                for module in sys.modules
                if module == "microcosm.build.us_runtime"
                or module.startswith("microcosm.build.us_runtime.")
            ),
        }
    )
)
"""


def test_the_annual_parent_check_leaves_us_runtime_unloaded(tmp_path) -> None:
    root = paths_for("microcosm-build").repository
    # Prefer this worktree's sources to editable installs in another checkout,
    # and start the child the way this interpreter started (-S, -P).
    pythonpath = os.pathsep.join(
        [
            *(str(path) for path in sorted((root / "packages").glob("*/src"))),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    flags = [
        *(["-S"] if sys.flags.no_site else []),
        *(["-P"] if sys.flags.safe_path else []),
    ]
    completed = subprocess.run(
        [sys.executable, *flags, "-c", _GATE_PROBE, str(tmp_path)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": pythonpath},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    assert result["module"] == annual.__file__
    assert result["outcomes"]["flagged"] == "reached model provenance"
    assert "needs a parent that already carries" in result["outcomes"]["unflagged"]
    assert result["us_runtime_modules"] == []


def test_us_runtime_re_exports_the_same_objects() -> None:
    assert us_runtime_re_export.__all__ == us_local_jurisdictions.__all__
    for name in us_local_jurisdictions.__all__:
        assert getattr(us_runtime_re_export, name) is getattr(
            us_local_jurisdictions, name
        ), name
