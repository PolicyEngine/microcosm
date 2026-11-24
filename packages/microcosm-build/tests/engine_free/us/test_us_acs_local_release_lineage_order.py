"""Where the release tool requires the donor's reviewed lineage (microcosm#1661).

A national donor of another lineage than Build O needs its
``REVIEWED_DONOR_LINEAGES`` entry before calibrate starts: calibrate fixes the
commit that package accepts, and finalize writes the inherited limitations the
entry decides. Materialize writes nothing the entry decides, so it may run
while the new donor's evidence is reviewed.
"""

from __future__ import annotations

import importlib.util
import inspect

from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


def _load_tool_module():
    path = _TEST_PATHS.repository / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_local_release_lineage_order", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_TOOL = _load_tool_module()
_CHECK = "_require_reviewed_donor_lineage("


def test_materialize_does_not_require_the_donor_lineage() -> None:
    assert _CHECK not in inspect.getsource(_TOOL.do_materialize)


def test_calibrate_requires_it_before_loading_or_solving() -> None:
    source = inspect.getsource(_TOOL.do_calibrate)
    assert _CHECK in source
    assert source.index(_CHECK) < source.index("load_checkpoint_surface(")


def test_finalize_requires_it_again() -> None:
    assert _CHECK in inspect.getsource(_TOOL.do_finalize)
