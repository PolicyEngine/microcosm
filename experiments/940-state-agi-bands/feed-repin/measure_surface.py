"""Measure the pinned-feed release surface as test_us_fiscal_targets' fixture builds it.

Run from the repository root with PYTHONPATH=. (test_support is a repo package).
"""

import importlib.util

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.us_runtime import (
    apply_us_medicaid_enrollment_substitutions,
    compile_us_fiscal_target_registry,
    default_congressional_district_vintage_crosswalk_path,
    load_congressional_district_vintage_crosswalk,
)
from microcosm.build.us_runtime.chronicle_feed import load_us_chronicle_feed
from microcosm.calibrate import TargetRegistry
from test_support.paths import paths_for

root = paths_for("microcosm-build").repository


def tool(name):
    spec = importlib.util.spec_from_file_location(name, root / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


feed = tool("build_us_target_parity_manifest").DEFAULT_FEED_PATH
facts = load_ledger_consumer_artifact(
    feed,
    expected_facts_sha256=load_us_chronicle_feed().facts_sha256,
    expected_manifest_sha256=None,
).facts
cw = load_congressional_district_vintage_crosswalk(
    default_congressional_district_vintage_crosswalk_path()
)
reg = compile_us_fiscal_target_registry(
    facts,
    target_period=2024,
    congressional_district_vintage_crosswalk=cw,
    age_targets=True,
)
reg, _ = apply_us_medicaid_enrollment_substitutions(reg)
specs, _ = tool("build_us_fiscal_refresh_release")._select_target_surface(
    reg.specs, "national_state"
)
surface = TargetRegistry(specs, country="us")
bands = [
    s
    for s in surface.specs
    if s.metadata.get("requires_state_agi_band_rebase") == "true"
]
print(
    "registry",
    len(reg.specs),
    "national_state",
    len(surface.specs),
    "version",
    surface.version,
    "bands_in_surface",
    len(bands),
)
