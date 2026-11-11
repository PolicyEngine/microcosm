"""Import the retirement-contribution stage without the package __init__ chains.

``microcosm.build`` and ``microcosm.build.us_runtime`` have heavy __init__
modules (the whole build stack). Registering each as a package module built
from its real spec, without executing its __init__, lets the stage module and
its direct helpers import exactly as in production and nothing else.
``microcosm.build.us`` (packaged JSON, no __init__) imports as a namespace
package, so importlib.resources finds the stage manifest.
"""

import importlib.util
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GateResult:
    """Stand-in for microcosm.build.gates.GateResult, whose module imports
    microcosm.calibrate and through it torch. The imputation path never
    constructs one; only the stage's signal gates do."""

    name: str
    passed: bool
    failures: tuple[str, ...] = ()
    details: Mapping[str, object] = field(default_factory=dict)


def install(root: Path) -> None:
    packages = root / "packages"
    for name in (
        "microcosm-graph",
        "microcosm-frame",
        "microcosm-fit",
        "microcosm-calibrate",
        "microcosm-build",
        "microcosm-data",
        "microcosm-diagnostics",
    ):
        sys.path.insert(0, str(packages / name / "src"))
    for dotted, rel in (
        ("microcosm.build", "microcosm-build/src/microcosm/build"),
        (
            "microcosm.build.us_runtime",
            "microcosm-build/src/microcosm/build/us_runtime",
        ),
    ):
        directory = packages / rel
        spec = importlib.util.spec_from_file_location(
            dotted,
            directory / "__init__.py",
            submodule_search_locations=[str(directory)],
        )
        sys.modules[dotted] = importlib.util.module_from_spec(spec)
    gates = types.ModuleType("microcosm.build.gates")
    gates.GateResult = GateResult
    sys.modules["microcosm.build.gates"] = gates
