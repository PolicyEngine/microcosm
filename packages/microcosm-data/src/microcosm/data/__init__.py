"""microcosm.data: load the published microcosm populations.

The distribution end of the stack. Where the operator shards
(:mod:`microcosm.frame`, :mod:`microcosm.fit`, :mod:`microcosm.calibrate`) *build*
populations, this shard *serves* the ones microcosm has published: a registry of
``(country, year, variant)`` pointers to artifacts on the Hugging Face Hub, and
a loader that returns each as the country engine's dataset object.

    >>> from policyengine_us import Microsimulation
    >>> from microcosm.data import load
    >>> sim = Microsimulation(dataset=load("us", 2024))
    >>> sim.calculate("household_net_income", 2024).sum()

Publishing a new population is one :class:`DatasetSpec` entry in
:mod:`microcosm.data.registry` plus its uploaded artifact — never a new package
or repository, which is the whole point of a registry-driven shard.

This shard does not depend on the Frame kernel: a published population is an
engine-native dataset, so loading it needs only ``huggingface_hub`` and the
country engine (an optional extra). The loader still enforces the release's
certified country-model and PolicyEngine Core version specifiers before it
constructs that engine-native dataset.
"""

from microcosm.data.contract import (
    RELEASE_MANIFEST_SCHEMA_VERSION,
    REQUIRED_RELEASE_FILES,
    US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
    ReleaseContractError,
    required_release_files,
    validate_release_dir,
)
from microcosm.data.loader import (
    available,
    available_variants,
    download,
    latest_year,
    load,
    resolve,
)
from microcosm.data.registry import DEFAULT_VARIANT, REGISTRY, DatasetSpec, register
from microcosm.data.release import (
    LATEST_POINTER_PATH,
    LATEST_POINTER_SCHEMA_VERSION,
    LatestPointer,
    latest_pointer_payload,
    latest_release,
    publish_release,
)

__all__ = [
    "load",
    "download",
    "available",
    "available_variants",
    "resolve",
    "latest_year",
    "DatasetSpec",
    "DEFAULT_VARIANT",
    "REGISTRY",
    "register",
    "RELEASE_MANIFEST_SCHEMA_VERSION",
    "REQUIRED_RELEASE_FILES",
    "US_SOURCE_COVERAGE_DIAGNOSTICS_FILE",
    "ReleaseContractError",
    "required_release_files",
    "validate_release_dir",
    "LATEST_POINTER_PATH",
    "LATEST_POINTER_SCHEMA_VERSION",
    "LatestPointer",
    "latest_pointer_payload",
    "latest_release",
    "publish_release",
]

__version__ = "0.1.0"
