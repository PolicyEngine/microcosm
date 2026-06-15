"""populace.data: load the published populace populations.

The distribution end of the stack. Where the operator shards
(:mod:`populace.frame`, :mod:`populace.fit`, :mod:`populace.calibrate`) *build*
populations, this shard *serves* the ones populace has published: a registry of
``(country, year)`` pointers to artifacts on the Hugging Face Hub, and a loader
that returns each as the country engine's dataset object.

    >>> from policyengine_us import Microsimulation
    >>> from populace.data import load
    >>> sim = Microsimulation(dataset=load("us", 2024))
    >>> sim.calculate("household_net_income", 2024).sum()

Publishing a new population is one :class:`DatasetSpec` entry in
:mod:`populace.data.registry` plus its uploaded artifact — never a new package
or repository, which is the whole point of a registry-driven shard.

This shard does not depend on the Frame kernel: a published population is an
engine-native dataset, so loading it needs only ``huggingface_hub`` and the
country engine (an optional extra). It therefore carries no kernel-compat gate —
there is no kernel in its dependency closure to gate against.
"""

from populace.data.contract import (
    RELEASE_MANIFEST_SCHEMA_VERSION,
    REQUIRED_RELEASE_FILES,
    US_SOURCE_COVERAGE_DIAGNOSTICS_FILE,
    ReleaseContractError,
    required_release_files,
    validate_release_dir,
)
from populace.data.loader import (
    available,
    download,
    latest_year,
    load,
    resolve,
)
from populace.data.registry import REGISTRY, DatasetSpec, register
from populace.data.release import (
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
    "resolve",
    "latest_year",
    "DatasetSpec",
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
